#!/usr/bin/env python3
"""Explicit PyTorch training loop for the cross-encoder ranker."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import build_index, get_index
from src.config import load_config
from src.crossencoder import (
    DEFAULT_CROSSENCODER_NAME,
    CrossEncoderRanker,
    rerank,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


class PairDataset(Dataset[Tuple[str, str, float]]):
    """Binary labeled query-passage pairs."""

    def __init__(self, pairs: Sequence[Tuple[str, str, float]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[str, str, float]:
        return self.pairs[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the PyTorch cross-encoder passage ranker."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/config.yaml"),
        help="Path to YAML configuration.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "data/processed"),
        help="Directory containing corpus, train queries/qrels, and val data.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Initialization checkpoint; defaults to MS MARCO MiniLM.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "artifacts/crossencoder"),
        help="Best checkpoint output directory.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--negative-count", type=int, default=2)
    parser.add_argument("--candidate-depth", type=int, default=100)
    parser.add_argument(
        "--max-train-queries",
        type=int,
        default=None,
        help="Limit training to the first N queries (overfit/debug sanity runs).",
    )
    parser.add_argument("--log-steps", type=int, default=500)
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable CUDA mixed precision.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_parquet(
    data_dir: Path,
    filename: str,
    required_columns: set[str],
) -> pd.DataFrame:
    path = data_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required data file not found: {path}")
    frame = pd.read_parquet(path)
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"Required data file is empty: {path}")
    return frame


def mine_training_pairs(
    train_queries: pd.DataFrame,
    train_qrels: pd.DataFrame,
    triplets: pd.DataFrame,
    passage_text_by_pid: Dict[int, str],
    candidate_depth: int = 100,
    negative_count: int = 2,
) -> List[Tuple[str, str, float]]:
    """Use each query's positive and BM25 top-ranked non-relevant passages."""
    if candidate_depth <= 0 or negative_count <= 0:
        raise ValueError("candidate_depth and negative_count must be positive")

    relevant_by_qid: Dict[int, set[int]] = {}
    for row in train_qrels.itertuples(index=False):
        relevant_by_qid.setdefault(int(row.query_id), set()).add(int(row.relevant_pid))

    positive_by_qid: Dict[int, Tuple[str, int]] = {}
    for row in triplets.itertuples(index=False):
        qid = int(row.qid)
        if qid not in positive_by_qid:
            positive_by_qid[qid] = (str(row.pos_text), int(row.pos_pid))

    pairs: List[Tuple[str, str, float]] = []
    missing_query_count = 0
    for row in train_queries.itertuples(index=False):
        qid = int(row.query_id)
        query = str(row.query_text).strip()
        positive = positive_by_qid.get(qid)
        if not query or positive is None:
            missing_query_count += 1
            continue
        positive_text, positive_pid = positive
        candidates = get_index().retrieve_bm25(query, k=candidate_depth)
        known_relevant = relevant_by_qid.get(qid, set())
        negative_pids: List[int] = []
        for pid, _ in candidates:
            if pid == positive_pid or pid in known_relevant or pid in negative_pids:
                continue
            negative_pids.append(pid)
            if len(negative_pids) == negative_count:
                break
        if len(negative_pids) < negative_count:
            logger.warning(
                "Skipping qid=%d: only %d BM25 negatives found",
                qid,
                len(negative_pids),
            )
            continue

        pairs.append((query, positive_text, 1.0))
        pairs.extend(
            (query, passage_text_by_pid[pid], 0.0)
            for pid in negative_pids
        )

    if missing_query_count:
        logger.warning(
            "Skipped %d train queries without a usable positive triplet",
            missing_query_count,
        )
    if not pairs:
        raise ValueError("No labeled cross-encoder training pairs were produced")
    return pairs


def make_collate_fn(tokenizer: Any, max_length: int):
    """Create a pair-tokenizing collator with dynamic batch padding."""

    def collate(batch: Sequence[Tuple[str, str, float]]):
        queries = [item[0] for item in batch]
        passages = [item[1] for item in batch]
        labels = torch.tensor([item[2] for item in batch], dtype=torch.float32)
        encoded = tokenizer(
            queries,
            passages,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        return encoded, labels

    return collate


def prepare_validation_candidates(
    queries_df: pd.DataFrame,
    candidate_depth: int,
) -> Dict[int, List[Tuple[int, float]]]:
    """Retrieve BM25 validation candidates once for repeated epoch scoring."""
    validation_candidates: Dict[int, List[Tuple[int, float]]] = {}
    index = get_index()
    for row in queries_df.itertuples(index=False):
        qid = int(row.query_id)
        validation_candidates[qid] = index.retrieve_bm25(
            str(row.query_text),
            k=candidate_depth,
        )
    return validation_candidates


def evaluate_val_mrr(
    model: CrossEncoderRanker,
    tokenizer: Any,
    queries_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    candidates_by_qid: Dict[int, List[Tuple[int, float]]],
    passage_text_by_pid: Dict[int, str],
    batch_size: int,
    max_length: int,
    fp16: bool,
) -> float:
    """Compute MRR@10 by re-ranking the cached BM25 top-100 results."""
    relevant_by_qid: Dict[int, set[int]] = {}
    for row in qrels_df.itertuples(index=False):
        relevant_by_qid.setdefault(int(row.query_id), set()).add(int(row.relevant_pid))

    reciprocal_ranks: List[float] = []
    for row in queries_df.itertuples(index=False):
        qid = int(row.query_id)
        candidates = candidates_by_qid.get(qid, [])
        pids = [pid for pid, _ in candidates]
        texts = [passage_text_by_pid[pid] for pid in pids]
        ranked = rerank(
            str(row.query_text),
            pids,
            texts,
            k=10,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            fp16=fp16,
        )
        reciprocal_rank = 0.0
        relevant = relevant_by_qid.get(qid, set())
        for rank, (pid, _) in enumerate(ranked, start=1):
            if pid in relevant:
                reciprocal_rank = 1.0 / rank
                break
        reciprocal_ranks.append(reciprocal_rank)
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def main() -> None:
    args = parse_args()
    if (
        args.batch_size <= 0
        or args.lr <= 0
        or args.epochs <= 0
        or args.max_length <= 0
        or args.log_steps <= 0
        or not 0 <= args.warmup_ratio <= 1
        or (args.max_train_queries is not None and args.max_train_queries <= 0)
    ):
        raise ValueError("Invalid training hyperparameters")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = args.fp16 and device.type == "cuda"
    if args.fp16 and not use_fp16:
        logger.warning("CUDA is unavailable; disabling fp16 mixed precision")

    config = load_config(args.config)
    data_dir = Path(args.data_dir)
    corpus_df = load_parquet(
        data_dir,
        "corpus.parquet",
        {"pid", "passage_text"},
    )
    train_queries = load_parquet(
        data_dir,
        "train_queries.parquet",
        {"query_id", "query_text"},
    )
    train_qrels = load_parquet(
        data_dir,
        "train_qrels.parquet",
        {"query_id", "relevant_pid"},
    )
    train_triplets = load_parquet(
        data_dir,
        "train_triplets.parquet",
        {"qid", "pos_pid", "pos_text"},
    )
    val_queries = load_parquet(
        data_dir,
        "val_queries.parquet",
        {"query_id", "query_text"},
    )
    val_qrels = load_parquet(
        data_dir,
        "val_qrels.parquet",
        {"query_id", "relevant_pid"},
    )
    if args.max_train_queries is not None:
        sanity_qids = set(
            train_queries["query_id"].head(args.max_train_queries).astype(int)
        )
        train_queries = train_queries[
            train_queries["query_id"].astype(int).isin(sanity_qids)
        ].reset_index(drop=True)
        train_qrels = train_qrels[
            train_qrels["query_id"].astype(int).isin(sanity_qids)
        ].reset_index(drop=True)
        train_triplets = train_triplets[
            train_triplets["qid"].astype(int).isin(sanity_qids)
        ].reset_index(drop=True)
        val_queries = train_queries
        val_qrels = train_qrels
        logger.warning(
            "Overfit sanity mode: using %d training queries for validation MRR",
            len(train_queries),
        )

    artifacts_dir = PROJECT_ROOT / "artifacts"
    logger.info("Building/loading BM25 index for hard-negative mining and validation")
    build_index(corpus_df, artifacts_dir=artifacts_dir)
    passage_text_by_pid = dict(
        zip(corpus_df["pid"].astype(int), corpus_df["passage_text"].astype(str))
    )
    train_pairs = mine_training_pairs(
        train_queries,
        train_qrels,
        train_triplets,
        passage_text_by_pid,
        candidate_depth=args.candidate_depth,
        negative_count=args.negative_count,
    )
    logger.info(
        "Training on %d query-passage pairs from %d queries",
        len(train_pairs),
        len(train_pairs) // (args.negative_count + 1),
    )
    validation_candidates = prepare_validation_candidates(
        val_queries,
        candidate_depth=args.candidate_depth,
    )

    model_name = args.model_name or config.models.crossencoder_name or DEFAULT_CROSSENCODER_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = CrossEncoderRanker(
        model_name,
        dropout=0.1,
        max_length=args.max_length,
    ).to(device)
    loader = DataLoader(
        PairDataset(train_pairs),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer, args.max_length),
        drop_last=False,
    )
    total_steps = len(loader) * args.epochs
    if total_steps == 0:
        raise ValueError("Training dataset yielded no batches")
    warmup_steps = math.ceil(total_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    criterion = torch.nn.BCEWithLogitsLoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_mrr = float("-inf")
    global_step = 0
    loss_since_log: List[float] = []
    training_history: List[Dict[str, float]] = []
    last_validated_step = -1
    start_time = time.perf_counter()

    def validate_and_checkpoint(epoch: int) -> float:
        nonlocal best_mrr, last_validated_step
        model.eval()
        val_mrr = evaluate_val_mrr(
            model,
            tokenizer,
            val_queries,
            val_qrels,
            validation_candidates,
            passage_text_by_pid,
            batch_size=args.batch_size,
            max_length=args.max_length,
            fp16=use_fp16,
        )
        last_validated_step = global_step
        logger.info(
            "step=%d epoch=%d train_loss_window=%.6f val_mrr@10=%.6f",
            global_step,
            epoch,
            float(np.mean(loss_since_log)) if loss_since_log else float("nan"),
            val_mrr,
        )
        training_history.append(
            {
                "step": float(global_step),
                "epoch": float(epoch),
                "val_mrr@10": val_mrr,
            }
        )
        if val_mrr > best_mrr:
            best_mrr = val_mrr
            model.save_pretrained(output_dir, tokenizer=tokenizer)
            (output_dir / "training_metrics.json").write_text(
                json.dumps(
                    {
                        "best_val_mrr@10": best_mrr,
                        "history": training_history,
                        "global_steps": global_step,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            logger.info("Saved best cross-encoder checkpoint to %s", output_dir)
        return val_mrr

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        for encoded, labels in loader:
            inputs = {key: value.to(device) for key, value in encoded.items()}
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_fp16,
            ):
                logits = model(**inputs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            loss_value = float(loss.detach().float().cpu())
            epoch_losses.append(loss_value)
            loss_since_log.append(loss_value)
            if global_step % args.log_steps == 0:
                validate_and_checkpoint(epoch)
                loss_since_log.clear()
                model.train()

        mean_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        logger.info(
            "epoch=%d train_loss=%.6f steps=%d",
            epoch,
            mean_epoch_loss,
            global_step,
        )
        if last_validated_step != global_step:
            validate_and_checkpoint(epoch)
            loss_since_log.clear()

    (output_dir / "training_metrics.json").write_text(
        json.dumps(
            {
                "best_val_mrr@10": best_mrr,
                "history": training_history,
                "global_steps": global_step,
                "training_seconds": time.perf_counter() - start_time,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Training complete. Best validation MRR@10=%.6f; checkpoint=%s",
        best_mrr,
        output_dir,
    )


if __name__ == "__main__":
    main()
