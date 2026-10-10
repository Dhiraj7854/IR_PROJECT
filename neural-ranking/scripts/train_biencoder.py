#!/usr/bin/env python3
"""Fine-tune the MiniLM bi-encoder with hard-negative ranking loss."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers import losses as st_losses
from src.config import load_config
from src.biencoder import DEFAULT_MODEL_NAME
from src.experiments import sample_random_negative_texts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a sentence-transformers bi-encoder."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/config.yaml"),
        help="Path to YAML configuration.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "data/processed"),
        help="Directory containing triplets, corpus, and validation files.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Base model override; defaults to the configured MiniLM model.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "artifacts/biencoder"),
        help="Directory for the best model checkpoint and training metrics.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable mixed-precision training.",
    )
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--val-sample-size", type=int, default=2000)
    parser.add_argument(
        "--negatives",
        choices=("hard", "random"),
        default="hard",
        help="Use mined hard negatives from train_triplets or sample random corpus passages.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sample_validation_corpus(
    corpus_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    sample_size: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Select relevant validation passages first, then fill a random sample."""
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero")
    if sample_size >= len(corpus_df):
        return corpus_df.reset_index(drop=True)

    qrel_pids = list(dict.fromkeys(int(pid) for pid in qrels_df["relevant_pid"]))
    available_pids = set(corpus_df["pid"].astype(int))
    relevant_pids = [pid for pid in qrel_pids if pid in available_pids]
    if len(relevant_pids) > sample_size:
        rng = np.random.default_rng(seed)
        relevant_pids = rng.choice(
            relevant_pids,
            size=sample_size,
            replace=False,
        ).tolist()

    selected = corpus_df[corpus_df["pid"].isin(relevant_pids)]
    remaining_count = sample_size - len(selected)
    if remaining_count:
        remaining = corpus_df[~corpus_df["pid"].isin(relevant_pids)]
        fill = remaining.sample(n=remaining_count, random_state=seed)
        selected = pd.concat([selected, fill], ignore_index=True)
    return selected.reset_index(drop=True)


def evaluate_validation_mrr(
    model: Any,
    queries_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    batch_size: int,
) -> float:
    """Compute validation MRR@10 against the sampled passage corpus."""
    passages = model.encode(
        corpus_df["passage_text"].astype(str).tolist(),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    queries = model.encode(
        queries_df["query_text"].astype(str).tolist(),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    passages = torch.as_tensor(passages, device=model.device)
    queries = torch.as_tensor(queries, device=model.device)
    scores = torch.matmul(queries, passages.T)
    top_count = min(10, len(corpus_df))
    top_indices = torch.topk(scores, k=top_count, dim=1, sorted=True).indices.cpu()
    passage_ids = corpus_df["pid"].to_numpy()

    relevant: Dict[int, set[int]] = {}
    for row in qrels_df.itertuples(index=False):
        relevant.setdefault(int(row.query_id), set()).add(int(row.relevant_pid))

    reciprocal_ranks: List[float] = []
    for row_index, qid in enumerate(queries_df["query_id"].astype(int)):
        relevant_pids = relevant.get(qid, set())
        reciprocal_rank = 0.0
        for rank, passage_index in enumerate(top_indices[row_index].tolist(), start=1):
            if int(passage_ids[passage_index]) in relevant_pids:
                reciprocal_rank = 1.0 / rank
                break
        reciprocal_ranks.append(reciprocal_rank)
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def _load_frame(data_dir: Path, filename: str, required_columns: set[str]) -> pd.DataFrame:
    path = data_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required training data not found: {path}")
    frame = pd.read_parquet(path)
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"Required training data is empty: {path}")
    return frame


def _create_training_examples(
    triplets: pd.DataFrame,
    negatives: str = "hard",
    corpus_df: pd.DataFrame | None = None,
    seed: int = 42,
) -> List[InputExample]:
    required = {"query_text", "pos_text", "neg_text"}
    missing = required.difference(triplets.columns)
    if missing:
        raise ValueError(f"Training triplets are missing columns: {sorted(missing)}")
    if negatives not in {"hard", "random"}:
        raise ValueError("negatives must be either 'hard' or 'random'")
    if negatives == "random":
        if corpus_df is None or not {"pid", "passage_text"}.issubset(corpus_df.columns):
            raise ValueError(
                "Random negatives require corpus_df with pid and passage_text columns"
            )
        if len(corpus_df) < 2 or corpus_df["pid"].duplicated().any():
            raise ValueError(
                "Random-negative corpus must contain at least two unique passage IDs"
            )
        if "pos_pid" not in triplets.columns:
            raise ValueError("Random negatives require pos_pid in the training triplets")
    if triplets[list(required)].isna().any().any():
        raise ValueError("Training triplet text columns must not contain null values")

    random_negatives: List[str] | None = None
    if negatives == "random" and corpus_df is not None:
        random_negatives = sample_random_negative_texts(
            corpus_df,
            triplets["pos_pid"].astype(int).tolist(),
            seed=seed,
        )

    examples: List[InputExample] = []
    for index, row in enumerate(triplets.itertuples(index=False)):
        query = str(row.query_text).strip()
        positive = str(row.pos_text).strip()
        if negatives == "hard":
            negative = str(row.neg_text).strip()
        else:
            if random_negatives is None:
                raise RuntimeError("Random-negative corpus was not initialized")
            negative = random_negatives[index]
        if not query or not positive or not negative:
            continue
        # MNRL treats column 0 as the anchor, column 1 as its positive, and
        # column 2 as an explicit hard negative. A query-negative pair alone
        # would incorrectly label the negative passage as relevant.
        examples.append(InputExample(texts=[query, positive, negative]))
    if not examples:
        raise ValueError(f"No usable query/positive/{negatives}-negative triplets found")
    return examples


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    batch_size = (
        config.training.batch_size
        if args.batch_size is None
        else args.batch_size
    )
    learning_rate = config.training.lr if args.lr is None else args.lr
    epochs = config.training.epochs if args.epochs is None else args.epochs
    fp16 = config.training.fp16 if args.fp16 is None else args.fp16
    warmup_ratio = (
        config.training.warmup_ratio
        if args.warmup_ratio is None
        else args.warmup_ratio
    )
    patience = (
        config.training.early_stopping_patience
        if args.patience is None
        else args.patience
    )
    if batch_size < 2 or learning_rate <= 0 or epochs <= 0:
        raise ValueError("batch_size must be >= 2; learning rate and epochs must be positive")
    if not 0.0 <= warmup_ratio <= 1.0 or patience < 1:
        raise ValueError("warmup_ratio must be in [0, 1] and patience must be >= 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_dir = Path(args.data_dir)
    triplets_df = _load_frame(
        data_dir,
        "train_triplets.parquet",
        {"query_text", "pos_text", "neg_text"},
    )
    corpus_df = _load_frame(data_dir, "corpus.parquet", {"pid", "passage_text"})
    val_queries = _load_frame(
        data_dir,
        "val_queries.parquet",
        {"query_id", "query_text"},
    )
    val_qrels = _load_frame(
        data_dir,
        "val_qrels.parquet",
        {"query_id", "relevant_pid"},
    )
    if args.val_sample_size <= 0:
        raise ValueError("--val-sample-size must be greater than zero")
    sampled_corpus = sample_validation_corpus(
        corpus_df,
        val_qrels,
        sample_size=args.val_sample_size,
        seed=args.seed,
    )
    examples = _create_training_examples(
        triplets_df,
        negatives=args.negatives,
        corpus_df=corpus_df,
        seed=args.seed,
    )

    model_name = args.model_name or config.models.biencoder_name or DEFAULT_MODEL_NAME
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = fp16 and device == "cuda"
    if fp16 and not use_fp16:
        logger.warning("fp16 was requested, but CUDA is unavailable; using full precision")

    model = SentenceTransformer(model_name, device=device)
    train_loader = DataLoader(
        examples,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=model.smart_batching_collate,
        drop_last=len(examples) % batch_size == 1,
    )
    if not len(train_loader):
        raise ValueError("Training data does not contain a full batch")

    loss_fn = st_losses.MultipleNegativesRankingLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    warmup_steps = math.ceil(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_mrr = float("-inf")
    epochs_without_improvement = 0
    global_step = 0
    metrics: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        window_losses: List[float] = []
        epoch_losses: List[float] = []
        for sentence_features, labels in train_loader:
            sentence_features = [
                {key: value.to(device) for key, value in feature.items()}
                for feature in sentence_features
            ]
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device,
                dtype=torch.float16,
                enabled=use_fp16,
            ):
                loss_value = loss_fn(sentence_features, labels)

            scaler.scale(loss_value).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            loss_scalar = float(loss_value.detach().float().cpu())
            window_losses.append(loss_scalar)
            epoch_losses.append(loss_scalar)
            if global_step % 50 == 0:
                logger.info(
                    "step=%d epoch=%d mean_loss_50_steps=%.6f",
                    global_step,
                    epoch,
                    float(np.mean(window_losses)),
                )
                window_losses.clear()

        if window_losses:
            logger.info(
                "step=%d epoch=%d mean_loss_final_window=%.6f",
                global_step,
                epoch,
                float(np.mean(window_losses)),
            )

        val_mrr = evaluate_validation_mrr(
            model,
            val_queries,
            val_qrels,
            sampled_corpus,
            batch_size=batch_size,
        )
        mean_train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        logger.info(
            "epoch=%d train_loss=%.6f val_mrr@10=%.6f",
            epoch,
            mean_train_loss,
            val_mrr,
        )
        metrics.append(
            {
                "epoch": float(epoch),
                "train_loss": mean_train_loss,
                "val_mrr@10": val_mrr,
            }
        )

        if val_mrr > best_mrr:
            best_mrr = val_mrr
            epochs_without_improvement = 0
            model.save(str(output_dir))
            (output_dir / "training_metrics.json").write_text(
                json.dumps(
                    {
                        "negative_strategy": args.negatives,
                        "best_val_mrr@10": best_mrr,
                        "epochs": metrics,
                        "global_steps": global_step,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            logger.info("Saved new best checkpoint to %s", output_dir)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping after epoch %d", epoch)
                break

    logger.info(
        "Training complete: best val MRR@10=%.6f; checkpoint=%s",
        best_mrr,
        output_dir,
    )


if __name__ == "__main__":
    main()
