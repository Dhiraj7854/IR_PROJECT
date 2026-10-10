#!/usr/bin/env python3
"""BM25 retrieval followed by cross-encoder re-ranking for val and test."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import build_index, get_index, write_trec_run
from src.config import load_config
from src.crossencoder import load_crossencoder, rerank

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve BM25 candidates, then cross-encoder re-rank them."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/config.yaml"),
        help="Path to YAML configuration.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "data/processed"),
        help="Directory containing corpus and query parquet files.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=str(PROJECT_ROOT / "artifacts"),
        help="Directory containing checkpoints and output runs.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Cross-encoder checkpoint directory.",
    )
    parser.add_argument("--candidate-depth", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("val", "test"),
        default=("val", "test"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.top_k <= 0
        or args.batch_size <= 0
        or (args.candidate_depth is not None and args.candidate_depth <= 0)
    ):
        raise ValueError("Retrieval depth, rerank depth, and batch size must be positive")

    config = load_config(args.config)
    candidate_depth = (
        config.retrieval.stage1_top_k
        if args.candidate_depth is None
        else args.candidate_depth
    )
    model_path = Path(args.model) if args.model else Path(args.artifacts_dir) / "crossencoder"
    if not model_path.is_dir():
        raise FileNotFoundError(f"Cross-encoder checkpoint not found: {model_path}")

    data_dir = Path(args.data_dir)
    corpus_path = data_dir / "corpus.parquet"
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    corpus_df = pd.read_parquet(corpus_path)
    if not {"pid", "passage_text"}.issubset(corpus_df.columns):
        raise ValueError("Corpus must contain pid and passage_text columns")
    passage_text_by_pid: Dict[int, str] = dict(
        zip(corpus_df["pid"].astype(int), corpus_df["passage_text"].astype(str))
    )

    artifacts_dir = Path(args.artifacts_dir)
    build_index(
        corpus_df,
        artifacts_dir=artifacts_dir,
    )
    ranker, tokenizer = load_crossencoder(model_path)
    run_dir = artifacts_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        queries_path = data_dir / f"{split}_queries.parquet"
        if not queries_path.is_file():
            raise FileNotFoundError(f"Query split not found: {queries_path}")
        queries_df = pd.read_parquet(queries_path)
        if not {"query_id", "query_text"}.issubset(queries_df.columns):
            raise ValueError(f"{queries_path} must contain query_id and query_text")

        run_results: Dict[int, List[Tuple[int, float]]] = {}
        latencies_ms: List[float] = []
        for query in queries_df.itertuples(index=False):
            query_text = str(query.query_text)
            candidates = get_index().retrieve_bm25(
                query_text,
                k=candidate_depth,
            )
            candidate_pids = [pid for pid, _ in candidates]
            candidate_texts = [passage_text_by_pid[pid] for pid in candidate_pids]

            started = time.perf_counter()
            ranked = rerank(
                query_text,
                candidate_pids,
                candidate_texts,
                k=args.top_k,
                model=ranker,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_length=config.models.max_seq_len,
                fp16=True,
            )
            if next(ranker.parameters()).device.type == "cuda":
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - started) * 1000.0)
            run_results[int(query.query_id)] = ranked

        run_path = run_dir / f"crossencoder_{split}.txt"
        write_trec_run(run_results, run_path, system_name="crossencoder")
        mean_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0
        logger.info(
            "%s: re-ranking %d BM25 candidates averaged %.2f ms/query; "
            "wrote %d queries to %s",
            split,
            candidate_depth,
            mean_latency,
            len(run_results),
            run_path,
        )


if __name__ == "__main__":
    main()
