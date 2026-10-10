#!/usr/bin/env python3
"""Run TF-IDF and BM25 baselines over val and test splits; write TREC run files."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.baseline import (
    build_index,
    retrieve_all,
    retrieve_bm25,
    retrieve_tfidf,
    write_trec_run,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run TF-IDF and BM25 baselines and write TREC run files."
    )
    p.add_argument(
        "--corpus",
        default=str(PROJECT_ROOT / "data/processed/corpus.parquet"),
        help="Path to corpus parquet.",
    )
    p.add_argument(
        "--data_dir",
        default=str(PROJECT_ROOT / "data/processed"),
        help="Directory containing val/test query parquets.",
    )
    p.add_argument(
        "--artifacts_dir",
        default=str(PROJECT_ROOT / "artifacts"),
        help="Directory for index cache and run-file output.",
    )
    p.add_argument(
        "--top_k",
        "--top-k",
        type=int,
        default=100,
        help="Retrieval depth (number of passages per query).",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test"],
        choices=["train", "val", "test"],
        help="Which splits to evaluate.",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=["tfidf", "bm25"],
        choices=["tfidf", "bm25"],
        help="Which retrieval methods to run.",
    )
    p.add_argument(
        "--force_rebuild",
        action="store_true",
        help="Ignore cached indexes and rebuild from scratch.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top_k must be greater than zero")

    data_dir = Path(args.data_dir)
    artifacts_dir = Path(args.artifacts_dir)
    runs_dir = artifacts_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Build / load indexes
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("Building / loading baseline indexes…")
    timings = build_index(
        corpus_path=args.corpus,
        artifacts_dir=artifacts_dir,
        force_rebuild=args.force_rebuild,
    )
    logger.info(
        "Index ready — total indexing/load time: %.2fs "
        "(TF-IDF build: %.2fs, BM25 build: %.2fs)",
        timings["total_indexing_s"],
        timings.get("tfidf_build_s", 0.0),
        timings.get("bm25_build_s", 0.0),
    )

    # ------------------------------------------------------------------ #
    # 2. Evaluate every (method, split) combination
    # ------------------------------------------------------------------ #
    for split in args.splits:
        q_path = data_dir / f"{split}_queries.parquet"
        if not q_path.exists():
            raise FileNotFoundError(f"Query split not found: {q_path}")
        queries_df = pd.read_parquet(q_path)
        logger.info("Loaded %d %s queries.", len(queries_df), split)

        for method in args.methods:
            logger.info("--- %s / %s ---", method.upper(), split)
            results, lat = retrieve_all(queries_df, method=method, k=args.top_k)

            logger.info(
                "  Latency: mean=%.2f ms  median=%.2f ms  p95=%.2f ms  total=%.2fs",
                lat["mean_ms"], lat["median_ms"], lat["p95_ms"], lat["total_s"],
            )

            run_file = runs_dir / f"{method}_{split}.txt"
            write_trec_run(results, run_file, system_name=method)
            logger.info("  Run file written → %s", run_file)

    # ------------------------------------------------------------------ #
    # 3. Spot-check: print top-5 passages for 3 val queries
    # ------------------------------------------------------------------ #
    if "val" in args.splits:
        logger.info("=" * 60)
        logger.info("Spot-check: top-5 results for 3 val queries")
        val_q = pd.read_parquet(data_dir / "val_queries.parquet")
        corpus_df = pd.read_parquet(args.corpus)
        pid_to_text: dict = dict(zip(corpus_df["pid"], corpus_df["passage_text"]))

        for _, row in val_q.head(3).iterrows():
            qid = int(row["query_id"])
            q = str(row["query_text"])
            print(f"\nQID={qid}  Query: {q!r}")
            for method, retrieve in (
                ("tfidf", retrieve_tfidf),
                ("bm25", retrieve_bm25),
            ):
                print(f"  {method.upper()}")
                for rank, (pid, score) in enumerate(retrieve(q, k=5), 1):
                    snippet = pid_to_text.get(pid, "")[:120].replace("\n", " ")
                    print(f"    {rank}. pid={pid}  score={score:.4f}  |  {snippet}...")

    logger.info("Done.")


if __name__ == "__main__":
    main()
