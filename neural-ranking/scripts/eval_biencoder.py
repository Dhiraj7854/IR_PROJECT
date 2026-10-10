#!/usr/bin/env python3
"""Evaluate a trained bi-encoder over validation and test query splits."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import write_trec_run
from src.biencoder import (
    DEFAULT_MODEL_PATH,
    configure_retrieval,
    load_biencoder,
    mean_reciprocal_rank,
    retrieve_biencoder_batch,
)
from src.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write TREC runs for a trained dense bi-encoder."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/config.yaml"),
        help="Path to YAML configuration.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to a trained sentence-transformers checkpoint.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "data/processed"),
        help="Directory containing corpus, queries, and qrels parquet files.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=str(PROJECT_ROOT / "artifacts"),
        help="Directory for embedding cache and TREC runs.",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("val", "test"),
        default=("val", "test"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or (args.top_k is not None and args.top_k <= 0):
        raise ValueError("--top-k and --batch-size must be greater than zero")

    model_path = Path(args.model)
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Trained bi-encoder checkpoint was not found: {model_path}"
        )
    data_dir = Path(args.data_dir)
    artifacts_dir = Path(args.artifacts_dir)
    config = load_config(args.config)
    top_k = (
        config.retrieval.stage1_top_k
        if args.top_k is None
        else args.top_k
    )
    if top_k <= 0:
        raise ValueError("--top-k must be greater than zero")

    corpus_path = data_dir / "corpus.parquet"
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    corpus_df = pd.read_parquet(corpus_path)
    required_corpus_columns = {"pid", "passage_text"}
    if not required_corpus_columns.issubset(corpus_df.columns):
        raise ValueError(
            f"Corpus must contain columns {sorted(required_corpus_columns)}"
        )

    model = load_biencoder(model_path)
    embedding_cache = artifacts_dir / "biencoder" / "corpus_embeddings.npy"
    configure_retrieval(
        corpus_df["passage_text"].astype(str).tolist(),
        pids=corpus_df["pid"].tolist(),
        batch_size=args.batch_size,
        model=model,
        cache_path=embedding_cache,
    )

    runs_dir = artifacts_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        queries_path = data_dir / f"{split}_queries.parquet"
        qrels_path = data_dir / f"{split}_qrels.parquet"
        if not queries_path.is_file():
            raise FileNotFoundError(f"Query split not found: {queries_path}")
        if not qrels_path.is_file():
            raise FileNotFoundError(f"Qrels split not found: {qrels_path}")
        queries_df = pd.read_parquet(queries_path)
        qrels_df = pd.read_parquet(qrels_path)
        if not {"query_id", "query_text"}.issubset(queries_df.columns):
            raise ValueError(f"{queries_path} must contain query_id and query_text")
        if not {"query_id", "relevant_pid"}.issubset(qrels_df.columns):
            raise ValueError(f"{qrels_path} must contain query_id and relevant_pid")

        hits = retrieve_biencoder_batch(
            queries_df["query_text"].astype(str).tolist(),
            k=top_k,
            batch_size=args.batch_size,
        )
        results = {
            int(qid): query_hits
            for qid, query_hits in zip(queries_df["query_id"], hits)
        }
        mrr = mean_reciprocal_rank(
            hits,
            queries_df["query_id"].astype(int).tolist(),
            qrels_df,
            k=10,
        )
        run_path = runs_dir / f"biencoder_{split}.txt"
        write_trec_run(results, run_path, system_name="biencoder")
        logger.info(
            "%s: MRR@10=%.6f over %d queries; wrote %s",
            split,
            mrr,
            len(queries_df),
            run_path,
        )


if __name__ == "__main__":
    main()
