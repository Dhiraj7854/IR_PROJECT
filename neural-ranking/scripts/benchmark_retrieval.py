#!/usr/bin/env python3
"""Measure warm-query latency, throughput, memory, and retrieval quality."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import build_index, get_index
from src.biencoder import configure_retrieval, load_biencoder, retrieve_biencoder
from src.config import load_config
from src.evaluate import average_metrics, evaluate_run, load_qrels
from src.hybrid import fuse_ranked_lists

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark comparable retrieval methods on a fixed query split."
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/config.yaml"))
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data/processed"))
    parser.add_argument("--artifacts-dir", default=str(PROJECT_ROOT / "artifacts"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("tfidf", "bm25", "biencoder", "rrf"),
        default=("tfidf", "bm25"),
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--query-limit", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _synchronize() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    if min(args.top_k, args.query_limit) <= 0 or args.warmup < 0:
        raise ValueError("top-k/query-limit must be positive and warmup non-negative")
    data_dir = Path(args.data_dir)
    artifacts_dir = Path(args.artifacts_dir)
    queries_path = data_dir / f"{args.split}_queries.parquet"
    qrels_path = data_dir / f"{args.split}_qrels.parquet"
    corpus_path = data_dir / "corpus.parquet"
    for path in (queries_path, qrels_path, corpus_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required benchmark input is missing: {path}")

    config = load_config(args.config)
    corpus = pd.read_parquet(corpus_path)
    queries = pd.read_parquet(queries_path).head(args.query_limit)
    qrels = load_qrels(qrels_path)
    build_index(corpus_df=corpus, artifacts_dir=artifacts_dir)
    index = get_index()
    if "biencoder" in args.methods or "rrf" in args.methods:
        model_path = Path(args.model or artifacts_dir / "biencoder")
        if not model_path.is_dir():
            raise FileNotFoundError(f"Bi-encoder checkpoint not found: {model_path}")
        model = load_biencoder(model_path)
        configure_retrieval(
            corpus["passage_text"].astype(str).tolist(),
            pids=corpus["pid"].tolist(),
            model=model,
            cache_path=model_path / "corpus_embeddings.npy",
        )

    retrievers: Dict[str, Callable[[str], List[Tuple[int, float]]]] = {
        "tfidf": lambda query: index.retrieve_tfidf(query, args.top_k),
        "bm25": lambda query: index.retrieve_bm25(query, args.top_k),
    }
    if "biencoder" in args.methods:
        retrievers["biencoder"] = lambda query: retrieve_biencoder(query, args.top_k)
    if "rrf" in args.methods:
        retrievers["rrf"] = lambda query: [
            (item["pid"], item["score"])
            for item in fuse_ranked_lists(
                [
                    index.retrieve_bm25(query, args.top_k),
                    retrieve_biencoder(query, args.top_k),
                ],
                k=config.retrieval.rrf_k,
                weights=[
                    (config.retrieval.rrf_weights or {}).get("bm25", 1.0),
                    (config.retrieval.rrf_weights or {}).get("biencoder", 1.0),
                ],
                source_names=["bm25", "biencoder"],
            )[: args.top_k]
        ]

    query_rows = [
        (str(row.query_id), str(row.query_text))
        for row in queries.itertuples(index=False)
    ]
    results: Dict[str, object] = {
        "dataset": config.dataset.name,
        "split": args.split,
        "query_count": len(query_rows),
        "top_k": args.top_k,
        "warmup_queries": args.warmup,
        "hardware": "CUDA" if __import__("torch").cuda.is_available() else "CPU",
        "methods": {},
    }
    for name in args.methods:
        retriever = retrievers[name]
        for _, query in query_rows[: args.warmup]:
            retriever(query)
        latencies = []
        run: Dict[str, Dict[str, float]] = {}
        tracemalloc.start()
        started_all = time.perf_counter()
        for qid, query in query_rows:
            _synchronize()
            started = time.perf_counter()
            hits = retriever(query)
            _synchronize()
            latencies.append((time.perf_counter() - started) * 1000.0)
            run[qid] = {str(pid): float(score) for pid, score in hits}
        wall_seconds = time.perf_counter() - started_all
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        matched_qrels = {qid: qrels[qid] for qid, _ in query_rows if qid in qrels}
        quality = average_metrics(evaluate_run(matched_qrels, run)) if matched_qrels else {}
        recall_100_valid = args.top_k >= min(100, len(corpus))
        if not recall_100_valid:
            quality.pop("Recall@100", None)
        values = {
            "warm_query_latency_ms_mean": statistics.mean(latencies) if latencies else 0.0,
            "p50_latency_ms": statistics.median(latencies) if latencies else 0.0,
            "p95_latency_ms": (
                float(pd.Series(latencies).quantile(0.95)) if latencies else 0.0
            ),
            "throughput_queries_per_second": (
                len(query_rows) / wall_seconds if wall_seconds else 0.0
            ),
            "python_tracemalloc_peak_mib": peak_bytes / (1024 * 1024),
            "recall_100_valid": recall_100_valid,
            "quality": quality,
        }
        results["methods"][name] = values
        logger.info("%s benchmark: %s", name, json.dumps(values))

    output_path = Path(args.output) if args.output else (
        artifacts_dir / "benchmarks" / f"retrieval_{args.split}_{int(time.time())}.json"
    )
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Benchmark output exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote benchmark output to %s", output_path)


if __name__ == "__main__":
    main()
