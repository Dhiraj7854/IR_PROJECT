#!/usr/bin/env python3
"""Compare opt-in query transformations against unmodified queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import build_index, get_index, write_trec_run
from src.config import load_config
from src.evaluate import average_metrics, evaluate_run, load_qrels, paired_bootstrap_interval
from src.query_enhancement import enhance_query, load_synonyms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate query enhancement with the same queries and qrels."
    )
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs/config.yaml")
    )
    parser.add_argument(
        "--data-dir", default=str(PROJECT_ROOT / "data/processed")
    )
    parser.add_argument(
        "--artifacts-dir", default=str(PROJECT_ROOT / "artifacts")
    )
    parser.add_argument("--retriever", choices=("bm25", "tfidf"), default="bm25")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("original", "normalize", "spell", "synonym"),
        default=("original", "normalize", "synonym"),
    )
    parser.add_argument(
        "--synonyms",
        default=str(PROJECT_ROOT / "configs/synonyms.json"),
        help="Editable JSON synonym dictionary.",
    )
    parser.add_argument(
        "--splits", nargs="+", choices=("val", "test"), default=("val", "test")
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    data_dir = Path(args.data_dir)
    artifacts_dir = Path(args.artifacts_dir)
    config = load_config(args.config)
    config_path = Path(args.config)
    if not config_path.is_file():
        config_path = PROJECT_ROOT / config_path
    corpus_path = data_dir / "corpus.parquet"
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    corpus = pd.read_parquet(corpus_path)
    build_index(corpus_df=corpus, artifacts_dir=artifacts_dir)
    index = get_index()
    retrieval = index.retrieve_bm25 if args.retriever == "bm25" else index.retrieve_tfidf
    synonyms = load_synonyms(Path(args.synonyms)) if "synonym" in args.methods else {}
    output_dir = artifacts_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        queries_path = data_dir / f"{split}_queries.parquet"
        qrels_path = data_dir / f"{split}_qrels.parquet"
        if not queries_path.is_file() or not qrels_path.is_file():
            raise FileNotFoundError(f"Missing query or qrels file for split {split}")
        queries = pd.read_parquet(queries_path)
        qrels = load_qrels(qrels_path)
        comparison_path = output_dir / f"query_{args.retriever}_comparison_{split}.json"
        output_files = [
            output_dir / f"query_{args.retriever}_{method}_{split}.txt"
            for method in args.methods
        ]
        output_files.extend(path.with_suffix(".metrics.json") for path in output_files)
        output_files.append(comparison_path)
        existing = [path for path in output_files if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(
                "Query-enhancement outputs exist; pass --overwrite explicitly: "
                + ", ".join(str(path) for path in existing)
            )
        per_query_by_method = {}
        summary_by_method = {}
        for method in args.methods:
            run_path = output_dir / f"query_{args.retriever}_{method}_{split}.txt"
            metrics_path = run_path.with_suffix(".metrics.json")
            results: Dict[int, List[Tuple[int, float]]] = {}
            transformed: Dict[str, str] = {}
            latencies_ms = []
            for row in queries.itertuples(index=False):
                original = str(row.query_text or "")
                updated = enhance_query(
                    original,
                    method,
                    synonyms=synonyms,
                )
                transformed[str(row.query_id)] = updated
                started = time.perf_counter()
                results[int(row.query_id)] = retrieval(updated, k=args.top_k)
                latencies_ms.append((time.perf_counter() - started) * 1000.0)
            write_trec_run(
                results,
                run_path,
                system_name=f"{args.retriever}_{method}",
            )
            per_query = evaluate_run(qrels, run_path)
            averages = average_metrics(per_query)
            recall_100_valid = args.top_k >= min(100, len(corpus))
            if not recall_100_valid:
                averages.pop("Recall@100", None)
                for metrics in per_query.values():
                    metrics.pop("Recall@100", None)
            summary = {
                "dataset": config.dataset.name,
                "split": split,
                "retriever": args.retriever,
                "enhancement": method,
                "query_count": len(queries),
                "top_k": args.top_k,
                "qrels_sha256": hashlib.sha256(qrels_path.read_bytes()).hexdigest(),
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "seed": 42,
                "average_metrics": averages,
                "per_query_metrics": per_query,
                "Recall@100_valid": recall_100_valid,
                "mean_latency_ms": (
                    sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
                ),
                "transformed_queries": transformed,
                "run_file": str(run_path),
            }
            metrics_path.write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            per_query_by_method[method] = per_query
            summary_by_method[method] = summary
            logger.info(
                "%s/%s MRR@10=%.4f mean latency=%.2f ms; wrote %s",
                split,
                method,
                summary["average_metrics"]["MRR@10"],
                summary["mean_latency_ms"],
                run_path,
            )
        comparison = {}
        original_metrics = per_query_by_method.get("original")
        for method, per_query in per_query_by_method.items():
            if method == "original" or original_metrics is None:
                continue
            metric_changes = {}
            for metric in ("MRR@10", "NDCG@10", "Recall@10", "MAP@10", "Recall@100"):
                try:
                    metric_changes[metric] = paired_bootstrap_interval(
                        original_metrics,
                        per_query,
                        metric,
                        seed=42,
                    )
                except ValueError:
                    continue
            comparison[method] = {
                "metric_change_vs_original": metric_changes,
                "mean_latency_change_ms": (
                    summary_by_method[method]["mean_latency_ms"]
                    - summary_by_method["original"]["mean_latency_ms"]
                ),
                "run_file": summary_by_method[method]["run_file"],
            }
        comparison_path.write_text(
            json.dumps(
                {
                    "dataset": config.dataset.name,
                    "split": split,
                    "retriever": args.retriever,
                    "top_k": args.top_k,
                    "original_run": summary_by_method.get("original", {}).get("run_file"),
                    "changes_vs_original": comparison,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
