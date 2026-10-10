#!/usr/bin/env python3
"""Compare all retrieval systems on test metrics and measured latency."""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import build_index, get_index
from src.biencoder import (
    configure_retrieval,
    load_biencoder,
    retrieve_biencoder,
)
from src.config import load_config
from src.crossencoder import load_crossencoder, rerank
from src.evaluate import average_metrics, evaluate_run, load_qrels, load_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SYSTEMS = (
    ("TF-IDF", "tfidf_test.txt"),
    ("BM25", "bm25_test.txt"),
    ("Bi-encoder", "biencoder_test.txt"),
    ("BM25 → Cross-encoder", "crossencoder_test.txt"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all retrieval systems and plot quality vs latency."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/config.yaml"),
        help="Path to YAML configuration.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "data/processed"),
        help="Directory containing test queries, qrels, and corpus.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=str(PROJECT_ROOT / "artifacts"),
        help="Directory containing system runs and model checkpoints.",
    )
    parser.add_argument("--latency-queries", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--biencoder-batch-size", type=int, default=64)
    return parser.parse_args()


def _sync_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _median_latency_ms(
    queries: List[Tuple[int, str]],
    system_call: Callable[[str], object],
) -> float:
    latencies: List[float] = []
    for _, query in queries:
        _sync_if_cuda()
        started = time.perf_counter()
        system_call(query)
        _sync_if_cuda()
        latencies.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(latencies)


def _format_markdown(rows: List[Dict[str, object]]) -> str:
    columns = [
        ("System", "system"),
        ("MRR@10", "MRR@10"),
        ("NDCG@10", "NDCG@10"),
        ("P@5", "P@5"),
        ("Recall@100", "Recall@100"),
        ("Median latency (ms/query)", "median_latency_ms"),
        ("Latency sample", "latency_queries"),
        ("Flag", "flag"),
    ]
    lines = [
        "| " + " | ".join(label for label, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for _, key in columns:
            value = row[key]
            if key in {"MRR@10", "NDCG@10", "P@5", "Recall@100"}:
                values.append(f"{float(value):.4f}")
            elif key == "median_latency_ms":
                values.append(f"{float(value):.2f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.latency_queries <= 0:
        raise ValueError("--latency-queries must be greater than zero")
    if args.batch_size <= 0 or args.biencoder_batch_size <= 0:
        raise ValueError("Batch sizes must be greater than zero")

    config = load_config(args.config)
    artifacts_dir = Path(args.artifacts_dir)
    data_dir = Path(args.data_dir)
    test_queries_path = data_dir / "test_queries.parquet"
    test_qrels_path = data_dir / "test_qrels.parquet"
    corpus_path = data_dir / "corpus.parquet"
    required_paths = [test_queries_path, test_qrels_path, corpus_path]
    run_paths = [artifacts_dir / "runs" / filename for _, filename in SYSTEMS]
    missing_paths = [path for path in required_paths + run_paths if not path.is_file()]
    if missing_paths:
        missing_list = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Run all retrieval/training/evaluation stages before comparison; "
            f"required files are missing:\n{missing_list}"
        )

    crossencoder_path = artifacts_dir / "crossencoder"
    biencoder_path = artifacts_dir / "biencoder"
    if not crossencoder_path.is_dir():
        raise FileNotFoundError(
            f"Cross-encoder checkpoint is required for latency measurement: {crossencoder_path}"
        )
    if not biencoder_path.is_dir():
        raise FileNotFoundError(
            f"Bi-encoder checkpoint is required for latency measurement: {biencoder_path}"
        )

    queries_df = pd.read_parquet(test_queries_path)
    if not {"query_id", "query_text"}.issubset(queries_df.columns):
        raise ValueError("test_queries.parquet must contain query_id and query_text")
    qrels = load_qrels(test_qrels_path)
    run_contents = {label: load_run(path) for (label, _), path in zip(SYSTEMS, run_paths)}
    query_sample = [
        (int(row.query_id), str(row.query_text))
        for row in queries_df.head(args.latency_queries).itertuples(index=False)
    ]
    if len(query_sample) < args.latency_queries:
        logger.warning(
            "Only %d test queries are available for latency measurement (requested %d)",
            len(query_sample),
            args.latency_queries,
        )

    corpus_df = pd.read_parquet(corpus_path)
    if not {"pid", "passage_text"}.issubset(corpus_df.columns):
        raise ValueError("corpus.parquet must contain pid and passage_text")
    passage_text_by_pid = dict(
        zip(corpus_df["pid"].astype(int), corpus_df["passage_text"].astype(str))
    )

    logger.info("Preparing shared lexical indexes")
    build_index(corpus_df, artifacts_dir=artifacts_dir)
    index = get_index()

    logger.info("Loading bi-encoder and cached corpus embeddings")
    biencoder_model = load_biencoder(biencoder_path)
    configure_retrieval(
        corpus_df["passage_text"].astype(str).tolist(),
        pids=corpus_df["pid"].tolist(),
        batch_size=args.biencoder_batch_size,
        model=biencoder_model,
        cache_path=biencoder_path / "corpus_embeddings.npy",
    )

    logger.info("Loading cross-encoder")
    crossencoder_model, crossencoder_tokenizer = load_crossencoder(crossencoder_path)
    latency_by_system: Dict[str, float] = {}

    for method, label in (("tfidf", "TF-IDF"), ("bm25", "BM25")):
        retrieval = (
            index.retrieve_tfidf if method == "tfidf" else index.retrieve_bm25
        )
        latency_by_system[label] = _median_latency_ms(
            query_sample,
            lambda query, fn=retrieval: fn(query, k=100),
        )

    latency_by_system["Bi-encoder"] = _median_latency_ms(
        query_sample,
        lambda query: retrieve_biencoder(query, k=100),
    )

    def bm25_crossencoder(query: str) -> object:
        candidates = index.retrieve_bm25(query, k=100)
        candidate_pids = [pid for pid, _ in candidates]
        candidate_texts = [passage_text_by_pid[pid] for pid in candidate_pids]
        return rerank(
            query,
            candidate_pids,
            candidate_texts,
            k=10,
            model=crossencoder_model,
            tokenizer=crossencoder_tokenizer,
            batch_size=args.batch_size,
            max_length=config.models.max_seq_len,
            fp16=True,
        )

    latency_by_system["BM25 → Cross-encoder"] = _median_latency_ms(
        query_sample,
        bm25_crossencoder,
    )

    rows: List[Dict[str, object]] = []
    per_query_rows: List[Dict[str, object]] = []
    for system, _ in SYSTEMS:
        per_query = evaluate_run(qrels, run_contents[system])
        averages = average_metrics(per_query)
        per_query_rows.extend(
            {
                "system": system,
                "query_id": qid,
                **metrics,
            }
            for qid, metrics in per_query.items()
        )
        row: Dict[str, object] = {
            "system": system,
            **averages,
            "median_latency_ms": latency_by_system[system],
            "latency_queries": len(query_sample),
            "flag": "",
        }
        rows.append(row)
        logger.info(
            "%s MRR@10=%.4f NDCG@10=%.4f P@5=%.4f Recall@100=%.4f "
            "median_latency=%.2f ms/query",
            system,
            averages["MRR@10"],
            averages["NDCG@10"],
            averages["P@5"],
            averages["Recall@100"],
            latency_by_system[system],
        )

    bm25_mrr = next(row["MRR@10"] for row in rows if row["system"] == "BM25")
    flags: List[str] = []
    if not 0.15 <= float(bm25_mrr) <= 0.35:
        message = (
            f"BM25 MRR@10={float(bm25_mrr):.4f} is outside the expected "
            "0.15–0.35 sanity range"
        )
        logger.warning("SUSPICIOUS RESULT: %s", message)
        flags.append(message)
    high_systems = [
        str(row["system"]) for row in rows if float(row["MRR@10"]) > 0.8
    ]
    if high_systems:
        message = f"MRR@10 above 0.8 for: {', '.join(high_systems)}"
        logger.warning("SUSPICIOUS RESULT: %s", message)
        flags.append(message)
    tfidf_mrr = next(row["MRR@10"] for row in rows if row["system"] == "TF-IDF")
    best_other = max(
        float(row["MRR@10"])
        for row in rows
        if row["system"] != "TF-IDF"
    )
    if float(tfidf_mrr) > best_other:
        message = (
            f"TF-IDF MRR@10={float(tfidf_mrr):.4f} exceeds every other system "
            f"(best other={best_other:.4f})"
        )
        logger.warning("SUSPICIOUS RESULT: %s", message)
        flags.append(message)
    if flags:
        rows[0]["flag"] = "WARNING: " + "; ".join(flags)

    results_dir = artifacts_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(results_dir / "comparison_table.csv", index=False)
    pd.DataFrame(per_query_rows).to_csv(
        results_dir / "per_query_metrics.csv",
        index=False,
    )
    (results_dir / "comparison_table.md").write_text(
        "# Test split system comparison\n\n"
        + _format_markdown(rows)
        + "\n"
        + (
            "**Suspicious-result checks:** " + "; ".join(flags) + "\n"
            if flags
            else "No configured suspicious-result checks were triggered.\n"
        ),
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(8, 5))
    for row in rows:
        axis.scatter(
            float(row["median_latency_ms"]),
            float(row["MRR@10"]),
            s=70,
        )
        axis.annotate(
            str(row["system"]),
            (float(row["median_latency_ms"]), float(row["MRR@10"])),
            xytext=(6, 6),
            textcoords="offset points",
        )
    axis.set_xscale("log")
    axis.set_xlabel("Median end-to-end latency (ms/query, log scale)")
    axis.set_ylabel("MRR@10")
    axis.set_title("Test split: retrieval quality vs latency")
    axis.grid(True, which="both", linestyle="--", alpha=0.35)
    figure.tight_layout()
    figure.savefig(results_dir / "pareto.png", dpi=180)
    plt.close(figure)
    logger.info("Saved comparison table and Pareto plot under %s", results_dir)


if __name__ == "__main__":
    main()
