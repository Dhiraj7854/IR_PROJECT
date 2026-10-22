#!/usr/bin/env python3
"""Compare all retrieval systems on test metrics and measured latency."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
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
from src.evaluate import (
    average_metrics,
    evaluate_run,
    load_qrels,
    load_run,
    paired_bootstrap_interval,
)
from src.hybrid import fuse_ranked_lists

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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing comparison outputs (they are preserved by default).",
    )
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
        ("Recall@10", "Recall@10"),
        ("Recall@100", "Recall@100"),
        ("MAP@10", "MAP@10"),
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
            value = row.get(key)
            if key in {"MRR@10", "NDCG@10", "P@5", "Recall@10", "Recall@100", "MAP@10"}:
                values.append("not available" if value is None else f"{float(value):.4f}")
            elif key == "median_latency_ms":
                values.append(f"{float(value):.2f}")
            elif value is None:
                values.append("not available")
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
    results_dir = artifacts_dir / "results"
    comparison_outputs = [
        results_dir / "comparison_table.csv",
        results_dir / "comparison_table.md",
        results_dir / "per_query_metrics.csv",
        results_dir / "pareto.png",
        results_dir / "evaluation_manifest.json",
        results_dir / "paired_differences.json",
    ]
    existing_outputs = [path for path in comparison_outputs if path.exists()]
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            "Comparison results already exist; use --overwrite explicitly: "
            + ", ".join(str(path) for path in existing_outputs)
        )
    test_queries_path = data_dir / "test_queries.parquet"
    test_qrels_path = data_dir / "test_qrels.parquet"
    corpus_path = data_dir / "corpus.parquet"
    required_paths = [test_queries_path, test_qrels_path, corpus_path]
    run_specs = list(SYSTEMS)
    hybrid_depth = config.retrieval.stage1_top_k
    rerank_depth = config.retrieval.rerank_depth
    optional_runs = [
        ("Hybrid RRF", f"hybrid_rrf_c{hybrid_depth}_test.txt"),
        (
            "Hybrid RRF → Cross-encoder",
            f"hybrid_crossencoder_c{hybrid_depth}_r{rerank_depth}_test.txt",
        ),
        (
            "BM25 → Cross-encoder (fixed candidates)",
            f"bm25_crossencoder_c{hybrid_depth}_r{rerank_depth}_test.txt",
        ),
    ]
    for label, filename in optional_runs:
        path = artifacts_dir / "runs" / filename
        if path.is_file():
            run_specs.append((label, filename))
    run_paths = [artifacts_dir / "runs" / filename for _, filename in run_specs]
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
    run_contents = {label: load_run(path) for (label, _), path in zip(run_specs, run_paths)}
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
    required_run_depth = min(100, len(corpus_df))

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
    if "Hybrid RRF" in run_contents or "Hybrid RRF → Cross-encoder" in run_contents:
        def hybrid_candidates(query: str) -> List[Tuple[int, float]]:
            bm25_hits = index.retrieve_bm25(query, k=hybrid_depth)
            dense_hits = retrieve_biencoder(query, k=hybrid_depth)
            fused = fuse_ranked_lists(
                [bm25_hits, dense_hits],
                k=config.retrieval.rrf_k,
                weights=[
                    (config.retrieval.rrf_weights or {}).get("bm25", 1.0),
                    (config.retrieval.rrf_weights or {}).get("biencoder", 1.0),
                ],
                source_names=["bm25", "biencoder"],
            )
            return [(entry["pid"], entry["score"]) for entry in fused]

        if "Hybrid RRF" in run_contents:
            latency_by_system["Hybrid RRF"] = _median_latency_ms(
                query_sample,
                lambda query: hybrid_candidates(query)[:100],
            )
        if "Hybrid RRF → Cross-encoder" in run_contents:
            def rrf_crossencoder(query: str) -> object:
                candidates = hybrid_candidates(query)[:rerank_depth]
                return rerank(
                    query,
                    [pid for pid, _ in candidates],
                    [passage_text_by_pid[pid] for pid, _ in candidates],
                    k=100,
                    model=crossencoder_model,
                    tokenizer=crossencoder_tokenizer,
                    batch_size=args.batch_size,
                    max_length=config.models.max_seq_len,
                    fp16=True,
                )

            latency_by_system["Hybrid RRF → Cross-encoder"] = _median_latency_ms(
                query_sample,
                rrf_crossencoder,
            )
    if "BM25 → Cross-encoder (fixed candidates)" in run_contents:
        def fixed_bm25_crossencoder(query: str) -> object:
            candidates = index.retrieve_bm25(query, k=hybrid_depth)[:rerank_depth]
            return rerank(
                query,
                [pid for pid, _ in candidates],
                [passage_text_by_pid[pid] for pid, _ in candidates],
                k=100,
                model=crossencoder_model,
                tokenizer=crossencoder_tokenizer,
                batch_size=args.batch_size,
                max_length=config.models.max_seq_len,
                fp16=True,
            )

        latency_by_system["BM25 → Cross-encoder (fixed candidates)"] = _median_latency_ms(
            query_sample,
            fixed_bm25_crossencoder,
        )

    rows: List[Dict[str, object]] = []
    per_query_rows: List[Dict[str, object]] = []
    per_query_by_system: Dict[str, Dict[str, Dict[str, float]]] = {}
    for system, _ in run_specs:
        per_query = evaluate_run(qrels, run_contents[system])
        run = run_contents[system]
        has_valid_top100 = all(
            not run.get(qid) or len(run[qid]) >= required_run_depth
            for qid in qrels
        )
        if not has_valid_top100:
            for metrics in per_query.values():
                metrics.pop("Recall@100", None)
            logger.warning(
                "%s run is truncated below %d documents for at least one query; "
                "Recall@100 will not be reported",
                system,
                required_run_depth,
            )
        averages = average_metrics(per_query)
        per_query_by_system[system] = per_query
        per_query_rows.extend(
            {
                "system": system,
                "query_id": qid,
                "Recall@100_valid": has_valid_top100,
                **metrics,
            }
            for qid, metrics in per_query.items()
        )
        if system not in latency_by_system:
            logger.warning("No measured latency available for %s; skipping comparison row", system)
            continue
        row: Dict[str, object] = {
            "system": system,
            **averages,
            "Recall@100_valid": has_valid_top100,
            "median_latency_ms": latency_by_system[system],
            "latency_queries": len(query_sample),
            "flag": "",
        }
        rows.append(row)
        logger.info(
            "%s MRR@10=%.4f NDCG@10=%.4f P@5=%.4f Recall@100=%s "
            "median_latency=%.2f ms/query",
            system,
            averages["MRR@10"],
            averages["NDCG@10"],
            averages["P@5"],
            f"{averages['Recall@100']:.4f}" if "Recall@100" in averages else "not available",
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
    manifest = {
        "dataset": config.dataset.name,
        "split": "test",
        "query_count": len(queries_df),
        "qrels_sha256": hashlib.sha256(test_qrels_path.read_bytes()).hexdigest(),
        "queries_sha256": hashlib.sha256(test_queries_path.read_bytes()).hexdigest(),
        "seed": args.seed,
        "effective_config": asdict(config),
        "config_sha256": hashlib.sha256(
            (Path(args.config) if Path(args.config).is_file() else PROJECT_ROOT / args.config).read_bytes()
        ).hexdigest(),
        "config": {
            "biencoder": str(biencoder_path),
            "crossencoder": str(crossencoder_path),
            "candidate_depth": hybrid_depth,
            "rerank_depth": rerank_depth,
            "rrf_k": config.retrieval.rrf_k,
            "rrf_weights": config.retrieval.rrf_weights,
        },
        "latency_query_count": len(query_sample),
        "ranked_runs": {
            label: str(artifacts_dir / "runs" / filename)
            for label, filename in run_specs
        },
    }
    (results_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    paired_differences: Dict[str, object] = {}
    baseline_metrics = per_query_by_system.get("BM25", {})
    for system, query_metrics in per_query_by_system.items():
        if system == "BM25":
            continue
        metric_intervals = {}
        for metric in ("MRR@10", "NDCG@10", "Recall@10", "MAP@10", "Recall@100"):
            try:
                metric_intervals[metric] = paired_bootstrap_interval(
                    baseline_metrics,
                    query_metrics,
                    metric,
                    seed=args.seed,
                )
            except ValueError:
                continue
        paired_differences[system] = {
            "baseline": "BM25",
            "comparison_minus_baseline": metric_intervals,
        }
    (results_dir / "paired_differences.json").write_text(
        json.dumps(paired_differences, indent=2) + "\n",
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
