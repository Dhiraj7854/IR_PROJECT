#!/usr/bin/env python3
"""Run retrieval ablations and write metrics/figures under artifacts/results."""

from __future__ import annotations

import argparse
import logging
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import build_index, get_index
from src.biencoder import (
    configure_retrieval,
    load_biencoder,
    retrieve_biencoder_batch,
)
from src.config import load_config
from src.crossencoder import load_crossencoder, rerank
from src.evaluate import average_metrics, evaluate_run, load_qrels
from src.experiments import (
    QUERY_LENGTH_BUCKETS,
    bucket_queries_by_length,
    mrr_for_queries,
    query_length_range,
    rankings_to_run,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

RERANK_DEPTHS = (50, 100, 200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hard-negative, rerank-depth, and query-length ablations."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/config.yaml"),
        help="Path to YAML configuration.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "data/processed"),
        help="Directory containing corpus, query, qrels, and triplet parquet files.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=str(PROJECT_ROOT / "artifacts"),
        help="Directory containing trained models and generated artifacts.",
    )
    parser.add_argument(
        "--biencoder-model",
        default="biencoder",
        help="Hard-negative bi-encoder checkpoint relative to --artifacts-dir or absolute.",
    )
    parser.add_argument(
        "--crossencoder-model",
        default="crossencoder",
        help="Cross-encoder checkpoint relative to --artifacts-dir or absolute.",
    )
    parser.add_argument(
        "--random-biencoder-model",
        default="biencoder_random",
        help="Random-negative bi-encoder checkpoint relative to --artifacts-dir or absolute.",
    )
    parser.add_argument("--biencoder-batch-size", type=int, default=64)
    parser.add_argument("--crossencoder-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reuse-random-checkpoint",
        action="store_true",
        help="Reuse artifacts/biencoder_random instead of retraining random negatives.",
    )
    return parser.parse_args()


def _require_frame(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required experiment input not found: {path}")
    frame = pd.read_parquet(path)
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"Required experiment input is empty: {path}")
    return frame


def _sync_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _train_random_biencoder(
    config_path: Path,
    data_dir: Path,
    output_dir: Path,
    seed: int,
) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train_biencoder.py"),
        "--config",
        str(config_path),
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--negatives",
        "random",
        "--epochs",
        "2",
        "--seed",
        str(seed),
    ]
    logger.info("Training random-negative bi-encoder: %s", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _biencoder_run(
    checkpoint: Path,
    corpus_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    batch_size: int,
) -> Dict[str, Dict[str, float]]:
    model = load_biencoder(checkpoint)
    configure_retrieval(
        corpus_df["passage_text"].astype(str).tolist(),
        pids=corpus_df["pid"].tolist(),
        batch_size=batch_size,
        model=model,
        cache_path=checkpoint / "corpus_embeddings.npy",
    )
    hits = retrieve_biencoder_batch(
        queries_df["query_text"].astype(str).tolist(),
        k=10,
        batch_size=batch_size,
    )
    result = rankings_to_run(queries_df["query_id"].astype(int).tolist(), hits)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _plot_negative_comparison(
    values: Mapping[str, float],
    output_path: Path,
) -> None:
    labels = list(values)
    scores = [values[label] for label in labels]
    figure, axis = plt.subplots(figsize=(6, 4))
    bars = axis.bar(labels, scores, color=["#4472C4", "#ED7D31"])
    axis.set_ylabel("Validation MRR@10")
    axis.set_xlabel("Negative sampling strategy")
    axis.set_title("Bi-encoder: hard vs random negatives")
    axis.set_ylim(0.0, max(0.05, max(scores, default=0.0) * 1.15))
    axis.bar_label(bars, fmt="%.4f", padding=3)
    axis.legend(handles=bars, labels=labels, title="Negative strategy")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _plot_depth_comparison(
    rows: Sequence[Mapping[str, float]],
    output_path: Path,
) -> None:
    depths = [int(row["depth"]) for row in rows]
    mrr = [float(row["mrr"]) for row in rows]
    latency = [float(row["median_latency_ms"]) for row in rows]
    figure, left_axis = plt.subplots(figsize=(7, 4.5))
    mrr_line = left_axis.plot(
        depths,
        mrr,
        marker="o",
        color="#4472C4",
        label="MRR@10",
    )
    left_axis.set_xlabel("BM25 candidate depth")
    left_axis.set_ylabel("Validation MRR@10", color="#4472C4")
    left_axis.tick_params(axis="y", labelcolor="#4472C4")
    left_axis.set_xticks(depths)
    right_axis = left_axis.twinx()
    latency_line = right_axis.plot(
        depths,
        latency,
        marker="s",
        color="#ED7D31",
        label="Median end-to-end latency",
    )
    right_axis.set_ylabel(
        "Median BM25 + re-ranking latency (ms/query)",
        color="#ED7D31",
    )
    right_axis.tick_params(axis="y", labelcolor="#ED7D31")
    left_axis.set_title("Cross-encoder re-ranking depth")
    left_axis.grid(True, axis="both", linestyle="--", alpha=0.35)
    lines = mrr_line + latency_line
    left_axis.legend(lines, [line.get_label() for line in lines], loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _plot_length_comparison(
    values: Mapping[str, Mapping[str, float]],
    output_path: Path,
) -> None:
    x = np.arange(len(QUERY_LENGTH_BUCKETS))
    width = 0.36
    bm25 = [values[bucket]["BM25"] for bucket in QUERY_LENGTH_BUCKETS]
    crossencoder = [
        values[bucket]["BM25 → Cross-encoder"]
        for bucket in QUERY_LENGTH_BUCKETS
    ]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    bm25_bars = axis.bar(
        x - width / 2,
        bm25,
        width,
        label="BM25",
        color="#4472C4",
    )
    crossencoder_bars = axis.bar(
        x + width / 2,
        crossencoder,
        width,
        label="BM25 → Cross-encoder",
        color="#ED7D31",
    )
    axis.set_xticks(x, QUERY_LENGTH_BUCKETS)
    axis.set_xlabel("Test query length bucket (word-count tercile)")
    axis.set_ylabel("Test MRR@10")
    axis.set_title("Retrieval quality by query length")
    axis.legend()
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.bar_label(bm25_bars, fmt="%.3f", padding=2)
    axis.bar_label(crossencoder_bars, fmt="%.3f", padding=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _write_report(
    output_path: Path,
    negative_mrr: Mapping[str, float],
    depth_rows: Sequence[Mapping[str, float]],
    length_rows: Sequence[Mapping[str, object]],
) -> None:
    hard_score = negative_mrr["Hard"]
    random_score = negative_mrr["Random"]
    lines = [
        "# Retrieval ablation results",
        "",
        "Experiments use the full corpus. Negative-sampling and re-ranking-depth",
        "ablations are measured on validation; query-length analysis uses test,",
        "as requested. MRR is macro-averaged over judged queries.",
        "",
        "## 1. Bi-encoder negative sampling (validation)",
        "",
        "| Negative strategy | MRR@10 | Difference vs hard |",
        "|---|---:|---:|",
        f"| Hard | {hard_score:.4f} | +0.0000 |",
        f"| Random | {random_score:.4f} | {random_score - hard_score:+.4f} |",
        "",
        "![Hard vs random negative sampling](ablations_negatives.png)",
        "",
        "## 2. Cross-encoder re-ranking depth (validation)",
        "",
        "| BM25 candidate depth | MRR@10 | Median BM25 + re-ranking latency (ms/query) |",
        "|---:|---:|---:|",
    ]
    lines.extend(
        f"| {int(row['depth'])} | {float(row['mrr']):.4f} | "
        f"{float(row['median_latency_ms']):.2f} |"
        for row in depth_rows
    )
    lines.extend(
        [
            "",
            "Latency is measured per query from BM25 candidate retrieval through",
            "cross-encoder scoring and includes CUDA synchronization when applicable.",
            "",
            "![Re-ranking depth](ablations_rerank_depth.png)",
            "",
            "## 3. Test query length (word-count terciles)",
            "",
            "| Query bucket | Queries | Word-count range | BM25 MRR@10 | Cross-encoder MRR@10 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        f"| {row['bucket']} | {int(row['query_count'])} | "
        f"{row['word_count_range']} | {float(row['bm25_mrr']):.4f} | "
        f"{float(row['crossencoder_mrr']):.4f} |"
        for row in length_rows
    )
    lines.extend(
        [
            "",
            "Length is whitespace-delimited word count. Queries are sorted by word",
            "count then query ID and split into balanced terciles; boundary ties may",
            "be divided across adjacent buckets.",
            "",
            "![Query length analysis](ablations_query_length.png)",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_experiments(args: argparse.Namespace) -> Path:
    if args.biencoder_batch_size <= 0 or args.crossencoder_batch_size <= 0:
        raise ValueError("Inference batch sizes must be greater than zero")

    config_path = Path(args.config).resolve()
    data_dir = Path(args.data_dir).resolve()
    artifacts_dir = Path(args.artifacts_dir).resolve()
    config = load_config(config_path)
    corpus_df = _require_frame(
        data_dir / "corpus.parquet",
        {"pid", "passage_text"},
    )
    val_queries = _require_frame(
        data_dir / "val_queries.parquet",
        {"query_id", "query_text"},
    )
    val_qrels_df = _require_frame(
        data_dir / "val_qrels.parquet",
        {"query_id", "relevant_pid"},
    )
    test_queries = _require_frame(
        data_dir / "test_queries.parquet",
        {"query_id", "query_text"},
    )
    test_qrels_df = _require_frame(
        data_dir / "test_qrels.parquet",
        {"query_id", "relevant_pid"},
    )
    if (
        val_queries["query_id"].duplicated().any()
        or test_queries["query_id"].duplicated().any()
    ):
        raise ValueError("Validation and test query IDs must be unique")

    hard_checkpoint = Path(args.biencoder_model)
    if not hard_checkpoint.is_absolute():
        hard_checkpoint = artifacts_dir / hard_checkpoint
    random_checkpoint = Path(args.random_biencoder_model)
    if not random_checkpoint.is_absolute():
        random_checkpoint = artifacts_dir / random_checkpoint
    crossencoder_checkpoint = Path(args.crossencoder_model)
    if not crossencoder_checkpoint.is_absolute():
        crossencoder_checkpoint = artifacts_dir / crossencoder_checkpoint
    if not hard_checkpoint.is_dir():
        raise FileNotFoundError(
            f"Train the hard-negative bi-encoder first: {hard_checkpoint}"
        )
    if not crossencoder_checkpoint.is_dir():
        raise FileNotFoundError(
            f"Train the cross-encoder first: {crossencoder_checkpoint}"
        )
    if args.reuse_random_checkpoint:
        if not random_checkpoint.is_dir():
            raise FileNotFoundError(
                f"Random-negative checkpoint not found: {random_checkpoint}"
            )
    else:
        _train_random_biencoder(
            config_path,
            data_dir,
            random_checkpoint,
            args.seed,
        )

    results_dir = artifacts_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Evaluating hard-negative bi-encoder on validation")
    hard_run = _biencoder_run(
        hard_checkpoint,
        corpus_df,
        val_queries,
        args.biencoder_batch_size,
    )
    logger.info("Evaluating random-negative bi-encoder on validation")
    random_run = _biencoder_run(
        random_checkpoint,
        corpus_df,
        val_queries,
        args.biencoder_batch_size,
    )
    val_qrels = load_qrels(val_qrels_df)
    negative_mrr = {
        "Hard": average_metrics(evaluate_run(val_qrels, hard_run))["MRR@10"],
        "Random": average_metrics(evaluate_run(val_qrels, random_run))["MRR@10"],
    }
    _plot_negative_comparison(
        negative_mrr,
        results_dir / "ablations_negatives.png",
    )

    logger.info("Building or loading BM25 index")
    build_index(corpus_df, artifacts_dir=artifacts_dir)
    index = get_index()
    passage_text_by_pid = dict(
        zip(corpus_df["pid"].astype(int), corpus_df["passage_text"].astype(str))
    )
    crossencoder_model, tokenizer = load_crossencoder(crossencoder_checkpoint)
    depth_rows: List[Dict[str, float]] = []
    for depth in RERANK_DEPTHS:
        results: Dict[str, Dict[str, float]] = {}
        latencies_ms: List[float] = []
        for query in val_queries.itertuples(index=False):
            query_text = str(query.query_text)
            _sync_if_cuda()
            started = time.perf_counter()
            candidates = index.retrieve_bm25(query_text, k=depth)
            candidate_pids = [pid for pid, _ in candidates]
            candidate_texts = [passage_text_by_pid[pid] for pid in candidate_pids]
            ranked = rerank(
                query_text,
                candidate_pids,
                candidate_texts,
                k=10,
                model=crossencoder_model,
                tokenizer=tokenizer,
                batch_size=args.crossencoder_batch_size,
                max_length=config.models.max_seq_len,
                fp16=True,
            )
            _sync_if_cuda()
            latencies_ms.append((time.perf_counter() - started) * 1000.0)
            results[str(int(query.query_id))] = {
                str(pid): float(score) for pid, score in ranked
            }
        metrics = average_metrics(evaluate_run(val_qrels, results))
        median_latency = statistics.median(latencies_ms)
        depth_rows.append(
            {
                "depth": float(depth),
                "mrr": metrics["MRR@10"],
                "median_latency_ms": median_latency,
            }
        )
        logger.info(
            "Validation rerank depth=%d MRR@10=%.4f median=%.2f ms/query",
            depth,
            metrics["MRR@10"],
            median_latency,
        )
    _plot_depth_comparison(
        depth_rows,
        results_dir / "ablations_rerank_depth.png",
    )

    logger.info("Analyzing test MRR by query-length tercile")
    query_buckets = bucket_queries_by_length(test_queries)
    test_qrels = load_qrels(test_qrels_df)
    bucket_query_ids = {
        bucket: frame["query_id"].astype(int).tolist()
        for bucket, frame in query_buckets.items()
    }
    bm25_rankings = [
        index.retrieve_bm25(str(text), k=100)
        for text in test_queries["query_text"].astype(str)
    ]
    bm25_run = rankings_to_run(
        test_queries["query_id"].astype(int).tolist(),
        bm25_rankings,
    )
    crossencoder_rankings: List[List[Tuple[int, float]]] = []
    for query, candidates in zip(test_queries.itertuples(index=False), bm25_rankings):
        candidate_pids = [pid for pid, _ in candidates]
        candidate_texts = [passage_text_by_pid[pid] for pid in candidate_pids]
        crossencoder_rankings.append(
            rerank(
                str(query.query_text),
                candidate_pids,
                candidate_texts,
                k=10,
                model=crossencoder_model,
                tokenizer=tokenizer,
                batch_size=args.crossencoder_batch_size,
                max_length=config.models.max_seq_len,
                fp16=True,
            )
        )
    crossencoder_run = rankings_to_run(
        test_queries["query_id"].astype(int).tolist(),
        crossencoder_rankings,
    )

    length_rows: List[Dict[str, object]] = []
    length_plot_values: Dict[str, Dict[str, float]] = {}
    for bucket in QUERY_LENGTH_BUCKETS:
        frame = query_buckets[bucket]
        query_ids = bucket_query_ids[bucket]
        word_min, word_max = query_length_range(frame)
        bm25_mrr = mrr_for_queries(test_qrels, bm25_run, query_ids)
        crossencoder_mrr = mrr_for_queries(test_qrels, crossencoder_run, query_ids)
        length_plot_values[bucket] = {
            "BM25": bm25_mrr,
            "BM25 → Cross-encoder": crossencoder_mrr,
        }
        length_rows.append(
            {
                "bucket": bucket,
                "query_count": len(frame),
                "word_count_range": f"{word_min}–{word_max}",
                "bm25_mrr": bm25_mrr,
                "crossencoder_mrr": crossencoder_mrr,
            }
        )
    _plot_length_comparison(
        length_plot_values,
        results_dir / "ablations_query_length.png",
    )

    report_path = results_dir / "ablations.md"
    _write_report(report_path, negative_mrr, depth_rows, length_rows)
    logger.info("Saved ablation report and figures under %s", results_dir)
    return report_path


def main() -> None:
    report_path = run_experiments(parse_args())
    print(f"Ablation results saved to {report_path}")


if __name__ == "__main__":
    main()
