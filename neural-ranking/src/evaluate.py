"""TREC-format loading and per-query/aggregate information retrieval metrics."""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import Dict, Mapping, Sequence, Union

import pandas as pd

logger = logging.getLogger(__name__)

Qrels = Dict[str, Dict[str, int]]
Run = Dict[str, Dict[str, float]]


def load_qrels(source: Union[str, Path, pd.DataFrame]) -> Qrels:
    """Load qrels from a parquet DataFrame/file or four-column TREC text file."""
    if isinstance(source, pd.DataFrame):
        frame = source
    else:
        path = Path(source)
        if path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path)
        else:
            records = []
            with path.open(encoding="utf-8") as qrels_file:
                for line_number, line in enumerate(qrels_file, start=1):
                    fields = line.split()
                    if not fields:
                        continue
                    if len(fields) != 4:
                        raise ValueError(
                            f"Invalid qrels line {line_number} in {path}: {line.rstrip()!r}"
                        )
                    qid, _, pid, relevance = fields
                    records.append(
                        {
                            "query_id": qid,
                            "relevant_pid": pid,
                            "relevance": relevance,
                        }
                    )
            frame = pd.DataFrame(records)

    if {"query_id", "relevant_pid"}.issubset(frame.columns):
        qid_column, pid_column = "query_id", "relevant_pid"
        relevance_column = "relevance" if "relevance" in frame.columns else None
    elif {"qid", "docno", "relevance"}.issubset(frame.columns):
        qid_column, pid_column, relevance_column = "qid", "docno", "relevance"
    else:
        raise ValueError(
            "Qrels must contain query_id/relevant_pid or qid/docno/relevance columns"
        )

    qrels: Qrels = {}
    for row in frame.itertuples(index=False):
        qid = str(getattr(row, qid_column))
        pid = str(getattr(row, pid_column))
        judgments = qrels.setdefault(qid, {})
        if relevance_column:
            raw_relevance = getattr(row, relevance_column)
            if pd.isna(raw_relevance):
                logger.warning("Skipping qrels row with missing relevance for qid=%s pid=%s", qid, pid)
                continue
            relevance = int(raw_relevance)
        else:
            relevance = 1
        judgments[pid] = relevance
    if not qrels:
        raise ValueError("Qrels are empty")
    return qrels


def load_run(source: Union[str, Path]) -> Run:
    """Read and validate a six-column TREC run, preserving TREC rank ordering."""
    path = Path(source)
    run: Run = {}
    ranks: Dict[str, Dict[str, int]] = {}
    seen_ranks: Dict[str, set[int]] = {}
    with path.open(encoding="utf-8") as run_file:
        for line_number, line in enumerate(run_file, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 6 or fields[1] != "Q0":
                raise ValueError(
                    f"Invalid TREC run line {line_number} in {path}: {line.rstrip()!r}"
                )
            qid, _, pid, raw_rank, raw_score, _ = fields
            try:
                rank = int(raw_rank)
                score = float(raw_score)
            except ValueError as error:
                raise ValueError(
                    f"Invalid rank/score at {path}:{line_number}"
                ) from error
            if rank < 1:
                raise ValueError(f"Rank must be positive at {path}:{line_number}")
            if pid in run.setdefault(qid, {}):
                raise ValueError(f"Duplicate document {pid!r} for qid {qid!r} in {path}")
            if rank in seen_ranks.setdefault(qid, set()):
                raise ValueError(f"Duplicate rank {rank} for qid {qid!r} in {path}")
            run[qid][pid] = score
            ranks.setdefault(qid, {})[pid] = rank
            seen_ranks[qid].add(rank)

    return {
        qid: {
            pid: score
            for pid, score in sorted(
                documents.items(),
                key=lambda item: ranks[qid][item[0]],
            )
        }
        for qid, documents in run.items()
    }


def _evaluate_single_query(
    qid: str,
    qrels: Mapping[str, Mapping[str, int]],
    run: Mapping[str, Mapping[str, float]],
    *,
    k10: int = 10,
    k100: int = 100,
) -> Dict[str, float]:
    """Compute common IR metrics for one query using binary or graded qrels."""
    qid_key = str(qid)
    rel_map = {
        str(pid): int(score)
        for pid, score in qrels.get(qid_key, {}).items()
        if score is not None
    }
    if not rel_map:
        return {
            "MRR@10": 0.0,
            "NDCG@10": 0.0,
            "P@5": 0.0,
            "Recall@10": 0.0,
            "Recall@100": 0.0,
            "MAP@10": 0.0,
        }

    retrieved = list(run.get(qid_key, {}).keys())
    relevant_total = sum(1 for rel in rel_map.values() if rel > 0)
    retrieved_top10 = retrieved[:k10]
    retrieved_top100 = retrieved[:k100]
    retrieved_top5 = retrieved[:5]

    reciprocal_rank = 0.0
    for rank, pid in enumerate(retrieved_top10, start=1):
        if int(rel_map.get(str(pid), 0)) > 0:
            reciprocal_rank = 1.0 / rank
            break

    dcg = 0.0
    for rank, pid in enumerate(retrieved_top10, start=1):
        relevance = int(rel_map.get(str(pid), 0))
        if relevance > 0:
            dcg += (2**relevance - 1) / math.log2(rank + 1)
    ideal_relevances = sorted(
        (relevance for relevance in rel_map.values() if relevance > 0),
        reverse=True,
    )[:k10]
    if ideal_relevances:
        ideal_dcg = 0.0
        for rank, relevance in enumerate(ideal_relevances, start=1):
            ideal_dcg += (2**relevance - 1) / math.log2(rank + 1)
        ndcg = dcg / ideal_dcg if ideal_dcg > 0.0 else 0.0
    else:
        ndcg = 0.0

    hits_5 = sum(1 for pid in retrieved_top5 if int(rel_map.get(str(pid), 0)) > 0)
    p_at_5 = hits_5 / 5.0

    hits_10 = sum(1 for pid in retrieved_top10 if int(rel_map.get(str(pid), 0)) > 0)
    recall_10 = hits_10 / relevant_total if relevant_total else 0.0

    hits_100 = sum(1 for pid in retrieved_top100 if int(rel_map.get(str(pid), 0)) > 0)
    recall_100 = hits_100 / relevant_total if relevant_total else 0.0

    relevant_seen = 0
    ap_sum = 0.0
    for rank, pid in enumerate(retrieved_top10, start=1):
        if int(rel_map.get(str(pid), 0)) > 0:
            relevant_seen += 1
            ap_sum += relevant_seen / rank
    map_at_10 = ap_sum / relevant_total if relevant_total else 0.0

    return {
        "MRR@10": reciprocal_rank,
        "NDCG@10": ndcg,
        "P@5": p_at_5,
        "Recall@10": recall_10,
        "Recall@100": recall_100,
        "MAP@10": map_at_10,
    }


def evaluate_run(
    qrels: Union[Qrels, str, Path, pd.DataFrame],
    run: Union[Run, str, Path],
) -> Dict[str, Dict[str, float]]:
    """Compute common IR metrics per query.

    Returns a mapping from query ID to metric names. All qrels query IDs are
    retained, with zero-valued metrics for queries omitted from the run.
    """
    parsed_qrels = load_qrels(qrels) if not isinstance(qrels, dict) else {
        str(qid): {str(pid): int(score) for pid, score in doc_map.items()}
        for qid, doc_map in qrels.items()
    }
    parsed_run = load_run(run) if isinstance(run, (str, Path)) else {
        str(qid): {str(pid): float(score) for pid, score in doc_map.items()}
        for qid, doc_map in run.items()
    }
    query_ids = sorted({str(qid) for qid in parsed_qrels})
    per_query: Dict[str, Dict[str, float]] = {}
    for qid in query_ids:
        per_query[qid] = _evaluate_single_query(qid, parsed_qrels, parsed_run)
    return per_query


def average_metrics(
    per_query: Mapping[str, Mapping[str, float]],
) -> Dict[str, float]:
    """Macro-average each available metric across qrels query IDs."""
    if not per_query:
        raise ValueError("Cannot average an empty per-query metric mapping")

    metric_names = [
        "MRR@10",
        "NDCG@10",
        "P@5",
        "Recall@10",
        "Recall@100",
        "MAP@10",
    ]
    observed = set()
    for metrics in per_query.values():
        observed.update(metrics)
    available = [name for name in metric_names if name in observed]
    if not available:
        return {name: 0.0 for name in metric_names}
    averages: Dict[str, float] = {}
    for name in available:
        values = [
            metrics[name]
            for metrics in per_query.values()
            if name in metrics and metrics[name] is not None
        ]
        if values:
            averages[name] = sum(values) / len(values)
    return averages


def paired_bootstrap_interval(
    baseline: Mapping[str, Mapping[str, float]],
    comparison: Mapping[str, Mapping[str, float]],
    metric: str,
    *,
    seed: int = 42,
    resamples: int = 2000,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """Paired percentile bootstrap CI for comparison-minus-baseline query metrics."""
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    query_ids = sorted(
        qid
        for qid in set(baseline).intersection(comparison)
        if metric in baseline[qid] and metric in comparison[qid]
    )
    if not query_ids:
        raise ValueError(f"No paired query values found for metric {metric!r}")
    differences = [
        float(comparison[qid][metric]) - float(baseline[qid][metric])
        for qid in query_ids
    ]
    mean_difference = sum(differences) / len(differences)
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sample = [differences[rng.randrange(len(differences))] for _ in differences]
        estimates.append(sum(sample) / len(sample))
    estimates.sort()
    tail = (1.0 - confidence) / 2.0
    lower = estimates[int(tail * (resamples - 1))]
    upper = estimates[int((1.0 - tail) * (resamples - 1))]
    return {
        "mean_difference": mean_difference,
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence": confidence,
        "paired_queries": float(len(query_ids)),
        "resamples": float(resamples),
    }


def paired_bootstrap_comparisons(
    per_query_by_system: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    baselines: Sequence[str] = ("BM25", "TF-IDF"),
    metrics: Sequence[str] = (
        "MRR@10",
        "NDCG@10",
        "Recall@10",
        "MAP@10",
        "Recall@100",
    ),
    seed: int = 42,
) -> Dict[str, object]:
    """Build paired intervals against the primary and additional baselines."""
    if not baselines:
        raise ValueError("At least one baseline system is required")
    primary_baseline = baselines[0]
    comparisons: Dict[str, object] = {}
    for system, query_metrics in per_query_by_system.items():
        if system == primary_baseline:
            continue
        result: Dict[str, object] = {"baseline": primary_baseline}
        primary_metrics: Dict[str, Dict[str, float]] = {}
        if primary_baseline in per_query_by_system:
            for metric in metrics:
                try:
                    primary_metrics[metric] = paired_bootstrap_interval(
                        per_query_by_system[primary_baseline],
                        query_metrics,
                        metric,
                        seed=seed,
                    )
                except ValueError:
                    continue
        result["comparison_minus_baseline"] = primary_metrics

        additional: Dict[str, object] = {}
        for baseline in baselines[1:]:
            if baseline == system or baseline not in per_query_by_system:
                continue
            metric_intervals: Dict[str, Dict[str, float]] = {}
            for metric in metrics:
                try:
                    metric_intervals[metric] = paired_bootstrap_interval(
                        per_query_by_system[baseline],
                        query_metrics,
                        metric,
                        seed=seed,
                    )
                except ValueError:
                    continue
            additional[baseline] = {
                "comparison_minus_baseline": metric_intervals,
            }
        if additional:
            result["additional_comparisons"] = additional
        comparisons[system] = result
    return comparisons
