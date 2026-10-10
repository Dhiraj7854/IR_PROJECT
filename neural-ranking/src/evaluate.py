"""TREC-format loading and per-query/aggregate information retrieval metrics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Mapping, Union

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
        relevance = int(getattr(row, relevance_column)) if relevance_column else 1
        qrels.setdefault(qid, {})[pid] = relevance
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

    if not run:
        raise ValueError(f"TREC run is empty: {path}")
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


def evaluate_run(
    qrels: Union[Qrels, str, Path, pd.DataFrame],
    run: Union[Run, str, Path],
) -> Dict[str, Dict[str, float]]:
    """Compute MRR@10, NDCG@10, P@5, and Recall@100 per query.

    Returns a mapping from query ID to metric names. All qrels query IDs are
    retained, with zero-valued metrics for queries omitted from the run.
    """
    from ranx import Qrels as RanxQrels
    from ranx import Run as RanxRun
    from ranx import evaluate as ranx_evaluate

    parsed_qrels = load_qrels(qrels) if not isinstance(qrels, dict) else qrels
    parsed_run = load_run(run) if isinstance(run, (str, Path)) else run
    if not parsed_run:
        raise ValueError("Run is empty")
    rank_ordered_run = {
        qid: {
            pid: float(len(documents) - rank)
            for rank, pid in enumerate(documents)
        }
        for qid, documents in parsed_run.items()
    }
    ranx_qrels = RanxQrels(parsed_qrels)
    ranx_run = RanxRun(rank_ordered_run)
    metric_results = ranx_evaluate(
        ranx_qrels,
        ranx_run,
        ["mrr@10", "ndcg@10", "precision@5", "recall@100"],
        return_mean=False,
        save_results_in_run=False,
        make_comparable=True,
    )
    query_ids = ranx_qrels.get_query_ids()

    per_query: Dict[str, Dict[str, float]] = {}
    for index, qid in enumerate(query_ids):
        per_query[qid] = {
            "MRR@10": float(metric_results["mrr@10"][index]),
            "NDCG@10": float(metric_results["ndcg@10"][index]),
            "P@5": float(metric_results["precision@5"][index]),
            "Recall@100": float(metric_results["recall@100"][index]),
        }
    return per_query


def average_metrics(
    per_query: Mapping[str, Mapping[str, float]],
) -> Dict[str, float]:
    """Macro-average each metric across qrels query IDs."""
    if not per_query:
        raise ValueError("Cannot average an empty per-query metric mapping")
    metric_names = ("MRR@10", "NDCG@10", "P@5", "Recall@100")
    return {
        name: sum(metrics.get(name, 0.0) for metrics in per_query.values())
        / len(per_query)
        for name in metric_names
    }
