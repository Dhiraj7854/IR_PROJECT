#!/usr/bin/env python3
"""Evaluate one TREC run using the project's standard retrieval metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import average_metrics, evaluate_run, load_qrels, load_run

_METRIC_ALIASES = {
    "mrr@10": "MRR@10",
    "recip_rank": "MRR@10",
    "ndcg@10": "NDCG@10",
    "ndcg_cut_10": "NDCG@10",
    "p@5": "P@5",
    "precision_5": "P@5",
    "recall@10": "Recall@10",
    "recall_10": "Recall@10",
    "recall@100": "Recall@100",
    "recall_100": "Recall@100",
    "map@10": "MAP@10",
    "map_cut_10": "MAP@10",
}


def parse_args() -> argparse.Namespace:
    """Parse TREC run, qrels, metric selection, and output paths."""
    parser = argparse.ArgumentParser(
        description="Compute per-query and macro retrieval metrics for a TREC run."
    )
    parser.add_argument(
        "--run-file",
        "--run_file",
        required=True,
        help="Path to a six-column TREC run file.",
    )
    parser.add_argument(
        "--qrels-file",
        "--qrels_file",
        required=True,
        help="Path to parquet or four-column TREC qrels.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=sorted(_METRIC_ALIASES),
        default=[
            "mrr@10",
            "ndcg@10",
            "p@5",
            "recall@10",
            "recall@100",
            "map@10",
        ],
        help="Metrics to include (accepted aliases map to the project's fixed cutoffs).",
    )
    parser.add_argument(
        "--output-file",
        "--output_file",
        default="artifacts/eval_results.json",
        help="JSON output path.",
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate the run and write per-query and average JSON results."""
    args = parse_args()
    parsed_run = load_run(args.run_file)
    metric_names = list(dict.fromkeys(_METRIC_ALIASES[name] for name in args.metrics))
    if "Recall@100" in metric_names:
        truncated = [
            (qid, len(documents))
            for qid, documents in parsed_run.items()
            if documents and len(documents) < 100
        ]
        if truncated:
            raise ValueError(
                "Cannot report Recall@100 from a run truncated below rank 100; "
                f"examples (query ID, run depth): {truncated[:5]}. Regenerate the run "
                "with at least 100 ranked results per query or omit Recall@100."
            )
    per_query = evaluate_run(load_qrels(args.qrels_file), parsed_run)
    averages = average_metrics(per_query)
    result: Dict[str, object] = {
        "metrics": metric_names,
        "per_query": {
            qid: {metric: scores[metric] for metric in metric_names}
            for qid, scores in per_query.items()
        },
        "average": {metric: averages[metric] for metric in metric_names},
    }
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["average"], indent=2))
    print(f"Wrote per-query and average metrics to {output_path}")


if __name__ == "__main__":
    main()
