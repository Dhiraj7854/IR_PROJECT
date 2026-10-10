"""Tests for TREC metrics and manual MRR@10 verification."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import average_metrics, evaluate_run, load_run


def test_mrr_at_10_matches_hand_calculation(tmp_path):
    qrels = pd.DataFrame(
        {
            "query_id": [101, 101, 202],
            "relevant_pid": [12, 13, 20],
        }
    )
    run_path = tmp_path / "sample.run"
    run_path.write_text(
        "101 Q0 11 1 3.0 toy\n"
        "101 Q0 12 2 2.0 toy\n"
        "101 Q0 14 3 1.0 toy\n",
        encoding="utf-8",
    )

    per_query = evaluate_run(qrels, run_path)

    # qid 101's first relevant document is at rank 2: reciprocal rank = 1/2.
    # qid 202 has no relevant result: reciprocal rank = 0.
    assert per_query["101"]["MRR@10"] == pytest.approx(0.5)
    assert per_query["202"]["MRR@10"] == pytest.approx(0.0)
    assert average_metrics(per_query)["MRR@10"] == pytest.approx(0.25)
    assert per_query["101"]["P@5"] == pytest.approx(0.2)
    assert per_query["101"]["Recall@100"] == pytest.approx(0.5)


def test_trec_run_is_sorted_using_rank_column_not_file_order(tmp_path):
    run_path = tmp_path / "ranked.run"
    run_path.write_text(
        "1 Q0 20 2 9.0 sys\n"
        "1 Q0 10 1 1.0 sys\n",
        encoding="utf-8",
    )

    run = load_run(run_path)

    assert list(run["1"]) == ["10", "20"]
