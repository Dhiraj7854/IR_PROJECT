"""Tests for TREC metrics and manual MRR@10 verification."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import math

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import (
    average_metrics,
    evaluate_run,
    load_run,
    paired_bootstrap_interval,
)


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


def test_metrics_match_hand_calculated_reference_at_cutoffs():
    qrels = {
        "q": {"a": 1, "b": 1, "c": 0},
    }
    run = {
        "q": {
            "x": 4.0,
            "a": 3.0,
            "b": 2.0,
            **{f"n{index}": 1.0 for index in range(4, 101)},
        },
    }

    metrics = evaluate_run(qrels, run)["q"]

    # Relevant docs are at ranks 2 and 3; ideal DCG@10 is 1 + 1/log2(3).
    expected_dcg = 1 / math.log2(3) + 1 / math.log2(4)
    expected_idcg = 1 + 1 / math.log2(3)
    assert metrics["MRR@10"] == pytest.approx(0.5)
    assert metrics["NDCG@10"] == pytest.approx(expected_dcg / expected_idcg)
    assert metrics["Recall@10"] == pytest.approx(1.0)
    assert metrics["Recall@100"] == pytest.approx(1.0)
    assert metrics["MAP@10"] == pytest.approx((1 / 2 + 2 / 3) / 2)


def test_empty_run_and_missing_relevance_labels_are_zero(tmp_path):
    qrels = pd.DataFrame(
        {
            "query_id": ["empty", "unlabeled"],
            "relevant_pid": ["1", "2"],
            "relevance": [None, None],
        }
    )
    empty_run = tmp_path / "empty.run"
    empty_run.write_text("", encoding="utf-8")

    per_query = evaluate_run(qrels, empty_run)

    assert per_query["empty"]["MRR@10"] == 0.0
    assert per_query["empty"]["Recall@100"] == 0.0
    assert per_query["unlabeled"]["MAP@10"] == 0.0


def test_trec_run_with_zero_results_for_a_query_is_scored_as_zero():
    per_query = evaluate_run(
        {"1": {"10": 1}, "2": {"20": 1}},
        {"1": {"11": 1.0}},
    )

    assert per_query["2"]["MRR@10"] == 0.0


def test_paired_bootstrap_is_deterministic_and_reports_comparison_minus_baseline():
    baseline = {"1": {"MRR@10": 0.2}, "2": {"MRR@10": 0.0}}
    comparison = {"1": {"MRR@10": 0.5}, "2": {"MRR@10": 0.2}}

    interval = paired_bootstrap_interval(
        baseline, comparison, "MRR@10", seed=7, resamples=500
    )

    assert interval["mean_difference"] == pytest.approx(0.25)
    assert interval["ci_lower"] <= interval["mean_difference"]
    assert interval["ci_upper"] >= interval["mean_difference"]
    assert interval == paired_bootstrap_interval(
        baseline, comparison, "MRR@10", seed=7, resamples=500
    )
