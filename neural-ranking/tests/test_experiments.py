"""CPU-only tests for ablation data preparation and reporting helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import (
    QUERY_LENGTH_BUCKETS,
    bucket_queries_by_length,
    mrr_for_queries,
    query_length_range,
    rankings_to_run,
    sample_random_negative_texts,
)


def test_query_buckets_are_balanced_and_sorted_by_length():
    queries = pd.DataFrame(
        {
            "query_id": [4, 2, 7, 1, 3, 6, 5],
            "query_text": [
                "a b c d",
                "a",
                "a b c",
                "a",
                "a b",
                "a b c d e",
                "a b c d f",
            ],
        }
    )

    buckets = bucket_queries_by_length(queries)

    assert tuple(buckets) == QUERY_LENGTH_BUCKETS
    assert [len(buckets[bucket]) for bucket in QUERY_LENGTH_BUCKETS] == [3, 2, 2]
    assert query_length_range(buckets["short"]) == (1, 2)
    assert query_length_range(buckets["long"]) == (5, 5)


def test_rankings_to_run_preserves_rank_order_and_rejects_duplicates():
    run = rankings_to_run([1], [[(20, 0.2), (10, 0.9)]])

    assert list(run["1"]) == ["20", "10"]
    with pytest.raises(ValueError, match="Duplicate document"):
        rankings_to_run([1], [[(20, 0.2), (20, 0.1)]])


def test_mrr_for_queries_matches_hand_calculation():
    qrels = {
        "1": {"20": 1},
        "2": {"40": 1},
        "3": {"60": 1},
    }
    run = {
        "1": {"10": 0.9, "20": 0.8},
        "2": {"40": 0.9},
        "3": {"50": 0.9},
    }

    assert mrr_for_queries(qrels, run, [1, 2]) == pytest.approx(0.75)


def test_random_negatives_are_deterministic_and_exclude_positive_pids():
    corpus = pd.DataFrame(
        {
            "pid": [1, 2, 3, 4],
            "passage_text": ["positive one", "random two", "positive three", "random four"],
        }
    )

    first = sample_random_negative_texts(corpus, [1, 3], seed=17)
    second = sample_random_negative_texts(corpus, [1, 3], seed=17)

    assert first == second
    assert first[0] != "positive one"
    assert first[1] != "positive three"
