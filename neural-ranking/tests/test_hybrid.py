"""Unit tests for reciprocal rank fusion (RRF) hybrid retrieval."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid import fuse_ranked_lists, rerank_with_tail, rrf_score


def test_rrf_score_matches_reference():
    assert rrf_score(1, k=60) == pytest.approx(1.0 / 61.0)
    assert rrf_score(2, k=60) == pytest.approx(1.0 / 62.0)


def test_fuse_ranked_lists_deduplicates_and_sorts_by_score():
    bm25 = [(10, 0.90), (20, 0.80), (30, 0.70)]
    dense = [(20, 0.95), (10, 0.85), (40, 0.60)]

    fused = fuse_ranked_lists([bm25, dense], k=60)

    assert [entry["pid"] for entry in fused[:3]] == [10, 20, 30]
    assert fused[0]["score"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[1]["score"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[0]["score"] > fused[2]["score"]
    assert fused[0]["ranks"]["0"] == 1
    assert fused[0]["ranks"]["1"] == 2


def test_fuse_ranked_lists_handles_empty_and_partial_overlap():
    empty = []
    ranking = [(11, 1.0), (22, 0.9)]
    fused = fuse_ranked_lists([empty, ranking], k=60)

    assert fused == [
        {
            "pid": 11,
            "score": pytest.approx(1.0 / 61.0),
            "ranks": {"1": 1},
            "sources": ["1"],
        },
        {
            "pid": 22,
            "score": pytest.approx(1.0 / 62.0),
            "ranks": {"1": 2},
            "sources": ["1"],
        },
    ]


def test_fuse_ranked_lists_ignores_duplicate_ids_within_a_source():
    fused = fuse_ranked_lists([[(7, 1.0), (7, 0.5), (8, 0.4)]])

    assert [entry["pid"] for entry in fused] == [7, 8]
    assert fused[0]["score"] == pytest.approx(1 / 61)
    assert fused[0]["ranks"] == {"0": 1}


def test_rerank_prefix_preserves_candidate_depth_and_tail_order():
    full_pool = [(1, 0.5), (2, 0.4), (3, 0.3), (4, 0.2)]
    reranked = [(2, 0.9), (1, 0.8)]

    final = rerank_with_tail(full_pool, reranked, output_depth=4)

    assert [pid for pid, _ in final] == [2, 1, 3, 4]


def test_rerank_prefix_rejects_non_candidate_ids():
    with pytest.raises(ValueError, match="only candidate"):
        rerank_with_tail([(1, 0.5)], [(2, 0.9)], output_depth=1)
