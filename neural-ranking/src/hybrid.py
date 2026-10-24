"""Hybrid retrieval utilities using Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

RankedList = Sequence[Tuple[int, float]]


def rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal rank fusion contribution for a single list position."""
    if k <= 0:
        raise ValueError("k must be positive")
    if rank <= 0:
        raise ValueError("Rank must be positive")
    return 1.0 / (k + rank)


def _normalise_weights(
    count: int,
    weights: Optional[Sequence[float]] = None,
) -> List[float]:
    """Return per-list weights; default weights preserve the standard RRF sum."""
    if weights is None:
        return [1.0 for _ in range(count)]
    if len(weights) != count:
        raise ValueError("weights must match the number of ranked lists")
    if not weights:
        return []
    if any(not math.isfinite(float(weight)) or weight < 0.0 for weight in weights):
        raise ValueError("weights must be finite and non-negative")
    if not any(weight > 0.0 for weight in weights):
        raise ValueError("at least one weight must be positive")
    return [float(weight) for weight in weights]


def fuse_ranked_lists(
    ranked_lists: Sequence[RankedList],
    *,
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
    source_names: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Fuse multiple ranked lists with reciprocal rank fusion.

    The returned entries contain the fused passage ID, the combined RRF score,
    the rank observed in each source list, and the source list identifiers.
    """
    if not ranked_lists:
        return []
    if k <= 0:
        raise ValueError("k must be positive")

    if source_names is None:
        source_names = [str(index) for index in range(len(ranked_lists))]
    if len(source_names) != len(ranked_lists):
        raise ValueError("source_names must match the number of ranked lists")
    if len(set(source_names)) != len(source_names) or any(
        not name for name in source_names
    ):
        raise ValueError("source_names must be unique and non-empty")

    norm_weights = _normalise_weights(len(ranked_lists), weights)
    fused_scores: Dict[int, float] = {}
    rank_map: Dict[int, Dict[str, int]] = {}
    sources: Dict[int, set[str]] = {}

    for list_index, ranking in enumerate(ranked_lists):
        if ranking is None:
            continue
        weight = norm_weights[list_index]
        seen_in_source = set()
        for rank_value, (pid, _) in enumerate(ranking, start=1):
            passage_id = int(pid)
            if passage_id in seen_in_source:
                continue
            seen_in_source.add(passage_id)
            fused_scores.setdefault(passage_id, 0.0)
            fused_scores[passage_id] += weight * rrf_score(rank_value, k=k)
            rank_map.setdefault(passage_id, {})[source_names[list_index]] = rank_value
            sources.setdefault(passage_id, set()).add(source_names[list_index])

    ordered = sorted(
        fused_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )
    results: List[Dict[str, Any]] = []
    for passage_id, score in ordered:
        result = {
            "pid": passage_id,
            "score": float(score),
            "ranks": rank_map.get(passage_id, {}),
            "sources": sorted(sources.get(passage_id, set())),
        }
        results.append(result)
    return results


def retrieve_hybrid(
    bm25_results: Sequence[Tuple[int, float]],
    dense_results: Sequence[Tuple[int, float]],
    *,
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """Convenience wrapper for a BM25 + bi-encoder hybrid retrieval run."""
    return fuse_ranked_lists(
        [bm25_results, dense_results],
        k=k,
        weights=weights,
        source_names=["bm25", "bi-encoder"],
    )


def rerank_with_tail(
    candidate_ranking: RankedList,
    reranked_prefix: RankedList,
    *,
    output_depth: int,
) -> List[Tuple[int, float]]:
    """Keep a full candidate-depth run while reranking only its leading prefix."""
    if output_depth < 0:
        raise ValueError("output_depth must be non-negative")
    candidate_ids = [int(pid) for pid, _ in candidate_ranking]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate_ranking contains duplicate passage IDs")
    candidate_set = set(candidate_ids)
    reranked_ids = [int(pid) for pid, _ in reranked_prefix]
    if len(reranked_ids) != len(set(reranked_ids)):
        raise ValueError("reranked_prefix contains duplicate passage IDs")
    if not set(reranked_ids).issubset(candidate_set):
        raise ValueError("reranked_prefix must contain only candidate passage IDs")

    reranked_set = set(reranked_ids)
    tail = [
        (int(pid), float(score))
        for pid, score in candidate_ranking
        if int(pid) not in reranked_set
    ]
    return [
        *((int(pid), float(score)) for pid, score in reranked_prefix),
        *tail,
    ][:output_depth]
