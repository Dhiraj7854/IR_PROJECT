"""Helpers for validation-split retrieval ablation experiments."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from src.evaluate import average_metrics, evaluate_run


QUERY_LENGTH_BUCKETS = ("short", "medium", "long")


def sample_random_negative_texts(
    corpus_df: pd.DataFrame,
    positive_pids: Sequence[int],
    seed: int = 42,
) -> list[str]:
    """Sample one non-positive, non-empty corpus passage per positive pid."""
    required = {"pid", "passage_text"}
    missing = required.difference(corpus_df.columns)
    if missing:
        raise ValueError(f"Corpus is missing required columns: {sorted(missing)}")
    if corpus_df[list(required)].isna().any().any():
        raise ValueError("Corpus pid and passage_text values must not be null")
    if corpus_df["pid"].duplicated().any():
        raise ValueError("Corpus passage IDs must be unique")

    usable_corpus = corpus_df[
        corpus_df["passage_text"].astype(str).str.strip().ne("")
    ]
    if len(usable_corpus) < 2:
        raise ValueError(
            "Random-negative corpus must contain at least two non-empty passages"
        )

    corpus_pids = usable_corpus["pid"].to_numpy()
    corpus_texts = usable_corpus["passage_text"].astype(str).to_numpy()
    pid_to_index = {
        int(pid): index for index, pid in enumerate(corpus_pids.tolist())
    }
    rng = np.random.default_rng(seed)
    negative_texts: list[str] = []
    for positive_pid in positive_pids:
        positive_index = pid_to_index.get(int(positive_pid))
        if positive_index is None:
            sampled_index = int(rng.integers(len(corpus_pids)))
        else:
            sampled_index = int(rng.integers(len(corpus_pids) - 1))
            if sampled_index >= positive_index:
                sampled_index += 1
        negative_texts.append(str(corpus_texts[sampled_index]).strip())
    return negative_texts


def bucket_queries_by_length(
    queries_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Split queries into balanced length terciles using whitespace word counts.

    Queries are sorted by word count and query ID before splitting, making
    bucket membership deterministic. Ties at a boundary may fall into either
    adjacent bucket to keep the groups as balanced as possible.
    """
    required = {"query_id", "query_text"}
    missing = required.difference(queries_df.columns)
    if missing:
        raise ValueError(f"Queries are missing required columns: {sorted(missing)}")
    if queries_df["query_id"].duplicated().any():
        raise ValueError("Query IDs must be unique for length bucketing")
    if len(queries_df) < len(QUERY_LENGTH_BUCKETS):
        raise ValueError("At least three queries are required to form length buckets")

    frame = queries_df.loc[:, ["query_id", "query_text"]].copy()
    frame["_word_count"] = frame["query_text"].fillna("").astype(str).map(
        lambda text: len(text.split())
    )
    frame = frame.sort_values(
        ["_word_count", "query_id"],
        kind="stable",
    ).reset_index(drop=True)
    chunks = np.array_split(
        np.arange(len(frame)),
        len(QUERY_LENGTH_BUCKETS),
    )
    return {
        bucket: frame.iloc[chunk]
        .drop(columns="_word_count")
        .reset_index(drop=True)
        for bucket, chunk in zip(QUERY_LENGTH_BUCKETS, chunks)
    }


def rankings_to_run(
    query_ids: Sequence[int],
    rankings: Sequence[Sequence[Tuple[int, float]]],
) -> Dict[str, Dict[str, float]]:
    """Convert aligned query IDs and ranked (pid, score) rows to a run mapping."""
    if len(query_ids) != len(rankings):
        raise ValueError("query_ids and rankings must have equal lengths")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("query_ids must be unique")

    run: Dict[str, Dict[str, float]] = {}
    for query_id, hits in zip(query_ids, rankings):
        qid = str(query_id)
        documents: Dict[str, float] = {}
        for pid, score in hits:
            document_id = str(pid)
            if document_id in documents:
                raise ValueError(f"Duplicate document {document_id!r} for qid {qid!r}")
            documents[document_id] = float(score)
        run[qid] = documents
    return run


def mrr_for_queries(
    qrels: Mapping[str, Mapping[str, int]],
    run: Mapping[str, Mapping[str, float]],
    query_ids: Sequence[int],
) -> float:
    """Return macro MRR@10 for the requested query IDs."""
    requested_ids = {str(query_id) for query_id in query_ids}
    subset_qrels = {
        qid: dict(relevant)
        for qid, relevant in qrels.items()
        if qid in requested_ids
    }
    if not subset_qrels:
        raise ValueError("No qrels are available for the requested query group")
    metrics = evaluate_run(subset_qrels, dict(run))
    return average_metrics(metrics)["MRR@10"]


def query_length_range(queries_df: pd.DataFrame) -> Tuple[int, int]:
    """Return min/max whitespace-delimited word counts for a query group."""
    counts = queries_df["query_text"].fillna("").astype(str).map(
        lambda text: len(text.split())
    )
    if counts.empty:
        raise ValueError("Cannot calculate a word-count range for an empty group")
    return int(counts.min()), int(counts.max())
