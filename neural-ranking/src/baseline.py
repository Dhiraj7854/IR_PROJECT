"""TF-IDF and BM25 lexical retrieval baselines with indexing, caching, and TREC-format output."""

from __future__ import annotations

import hashlib
import logging
import math
import pickle
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_BM25_STOP_WORDS = frozenset(ENGLISH_STOP_WORDS)


def tokenize(text: str) -> List[str]:
    """Lowercase and regex-tokenize text, excluding common English stopwords."""
    return [
        token
        for token in _TOKEN_RE.findall(text.lower())
        if token not in _BM25_STOP_WORDS
    ] if text else []


# ---------------------------------------------------------------------------
# Index paths
# ---------------------------------------------------------------------------

_DEFAULT_ARTIFACTS = Path("artifacts")
_INDEX_CACHE_VERSION = 2
_INDEX_CACHE_NAME = "baseline_index.pkl"


def _corpus_fingerprint(corpus_df: pd.DataFrame) -> str:
    """Hash passage IDs and text so cached indexes cannot outlive their corpus."""
    corpus_hashes = pd.util.hash_pandas_object(
        corpus_df[["pid", "passage_text"]], index=False
    )
    return hashlib.sha256(corpus_hashes.to_numpy(dtype=np.uint64).tobytes()).hexdigest()


def _top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return up to k score-ranked row indexes without duplicate documents."""
    if k < 0:
        raise ValueError("k must be non-negative")
    count = min(k, len(scores))
    if count == 0:
        return np.empty(0, dtype=np.intp)

    candidates = np.argpartition(scores, len(scores) - count)[-count:]
    return candidates[np.lexsort((candidates, -scores[candidates]))]


def _compact_corpus_arrays(corpus_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    required_columns = {"pid", "passage_text"}
    missing_columns = required_columns.difference(corpus_df.columns)
    if missing_columns:
        raise ValueError(f"Corpus is missing required columns: {sorted(missing_columns)}")
    if corpus_df.empty:
        raise ValueError("Cannot build a retrieval index from an empty corpus")
    if corpus_df[["pid", "passage_text"]].isna().any().any():
        raise ValueError("Corpus pid and passage_text values must not be null")
    if corpus_df["pid"].duplicated().any():
        raise ValueError("Corpus passage IDs must be unique")
    return (
        corpus_df["pid"].to_numpy(dtype=np.int64),
        corpus_df["passage_text"].astype(str).tolist(),
    )


class CompactBM25Index:
    """Sparse BM25 index without the large Python-object tokenized corpus."""

    def __init__(
        self,
        corpus_df: pd.DataFrame,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        if (
            not math.isfinite(k1)
            or k1 <= 0
            or not 0.0 <= b <= 1.0
            or not math.isfinite(epsilon)
            or epsilon < 0
        ):
            raise ValueError("Invalid BM25 parameters")
        self._pids, texts = _compact_corpus_arrays(corpus_df)
        self._vectorizer = CountVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            lowercase=False,
            dtype=np.int32,
        )
        counts = sp.csr_matrix(
            self._vectorizer.fit_transform(texts),
            dtype=np.int32,
        )
        self._postings = sp.csc_matrix(counts)
        self._document_lengths = np.asarray(
            counts.sum(axis=1),
            dtype=np.float64,
        ).ravel()
        self._average_document_length = float(self._document_lengths.mean())
        if self._average_document_length <= 0:
            raise ValueError("Cannot build BM25 index: corpus contains no indexable terms")
        document_frequencies = np.diff(self._postings.indptr).astype(np.float64)
        document_count = len(self._pids)
        idf = np.log(document_count - document_frequencies + 0.5) - np.log(
            document_frequencies + 0.5
        )
        average_idf = float(idf.sum() / len(idf))
        idf[idf < 0] = epsilon * average_idf
        self._idf = idf
        self._k1 = k1
        self._b = b

    def retrieve(self, query: str, k: int = 100) -> List[Tuple[int, float]]:
        """Return BM25-ranked passage IDs and scores."""
        if k < 0:
            raise ValueError("k must be non-negative")
        scores = np.zeros(len(self._pids), dtype=np.float64)
        vocabulary = self._vectorizer.vocabulary_
        for term in tokenize(query):
            column = vocabulary.get(term)
            if column is None:
                continue
            start, end = self._postings.indptr[column : column + 2]
            rows = self._postings.indices[start:end]
            term_frequencies = self._postings.data[start:end].astype(np.float64)
            lengths = self._document_lengths[rows]
            denominator = term_frequencies + self._k1 * (
                1.0
                - self._b
                + self._b * lengths / self._average_document_length
            )
            scores[rows] += (
                self._idf[column]
                * term_frequencies
                * (self._k1 + 1.0)
                / denominator
            )
        top_indices = _top_k_indices(scores, k)
        return [
            (int(self._pids[index]), float(scores[index]))
            for index in top_indices
        ]


class CompactTfidfIndex:
    """TF-IDF-only index that avoids constructing an unused BM25 object."""

    def __init__(self, corpus_df: pd.DataFrame) -> None:
        self._pids, texts = _compact_corpus_arrays(corpus_df)
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            sublinear_tf=True,
            dtype=np.float32,
        )
        self._matrix = sp.csr_matrix(self._vectorizer.fit_transform(texts))

    def retrieve(self, query: str, k: int = 100) -> List[Tuple[int, float]]:
        """Return cosine-ranked passage IDs and scores."""
        if k < 0:
            raise ValueError("k must be non-negative")
        query_vector = self._vectorizer.transform([query])
        scores = (query_vector @ self._matrix.T).toarray().ravel()
        top_indices = _top_k_indices(scores, k)
        return [
            (int(self._pids[index]), float(scores[index]))
            for index in top_indices
        ]


# ---------------------------------------------------------------------------
# BaselineIndex — builds and owns both indexes
# ---------------------------------------------------------------------------

class BaselineIndex:
    """Holds TF-IDF and BM25 indexes over the 100k passage corpus.

    Build once with :meth:`build`, then use :meth:`retrieve_tfidf` /
    :meth:`retrieve_bm25`. Both indexes, the tokenized corpus, and the pid
    mapping are cached together in *artifacts_dir* so subsequent runs skip
    re-building.
    """

    def __init__(self) -> None:
        # TF-IDF state
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix: Optional[sp.csr_matrix] = None  # shape (N, V)

        # BM25 state
        self._bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: Optional[List[List[str]]] = None

        # Shared pid array (row i → pid)
        self._pids: Optional[np.ndarray] = None
        self._pid_to_row: Optional[Dict[int, int]] = None

        self._built = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        corpus_df: pd.DataFrame,
        artifacts_dir: Union[str, Path] = _DEFAULT_ARTIFACTS,
        force_rebuild: bool = False,
    ) -> Dict[str, float]:
        """Build (or load cached) TF-IDF and BM25 indexes.

        Args:
            corpus_df: DataFrame with columns [pid, passage_text].
            artifacts_dir: Directory where the combined index cache is stored.
            force_rebuild: Skip cache and always rebuild.

        Returns:
            Build/load timings, including 'total_indexing_s'.
        """
        total_started = time.perf_counter()
        required_columns = {"pid", "passage_text"}
        missing_columns = required_columns.difference(corpus_df.columns)
        if missing_columns:
            raise ValueError(f"Corpus is missing required columns: {sorted(missing_columns)}")
        if corpus_df.empty:
            raise ValueError("Cannot build retrieval indexes from an empty corpus")
        if corpus_df[["pid", "passage_text"]].isna().any().any():
            raise ValueError("Corpus pid and passage_text values must not be null")
        if corpus_df["pid"].duplicated().any():
            raise ValueError("Corpus passage IDs must be unique")

        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = _corpus_fingerprint(corpus_df)
        cache_path = artifacts_dir / _INDEX_CACHE_NAME

        if not force_rebuild and cache_path.exists():
            logger.info("Loading cached retrieval indexes from %s", cache_path)
            with cache_path.open("rb") as cache_file:
                state = pickle.load(cache_file)
            if (
                state.get("cache_version") == _INDEX_CACHE_VERSION
                and state.get("corpus_fingerprint") == fingerprint
            ):
                self._vectorizer = state["vectorizer"]
                self._tfidf_matrix = state["tfidf_matrix"]
                self._bm25 = state["bm25"]
                self._tokenized_corpus = state["tokenized_corpus"]
                self._pids = state["pids"]
                self._pid_to_row = {
                    int(pid): row for row, pid in enumerate(self._pids)
                }
                self._built = True
                elapsed = time.perf_counter() - total_started
                logger.info("Cached retrieval indexes loaded in %.2fs", elapsed)
                return {
                    "tfidf_build_s": 0.0,
                    "bm25_build_s": 0.0,
                    "total_indexing_s": elapsed,
                }
            logger.info("Index cache does not match the corpus; rebuilding it")

        self._pids = corpus_df["pid"].to_numpy(dtype=np.int64)
        self._pid_to_row = {
            int(pid): row for row, pid in enumerate(self._pids)
        }
        texts = corpus_df["passage_text"].astype(str).tolist()

        logger.info("Building TF-IDF index over %d passages", len(corpus_df))
        tfidf_started = time.perf_counter()
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            sublinear_tf=True,
            dtype=np.float32,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(texts)
        tfidf_build_s = time.perf_counter() - tfidf_started

        logger.info("Tokenizing corpus and building BM25 index over %d passages", len(corpus_df))
        bm25_started = time.perf_counter()
        self._tokenized_corpus = [tokenize(text) for text in texts]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        bm25_build_s = time.perf_counter() - bm25_started

        state = {
            "cache_version": _INDEX_CACHE_VERSION,
            "corpus_fingerprint": fingerprint,
            "vectorizer": self._vectorizer,
            "tfidf_matrix": self._tfidf_matrix,
            "bm25": self._bm25,
            "tokenized_corpus": self._tokenized_corpus,
            "pids": self._pids,
        }
        with cache_path.open("wb") as cache_file:
            pickle.dump(state, cache_file, protocol=pickle.HIGHEST_PROTOCOL)

        self._built = True
        total_indexing_s = time.perf_counter() - total_started
        logger.info(
            "Indexes built and cached in %.2fs (TF-IDF %.2fs, BM25 %.2fs)",
            total_indexing_s,
            tfidf_build_s,
            bm25_build_s,
        )
        return {
            "tfidf_build_s": tfidf_build_s,
            "bm25_build_s": bm25_build_s,
            "total_indexing_s": total_indexing_s,
        }

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def _check_built(self) -> None:
        if not self._built:
            raise RuntimeError("Index not built. Call .build() first.")

    def retrieve_tfidf(self, query: str, k: int = 100) -> List[Tuple[int, float]]:
        """Retrieve top-k passages by TF-IDF cosine similarity.

        Args:
            query: Free-text query string.
            k: Number of results to return.

        Returns:
            List of (pid, score) tuples sorted by descending score.
        """
        self._check_built()
        if self._vectorizer is None or self._tfidf_matrix is None or self._pids is None:
            raise RuntimeError("TF-IDF index state is incomplete")
        q_vec = self._vectorizer.transform([query])  # (1, V), L2-normalized by default
        scores = (q_vec @ self._tfidf_matrix.T).toarray().ravel()  # (N,)
        top_idx = _top_k_indices(scores, k)
        return [(int(self._pids[i]), float(scores[i])) for i in top_idx]

    def retrieve_bm25(self, query: str, k: int = 100) -> List[Tuple[int, float]]:
        """Retrieve top-k passages by BM25Okapi score.

        Args:
            query: Free-text query string.
            k: Number of results to return.

        Returns:
            List of (pid, score) tuples sorted by descending score.
        """
        self._check_built()
        if self._bm25 is None or self._pids is None:
            raise RuntimeError("BM25 index state is incomplete")
        tokens = tokenize(query)
        scores = self._bm25.get_scores(tokens)  # numpy array (N,)
        top_idx = _top_k_indices(scores, k)
        return [(int(self._pids[i]), float(scores[i])) for i in top_idx]

    def retrieve_tfidf_feedback(
        self,
        query: str,
        relevant_pids: Sequence[int],
        irrelevant_pids: Sequence[int] = (),
        k: int = 100,
        *,
        alpha: float = 1.0,
        beta: float = 0.75,
        gamma: float = 0.15,
    ) -> List[Tuple[int, float]]:
        """Rank with a transparent Rocchio query updated by explicit feedback."""
        self._check_built()
        if self._vectorizer is None or self._tfidf_matrix is None or self._pids is None:
            raise RuntimeError("TF-IDF index state is incomplete")
        if self._pid_to_row is None:
            raise RuntimeError("TF-IDF passage ID mapping is unavailable")
        if k < 0:
            raise ValueError("k must be non-negative")
        if any(
            not math.isfinite(value) or value < 0
            for value in (alpha, beta, gamma)
        ):
            raise ValueError("Rocchio parameters must be finite and non-negative")
        positives = [int(pid) for pid in relevant_pids]
        negatives = [int(pid) for pid in irrelevant_pids]
        if len(set(positives)) != len(positives):
            raise ValueError("Relevant feedback passage IDs must be unique")
        if len(set(negatives)) != len(negatives):
            raise ValueError("Irrelevant feedback passage IDs must be unique")
        if set(positives).intersection(negatives):
            raise ValueError("A passage cannot be both relevant and irrelevant")
        unknown = (set(positives) | set(negatives)).difference(self._pid_to_row)
        if unknown:
            raise ValueError(f"Feedback passage IDs are not in the corpus: {sorted(unknown)}")
        if k == 0:
            return []

        query_vector = self._vectorizer.transform([query]).tocsr()
        updated_query = query_vector.multiply(alpha)
        if positives:
            positive_rows = [self._pid_to_row[pid] for pid in positives]
            updated_query = updated_query + beta * self._tfidf_matrix[positive_rows].mean(axis=0)
        if negatives:
            negative_rows = [self._pid_to_row[pid] for pid in negatives]
            updated_query = updated_query - gamma * self._tfidf_matrix[negative_rows].mean(axis=0)
        updated_query = normalize(sp.csr_matrix(updated_query), norm="l2")
        scores = (updated_query @ self._tfidf_matrix.T).toarray().ravel()
        top_idx = _top_k_indices(scores, k)
        return [(int(self._pids[i]), float(scores[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# Module-level convenience API (lazy singleton)
# ---------------------------------------------------------------------------

_INDEX: Optional[BaselineIndex] = None


def get_index() -> BaselineIndex:
    """Return the module-level singleton index (must have called build_index first)."""
    if _INDEX is None or not _INDEX._built:
        raise RuntimeError("Call build_index() before using retrieve_*")
    return _INDEX


def build_index(
    corpus_df: Optional[pd.DataFrame] = None,
    corpus_path: Union[str, Path] = "data/processed/corpus.parquet",
    artifacts_dir: Union[str, Path] = "artifacts",
    force_rebuild: bool = False,
) -> Dict[str, float]:
    """Build (or load cached) global TF-IDF and BM25 indexes.

    Args:
        corpus_df: Pre-loaded corpus DataFrame; loaded from disk if None.
        corpus_path: Path to corpus parquet (used when corpus_df is None).
        artifacts_dir: Directory for caching index files.
        force_rebuild: Ignore cached indexes and rebuild from scratch.

    Returns:
        dict with build timing information.
    """
    global _INDEX
    if corpus_df is None:
        logger.info("Loading corpus from %s", corpus_path)
        corpus_df = pd.read_parquet(corpus_path)

    _INDEX = BaselineIndex()
    return _INDEX.build(corpus_df, artifacts_dir=artifacts_dir, force_rebuild=force_rebuild)


def retrieve_tfidf(query: str, k: int = 100) -> List[Tuple[int, float]]:
    """Module-level TF-IDF retrieval using the cached singleton index."""
    return get_index().retrieve_tfidf(query, k)


def retrieve_bm25(query: str, k: int = 100) -> List[Tuple[int, float]]:
    """Module-level BM25 retrieval using the cached singleton index."""
    return get_index().retrieve_bm25(query, k)


def retrieve_tfidf_feedback(
    query: str,
    relevant_pids: Sequence[int],
    irrelevant_pids: Sequence[int] = (),
    k: int = 100,
    *,
    alpha: float = 1.0,
    beta: float = 0.75,
    gamma: float = 0.15,
) -> List[Tuple[int, float]]:
    """Module-level TF-IDF retrieval with explicit Rocchio feedback."""
    return get_index().retrieve_tfidf_feedback(
        query,
        relevant_pids,
        irrelevant_pids,
        k,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )


# ---------------------------------------------------------------------------
# TREC-format writing
# ---------------------------------------------------------------------------

def write_trec_run(
    results: Dict[int, List[Tuple[int, float]]],
    run_file: Union[str, Path],
    system_name: str,
) -> None:
    """Write retrieval results as a valid TREC-format run file.

    Format per line:
        ``qid Q0 pid rank score system_name``

    Args:
        results: Mapping of query_id → [(pid, score), …] sorted by score desc.
        run_file: Output path for the TREC run file.
        system_name: Identifier appearing in column 6.
    """
    run_file = Path(run_file)
    run_file.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for qid, hits in results.items():
        for rank, (pid, score) in enumerate(hits, start=1):
            lines.append(f"{qid} Q0 {pid} {rank} {score:.6f} {system_name}")
    run_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %d lines to %s", len(lines), run_file)


# ---------------------------------------------------------------------------
# Batch retrieval helper
# ---------------------------------------------------------------------------

def retrieve_all(
    queries_df: pd.DataFrame,
    method: str,
    k: int = 100,
) -> Tuple[Dict[int, List[Tuple[int, float]]], Dict[str, float]]:
    """Run retrieval for all queries in *queries_df* and return timing stats.

    Args:
        queries_df: DataFrame with columns [query_id, query_text].
        method: ``'tfidf'`` or ``'bm25'``.
        k: Depth of retrieval.

    Returns:
        (results_dict, latency_stats) where latency_stats contains
        ``mean_ms``, ``median_ms``, ``p95_ms``, ``total_s``.
    """
    index = get_index()
    retrieval_methods = {
        "tfidf": index.retrieve_tfidf,
        "bm25": index.retrieve_bm25,
    }
    if method not in retrieval_methods:
        raise ValueError(f"Unknown retrieval method: {method!r}")
    if k < 0:
        raise ValueError("k must be non-negative")
    fn = retrieval_methods[method]

    results: Dict[int, List[Tuple[int, float]]] = {}
    latencies: List[float] = []

    t_total = time.perf_counter()
    for _, row in queries_df.iterrows():
        qid = int(row["query_id"])
        q = str(row["query_text"])
        t0 = time.perf_counter()
        results[qid] = fn(q, k)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)
        logger.info("%s qid=%s retrieval latency: %.2f ms", method, qid, latency_ms)

    total_s = time.perf_counter() - t_total
    arr = np.asarray(latencies, dtype=np.float64)
    stats: Dict[str, float] = {
        "mean_ms": float(arr.mean()) if arr.size else 0.0,
        "median_ms": float(np.median(arr)) if arr.size else 0.0,
        "p95_ms": float(np.percentile(arr, 95)) if arr.size else 0.0,
        "total_s": total_s,
    }
    return results, stats
