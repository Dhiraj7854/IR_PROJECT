"""Pytest tests for TF-IDF and BM25 baseline retrieval correctness."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import (
    BaselineIndex,
    CompactBM25Index,
    CompactTfidfIndex,
    tokenize,
)

# ---------------------------------------------------------------------------
# Fixtures — build a tiny in-memory index so tests run instantly
# ---------------------------------------------------------------------------

MINI_CORPUS = pd.DataFrame(
    {
        "pid": list(range(20)),
        "passage_text": [
            "the cat sat on the mat in the living room",
            "dogs are friendly domestic animals often kept as pets",
            "python is a popular programming language used for data science",
            "machine learning is a subset of artificial intelligence",
            "the stock market crashed after the pandemic outbreak",
            "pizza is an italian dish made with tomato sauce and cheese",
            "climate change is causing rising sea levels around the world",
            "the human brain contains approximately 86 billion neurons",
            "solar panels convert sunlight into electrical energy efficiently",
            "vaccines protect against infectious diseases by building immunity",
            "the amazon rainforest is home to millions of plant and animal species",
            "quantum computing uses qubits to perform calculations simultaneously",
            "the eiffel tower is located in paris france and was built in 1889",
            "water molecules consist of two hydrogen atoms and one oxygen atom",
            "marathon runners train for months to complete 26 miles",
            "deep sea fish have adapted to survive in extreme pressure darkness",
            "the roman empire fell in 476 ad ending the ancient period",
            "black holes have gravitational pull so strong light cannot escape",
            "meditation reduces stress improves focus and promotes well being",
            "antibiotics are used to treat bacterial infections not viral ones",
        ],
    }
)


@pytest.fixture(scope="module")
def mini_index(tmp_path_factory):
    """Build a tiny BaselineIndex once for the whole test module."""
    tmp = tmp_path_factory.mktemp("artifacts")
    idx = BaselineIndex()
    idx.build(MINI_CORPUS, artifacts_dir=tmp)
    return idx


# ---------------------------------------------------------------------------
# Core acceptance test: both methods return k unique pids
# ---------------------------------------------------------------------------

def test_bm25_tokenization_lowercases_and_removes_stopwords():
    assert tokenize("What does heterogeneous mean in chemistry?") == [
        "does",
        "heterogeneous",
        "mean",
        "chemistry",
    ]


@pytest.mark.parametrize("method", ["tfidf", "bm25"])
@pytest.mark.parametrize("k", [1, 5, 10, 20])
def test_returns_k_unique_pids(mini_index, method, k):
    """Both TF-IDF and BM25 return exactly k unique pids for a known query."""
    query = "machine learning artificial intelligence neural network"
    if method == "tfidf":
        results = mini_index.retrieve_tfidf(query, k=k)
    else:
        results = mini_index.retrieve_bm25(query, k=k)

    assert len(results) == k, f"{method} returned {len(results)} results, expected {k}"
    pids = [pid for pid, _ in results]
    assert len(set(pids)) == k, f"{method} returned duplicate pids: {pids}"


def test_tfidf_k_unique_pids_corpus_query(mini_index):
    """TF-IDF returns k unique pids for a known query — explicit check."""
    k = 10
    results = mini_index.retrieve_tfidf("python programming language data science", k=k)
    pids = [p for p, _ in results]
    assert len(pids) == k
    assert len(set(pids)) == k


def test_bm25_k_unique_pids_corpus_query(mini_index):
    """BM25 returns k unique pids for a known query — explicit check."""
    k = 10
    results = mini_index.retrieve_bm25("python programming language data science", k=k)
    pids = [p for p, _ in results]
    assert len(pids) == k
    assert len(set(pids)) == k


@pytest.mark.parametrize(
    ("compact_type", "baseline_method"),
    [
        (CompactBM25Index, "retrieve_bm25"),
        (CompactTfidfIndex, "retrieve_tfidf"),
    ],
)
@pytest.mark.parametrize(
    "query",
    [
        "python programming language data science",
        "artificial artificial intelligence",
        "wordnotpresent",
    ],
)
def test_compact_index_matches_existing_lexical_index(
    mini_index,
    compact_type,
    baseline_method,
    query,
):
    compact = compact_type(MINI_CORPUS)

    expected = getattr(mini_index, baseline_method)(query, k=10)
    actual = compact.retrieve(query, k=10)

    assert [pid for pid, _ in actual] == [pid for pid, _ in expected]
    assert [score for _, score in actual] == pytest.approx(
        [score for _, score in expected],
        rel=1e-5,
        abs=1e-6,
    )


# ---------------------------------------------------------------------------
# Relevance: top result should make topical sense
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["tfidf", "bm25"])
def test_top_result_topical(mini_index, method):
    """The top-1 result for 'python programming' should be the programming passage."""
    query = "python programming language"
    if method == "tfidf":
        results = mini_index.retrieve_tfidf(query, k=5)
    else:
        results = mini_index.retrieve_bm25(query, k=5)

    top_pid = results[0][0]
    # pid=2 is the Python programming passage
    assert top_pid == 2, (
        f"{method}: expected top pid=2 (Python), got {top_pid}. "
        f"Top-5: {results}"
    )


@pytest.mark.parametrize("method", ["tfidf", "bm25"])
def test_scores_descending(mini_index, method):
    """Returned scores must be in non-increasing order."""
    query = "climate change rising sea levels global warming"
    if method == "tfidf":
        results = mini_index.retrieve_tfidf(query, k=10)
    else:
        results = mini_index.retrieve_bm25(query, k=10)

    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True), (
        f"{method}: scores not sorted descending: {scores}"
    )


# ---------------------------------------------------------------------------
# Cache round-trip
# ---------------------------------------------------------------------------

def test_cache_roundtrip(tmp_path):
    """Index built once, reloaded from cache, and still returns correct results."""
    idx1 = BaselineIndex()
    idx1.build(MINI_CORPUS, artifacts_dir=tmp_path)
    cache_path = tmp_path / "baseline_index.pkl"
    assert cache_path.exists()
    with cache_path.open("rb") as cache_file:
        cached_state = pickle.load(cache_file)
    assert len(cached_state["tokenized_corpus"]) == len(MINI_CORPUS)

    q = "human brain neurons"
    r1_tfidf = idx1.retrieve_tfidf(q, k=3)
    r1_bm25 = idx1.retrieve_bm25(q, k=3)

    # Reload from cache
    idx2 = BaselineIndex()
    idx2.build(MINI_CORPUS, artifacts_dir=tmp_path)

    r2_tfidf = idx2.retrieve_tfidf(q, k=3)
    r2_bm25 = idx2.retrieve_bm25(q, k=3)

    assert [p for p, _ in r1_tfidf] == [p for p, _ in r2_tfidf], "TF-IDF cache mismatch"
    assert [p for p, _ in r1_bm25] == [p for p, _ in r2_bm25], "BM25 cache mismatch"


def test_cache_is_rebuilt_for_changed_corpus(tmp_path):
    """A cache from a different corpus must not be reused."""
    BaselineIndex().build(MINI_CORPUS, artifacts_dir=tmp_path)
    changed_corpus = MINI_CORPUS.copy()
    changed_corpus.loc[2, "passage_text"] = "orchards grow apples and pears"

    rebuilt = BaselineIndex()
    rebuilt.build(changed_corpus, artifacts_dir=tmp_path)

    assert rebuilt.retrieve_bm25("orchards apples", k=1)[0][0] == 2


def test_rocchio_feedback_moves_relevant_passage_up(mini_index):
    before = mini_index.retrieve_tfidf("programming", k=20)
    after = mini_index.retrieve_tfidf_feedback(
        "programming",
        relevant_pids=[3],
        irrelevant_pids=[2],
        k=20,
    )
    before_rank = [pid for pid, _ in before].index(3)
    after_rank = [pid for pid, _ in after].index(3)

    assert after_rank <= before_rank
    assert after[0][0] == 3


def test_rocchio_rejects_duplicate_or_conflicting_feedback(mini_index):
    with pytest.raises(ValueError, match="unique"):
        mini_index.retrieve_tfidf_feedback("python", [2, 2], k=5)
    with pytest.raises(ValueError, match="both relevant and irrelevant"):
        mini_index.retrieve_tfidf_feedback("python", [2], [2], k=5)


# ---------------------------------------------------------------------------
# TREC format
# ---------------------------------------------------------------------------

def test_trec_run_format(mini_index, tmp_path):
    """write_trec_run produces valid TREC format lines."""
    from src.baseline import write_trec_run

    results = {
        12345: mini_index.retrieve_tfidf("climate change", k=5),
        99999: mini_index.retrieve_bm25("water molecules hydrogen", k=5),
    }
    run_file = tmp_path / "test_run.txt"
    write_trec_run(results, run_file, system_name="tfidf")

    lines = run_file.read_text().strip().splitlines()
    assert len(lines) == 10  # 2 queries × 5 results

    for line in lines:
        parts = line.split()
        assert len(parts) == 6, f"Expected 6 fields, got {len(parts)}: {line!r}"
        qid, q0, pid, rank, score, sysname = parts
        assert q0 == "Q0", f"Second field must be 'Q0', got {q0!r}"
        int(qid)    # must parse as int
        int(pid)    # must parse as int
        int(rank)   # must parse as int
        float(score)  # must parse as float
        assert sysname == "tfidf"


# ---------------------------------------------------------------------------
# Live data integration (skip if processed data absent)
# ---------------------------------------------------------------------------

PROCESSED = PROJECT_ROOT / "data" / "processed"


@pytest.mark.skipif(
    not (PROCESSED / "corpus.parquet").exists(),
    reason="Processed corpus not found; run scripts/prepare_data.py first",
)
def test_live_index_returns_k_unique(tmp_path):
    """Build from real 100k corpus and confirm k=100 unique pids returned."""
    corpus_df = pd.read_parquet(PROCESSED / "corpus.parquet")
    idx = BaselineIndex()
    idx.build(corpus_df, artifacts_dir=tmp_path)

    k = 100
    query = "what causes climate change"

    tfidf_results = idx.retrieve_tfidf(query, k=k)
    bm25_results = idx.retrieve_bm25(query, k=k)

    assert len(tfidf_results) == k
    assert len(set(p for p, _ in tfidf_results)) == k

    assert len(bm25_results) == k
    assert len(set(p for p, _ in bm25_results)) == k
