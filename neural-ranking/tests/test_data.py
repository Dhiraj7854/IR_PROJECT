"""Pytest tests for data preparation, split integrity, and triplet mining validation."""

import sys
from pathlib import Path
import pandas as pd
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "processed"


@pytest.fixture(scope="module")
def corpus_df():
    """Load corpus parquet file."""
    path = DATA_DIR / "corpus.parquet"
    assert path.exists(), f"Missing corpus file: {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def train_queries():
    path = DATA_DIR / "train_queries.parquet"
    assert path.exists(), f"Missing train queries file: {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def val_queries():
    path = DATA_DIR / "val_queries.parquet"
    assert path.exists(), f"Missing val queries file: {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def test_queries():
    path = DATA_DIR / "test_queries.parquet"
    assert path.exists(), f"Missing test queries file: {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def train_qrels():
    path = DATA_DIR / "train_qrels.parquet"
    assert path.exists(), f"Missing train qrels file: {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def val_qrels():
    path = DATA_DIR / "val_qrels.parquet"
    assert path.exists(), f"Missing val qrels file: {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def test_qrels():
    path = DATA_DIR / "test_qrels.parquet"
    assert path.exists(), f"Missing test qrels file: {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def train_triplets():
    path = DATA_DIR / "train_triplets.parquet"
    assert path.exists(), f"Missing train triplets file: {path}"
    return pd.read_parquet(path)


def test_corpus_size_and_schema(corpus_df):
    """Verify corpus has 100,000 unique passages with correct schema."""
    assert len(corpus_df) == 100000
    assert list(corpus_df.columns) == ["pid", "passage_text"]
    assert corpus_df["pid"].nunique() == 100000
    assert corpus_df["passage_text"].nunique() == 100000


def test_query_splits_are_strictly_disjoint(train_queries, val_queries, test_queries):
    """Test requirement (b): query splits (train/val/test) are strictly disjoint."""
    train_qids = set(train_queries["query_id"])
    val_qids = set(val_queries["query_id"])
    test_qids = set(test_queries["query_id"])

    assert len(train_qids) == 3500
    assert len(val_qids) == 750
    assert len(test_qids) == 750

    # Test vs Train/Val disjointness
    assert len(test_qids.intersection(train_qids)) == 0, "Test queries overlap with train queries!"
    assert len(test_qids.intersection(val_qids)) == 0, "Test queries overlap with val queries!"

    # Train vs Val disjointness
    assert len(train_qids.intersection(val_qids)) == 0, "Train queries overlap with val queries!"


def test_no_negative_pid_in_qrels(train_triplets, train_qrels):
    """Test requirement (a): no pid in the negative set appears in query's qrels."""
    # Build ground truth relevance map: qid -> set of relevant pids
    rel_map = {}
    for _, row in train_qrels.iterrows():
        qid = int(row["query_id"])
        pid = int(row["relevant_pid"])
        if qid not in rel_map:
            rel_map[qid] = set()
        rel_map[qid].add(pid)

    assert len(train_triplets) > 0, "Triplets dataframe is empty!"

    for idx, row in train_triplets.iterrows():
        qid = int(row["qid"])
        neg_pid = int(row["neg_pid"])
        relevant_pids = rel_map.get(qid, set())

        assert neg_pid not in relevant_pids, (
            f"Row {idx}: negative pid {neg_pid} is marked as relevant in qrels for qid {qid}!"
        )


def test_no_empty_texts(corpus_df, train_queries, val_queries, test_queries, train_triplets):
    """Test requirement (c): no empty or whitespace-only texts in corpus, queries, or triplets."""
    # Corpus passages
    assert not corpus_df["passage_text"].isna().any(), "Found NaN in corpus passage_text"
    assert (corpus_df["passage_text"].str.strip().str.len() > 0).all(), "Found empty passage in corpus"

    # Queries
    for split_name, q_df in [("train", train_queries), ("val", val_queries), ("test", test_queries)]:
        assert not q_df["query_text"].isna().any(), f"Found NaN in {split_name} queries"
        assert (q_df["query_text"].str.strip().str.len() > 0).all(), f"Found empty text in {split_name} queries"

    # Triplets
    assert not train_triplets["query_text"].isna().any(), "Found NaN in triplet query_text"
    assert (train_triplets["query_text"].str.strip().str.len() > 0).all(), "Found empty triplet query_text"

    assert not train_triplets["pos_text"].isna().any(), "Found NaN in triplet pos_text"
    assert (train_triplets["pos_text"].str.strip().str.len() > 0).all(), "Found empty triplet pos_text"

    assert not train_triplets["neg_text"].isna().any(), "Found NaN in triplet neg_text"
    assert (train_triplets["neg_text"].str.strip().str.len() > 0).all(), "Found empty triplet neg_text"


def test_all_qrels_pids_in_corpus(corpus_df, train_qrels, val_qrels, test_qrels):
    """Verify that all relevant passages in qrels exist within the 100k corpus."""
    corpus_pids = set(corpus_df["pid"])

    for split_name, qrels_df in [("train", train_qrels), ("val", val_qrels), ("test", test_qrels)]:
        qrel_pids = set(qrels_df["relevant_pid"])
        missing = qrel_pids - corpus_pids
        assert len(missing) == 0, f"{split_name} qrels has pids not in the 100k corpus: {missing}"
