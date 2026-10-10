"""Data loading, preprocessing, sampling, split management, and triplet mining for MS MARCO."""

from __future__ import annotations

import math
import os
import re
import socket
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.config import Config, load_config

TOKEN_PATTERN = re.compile(r"\w+")


def configure_ipv4_preference() -> None:
    """Ensure socket resolution prioritizes IPv4 to avoid broken IPv6 network routes."""
    if getattr(socket, "_ipv4_patched", False):
        return
    orig_getaddrinfo = socket.getaddrinfo

    def getaddrinfo_v4(host, port, family=0, type=0, proto=0, flags=0):
        if family in (0, socket.AF_UNSPEC):
            family = socket.AF_INET
        return orig_getaddrinfo(host, port, family, type, proto, flags)

    socket.getaddrinfo = getaddrinfo_v4
    socket._ipv4_patched = True


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    if not text:
        return []
    return TOKEN_PATTERN.findall(text.lower())


class FastBM25Searcher:
    """Fast BM25 retrieval indexer using an inverted index with BM25Okapi scoring."""

    def __init__(self, tokenized_corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus_size = len(tokenized_corpus)
        self.doc_lens = np.array([len(doc) for doc in tokenized_corpus], dtype=np.float32)
        self.avgdl = float(self.doc_lens.mean()) if self.corpus_size > 0 else 1.0
        self.k1 = k1
        self.b = b

        # Build document frequencies and inverted index
        inv_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        df: Dict[str, int] = defaultdict(int)

        for doc_id, doc in enumerate(tokenized_corpus):
            term_counts: Dict[str, int] = defaultdict(int)
            for token in doc:
                term_counts[token] += 1
            for token, count in term_counts.items():
                inv_index[token].append((doc_id, count))
                df[token] += 1

        # Calculate standard BM25 IDF
        self.idf: Dict[str, float] = {}
        for term, freq in df.items():
            self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

        # Convert inverted index to numpy arrays
        self.inverted_index: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for term, postings in inv_index.items():
            doc_ids = np.array([p[0] for p in postings], dtype=np.int32)
            counts = np.array([p[1] for p in postings], dtype=np.float32)
            self.inverted_index[term] = (doc_ids, counts)

    def search_top_negative(
        self, query_tokens: List[str], exclude_pids: Set[int], top_n: int = 50
    ) -> Tuple[int, float]:
        """Find the highest-scoring passage that is not in the exclude_pids set.

        Returns:
            Tuple of (negative_pid, bm25_score).
        """
        if not query_tokens:
            # Fallback to any non-excluded passage
            for pid in range(self.corpus_size):
                if pid not in exclude_pids:
                    return pid, 0.0
            return 0, 0.0

        scores = np.zeros(self.corpus_size, dtype=np.float32)
        denom_cache = self.k1 * (1.0 - self.b + self.b * (self.doc_lens / self.avgdl))

        has_matches = False
        for term in query_tokens:
            if term in self.inverted_index:
                doc_ids, counts = self.inverted_index[term]
                idf = self.idf.get(term, 0.0)
                if idf <= 0:
                    continue
                has_matches = True
                numer = counts * (self.k1 + 1.0)
                denom = counts + denom_cache[doc_ids]
                scores[doc_ids] += idf * (numer / denom)

        if not has_matches:
            for pid in range(self.corpus_size):
                if pid not in exclude_pids:
                    return pid, 0.0
            return 0, 0.0

        # Retrieve top candidates
        cand_k = min(top_n, self.corpus_size)
        top_candidates = np.argpartition(scores, -cand_k)[-cand_k:]
        sorted_top = top_candidates[np.argsort(scores[top_candidates])[::-1]]

        for pid in sorted_top:
            if int(pid) not in exclude_pids:
                return int(pid), float(scores[pid])

        # If all top candidates were relevant, fallback to first non-relevant
        for pid in range(self.corpus_size):
            if pid not in exclude_pids:
                return pid, float(scores[pid])

        return 0, 0.0


def download_and_sample_msmarco(
    dataset_name: str = "microsoft/ms_marco",
    dataset_version: str = "v1.1",
    corpus_size: int = 100000,
    num_queries: int = 5000,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Stream MS MARCO from HuggingFace and extract 100k unique passages and 5k queries.

    Returns:
        corpus_df: DataFrame with columns ['pid', 'passage_text']
        queries_records: List of dicts with query_id, query_text, and relevant passages
    """
    configure_ipv4_preference()

    # Normalize HF dataset identifier
    hf_dataset_id = "microsoft/ms_marco" if "ms_marco" in dataset_name or "msmarco" in dataset_name else dataset_name
    print(f"Loading dataset '{hf_dataset_id}' (version '{dataset_version}')...")

    ds = load_dataset(hf_dataset_id, dataset_version, split="train", streaming=True)

    selected_queries: List[Dict[str, Any]] = []
    unique_passages: Set[str] = set()
    needed_relevant_passages: Set[str] = set()

    print("Streaming dataset to extract queries and unique passages...")
    pbar = tqdm(total=corpus_size, desc="Collecting passages")

    for item in ds:
        q_text = item["query"].strip()
        qid = int(item["query_id"])
        passages_raw = item["passages"]["passage_text"]
        is_selected_raw = item["passages"]["is_selected"]

        valid_p_and_sel = [
            (p.strip(), sel)
            for p, sel in zip(passages_raw, is_selected_raw)
            if p and p.strip()
        ]
        if not q_text or not valid_p_and_sel:
            continue

        relevant_passages = [p for p, sel in valid_p_and_sel if sel == 1]
        all_candidate_passages = [p for p, _ in valid_p_and_sel]

        # Only select queries that have at least one relevant passage
        if not relevant_passages:
            continue

        if len(selected_queries) < num_queries:
            selected_queries.append({
                "query_id": qid,
                "query_text": q_text,
                "relevant_passages": relevant_passages,
            })
            for p in relevant_passages:
                needed_relevant_passages.add(p)

        # Collect candidate passages into our corpus pool
        for p in all_candidate_passages:
            if p not in unique_passages:
                unique_passages.add(p)
                pbar.update(1)
                if len(unique_passages) >= corpus_size and len(selected_queries) >= num_queries:
                    break

        if len(unique_passages) >= corpus_size and len(selected_queries) >= num_queries:
            break

    pbar.close()

    # Ensure all relevant passages for our selected queries are in the corpus
    for p in needed_relevant_passages:
        unique_passages.add(p)

    # Convert to sorted list and trim to exact corpus_size
    # Prioritize needed relevant passages so none are dropped
    other_passages = list(unique_passages - needed_relevant_passages)
    needed_list = list(needed_relevant_passages)
    remaining_slots = corpus_size - len(needed_list)
    final_passages = needed_list + other_passages[:remaining_slots]

    # Build corpus DataFrame
    corpus_records = [{"pid": i, "passage_text": text} for i, text in enumerate(final_passages)]
    corpus_df = pd.DataFrame(corpus_records)

    return corpus_df, selected_queries


def split_queries_and_qrels(
    selected_queries: List[Dict[str, Any]],
    passage_to_pid: Dict[str, int],
    train_size: int = 3500,
    val_size: int = 750,
    test_size: int = 750,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Split 5,000 queries into disjoint train, val, and test splits with corresponding qrels."""
    total_needed = train_size + val_size + test_size
    if len(selected_queries) < total_needed:
        raise ValueError(
            f"Not enough queries: need {total_needed}, but got {len(selected_queries)}"
        )

    # Slices are strictly disjoint
    train_raw = selected_queries[:train_size]
    val_raw = selected_queries[train_size : train_size + val_size]
    test_raw = selected_queries[train_size + val_size : total_needed]

    splits_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}

    for name, raw_list in [("train", train_raw), ("val", val_raw), ("test", test_raw)]:
        query_rows = []
        qrel_rows = []

        for item in raw_list:
            qid = item["query_id"]
            q_text = item["query_text"]
            query_rows.append({"query_id": qid, "query_text": q_text})

            for p_text in item["relevant_passages"]:
                if p_text in passage_to_pid:
                    qrel_rows.append({
                        "query_id": qid,
                        "relevant_pid": passage_to_pid[p_text],
                    })

        q_df = pd.DataFrame(query_rows).drop_duplicates(subset=["query_id"])
        qrels_df = pd.DataFrame(qrel_rows).drop_duplicates()
        splits_data[name] = (q_df, qrels_df)

    return splits_data


def generate_train_triplets(
    train_queries_df: pd.DataFrame,
    train_qrels_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    searcher: FastBM25Searcher,
) -> pd.DataFrame:
    """Generate training triplets with BM25 hard negatives for train queries.

    Returns:
        DataFrame with columns: [qid, query_text, pos_pid, pos_text, neg_pid, neg_text]
    """
    pid_to_text = dict(zip(corpus_df["pid"], corpus_df["passage_text"]))

    # Group relevant pids per query_id
    qrels_map: Dict[int, List[int]] = defaultdict(list)
    for _, row in train_qrels_df.iterrows():
        qrels_map[int(row["query_id"])].append(int(row["relevant_pid"]))

    triplets: List[Dict[str, Any]] = []

    print("Mining BM25 hard negatives for training triplets...")
    for _, row in tqdm(train_queries_df.iterrows(), total=len(train_queries_df), desc="Mining triplets"):
        qid = int(row["query_id"])
        q_text = str(row["query_text"])
        rel_pids = qrels_map.get(qid, [])
        if not rel_pids:
            continue

        pos_pid = rel_pids[0]
        pos_text = pid_to_text.get(pos_pid, "")
        if not pos_text:
            continue

        q_tokens = tokenize(q_text)
        exclude_set = set(rel_pids)

        neg_pid, _ = searcher.search_top_negative(q_tokens, exclude_set)
        neg_text = pid_to_text.get(neg_pid, "")

        triplets.append({
            "qid": qid,
            "query_text": q_text,
            "pos_pid": pos_pid,
            "pos_text": pos_text,
            "neg_pid": neg_pid,
            "neg_text": neg_text,
        })

    return pd.DataFrame(triplets)


def prepare_msmarco_pipeline(
    config_path: Union[str, Path] = "configs/config.yaml",
    output_dir: Union[str, Path] = "data/processed",
) -> Dict[str, Any]:
    """Execute end-to-end data preparation pipeline based on config."""
    cfg = load_config(config_path)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    corpus_size = cfg.corpus_subset_size
    train_size = cfg.query_splits.train
    val_size = cfg.query_splits.val
    test_size = cfg.query_splits.test
    total_queries = train_size + val_size + test_size

    # 1. Download and sample corpus & queries
    corpus_df, selected_queries = download_and_sample_msmarco(
        dataset_name=cfg.dataset.name,
        corpus_size=corpus_size,
        num_queries=total_queries,
    )

    # Save corpus parquet
    corpus_file = out_path / "corpus.parquet"
    corpus_df.to_parquet(corpus_file, index=False)
    print(f"Saved corpus to {corpus_file} (rows: {len(corpus_df)})")

    # 2. Split queries and qrels
    passage_to_pid = dict(zip(corpus_df["passage_text"], corpus_df["pid"]))
    splits_data = split_queries_and_qrels(
        selected_queries=selected_queries,
        passage_to_pid=passage_to_pid,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )

    # Save splits
    for split_name, (q_df, qrels_df) in splits_data.items():
        q_file = out_path / f"{split_name}_queries.parquet"
        qrels_file = out_path / f"{split_name}_qrels.parquet"
        q_df.to_parquet(q_file, index=False)
        qrels_df.to_parquet(qrels_file, index=False)
        print(f"Saved {split_name} queries ({len(q_df)}) and qrels ({len(qrels_df)})")

    # 3. Build BM25 index & mine hard negatives
    print("Building BM25 index over 100k corpus...")
    tokenized_corpus = [tokenize(text) for text in corpus_df["passage_text"]]
    searcher = FastBM25Searcher(tokenized_corpus)

    train_q_df, train_qrels_df = splits_data["train"]
    triplets_df = generate_train_triplets(train_q_df, train_qrels_df, corpus_df, searcher)

    triplets_file = out_path / "train_triplets.parquet"
    triplets_df.to_parquet(triplets_file, index=False)
    print(f"Saved train triplets to {triplets_file} (rows: {len(triplets_df)})")

    # Calculate statistics
    words_per_doc = [len(toks) for toks in tokenized_corpus]
    avg_words = float(np.mean(words_per_doc))
    chars_per_doc = [len(text) for text in corpus_df["passage_text"]]
    avg_chars = float(np.mean(chars_per_doc))

    stats = {
        "corpus_size": len(corpus_df),
        "queries_per_split": {
            "train": len(splits_data["train"][0]),
            "val": len(splits_data["val"][0]),
            "test": len(splits_data["test"][0]),
        },
        "triplets_generated": len(triplets_df),
        "avg_passage_length_words": avg_words,
        "avg_passage_length_chars": avg_chars,
    }

    return stats


def load_corpus(path: Union[str, Path] = "data/processed/corpus.parquet") -> pd.DataFrame:
    """Load corpus parquet file."""
    return pd.read_parquet(path)


def load_queries(split: str = "train", data_dir: Union[str, Path] = "data/processed") -> pd.DataFrame:
    """Load query split parquet file."""
    return pd.read_parquet(Path(data_dir) / f"{split}_queries.parquet")


def load_qrels(split: str = "train", data_dir: Union[str, Path] = "data/processed") -> pd.DataFrame:
    """Load qrels split parquet file."""
    return pd.read_parquet(Path(data_dir) / f"{split}_qrels.parquet")


def load_triplets(path: Union[str, Path] = "data/processed/train_triplets.parquet") -> pd.DataFrame:
    """Load train triplets parquet file."""
    return pd.read_parquet(path)
