"""Tests for bi-encoder retrieval without downloading a transformer checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.biencoder import configure_retrieval, retrieve_biencoder


class ToyEncoder:
    """Deterministic encoder separating machine-learning passages from others."""

    device = torch.device("cpu")

    def encode(
        self,
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ):
        del batch_size, show_progress_bar, convert_to_numpy, normalize_embeddings
        vectors = [
            [1.0, 0.0] if "machine learning" in text.lower() else [0.0, 1.0]
            for text in texts
        ]
        return np.asarray(vectors, dtype=np.float32)


def test_machine_learning_query_retrieves_relevant_passage_in_top_three(tmp_path):
    corpus = [
        f"An unrelated passage about astronomy and stars, document {index}."
        for index in range(50)
    ]
    relevant_index = 23
    corpus[relevant_index] = "This passage explains machine learning algorithms."
    pids = list(range(1000, 1050))

    configure_retrieval(
        corpus,
        pids=pids,
        model=ToyEncoder(),
        cache_path=tmp_path / "corpus_embeddings.npy",
    )

    results = retrieve_biencoder("what is machine learning", k=3)

    assert len(results) == 3
    assert len({pid for pid, _ in results}) == 3
    assert pids[relevant_index] in {pid for pid, _ in results}
