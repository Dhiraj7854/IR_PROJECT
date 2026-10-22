"""Tests for model-independent bi-encoder training helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_biencoder import _make_training_collator
from src.biencoder import load_biencoder


class ToyTokenizer:
    def tokenize(self, texts):
        return {
            "input_ids": torch.tensor([[len(text)] for text in texts]),
            "attention_mask": torch.ones((len(texts), 1), dtype=torch.int64),
            "text": list(texts),
        }


def test_training_collator_returns_tokenized_tensor_batches():
    collate = _make_training_collator(ToyTokenizer())
    examples = [
        SimpleNamespace(texts=["query one", "positive", "negative"], label=0),
        SimpleNamespace(texts=["query two", "positive two", "negative two"], label=0),
    ]

    features, labels = collate(examples)

    assert len(features) == 3
    assert all(torch.is_tensor(value) for item in features for value in item.values())
    assert all("text" not in item for item in features)
    assert [item["input_ids"].shape for item in features] == [(2, 1)] * 3
    assert labels.shape == (2,)


def test_load_biencoder_rejects_incomplete_local_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="no modules.json"):
        load_biencoder(tmp_path)
