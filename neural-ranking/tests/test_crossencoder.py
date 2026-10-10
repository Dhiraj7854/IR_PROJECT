"""CPU-only tests for cross-encoder candidate re-ranking."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.crossencoder import rerank


class ToyTokenizer:
    def __call__(
        self,
        queries,
        passages,
        truncation=True,
        max_length=256,
        padding=True,
        return_tensors="pt",
    ):
        del queries, truncation, max_length, padding, return_tensors
        relevance = {
            "irrelevant": 0.1,
            "somewhat relevant": 0.5,
            "machine learning explanation": 0.9,
        }
        return {
            "input_ids": torch.tensor(
                [[relevance[passage]] for passage in passages],
                dtype=torch.float32,
            )
        }


class ToyRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.max_length = 256

    def forward(self, input_ids):
        return input_ids[:, 0] * self.weight


def test_rerank_orders_candidates_by_descending_score_and_pid():
    ranked = rerank(
        "what is machine learning",
        candidate_pids=[100, 200, 300],
        candidate_texts=[
            "irrelevant",
            "machine learning explanation",
            "somewhat relevant",
        ],
        k=3,
        model=ToyRanker(),
        tokenizer=ToyTokenizer(),
        fp16=False,
    )

    assert [pid for pid, _ in ranked] == [200, 300, 100]
    assert [score for _, score in ranked] == sorted(
        (score for _, score in ranked),
        reverse=True,
    )
