"""CPU-only tests for cross-encoder candidate re-ranking."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.crossencoder import CrossEncoderRanker, rerank
from scripts.train_crossencoder import make_collate_fn


class ToyTokenizer:
    def __init__(self):
        self.truncation_modes = []

    def __call__(
        self,
        queries,
        passages,
        truncation=True,
        max_length=256,
        padding=True,
        return_tensors="pt",
    ):
        del queries, max_length, padding, return_tensors
        self.truncation_modes.append(truncation)
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
    tokenizer = ToyTokenizer()
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
        tokenizer=tokenizer,
        fp16=False,
    )

    assert [pid for pid, _ in ranked] == [200, 300, 100]
    assert tokenizer.truncation_modes == ["only_second"]
    assert [score for _, score in ranked] == sorted(
        (score for _, score in ranked),
        reverse=True,
    )


def test_training_pair_tokenization_truncates_passage_only():
    tokenizer = ToyTokenizer()
    collate = make_collate_fn(tokenizer, max_length=128)

    encoded, labels = collate([("a long query", "irrelevant", 1.0)])

    assert tokenizer.truncation_modes == ["only_second"]
    assert labels.tolist() == [1.0]
    assert "input_ids" in encoded


class ToySequenceClassifier(nn.Module):
    def __init__(self, logits, labels):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(logits, dtype=torch.float32))
        self.config = SimpleNamespace(
            num_labels=len(logits),
            id2label={index: label for index, label in enumerate(labels)},
        )

    def forward(self, input_ids, return_dict=True):
        del return_dict
        return SimpleNamespace(
            logits=self.logits.expand(input_ids.shape[0], -1)
        )

    def save_pretrained(self, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)


def test_nli_classifier_uses_entailment_minus_contradiction():
    classifier = ToySequenceClassifier(
        [1.0, -0.5, 3.25],
        ["contradiction", "neutral", "entailment"],
    )
    ranker = CrossEncoderRanker(sequence_classifier=classifier)

    scores = ranker(input_ids=torch.zeros((2, 1), dtype=torch.long))

    assert ranker.score_mode == "nli_margin"
    assert scores.tolist() == pytest.approx([2.25, 2.25])


def test_single_logit_classifier_preserves_its_relevance_score():
    classifier = ToySequenceClassifier([1.75], ["relevance"])
    ranker = CrossEncoderRanker(sequence_classifier=classifier)

    scores = ranker(input_ids=torch.zeros((1, 1), dtype=torch.long))

    assert ranker.score_mode == "single_logit"
    assert scores.tolist() == pytest.approx([1.75])


def test_nli_checkpoint_round_trips_score_mode_without_network_access(
    tmp_path,
    monkeypatch,
):
    transformers = pytest.importorskip("transformers")
    classifier = ToySequenceClassifier(
        [0.25, -0.5, 1.0],
        ["contradiction", "neutral", "entailment"],
    )
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: classifier),
    )
    model = CrossEncoderRanker(sequence_classifier=classifier)
    model.save_pretrained(tmp_path, tokenizer=ToySaveableTokenizer())

    restored = CrossEncoderRanker.from_pretrained(tmp_path)
    scores = restored(input_ids=torch.zeros((1, 1), dtype=torch.long))

    assert restored.score_mode == "nli_margin"
    assert scores.tolist() == pytest.approx([0.75])


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=2)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, return_dict=True):
        del return_dict
        hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 2)
        return SimpleNamespace(last_hidden_state=hidden * self.scale)

    def save_pretrained(self, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)


class ToySaveableTokenizer:
    def save_pretrained(self, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)


def test_legacy_checkpoint_round_trips_without_network_access(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: ToyBackbone()),
    )
    model = CrossEncoderRanker(backbone=ToyBackbone())
    model.scoring_head.weight.data.fill_(0.75)
    model.scoring_head.bias.data.fill_(0.25)
    model.eval()
    expected = model(input_ids=torch.tensor([[1, 2]], dtype=torch.long))
    model.save_pretrained(tmp_path, tokenizer=ToySaveableTokenizer())

    restored = CrossEncoderRanker.from_pretrained(tmp_path)
    restored.eval()
    actual = restored(input_ids=torch.tensor([[1, 2]], dtype=torch.long))

    assert actual.tolist() == pytest.approx(expected.tolist())
    assert restored.score_mode == "legacy_cls"
