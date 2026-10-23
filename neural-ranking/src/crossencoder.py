"""PyTorch cross-encoder scorer for query-passage relevance."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, List, Mapping, Optional, Protocol, Sequence, Tuple, Union

import torch
from torch import nn

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CROSSENCODER_PATH = PROJECT_ROOT / "artifacts" / "crossencoder"
DEFAULT_MAX_LENGTH = 256
DEFAULT_CROSSENCODER_NAME = "cross-encoder/nli-MiniLM2-L6-H768"


class _SequenceClassifierConfig(Protocol):
    num_labels: int
    id2label: Mapping[int, str]


class _SequenceClassifierOutput(Protocol):
    logits: torch.Tensor


class _SequenceClassifier(Protocol):
    config: _SequenceClassifierConfig

    def __call__(self, **inputs: Any) -> _SequenceClassifierOutput: ...

    def save_pretrained(self, save_directory: Union[str, Path]) -> Any: ...


class _BackboneConfig(Protocol):
    hidden_size: int


class _BackboneOutput(Protocol):
    last_hidden_state: torch.Tensor


class _Backbone(Protocol):
    config: _BackboneConfig

    def __call__(self, **inputs: Any) -> _BackboneOutput: ...

    def save_pretrained(self, save_directory: Union[str, Path]) -> Any: ...


class CrossEncoderRanker(nn.Module):
    """Sequence classifier adapted to a scalar query-passage relevance score."""

    def __init__(
        self,
        model_name_or_path: Union[str, Path] = DEFAULT_CROSSENCODER_NAME,
        dropout: float = 0.1,
        max_length: int = DEFAULT_MAX_LENGTH,
        backbone: Optional[_Backbone] = None,
        sequence_classifier: Optional[_SequenceClassifier] = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")
        if backbone is not None and sequence_classifier is not None:
            raise ValueError("Pass either a legacy backbone or sequence classifier, not both")
        if backbone is None and sequence_classifier is None:
            from transformers import AutoModelForSequenceClassification

            sequence_classifier = AutoModelForSequenceClassification.from_pretrained(
                str(model_name_or_path)
            )
        self.sequence_classifier = sequence_classifier
        self.backbone = backbone
        self.scoring_head: Optional[nn.Linear] = None
        self.score_mode = "legacy_cls"
        self.entailment_label_id: Optional[int] = None
        self.contradiction_label_id: Optional[int] = None
        if self.sequence_classifier is not None:
            labels = {
                int(label_id): str(label).strip().lower()
                for label_id, label in self.sequence_classifier.config.id2label.items()
            }
            num_labels = int(self.sequence_classifier.config.num_labels)
            if num_labels == 1:
                self.score_mode = "single_logit"
            elif num_labels == 3:
                label_ids = {label: label_id for label_id, label in labels.items()}
                if "entailment" not in label_ids or "contradiction" not in label_ids:
                    raise ValueError(
                        "Three-label cross-encoder initialization requires "
                        "entailment and contradiction labels"
                    )
                self.score_mode = "nli_margin"
                self.entailment_label_id = label_ids["entailment"]
                self.contradiction_label_id = label_ids["contradiction"]
            else:
                raise ValueError(
                    f"Unsupported pretrained sequence-classification head "
                    f"with {num_labels} labels"
                )
        elif self.backbone is None:
            raise RuntimeError("Cross-encoder backbone was not initialized")
        else:
            hidden_size = int(self.backbone.config.hidden_size)
            self.dropout = nn.Dropout(dropout)
            self.scoring_head = nn.Linear(hidden_size, 1)
        self.max_length = max_length

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        """Return one relevance logit for each encoded query-passage pair."""
        if self.sequence_classifier is not None:
            logits = self.sequence_classifier(**inputs, return_dict=True).logits
            if self.score_mode == "single_logit":
                return logits[:, 0]
            if self.score_mode == "nli_margin":
                if (
                    self.entailment_label_id is None
                    or self.contradiction_label_id is None
                ):
                    raise RuntimeError("NLI relevance labels were not initialized")
                return (
                    logits[:, self.entailment_label_id]
                    - logits[:, self.contradiction_label_id]
                )
            raise RuntimeError(f"Unsupported sequence-classification score mode: {self.score_mode}")
        if self.backbone is None or self.scoring_head is None:
            raise RuntimeError("Legacy cross-encoder components were not initialized")
        outputs = self.backbone(**inputs, return_dict=True)
        cls_embedding = outputs.last_hidden_state[:, 0]
        return self.scoring_head(self.dropout(cls_embedding)).squeeze(-1)

    def save_pretrained(
        self,
        output_dir: Union[str, Path],
        tokenizer: Any,
        dropout: float = 0.1,
    ) -> None:
        """Save backbone, tokenizer, head, and inference settings."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if self.sequence_classifier is not None:
            self.sequence_classifier.save_pretrained(output_path / "backbone")
        elif self.backbone is not None and self.scoring_head is not None:
            self.backbone.save_pretrained(output_path / "backbone")
            torch.save(self.scoring_head.state_dict(), output_path / "scoring_head.pt")
        else:
            raise RuntimeError("Cross-encoder components were not initialized")
        tokenizer.save_pretrained(output_path)
        metadata = {
            "dropout": dropout,
            "max_length": self.max_length,
            "architecture": (
                "sequence-classification"
                if self.sequence_classifier is not None
                else "legacy-cls"
            ),
            "score_mode": self.score_mode,
        }
        (output_path / "crossencoder_config.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> "CrossEncoderRanker":
        """Restore a checkpoint created by :meth:`save_pretrained`."""
        model_path = Path(model_path)
        metadata_path = model_path / "crossencoder_config.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Cross-encoder metadata not found: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("architecture") == "sequence-classification":
            from transformers import AutoModelForSequenceClassification

            classifier = AutoModelForSequenceClassification.from_pretrained(
                model_path / "backbone"
            )
            model = cls(
                dropout=float(metadata["dropout"]),
                max_length=int(metadata["max_length"]),
                sequence_classifier=classifier,
            )
            if model.score_mode != metadata.get("score_mode"):
                raise ValueError(
                    "Saved cross-encoder score mode does not match its classifier labels"
                )
            return model

        from transformers import AutoModel

        model = cls(
            dropout=float(metadata["dropout"]),
            max_length=int(metadata["max_length"]),
            backbone=AutoModel.from_pretrained(model_path / "backbone"),
        )
        state = torch.load(
            model_path / "scoring_head.pt",
            map_location=map_location or "cpu",
            weights_only=True,
        )
        if model.scoring_head is None:
            raise RuntimeError("Legacy cross-encoder scoring head was not initialized")
        model.scoring_head.load_state_dict(state)
        return model


_RANKER: Optional[CrossEncoderRanker] = None
_TOKENIZER: Optional[Any] = None


def load_crossencoder(
    model_path: Union[str, Path] = DEFAULT_CROSSENCODER_PATH,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[CrossEncoderRanker, Any]:
    """Load the saved ranker checkpoint and its paired tokenizer."""
    global _RANKER, _TOKENIZER
    model_path = Path(model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Cross-encoder checkpoint not found: {model_path}")
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _RANKER = CrossEncoderRanker.from_pretrained(model_path)
    _RANKER.to(target_device)
    _RANKER.eval()
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(str(model_path))
    return _RANKER, _TOKENIZER


def rerank(
    query: str,
    candidate_pids: Sequence[int],
    candidate_texts: Sequence[str],
    k: Optional[int] = None,
    *,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    batch_size: int = 32,
    max_length: Optional[int] = None,
    fp16: bool = True,
) -> List[Tuple[int, float]]:
    """Re-score candidate passages and return (pid, score) in descending order.

    Inference is performed in batches under ``torch.no_grad``. FP16 autocast is
    enabled when requested and a CUDA device is available.
    """
    if len(candidate_pids) != len(candidate_texts):
        raise ValueError("candidate_pids and candidate_texts must have equal lengths")
    if len(set(candidate_pids)) != len(candidate_pids):
        raise ValueError("candidate_pids must be unique")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if k is not None and k < 0:
        raise ValueError("k must be non-negative")
    if not candidate_pids or k == 0:
        return []

    ranker = model or _RANKER
    passage_tokenizer = tokenizer or _TOKENIZER
    if ranker is None or passage_tokenizer is None:
        raise RuntimeError("Load a cross-encoder or pass model and tokenizer to rerank()")
    device = next(ranker.parameters()).device
    use_fp16 = fp16 and device.type == "cuda"
    limit = max_length or getattr(ranker, "max_length", DEFAULT_MAX_LENGTH)

    scored: List[Tuple[int, float]] = []
    ranker.eval()
    with torch.no_grad():
        for start in range(0, len(candidate_pids), batch_size):
            batch_texts = candidate_texts[start : start + batch_size]
            encoded = passage_tokenizer(
                [query] * len(batch_texts),
                list(batch_texts),
                truncation="only_second",
                max_length=limit,
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in encoded.items()}
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_fp16,
            ):
                logits = ranker(**inputs)
            scores = logits.float().reshape(-1).cpu().tolist()
            scored.extend(
                (int(pid), float(score))
                for pid, score in zip(
                    candidate_pids[start : start + len(batch_texts)],
                    scores,
                )
            )

    scored.sort(key=lambda result: (-result[1], result[0]))
    return scored if k is None else scored[:k]


def measure_rerank_latency_ms(
    query: str,
    candidate_pids: Sequence[int],
    candidate_texts: Sequence[str],
    *,
    warmup: int = 1,
    repeats: int = 3,
    **rerank_kwargs: Any,
) -> float:
    """Measure mean end-to-end reranking latency in milliseconds."""
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    for _ in range(warmup):
        rerank(
            query,
            candidate_pids,
            candidate_texts,
            **rerank_kwargs,
        )
    latencies: List[float] = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        rerank(
            query,
            candidate_pids,
            candidate_texts,
            **rerank_kwargs,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - started) * 1000.0)
    return sum(latencies) / len(latencies)
