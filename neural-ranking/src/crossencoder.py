"""PyTorch cross-encoder scorer for query-passage relevance."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CROSSENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_CROSSENCODER_PATH = PROJECT_ROOT / "artifacts" / "crossencoder"
DEFAULT_MAX_LENGTH = 256


class CrossEncoderRanker(nn.Module):
    """AutoModel backbone with dropout and a scalar [CLS] scoring head."""

    def __init__(
        self,
        model_name_or_path: Union[str, Path] = DEFAULT_CROSSENCODER_NAME,
        dropout: float = 0.1,
        max_length: int = DEFAULT_MAX_LENGTH,
        backbone: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")
        self.backbone = backbone or AutoModel.from_pretrained(str(model_name_or_path))
        hidden_size = int(self.backbone.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.scoring_head = nn.Linear(hidden_size, 1)
        self.max_length = max_length

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        """Return one relevance logit for each encoded query-passage pair."""
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
        self.backbone.save_pretrained(output_path / "backbone")
        tokenizer.save_pretrained(output_path)
        torch.save(self.scoring_head.state_dict(), output_path / "scoring_head.pt")
        metadata = {
            "dropout": dropout,
            "max_length": self.max_length,
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
        model = cls(
            model_path / "backbone",
            dropout=float(metadata["dropout"]),
            max_length=int(metadata["max_length"]),
        )
        state = torch.load(
            model_path / "scoring_head.pt",
            map_location=map_location or "cpu",
            weights_only=True,
        )
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
                truncation=True,
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
