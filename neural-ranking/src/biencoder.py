"""Sentence-transformer bi-encoder training helpers and dense retrieval."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "biencoder"
DEFAULT_EMBEDDINGS_PATH = DEFAULT_MODEL_PATH / "corpus_embeddings.npy"

_MODEL: Optional[Any] = None
_MODEL_CACHE_KEY: Optional[str] = None
_CORPUS_EMBEDDINGS: Optional[torch.Tensor] = None
_CORPUS_PIDS: Optional[np.ndarray] = None


def load_biencoder(
    model_name_or_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    device: Optional[str] = None,
) -> Any:
    """Load a trained checkpoint, falling back to the MiniLM base model."""
    global _MODEL, _MODEL_CACHE_KEY
    from sentence_transformers import SentenceTransformer

    model_path = Path(model_name_or_path)
    if model_path.exists():
        model_source = str(model_path.resolve())
    elif str(model_name_or_path) == str(DEFAULT_MODEL_PATH):
        model_source = DEFAULT_MODEL_NAME
    else:
        model_source = str(model_name_or_path)

    _MODEL = SentenceTransformer(model_source, device=device)
    _MODEL_CACHE_KEY = model_source
    setattr(_MODEL, "_biencoder_cache_key", model_source)
    return _MODEL


def _get_model(model: Optional[Any] = None) -> Any:
    if model is not None:
        return model
    if _MODEL is None:
        return load_biencoder()
    return _MODEL


def _model_cache_key(model: Any) -> str:
    model_key = getattr(model, "_biencoder_cache_key", None)
    if model_key is None:
        model_key = f"{type(model).__module__}.{type(model).__qualname__}"
    path = Path(str(model_key))
    if path.is_dir():
        digest = hashlib.sha256()
        ignored = {
            "corpus_embeddings.npy",
            "corpus_embeddings.json",
            "training_metrics.json",
        }
        for file_path in sorted(
            p for p in path.rglob("*") if p.is_file() and p.name not in ignored
        ):
            stat = file_path.stat()
            digest.update(str(file_path.relative_to(path)).encode("utf-8"))
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
        return f"{model_key}:{digest.hexdigest()}"
    return str(model_key)


def _texts_fingerprint(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = str(text).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def _encode(
    model: Any,
    texts: Sequence[str],
    batch_size: int,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=len(texts) > batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    result = np.asarray(embeddings, dtype=np.float32)
    if result.ndim != 2 or result.shape[0] != len(texts):
        raise ValueError("The encoder returned an unexpected embedding shape")
    if not np.isfinite(result).all():
        raise ValueError("The encoder returned non-finite embeddings")
    return result


def encode_corpus(
    texts: Sequence[str],
    batch_size: int = 64,
    model: Optional[Any] = None,
    cache_path: Union[str, Path] = DEFAULT_EMBEDDINGS_PATH,
) -> np.ndarray:
    """Encode corpus passages once and cache normalized vectors as an .npy file.

    A sidecar stores corpus and model fingerprints; a cache is reused only when
    both match the current inputs.
    """
    encoder = _get_model(model)
    corpus_texts = [str(text) for text in texts]
    if not corpus_texts:
        raise ValueError("Cannot encode an empty corpus")
    if any(not text.strip() for text in corpus_texts):
        raise ValueError("Corpus texts must not be empty")

    cache_path = Path(cache_path)
    metadata_path = cache_path.with_suffix(".json")
    metadata = {
        "corpus_fingerprint": _texts_fingerprint(corpus_texts),
        "model_fingerprint": _model_cache_key(encoder),
        "count": len(corpus_texts),
    }
    if cache_path.exists() and metadata_path.exists():
        cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if cached_metadata == metadata:
            embeddings = np.load(cache_path, mmap_mode=None)
            if embeddings.ndim == 2 and embeddings.shape[0] == len(corpus_texts):
                logger.info("Loaded corpus embeddings from %s", cache_path)
                return embeddings

    logger.info("Encoding %d corpus passages", len(corpus_texts))
    embeddings = _encode(encoder, corpus_texts, batch_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as cache_file:
        np.save(cache_file, embeddings)
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Cached corpus embeddings at %s", cache_path)
    return embeddings


def configure_retrieval(
    corpus_texts: Sequence[str],
    pids: Optional[Sequence[int]] = None,
    batch_size: int = 64,
    model: Optional[Any] = None,
    cache_path: Union[str, Path] = DEFAULT_EMBEDDINGS_PATH,
) -> np.ndarray:
    """Initialize the module-level model and normalized corpus embedding index."""
    global _MODEL, _MODEL_CACHE_KEY, _CORPUS_EMBEDDINGS, _CORPUS_PIDS
    encoder = _get_model(model)
    if pids is not None and len(pids) != len(corpus_texts):
        raise ValueError("Number of passage IDs must match the number of corpus texts")
    if pids is not None and len(np.unique(pids)) != len(pids):
        raise ValueError("Corpus passage IDs must be unique")
    if model is not None:
        _MODEL = model
        _MODEL_CACHE_KEY = getattr(
            model,
            "_biencoder_cache_key",
            f"{type(model).__module__}.{type(model).__qualname__}",
        )

    corpus_embeddings = encode_corpus(
        corpus_texts,
        batch_size=batch_size,
        model=encoder,
        cache_path=cache_path,
    )
    if pids is None:
        corpus_pids = np.arange(len(corpus_embeddings), dtype=np.int64)
    else:
        corpus_pids = np.asarray(pids)
    device = getattr(encoder, "device", torch.device("cpu"))
    _CORPUS_EMBEDDINGS = torch.as_tensor(
        corpus_embeddings,
        dtype=torch.float32,
        device=device,
    )
    _CORPUS_PIDS = corpus_pids
    return corpus_embeddings


def retrieve_biencoder(query: str, k: int = 100) -> List[Tuple[int, float]]:
    """Return top-k unique passage IDs by cosine similarity."""
    if k < 0:
        raise ValueError("k must be non-negative")
    if _CORPUS_EMBEDDINGS is None or _CORPUS_PIDS is None:
        raise RuntimeError("Call configure_retrieval() before retrieval")
    if k == 0:
        return []

    encoder = _get_model()
    query_embedding = _encode(encoder, [query], batch_size=1)
    query_tensor = torch.as_tensor(
        query_embedding[0],
        dtype=_CORPUS_EMBEDDINGS.dtype,
        device=_CORPUS_EMBEDDINGS.device,
    )
    scores = torch.matmul(_CORPUS_EMBEDDINGS, query_tensor)
    count = min(k, scores.numel())
    top_scores, top_indices = torch.topk(scores, k=count, largest=True, sorted=True)
    return [
        (int(_CORPUS_PIDS[index]), float(score))
        for index, score in zip(top_indices.cpu().tolist(), top_scores.cpu().tolist())
    ]


def retrieve_biencoder_batch(
    queries: Sequence[str],
    k: int = 100,
    batch_size: int = 64,
) -> List[List[Tuple[int, float]]]:
    """Batch-encode queries and score them against the indexed corpus."""
    if k < 0:
        raise ValueError("k must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if _CORPUS_EMBEDDINGS is None or _CORPUS_PIDS is None:
        raise RuntimeError("Call configure_retrieval() before retrieval")
    if k == 0:
        return [[] for _ in queries]

    encoder = _get_model()
    all_results: List[List[Tuple[int, float]]] = []
    count = min(k, len(_CORPUS_PIDS))
    for start in range(0, len(queries), batch_size):
        query_embeddings = _encode(
            encoder,
            queries[start : start + batch_size],
            batch_size=batch_size,
        )
        query_tensor = torch.as_tensor(
            query_embeddings,
            dtype=_CORPUS_EMBEDDINGS.dtype,
            device=_CORPUS_EMBEDDINGS.device,
        )
        scores = torch.matmul(query_tensor, _CORPUS_EMBEDDINGS.T)
        top_scores, top_indices = torch.topk(
            scores,
            k=count,
            dim=1,
            largest=True,
            sorted=True,
        )
        for row_scores, row_indices in zip(top_scores.cpu(), top_indices.cpu()):
            all_results.append(
                [
                    (int(_CORPUS_PIDS[index]), float(score))
                    for index, score in zip(row_indices.tolist(), row_scores.tolist())
                ]
            )
    return all_results


def mean_reciprocal_rank(
    results: Sequence[Sequence[Tuple[int, float]]],
    query_ids: Sequence[int],
    qrels: Any,
    k: int = 10,
) -> float:
    """Compute MRR@k from ranked results and a qrels DataFrame."""
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if len(results) != len(query_ids):
        raise ValueError("Results and query IDs must have the same length")

    relevant_by_qid: Dict[int, set[int]] = {}
    for row in qrels.itertuples(index=False):
        qid = int(getattr(row, "query_id"))
        pid = int(getattr(row, "relevant_pid"))
        relevant_by_qid.setdefault(qid, set()).add(pid)

    reciprocal_ranks: List[float] = []
    for qid_value, hits in zip(query_ids, results):
        relevant = relevant_by_qid.get(int(qid_value), set())
        reciprocal_rank = 0.0
        for rank, (pid, _) in enumerate(hits[:k], start=1):
            if int(pid) in relevant:
                reciprocal_rank = 1.0 / rank
                break
        reciprocal_ranks.append(reciprocal_rank)
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
