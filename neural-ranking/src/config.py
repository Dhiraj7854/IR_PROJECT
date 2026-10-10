"""Configuration loader and typed schema definitions for neural ranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union
import os
import re

try:
    import yaml
except ImportError:
    yaml = None


def _fallback_yaml_parse(text: str) -> Dict[str, Any]:
    """Lightweight fallback parser for basic nested YAML structures."""
    lines = text.splitlines()
    root: Dict[str, Any] = {}
    stack: list[tuple[int, Dict[str, Any]]] = [(-1, root)]

    for raw_line in lines:
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        if ":" not in stripped:
            continue

        key, val = stripped.split(":", 1)
        key = key.strip()
        val = val.strip()

        # Unwind stack to find current parent
        while len(stack) > 1 and stack[-1][0] >= indent:
            stack.pop()
        current_dict = stack[-1][1]

        if val == "":
            new_dict: Dict[str, Any] = {}
            current_dict[key] = new_dict
            stack.append((indent, new_dict))
        else:
            # Parse value type (int, float, bool, string)
            if val.lower() == "true":
                parsed_val: Any = True
            elif val.lower() == "false":
                parsed_val = False
            elif val.startswith('"') and val.endswith('"'):
                parsed_val = val[1:-1]
            elif val.startswith("'") and val.endswith("'"):
                parsed_val = val[1:-1]
            else:
                try:
                    if "." in val or "e" in val.lower():
                        parsed_val = float(val)
                    else:
                        parsed_val = int(val)
                except ValueError:
                    parsed_val = val
            current_dict[key] = parsed_val

    return root


@dataclass
class QuerySplitsConfig:
    """Query split counts for train, validation, and evaluation."""

    train: int = 3500
    val: int = 750
    test: int = 750


@dataclass
class DatasetConfig:
    """Dataset configuration parameters."""

    name: str = "msmarco-passage"
    corpus_subset_size: int = 100000
    query_splits: QuerySplitsConfig = field(default_factory=QuerySplitsConfig)


@dataclass
class ModelsConfig:
    """Pre-trained neural model names and sequence length settings."""

    biencoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    crossencoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_seq_len: int = 256


@dataclass
class TrainingConfig:
    """Model training hyperparameters."""

    batch_size: int = 32
    lr: float = 0.00002
    epochs: int = 3


@dataclass
class RetrievalConfig:
    """Top-k retrieval and re-ranking depth parameters."""

    stage1_top_k: int = 100
    rerank_depth: int = 50

    @property
    def top_k_stage1(self) -> int:
        """Alias for stage1_top_k."""
        return self.stage1_top_k


@dataclass
class Config:
    """Master typed configuration schema for two-stage neural ranking."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    # Top-level convenience properties
    @property
    def dataset_name(self) -> str:
        """Name of the dataset."""
        return self.dataset.name

    @property
    def corpus_subset_size(self) -> int:
        """Subsampled corpus size."""
        return self.dataset.corpus_subset_size

    @property
    def query_splits(self) -> QuerySplitsConfig:
        """Query split sizes."""
        return self.dataset.query_splits

    @property
    def biencoder_name(self) -> str:
        """Bi-encoder checkpoint name."""
        return self.models.biencoder_name

    @property
    def crossencoder_name(self) -> str:
        """Cross-encoder checkpoint name."""
        return self.models.crossencoder_name

    @property
    def max_seq_len(self) -> int:
        """Maximum sequence length."""
        return self.models.max_seq_len

    @property
    def batch_size(self) -> int:
        """Batch size for training and inference."""
        return self.training.batch_size

    @property
    def lr(self) -> float:
        """Learning rate."""
        return self.training.lr

    @property
    def epochs(self) -> int:
        """Total training epochs."""
        return self.training.epochs

    @property
    def stage1_top_k(self) -> int:
        """Stage 1 candidate pool size."""
        return self.retrieval.stage1_top_k

    @property
    def top_k_stage1(self) -> int:
        """Alias for stage 1 candidate pool size."""
        return self.retrieval.stage1_top_k

    @property
    def rerank_depth(self) -> int:
        """Re-ranking candidate depth."""
        return self.retrieval.rerank_depth


def load_config(path: Union[str, Path] = "configs/config.yaml") -> Config:
    """Load configuration from a YAML file into a typed Config dataclass.

    Args:
        path: Filepath to YAML configuration. Defaults to 'configs/config.yaml'.

    Returns:
        Config: Populated typed configuration dataclass.

    Raises:
        FileNotFoundError: If the configuration file cannot be found.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        # Fallback: check relative to project root
        project_root = Path(__file__).resolve().parent.parent
        candidate = project_root / path
        if candidate.exists():
            path_obj = candidate
        else:
            raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path_obj, "r", encoding="utf-8") as f:
        content = f.read()

    if yaml is not None:
        raw_cfg = yaml.safe_load(content) or {}
    else:
        raw_cfg = _fallback_yaml_parse(content)

    # Parse dataset section
    dataset_raw = raw_cfg.get("dataset", {})
    splits_raw = dataset_raw.get("query_splits", {})
    query_splits = QuerySplitsConfig(
        train=int(splits_raw.get("train", 3500)),
        val=int(splits_raw.get("val", 750)),
        test=int(splits_raw.get("test", 750)),
    )
    dataset = DatasetConfig(
        name=str(dataset_raw.get("name", "msmarco-passage")),
        corpus_subset_size=int(dataset_raw.get("corpus_subset_size", 100000)),
        query_splits=query_splits,
    )

    # Parse models section
    models_raw = raw_cfg.get("models", {})
    models = ModelsConfig(
        biencoder_name=str(
            models_raw.get("biencoder_name", "sentence-transformers/all-MiniLM-L6-v2")
        ),
        crossencoder_name=str(
            models_raw.get(
                "crossencoder_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
        ),
        max_seq_len=int(models_raw.get("max_seq_len", 256)),
    )

    # Parse training section
    training_raw = raw_cfg.get("training", {})
    training = TrainingConfig(
        batch_size=int(training_raw.get("batch_size", 32)),
        lr=float(training_raw.get("lr", 0.00002)),
        epochs=int(training_raw.get("epochs", 3)),
    )

    # Parse retrieval section
    retrieval_raw = raw_cfg.get("retrieval", {})
    retrieval = RetrievalConfig(
        stage1_top_k=int(
            retrieval_raw.get(
                "stage1_top_k", retrieval_raw.get("top_k_stage1", 100)
            )
        ),
        rerank_depth=int(retrieval_raw.get("rerank_depth", 50)),
    )

    return Config(
        dataset=dataset,
        models=models,
        training=training,
        retrieval=retrieval,
    )

