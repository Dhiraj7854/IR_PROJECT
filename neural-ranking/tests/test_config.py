"""Tests for configuration loading and validation."""

import math
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pytest
except ImportError:
    pytest = None

from src.config import (
    Config,
    DatasetConfig,
    ModelsConfig,
    QuerySplitsConfig,
    RetrievalConfig,
    TrainingConfig,
    load_config,
    _coerce_bool,
)


def test_load_config_valid():
    """Test loading valid configuration file from default path."""
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    cfg = load_config(str(config_path))

    assert isinstance(cfg, Config)

    # Dataset checks
    assert isinstance(cfg.dataset, DatasetConfig)
    assert cfg.dataset.name == "msmarco-passage"
    assert cfg.dataset.corpus_subset_size == 100000

    # Query split checks
    assert isinstance(cfg.dataset.query_splits, QuerySplitsConfig)
    assert cfg.dataset.query_splits.train == 3500
    assert cfg.dataset.query_splits.val == 750
    assert cfg.dataset.query_splits.test == 750

    # Models checks
    assert isinstance(cfg.models, ModelsConfig)
    assert cfg.models.biencoder_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.models.crossencoder_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert cfg.models.max_seq_len == 256

    # Training checks
    assert isinstance(cfg.training, TrainingConfig)
    assert cfg.training.batch_size == 64
    assert math.isclose(cfg.training.lr, 2.0e-5, rel_tol=1e-6)
    assert cfg.training.epochs == 2
    assert cfg.training.fp16 is True
    assert math.isclose(cfg.training.warmup_ratio, 0.1)
    assert cfg.training.early_stopping_patience == 1

    # Retrieval top-k checks
    assert isinstance(cfg.retrieval, RetrievalConfig)
    assert cfg.retrieval.stage1_top_k == 100
    assert cfg.retrieval.rerank_depth == 50


def test_config_convenience_properties():
    """Test top-level convenience property access."""
    cfg = load_config("configs/config.yaml")

    assert cfg.dataset_name == "msmarco-passage"
    assert cfg.corpus_subset_size == 100000
    assert cfg.query_splits.train == 3500
    assert cfg.query_splits.val == 750
    assert cfg.query_splits.test == 750
    assert cfg.biencoder_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.crossencoder_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert cfg.max_seq_len == 256
    assert cfg.batch_size == 64
    assert math.isclose(cfg.lr, 2.0e-5, rel_tol=1e-6)
    assert cfg.epochs == 2
    assert cfg.stage1_top_k == 100
    assert cfg.top_k_stage1 == 100
    assert cfg.rerank_depth == 50


def test_boolean_configuration_parsing_handles_false_strings_and_invalid_values():
    assert _coerce_bool("false", default=True) is False
    assert _coerce_bool("yes") is True
    with pytest.raises(ValueError, match="Invalid boolean"):
        _coerce_bool("perhaps")


def test_load_config_file_not_found():
    """Test that FileNotFoundError is raised when config path does not exist."""
    if pytest is not None:
        with pytest.raises(FileNotFoundError):
            load_config("configs/non_existent_config.yaml")
    else:
        try:
            load_config("configs/non_existent_config.yaml")
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    test_load_config_valid()
    test_config_convenience_properties()
    test_load_config_file_not_found()
    print("All tests passed successfully!")
