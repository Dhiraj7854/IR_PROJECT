"""Unit tests for BEIR result normalization that do not need BEIR installed."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_beir import _qrels_dict


def test_beir_qrels_normalization_handles_nested_mapping_and_dataframe():
    assert _qrels_dict({"q1": {"doc-a": 2}}) == {"q1": {"doc-a": 2}}

    frame = pd.DataFrame(
        {
            "query-id": ["q1", "q1", "q2"],
            "corpus-id": ["doc-a", "doc-b", "doc-c"],
            "score": [1, 0, None],
        }
    )
    assert _qrels_dict(frame) == {
        "q1": {"doc-a": 1, "doc-b": 0},
        "q2": {},
    }


def test_beir_qrels_rejects_unknown_representation():
    with pytest.raises(TypeError, match="Unsupported"):
        _qrels_dict(["invalid"])
