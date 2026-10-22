"""Tests for the command-line Recall@100 completeness guard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import main


def test_cli_rejects_truncated_recall_100_and_accepts_full_depth(
    tmp_path, monkeypatch
):
    qrels = tmp_path / "qrels.txt"
    qrels.write_text("q 0 p1 1\n", encoding="utf-8")
    run = tmp_path / "run.txt"
    run.write_text(
        "".join(f"q Q0 p{index} {index} {101-index} system\n" for index in range(1, 11)),
        encoding="utf-8",
    )
    output = tmp_path / "metrics.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--run-file",
            str(run),
            "--qrels-file",
            str(qrels),
            "--output-file",
            str(output),
        ],
    )

    with pytest.raises(ValueError, match="Cannot report Recall@100"):
        main()

    run.write_text(
        "".join(f"q Q0 p{index} {index} {101-index} system\n" for index in range(1, 101)),
        encoding="utf-8",
    )
    main()
    result = json.loads(output.read_text(encoding="utf-8"))

    assert result["average"]["Recall@100"] == pytest.approx(1.0)
