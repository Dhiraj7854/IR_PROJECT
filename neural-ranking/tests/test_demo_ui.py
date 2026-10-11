"""Tests for the model-free parts of the Gradio demo UI."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import demo


def test_benchmark_renders_native_charts_from_result_files(tmp_path, monkeypatch):
    (tmp_path / "comparison_table.csv").write_text(
        "system,MRR@10,NDCG@10,P@5,Recall@100,median_latency_ms,latency_queries,flag\n"
        "BM25,0.33,0.42,0.12,0.91,153.4,200,\n"
        "Bi-encoder,0.49,0.58,0.15,0.98,7.4,200,\n",
        encoding="utf-8",
    )
    (tmp_path / "ablations.md").write_text(
        "## 1. Negative sampling\n\n"
        "| Negative strategy | MRR@10 | Difference vs hard |\n"
        "|---|---:|---:|\n| Hard | 0.49 | +0.0000 |\n| Random | 0.52 | +0.0300 |\n\n"
        "## 2. Candidate depth\n\n"
        "| BM25 candidate depth | MRR@10 | Median BM25 + re-ranking latency (ms/query) |\n"
        "|---:|---:|---:|\n| 50 | 0.44 | 203.8 |\n| 100 | 0.41 | 265.5 |\n\n"
        "## 3. Query length\n\n"
        "| Query bucket | Queries | Word-count range | BM25 MRR@10 | Cross-encoder MRR@10 |\n"
        "|---|---:|---|---:|---:|\n| short | 250 | 2–5 | 0.36 | 0.43 |\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(demo, "RESULTS_DIR", tmp_path)

    rendered = demo.render_benchmark()

    assert "Quality vs. speed" in rendered
    assert "Negative sampling" in rendered
    assert "Candidate-depth trade-off" in rendered
    assert "Query-length analysis" in rendered
    assert "median latency" in rendered
    assert rendered.count("<svg") >= 4
    assert "<img" not in rendered
    assert "BM25" in rendered
    assert "Bi-encoder" in rendered


def test_search_insights_explain_rank_overlap():
    rendered = demo.render_insights(
        query="machine learning",
        top_k=3,
        depth=10,
        bm25_hits=[(1, 3.0), (2, 2.0), (3, 1.0)],
        cross_hits=[(2, 0.9), (4, 0.8), (3, 0.7)],
        bm25_rank={1: 1, 2: 2, 3: 3},
        timings={"BM25": 12.0, "BM25 → Cross-encoder": 45.0},
    )

    assert "2/3" in rendered
    assert "Result-set overlap" in rendered
    assert "2 shared passages" in rendered
    assert "2 unique to one ranking" in rendered
