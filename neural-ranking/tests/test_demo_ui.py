"""Tests for the model-free parts of the Gradio demo UI."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import demo


def test_embedding_cache_uses_persistent_space_volume(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACE_ID", "DhirajD/neural-passage-search")
    expected = tmp_path / "corpus_embeddings.npy"

    assert demo._embedding_cache_path(tmp_path / "biencoder", tmp_path) == expected


def test_neural_inference_bypasses_unavailable_zerogpu_on_space(monkeypatch):
    calls = []
    expected = {"dense": ([(17, 0.8)], 1.0)}

    def cpu_inference(*args):
        calls.append(args)
        return expected

    def unexpected_zerogpu_call(*args):
        raise AssertionError("ZeroGPU should be bypassed by default on a Space")

    monkeypatch.setattr(demo, "_NEURAL_INFERENCE", cpu_inference)
    monkeypatch.setattr(demo, "_run_neural_inference", unexpected_zerogpu_call)
    monkeypatch.setenv("SPACE_ID", "DhirajD/neural-passage-search")
    monkeypatch.delenv("IR_USE_ZEROGPU", raising=False)

    result = demo._run_neural_inference_with_fallback(
        "example query", 10, [], [], True, False
    )

    assert result == expected
    assert calls == [
        ("example query", 10, [], [], True, False, True)
    ]


def test_search_callback_runs_hybrid_retrieval_and_reranking():
    backend = demo.Backend(
        passages={1: "photosynthesis in green plants", 2: "sunlight supports plant growth"},
        bm25=lambda query, k: [(1, 2.0), (2, 1.0)][:k],
        tfidf=lambda query, k: [(1, 2.0), (2, 1.0)][:k],
        dense=lambda query, k: [(2, 0.9), (1, 0.7)][:k],
        rerank=lambda query, pids, k: [(pid, 0.8) for pid in pids[:k]],
    )
    app, _ = demo.build_app(backend)
    search_index = next(
        index
        for index, dependency in enumerate(app.config["dependencies"])
        if dependency.get("api_name") == "search"
    )
    search = app.fns[search_index].fn

    _, rrf_results = search("photosynthesis", ["Hybrid RRF"], 2, 2, True)
    _, reranked_results = search(
        "photosynthesis",
        ["Hybrid RRF → Cross-encoder"],
        2,
        2,
        True,
    )

    assert "Hybrid RRF" in rrf_results
    assert "Hybrid RRF → Cross-encoder" in reranked_results


def test_empty_benchmark_explains_missing_space_specific_qrels(tmp_path, monkeypatch):
    monkeypatch.setattr(demo, "RESULTS_DIR", tmp_path)

    rendered = demo.render_benchmark()

    assert "Benchmark results unavailable for this corpus" in rendered
    assert "matching qrels" in rendered
    assert "IR_RESULTS_DIR" in rendered
    assert "No fallback or fabricated metrics" in rendered


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
    assert "Local MS MARCO evaluation" in rendered
    assert "do not describe a separate Hugging Face Space" in rendered


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
