#!/usr/bin/env python3
"""Interactive BM25 and cross-encoder passage retrieval demo."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, List, Tuple

import gradio as gr
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import BaselineIndex, build_index, get_index
from src.biencoder import configure_retrieval, load_biencoder
from src.config import load_config
from src.crossencoder import CrossEncoderRanker, load_crossencoder, rerank

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CORPUS_PATH = DATA_DIR / "corpus.parquet"
BIENCODER_PATH = ARTIFACTS_DIR / "biencoder"
CROSSENCODER_PATH = ARTIFACTS_DIR / "crossencoder"
EMBEDDINGS_PATH = BIENCODER_PATH / "corpus_embeddings.npy"
CONFIG = load_config(PROJECT_ROOT / "configs" / "config.yaml")
MAX_SEQ_LEN = CONFIG.models.max_seq_len

# Representative examples from the test query split.
EXAMPLE_QUERIES = [
    "where does the water that is produced by cellular respiration go",
    "how do i use sunpass on peach pass roads",
    "who was the first woman in space in india",
    "what does the anterior cingulate cortex do",
]


def _load_models_and_indexes() -> Tuple[
    pd.DataFrame,
    BaselineIndex,
    Any,
    CrossEncoderRanker,
    Any,
]:
    if not CORPUS_PATH.is_file():
        raise FileNotFoundError(f"Corpus parquet not found: {CORPUS_PATH}")
    if not BIENCODER_PATH.is_dir():
        raise FileNotFoundError(
            f"Trained bi-encoder checkpoint not found: {BIENCODER_PATH}"
        )
    if not EMBEDDINGS_PATH.is_file():
        raise FileNotFoundError(
            f"Cached corpus embeddings not found: {EMBEDDINGS_PATH}. "
            "Run scripts/eval_biencoder.py once to create them."
        )
    if not CROSSENCODER_PATH.is_dir():
        raise FileNotFoundError(
            f"Trained cross-encoder checkpoint not found: {CROSSENCODER_PATH}"
        )

    corpus_df = pd.read_parquet(CORPUS_PATH)
    required_columns = {"pid", "passage_text"}
    missing = required_columns.difference(corpus_df.columns)
    if missing:
        raise ValueError(f"{CORPUS_PATH} is missing columns: {sorted(missing)}")
    if corpus_df.empty:
        raise ValueError(f"Corpus is empty: {CORPUS_PATH}")
    if corpus_df["pid"].duplicated().any():
        raise ValueError("Corpus passage IDs must be unique")

    started = time.perf_counter()
    build_index(corpus_df=corpus_df, artifacts_dir=ARTIFACTS_DIR)
    baseline_index = get_index()
    baseline_elapsed = time.perf_counter() - started
    print(f"BM25 index load/build: {baseline_elapsed:.2f}s", flush=True)

    started = time.perf_counter()
    biencoder_model = load_biencoder(BIENCODER_PATH)
    configure_retrieval(
        corpus_df["passage_text"].astype(str).tolist(),
        pids=corpus_df["pid"].tolist(),
        batch_size=64,
        model=biencoder_model,
        cache_path=EMBEDDINGS_PATH,
    )
    biencoder_elapsed = time.perf_counter() - started
    print(f"Bi-encoder and cached corpus embedding load: {biencoder_elapsed:.2f}s", flush=True)

    started = time.perf_counter()
    crossencoder_model, crossencoder_tokenizer = load_crossencoder(
        CROSSENCODER_PATH
    )
    crossencoder_elapsed = time.perf_counter() - started
    print(f"Cross-encoder load: {crossencoder_elapsed:.2f}s", flush=True)
    return (
        corpus_df,
        baseline_index,
        biencoder_model,
        crossencoder_model,
        crossencoder_tokenizer,
    )


(
    CORPUS,
    BASELINE_INDEX,
    BIENCODER_MODEL,
    CROSSENCODER_MODEL,
    CROSSENCODER_TOKENIZER,
) = _load_models_and_indexes()
PASSAGE_TEXT_BY_PID = dict(
    zip(CORPUS["pid"].astype(int), CORPUS["passage_text"].astype(str))
)


def _warm_up_inference() -> None:
    sample = CORPUS["passage_text"].astype(str).head(100).tolist()
    if not sample:
        raise ValueError("Cannot warm up inference with an empty corpus")
    started = time.perf_counter()
    BASELINE_INDEX.retrieve_bm25("warm up retrieval", k=100)
    rerank(
        "warm up neural ranking",
        list(range(len(sample))),
        sample,
        k=10,
        model=CROSSENCODER_MODEL,
        tokenizer=CROSSENCODER_TOKENIZER,
        batch_size=32,
        max_length=MAX_SEQ_LEN,
        fp16=True,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(
        f"First-query inference warm-up: {time.perf_counter() - started:.2f}s",
        flush=True,
    )


_warm_up_inference()


def _passage_excerpt(text: str, limit: int = 300) -> str:
    excerpt = " ".join(text.split())
    return excerpt if len(excerpt) <= limit else excerpt[:limit].rstrip() + "..."


def _to_table(
    hits: List[Tuple[int, float]],
) -> pd.DataFrame:
    rows = [
        {
            "Rank": rank,
            "PID": pid,
            "Score": f"{score:.6f}",
            "Passage": _passage_excerpt(PASSAGE_TEXT_BY_PID[int(pid)]),
        }
        for rank, (pid, score) in enumerate(hits, start=1)
    ]
    return pd.DataFrame(rows, columns=["Rank", "PID", "Score", "Passage"])


def _sync_if_cuda() -> None:
    if next(CROSSENCODER_MODEL.parameters()).device.type == "cuda":
        torch.cuda.synchronize()


def search(query: str) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Return the BM25 baseline and neural reranking result tables."""
    query = query.strip()
    if not query:
        raise gr.Error("Enter a query before searching.")

    baseline_started = time.perf_counter()
    baseline_hits = BASELINE_INDEX.retrieve_bm25(query, k=10)
    baseline_latency = (time.perf_counter() - baseline_started) * 1000.0

    neural_started = time.perf_counter()
    candidates = BASELINE_INDEX.retrieve_bm25(query, k=100)
    candidate_pids = [pid for pid, _ in candidates]
    candidate_texts = [PASSAGE_TEXT_BY_PID[pid] for pid in candidate_pids]
    neural_hits = rerank(
        query,
        candidate_pids,
        candidate_texts,
        k=10,
        model=CROSSENCODER_MODEL,
        tokenizer=CROSSENCODER_TOKENIZER,
        batch_size=32,
        max_length=MAX_SEQ_LEN,
        fp16=True,
    )
    _sync_if_cuda()
    neural_latency = (time.perf_counter() - neural_started) * 1000.0

    return (
        _to_table(baseline_hits),
        _to_table(neural_hits),
        f"Total latency: **{baseline_latency:.1f} ms**",
        f"Total latency: **{neural_latency:.1f} ms** (BM25 top-100 + re-ranking)",
    )


with gr.Blocks(title="Neural Passage Search") as demo:
    gr.Markdown(
        "# Neural Passage Search\n"
        "Compare BM25 retrieval with cross-encoder re-ranking over the passage corpus."
    )
    query_input = gr.Textbox(
        label="Query",
        placeholder="Enter a question to search the passage corpus",
        lines=2,
    )
    search_button = gr.Button("Search", variant="primary")
    with gr.Row():
        with gr.Column():
            gr.Markdown("## BM25 Baseline top-10")
            baseline_results = gr.Dataframe(
                headers=["Rank", "PID", "Score", "Passage"],
                datatype=["number", "number", "str", "str"],
                interactive=False,
                wrap=True,
                label="Retrieved passages",
            )
            baseline_latency_output = gr.Markdown()
        with gr.Column():
            gr.Markdown("## Neural Re-ranked top-10")
            neural_results = gr.Dataframe(
                headers=["Rank", "PID", "Score", "Passage"],
                datatype=["number", "number", "str", "str"],
                interactive=False,
                wrap=True,
                label="Re-ranked passages",
            )
            neural_latency_output = gr.Markdown()

    search_button.click(
        fn=search,
        inputs=query_input,
        outputs=[
            baseline_results,
            neural_results,
            baseline_latency_output,
            neural_latency_output,
        ],
    )
    query_input.submit(
        fn=search,
        inputs=query_input,
        outputs=[
            baseline_results,
            neural_results,
            baseline_latency_output,
            neural_latency_output,
        ],
    )
    gr.Examples(
        examples=[[query] for query in EXAMPLE_QUERIES],
        inputs=query_input,
        label="Example queries",
    )


if __name__ == "__main__":
    demo.launch()
    # For remote access, use: demo.launch(server_name="0.0.0.0")
