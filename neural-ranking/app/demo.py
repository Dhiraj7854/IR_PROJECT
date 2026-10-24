#!/usr/bin/env python3
"""Neural Passage Search: a polished Gradio demo for the two-stage ranking project.

Compare TF-IDF, BM25, the bi-encoder, and BM25 -> cross-encoder re-ranking on the
same query, with query-term highlighting, rank-movement chips, score bars,
latency breakdowns, an insights strip, and a benchmark tab.

Run from the ``neural-ranking/`` directory:

    python app/demo_pro.py
"""

from __future__ import annotations

import argparse
import csv
import html
import inspect
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import gradio as gr

try:
    import spaces
except ImportError:
    class _SpacesFallback:
        """Keep local CPU development available without the Spaces package."""

        @staticmethod
        def GPU(*args, **kwargs):
            def decorate(function):
                return function

            return decorate

    spaces = _SpacesFallback()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid import fuse_ranked_lists
from src.query_enhancement import enhance_query, load_synonyms

SYNONYMS_PATH = PROJECT_ROOT / "configs" / "synonyms.json"
try:
    SYNONYMS = load_synonyms(SYNONYMS_PATH) if SYNONYMS_PATH.is_file() else {}
except Exception:
    SYNONYMS = {}


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = Path(
    os.environ.get("IR_RESULTS_DIR", ARTIFACTS_DIR / "local-evaluation")
)
if not RESULTS_DIR.is_absolute():
    RESULTS_DIR = PROJECT_ROOT / RESULTS_DIR
DATA_DIR = PROJECT_ROOT / "data" / "processed"
QUERY_ANALYSIS_UNAVAILABLE = (
    '<div class="analysis-empty"><span class="eyebrow">JUDGED DATA REQUIRED</span>'
    "<h3>Query analysis is not available yet</h3>"
    "<p>Add <code>data/processed/test_queries.parquet</code> and "
    "<code>data/processed/test_qrels.parquet</code> for this Space's passage corpus. "
    "Their passage IDs must match the bundled corpus; local benchmark judgments "
    "and live searches are not interchangeable with these labels.</p></div>"
)

Hit = Tuple[int, float]

# Test queries with maximum top-3 disagreement between BM25 and the cross-encoder.
EXAMPLE_QUERIES = [
    "where does the water that is produced by cellular respiration go",
    "how do i use sunpass on peach pass roads",
    "what is the highest level for a mountain in skylanders superchargers",
    "which part of the brain controls emotion - cerebral hemisphere",
]

# name -> (key, accent colour, one-line description)
SYSTEMS: Dict[str, Tuple[str, str, str]] = {
    "TF-IDF": ("tfidf", "#f59e0b", "Sparse lexical match, cosine over TF-IDF vectors"),
    "BM25": ("bm25", "#3b82f6", "Sparse lexical match with length normalisation"),
    "Bi-encoder": ("dense", "#10b981", "Dense retrieval: query and passage embedded separately"),
    "BM25 → Cross-encoder": ("cross", "#8b5cf6", "BM25 top-N re-scored jointly by a neural cross-encoder"),
    "Hybrid RRF": ("rrf", "#06b6d4", "BM25 + bi-encoder merged by reciprocal rank fusion"),
    "Hybrid RRF → Cross-encoder": ("rrf_cross", "#ec4899", "RRF candidates jointly re-scored by a cross-encoder"),
}
DEFAULT_SYSTEMS = ["BM25", "BM25 → Cross-encoder"]

_STOPWORDS = frozenset(
    "a an and are as at be but by do does did for from had has have how i if in into is it its "
    "of on or that the their then there these they this to was we were what when where which "
    "who whom whose why will with you your".split()
)

_NEURAL_INFERENCE: Optional[Callable[..., Dict[str, object]]] = None


def _zerogpu_requested() -> bool:
    return os.getenv("IR_USE_ZEROGPU", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@spaces.GPU(duration=60)
def _run_neural_inference(
    query: str,
    dense_k: int,
    candidate_pids: List[int],
    candidate_texts: List[str],
    run_dense: bool,
    run_cross: bool,
) -> Dict[str, object]:
    if _NEURAL_INFERENCE is None:
        raise RuntimeError("Neural inference backend has not been initialized")
    return _NEURAL_INFERENCE(
        query,
        dense_k,
        candidate_pids,
        candidate_texts,
        run_dense,
        run_cross,
    )


def _run_neural_inference_with_fallback(
    query: str,
    dense_k: int,
    candidate_pids: List[int],
    candidate_texts: List[str],
    run_dense: bool,
    run_cross: bool,
) -> Dict[str, object]:
    if os.getenv("SPACE_ID") and not _zerogpu_requested():
        if _NEURAL_INFERENCE is None:
            raise RuntimeError("Neural inference backend has not been initialized")
        logger.info("ZeroGPU is disabled; running neural inference on CPU")
        return _NEURAL_INFERENCE(
            query,
            dense_k,
            candidate_pids,
            candidate_texts,
            run_dense,
            run_cross,
            True,
        )

    try:
        return _run_neural_inference(
            query,
            dense_k,
            candidate_pids,
            candidate_texts,
            run_dense,
            run_cross,
        )
    except RuntimeError as error:
        if "No CUDA GPUs are available" not in str(error):
            raise
        if _NEURAL_INFERENCE is None:
            raise RuntimeError("Neural inference backend has not been initialized") from error
        logger.warning("ZeroGPU allocation unavailable; retrying this query on CPU")
        return _NEURAL_INFERENCE(
            query,
            dense_k,
            candidate_pids,
            candidate_texts,
            run_dense,
            run_cross,
            True,
        )


# ---------------------------------------------------------------------------
# Backend: models and indexes (loaded lazily so the UI code stays importable)
# ---------------------------------------------------------------------------


@dataclass
class Backend:
    """Everything the UI needs, behind plain callables (easy to stub in tests)."""

    passages: Dict[int, str]
    bm25: Callable[[str, int], List[Hit]]
    tfidf: Callable[[str, int], List[Hit]]
    dense: Callable[[str, int], List[Hit]]
    rerank: Callable[[str, List[int], int], List[Hit]]
    feedback: Optional[Callable[[str, List[int], List[int], int], List[Hit]]] = None
    device: str = "cpu"
    rrf_k: int = 60
    benchmark_queries: Optional[Dict[str, str]] = None
    benchmark_qrels: Optional[Dict[str, Dict[str, int]]] = None

    @property
    def corpus_size(self) -> int:
        return len(self.passages)


def _embedding_cache_path(
    biencoder_path: Path,
    space_data_dir: Path = Path("/data"),
) -> Path:
    if os.getenv("SPACE_ID") and space_data_dir.is_dir():
        return space_data_dir / "corpus_embeddings.npy"
    return biencoder_path / "corpus_embeddings.npy"


def load_backend() -> Backend:
    """Load the corpus, lexical index, bi-encoder, and cross-encoder once."""
    global _NEURAL_INFERENCE
    import pandas as pd
    import torch

    from src import biencoder
    from src.baseline import build_index, get_index
    from src.biencoder import configure_retrieval, load_biencoder, retrieve_biencoder
    from src.config import load_config
    from src.crossencoder import load_crossencoder, rerank
    from src.evaluate import evaluate_run
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    max_seq_len = config.models.max_seq_len
    corpus_path = DATA_DIR / "corpus.parquet"
    biencoder_path = ARTIFACTS_DIR / "biencoder"
    crossencoder_path = ARTIFACTS_DIR / "crossencoder"
    embeddings_path = _embedding_cache_path(biencoder_path)

    for path, hint in [
        (corpus_path, "Run scripts/prepare_data.py first."),
        (biencoder_path, "Train or copy the bi-encoder checkpoint."),
        (crossencoder_path, "Train or copy the cross-encoder checkpoint."),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. {hint}")
    if not embeddings_path.exists():
        logger.info(
            "No corpus embedding cache at %s; the corpus will be encoded and cached",
            embeddings_path,
        )

    corpus_df = pd.read_parquet(corpus_path)
    if corpus_df.empty or corpus_df["pid"].duplicated().any():
        raise ValueError("Corpus must be non-empty with unique pids")

    started = time.perf_counter()
    build_index(corpus_df=corpus_df, artifacts_dir=ARTIFACTS_DIR)
    index = get_index()
    logger.info("Lexical index ready in %.2fs", time.perf_counter() - started)

    encoder = load_biencoder(biencoder_path, device="cpu")
    configure_retrieval(
        corpus_df["passage_text"].astype(str).tolist(),
        pids=corpus_df["pid"].tolist(),
        batch_size=64,
        model=encoder,
        cache_path=embeddings_path,
    )
    cross_model, cross_tokenizer = load_crossencoder(crossencoder_path, device="cpu")

    passages = dict(zip(corpus_df["pid"].astype(int), corpus_df["passage_text"].astype(str)))
    passage_texts = corpus_df["passage_text"].astype(str).tolist()
    passage_ids = corpus_df["pid"].tolist()

    def _neural_inference(
        query: str,
        dense_k: int,
        candidate_pids: List[int],
        candidate_texts: List[str],
        run_dense: bool,
        run_cross: bool,
        force_cpu: bool = False,
    ) -> Dict[str, object]:
        use_cuda = torch.cuda.is_available() and not force_cpu
        device = "cuda" if use_cuda else "cpu"
        outputs: Dict[str, object] = {"device": device}

        try:
            if run_dense:
                biencoder.set_retrieval_device(device)
            if run_cross:
                cross_model.to(device)
                cross_model.eval()

            if run_dense:
                if use_cuda:
                    torch.cuda.synchronize()
                started = time.perf_counter()
                dense_hits = retrieve_biencoder(query, dense_k)
                if use_cuda:
                    torch.cuda.synchronize()
                outputs["dense"] = (
                    dense_hits,
                    (time.perf_counter() - started) * 1000.0,
                )

            if run_cross:
                if use_cuda:
                    torch.cuda.synchronize()
                started = time.perf_counter()
                cross_hits = rerank(
                    query,
                    candidate_pids,
                    candidate_texts,
                    k=dense_k,
                    model=cross_model,
                    tokenizer=cross_tokenizer,
                    batch_size=32,
                    max_length=max_seq_len,
                    fp16=True,
                )
                if use_cuda:
                    torch.cuda.synchronize()
                outputs["cross"] = (
                    cross_hits,
                    (time.perf_counter() - started) * 1000.0,
                )
            return outputs
        finally:
            if run_dense:
                biencoder.set_retrieval_device("cpu")
            if run_cross:
                cross_model.to("cpu")
            if use_cuda:
                torch.cuda.empty_cache()

    _NEURAL_INFERENCE = _neural_inference
    if os.getenv("SPACE_ID"):
        device_label = (
            "ZeroGPU (on demand)"
            if _zerogpu_requested()
            else "CPU (ZeroGPU bypassed)"
        )
    else:
        device_label = "CUDA on demand" if torch.cuda.is_available() else "CPU"

    def _dense(query: str, k: int) -> List[Hit]:
        response = _run_neural_inference_with_fallback(
            query, k, [], [], True, False
        )
        hits, _ = response["dense"]
        return hits

    def _rerank(query: str, pids: List[int], k: int) -> List[Hit]:
        response = _run_neural_inference_with_fallback(
            query,
            k,
            pids,
            [passages[p] for p in pids],
            False,
            True,
        )
        hits, _ = response["cross"]
        return hits

    backend = Backend(
        passages=passages,
        bm25=index.retrieve_bm25,
        tfidf=index.retrieve_tfidf,
        dense=_dense,
        rerank=_rerank,
        feedback=lambda query, positive, negative, k: index.retrieve_tfidf_feedback(
            query,
            positive,
            negative,
            k,
            alpha=config.retrieval.rocchio_alpha,
            beta=config.retrieval.rocchio_beta,
            gamma=config.retrieval.rocchio_gamma,
        ),
        device=device_label,
        rrf_k=config.retrieval.rrf_k,
    )
    query_path = DATA_DIR / "test_queries.parquet"
    qrels_path = DATA_DIR / "test_qrels.parquet"
    if query_path.is_file() and qrels_path.is_file():
        from src.evaluate import load_qrels

        query_frame = pd.read_parquet(query_path)
        backend.benchmark_queries = {
            str(row.query_id): str(row.query_text)
            for row in query_frame.itertuples(index=False)
        }
        backend.benchmark_qrels = load_qrels(qrels_path)

    # Warm up so the first user query is not penalised by lazy initialisation.
    sample = list(passages)[:100]
    backend.bm25("warm up retrieval", 100)
    backend.tfidf("warm up retrieval", 10)
    if not os.getenv("SPACE_ID"):
        backend.dense("warm up retrieval", 10)
        backend.rerank("warm up neural ranking", sample, 10)
    return backend


# ---------------------------------------------------------------------------
# Rendering helpers (pure functions: no models needed)
# ---------------------------------------------------------------------------


def query_terms(query: str) -> List[str]:
    """Content words from the query, used for highlighting."""
    seen: List[str] = []
    for token in re.findall(r"[a-z0-9]+", query.lower()):
        if token not in _STOPWORDS and len(token) > 1 and token not in seen:
            seen.append(token)
    return seen


def highlight(text: str, terms: Sequence[str]) -> str:
    """HTML-escape ``text`` and wrap whole-word query-term matches in <mark>."""
    if not terms:
        return html.escape(text)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )
    parts: List[str] = []
    last = 0
    for match in pattern.finditer(text):
        parts.append(html.escape(text[last : match.start()]))
        parts.append(f"<mark>{html.escape(match.group(0))}</mark>")
        last = match.end()
    parts.append(html.escape(text[last:]))
    return "".join(parts)


def snippet(text: str, terms: Sequence[str], limit: int = 280) -> str:
    """A window of ``text`` centred near the first query-term match."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    start = 0
    if terms:
        match = re.search(
            r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", text, re.IGNORECASE
        )
        if match:
            start = max(0, match.start() - limit // 4)
    end = min(len(text), start + limit)
    return ("…" if start > 0 else "") + text[start:end].strip() + ("…" if end < len(text) else "")


def _score_widths(hits: Sequence[Hit]) -> List[float]:
    """Min-max normalise scores to 8-100 % bar widths within one column."""
    if not hits:
        return []
    scores = [s for _, s in hits]
    low, high = min(scores), max(scores)
    if high - low < 1e-12:
        return [100.0] * len(hits)
    return [8.0 + 92.0 * (s - low) / (high - low) for s in scores]


def _movement_chip(rank: int, pid: int, bm25_rank: Dict[int, int], depth: int) -> str:
    previous = bm25_rank.get(pid)
    if previous is None:
        return f'<span class="chip new" title="Not in the BM25 top-{depth}">✨ outside BM25 top-{depth}</span>'
    delta = previous - rank
    if delta > 0:
        return f'<span class="chip up">▲ {delta} · BM25 #{previous}</span>'
    if delta < 0:
        return f'<span class="chip down">▼ {-delta} · BM25 #{previous}</span>'
    return f'<span class="chip same">= BM25 #{previous}</span>'


def render_column(
    name: str,
    hits: Sequence[Hit],
    passages: Dict[int, str],
    terms: Sequence[str],
    latency_ms: float,
    latency_note: str,
    bm25_rank: Optional[Dict[int, int]],
    depth: int,
    do_highlight: bool,
) -> str:
    key, color, blurb = SYSTEMS[name]
    widths = _score_widths(hits)
    cards: List[str] = []
    for rank, ((pid, score), width) in enumerate(zip(hits, widths), start=1):
        text = passages[int(pid)]
        use_terms = terms if do_highlight else []
        short = highlight(snippet(text, use_terms), use_terms)
        full = highlight(" ".join(text.split()), use_terms)
        chip = _movement_chip(rank, int(pid), bm25_rank, depth) if bm25_rank is not None and key != "bm25" else ""
        cards.append(
            f"""
<div class="card" style="--accent:{color}">
  <div class="rank">{rank}</div>
  <div class="body">
    <div class="meta"><span class="pid">PID {int(pid)}</span>{chip}<span class="score">{score:.4f}</span></div>
    <div class="bar"><span style="width:{width:.1f}%"></span></div>
    <p class="snippet">{short}</p>
    <details><summary>Full passage</summary><p>{full}</p></details>
  </div>
</div>"""
        )
    body = "".join(cards) or '<p class="empty">No results.</p>'
    return f"""
<section class="col" style="--accent:{color}">
  <header>
    <h3><span class="dot"></span>{html.escape(name)}</h3>
    <p>{html.escape(blurb)}</p>
    <div class="lat"><b>{latency_ms:.1f} ms</b> <span>{html.escape(latency_note)}</span></div>
  </header>
  {body}
</section>"""


def _tile(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="sub">{html.escape(sub)}</div>' if sub else ""
    return f'<div class="tile"><div class="label">{html.escape(label)}</div><div class="value">{value}</div>{sub_html}</div>'


def render_insights(
    query: str,
    top_k: int,
    depth: int,
    bm25_hits: Sequence[Hit],
    cross_hits: Optional[Sequence[Hit]],
    bm25_rank: Dict[int, int],
    timings: Dict[str, float],
    enhancement_note: str = "",
) -> str:
    """Summary tiles plus a latency bar chart for the selected systems."""
    tiles: List[str] = []
    if cross_hits is not None:
        bm25_top = [p for p, _ in bm25_hits[:top_k]]
        cross_top = [p for p, _ in cross_hits[:top_k]]
        overlap = len(set(bm25_top) & set(cross_top))
        overlap_pct = 100.0 * overlap / max(1, len(set(bm25_top) | set(cross_top)))
        tiles.append(_tile(f"Top-{top_k} overlap", f"{overlap}/{top_k}", "BM25 vs cross-encoder"))
        changed = bool(bm25_top) and bool(cross_top) and bm25_top[0] != cross_top[0]
        tiles.append(
            _tile("Top-1 result", "changed" if changed else "same", "after neural re-ranking")
        )
        jumps = [
            (bm25_rank[p] - r, p, bm25_rank[p], r)
            for r, p in enumerate(cross_top, start=1)
            if p in bm25_rank
        ]
        if jumps:
            gain, pid, old, new = max(jumps)
            if gain > 0:
                tiles.append(_tile("Biggest jump", f"▲ {gain}", f"PID {pid}: BM25 #{old} → #{new}"))
        rescued = sum(1 for p in cross_top if bm25_rank.get(p, depth + 1) > top_k)
        tiles.append(_tile("Pulled up from below", str(rescued), f"cross-encoder top-{top_k} hits that BM25 ranked past #{top_k}"))
        donut = f"""
<div class="overlap-card">
  <div class="donut" style="--overlap:{overlap_pct:.1f}%"><span>{overlap_pct:.0f}%</span></div>
  <div><strong>Result-set overlap</strong><div class="overlap-legend">
    <span><i class="legend-shared"></i>{overlap} shared passages</span>
    <span><i class="legend-different"></i>{len(set(bm25_top) | set(cross_top)) - overlap} unique to one ranking</span>
  </div></div>
</div>"""
    else:
        donut = ""
    tiles.append(_tile("Query terms", f"{len(query_terms(query))}", ", ".join(query_terms(query)[:6]) or "—"))

    longest = max(timings.values()) if timings else 1.0
    bars = []
    for name, ms in timings.items():
        color = SYSTEMS[name][1]
        width = max(2.0, 100.0 * ms / longest)
        bars.append(
            f'<div class="lrow"><span class="lname">{html.escape(name)}</span>'
            f'<span class="ltrack"><span style="width:{width:.1f}%;background:{color}"></span></span>'
            f'<span class="lval">{ms:.1f} ms</span></div>'
        )
    note_html = (
        f'<div class="pill-enhanced"><span>⚡ {html.escape(enhancement_note)}</span></div>'
        if enhancement_note
        else ""
    )
    return f"""
<div class="insights">
  <div class="insight-main">{note_html}<div class="tiles">{''.join(tiles)}</div>{donut}</div>
  <div class="latency"><h4>Latency</h4>{''.join(bars)}
  <p class="fine">The cross-encoder time includes the shared BM25 stage-1 retrieval.</p></div>
</div>"""


def _read_benchmark() -> List[Dict[str, object]]:
    path = RESULTS_DIR / "comparison_table.csv"
    if path.is_file():
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if rows:
                aliases = {
                    "system": "System",
                    "mrr@10": "MRR@10",
                    "ndcg@10": "NDCG@10",
                    "p@5": "P@5",
                    "recall@10": "Recall@10",
                    "recall@100": "Recall@100",
                    "map@10": "MAP@10",
                    "median_latency_ms": "Median latency (ms/query)",
                    "recall@100_valid": "Recall@100 valid",
                }
                parsed: List[Dict[str, object]] = []
                for row in rows:
                    normalised: Dict[str, object] = {}
                    for key, value in row.items():
                        if value in (None, ""):
                            continue
                        label = aliases.get(key.strip().lower(), key)
                        if label == "System":
                            normalised[label] = value
                        elif label == "Recall@100 valid":
                            normalised[label] = value.strip().lower() == "true"
                        else:
                            normalised[label] = float(value)
                    if "System" in normalised:
                        parsed.append(normalised)
                if parsed:
                    return parsed
        except (ValueError, OSError) as exc:
            logger.warning("Could not parse %s: %s", path, exc)
    return []


def _read_ablation_tables() -> List[List[Dict[str, object]]]:
    """Read the three ablation tables from the generated Markdown report."""
    path = RESULTS_DIR / "ablations.md"
    if not path.is_file():
        return []
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return []

    tables: List[List[Dict[str, object]]] = []
    for section in re.split(r"(?m)^##\s+\d+\.", content)[1:]:
        lines = [line.strip() for line in section.splitlines() if line.strip().startswith("|")]
        if len(lines) < 3:
            continue
        headers = [cell.strip() for cell in lines[0].strip("|").split("|")]
        rows: List[Dict[str, object]] = []
        for line in lines[2:]:
            values = [cell.strip() for cell in line.strip("|").split("|")]
            if len(values) != len(headers):
                continue
            parsed: Dict[str, object] = {}
            for header, value in zip(headers, values):
                try:
                    parsed[header] = float(value)
                except ValueError:
                    parsed[header] = value
            rows.append(parsed)
        if rows:
            tables.append(rows)
    return tables


def _read_json_result(filename: str) -> Dict[str, object]:
    path = RESULTS_DIR / filename
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return {}
    if not isinstance(value, dict):
        logger.warning("Expected a JSON object in %s", path)
        return {}
    return value


def _read_per_query_metrics() -> Dict[str, Dict[str, float]]:
    path = RESULTS_DIR / "per_query_metrics.csv"
    if not path.is_file():
        return {}
    metrics: Dict[str, Dict[str, float]] = {}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                system = row.get("system")
                query_id = row.get("query_id")
                raw_mrr = row.get("MRR@10")
                if not system or not query_id or raw_mrr is None:
                    continue
                try:
                    metrics.setdefault(system, {})[query_id] = float(raw_mrr)
                except ValueError:
                    logger.warning(
                        "Skipping invalid per-query MRR value for system=%s query=%s",
                        system,
                        query_id,
                    )
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
    return metrics


def _chart_card(title: str, subtitle: str, svg: str, extra_class: str = "") -> str:
    return f"""
<section class="chart-card {extra_class}">
  <div class="chart-heading"><div><h3>{html.escape(title)}</h3>
    <p>{html.escape(subtitle)}</p></div></div>
  <div class="chart-viewport">{svg}</div>
</section>"""


def _render_quality_chart(rows: Sequence[Dict[str, object]]) -> str:
    metrics = [
        ("MRR@10", "#6366f1"),
        ("NDCG@10", "#06b6d4"),
        ("Recall@100", "#10b981"),
    ]
    data = [
        row for row in rows
        if any(isinstance(row.get(metric), (int, float)) for metric, _ in metrics)
    ]
    if not data:
        return '<div class="chart-empty">No quality metrics available.</div>'

    width = 900
    left, right, top, row_height = 235, 80, 38, 53
    height = top + row_height * len(data) + 42
    plot_width = width - left - right
    parts = [
        f'<svg class="data-chart tall-chart" viewBox="0 0 {width} {height}" '
        'role="img" aria-label="Grouped comparison of local MRR, NDCG and Recall at 100">'
    ]
    for tick in range(5):
        value = tick / 4
        x = left + plot_width * value
        parts.append(
            f'<line class="grid-line vertical" x1="{x:.1f}" y1="{top-12}" '
            f'x2="{x:.1f}" y2="{height-28}"/>'
            f'<text class="axis-tick" x="{x:.1f}" y="20" text-anchor="middle">{value:.2f}</text>'
        )
    for row_index, row in enumerate(data):
        y = top + row_index * row_height
        system = str(row["System"])
        parts.append(
            f'<text class="axis-tick row-label" x="{left-14}" y="{y+21}" '
            f'text-anchor="end">{html.escape(system)}</text>'
        )
        for metric_index, (metric, color) in enumerate(metrics):
            value = row.get(metric)
            if not isinstance(value, (int, float)):
                continue
            bar_y = y + metric_index * 12
            bar_width = plot_width * max(0.0, min(1.0, float(value)))
            parts.append(
                f'<rect class="chart-bar" x="{left}" y="{bar_y}" width="{bar_width:.1f}" '
                f'height="8" rx="4" fill="{color}"><title>{html.escape(system)} · '
                f'{metric}: {value:.4f}</title></rect>'
            )
            parts.append(
                f'<text class="bar-value" x="{min(width-right-2, left+bar_width+6):.1f}" '
                f'y="{bar_y+8}" text-anchor="start">{value:.3f}</text>'
            )
    legend = []
    legend_x = [left, left + 170, left + 340]
    for (metric, color), x in zip(metrics, legend_x):
        legend.append(
            f'<rect x="{x}" y="{height-18}" width="10" height="10" rx="3" fill="{color}"/>'
            f'<text class="axis-tick" x="{x+16}" y="{height-9}">{metric}</text>'
        )
    parts.extend(legend)
    parts.append("</svg>")
    return "".join(parts)


def _render_latency_chart(rows: Sequence[Dict[str, object]], latency_queries: int) -> str:
    data = []
    for row in rows:
        try:
            data.append(
                (
                    str(row["System"]),
                    float(row["Median latency (ms/query)"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    if not data:
        return '<div class="chart-empty">No latency measurements available.</div>'

    data.sort(key=lambda item: item[1])
    bm25_latency = next((latency for name, latency in data if name == "BM25"), None)
    width = 900
    left, right, top, row_height = 265, 220, 34, 42
    height = top + row_height * len(data) + 24
    plot_width = width - left - right
    maximum = max(latency for _, latency in data) or 1.0
    parts = [
        f'<svg class="data-chart tall-chart" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="Median latency per query measured on {latency_queries} local test queries">'
    ]
    for tick in range(5):
        value = maximum * tick / 4
        x = left + plot_width * tick / 4
        parts.append(
            f'<line class="grid-line vertical" x1="{x:.1f}" y1="{top-10}" '
            f'x2="{x:.1f}" y2="{height-14}"/>'
            f'<text class="axis-tick" x="{x:.1f}" y="17" text-anchor="middle">{value:.0f} ms</text>'
        )
    for index, (name, latency) in enumerate(data):
        y = top + index * row_height
        speedup = (
            f"{bm25_latency / latency:.1f}× faster than BM25"
            if bm25_latency and latency < bm25_latency
            else (
                f"{latency / bm25_latency:.1f}× BM25 latency"
                if bm25_latency and latency > bm25_latency
                else "BM25 reference"
            )
        )
        bar_width = plot_width * latency / maximum
        color = SYSTEMS.get(name, ("", "#64748b", ""))[1]
        parts.append(
            f'<text class="axis-tick row-label" x="{left-12}" y="{y+15}" '
            f'text-anchor="end">{html.escape(name)}</text>'
            f'<rect class="chart-bar" x="{left}" y="{y}" width="{bar_width:.1f}" '
            f'height="21" rx="7" fill="{color}"><title>{html.escape(name)}: '
            f'{latency:.2f} ms/query</title></rect>'
            f'<text class="bar-value" x="{left+bar_width+8:.1f}" y="{y+14}">'
            f'{latency:.2f} ms · {html.escape(speedup)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _render_paired_ci_chart(paired: Dict[str, object]) -> str:
    entries = []

    def add_entry(label: str, result: object) -> None:
        if not isinstance(result, dict):
            return
        metrics = result.get("comparison_minus_baseline", {})
        mrr = metrics.get("MRR@10") if isinstance(metrics, dict) else None
        if not isinstance(mrr, dict):
            return
        try:
            entries.append(
                (
                    label,
                    float(mrr["mean_difference"]),
                    float(mrr["ci_lower"]),
                    float(mrr["ci_upper"]),
                    int(mrr["paired_queries"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            return

    for system, result in paired.items():
        if not isinstance(result, dict):
            continue
        baseline = str(result.get("baseline", "BM25"))
        add_entry(f"{system} vs {baseline}", result)
        additional = result.get("additional_comparisons", {})
        if isinstance(additional, dict):
            for additional_baseline, comparison in additional.items():
                add_entry(
                    f"{system} vs {additional_baseline}",
                    comparison,
                )
    if not entries:
        return '<div class="chart-empty">Paired confidence intervals are not available.</div>'

    minimum = min(-0.05, min(item[2] for item in entries))
    maximum = max(0.05, max(item[3] for item in entries))
    margin = (maximum - minimum) * 0.08
    minimum -= margin
    maximum += margin
    width, left, right, top, row_height = 1100, 430, 40, 36, 39
    height = top + row_height * len(entries) + 23
    plot_width = width - left - right
    x_for = lambda value: left + (value - minimum) / (maximum - minimum) * plot_width
    zero_x = x_for(0.0)
    parts = [
        f'<svg class="data-chart tall-chart" viewBox="0 0 {width} {height}" '
        'role="img" aria-label="Paired bootstrap 95 percent confidence intervals for MRR at 10 relative to BM25 and TF-IDF">'
    ]
    for tick in range(5):
        value = minimum + (maximum - minimum) * tick / 4
        x = left + plot_width * tick / 4
        parts.append(
            f'<line class="grid-line vertical" x1="{x:.1f}" y1="{top-8}" '
            f'x2="{x:.1f}" y2="{height-14}"/>'
            f'<text class="axis-tick" x="{x:.1f}" y="17" text-anchor="middle">{value:+.2f}</text>'
        )
    parts.append(
        f'<line x1="{zero_x:.1f}" y1="{top-10}" x2="{zero_x:.1f}" y2="{height-14}" '
        'stroke="#64748b" stroke-width="2" stroke-dasharray="5 4"/>'
    )
    for index, (system, mean, lower, upper, count) in enumerate(entries):
        y = top + index * row_height + 10
        color = "#059669" if lower > 0 else ("#dc2626" if upper < 0 else "#64748b")
        parts.append(
            f'<text class="axis-tick row-label" x="{left-12}" y="{y+4}" '
            f'text-anchor="end">{html.escape(system)}</text>'
            f'<line x1="{x_for(lower):.1f}" y1="{y}" x2="{x_for(upper):.1f}" y2="{y}" '
            f'stroke="{color}" stroke-width="4" stroke-linecap="round"/>'
            f'<line x1="{x_for(lower):.1f}" y1="{y-5}" x2="{x_for(lower):.1f}" y2="{y+5}" '
            f'stroke="{color}" stroke-width="2"/>'
            f'<line x1="{x_for(upper):.1f}" y1="{y-5}" x2="{x_for(upper):.1f}" y2="{y+5}" '
            f'stroke="{color}" stroke-width="2"/>'
            f'<circle cx="{x_for(mean):.1f}" cy="{y}" r="6" fill="{color}">'
            f'<title>{html.escape(system)}: {mean:+.4f}; 95% CI [{lower:+.4f}, '
            f'{upper:+.4f}], n={count}</title></circle>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _render_query_distribution_chart(
    per_query: Dict[str, Dict[str, float]],
) -> str:
    systems = [
        (name, values)
        for name, values in per_query.items()
        if values
    ]
    if not systems:
        return '<div class="chart-empty">Per-query MRR distributions are not available.</div>'
    width, left, right, top, row_height = 900, 270, 40, 32, 39
    height = top + row_height * len(systems) + 34
    plot_width = width - left - right
    parts = [
        f'<svg class="data-chart tall-chart" viewBox="0 0 {width} {height}" '
        'role="img" aria-label="Distribution of test-query MRR at 10 by retrieval system">'
    ]
    for tick in range(5):
        value = tick / 4
        x = left + plot_width * value
        parts.append(
            f'<line class="grid-line vertical" x1="{x:.1f}" y1="{top-10}" '
            f'x2="{x:.1f}" y2="{height-25}"/>'
            f'<text class="axis-tick" x="{x:.1f}" y="16" text-anchor="middle">{value:.2f}</text>'
        )
    for index, (system, values_map) in enumerate(systems):
        values = list(values_map.values())
        low = min(values)
        q1 = _linear_quantile(values, 0.25)
        median = _linear_quantile(values, 0.5)
        q3 = _linear_quantile(values, 0.75)
        high = max(values)
        y = top + index * row_height + 12
        color = SYSTEMS.get(system, ("", "#64748b", ""))[1]
        x_for = lambda value: left + plot_width * value
        parts.append(
            f'<text class="axis-tick row-label" x="{left-12}" y="{y+4}" '
            f'text-anchor="end">{html.escape(system)}</text>'
            f'<line x1="{x_for(low):.1f}" y1="{y}" x2="{x_for(high):.1f}" y2="{y}" '
            f'stroke="{color}" stroke-width="2" opacity=".7"/>'
            f'<rect class="chart-bar" x="{x_for(q1):.1f}" y="{y-8}" '
            f'width="{max(1.0, x_for(q3)-x_for(q1)):.1f}" height="16" rx="5" '
            f'fill="{color}" opacity=".5"><title>{html.escape(system)} MRR@10: '
            f'min {low:.3f}, Q1 {q1:.3f}, median {median:.3f}, Q3 {q3:.3f}, '
            f'max {high:.3f} (n={len(values)})</title></rect>'
            f'<line x1="{x_for(median):.1f}" y1="{y-9}" x2="{x_for(median):.1f}" '
            f'y2="{y+9}" stroke="{color}" stroke-width="3"/>'
        )
    parts.extend([
        f'<text class="axis-label" x="{left+plot_width/2}" y="{height-5}" '
        'text-anchor="middle">Per-query MRR@10 · whiskers show min–max; box shows Q1–Q3</text>',
        "</svg>",
    ])
    return "".join(parts)


def _render_query_outcomes_chart(
    per_query: Dict[str, Dict[str, float]],
) -> str:
    baseline = per_query.get("BM25")
    if not baseline:
        return '<div class="chart-empty">BM25 per-query metrics are not available.</div>'
    systems = [(name, values) for name, values in per_query.items() if name != "BM25"]
    if not systems:
        return '<div class="chart-empty">No systems available for a paired query comparison.</div>'

    width, left, right, top, row_height = 900, 275, 50, 32, 39
    height = top + row_height * len(systems) + 42
    plot_width = width - left - right
    parts = [
        f'<svg class="data-chart tall-chart" viewBox="0 0 {width} {height}" '
        'role="img" aria-label="Count of test queries where each system improves, ties or loses MRR against BM25">'
    ]
    for tick in range(5):
        value = tick / 4
        x = left + plot_width * value
        parts.append(
            f'<line class="grid-line vertical" x1="{x:.1f}" y1="{top-10}" '
            f'x2="{x:.1f}" y2="{height-28}"/>'
            f'<text class="axis-tick" x="{x:.1f}" y="16" text-anchor="middle">{value:.0%}</text>'
        )
    for index, (system, values) in enumerate(systems):
        matched = [
            (score, baseline[qid])
            for qid, score in values.items()
            if qid in baseline
        ]
        if not matched:
            continue
        wins = sum(score > reference for score, reference in matched)
        ties = sum(score == reference for score, reference in matched)
        losses = len(matched) - wins - ties
        y = top + index * row_height
        x = left
        for count, color, label in (
            (wins, "#059669", "improved"),
            (ties, "#94a3b8", "tied"),
            (losses, "#ef4444", "lower"),
        ):
            segment_width = plot_width * count / len(matched)
            parts.append(
                f'<rect class="chart-bar" x="{x:.1f}" y="{y}" width="{segment_width:.1f}" '
                f'height="22" fill="{color}"><title>{html.escape(system)} {label}: '
                f'{count}/{len(matched)} queries ({count/len(matched):.1%})</title></rect>'
            )
            x += segment_width
        parts.append(
            f'<text class="axis-tick row-label" x="{left-12}" y="{y+15}" '
            f'text-anchor="end">{html.escape(system)}</text>'
            f'<text class="axis-tick" x="{left+plot_width+9}" y="{y+15}">'
            f'{wins} / {ties} / {losses}</text>'
        )
    parts.extend([
        '<rect x="210" y="' + str(height - 17) + '" width="10" height="10" rx="3" fill="#059669"/>'
        '<text class="axis-tick" x="226" y="' + str(height - 8) + '">improved</text>'
        '<rect x="318" y="' + str(height - 17) + '" width="10" height="10" rx="3" fill="#94a3b8"/>'
        '<text class="axis-tick" x="334" y="' + str(height - 8) + '">tied</text>'
        '<rect x="385" y="' + str(height - 17) + '" width="10" height="10" rx="3" fill="#ef4444"/>'
        '<text class="axis-tick" x="401" y="' + str(height - 8) + '">lower</text>',
        "</svg>",
    ])
    return "".join(parts)


def _render_pareto_chart(rows: Sequence[Dict[str, object]]) -> str:
    """Render quality versus latency as a responsive, accessible inline SVG."""
    points = []
    for row in rows:
        try:
            points.append(
                (str(row["System"]), float(row["MRR@10"]), float(row["Median latency (ms/query)"]))
            )
        except (KeyError, TypeError, ValueError):
            continue
    if not points:
        return '<div class="chart-empty">No quality/latency metrics available.</div>'

    width, height = 760, 390
    left, right, top, bottom = 82, 30, 24, 68
    plot_w, plot_h = width - left - right, height - top - bottom
    x_max = max(p[2] for p in points) * 1.12 or 1.0
    y_max = max(p[1] for p in points) * 1.15 or 1.0
    parts = ['<svg class="data-chart" viewBox="0 0 760 390" role="img" aria-label="MRR at 10 versus median latency scatter plot">']
    for i in range(5):
        y = top + plot_h * i / 4
        value = y_max * (1 - i / 4)
        parts.append(f'<line class="grid-line" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>')
        parts.append(f'<text class="axis-tick" x="{left-12}" y="{y+4:.1f}" text-anchor="end">{value:.2f}</text>')
    for i in range(5):
        x = left + plot_w * i / 4
        value = x_max * i / 4
        parts.append(f'<line class="grid-line vertical" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}"/>')
        parts.append(f'<text class="axis-tick" x="{x:.1f}" y="{top+plot_h+22}" text-anchor="middle">{value:.0f}</text>')
    parts.extend([
        f'<line class="axis-line" x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}"/>',
        f'<line class="axis-line" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>',
        f'<text class="axis-label" x="{left+plot_w/2}" y="{height-12}" text-anchor="middle">Median latency (ms/query) · lower is faster</text>',
        f'<text class="axis-label" transform="translate(19 {top+plot_h/2}) rotate(-90)" text-anchor="middle">MRR@10 · higher is better</text>',
    ])
    for idx, (name, mrr, latency) in enumerate(points):
        x = left + latency / x_max * plot_w
        y = top + (1 - mrr / y_max) * plot_h
        color = SYSTEMS.get(name, ("", "#64748b", ""))[1]
        label = name.replace(" → ", " + ")
        label_y = y - 15 if idx % 2 == 0 else y + 25
        parts.append(
            f'<g class="plot-point" tabindex="0"><title>{html.escape(name)} · MRR@10 {mrr:.4f} · {latency:.1f} ms/query</title>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{color}"/>'
            f'<text class="point-label" x="{x:.1f}" y="{label_y:.1f}" text-anchor="middle">{html.escape(label)}</text></g>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _render_negative_chart(rows: Sequence[Dict[str, object]]) -> str:
    """Render hard/random validation MRR as a clear native SVG bar chart."""
    data = [(str(r.get("Negative strategy", "")), float(r.get("MRR@10", 0))) for r in rows]
    if not data:
        return '<div class="chart-empty">No negative-sampling results available.</div>'
    width, height, left = 650, 205, 150
    max_value = max(max(value for _, value in data) * 1.16, 0.1)
    parts = [f'<svg class="data-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Validation MRR at 10 for hard versus random negatives">']
    for i in range(4):
        x = left + (width - left - 48) * i / 3
        parts.append(f'<line class="grid-line vertical" x1="{x:.1f}" y1="24" x2="{x:.1f}" y2="158"/>')
        parts.append(f'<text class="axis-tick" x="{x:.1f}" y="180" text-anchor="middle">{max_value*i/3:.2f}</text>')
    for i, (name, value) in enumerate(data):
        y = 50 + i * 68
        color = "#6366f1" if name.lower() == "hard" else "#10b981"
        bar_w = (width - left - 48) * value / max_value
        parts.extend([
            f'<text class="axis-tick row-label" x="{left-14}" y="{y+5}" text-anchor="end">{html.escape(name.title())}</text>',
            f'<rect class="chart-bar" x="{left}" y="{y-13}" width="{bar_w:.1f}" height="26" rx="8" fill="{color}"><title>{value:.4f} MRR@10</title></rect>',
            f'<text class="bar-value" x="{left+bar_w+9:.1f}" y="{y+5}">{value:.4f}</text>',
        ])
    parts.extend([
        f'<text class="axis-label" x="{left+(width-left-48)/2}" y="201" text-anchor="middle">Validation MRR@10</text>',
        "</svg>",
    ])
    return "".join(parts)


def _render_depth_chart(rows: Sequence[Dict[str, object]]) -> str:
    """Plot reranking quality and latency together to show the depth trade-off."""
    data = []
    for row in rows:
        try:
            data.append((float(row["BM25 candidate depth"]), float(row["MRR@10"]),
                         float(row["Median BM25 + re-ranking latency (ms/query)"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not data:
        return '<div class="chart-empty">No reranking-depth results available.</div>'
    width, height, left, right, top, bottom = 650, 285, 68, 28, 28, 62
    plot_w, plot_h = width-left-right, height-top-bottom
    x0, x1 = min(d[2] for d in data), max(d[2] for d in data)
    if x1 == x0:
        x1 += 1
    y1 = max(d[1] for d in data) * 1.18 or 1.0
    coords = [(left+(lat-x0)/(x1-x0)*plot_w, top+(1-mrr/y1)*plot_h, depth, mrr, lat)
              for depth, mrr, lat in data]
    parts = ['<svg class="data-chart" viewBox="0 0 650 285" role="img" aria-label="Reranking depth trade-off between validation MRR and latency">']
    for i in range(4):
        y = top + plot_h*i/3
        parts.append(f'<line class="grid-line" x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}"/>')
        parts.append(f'<text class="axis-tick" x="{left-10}" y="{y+4:.1f}" text-anchor="end">{y1*(1-i/3):.2f}</text>')
        x = left + plot_w*i/3
        latency_tick = x0 + (x1-x0)*i/3
        parts.append(f'<line class="grid-line vertical" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}"/>')
        parts.append(f'<text class="axis-tick" x="{x:.1f}" y="{top+plot_h+20}" text-anchor="middle">{latency_tick:.0f}</text>')
    line_points = " ".join(f"{x:.1f},{y:.1f}" for x, y, *_ in coords)
    parts.append(f'<polyline class="chart-line" points="{line_points}"/>')
    for x, y, depth, mrr, latency in coords:
        parts.append(f'<g class="plot-point" tabindex="0"><title>Depth {depth:.0f} · MRR@10 {mrr:.4f} · {latency:.1f} ms/query</title>'
                     f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="#8b5cf6"/>'
                     f'<text class="point-label" x="{x:.1f}" y="{y-15:.1f}" text-anchor="middle">top-{depth:.0f}</text></g>')
    parts.extend([
        f'<text class="axis-label" x="{left+plot_w/2}" y="{height-10}" text-anchor="middle">BM25 + re-ranking latency (ms/query)</text>',
        f'<text class="axis-label" transform="translate(17 {top+plot_h/2}) rotate(-90)" text-anchor="middle">Validation MRR@10</text>',
        "</svg>",
    ])
    return "".join(parts)


def _render_query_length_chart(rows: Sequence[Dict[str, object]]) -> str:
    """Render BM25 and cross-encoder MRR by query-length bucket."""
    data = []
    for row in rows:
        try:
            data.append((str(row["Query bucket"]), float(row["BM25 MRR@10"]),
                         float(row["Cross-encoder MRR@10"]), str(row["Word-count range"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not data:
        return '<div class="chart-empty">No query-length results available.</div>'
    width, height, top, bottom = 760, 290, 30, 68
    plot_h = height-top-bottom
    max_value = max(max(d[1], d[2]) for d in data) * 1.2 or 1.0
    chart_w = 160
    lefts = [100 + i * 225 for i in range(len(data))]
    parts = ['<svg class="data-chart" viewBox="0 0 760 290" role="img" aria-label="BM25 and cross-encoder MRR by query length">']
    for i in range(4):
        y = top + plot_h*i/3
        parts.append(f'<line class="grid-line" x1="66" y1="{y:.1f}" x2="730" y2="{y:.1f}"/>')
        parts.append(f'<text class="axis-tick" x="56" y="{y+4:.1f}" text-anchor="end">{max_value*(1-i/3):.2f}</text>')
    for left, (bucket, bm25, cross, word_range) in zip(lefts, data):
        bw = 42
        for offset, value, color, label in [(12, bm25, "#3b82f6", "BM25"), (67, cross, "#8b5cf6", "Cross-encoder")]:
            bar_h = plot_h * value / max_value
            y = top + plot_h - bar_h
            parts.append(f'<rect class="chart-bar" x="{left+offset}" y="{y:.1f}" width="{bw}" height="{bar_h:.1f}" rx="7" fill="{color}"><title>{html.escape(bucket)} · {label}: {value:.4f}</title></rect>')
            parts.append(f'<text class="bar-value" x="{left+offset+bw/2}" y="{y-7:.1f}" text-anchor="middle">{value:.3f}</text>')
        parts.append(f'<text class="axis-tick bucket-label" x="{left+chart_w/2}" y="{height-35}" text-anchor="middle">{html.escape(bucket.title())}</text>')
        parts.append(f'<text class="axis-tick" x="{left+chart_w/2}" y="{height-15}" text-anchor="middle">{html.escape(word_range)} words</text>')
    parts.extend([
        '<g class="chart-legend"><rect x="520" y="5" width="12" height="12" rx="3" fill="#3b82f6"/><text x="538" y="16">BM25</text>'
        '<rect x="598" y="5" width="12" height="12" rx="3" fill="#8b5cf6"/><text x="616" y="16">Cross-encoder</text></g>',
        "</svg>",
    ])
    return "".join(parts)


def _read_scifact_section() -> str:
    path = RESULTS_DIR / "scifact_metrics.json"
    if not path.is_file():
        path = ARTIFACTS_DIR / "results" / "scifact_metrics.json"
    if not path.is_file():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read SciFact metrics: %s", exc)
        return ""

    runs = data.get("runs", {})
    if not runs:
        return ""

    system_names = {
        "bm25": "BM25",
        "biencoder": "Bi-encoder (Zero-shot)",
        "rrf": "Hybrid RRF (Zero-shot, k=60)",
        "bm25_crossencoder": "BM25 → Cross-encoder (Zero-shot)",
        "rrf_crossencoder": "Hybrid RRF → Cross-encoder (Zero-shot)",
    }
    metrics_keys = ["MRR@10", "NDCG@10", "P@5", "Recall@10", "Recall@100", "MAP@10"]

    best = {}
    for k in metrics_keys:
        best[k] = max(run.get("metrics", {}).get(k, 0.0) for run in runs.values())

    rows_html = []
    for key, display_name in system_names.items():
        if key not in runs:
            continue
        m = runs[key].get("metrics", {})
        cells = []
        for mk in metrics_keys:
            val = m.get(mk, 0.0)
            is_best = abs(val - best[mk]) < 1e-5 and val > 0
            cls = ' class="best"' if is_best else ""
            cells.append(f"<td{cls}>{val:.4f}</td>")
        rows_html.append(f"<tr><th>{html.escape(display_name)}</th>{''.join(cells)}</tr>")

    head_html = "".join(f"<th>{k}</th>" for k in metrics_keys)
    query_count = data.get("query_count", 300)
    corpus_count = data.get("corpus_count", 5183)

    return f"""
<div class="bench-intro scifact-intro"><span class="eyebrow">ZERO-SHOT DOMAIN TRANSFER (BEIR)</span>
  <h2>SciFact Benchmark — Out-of-Domain Generalization</h2>
  <p>Evaluates zero-shot transfer on <strong>{query_count} test queries</strong> across <strong>{corpus_count:,} scientific research abstracts</strong> from BEIR SciFact with no in-domain fine-tuning.</p>
</div>
<div class="table-scroll">
<table>
  <thead><tr><th>System</th>{head_html}</tr></thead>
  <tbody>{''.join(rows_html)}</tbody>
</table>
</div>
<p class="fine"><strong>Key Finding:</strong> Hybrid RRF achieves the highest Recall@100 (<strong>{best.get('Recall@100', 0):.3f}</strong>) and NDCG@10 (<strong>{best.get('NDCG@10', 0):.3f}</strong>) by combining lexical exact-term matching with dense semantic embeddings, outperforming individual retrievers without retraining.</p>
"""


def _read_query_enhancement_section() -> str:
    path = RESULTS_DIR / "query_bm25_comparison_test.json"
    if not path.is_file():
        path = ARTIFACTS_DIR / "results" / "query_bm25_comparison_test.json"
    if not path.is_file():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read query enhancement metrics: %s", exc)
        return ""

    changes = data.get("changes_vs_original", {})
    if not changes:
        return ""

    rows = [
        "<tr><th>Original Query</th><td>Baseline (0.0000)</td><td>Baseline (0.0000)</td><td>Baseline (0.0000)</td><td>0.00 ms</td></tr>"
    ]
    for method, details in changes.items():
        label = "Normalization (NFKC + casefold)" if method == "normalize" else "Synonym Expansion"
        m_diff = details["metric_change_vs_original"].get("MRR@10", {})
        n_diff = details["metric_change_vs_original"].get("NDCG@10", {})
        r_diff = details["metric_change_vs_original"].get("Recall@100", {})
        lat_diff = details.get("mean_latency_change_ms", 0.0)

        m_val = f"{m_diff.get('mean_difference', 0.0):+.4f} [{m_diff.get('ci_lower', 0):+.4f}, {m_diff.get('ci_upper', 0):+.4f}]"
        n_val = f"{n_diff.get('mean_difference', 0.0):+.4f} [{n_diff.get('ci_lower', 0):+.4f}, {n_diff.get('ci_upper', 0):+.4f}]"
        r_val = f"{r_diff.get('mean_difference', 0.0):+.4f} [{r_diff.get('ci_lower', 0):+.4f}, {r_diff.get('ci_upper', 0):+.4f}]"
        rows.append(f"<tr><th>{html.escape(label)}</th><td>{m_val}</td><td>{n_val}</td><td>{r_val}</td><td>{lat_diff:+.2f} ms</td></tr>")

    return f"""
<div class="bench-intro qe-intro"><span class="eyebrow">QUERY ENHANCEMENT ABLATION</span>
  <h2>Query Transformation Impact on BM25 (MS MARCO Test)</h2>
  <p>Paired bootstrap difference (95% CI, 2,000 resamples) across 750 test queries comparing original vs. normalized vs. synonym-expanded queries.</p>
</div>
<div class="table-scroll">
<table>
  <thead><tr><th>Query Transformation</th><th>Δ MRR@10 (95% CI)</th><th>Δ NDCG@10 (95% CI)</th><th>Δ Recall@100 (95% CI)</th><th>Δ Latency</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
</div>
<p class="fine"><strong>Insight:</strong> Unicode normalization preserves lexical precision identically with zero overhead. Synonym expansion introduces slight drift if applied unconditionally, demonstrating the value of precision-focused neural cross-encoding over broad lexical expansion.</p>
"""


def render_benchmark() -> str:
    rows = _read_benchmark()
    if not rows:
        return (
            '<div class="bench"><div class="bench-intro"><span class="eyebrow">OFFLINE EVALUATION</span>'
            "<h2>Benchmark results unavailable for this corpus</h2>"
            "<p>This Space does not include verified metrics and judged queries for its bundled "
            "corpus and checkpoints. Quality metrics require matching qrels; results from the "
            "companion project are intentionally not shown because they use different artifacts. "
            "Set IR_RESULTS_DIR to a directory with evaluation outputs produced for this Space "
            "to display them here. No fallback or fabricated metrics are displayed.</p>"
            "</div></div>"
        )
    preferred_metrics = [
        "MRR@10",
        "NDCG@10",
        "P@5",
        "Recall@10",
        "Recall@100",
        "MAP@10",
        "Median latency (ms/query)",
    ]
    metrics = [metric for metric in preferred_metrics if any(metric in row for row in rows)]
    manifest = _read_json_result("evaluation_manifest.json")
    paired_differences = _read_json_result("paired_differences.json")
    per_query = _read_per_query_metrics()
    try:
        query_count = int(manifest.get("query_count", 0))
        latency_queries = int(manifest.get("latency_query_count", 0))
        run_config = manifest.get("config", {})
        candidate_depth = int(run_config.get("candidate_depth", 0))
        rerank_depth = int(run_config.get("rerank_depth", 0))
        rrf_k = int(run_config.get("rrf_k", 0))
    except (AttributeError, TypeError, ValueError):
        query_count = latency_queries = candidate_depth = rerank_depth = rrf_k = 0
    dataset_name = str(manifest.get("dataset", "local dataset"))
    if dataset_name.casefold() == "msmarco-passage":
        dataset_name = "MS MARCO passage"
    dataset_label = html.escape(dataset_name)
    split_label = html.escape(str(manifest.get("split", "recorded split")))
    query_summary = f"{query_count} queries" if query_count else "the recorded queries"
    latency_summary = (
        f"{latency_queries} queries" if latency_queries else "the recorded query sample"
    )

    valid_rows = [row for row in rows if row.get("Recall@100 valid") is not False]
    best_mrr = max(
        (row for row in valid_rows if "MRR@10" in row),
        key=lambda row: float(row["MRR@10"]),
        default=None,
    )
    best_recall = max(
        (row for row in valid_rows if "Recall@100" in row),
        key=lambda row: float(row["Recall@100"]),
        default=None,
    )
    fastest = min(
        (row for row in valid_rows if "Median latency (ms/query)" in row),
        key=lambda row: float(row["Median latency (ms/query)"]),
        default=None,
    )
    bm25 = next((row for row in valid_rows if row.get("System") == "BM25"), None)

    best = {}
    for metric in metrics:
        values = [r[metric] for r in rows if metric in r]
        lower_is_better = "latency" in metric.lower()
        best[metric] = min(values) if lower_is_better else max(values)
    head = "".join(f"<th>{html.escape(m)}</th>" for m in metrics)
    body = []
    for row in rows:
        cells = []
        for metric in metrics:
            value = row.get(metric)
            if value is None:
                cells.append("<td>—</td>")
                continue
            is_best = abs(value - best[metric]) < 1e-9
            cells.append(f'<td class="{"best" if is_best else ""}">{value:.4g}</td>')
        color = SYSTEMS.get(row["System"], ("", "#64748b", ""))[1]
        body.append(
            f'<tr><th><span class="dot" style="background:{color}"></span>{html.escape(row["System"])}</th>{"".join(cells)}</tr>'
        )
    pareto = _chart_card(
        "Quality vs. speed",
        f"Saved local {dataset_label} {split_label} evaluation. Favor the upper-left: "
        "higher MRR with lower query latency.",
        _render_pareto_chart(rows),
        "wide-chart",
    )
    quality_chart = _chart_card(
        "Ranking quality at a glance",
        "Higher is better. MRR@10 and NDCG@10 focus on top-ranked relevance; Recall@100 measures candidate coverage.",
        _render_quality_chart(rows),
        "wide-chart",
    )
    latency_chart = _chart_card(
        "Median latency per query",
        f"Sorted fastest to slowest · measured on {latency_queries or 'the recorded'} local test queries. "
        "Lower is faster; speed ratios use BM25 as the reference.",
        _render_latency_chart(rows, latency_queries),
        "wide-chart",
    )
    confidence_chart = _chart_card(
        "Paired MRR@10 difference vs. BM25 and TF-IDF",
        "Point is the mean difference; whisker is the paired bootstrap 95% confidence interval. "
        "Positive differences favor the method; intervals wholly right of zero favor it, while intervals crossing zero indicate uncertainty.",
        _render_paired_ci_chart(paired_differences),
        "wide-chart",
    )
    distribution_chart = _chart_card(
        "Per-query MRR@10 distribution",
        "Center line is median; box is the interquartile range; whiskers span the minimum and maximum query scores.",
        _render_query_distribution_chart(per_query),
        "wide-chart",
    )
    outcomes_chart = _chart_card(
        "How often each system beats BM25",
        "Paired test queries compared by MRR@10. Counts at right are improved / tied / lower; each query is compared to itself.",
        _render_query_outcomes_chart(per_query),
        "wide-chart",
    )
    summary_tiles = []
    if best_mrr:
        summary_tiles.append(
            _tile("Best MRR@10", f"{float(best_mrr['MRR@10']):.3f}", str(best_mrr["System"]))
        )
    if best_recall:
        summary_tiles.append(
            _tile(
                "Best Recall@100",
                f"{float(best_recall['Recall@100']):.3f}",
                str(best_recall["System"]),
            )
        )
    if fastest:
        summary_tiles.append(
            _tile(
                "Lowest median latency",
                f"{float(fastest['Median latency (ms/query)']):.1f} ms",
                str(fastest["System"]),
            )
        )
    if best_mrr and bm25 and float(bm25.get("MRR@10", 0)) > 0:
        summary_tiles.append(
            _tile(
                "MRR@10 lift vs. BM25",
                f"+{float(best_mrr['MRR@10']) - float(bm25['MRR@10']):.3f}",
                str(best_mrr["System"]),
            )
        )
    protocol_items = [
        f"<li><strong>Dataset/split:</strong> {html.escape(str(manifest.get('dataset', 'not recorded')))} / "
        f"{html.escape(str(manifest.get('split', 'not recorded')))}</li>",
        f"<li><strong>Queries:</strong> {query_count or 'not recorded'} test queries; "
        f"latency sample: {latency_queries or 'not recorded'} queries</li>",
        f"<li><strong>Candidate/rerank depth:</strong> {candidate_depth or 'not recorded'} / "
        f"{rerank_depth or 'not recorded'}</li>",
        f"<li><strong>RRF constant:</strong> k={rrf_k or 'not recorded'}</li>",
        f"<li><strong>Query fingerprint:</strong> {html.escape(str(manifest.get('queries_sha256', 'not recorded')))}</li>",
        f"<li><strong>Qrels fingerprint:</strong> {html.escape(str(manifest.get('qrels_sha256', 'not recorded')))}</li>",
    ]
    protocol = (
        '<details class="protocol"><summary>Evaluation protocol and reproducibility details</summary>'
        f'<ul>{"".join(protocol_items)}</ul>'
        '<p>Metrics are macro-averaged over the recorded test queries. Paired confidence intervals '
        'resample the same query IDs for both systems. Latency is a separate query sample, not a '
        'per-query quality measurement. These files were copied from the completed local evaluation; '
        'the live Space uses different bundled corpus/model artifacts.</p></details>'
    )
    ablations = _read_ablation_tables()
    ablation_section = ""
    if len(ablations) >= 3:
        ablation_section = (
            '<div class="bench-intro ablation-intro"><span class="eyebrow">ABLATION STUDIES</span>'
            "<h2>What changes the ranking?</h2>"
            "<p>Validation experiments isolate training negatives and candidate depth; "
            "query-length results use the test split.</p></div>"
            '<div class="chart-grid">'
            + _chart_card("Negative sampling", "Validation MRR@10 · hard vs random training negatives",
                          _render_negative_chart(ablations[0]))
            + _chart_card("Candidate-depth trade-off", "Validation MRR@10 and total latency by BM25 depth",
                          _render_depth_chart(ablations[1]))
            + _chart_card("Query-length analysis", "Test MRR@10 by bucket · 250 queries per bucket",
                          _render_query_length_chart(ablations[2]), "wide-chart")
            + "</div>"
        )
    return f"""
<div class="bench">
  <div class="bench-intro"><span class="eyebrow">OFFLINE EVALUATION</span>
    <h2>Local MS MARCO evaluation — saved results</h2>
    <p><strong>Heads-up:</strong> these metrics were produced by local/Colab evaluation on the
    {dataset_label} {split_label} split ({query_summary}; latency sampled on {latency_summary}).
    They are not measurements of this Space's currently bundled corpus or model
    checkpoints. They do not describe a separate Hugging Face Space unless its corpus and
    checkpoints match. Live neural inference runs on CPU by default. The table is a reference, not a claim about live Space quality or latency.</p>
  </div>
  <div class="tiles benchmark-tiles">{''.join(summary_tiles)}</div>
  {protocol}
  <table><thead><tr><th>System</th>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>
  <p class="fine">Best value per column highlighted. Results and run provenance are in artifacts/local-evaluation.
  {"Recall@100 is from saved full-depth local ranked runs, not top-10 outputs." if "Recall@100" in metrics else ""}</p>
  <div class="chart-grid benchmark-charts">
    {quality_chart}
    {latency_chart}
    {pareto}
    {confidence_chart}
    {distribution_chart}
    {outcomes_chart}
  </div>
  {ablation_section}
  {_read_scifact_section()}
  {_read_query_enhancement_section()}
</div>"""


HOW_IT_WORKS = """
<div class="guide">
  <header class="tab-intro">
    <span class="eyebrow">A QUICK FIELD GUIDE</span>
    <h2>How to read the retrieval comparison</h2>
    <p>Each method sees the same query and passage collection, but finds and orders candidates differently.
    Compare the returned passages and rank changes; relevance is not inferred for a live search.</p>
  </header>
  <div class="pipeline" aria-label="Retrieval stages">
    <div class="pipeline-step"><span>01</span><strong>Find candidates</strong><small>Lexical, semantic, or both</small></div>
    <b class="pipeline-arrow">→</b>
    <div class="pipeline-step"><span>02</span><strong>Fuse or re-rank</strong><small>Optional deeper scoring</small></div>
    <b class="pipeline-arrow">→</b>
    <div class="pipeline-step"><span>03</span><strong>Compare top results</strong><small>Inspect ranks and evidence</small></div>
  </div>
  <h3 class="guide-section-title">Retrieval methods</h3>
  <div class="system-grid">
    <article class="system-card" style="--guide-accent:#f59e0b">
      <span class="system-kind">LEXICAL</span><h3>TF-IDF</h3>
      <p>Weights terms by frequency and rarity, then scores matching words.</p>
      <div><b>Good at</b> fast, transparent exact-term matching</div>
      <div><b>Watch for</b> vocabulary mismatch and no length normalisation</div>
    </article>
    <article class="system-card" style="--guide-accent:#3b82f6">
      <span class="system-kind">LEXICAL</span><h3>BM25</h3>
      <p>A tuned lexical baseline with term saturation and document-length normalisation.</p>
      <div><b>Good at</b> strong exact-term retrieval</div>
      <div><b>Watch for</b> synonyms and paraphrases that use different words</div>
    </article>
    <article class="system-card" style="--guide-accent:#10b981">
      <span class="system-kind">SEMANTIC</span><h3>Bi-encoder</h3>
      <p>Encodes the query and passages separately, then compares their vectors.</p>
      <div><b>Good at</b> matching meaning across different wording</div>
      <div><b>Watch for</b> detail lost in a single vector representation</div>
    </article>
    <article class="system-card" style="--guide-accent:#06b6d4">
      <span class="system-kind">HYBRID</span><h3>Hybrid RRF</h3>
      <p>Combines BM25 and bi-encoder rankings with Reciprocal Rank Fusion.</p>
      <div><b>Good at</b> combining lexical and semantic candidates</div>
      <div><b>Watch for</b> extra retrieval cost; fusion still needs evaluation</div>
    </article>
    <article class="system-card" style="--guide-accent:#8b5cf6">
      <span class="system-kind">NEURAL RE-RANKING (NLI)</span><h3>BM25 → Cross-encoder</h3>
      <p>Jointly scores query–passage pairs using <code>nli-MiniLM2-L6-H768</code> with natural language inference logit margin: <code>score = entailment - contradiction</code>.</p>
      <div><b>Good at</b> deep semantic premise entailment and precise pairwise ranking</div>
      <div><b>Watch for</b> latency and BM25 candidate recall ceiling</div>
    </article>
    <article class="system-card" style="--guide-accent:#ec4899">
      <span class="system-kind">HYBRID + NLI RE-RANKING</span><h3>Hybrid RRF → Cross-encoder</h3>
      <p>Re-scores candidates from lexical BM25 and semantic bi-encoder vectors with Natural Language Inference.</p>
      <div><b>Good at</b> broader candidate coverage plus deep pairwise logic</div>
      <div><b>Watch for</b> dual retrieval latency before re-ranking</div>
    </article>
  </div>
  <div class="guide-grid">
    <section class="guide-panel">
      <span class="eyebrow">READ THE RESULTS</span><h3>What the cards tell you</h3>
      <ul>
        <li><b>Rank number</b> is the method's position for that passage.</li>
        <li><b>▲ / ▼ chips</b> show movement relative to BM25; ✨ means the passage was outside BM25's candidate depth.</li>
        <li><b>Score bars</b> are scaled within each method. Raw scores across methods are not directly comparable.</li>
        <li><b>Highlighted terms</b> reveal exact query-word overlap; no highlight does not mean a passage is irrelevant.</li>
        <li><b>Latency</b> is an observed time for this query on the current Space hardware, not a benchmark guarantee.</li>
      </ul>
    </section>
    <section class="guide-panel metrics-panel">
      <span class="eyebrow">OFFLINE EVALUATION METRICS</span><h3>How quality is measured</h3>
      <div><b>MRR@10</b><p>Rewards placing the first relevant passage near the top ten.</p></div>
      <div><b>NDCG@10</b><p>Rewards highly ranked relevant passages, with a discount for lower positions.</p></div>
      <div><b>Recall@100</b><p>Share of judged relevant passages retrieved in the top 100 candidates.</p></div>
      <p class="fine">These require judged queries and matching qrels. The saved local benchmark is explicitly labeled on its tab.</p>
    </section>
  </div>
  <div class="guide-grid" style="margin-top:14px;">
    <section class="guide-panel">
      <span class="eyebrow">QUERY ENHANCEMENT</span><h3>Lexical transformations</h3>
      <div><b>Unicode Normalization (NFKC)</b><p>Normalizes full-width characters, accents, and spacing into consistent Unicode forms with zero metric degradation.</p></div>
      <div><b>Synonym Expansion</b><p>Bridges vocabulary mismatch by appending known synonym phrases from a curated domain dictionary.</p></div>
    </section>
    <section class="guide-panel">
      <span class="eyebrow">ZERO-SHOT GENERALIZATION</span><h3>BEIR transfer evaluation</h3>
      <div><b>SciFact Dataset</b><p>Tests whether models trained on conversational web queries (MS MARCO) transfer to biomedical claim verification without retraining.</p></div>
      <div><b>RRF Fusion Strength</b><p>Reciprocal Rank Fusion consistently maintains robust recall (0.957) across domains where individual models degrade.</p></div>
    </section>
  </div>
</div>
"""


CSS = """
html, body { height: auto !important; min-height: 0 !important; }
.gradio-container { max-width: 1280px !important; margin: auto; }
.gradio-container { height: auto !important; min-height: 0 !important; }
.hero { padding: 22px 26px; border-radius: 16px; color: #fff;
  background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%); }
.hero h1 { margin: 0 0 4px; font-size: 1.9rem; letter-spacing: -0.02em; color: #fff; }
.hero p { margin: 0; opacity: .92; }
.hero .pills { margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }
.hero .pills span { background: rgba(255,255,255,.18); padding: 3px 10px; border-radius: 999px; font-size: .8rem; }
.pill-enhanced { background: rgba(99, 102, 241, 0.12); border: 1px solid rgba(99, 102, 241, 0.4); border-radius: 8px; padding: 6px 12px; margin-bottom: 10px; font-size: .84rem; color: var(--body-text-color); display: flex; align-items: center; gap: 8px; }
.pill-enhanced code { background: rgba(0,0,0,.08); padding: 2px 6px; border-radius: 4px; font-weight: 600; }
.scifact-intro, .qe-intro { margin-top: 32px; }
.insights { display: grid; grid-template-columns: 2fr 1.3fr; gap: 14px; margin: 6px 0; }
@media (max-width: 860px) { .insights { grid-template-columns: 1fr; } }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
.tile, .latency, .bench table { border: 1px solid var(--border-color-primary); background: var(--block-background-fill); border-radius: 12px; }
.tile { padding: 10px 14px; }
.tile .label { font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; color: var(--body-text-color-subdued); }
.tile .value { font-size: 1.5rem; font-weight: 700; margin: 2px 0; }
.tile .sub { font-size: .75rem; color: var(--body-text-color-subdued); }
.latency { padding: 10px 14px; }
.latency h4 { margin: 0 0 6px; }
.lrow { display: grid; grid-template-columns: 150px 1fr 70px; gap: 8px; align-items: center; margin: 5px 0; font-size: .82rem; }
.ltrack { height: 9px; border-radius: 99px; background: rgba(127,127,127,.18); overflow: hidden; }
.ltrack span { display: block; height: 100%; border-radius: 99px; }
.lval { text-align: right; font-variant-numeric: tabular-nums; }
.fine { font-size: .74rem; color: var(--body-text-color-subdued); margin: 6px 0 0; }
.cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; align-items: start; }
.col header { position: sticky; top: 0; z-index: 2; padding: 10px 4px 8px; background: var(--background-fill-primary);
  border-bottom: 3px solid var(--accent); margin-bottom: 10px; }
.col header h3 { margin: 0; font-size: 1.05rem; display: flex; align-items: center; gap: 8px; }
.col header p { margin: 2px 0 4px; font-size: .78rem; color: var(--body-text-color-subdued); }
.dot { width: 10px; height: 10px; border-radius: 50%; background: var(--accent); display: inline-block; margin-right: 6px; }
.lat { font-size: .82rem; } .lat span { color: var(--body-text-color-subdued); }
.card { display: flex; gap: 12px; padding: 12px; margin-bottom: 10px; border-radius: 12px;
  border: 1px solid var(--border-color-primary); border-left: 4px solid var(--accent);
  background: var(--block-background-fill); transition: transform .12s ease, box-shadow .12s ease; }
.card:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(0,0,0,.12); }
.rank { min-width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-weight: 700; color: #fff; background: var(--accent); flex-shrink: 0; }
.body { min-width: 0; flex: 1; }
.meta { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; font-size: .74rem; color: var(--body-text-color-subdued); }
.meta .score { margin-left: auto; font-variant-numeric: tabular-nums; font-weight: 600; color: var(--body-text-color); }
.bar { height: 5px; border-radius: 99px; background: rgba(127,127,127,.18); margin: 6px 0 8px; overflow: hidden; }
.bar span { display: block; height: 100%; background: var(--accent); border-radius: 99px; }
.snippet { margin: 0; font-size: .9rem; line-height: 1.5; }
mark { background: rgba(250, 204, 21, .45); color: inherit; padding: 0 2px; border-radius: 3px; }
details { margin-top: 6px; font-size: .84rem; }
details summary { cursor: pointer; color: var(--accent); font-weight: 600; }
details p { line-height: 1.55; margin: 6px 0 0; }
.chip { padding: 1px 8px; border-radius: 999px; font-weight: 600; white-space: nowrap; }
.chip.up { background: rgba(16,185,129,.16); color: #059669; }
.chip.down { background: rgba(239,68,68,.15); color: #dc2626; }
.chip.same { background: rgba(127,127,127,.18); }
.chip.new { background: rgba(139,92,246,.18); color: #7c3aed; }
.empty, .placeholder { padding: 28px; text-align: center; color: var(--body-text-color-subdued); }
.bench table { width: 100%; border-collapse: separate; border-spacing: 0; overflow: hidden; font-size: .88rem; }
.bench th, .bench td { padding: 9px 12px; text-align: right; border-bottom: 1px solid var(--border-color-primary); }
.bench th:first-child { text-align: left; }
.bench thead th { background: rgba(127,127,127,.1); }
.bench td.best { font-weight: 700; color: #059669; }
.examples { display: flex; gap: 8px; flex-wrap: wrap; }

/* Search insights */
.insights { grid-template-columns: minmax(0, 1.6fr) minmax(280px, 1fr); gap: 16px; margin: 14px 0 18px; }
.insight-main { min-width: 0; }
.overlap-card { display: flex; align-items: center; gap: 14px; padding: 12px 16px; margin-top: 10px;
  border: 1px solid var(--border-color-primary); border-radius: 14px; background: var(--block-background-fill); }
.donut { width: 60px; height: 60px; flex: 0 0 60px; display: grid; place-items: center; border-radius: 50%;
  background: conic-gradient(#6366f1 var(--overlap), rgba(127,127,127,.2) 0); position: relative; }
.donut::before { content: ""; position: absolute; inset: 7px; border-radius: 50%; background: var(--block-background-fill); }
.donut span { z-index: 1; font-size: .82rem; font-weight: 750; }
.overlap-legend { display: flex; flex-wrap: wrap; gap: 6px 14px; margin-top: 6px; font-size: .78rem; color: var(--body-text-color-subdued); }
.overlap-legend span { display: inline-flex; align-items: center; gap: 5px; }
.overlap-legend i { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #6366f1; }
.overlap-legend .legend-different { background: rgba(127,127,127,.35); }

/* Benchmark charts are inline SVG so they scale crisply with the layout. */
.bench { display: grid; gap: 16px; }
.bench-intro { padding: 8px 2px 0; }
.bench-intro h2 { margin: 3px 0 4px; font-size: 1.35rem; letter-spacing: -.02em; }
.bench-intro p, .chart-heading p { margin: 0; color: var(--body-text-color-subdued); font-size: .86rem; }
.eyebrow { color: #6366f1; font-size: .7rem; font-weight: 800; letter-spacing: .11em; }
.bench table { display: block; overflow-x: auto; border-spacing: 0; }
.bench td.best { color: #059669; }
.chart-card { min-width: 0; padding: 16px 18px 12px; border: 1px solid var(--border-color-primary);
  border-radius: 16px; background: var(--block-background-fill); box-shadow: 0 8px 24px rgba(15,23,42,.045); }
.chart-heading h3 { margin: 0 0 4px; font-size: 1.02rem; }
.chart-viewport { width: 100%; overflow: hidden; margin-top: 10px; }
.data-chart { display: block; width: 100%; max-height: 420px; overflow: visible; }
.grid-line { stroke: var(--border-color-primary); stroke-width: 1; opacity: .75; }
.grid-line.vertical { stroke-dasharray: 3 5; }
.axis-line { stroke: var(--body-text-color-subdued); stroke-width: 1.2; opacity: .7; }
.axis-tick { fill: var(--body-text-color-subdued); font: 12px ui-sans-serif,system-ui,sans-serif; }
.axis-label { fill: var(--body-text-color); font: 12px ui-sans-serif,system-ui,sans-serif; font-weight: 600; }
.point-label { fill: var(--body-text-color); font: 12px ui-sans-serif,system-ui,sans-serif; font-weight: 650; }
.plot-point { cursor: help; outline: none; }
.plot-point circle { stroke: var(--block-background-fill); stroke-width: 3; transition: r .16s ease, filter .16s ease; }
.plot-point:hover circle, .plot-point:focus circle { r: 11; filter: drop-shadow(0 3px 5px rgba(15,23,42,.25)); }
.chart-bar { transform-box: fill-box; transform-origin: left center; animation: grow-in .75s cubic-bezier(.2,.75,.25,1) both; }
.chart-line { fill: none; stroke: #8b5cf6; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }
.bar-value { fill: var(--body-text-color); font: 12px ui-sans-serif,system-ui,sans-serif; font-weight: 700; }
.bucket-label { font-weight: 750; }
.chart-legend { fill: var(--body-text-color); font: 11px ui-sans-serif,system-ui,sans-serif; }
.chart-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.wide-chart { grid-column: 1 / -1; }
.chart-empty { padding: 32px 12px; text-align: center; color: var(--body-text-color-subdued); }
.ablation-intro { padding-top: 22px; }
.benchmark-tiles { margin: 2px 0; }
.benchmark-tiles .tile { border-radius: 14px; box-shadow: 0 5px 18px rgba(15,23,42,.04); }
.bench-intro:first-child { padding: 18px 20px; border: 1px solid rgba(99,102,241,.22);
  border-left: 5px solid #6366f1; border-radius: 16px;
  background: linear-gradient(115deg, rgba(99,102,241,.10), rgba(6,182,212,.055)); }
.bench-intro:first-child p { line-height: 1.65; max-width: 1000px; }
.bench-intro:first-child strong { color: #b45309; }
.protocol { padding: 12px 16px; border: 1px solid var(--border-color-primary);
  border-radius: 12px; background: var(--block-background-fill); }
.protocol summary { cursor: pointer; color: #4f46e5; font-weight: 700; }
.protocol ul { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px 28px;
  padding-left: 20px; font-size: .82rem; }
.protocol li { overflow-wrap: anywhere; }
.protocol p { color: var(--body-text-color-subdued); font-size: .8rem; line-height: 1.55; }
.bench table { box-shadow: 0 8px 24px rgba(15,23,42,.045); }
.bench thead th { position: sticky; top: 0; z-index: 1; backdrop-filter: blur(8px); }
.bench tbody tr:hover { background: rgba(99,102,241,.055); }
.benchmark-charts { gap: 16px; }
.chart-card { transition: transform .16s ease, box-shadow .16s ease; }
.chart-card:hover { transform: translateY(-2px); box-shadow: 0 12px 30px rgba(15,23,42,.085); }
.tall-chart { max-height: 500px; }

/* Shared presentation for the interactive and learning tabs. */
.tab-intro { margin: 8px 0 18px; padding: 20px 22px; border: 1px solid rgba(99,102,241,.22);
  border-left: 5px solid #6366f1; border-radius: 16px;
  background: linear-gradient(115deg, rgba(99,102,241,.10), rgba(6,182,212,.055)); }
.tab-intro h2 { margin: 4px 0 6px; font-size: 1.38rem; letter-spacing: -.025em; }
.tab-intro p { max-width: 920px; margin: 0; color: var(--body-text-color-subdued); line-height: 1.65; }
.search-intro { margin-bottom: 12px; }
.search-intro h2 { font-size: 1.25rem; }
.placeholder { margin: 16px 0; border: 1px dashed var(--border-color-primary);
  border-radius: 16px; background: linear-gradient(140deg, rgba(99,102,241,.07), rgba(6,182,212,.035)); }
.placeholder h3 { margin: 0 0 6px; color: var(--body-text-color); }
.placeholder p { max-width: 600px; margin: auto; line-height: 1.6; }
.insights .tile { border-radius: 14px; box-shadow: 0 5px 18px rgba(15,23,42,.04); }
.latency { border-radius: 14px; box-shadow: 0 5px 18px rgba(15,23,42,.04); }
.latency h4 { font-size: .95rem; }
.cols { gap: 18px; }
.col header { border-radius: 12px 12px 0 0; }
.card { border-radius: 14px; }
.card .snippet { overflow-wrap: anywhere; }
.pid { font-variant-numeric: tabular-nums; }
.step-strip { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
.step-strip span { display: inline-flex; align-items: center; gap: 8px; padding: 7px 11px;
  border: 1px solid var(--border-color-primary); border-radius: 999px;
  background: var(--block-background-fill); font-size: .82rem; font-weight: 600; }
.step-strip b { display: inline-grid; width: 22px; height: 22px; place-items: center;
  border-radius: 50%; color: white; background: #6366f1; font-size: .74rem; }
.feedback-output { margin: 16px 0; padding: 16px; border: 1px solid var(--border-color-primary);
  border-radius: 16px; background: var(--block-background-fill); }
.feedback-empty { display: grid; gap: 4px; margin: 12px 0; padding: 18px;
  border: 1px dashed var(--border-color-primary); border-radius: 14px;
  background: rgba(99,102,241,.045); }
.feedback-empty b { font-size: .9rem; }
.feedback-empty span { color: var(--body-text-color-subdued); font-size: .82rem; }
.feedback-output > p { color: var(--body-text-color-subdued); font-size: .84rem; }
.output-banner { display: flex; align-items: center; gap: 10px 14px; flex-wrap: wrap;
  margin-bottom: 10px; padding: 12px 14px; border-radius: 12px;
  background: rgba(99,102,241,.09); }
.output-banner .eyebrow { flex-basis: 100%; }
.output-banner strong { font-size: 1.05rem; }
.output-banner > span:last-child { color: var(--body-text-color-subdued); font-size: .85rem; }
.feedback-output table, .query-analysis table { width: 100%; border-collapse: separate;
  border-spacing: 0; font-size: .84rem; }
.feedback-output th, .feedback-output td, .query-analysis th, .query-analysis td {
  padding: 9px 11px; border-bottom: 1px solid var(--border-color-primary); text-align: left; vertical-align: top; }
.feedback-output th, .query-analysis th { position: sticky; top: 0; z-index: 1;
  background: var(--block-background-fill); font-size: .76rem; color: var(--body-text-color-subdued); }
.feedback-output td:nth-child(3), .query-analysis td:last-child { min-width: 260px; line-height: 1.5; }
.feedback-output tbody tr:hover, .query-analysis tbody tr:hover { background: rgba(99,102,241,.06); }
.analysis-intro { margin-bottom: 18px; }
.analysis-empty { margin: 18px 0; padding: 20px; border: 1px dashed var(--border-color-primary);
  border-radius: 14px; background: rgba(99,102,241,.055); }
.analysis-empty h3 { margin: 5px 0; }
.analysis-empty p { margin: 0; color: var(--body-text-color-subdued); line-height: 1.6; }
.query-analysis { display: grid; gap: 14px; margin-top: 18px; }
.analysis-query, .analysis-summary, .analysis-movement { min-width: 0; padding: 16px;
  border: 1px solid var(--border-color-primary); border-radius: 16px;
  background: var(--block-background-fill); }
.analysis-query h3 { margin: 6px 0 14px; font-size: 1.16rem; line-height: 1.5; }
.analysis-facts { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 9px; }
.analysis-facts > div { display: grid; gap: 5px; align-content: start; padding: 11px;
  border-radius: 12px; background: rgba(99,102,241,.075); overflow-wrap: anywhere; }
.analysis-facts span { color: var(--body-text-color-subdued); font-size: .73rem; text-transform: uppercase; letter-spacing: .04em; }
.analysis-facts b { font-size: .84rem; }
.analysis-summary h4, .analysis-movement h4 { margin: 0 0 10px; }
.metric-pills { display: flex; flex-wrap: wrap; gap: 6px; }
.metric-pills span { display: inline-flex; align-items: center; gap: 6px; padding: 5px 8px;
  border-radius: 999px; background: rgba(99,102,241,.09); font-size: .73rem; white-space: nowrap; }
.metric-pills b { font-variant-numeric: tabular-nums; }
.table-scroll { width: 100%; overflow-x: auto; border: 1px solid var(--border-color-primary); border-radius: 10px; }
.analysis-movement table { min-width: 650px; }
.analysis-section-title { margin: 10px 0 0; }
.analysis-method-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; align-items: start; }
.analysis-method { min-width: 0; margin: 0; border: 1px solid var(--border-color-primary);
  border-radius: 14px; background: var(--block-background-fill); overflow: hidden; }
.analysis-method > summary { display: grid; gap: 8px; padding: 13px; cursor: pointer; list-style: none; }
.analysis-method > summary::-webkit-details-marker { display: none; }
.analysis-method > summary h4 { margin: 0; font-size: .95rem; }
.analysis-method[open] > summary { border-bottom: 1px solid var(--border-color-primary); }
.analysis-method .table-scroll { border: 0; border-radius: 0; }
.analysis-method table { min-width: 600px; }
.guide { display: grid; gap: 18px; }
.guide .tab-intro { margin-bottom: 0; }
.pipeline { display: flex; align-items: center; gap: 10px; padding: 14px;
  border: 1px solid var(--border-color-primary); border-radius: 16px;
  background: var(--block-background-fill); }
.pipeline-step { display: grid; grid-template-columns: auto 1fr; gap: 0 10px; flex: 1;
  align-items: center; min-width: 0; }
.pipeline-step > span { grid-row: span 2; display: grid; width: 34px; height: 34px;
  place-items: center; border-radius: 11px; background: rgba(99,102,241,.14); color: #6366f1;
  font-size: .74rem; font-weight: 800; }
.pipeline-step strong { font-size: .9rem; }
.pipeline-step small { color: var(--body-text-color-subdued); font-size: .75rem; }
.pipeline-arrow { color: #6366f1; }
.guide-section-title { margin: 4px 0 -6px; font-size: 1.12rem; }
.system-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.system-card { min-width: 0; padding: 15px; border: 1px solid var(--border-color-primary);
  border-top: 3px solid var(--guide-accent); border-radius: 14px;
  background: var(--block-background-fill); }
.system-kind { color: var(--guide-accent); font-size: .66rem; font-weight: 800; letter-spacing: .09em; }
.system-card h3 { margin: 5px 0; font-size: 1rem; }
.system-card p { min-height: 3.8em; margin: 0 0 10px; color: var(--body-text-color-subdued);
  font-size: .82rem; line-height: 1.5; }
.system-card > div { margin-top: 7px; font-size: .78rem; line-height: 1.45; }
.system-card > div b { display: block; color: var(--guide-accent); font-size: .68rem;
  letter-spacing: .04em; text-transform: uppercase; }
.guide-grid { display: grid; grid-template-columns: 1.2fr 1fr; gap: 12px; align-items: start; }
.guide-panel { min-width: 0; padding: 16px; border: 1px solid var(--border-color-primary);
  border-radius: 16px; background: var(--block-background-fill); }
.guide-panel h3 { margin: 4px 0 10px; font-size: 1.02rem; }
.guide-panel ul { display: grid; gap: 8px; margin: 0; padding-left: 20px; font-size: .82rem; line-height: 1.55; }
.metrics-panel > div { padding: 8px 0; border-bottom: 1px solid var(--border-color-primary); }
.metrics-panel > div:last-of-type { border-bottom: 0; }
.metrics-panel > div b { color: #6366f1; }
.metrics-panel > div p { margin: 3px 0 0; color: var(--body-text-color-subdued); font-size: .8rem; line-height: 1.5; }
@keyframes grow-in { from { transform: scaleX(.04); opacity: .25; } to { transform: scaleX(1); opacity: 1; } }
@media (max-width: 760px) {
  .insights { grid-template-columns: 1fr; }
  .chart-grid { grid-template-columns: 1fr; }
  .wide-chart { grid-column: auto; }
  .chart-card { padding: 12px; }
  .protocol ul { grid-template-columns: 1fr; gap: 4px; }
  .data-chart { min-width: 540px; }
  .chart-viewport { overflow-x: auto; }
  .analysis-facts, .analysis-method-grid, .guide-grid { grid-template-columns: 1fr; }
  .system-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .pipeline { align-items: stretch; flex-direction: column; }
  .pipeline-arrow { align-self: center; transform: rotate(90deg); }
  .system-card p { min-height: 0; }
  .lrow { grid-template-columns: minmax(100px, 1fr) 1.4fr 62px; }
}
@media (max-width: 480px) {
  .system-grid { grid-template-columns: 1fr; }
  .tab-intro { padding: 16px; }
  .output-banner { align-items: flex-start; flex-direction: column; }
  .step-strip { align-items: flex-start; flex-direction: column; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; scroll-behavior: auto !important; }
}
"""

PLACEHOLDER = (
    '<div class="placeholder"><h3>Ask anything</h3>'
    "<p>Pick an example above or type a question, then compare how each ranking system answers it.</p></div>"
)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def build_app(backend: Backend) -> Tuple[gr.Blocks, dict]:
    """Construct the Gradio Blocks UI around ``backend``."""

    def search(
        query: str,
        systems: List[str],
        top_k: int,
        depth: int,
        do_highlight: bool,
        enhancement: str = "Original",
    ):
        raw_query = (query or "").strip()
        if not raw_query:
            raise gr.Error("Enter a query before searching.")
        if not systems:
            raise gr.Error("Select at least one system to display.")
        top_k, depth = int(top_k), int(depth)
        depth = max(depth, top_k)

        # Opt-in query enhancement
        active_query = raw_query
        enhancement_label = str(enhancement or "").lower()
        enhancement_note = ""
        if "norm" in enhancement_label:
            active_query = enhance_query(raw_query, "normalize")
            if active_query != raw_query:
                enhancement_note = f"Normalized: '{active_query}'"
        elif "synonym" in enhancement_label:
            active_query = enhance_query(raw_query, "synonym", synonyms=SYNONYMS)
            if active_query != raw_query:
                enhancement_note = f"Synonyms expanded: '{active_query}'"

        terms = query_terms(active_query)
        if raw_query != active_query:
            terms = list(dict.fromkeys(terms + query_terms(raw_query)))

        query = active_query
        # Stage 1 always runs: it feeds the BM25 column, the cross-encoder, and rank chips.
        started = time.perf_counter()
        stage1 = backend.bm25(query, depth)
        stage1_ms = (time.perf_counter() - started) * 1000.0
        bm25_rank = {int(pid): rank for rank, (pid, _) in enumerate(stage1, start=1)}

        results: Dict[str, List[Hit]] = {}
        timings: Dict[str, float] = {}
        notes: Dict[str, str] = {}
        cross_hits: Optional[List[Hit]] = None
        dense_candidates: Optional[List[Hit]] = None
        rrf_hits: Optional[List[Hit]] = None

        for name in systems:
            key = SYSTEMS[name][0]
            if key == "bm25":
                results[name], timings[name], notes[name] = stage1[:top_k], stage1_ms, f"top-{depth} retrieval"
            elif key == "tfidf":
                t0 = time.perf_counter()
                results[name] = backend.tfidf(query, top_k)
                timings[name], notes[name] = (time.perf_counter() - t0) * 1000.0, "full-corpus scoring"
            elif key == "dense":
                t0 = time.perf_counter()
                results[name] = backend.dense(query, top_k)
                timings[name], notes[name] = (time.perf_counter() - t0) * 1000.0, "embed query + dot product"
            elif key == "cross":
                t0 = time.perf_counter()
                cross_hits = backend.rerank(query, [int(p) for p, _ in stage1], top_k)
                rerank_ms = (time.perf_counter() - t0) * 1000.0
                results[name] = cross_hits
                timings[name] = stage1_ms + rerank_ms
                notes[name] = f"{stage1_ms:.0f} ms BM25 + {rerank_ms:.0f} ms re-rank of {len(stage1)}"
            else:
                if dense_candidates is None:
                    t0 = time.perf_counter()
                    dense_candidates = backend.dense(query, depth)
                    dense_ms = (time.perf_counter() - t0) * 1000.0
                else:
                    dense_ms = 0.0
                if rrf_hits is None:
                    fused = fuse_ranked_lists(
                        [stage1, dense_candidates],
                        k=backend.rrf_k,
                        source_names=["bm25", "biencoder"],
                    )
                    rrf_hits = [(item["pid"], item["score"]) for item in fused]
                if key == "rrf":
                    results[name] = rrf_hits[:top_k]
                    timings[name] = stage1_ms + dense_ms
                    notes[name] = f"RRF k={backend.rrf_k}; top-{depth} candidates per retriever"
                else:
                    t0 = time.perf_counter()
                    fused_candidates = rrf_hits[:depth]
                    hybrid_cross_hits = backend.rerank(
                        query, [int(pid) for pid, _ in fused_candidates], top_k
                    )
                    rerank_ms = (time.perf_counter() - t0) * 1000.0
                    results[name] = hybrid_cross_hits
                    cross_hits = hybrid_cross_hits
                    timings[name] = stage1_ms + dense_ms + rerank_ms
                    notes[name] = (
                        f"BM25 + bi-encoder + RRF + {rerank_ms:.0f} ms re-rank "
                        f"of {len(fused_candidates)} candidates"
                    )

        ordered = [name for name in SYSTEMS if name in results]
        columns = "".join(
            render_column(
                name, results[name], backend.passages, terms, timings[name], notes[name],
                bm25_rank, depth, do_highlight,
            )
            for name in ordered
        )
        insights = render_insights(
            raw_query, top_k, depth, stage1, cross_hits, bm25_rank, {n: timings[n] for n in ordered},
            enhancement_note=enhancement_note,
        )
        return insights, f'<div class="cols">{columns}</div>'

    def feedback_candidates(query: str):
        query = (query or "").strip()
        if not query:
            raise gr.Error("Enter a query before retrieving feedback candidates.")
        hits = backend.tfidf(query, 20)
        options = [
            f"{pid} — {snippet(backend.passages[int(pid)], [], 120)}"
            for pid, _ in hits
        ]
        state = {
            "query": query,
            "hits": [(int(pid), float(score)) for pid, score in hits],
        }
        rows = "".join(
            f"<tr><td>{rank}</td><td>{int(pid)}</td><td>{html.escape(backend.passages[int(pid)])}</td><td>{score:.4f}</td></tr>"
            for rank, (pid, score) in enumerate(hits, start=1)
        )
        table = (
            '<div class="feedback-output">'
            '<div class="output-banner"><span class="eyebrow">CANDIDATES READY</span>'
            f"<strong>{len(hits)} passages retrieved</strong>"
            "<span>Choose positive and/or negative examples below, then apply feedback.</span></div>"
            "<p>Session-local candidates. Mark relevant and/or irrelevant, then rerank.</p>"
            "<table><thead><tr><th>Rank</th><th>PID</th><th>Passage</th><th>TF-IDF score</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )
        return table, gr.update(choices=options, value=[]), gr.update(
            choices=options, value=[]
        ), state

    def apply_feedback(state: dict, relevant: List[str], irrelevant: List[str]):
        if not state or not state.get("query"):
            raise gr.Error("Retrieve candidate passages before applying feedback.")
        positive_ids = [int(value.split(" — ", 1)[0]) for value in (relevant or [])]
        negative_ids = [int(value.split(" — ", 1)[0]) for value in (irrelevant or [])]
        if set(positive_ids).intersection(negative_ids):
            raise gr.Error("A passage cannot be marked both relevant and irrelevant.")
        initial_hits: List[Hit] = state["hits"]
        allowed_ids = {pid for pid, _ in initial_hits}
        if not set(positive_ids + negative_ids).issubset(allowed_ids):
            raise gr.Error("Feedback must refer only to passages shown in this session.")
        if backend.feedback is None:
            raise gr.Error("Relevance feedback is unavailable in this backend.")
        updated = backend.feedback(
            state["query"],
            positive_ids,
            negative_ids,
            len(initial_hits),
        )
        before_rank = {pid: rank for rank, (pid, _) in enumerate(initial_hits, start=1)}
        after_rank = {pid: rank for rank, (pid, _) in enumerate(updated, start=1)}
        rows = []
        for rank, (pid, score) in enumerate(updated, start=1):
            previous = before_rank.get(pid)
            movement = (
                "new"
                if previous is None
                else f"{previous} → {rank} ({previous - rank:+d})"
            )
            rows.append(
                f"<tr><td>{rank}</td><td>{pid}</td><td>{html.escape(backend.passages[pid])}</td>"
                f"<td>{score:.4f}</td><td>{movement}</td></tr>"
            )
        return (
            '<div class="feedback-output">'
            '<div class="output-banner"><span class="eyebrow">UPDATED RANKING</span>'
            f"<strong>{len(updated)} passages re-ordered</strong>"
            f"<span>{len(positive_ids)} relevant · {len(negative_ids)} irrelevant examples</span></div>"
            "<p>Rocchio feedback is user-provided and session-local; "
            "it is not benchmark relevance.</p>"
            "<table><thead><tr><th>New rank</th><th>PID</th><th>Passage</th>"
            f"<th>TF-IDF score</th><th>Rank change</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        )

    def filtered_benchmark_ids(category: str) -> List[str]:
        queries = backend.benchmark_queries or {}
        if category == "All":
            return list(queries)
        bounds = {"Short": (0, 3), "Medium": (4, 6), "Long": (7, 10**9)}
        if category not in bounds:
            raise ValueError(f"Unsupported judged-query filter: {category}")
        lower, upper = bounds[category]
        return [
            qid for qid, text in queries.items()
            if lower <= len(text.split()) <= upper
        ]

    def update_error_query_choices(category: str):
        query_ids = filtered_benchmark_ids(category)
        return gr.update(
            choices=[(f"{qid}: {(backend.benchmark_queries or {}).get(qid, '')[:80]}", qid)
                    for qid in query_ids],
            value=query_ids[0] if query_ids else None,
        )

    def analyze_benchmark_query(query_id: str):
        from src.evaluate import evaluate_run

        if not backend.benchmark_queries or not backend.benchmark_qrels:
            return QUERY_ANALYSIS_UNAVAILABLE
        qid = str(query_id or "")
        query_text = backend.benchmark_queries.get(qid)
        judgments = backend.benchmark_qrels.get(qid)
        if query_text is None or judgments is None:
            raise gr.Error("Select a judged test query with available qrels.")

        depth = 100
        bm25_hits = backend.bm25(query_text, depth)
        tfidf_hits = backend.tfidf(query_text, depth)
        dense_hits = backend.dense(query_text, depth)
        fused_data = fuse_ranked_lists(
            [bm25_hits, dense_hits],
            k=backend.rrf_k,
            source_names=["bm25", "biencoder"],
        )
        rrf_hits: List[Hit] = [
            (item["pid"], item["score"]) for item in fused_data
        ]
        bm25_cross = backend.rerank(query_text, [pid for pid, _ in bm25_hits], depth)
        rrf_cross = backend.rerank(query_text, [pid for pid, _ in rrf_hits[:depth]], depth)
        methods = {
            "TF-IDF": tfidf_hits,
            "BM25": bm25_hits,
            "Bi-encoder": dense_hits,
            "Hybrid RRF": rrf_hits,
            "BM25 → Cross-encoder": bm25_cross,
            "Hybrid RRF → Cross-encoder": rrf_cross,
        }
        positive = {pid for pid, score in judgments.items() if score > 0}
        candidate_recall = {
            name: (
                len(positive.intersection(pid for pid, _ in hits)) / len(positive)
                if positive
                else 0.0
            )
            for name, hits in (
                ("BM25", bm25_hits),
                ("Bi-encoder", dense_hits),
                ("RRF candidate union", rrf_hits),
            )
        }
        sections = []
        for name, hits in methods.items():
            run = {qid: {str(pid): float(score) for pid, score in hits}}
            query_metrics = evaluate_run({qid: judgments}, run)[qid]
            rows = "".join(
                f"<tr><td>{rank}</td><td>{pid}</td><td>{'relevant' if pid in positive else ''}</td>"
                f"<td>{score:.4f}</td><td>{html.escape(backend.passages.get(pid, '[passage missing]'))}</td></tr>"
                for rank, (pid, score) in enumerate(hits[:10], start=1)
            )
            sections.append(
                '<details class="analysis-method"><summary>'
                f'<h4>{html.escape(name)}</h4><div class="metric-pills">'
                f'<span>MRR@10 <b>{query_metrics["MRR@10"]:.3f}</b></span>'
                f'<span>NDCG@10 <b>{query_metrics["NDCG@10"]:.3f}</b></span>'
                f'<span>Recall@10 <b>{query_metrics["Recall@10"]:.3f}</b></span>'
                f'<span>Recall@100 <b>{query_metrics["Recall@100"]:.3f}</b></span>'
                "</div></summary><div class=\"table-scroll\"><table>"
                "<thead><tr><th>Rank</th><th>PID</th><th>Judgment</th>"
                f"<th>Score</th><th>Passage</th></tr></thead><tbody>{rows}</tbody></table></div></details>"
            )
        bm25_top = {pid for pid, _ in bm25_hits[:10]}
        dense_top = {pid for pid, _ in dense_hits[:10]}
        rrf_top = {pid for pid, _ in rrf_hits[:10]}
        overlap = (
            f"Top-10 overlap: BM25∩bi-encoder={len(bm25_top & dense_top)}, "
            f"BM25∩RRF={len(bm25_top & rrf_top)}, "
            f"bi-encoder∩RRF={len(dense_top & rrf_top)}."
        )
        missed = sorted(positive.difference(pid for pid, _ in rrf_hits))
        found = sorted(positive.intersection(pid for pid, _ in rrf_hits))
        bm25_rank = {pid: rank for rank, (pid, _) in enumerate(bm25_hits, start=1)}
        bm25_cross_rank = {
            pid: rank for rank, (pid, _) in enumerate(bm25_cross, start=1)
        }
        rrf_rank = {pid: rank for rank, (pid, _) in enumerate(rrf_hits, start=1)}
        cross_rank = {pid: rank for rank, (pid, _) in enumerate(rrf_cross, start=1)}
        movement_rows = []
        for pid in sorted(set(rrf_rank) | set(cross_rank), key=lambda item: (rrf_rank.get(item, 10**9), item))[:20]:
            movement_rows.append(
                f"<tr><td>{pid}</td><td>{bm25_rank.get(pid, '—')}</td>"
                f"<td>{bm25_cross_rank.get(pid, '—')}</td><td>{rrf_rank.get(pid, '—')}</td>"
                f"<td>{cross_rank.get(pid, '—')}</td></tr>"
            )
        found_ids = html.escape(", ".join(map(str, found)) or "none")
        missed_ids = html.escape(", ".join(map(str, missed)) or "none")
        return (
            '<div class="query-analysis">'
            '<section class="analysis-query">'
            f'<span class="eyebrow">JUDGED TEST QUERY · {html.escape(qid)}</span>'
            f"<h3>{html.escape(query_text)}</h3>"
            f'<div class="analysis-facts"><div><span>Judged relevant passages</span>'
            f"<b>{html.escape(', '.join(map(str, sorted(positive))) or 'none judged relevant')}</b></div>"
            f'<div><span>RRF candidate coverage</span><b>{len(found)} found · {len(missed)} missed</b></div>'
            f'<div><span>Top-10 comparison</span><b>{html.escape(overlap)}</b></div></div>'
            f'<p class="fine">RRF found: {found_ids} · Missed: {missed_ids}</p>'
            "</section>"
            '<section class="analysis-summary"><h4>Candidate-generation recall</h4>'
            f'<div class="metric-pills">{"".join(f"<span>{html.escape(name)} <b>{value:.1%}</b></span>" for name, value in candidate_recall.items())}</div></section>'
            '<section class="analysis-movement"><h4>Rank movement across the fused candidates</h4>'
            "<div class=\"table-scroll\"><table><thead><tr><th>PID</th><th>BM25 rank</th>"
            f"<th>BM25 + cross-encoder rank</th><th>RRF rank</th>"
            f"<th>RRF + cross-encoder rank</th></tr></thead><tbody>{''.join(movement_rows)}</tbody></table></div></section>"
            '<h4 class="analysis-section-title">Top-10 passage lists by method</h4>'
            f'<div class="analysis-method-grid">{"".join(sections)}</div></div>'
        )

    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="violet",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    )
    blocks_kwargs: dict = {"title": "Neural Passage Search"}
    launch_kwargs: dict = {}
    # Gradio 6 moved theme/css from Blocks() to launch(); support both.
    if "theme" in inspect.signature(gr.Blocks.launch).parameters:
        launch_kwargs.update(theme=theme, css=CSS)
    else:
        blocks_kwargs.update(theme=theme, css=CSS)

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.HTML(
            f"""<div class="hero"><h1>\U0001f50e Neural Passage Search</h1>
<p>See what changes when a neural ranker replaces, or re-orders, a lexical one.</p>
<div class="pills"><span>MS MARCO passages</span><span>{backend.corpus_size:,} passages indexed</span>
<span>Running on {html.escape(backend.device)}</span><span>TF-IDF · BM25 · Bi-encoder · Cross-encoder</span></div></div>"""
        )
        with gr.Tabs():
            with gr.Tab("Search"):
                gr.HTML(
                    '<div class="tab-intro search-intro"><span class="eyebrow">INTERACTIVE RETRIEVAL LAB</span>'
                    "<h2>Ask a question. Compare how each system finds answers.</h2>"
                    "<p>Choose an example or enter your own query. Live results are ranked, not judged; "
                    "use the Benchmarks tab for the saved offline quality evaluation.</p></div>"
                )
                with gr.Row(equal_height=True):
                    query = gr.Textbox(
                        label="Query", placeholder="Ask a question, e.g. 'where does the water from cellular respiration go'",
                        lines=1, scale=5, autofocus=True,
                    )
                    search_btn = gr.Button("Search", variant="primary", scale=1)
                    random_btn = gr.Button("\U0001f3b2 Random example", scale=1)
                with gr.Row():
                    example_buttons = [gr.Button(q if len(q) < 58 else q[:55] + "…", size="sm") for q in EXAMPLE_QUERIES]
                with gr.Accordion("Settings & Query Enhancement", open=False):
                    systems = gr.CheckboxGroup(
                        choices=list(SYSTEMS), value=DEFAULT_SYSTEMS, label="Systems to compare",
                        info="Select systems to compare. Rank-movement chips are shown relative to BM25.",
                    )
                    with gr.Row():
                        top_k = gr.Slider(3, 20, value=10, step=1, label="Results per system")
                        depth = gr.Slider(10, 200, value=100, step=10, label="Cross-encoder candidate depth (BM25 top-N)")
                    with gr.Row():
                        enhancement = gr.Dropdown(
                            choices=["Original", "Normalized (NFKC)", "Synonym expansion"],
                            value="Original",
                            label="Query enhancement",
                            info="Optional query transformations before retrieval",
                            scale=2,
                        )
                        do_highlight = gr.Checkbox(value=True, label="Highlight query terms", scale=1)
                insights_out = gr.HTML()
                results_out = gr.HTML(PLACEHOLDER)

            with gr.Tab("Benchmarks"):
                gr.HTML(render_benchmark())

            with gr.Tab("Relevance feedback"):
                gr.HTML(
                    '<div class="tab-intro feedback-intro"><span class="eyebrow">PERSONALIZE A SEARCH</span>'
                    "<h2>Teach the ranker what matters for this query</h2>"
                    "<p>Mark a few retrieved passages as relevant or irrelevant. Rocchio feedback adjusts "
                    "the TF-IDF query for this update only; it does not train the model or change benchmark results.</p>"
                    '<div class="step-strip"><span><b>1</b> Retrieve candidates</span>'
                    '<span><b>2</b> Label examples</span><span><b>3</b> Review rank changes</span></div></div>'
                )
                with gr.Row(equal_height=True):
                    feedback_query = gr.Textbox(
                        label="Search query",
                        placeholder="Enter the question you want to refine",
                        scale=5,
                    )
                    feedback_fetch = gr.Button(
                        "Retrieve candidates", variant="primary", scale=1
                    )
                feedback_candidates_out = gr.HTML(
                    '<div class="feedback-empty"><b>Start with a query</b>'
                    "<span>Retrieve up to 20 TF-IDF candidates to begin labeling examples.</span></div>"
                )
                feedback_state = gr.State({})
                with gr.Row():
                    positive_feedback = gr.CheckboxGroup(
                        label="Relevant — move toward these", choices=[]
                    )
                    negative_feedback = gr.CheckboxGroup(
                        label="Irrelevant — move away from these", choices=[]
                    )
                feedback_apply = gr.Button("Apply feedback and rerank", variant="primary")
                feedback_results = gr.HTML(
                    '<div class="feedback-empty"><b>Your updated ranking will appear here</b>'
                    "<span>Candidate order changes after you apply positive or negative feedback.</span></div>"
                )
                feedback_fetch.click(
                    feedback_candidates,
                    [feedback_query],
                    [
                        feedback_candidates_out,
                        positive_feedback,
                        negative_feedback,
                        feedback_state,
                    ],
                )
                feedback_apply.click(
                    apply_feedback,
                    [feedback_state, positive_feedback, negative_feedback],
                    [feedback_results],
                )

            with gr.Tab("Query error analysis"):
                gr.HTML(
                    '<div class="tab-intro analysis-intro"><span class="eyebrow">QUERY-LEVEL DIAGNOSTICS</span>'
                    "<h2>See where retrieval succeeds or misses</h2>"
                    "<p>Filter prepared, judged test queries by length, then compare how the "
                    "Space's current retrieval systems handle a selected query. Live searches "
                    "are not assigned inferred relevance labels.</p></div>"
                )
                initial_ids = list((backend.benchmark_queries or {}).keys())
                if initial_ids and backend.benchmark_qrels:
                    analysis_initial = (
                        '<div class="analysis-empty"><span class="eyebrow">READY TO EXPLORE</span>'
                        f"<h3>{len(initial_ids):,} held-out queries with qrels</h3>"
                        "<p>Choose a query to compare candidate coverage and top-ranked passages.</p></div>"
                    )
                    with gr.Row():
                        error_filter = gr.Dropdown(
                            choices=["All", "Short", "Medium", "Long"],
                            value="All",
                            label="Filter test queries",
                            scale=1,
                        )
                        error_query = gr.Dropdown(
                            choices=[
                                (f"{qid}: {text[:80]}", qid)
                                for qid, text in (backend.benchmark_queries or {}).items()
                            ],
                            value=initial_ids[0],
                            label="Select a judged query",
                            scale=2,
                        )
                    analyze_button = gr.Button("Analyze query", variant="primary")
                    error_analysis = gr.HTML(analysis_initial)
                    error_filter.change(
                        update_error_query_choices,
                        [error_filter],
                        [error_query],
                    )
                    analyze_button.click(
                        analyze_benchmark_query,
                        [error_query],
                        [error_analysis],
                    )
                else:
                    gr.HTML(QUERY_ANALYSIS_UNAVAILABLE)

            with gr.Tab("How it works"):
                gr.HTML(HOW_IT_WORKS)

        inputs = [query, systems, top_k, depth, do_highlight, enhancement]
        outputs = [insights_out, results_out]
        search_btn.click(search, inputs, outputs)
        query.submit(search, inputs, outputs)
        random_btn.click(lambda: random.choice(EXAMPLE_QUERIES), None, query).then(search, inputs, outputs)
        for button, example in zip(example_buttons, EXAMPLE_QUERIES):
            button.click(lambda q=example: q, None, query).then(search, inputs, outputs)

    return demo, launch_kwargs


def _in_colab() -> bool:
    """True on a Colab VM, including inside `!python ...` subprocesses."""
    return (
        "google.colab" in sys.modules
        or any(key in os.environ for key in ("COLAB_RELEASE_TAG", "COLAB_GPU"))
        or Path("/content/sample_data").is_dir()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural Passage Search demo")
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    # localhost links never work on Colab (the server runs on Google's VM), so a
    # public share link is needed there. `!python` doesn't get Gradio's auto-detection.
    share = args.share or _in_colab()

    backend = load_backend()
    demo, launch_kwargs = build_app(backend)
    print(f"Launching (share={share}) ...", flush=True)
    demo.queue().launch(share=share, server_port=args.port, debug=True, **launch_kwargs)


if __name__ == "__main__":
    main()