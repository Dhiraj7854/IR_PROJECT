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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = ARTIFACTS_DIR / "results"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

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
}
DEFAULT_SYSTEMS = ["BM25", "BM25 → Cross-encoder"]

# Numbers from the README's completed Colab evaluation (750 test queries). Used only
# if artifacts/results/comparison_table.csv is not present locally.
FALLBACK_BENCHMARK = [
    {"System": "TF-IDF", "MRR@10": 0.2943, "NDCG@10": 0.3854, "P@5": 0.1091, "Recall@100": 0.9073, "Median latency (ms/query)": 55.13},
    {"System": "BM25", "MRR@10": 0.3316, "NDCG@10": 0.4230, "P@5": 0.1192, "Recall@100": 0.9087, "Median latency (ms/query)": 153.47},
    {"System": "Bi-encoder", "MRR@10": 0.4895, "NDCG@10": 0.5790, "P@5": 0.1525, "Recall@100": 0.9820, "Median latency (ms/query)": 7.42},
    {"System": "BM25 → Cross-encoder", "MRR@10": 0.4016, "NDCG@10": 0.4835, "P@5": 0.1272, "Recall@100": 0.7544, "Median latency (ms/query)": 284.89},
]

_STOPWORDS = frozenset(
    "a an and are as at be but by do does did for from had has have how i if in into is it its "
    "of on or that the their then there these they this to was we were what when where which "
    "who whom whose why will with you your".split()
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
    device: str = "cpu"

    @property
    def corpus_size(self) -> int:
        return len(self.passages)


def load_backend() -> Backend:
    """Load the corpus, lexical index, bi-encoder, and cross-encoder once."""
    import pandas as pd
    import torch

    from src.baseline import build_index, get_index
    from src.biencoder import configure_retrieval, load_biencoder, retrieve_biencoder
    from src.config import load_config
    from src.crossencoder import load_crossencoder, rerank

    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    max_seq_len = config.models.max_seq_len
    corpus_path = DATA_DIR / "corpus.parquet"
    biencoder_path = ARTIFACTS_DIR / "biencoder"
    crossencoder_path = ARTIFACTS_DIR / "crossencoder"
    embeddings_path = biencoder_path / "corpus_embeddings.npy"

    for path, hint in [
        (corpus_path, "Run scripts/prepare_data.py first."),
        (biencoder_path, "Train or copy the bi-encoder checkpoint."),
        (embeddings_path, "Run scripts/eval_biencoder.py once to cache embeddings."),
        (crossencoder_path, "Train or copy the cross-encoder checkpoint."),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. {hint}")

    corpus_df = pd.read_parquet(corpus_path)
    if corpus_df.empty or corpus_df["pid"].duplicated().any():
        raise ValueError("Corpus must be non-empty with unique pids")

    started = time.perf_counter()
    build_index(corpus_df=corpus_df, artifacts_dir=ARTIFACTS_DIR)
    index = get_index()
    logger.info("Lexical index ready in %.2fs", time.perf_counter() - started)

    encoder = load_biencoder(biencoder_path)
    configure_retrieval(
        corpus_df["passage_text"].astype(str).tolist(),
        pids=corpus_df["pid"].tolist(),
        batch_size=64,
        model=encoder,
        cache_path=embeddings_path,
    )
    cross_model, cross_tokenizer = load_crossencoder(crossencoder_path)

    passages = dict(zip(corpus_df["pid"].astype(int), corpus_df["passage_text"].astype(str)))
    use_cuda = next(cross_model.parameters()).device.type == "cuda"

    def _rerank(query: str, pids: List[int], k: int) -> List[Hit]:
        hits = rerank(
            query,
            pids,
            [passages[p] for p in pids],
            k=k,
            model=cross_model,
            tokenizer=cross_tokenizer,
            batch_size=32,
            max_length=max_seq_len,
            fp16=True,
        )
        if use_cuda:
            torch.cuda.synchronize()
        return hits

    backend = Backend(
        passages=passages,
        bm25=index.retrieve_bm25,
        tfidf=index.retrieve_tfidf,
        dense=retrieve_biencoder,
        rerank=_rerank,
        device="GPU" if use_cuda else "CPU",
    )

    # Warm up so the first user query is not penalised by lazy initialisation.
    sample = list(passages)[:100]
    backend.bm25("warm up retrieval", 100)
    backend.tfidf("warm up retrieval", 10)
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
    return f"""
<div class="insights">
  <div class="insight-main"><div class="tiles">{''.join(tiles)}</div>{donut}</div>
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
                    "recall@100": "Recall@100",
                    "median_latency_ms": "Median latency (ms/query)",
                }
                parsed: List[Dict[str, object]] = []
                for row in rows:
                    normalised: Dict[str, object] = {}
                    for key, value in row.items():
                        if value in (None, ""):
                            continue
                        label = aliases.get(key.strip().lower(), key)
                        normalised[label] = value if label == "System" else float(value)
                    if "System" in normalised:
                        parsed.append(normalised)
                if parsed:
                    return parsed
        except (ValueError, OSError) as exc:  # fall back to the README numbers
            logger.warning("Could not parse %s: %s", path, exc)
    return FALLBACK_BENCHMARK


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


def _chart_card(title: str, subtitle: str, svg: str, extra_class: str = "") -> str:
    return f"""
<section class="chart-card {extra_class}">
  <div class="chart-heading"><div><h3>{html.escape(title)}</h3>
    <p>{html.escape(subtitle)}</p></div></div>
  <div class="chart-viewport">{svg}</div>
</section>"""


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


def render_benchmark() -> str:
    rows = _read_benchmark()
    preferred_metrics = ["MRR@10", "NDCG@10", "P@5", "Recall@100", "Median latency (ms/query)"]
    metrics = [metric for metric in preferred_metrics if any(metric in row for row in rows)]
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
            star = "*" if row["System"].endswith("Cross-encoder") and metric == "Recall@100" else ""
            cells.append(f'<td class="{"best" if is_best else ""}">{value:.4g}{star}</td>')
        color = SYSTEMS.get(row["System"], ("", "#64748b", ""))[1]
        body.append(
            f'<tr><th><span class="dot" style="background:{color}"></span>{html.escape(row["System"])}</th>{"".join(cells)}</tr>'
        )
    pareto = _chart_card(
        "Quality vs. speed",
        "Each point is one retriever. Favor the upper-left: higher MRR with lower query latency.",
        _render_pareto_chart(rows),
    )
    ablations = _read_ablation_tables()
    ablation_cards = ""
    if len(ablations) >= 3:
        ablation_cards = (
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
    <h2>How the rankers compare</h2>
    <p>Quality metrics are computed on the held-out test queries. Latency is the median over 200 queries.</p>
  </div>
  <table><thead><tr><th>System</th>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>
  <p class="fine">Best value per column highlighted. 750 test queries, MS MARCO passages.
  *Cross-encoder Recall@100 was computed over a run file holding only its top-10, so it is not a true Recall@100.</p>
  {pareto}
  <div class="bench-intro ablation-intro"><span class="eyebrow">ABLATION STUDIES</span>
    <h2>What changes the ranking?</h2>
    <p>Validation experiments isolate training negatives and candidate depth; query-length results use the test split.</p>
  </div>
  {ablation_cards}
</div>"""


HOW_IT_WORKS = """
### How the systems differ

| System | Idea | Strength | Weakness |
|---|---|---|---|
| **TF-IDF** | Weight terms by frequency and rarity, cosine similarity | Fast, transparent | No length normalisation, exact words only |
| **BM25** | TF-IDF refinement with saturation and length normalisation | Strong lexical baseline | Vocabulary mismatch (synonyms, paraphrase) |
| **Bi-encoder** | Embed query and passage independently, compare vectors | Very fast at query time, semantic | Single vector per passage loses detail |
| **BM25 → Cross-encoder** | Re-score BM25's top-N by reading query and passage together | Most precise per pair | Slow; **capped by BM25's recall** |

### Reading the result cards
- **Coloured bar:** the score, scaled within that column. Scores from different systems are on different scales, so compare ranks, not raw values.
- **▲ / ▼ chips:** how far a passage moved relative to BM25's ranking.
- **✨ outside BM25 top-N:** the system found a passage BM25's candidate list never contained. For the cross-encoder this can't happen (it only re-orders BM25 candidates), which is exactly why a dense first stage helps.
- **Highlighted words:** query terms, so you can see when a system retrieves *without* lexical overlap.

### Metrics
- **MRR@10:** average of 1/rank of the first relevant passage in the top 10.
- **NDCG@10:** rewards putting relevant passages higher, with a log discount.
- **Recall@100:** share of relevant passages found in the top 100.
"""


CSS = """
.gradio-container { max-width: 1280px !important; margin: auto; }
.hero { padding: 22px 26px; border-radius: 16px; color: #fff;
  background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%); }
.hero h1 { margin: 0 0 4px; font-size: 1.9rem; letter-spacing: -0.02em; color: #fff; }
.hero p { margin: 0; opacity: .92; }
.hero .pills { margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }
.hero .pills span { background: rgba(255,255,255,.18); padding: 3px 10px; border-radius: 999px; font-size: .8rem; }
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
@keyframes grow-in { from { transform: scaleX(.04); opacity: .25; } to { transform: scaleX(1); opacity: 1; } }
@media (max-width: 760px) {
  .insights { grid-template-columns: 1fr; }
  .chart-grid { grid-template-columns: 1fr; }
  .wide-chart { grid-column: auto; }
  .chart-card { padding: 12px; }
  .data-chart { min-width: 540px; }
  .chart-viewport { overflow-x: auto; }
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

    def search(query: str, systems: List[str], top_k: int, depth: int, do_highlight: bool):
        query = (query or "").strip()
        if not query:
            raise gr.Error("Enter a query before searching.")
        if not systems:
            raise gr.Error("Select at least one system to display.")
        top_k, depth = int(top_k), int(depth)
        depth = max(depth, top_k)
        terms = query_terms(query)

        # Stage 1 always runs: it feeds the BM25 column, the cross-encoder, and rank chips.
        started = time.perf_counter()
        stage1 = backend.bm25(query, depth)
        stage1_ms = (time.perf_counter() - started) * 1000.0
        bm25_rank = {int(pid): rank for rank, (pid, _) in enumerate(stage1, start=1)}

        results: Dict[str, List[Hit]] = {}
        timings: Dict[str, float] = {}
        notes: Dict[str, str] = {}
        cross_hits: Optional[List[Hit]] = None

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
            else:
                t0 = time.perf_counter()
                cross_hits = backend.rerank(query, [int(p) for p, _ in stage1], top_k)
                rerank_ms = (time.perf_counter() - t0) * 1000.0
                results[name] = cross_hits
                timings[name] = stage1_ms + rerank_ms
                notes[name] = f"{stage1_ms:.0f} ms BM25 + {rerank_ms:.0f} ms re-rank of {len(stage1)}"

        ordered = [name for name in SYSTEMS if name in results]
        columns = "".join(
            render_column(
                name, results[name], backend.passages, terms, timings[name], notes[name],
                bm25_rank, depth, do_highlight,
            )
            for name in ordered
        )
        insights = render_insights(
            query, top_k, depth, stage1, cross_hits, bm25_rank, {n: timings[n] for n in ordered}
        )
        return insights, f'<div class="cols">{columns}</div>'

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
                with gr.Row(equal_height=True):
                    query = gr.Textbox(
                        label="Query", placeholder="Ask a question, e.g. 'where does the water from cellular respiration go'",
                        lines=1, scale=5, autofocus=True,
                    )
                    search_btn = gr.Button("Search", variant="primary", scale=1)
                    random_btn = gr.Button("\U0001f3b2 Random example", scale=1)
                with gr.Row():
                    example_buttons = [gr.Button(q if len(q) < 58 else q[:55] + "…", size="sm") for q in EXAMPLE_QUERIES]
                with gr.Accordion("Settings", open=False):
                    systems = gr.CheckboxGroup(
                        choices=list(SYSTEMS), value=DEFAULT_SYSTEMS, label="Systems to compare",
                        info="Select 1–4. Rank-movement chips are shown relative to BM25.",
                    )
                    with gr.Row():
                        top_k = gr.Slider(3, 20, value=10, step=1, label="Results per system")
                        depth = gr.Slider(10, 200, value=100, step=10, label="Cross-encoder candidate depth (BM25 top-N)")
                    do_highlight = gr.Checkbox(value=True, label="Highlight query terms")
                insights_out = gr.HTML()
                results_out = gr.HTML(PLACEHOLDER)

            with gr.Tab("Benchmarks"):
                gr.HTML(render_benchmark())

            with gr.Tab("How it works"):
                gr.Markdown(HOW_IT_WORKS)

        inputs = [query, systems, top_k, depth, do_highlight]
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