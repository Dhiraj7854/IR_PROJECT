#!/usr/bin/env python3
"""Evaluate BM25, RRF, and their cross-encoder rerankings on a data split."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import build_index, get_index, write_trec_run
from src.biencoder import configure_retrieval, load_biencoder, retrieve_biencoder
from src.config import load_config
from src.crossencoder import load_crossencoder, rerank
from src.evaluate import average_metrics, evaluate_run, load_qrels, load_run
from src.hybrid import fuse_ranked_lists, rerank_with_tail

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate RRF and fixed-candidate cross-encoder runs. "
            "Existing outputs are preserved unless --overwrite is passed."
        )
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/config.yaml"),
        help="Path to YAML configuration.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "data/processed"),
        help="Directory containing corpus and query/qrels parquet files.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=str(PROJECT_ROOT / "artifacts"),
        help="Directory containing checkpoints and output runs.",
    )
    parser.add_argument("--biencoder-model", default=None)
    parser.add_argument("--crossencoder-model", default=None)
    parser.add_argument("--candidate-depth", type=int, default=None)
    parser.add_argument("--rerank-depth", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--biencoder-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("val", "test"),
        default=("val", "test"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _candidate_recall(
    qrels: Dict[str, Dict[str, int]],
    candidate_ids: Dict[str, Sequence[int]],
) -> Dict[str, float]:
    per_query = []
    for qid, judgments in qrels.items():
        relevant = {pid for pid, score in judgments.items() if score > 0}
        if not relevant:
            continue
        per_query.append(
            len(relevant.intersection(str(pid) for pid in candidate_ids.get(qid, [])))
            / len(relevant)
        )
    return {
        "macro_candidate_recall": sum(per_query) / len(per_query) if per_query else 0.0,
        "judged_queries": float(len(per_query)),
    }


def _save_run_metrics(
    run_path: Path,
    qrels: Dict[str, Dict[str, int]],
    *,
    candidate_recall: Dict[str, float],
    metadata: Dict[str, object],
    corpus_size: int,
) -> Dict[str, object]:
    per_query = evaluate_run(qrels, run_path)
    ranked_run = load_run(run_path)
    required_depth = min(100, corpus_size)
    recall_100_valid = all(
        not ranked_run.get(qid) or len(ranked_run[qid]) >= required_depth
        for qid in qrels
    )
    if not recall_100_valid:
        for metrics in per_query.values():
            metrics.pop("Recall@100", None)
    summary: Dict[str, object] = {
        "run_file": str(run_path),
        "candidate_recall": candidate_recall,
        "average_metrics": average_metrics(per_query),
        "per_query_metrics": per_query,
        "Recall@100_valid": recall_100_valid,
        **metadata,
    }
    metrics_path = run_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    if min(args.top_k, args.batch_size, args.biencoder_batch_size) <= 0:
        raise ValueError("Output depth and batch sizes must be positive")
    if args.candidate_depth is not None and args.candidate_depth <= 0:
        raise ValueError("--candidate-depth must be positive")
    if args.rerank_depth is not None and args.rerank_depth <= 0:
        raise ValueError("--rerank-depth must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    config = load_config(args.config)
    config_path = Path(args.config)
    if not config_path.is_file():
        config_path = PROJECT_ROOT / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    candidate_depth = args.candidate_depth or config.retrieval.stage1_top_k
    rerank_depth = args.rerank_depth or config.retrieval.rerank_depth
    biencoder_path = Path(args.biencoder_model or Path(args.artifacts_dir) / "biencoder")
    crossencoder_path = Path(args.crossencoder_model or Path(args.artifacts_dir) / "crossencoder")
    for model_path, label in (
        (biencoder_path, "bi-encoder"),
        (crossencoder_path, "cross-encoder"),
    ):
        if not model_path.is_dir():
            raise FileNotFoundError(f"{label} checkpoint not found: {model_path}")

    data_dir = Path(args.data_dir)
    corpus_path = data_dir / "corpus.parquet"
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    corpus_df = pd.read_parquet(corpus_path)
    required_corpus = {"pid", "passage_text"}
    if not required_corpus.issubset(corpus_df.columns):
        raise ValueError(f"Corpus must contain {sorted(required_corpus)}")
    if corpus_df["pid"].duplicated().any():
        raise ValueError("Corpus passage IDs must be unique")

    artifacts_dir = Path(args.artifacts_dir)
    build_index(corpus_df=corpus_df, artifacts_dir=artifacts_dir)
    index = get_index()
    biencoder = load_biencoder(biencoder_path)
    configure_retrieval(
        corpus_df["passage_text"].astype(str).tolist(),
        pids=corpus_df["pid"].tolist(),
        batch_size=args.biencoder_batch_size,
        model=biencoder,
        cache_path=biencoder_path / "corpus_embeddings.npy",
    )
    crossencoder, tokenizer = load_crossencoder(crossencoder_path)
    passages = dict(
        zip(corpus_df["pid"].astype(int), corpus_df["passage_text"].astype(str))
    )
    run_dir = artifacts_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        queries_path = data_dir / f"{split}_queries.parquet"
        qrels_path = data_dir / f"{split}_qrels.parquet"
        for path in (queries_path, qrels_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required {split} file not found: {path}")
        queries = pd.read_parquet(queries_path)
        qrels = load_qrels(qrels_path)
        if not {"query_id", "query_text"}.issubset(queries.columns):
            raise ValueError(f"{queries_path} must contain query_id and query_text")

        run_paths = {
            "bm25_cross": run_dir
            / f"bm25_crossencoder_c{candidate_depth}_r{rerank_depth}_{split}.txt",
            "rrf": run_dir / f"hybrid_rrf_c{candidate_depth}_{split}.txt",
            "rrf_cross": run_dir
            / f"hybrid_crossencoder_c{candidate_depth}_r{rerank_depth}_{split}.txt",
        }
        metrics_paths = [path.with_suffix(".metrics.json") for path in run_paths.values()]
        existing = [
            path for path in [*run_paths.values(), *metrics_paths] if path.exists()
        ]
        if existing and not args.overwrite:
            raise FileExistsError(
                "Refusing to overwrite existing run(s); use --overwrite explicitly: "
                + ", ".join(str(path) for path in existing)
            )

        run_results: Dict[str, Dict[int, List[Tuple[int, float]]]] = {
            key: {} for key in run_paths
        }
        candidate_ids: Dict[str, Dict[str, List[int]]] = {
            "bm25": {},
            "rrf_union": {},
        }
        missing_passage_count = 0
        started = time.perf_counter()
        for query in queries.itertuples(index=False):
            qid = int(query.query_id)
            qid_key = str(qid)
            query_text = str(query.query_text)
            bm25_hits = index.retrieve_bm25(query_text, k=candidate_depth)
            dense_hits = retrieve_biencoder(query_text, k=candidate_depth)
            fused = fuse_ranked_lists(
                [bm25_hits, dense_hits],
                k=config.retrieval.rrf_k,
                weights=[
                    (config.retrieval.rrf_weights or {}).get("bm25", 1.0),
                    (config.retrieval.rrf_weights or {}).get("biencoder", 1.0),
                ],
                source_names=["bm25", "biencoder"],
            )
            fused_hits = [(entry["pid"], entry["score"]) for entry in fused]
            candidate_ids["bm25"][qid_key] = [pid for pid, _ in bm25_hits]
            candidate_ids["rrf_union"][qid_key] = [pid for pid, _ in fused_hits]
            run_results["rrf"][qid] = fused_hits[: args.top_k]

            bm25_pool = [(pid, score) for pid, score in bm25_hits if pid in passages]
            rrf_pool = [(pid, score) for pid, score in fused_hits if pid in passages]
            missing_passage_count += (
                len(bm25_hits) - len(bm25_pool)
                + len(fused_hits) - len(rrf_pool)
            )
            for run_key, candidate_pool in (
                ("bm25_cross", bm25_pool),
                ("rrf_cross", rrf_pool),
            ):
                candidates = candidate_pool[:rerank_depth]
                reranked = rerank(
                    query_text,
                    [pid for pid, _ in candidates],
                    [passages[pid] for pid, _ in candidates],
                    k=args.top_k,
                    model=crossencoder,
                    tokenizer=tokenizer,
                    batch_size=args.batch_size,
                    max_length=config.models.max_seq_len,
                    fp16=True,
                )
                run_results[run_key][qid] = rerank_with_tail(
                    candidate_pool,
                    reranked,
                    output_depth=args.top_k,
                )

        elapsed = time.perf_counter() - started
        qrels_hash = hashlib.sha256(qrels_path.read_bytes()).hexdigest()
        for key, path in run_paths.items():
            write_trec_run(run_results[key], path, system_name=key)
        metadata: Dict[str, object] = {
            "dataset": config.dataset.name,
            "split": split,
            "query_count": int(len(queries)),
            "qrels_sha256": qrels_hash,
            "queries_sha256": hashlib.sha256(queries_path.read_bytes()).hexdigest(),
            "seed": args.seed,
            "candidate_depth_per_retriever": candidate_depth,
            "rerank_depth": rerank_depth,
            "output_depth": args.top_k,
            "rrf_k": config.retrieval.rrf_k,
            "rrf_weights": config.retrieval.rrf_weights or {
                "bm25": 1.0,
                "biencoder": 1.0,
            },
            "effective_config": asdict(config),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "biencoder_model": str(biencoder_path),
            "crossencoder_model": str(crossencoder_path),
            "elapsed_seconds": elapsed,
            "missing_candidate_passage_occurrences": missing_passage_count,
            "evaluation_metric_convention": (
                "Positive relevance is rel>0; macro over qrels queries; "
                "queries with no positive judgments score zero."
            ),
        }
        recall_by_candidate = {
            key: _candidate_recall(qrels, values)
            for key, values in candidate_ids.items()
        }
        recall_by_candidate["rrf_rerank"] = recall_by_candidate["rrf_union"]
        for key, path in run_paths.items():
            recall_key = "bm25" if key == "bm25_cross" else (
                "rrf_union" if key == "rrf" else "rrf_rerank"
            )
            summary = _save_run_metrics(
                path,
                qrels,
                candidate_recall=recall_by_candidate[recall_key],
                metadata=metadata,
                corpus_size=len(corpus_df),
            )
            logger.info(
                "%s %s MRR@10=%.4f NDCG@10=%.4f Recall@100=%s; "
                "candidate recall=%.4f. Saved %s",
                split,
                key,
                summary["average_metrics"]["MRR@10"],
                summary["average_metrics"]["NDCG@10"],
                (
                    f"{summary['average_metrics']['Recall@100']:.4f}"
                    if "Recall@100" in summary["average_metrics"]
                    else "not available"
                ),
                summary["candidate_recall"]["macro_candidate_recall"],
                path,
            )


if __name__ == "__main__":
    main()
