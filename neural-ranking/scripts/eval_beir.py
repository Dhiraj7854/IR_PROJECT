#!/usr/bin/env python3
"""Evaluate existing retrieval models zero-shot on a prepared BEIR dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import pandas as pd
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.biencoder import configure_retrieval, load_biencoder, retrieve_biencoder
from src.config import load_config
from src.crossencoder import load_crossencoder, rerank
from src.evaluate import average_metrics, evaluate_run
from src.hybrid import fuse_ranked_lists, rerank_with_tail

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BM25, dense, RRF, and optional cross-encoder BEIR evaluation."
    )
    parser.add_argument("--dataset", choices=("scifact", "nfcorpus"), required=True)
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "beir"))
    parser.add_argument("--artifacts-dir", default=str(PROJECT_ROOT / "artifacts"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/config.yaml"))
    parser.add_argument("--biencoder-model", default=None)
    parser.add_argument("--crossencoder-model", default=None)
    parser.add_argument("--candidate-depth", type=int, default=100)
    parser.add_argument("--rerank-depth", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-crossencoder", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _qrels_dict(source: object) -> Dict[str, Dict[str, int]]:
    if isinstance(source, pd.DataFrame):
        if {"query_id", "relevant_pid"}.issubset(source.columns):
            qid_column, pid_column = "query_id", "relevant_pid"
            rel_column = "relevance" if "relevance" in source.columns else None
        elif {"query-id", "corpus-id", "score"}.issubset(source.columns):
            qid_column, pid_column, rel_column = "query-id", "corpus-id", "score"
        elif {"qid", "docno", "relevance"}.issubset(source.columns):
            qid_column, pid_column, rel_column = "qid", "docno", "relevance"
        else:
            raise ValueError("BEIR qrels DataFrame has unexpected columns")
        result: Dict[str, Dict[str, int]] = {}
        for _, row in source.iterrows():
            qid, pid = str(row[qid_column]), str(row[pid_column])
            judgments = result.setdefault(qid, {})
            if rel_column and pd.isna(row[rel_column]):
                continue
            judgments[pid] = int(row[rel_column]) if rel_column else 1
        return result
    if isinstance(source, Mapping):
        result: Dict[str, Dict[str, int]] = {}
        for qid, judgments in source.items():
            if not isinstance(judgments, Mapping):
                raise ValueError("BEIR qrels must map query IDs to document judgments")
            result[str(qid)] = {
                str(pid): int(relevance) for pid, relevance in judgments.items()
            }
        return result
    raise TypeError("Unsupported BEIR qrels representation")


def _write_trec(path: Path, run: Mapping[str, Mapping[str, float]], tag: str) -> None:
    lines = []
    for qid, docs in run.items():
        for rank, (doc_id, score) in enumerate(docs.items(), start=1):
            lines.append(f"{qid} Q0 {doc_id} {rank} {score:.8f} {tag}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if min(args.candidate_depth, args.rerank_depth, args.top_k, args.batch_size) <= 0:
        raise ValueError("Candidate, rerank, output depths, and batch size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    try:
        from beir.datasets.data_loader import GenericDataLoader
    except ImportError as error:
        raise RuntimeError(
            "BEIR evaluation requires the optional dependency: "
            "python -m pip install -e \".[beir]\""
        ) from error

    dataset_path = Path(args.data_dir) / args.dataset
    if not dataset_path.is_dir():
        raise FileNotFoundError(
            f"Dataset directory missing: {dataset_path}. Prepare explicitly with "
            f"python scripts/prepare_beir.py --dataset {args.dataset}"
        )
    corpus, queries, raw_qrels = GenericDataLoader(data_folder=str(dataset_path)).load(
        split="test"
    )
    qrels = _qrels_dict(raw_qrels)
    if not corpus or not queries or not qrels:
        raise ValueError(f"BEIR test split is empty or malformed: {dataset_path}")

    # Internal integer IDs keep the existing retrieval interfaces unchanged.
    external_ids = list(corpus)
    external_to_internal = {doc_id: index for index, doc_id in enumerate(external_ids)}
    internal_to_external = {index: doc_id for doc_id, index in external_to_internal.items()}
    passages = []
    for doc_id in external_ids:
        item = corpus[doc_id]
        title = str(item.get("title", "")).strip()
        text = str(item.get("text", "")).strip()
        passages.append(f"{title} {text}".strip())
    corpus_df = pd.DataFrame(
        {"pid": list(range(len(passages))), "passage_text": passages}
    )

    config = load_config(args.config)
    artifacts_dir = Path(args.artifacts_dir) / "beir" / args.dataset
    runs_dir = artifacts_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        runs_dir / f"bm25_test_top{args.top_k}.txt",
        runs_dir / f"biencoder_test_top{args.top_k}.txt",
        runs_dir / f"rrf_test_top{args.top_k}.txt",
    ]
    cross_path = args.crossencoder_model or str(Path(args.artifacts_dir) / "crossencoder")
    cross_available = not args.no_crossencoder and Path(cross_path).is_dir()
    if cross_available:
        output_paths.append(
            runs_dir / f"bm25_crossencoder_c{args.candidate_depth}_r{args.rerank_depth}.txt"
        )
        output_paths.append(
            runs_dir / f"rrf_crossencoder_c{args.candidate_depth}_r{args.rerank_depth}.txt"
        )
    existing = [path for path in output_paths if path.exists()]
    metrics_path = artifacts_dir / "metrics_test.json"
    if metrics_path.exists():
        existing.append(metrics_path)
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite BEIR runs; pass --overwrite explicitly: "
            + ", ".join(str(path) for path in existing)
        )

    from src.baseline import BaselineIndex

    index = BaselineIndex()
    index.build(corpus_df, artifacts_dir=artifacts_dir / "index")
    biencoder_path = args.biencoder_model or config.models.biencoder_name
    model = load_biencoder(biencoder_path)
    configure_retrieval(
        passages,
        pids=list(range(len(passages))),
        model=model,
        cache_path=artifacts_dir / "corpus_embeddings.npy",
    )

    bm25_run: Dict[str, Dict[str, float]] = {}
    dense_run: Dict[str, Dict[str, float]] = {}
    rrf_run: Dict[str, Dict[str, float]] = {}
    bm25_cross_run: Dict[str, Dict[str, float]] = {}
    cross_run: Dict[str, Dict[str, float]] = {}
    candidate_ids = {
        "bm25": {},
        "biencoder": {},
        "rrf": {},
    }
    ranker = tokenizer = None
    if cross_available:
        ranker, tokenizer = load_crossencoder(cross_path)

    for qid, query in queries.items():
        qid = str(qid)
        bm25_hits = index.retrieve_bm25(str(query), args.candidate_depth)
        dense_hits = retrieve_biencoder(str(query), args.candidate_depth)
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
        candidate_ids["bm25"][qid] = [pid for pid, _ in bm25_hits]
        candidate_ids["biencoder"][qid] = [pid for pid, _ in dense_hits]
        candidate_ids["rrf"][qid] = [entry["pid"] for entry in fused]
        for destination, hits in (
            (bm25_run, bm25_hits),
            (dense_run, dense_hits),
            (rrf_run, fused_hits),
        ):
            destination[qid] = {
                internal_to_external[pid]: float(score) for pid, score in hits[: args.top_k]
            }
        if cross_available and ranker is not None:
            bm25_candidates = bm25_hits[: args.rerank_depth]
            bm25_reranked = rerank(
                str(query),
                [pid for pid, _ in bm25_candidates],
                [passages[pid] for pid, _ in bm25_candidates],
                k=args.top_k,
                model=ranker,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_length=config.models.max_seq_len,
                fp16=True,
            )
            bm25_reranked = rerank_with_tail(
                bm25_hits,
                bm25_reranked,
                output_depth=args.top_k,
            )
            bm25_cross_run[qid] = {
                internal_to_external[pid]: float(score)
                for pid, score in bm25_reranked
            }
            cross_candidates = fused_hits[: args.rerank_depth]
            reranked = rerank(
                str(query),
                [pid for pid, _ in cross_candidates],
                [passages[pid] for pid, _ in cross_candidates],
                k=args.top_k,
                model=ranker,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_length=config.models.max_seq_len,
                fp16=True,
            )
            reranked = rerank_with_tail(
                fused_hits,
                reranked,
                output_depth=args.top_k,
            )
            cross_run[qid] = {
                internal_to_external[pid]: float(score) for pid, score in reranked
            }

    run_map = {
        "bm25": bm25_run,
        "biencoder": dense_run,
        "rrf": rrf_run,
    }
    if cross_run:
        run_map["bm25_crossencoder"] = bm25_cross_run
        run_map["rrf_crossencoder"] = cross_run
    for path, (name, run) in zip(output_paths, run_map.items()):
        _write_trec(path, run, name)

    split_qrels = {
        str(qid): judgments for qid, judgments in qrels.items()
    }
    qrels_path = dataset_path / "qrels" / "test.tsv"
    qrels_bytes = (
        qrels_path.read_bytes()
        if qrels_path.is_file()
        else json.dumps(split_qrels, sort_keys=True).encode("utf-8")
    )
    result = {
        "dataset": args.dataset,
        "split": "test",
        "evaluation_mode": "zero-shot; no BEIR relevance judgments used for tuning",
        "query_count": len(queries),
        "corpus_count": len(corpus),
        "queries_sha256": (
            hashlib.sha256((dataset_path / "queries.jsonl").read_bytes()).hexdigest()
            if (dataset_path / "queries.jsonl").is_file()
            else None
        ),
        "candidate_depth_per_retriever": args.candidate_depth,
        "rerank_depth": args.rerank_depth,
        "output_depth": args.top_k,
        "seed": args.seed,
        "Recall@100_valid": args.top_k >= min(100, len(corpus)),
        "rrf_k": config.retrieval.rrf_k,
        "biencoder_model": biencoder_path,
        "crossencoder_model": cross_path if cross_available else None,
        "qrels_sha256": hashlib.sha256(qrels_bytes).hexdigest(),
        "runs": {},
    }
    recall_100_valid = args.top_k >= min(100, len(corpus))
    for name, run in run_map.items():
        per_query_metrics = evaluate_run(split_qrels, run)
        if not recall_100_valid:
            for metric_values in per_query_metrics.values():
                metric_values.pop("Recall@100", None)
        metrics = average_metrics(per_query_metrics)
        candidate_source = {
            "bm25": "bm25",
            "biencoder": "biencoder",
            "rrf": "rrf",
            "rrf_crossencoder": "rrf",
            "bm25_crossencoder": "bm25",
        }[name]
        candidate_recall_values = []
        for qid, judgments in split_qrels.items():
            relevant = {pid for pid, score in judgments.items() if score > 0}
            if not relevant:
                continue
            ids = {
                internal_to_external[pid]
                for pid in candidate_ids[candidate_source].get(qid, [])
            }
            candidate_recall_values.append(
                len(relevant.intersection(ids)) / len(relevant)
            )
        result["runs"][name] = {
            "file": str(output_paths[list(run_map).index(name)]),
            "metrics": metrics,
            "per_query_metrics": per_query_metrics,
            "candidate_recall": (
                sum(candidate_recall_values) / len(candidate_recall_values)
                if candidate_recall_values else 0.0
            ),
        }
        logger.info("%s: %s", name, json.dumps(metrics))
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved BEIR metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
