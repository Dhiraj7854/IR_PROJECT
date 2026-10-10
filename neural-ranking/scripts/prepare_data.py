#!/usr/bin/env python3
"""Script to download MS MARCO, sample 100k corpus, create query splits, and mine BM25 hard negative triplets."""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import configure_ipv4_preference, prepare_msmarco_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for data preparation."""
    parser = argparse.ArgumentParser(
        description="Download MS MARCO, create 100k corpus, split queries, and generate BM25 triplets."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save processed parquet files.",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution entry point."""
    configure_ipv4_preference()
    args = parse_args()

    print("=" * 60)
    print("Neural Ranking - Data Preparation Pipeline")
    print(f"Config: {args.config}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 60)

    stats = prepare_msmarco_pipeline(
        config_path=args.config,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("Data Preparation Pipeline Completed Successfully!")
    print("=" * 60)
    print(f"Corpus size: {stats['corpus_size']:,} unique passages")
    print(
        f"Queries per split: Train={stats['queries_per_split']['train']:,}, "
        f"Val={stats['queries_per_split']['val']:,}, "
        f"Test={stats['queries_per_split']['test']:,} "
        f"(Total={sum(stats['queries_per_split'].values()):,})"
    )
    print(f"Triplets generated: {stats['triplets_generated']:,}")
    print(
        f"Average passage length: {stats['avg_passage_length_words']:.2f} words "
        f"({stats['avg_passage_length_chars']:.2f} characters)"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
