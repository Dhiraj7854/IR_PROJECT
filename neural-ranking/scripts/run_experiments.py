"""Placeholder script for executing comparative benchmarking across ranking methods."""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for running comparative experiments."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end benchmark comparison across TF-IDF, BM25, bi-encoder, and two-stage pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["tfidf", "bm25", "biencoder", "two_stage"],
        help="Systems to include in benchmark comparison.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/experiments",
        help="Directory to save experimental comparison tables and plots.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for comparative experiment runner."""
    args = parse_args()
    pass


if __name__ == "__main__":
    main()

