"""Placeholder script for evaluating lexical retrieval baselines (TF-IDF and BM25)."""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for baseline retrieval."""
    parser = argparse.ArgumentParser(
        description="Run lexical baselines (TF-IDF / BM25) on the retrieval corpus."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["bm25", "tfidf", "both"],
        default="both",
        help="Baseline retrieval method to execute.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Number of candidate passages to retrieve per query.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/baselines",
        help="Directory to save retrieval runs and metrics.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for baseline retrieval script."""
    args = parse_args()
    pass


if __name__ == "__main__":
    main()

