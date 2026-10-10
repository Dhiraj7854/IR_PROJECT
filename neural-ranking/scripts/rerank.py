"""Placeholder script for running stage-2 cross-encoder re-ranking on stage-1 candidates."""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for candidate re-ranking."""
    parser = argparse.ArgumentParser(
        description="Re-rank stage-1 retrieved candidate passages with a cross-encoder."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--candidate_run",
        type=str,
        default="artifacts/stage1_candidates.tsv",
        help="Path to stage 1 candidate run file.",
    )
    parser.add_argument(
        "--rerank_depth",
        type=int,
        default=50,
        help="Number of candidates to re-rank per query.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="artifacts/reranked_run.tsv",
        help="Path to save re-ranked run output.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for candidate re-ranking script."""
    args = parse_args()
    pass


if __name__ == "__main__":
    main()

