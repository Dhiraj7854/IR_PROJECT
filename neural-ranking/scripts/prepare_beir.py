#!/usr/bin/env python3
"""Explicitly download an official BEIR dataset (never run automatically)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and unpack SciFact or NFCorpus from the official BEIR archive."
    )
    parser.add_argument("--dataset", choices=("scifact", "nfcorpus"), required=True)
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "beir"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from beir import util
    except ImportError as error:
        raise RuntimeError(
            "BEIR preparation requires the optional dependency: "
            "python -m pip install -e \".[beir]\""
        ) from error

    destination = Path(args.data_dir)
    destination.mkdir(parents=True, exist_ok=True)
    url = (
        "https://public.ukp.informatik.tu-darmstadt.de/BEIR/datasets/"
        f"{args.dataset}.zip"
    )
    util.download_and_unzip(url, str(destination))
    extracted = destination / args.dataset
    if not extracted.is_dir():
        raise FileNotFoundError(
            f"BEIR downloader completed but expected dataset directory is absent: {extracted}"
        )
    print(f"Prepared {args.dataset} at {extracted}")


if __name__ == "__main__":
    main()
