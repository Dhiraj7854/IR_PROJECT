# Two-Stage Neural Ranking System

A Python framework for a two-stage neural ranking system (BM25 retrieval + cross-encoder re-ranking) compared against TF-IDF and bi-encoder baselines.

## Project Structure

```
neural-ranking/
├── configs/
│   └── config.yaml          # Dataset, model, training, and top-k retrieval configuration
├── src/
│   ├── __init__.py
│   ├── config.py            # Typed dataclass config loader
│   ├── data.py              # Data loading & preprocessing
│   ├── baseline.py          # TF-IDF & BM25 sparse baselines
│   ├── biencoder.py         # Dense bi-encoder candidate retrieval
│   ├── crossencoder.py      # Cross-encoder joint scoring re-ranker
│   ├── evaluate.py          # Evaluation metrics (MRR, NDCG, MAP, Recall)
│   └── experiments.py       # Comparative benchmarking pipeline
├── scripts/                 # CLI entry points with argparse skeletons
│   ├── run_baselines.py
│   ├── train_biencoder.py
│   ├── train_crossencoder.py
│   ├── rerank.py
│   ├── evaluate.py
│   └── run_experiments.py
├── notebooks/               # Jupyter notebooks for analysis
├── tests/                   # Test suite
│   ├── __init__.py
│   └── test_config.py       # Configuration loading tests
├── data/                    # Corpus and query datasets
├── artifacts/               # Checkpoints, runs, and evaluation metrics
├── requirements.txt         # Pinned dependency specifications
├── pyproject.toml           # Project build specifications
└── .gitignore               # Ignored files (data, artifacts, cache, venv)
```

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Verify configuration
python -c "from src.config import load_config; print(load_config('configs/config.yaml'))"

# Run tests
pytest tests/
```

