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

## Run lexical retrieval baselines

From the project root, run:

```bash
python scripts/run_baseline.py
```

This builds or loads the cached TF-IDF and BM25 indexes in `artifacts/`, logs
indexing time and per-query latency, and writes TREC-format runs for the
validation and test splits to `artifacts/runs/{tfidf,bm25}_{val,test}.txt`.

## Train and evaluate the bi-encoder in Colab

The processed parquet files are not checked into the repository. In Colab,
clone the repository, mount Drive, and copy `data/processed/` from Drive into
the project before training:

```python
!git clone https://github.com/Dhiraj7854/IR_PROJECT.git
%cd /content/IR_PROJECT/neural-ranking
from google.colab import drive
drive.mount("/content/drive")
!cp -r /content/drive/MyDrive/IR_PROJECT/neural-ranking/data/processed ./data/
%pip install "sentence-transformers==3.4.1" pandas pyarrow pyyaml
```

Run training and full-corpus evaluation from the project directory:

```bash
python scripts/train_biencoder.py
python scripts/eval_biencoder.py
```

Training uses the settings in `configs/config.yaml`, evaluates MRR@10 each
epoch on a 2,000-passage validation sample, and saves the best checkpoint to
`artifacts/biencoder/`. Full-corpus passage vectors are cached there at
`corpus_embeddings.npy`. Evaluation writes TREC runs to
`artifacts/runs/biencoder_val.txt` and `artifacts/runs/biencoder_test.txt`.
Copy `artifacts/biencoder/` and `artifacts/runs/` back to Drive if they need to
persist after the Colab runtime ends.

## Compare all systems on the test split

After producing the four test runs and saving both neural checkpoints, run:

```bash
python scripts/run_all_eval.py
```

The evaluator computes per-query and macro-average MRR@10, NDCG@10, P@5, and
Recall@100 with `ranx`. It measures median end-to-end latency over the first
200 test queries and writes comparison tables, per-query metrics, and a
log-latency Pareto plot under `artifacts/results/`. Suspicious MRR ranges are
reported as warnings in the log and comparison output.

With 3,500 triplets, batch size 64, and two epochs, training has about 110
optimizer steps, so the requested two-epoch configuration cannot produce a
500-step loss trace. The script logs a mean loss every 50 optimizer steps;
loss decrease is measured, not guaranteed for stochastic optimization.

## Train and evaluate the cross-encoder

Use the same Colab setup and processed data. The explicit PyTorch loop mines two
BM25 negatives per training query and initializes the backbone from
`cross-encoder/ms-marco-MiniLM-L-6-v2`:

```bash
python scripts/train_crossencoder.py
python scripts/eval_crossencoder.py
```

Training defaults to batch size 32, learning rate `2e-5`, two epochs, 10% linear
warmup, and CUDA fp16 when available. The best validation-MRR checkpoint is
saved in `artifacts/crossencoder/`. Evaluation re-ranks BM25's top 100 passages
and writes `artifacts/runs/crossencoder_{val,test}.txt`; it also reports
cross-encoder reranking milliseconds per query.

For a small overfit/sanity run, the training script supports `--max-train-queries
100 --epochs 10`; compare training loss and validation MRR in the logs. Use
`--no-fp16` to debug on CPU, though full training is intended for a Colab GPU.
