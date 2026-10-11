# Two-Stage Neural Passage Ranking

An MS MARCO passage-retrieval project comparing TF-IDF, BM25, a fine-tuned
bi-encoder, and BM25 followed by a neural cross-encoder reranker. It includes
data preparation, model training and evaluation, retrieval ablations, and a
Gradio search dashboard.

## Project layout

```text
IR_PROJECT/
├── neural-ranking/
│   ├── app/                 # Gradio search and benchmark dashboard
│   ├── configs/             # Model and training configuration
│   ├── data/processed/      # Prepared corpus, query splits, and qrels
│   ├── artifacts/           # Indexes, checkpoints, runs, and result tables
│   ├── scripts/             # Prepare, train, evaluate, and experiment CLIs
│   ├── src/                 # Retrieval, training, and evaluation modules
│   └── tests/
└── readme.md
```

## Quick start

The full workflow is documented in
[`neural-ranking/README.md`](neural-ranking/README.md). In short, install
dependencies, prepare the MS MARCO data, train or restore the model
checkpoints, then launch the demo. GPU training and full-corpus embedding are
intended for Google Colab or another CUDA-enabled environment.

```bash
cd neural-ranking
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
python scripts/prepare_data.py --config configs/config.yaml --output_dir data/processed
```

Follow the detailed guide for the exact training and evaluation commands.
Generated data, checkpoints, runs, and metrics are not committed to Git; see
the guide for restoring outputs from Google Drive.

## Gradio dashboard

The app in [`neural-ranking/app/demo.py`](neural-ranking/app/demo.py) compares
retrieval systems on the same query. It provides:

- BM25, TF-IDF, bi-encoder, and BM25 → cross-encoder result cards with passage
  IDs, scores, highlighted query terms, rank changes, and latency.
- Search controls for the systems to compare, result count, and BM25 candidate
  depth.
- A benchmark tab with responsive, in-page SVG charts for quality vs. latency,
  negative-sampling ablations, reranking depth, and query length. These charts
  are rendered in the page rather than embedded as static PNGs; chart points
  expose exact values on hover.
- A search-time overlap donut and latency comparison, plus a “How it works”
  guide.

Launch locally from `neural-ranking/`:

```bash
python app/demo.py
```

In Google Colab, the app automatically requests a share link. Open the
`https://....gradio.live` URL printed in the cell output; `127.0.0.1` points to
the Colab VM, not your local computer. Outside Colab, use `--share` to request
a public link:

```bash
python app/demo.py --share
```

The share link is publicly accessible while active. Only use it with data that
is appropriate to expose.

## Latest evaluation

The recorded Colab test comparison is in
[`neural-ranking/artifacts/results/comparison_table.csv`](neural-ranking/artifacts/results/comparison_table.csv).
The bi-encoder had the highest MRR@10 (0.4895) and lowest median latency
(7.42 ms/query) in that run. The cross-encoder run currently contains only
top-10 results, so its reported Recall@100 is not a true top-100 recall
measurement. See the detailed README for all metrics, ablations, and
reproducibility caveats.
