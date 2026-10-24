# Neural Passage Ranking: Experimental Report

## Introduction

Passage retrieval must identify a small set of relevant documents from a large
corpus while balancing ranking quality, coverage, and response time. Lexical
retrievers such as TF-IDF and BM25 are efficient and interpretable, but their
matching is sensitive to wording. Neural bi-encoders can represent queries and
passages in a shared semantic space and support efficient vector retrieval.
Cross-encoders jointly process a query and candidate passage and can refine a
short candidate list, at a higher per-candidate inference cost.

This project compares TF-IDF, BM25, a dense bi-encoder, BM25-to-cross-encoder
reranking, and reciprocal-rank fusion (RRF) on a 100,000-passage MS MARCO
corpus. It reports top-rank quality, recall, paired query-level uncertainty,
and measured latency. The current saved bundle also exposes an important
training issue: its legacy cross-encoder discarded a pretrained
sequence-classification head, and its test MRR@10 is below the bi-encoder's.
The implementation now initializes a fresh model from an NLI-trained
sequence-classification checkpoint and preserves its head, but new training
and test results must still be produced on Colab before any improvement is
claimed.

## Related Work

TF-IDF and BM25 are sparse lexical retrieval baselines; BM25 extends the
probabilistic relevance framework with term-frequency saturation and document
length normalization [1]. Neural ranking research includes the
Deep Structured Semantic Model (DSSM), which learns semantic representations
from query-document interaction data [2], and BERT-based passage reranking,
which jointly scores query-passage pairs [3]. Sentence-BERT makes transformer
representations practical for semantic search by enabling independently
encoded sentences [4]; Dense Passage Retrieval (DPR) applies a dual-encoder
architecture to open-domain retrieval [5].

BEIR is a heterogeneous benchmark for evaluating zero-shot retrieval transfer
across domains [6]. Reciprocal-rank fusion combines ranked lists without
requiring their raw scores to be calibrated [7]. The selected cross-encoder
initializer is trained for natural-language inference (NLI), using the
entailment-versus-contradiction logit margin as an initial relevance score;
this follows the three-way label setup studied in SNLI and MultiNLI [8, 9].

## Methodology

### Data and splits

The restored MS MARCO bundle contains a deterministic 100,000-passage corpus,
3,500 training queries, 750 validation queries, and 750 test queries with
associated relevance judgments. Reported test metrics are macro-averaged over
the 750 saved test queries. All systems are evaluated against the same saved
test qrels; the result manifest records query and qrels hashes.

### Retrieval systems

- **TF-IDF:** scikit-learn TF-IDF with English stop-word filtering,
  sublinear term frequency, and L2 normalization.
- **BM25:** BM25Okapi with the project's lowercase, alphanumeric tokenizer
  and English stop-word removal.
- **Bi-encoder:** `sentence-transformers/all-MiniLM-L6-v2`, fine-tuned with
  Multiple Negatives Ranking Loss and hard negatives, then used to retrieve
  from precomputed corpus embeddings.
- **Cross-encoder:** the saved legacy run used a pretrained MS MARCO MiniLM
  encoder with a newly initialized scalar head. The prior implementation
  loaded the base encoder with `AutoModel`, so it did not retain the
  pretrained sequence-classification head. Its MS MARCO pretraining also
  means that its test scores should be treated as exploratory, not as an
  uncontaminated held-out estimate.
- **Updated cross-encoder protocol:** training now defaults to
  `cross-encoder/nli-MiniLM2-L6-H768`, preserves its three-class classifier,
  scores each pair by entailment logit minus contradiction logit, and
  truncates only the passage so the query is retained. This is a new
  experiment; no results from it are included below.
- **Hybrid RRF:** fuses BM25 and bi-encoder rankings using one-based ranks and
  configurable weights, without combining their raw scores. Cross-encoder
  reranking is applied only after candidate generation.

The ranking metrics are MRR@10, NDCG@10, precision at five, Recall@10,
Recall@100, and MAP@10. Confidence intervals are paired percentile-bootstrap
intervals over query-level score differences (2,000 resamples, seed 42).
They express query-sampling variation on this single test split, not
variation across training seeds. Latency is the median measured in the
original Colab run over 200 queries and is hardware-dependent.

## Experiments

The results below are the saved legacy run, not a new training run. The
cross-encoder rows use the legacy MS MARCO-initialized model described above.

### Main test comparison

| System | MRR@10 | NDCG@10 | P@5 | Recall@10 | Recall@100 | MAP@10 | Median latency (ms/query) |
|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF | 0.2943 | 0.3854 | 0.1091 | 0.6910 | 0.9073 | 0.2874 | 45.96 |
| BM25 | 0.3316 | 0.4230 | 0.1192 | 0.7269 | 0.9087 | 0.3247 | 151.89 |
| Bi-encoder | 0.4892 | 0.5788 | 0.1528 | 0.8720 | 0.9820 | 0.4831 | 8.24 |
| BM25 → Cross-encoder (legacy) | 0.3970 | 0.4772 | 0.1251 | 0.7444 | 0.9087 | 0.3902 | 272.23 |
| Hybrid RRF | 0.4822 | 0.5687 | 0.1533 | 0.8531 | 0.9947 | 0.4746 | 149.74 |
| Hybrid RRF → Cross-encoder (legacy) | 0.3402 | 0.4237 | 0.1109 | 0.7086 | 0.9947 | 0.3319 | 209.05 |
| BM25 → Cross-encoder (fixed candidates, depth 100 / rerank 50; legacy) | 0.4312 | 0.5072 | 0.1368 | 0.7620 | 0.9087 | 0.4231 | 199.94 |

The saved runs have valid top-100 depth. The original configuration hash
differs from the current configuration because the cross-encoder initializer
has since been changed; the run manifest records its original effective
configuration.

### Paired MRR@10 differences

Positive differences favor the named system. Each interval compares the same
750 test queries; values below were regenerated from the saved per-query
metrics and are also stored in
`artifacts/results/paired_differences.json`.

| Comparison | Mean difference | Paired bootstrap 95% CI |
|---|---:|---:|
| Bi-encoder vs BM25 | +0.1575 | [+0.1267, +0.1912] |
| Bi-encoder vs TF-IDF | +0.1949 | [+0.1633, +0.2265] |
| BM25 → Cross-encoder (legacy) vs BM25 | +0.0654 | [+0.0287, +0.1020] |
| BM25 → Cross-encoder (legacy) vs TF-IDF | +0.1028 | [+0.0664, +0.1390] |
| Hybrid RRF vs BM25 | +0.1506 | [+0.1302, +0.1714] |
| Hybrid RRF vs TF-IDF | +0.1880 | [+0.1660, +0.2116] |
| Hybrid RRF → Cross-encoder (legacy) vs BM25 | +0.0085 | [-0.0302, +0.0453] |
| Hybrid RRF → Cross-encoder (legacy) vs TF-IDF | +0.0459 | [+0.0099, +0.0820] |
| BM25 → Cross-encoder, fixed candidates (legacy) vs TF-IDF | +0.1370 | [+0.1015, +0.1734] |

### TF-IDF-specific error analysis

The saved per-query runs illustrate vocabulary mismatch and morphology
limitations in lexical matching:

- For **“what is a length of a day”**, the judged relevant passage describes
  “a day” as a “unit of time” and an “interval equal to 24 hours.” TF-IDF
  retrieves it at rank 62, outside the MRR@10 cutoff; the bi-encoder retrieves
  it at rank 1.
- For **“what is bitcoins”**, the relevant passage begins “Bitcoin is a
  decentralized digital currency.” The plural query token does not match the
  singular passage token under this TF-IDF tokenizer, and the relevant passage
  is absent from TF-IDF's top 100; the bi-encoder places it at rank 1.

These are examples from the saved test run, not a claim that neural retrieval
always wins. TF-IDF remains a useful, fast baseline, and the held-out test
set should not be used to tune the new models.

### Candidate-depth diagnosis (validation)

Candidate recall is measured before reranking; final MRR@10 is measured after
reranking. RRF depth is per source retriever, so the fused candidate union can
contain up to twice that number of unique passages.

| Candidate source | Candidate depth | Rerank depth | Candidate recall | Final MRR@10 |
|---|---:|---:|---:|---:|
| BM25 | 10 | 10 | 0.7402 | 0.4602 |
| BM25 | 50 | 50 | 0.8920 | 0.4406 |
| BM25 | 100 | 50 | 0.9244 | 0.4407 |
| RRF | 10 per source | 10 | 0.9398 | 0.4660 |
| RRF | 50 per source | 50 | 0.9953 | 0.3177 |
| RRF | 100 per source | 50 | 0.9980 | 0.3235 |

The RRF candidates have high recall, but the legacy cross-encoder did not
preserve the strong RRF ordering. This motivates retraining and controlled
reranker evaluation; one seed and one validation split do not establish a
general property of cross-encoders.

### Training history and pending experiments

The saved hard-negative bi-encoder metadata records one completed epoch and
55 optimizer steps. Its best validation MRR@10 (0.9702) was measured on the
training script's sampled 2,000-passage validation corpus and is not
comparable to full-corpus test scores. The legacy cross-encoder records 658
optimizer steps; its best validation MRR@10 was 0.4056 at step 329, followed
by 0.3720 at step 658.

No result is claimed yet for the NLI-initialized cross-encoder, longer
bi-encoder training, hard-versus-random negative comparison, query-length
analysis, query enhancement, or BEIR SciFact. Query enhancement previously
failed with `SIGKILL`; reproducing it exposed an unbounded output-path list
construction that raised `MemoryError`. The evaluator now creates a finite
path list and builds only the requested sparse lexical index, including a
compact sparse-postings BM25 implementation. A 100,000-passage / 750-query
validation smoke run completed locally as a code-path check, but it is not a
new reported benchmark result. The test-split experiment still needs to be
run in Colab; the exact commands are in the project README.

## Results

![Test-set MRR@10 vs median latency](artifacts/results/pareto.png)

Detailed outputs are in `artifacts/results/`: `comparison_table.md`,
`comparison_table.csv`, `per_query_metrics.csv`, `paired_differences.json`,
and `evaluation_manifest.json`. Candidate-depth metrics are stored beside
their ranked runs in `artifacts/runs/`.

## Analysis

The bi-encoder has the highest saved test MRR@10 (0.4892) and NDCG@10
(0.5788), and the lowest measured median latency (8.24 ms/query). Its MRR
advantage over TF-IDF is +0.1949 with a paired 95% CI of [+0.1633, +0.2265].
Hybrid RRF approaches the bi-encoder's MRR@10 (0.4822) and has the highest
Recall@100 (0.9947), but does not exceed the bi-encoder on top-rank quality.

The legacy BM25-to-cross-encoder run improves MRR@10 over BM25, but remains
below the bi-encoder and incurs much higher measured latency. The hybrid
cross-encoder result has an interval versus BM25 that crosses zero and a
lower MRR@10 than unreranked RRF. These results, together with the discarded
pretrained classifier head and potential MS MARCO pretraining overlap, are
reasons to treat the old reranker as a diagnostic baseline, not a final
cross-encoder comparison. The NLI-initialized reranker is intended to address
the initialization problem; its validation behavior and held-out test score
remain unknown until the new experiment completes.

The paired intervals compare query samples from one saved test set. They are
not seed-level confidence intervals and do not remove dataset or
pretraining-related bias. Latencies came from the original Colab hardware and
are not portable to other devices.

## Limitations

The saved evaluation represents one MS MARCO test split and one recorded
training seed; it does not establish transfer to other domains or robustness
across seeds. The legacy bi-encoder's validation score was computed on a
sampled corpus. The legacy cross-encoder has a short training history and
potential benchmark-pretraining overlap. The saved Space benchmark table is
an offline Colab evaluation; the live Space runs on its separately packaged
corpus and checkpoints, by default on CPU. Its live latency and rankings
should not be compared as if they came from the table's offline setup.

BEIR, retraining, negative-sampling, query-length, and query-enhancement
results remain pending. The compact index and finite path-list changes have
been exercised locally on the complete corpus's validation queries; Colab
runtime measurements and held-out test results are still needed. Multiple
seeds and a fully recorded software/hardware manifest are also needed for
stronger conclusions.

## Conclusion

On the saved MS MARCO test split, dense retrieval substantially improves
top-rank quality over both lexical baselines while the hybrid RRF system
provides comparable MRR and the highest candidate coverage. TF-IDF's
vocabulary mismatch is visible in individual test queries, though it remains
a transparent and competitive baseline. The legacy cross-encoder results
expose an implementation and experimental weakness rather than settling the
value of reranking: the pretrained classifier head was discarded, validation
quality declined late in training, and the score is below the bi-encoder.

This project now has a separate NLI-initialized classifier path, paired
comparisons against TF-IDF, a lower-memory query-enhancement index path, and
documented Colab procedures for the missing experiments. Until those
experiments are run and their artifacts reviewed, no new cross-encoder,
generalization, or ablation improvement is claimed.

## References

1. Robertson, S., and Zaragoza, H. “The Probabilistic Relevance Framework:
   BM25 and Beyond.” *Foundations and Trends in Information Retrieval*, 2009.
2. Huang, P.-S. et al. “Learning Deep Structured Semantic Models for Web
   Search using Clickthrough Data.” *Proceedings of the 22nd ACM International
   Conference on Information & Knowledge Management*, 2013.
3. Nogueira, R., and Cho, K. “Passage Re-ranking with BERT.” arXiv:1901.04085,
   2019.
4. Reimers, N., and Gurevych, I. “Sentence-BERT: Sentence Embeddings using
   Siamese BERT-Networks.” *Proceedings of EMNLP-IJCNLP*, 2019.
5. Karpukhin, V. et al. “Dense Passage Retrieval for Open-Domain Question
   Answering.” *Proceedings of EMNLP*, 2020.
6. Thakur, N. et al. “BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation
   of Information Retrieval Models.” *NeurIPS Datasets and Benchmarks*, 2021.
7. Cormack, G. V., Clarke, C. L. A., and Buettcher, S. “Reciprocal Rank Fusion
   Outperforms Condorcet and Individual Rank Learning Methods.” *Proceedings
   of SIGIR*, 2009.
8. Bowman, S. R. et al. “A Large Annotated Corpus for Learning Natural
   Language Inference.” *Proceedings of EMNLP*, 2015.
9. Williams, A., Nangia, N., and Bowman, S. R. “A Broad-Coverage Challenge
   Corpus for Sentence Understanding through Inference.” *Proceedings of
   NAACL-HLT*, 2018.
10. Bajaj, P. et al. “MS MARCO: A Human Generated MAchine Reading
    COmprehension Dataset.” *Proceedings of the Workshop on Cognitive
    Computation: Integrating Neural and Symbolic Approaches*, 2016.
