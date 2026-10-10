# Neural Passage Ranking: Experimental Report

## Introduction

<!-- Introduce the retrieval task, motivation, objectives, and contributions. -->

## Related Work

<!-- Summarize the cited work and explain its relevance to this project. -->

- **BM25:** Robertson and Zaragoza, “The Probabilistic Relevance Framework: BM25 and Beyond” (2009).
- **DSSM:** Huang et al., “Learning Deep Structured Semantic Models for Web Search using Clickthrough Data” (2013).
- **BERT for re-ranking:** Nogueira and Cho, “Passage Re-ranking with BERT” (2019).
- **MS MARCO:** Bajaj et al., “MS MARCO: A Human Generated MAchine Reading COmprehension Dataset” (2016).

## Methodology

<!-- Describe data preparation, corpus/splits, qrels, preprocessing, retrieval, -->
<!-- model architectures, training objectives, hyperparameters, and metrics. -->

## Experiments

<!-- Record hardware/software, exact commands, split sizes, and experimental setup. -->

### Main comparison

| System | MRR@10 | NDCG@10 | P@5 | Recall@100 | Median latency (ms/query) |
|---|---:|---:|---:|---:|---:|
| TF-IDF | 0.2943 | 0.3854 | 0.1091 | 0.9073 | 55.13 |
| BM25 | 0.3316 | 0.4230 | 0.1192 | 0.9087 | 153.47 |
| Bi-encoder | 0.4895 | 0.5790 | 0.1525 | 0.9820 | 7.42 |
| BM25 → Cross-encoder | 0.4016 | 0.4835 | 0.1272 | 0.7544* | 284.89 |

Metrics are evaluated over 750 test queries; latency is median over 200 test
queries in the Colab run. `*` The cross-encoder run has ranks 1–10 only, so
0.7544 is recall of the returned top-10, not full Recall@100. A depth-100
cross-encoder run is needed for a comparable Recall@100 value. The CSV reports
no configured suspicious-result flags. The extracted bundle did not include
the exact Colab `test_qrels.parquet`; these values are transcribed from the
Colab-generated comparison artifacts and have not been independently
recomputed against the local qrels.

Training metadata records one completed epoch (55 optimizer steps) for each
bi-encoder run; their best sampled-corpus validation MRR@10 values were 0.9705
and 0.9712 for hard and random negatives, respectively. The cross-encoder ran
658 optimizer steps in 273.59 seconds; its best validation MRR@10 was 0.4134
at step 329. The bi-encoder validation score uses a 2,000-passage sample and
is not the full-corpus retrieval score shown above.

### Hard vs. random negative sampling

| Strategy | Validation MRR@10 | Difference vs. hard |
|---|---:|---:|
| Hard negatives | 0.4968 | +0.0000 |
| Random negatives | 0.5260 | +0.0292 |

### Cross-encoder re-ranking depth

| BM25 candidate depth | Validation MRR@10 | Median latency (ms/query) |
|---:|---:|---:|
| 50 | 0.4451 | 203.83 |
| 100 | 0.4134 | 265.49 |
| 200 | 0.3764 | 344.82 |

### Query-length analysis

| Test query bucket | Query count | Word-count range | BM25 MRR@10 | BM25 → Cross-encoder MRR@10 |
|---|---:|---:|---:|---:|
| Short | 250 | 2–5 | 0.3664 | 0.4322 |
| Medium | 250 | 5–7 | 0.3029 | 0.3808 |
| Long | 250 | 7–22 | 0.3256 | 0.3919 |

## Results

![Test-set MRR@10 vs median latency](artifacts/results/pareto.png)

Detailed table and per-query metrics: `artifacts/results/comparison_table.md`
and `artifacts/results/per_query_metrics.csv`. Ablation plots:
`artifacts/results/ablations_negatives.png`,
`artifacts/results/ablations_rerank_depth.png`, and
`artifacts/results/ablations_query_length.png`.

![Hard vs random negative sampling](artifacts/results/ablations_negatives.png)

![Cross-encoder reranking depth](artifacts/results/ablations_rerank_depth.png)

![Query-length ablation](artifacts/results/ablations_query_length.png)

## Analysis

<!-- Write your interpretation of the main comparison, latency/quality tradeoffs, -->
<!-- suspicious-result checks, and statistical or qualitative observations here. -->

### Ablation analysis

<!-- Interpret the negative-sampling, re-ranking-depth, and query-length results -->
<!-- from artifacts/results/ablations.md. -->

## Limitations

<!-- Discuss dataset coverage, qrels incompleteness, compute constraints, -->
<!-- sampled validation corpus, latency environment, and generalizability. -->

## Conclusion

<!-- Summarize findings and future work after the results have been generated. -->

## References

1. Robertson, S., and Zaragoza, H. “The Probabilistic Relevance Framework: BM25 and Beyond.” *Foundations and Trends in Information Retrieval*, 2009.
2. Huang, P.-S. et al. “Learning Deep Structured Semantic Models for Web Search using Clickthrough Data.” *Proceedings of the 22nd ACM International Conference on Information & Knowledge Management*, 2013.
3. Nogueira, R., and Cho, K. “Passage Re-ranking with BERT.” arXiv:1901.04085, 2019.
4. Bajaj, P. et al. “MS MARCO: A Human Generated MAchine Reading COmprehension Dataset.” *Proceedings of the Workshop on Cognitive Computation: Integrating Neural and Symbolic Approaches*, 2016.
