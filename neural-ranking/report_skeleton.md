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

<!-- Populate from artifacts/results/comparison_table.csv. -->

| System | MRR@10 | NDCG@10 | P@5 | Recall@100 | Median latency (ms/query) |
|---|---:|---:|---:|---:|---:|
| TF-IDF | 0.2943 | 0.3854 | 0.1091 | 0.9073 | 35.95 |
| BM25 | 0.3316 | 0.4230 | 0.1192 | 0.9087 | 97.32 |
| Bi-encoder | Not available | Not available | Not available | Not available | Not available |
| BM25 → Cross-encoder | Not available | Not available | Not available | Not available | Not available |

The lexical metrics are from the current TREC test runs evaluated against
`data/processed/test_qrels.parquet`. Lexical latency is median time over 200
test queries on the local CPU; see `artifacts/results/` for per-query metric
JSON and `artifacts/results/baseline_latency_local.json` for latency provenance.
No neural test runs/checkpoints or complete comparison outputs were available
when this skeleton was prepared.

### Hard vs. random negative sampling

<!-- Populate from artifacts/results/ablations.md. -->

| Strategy | Validation MRR@10 | Difference vs. hard |
|---|---:|---:|
| Hard negatives | Not available | Not available |
| Random negatives | Not available | Not available |

### Cross-encoder re-ranking depth

<!-- Populate from artifacts/results/ablations.md. -->

| BM25 candidate depth | Validation MRR@10 | Median latency (ms/query) |
|---:|---:|---:|
| 50 | Not available | Not available |
| 100 | Not available | Not available |
| 200 | Not available | Not available |

### Query-length analysis

<!-- Populate from artifacts/results/ablations.md. -->

| Test query bucket | Query count | Word-count range | BM25 MRR@10 | BM25 → Cross-encoder MRR@10 |
|---|---:|---:|---:|---:|
| Short | Not available | Not available | Not available | Not available |
| Medium | Not available | Not available | Not available | Not available |
| Long | Not available | Not available | Not available | Not available |

## Results

<!-- Insert the final comparison table and figures. Source the comparison from -->
<!-- artifacts/results/comparison_table.md and artifacts/results/pareto.png. -->

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
