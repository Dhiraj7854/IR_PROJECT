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

The restored experiment bundle contains the 100,000-passage MS MARCO corpus,
750-query validation and test splits, qrels, train triplets, model checkpoints,
ranked runs, and evaluation outputs. Test metrics below use 750 queries;
latency is median over 200 queries in the original Colab runtime.

### Main test comparison

| System | MRR@10 | NDCG@10 | P@5 | Recall@10 | Recall@100 | MAP@10 | Median latency (ms/query) |
|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF | 0.2943 | 0.3854 | 0.1091 | 0.6910 | 0.9073 | 0.2874 | 45.96 |
| BM25 | 0.3316 | 0.4230 | 0.1192 | 0.7269 | 0.9087 | 0.3247 | 151.89 |
| Bi-encoder | 0.4892 | 0.5788 | 0.1528 | 0.8720 | 0.9820 | 0.4831 | 8.24 |
| BM25 → Cross-encoder | 0.3970 | 0.4772 | 0.1251 | 0.7444 | 0.9087 | 0.3902 | 272.23 |
| Hybrid RRF | 0.4822 | 0.5687 | 0.1533 | 0.8531 | 0.9947 | 0.4746 | 149.74 |
| Hybrid RRF → Cross-encoder | 0.3402 | 0.4237 | 0.1109 | 0.7086 | 0.9947 | 0.3319 | 209.05 |
| BM25 → Cross-encoder (candidate depth 100, rerank depth 50) | 0.4312 | 0.5072 | 0.1368 | 0.7620 | 0.9087 | 0.4231 | 199.94 |

The saved test qrels and query SHA-256 hashes match the restored files. Every
listed run has valid top-100 depth. The effective configuration in the
manifest matches the current `configs/config.yaml` values, although its raw
configuration-file hash differs.

Paired bootstrap differences from BM25 (750 paired queries, 2,000 resamples):
bi-encoder MRR@10 +0.1575 (95% CI +0.1267 to +0.1912), and hybrid RRF MRR@10
+0.1506 (95% CI +0.1302 to +0.1714). These intervals represent query-sampling
variation on this test split, not variation across training seeds.

### Candidate-depth comparison (validation)

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

The RRF candidates have high recall, but the cross-encoder reranked output did
not retain the strong RRF ordering. Candidate and rerank depths vary together
for some sweep points, so this is descriptive rather than a completely
controlled reranker-depth experiment.

### Training metadata and unavailable experiments

The saved hard-negative bi-encoder metadata records one completed epoch and
55 optimizer steps. Its best validation MRR@10 (0.9702) used the training
script's sampled 2,000-passage validation corpus and is not directly
comparable to the full-corpus test results. Cross-encoder training records
658 optimizer steps and 243.57 seconds; its best validation MRR@10 was 0.4056.

Query enhancement was attempted but Colab killed the process with `SIGKILL`
while it was loading or building the lexical index. No query-enhancement
metrics were produced. BEIR, relevance-feedback, query-length, and
random-negative evaluation artifacts are absent from this bundle and remain
unreported.

## Results

![Test-set MRR@10 vs median latency](artifacts/results/pareto.png)

Detailed outputs: `artifacts/results/comparison_table.md`,
`comparison_table.csv`, `per_query_metrics.csv`, `paired_differences.json`,
and `evaluation_manifest.json`. Candidate-depth metrics are stored beside
their ranked runs in `artifacts/runs/`.

## Analysis

The bi-encoder has the strongest test MRR@10 (0.4892) and NDCG@10 (0.5788),
with the lowest measured median latency (8.24 ms/query). Hybrid RRF markedly
improves on BM25 (MRR@10 0.4822 vs. 0.3316) and has the highest Recall@100
(0.9947), but does not exceed the bi-encoder's top-rank quality.

BM25-to-cross-encoder reranking improves MRR@10 over BM25. RRF-to-cross-encoder
reranking instead lowers MRR@10 from 0.4822 to 0.3402 despite candidate recall
of 0.9947. This motivates investigation of candidate ordering and reranker
behavior; it does not show that cross-encoders are inherently worse.

The paired 95% bootstrap interval for RRF minus BM25 MRR@10 is +0.1302 to
+0.1714; for bi-encoder minus BM25 it is +0.1267 to +0.1912. No configured
suspicious-result flags were raised in the saved comparison. Latencies reflect
the original Colab run and are not hardware-independent.

## Limitations

This is one MS MARCO test split and one recorded training seed; it does not
establish transfer to BEIR or robustness across seeds. The sampled-corpus
bi-encoder validation score differs from full-corpus final evaluation.
Query-enhancement evaluation exceeded available Colab memory, and no complete
hardware/software latency manifest was captured.

## Conclusion

<!-- Summarize findings and future work after the results have been generated. -->

## References

1. Robertson, S., and Zaragoza, H. “The Probabilistic Relevance Framework: BM25 and Beyond.” *Foundations and Trends in Information Retrieval*, 2009.
2. Huang, P.-S. et al. “Learning Deep Structured Semantic Models for Web Search using Clickthrough Data.” *Proceedings of the 22nd ACM International Conference on Information & Knowledge Management*, 2013.
3. Nogueira, R., and Cho, K. “Passage Re-ranking with BERT.” arXiv:1901.04085, 2019.
4. Bajaj, P. et al. “MS MARCO: A Human Generated MAchine Reading COmprehension Dataset.” *Proceedings of the Workshop on Cognitive Computation: Integrating Neural and Symbolic Approaches*, 2016.
