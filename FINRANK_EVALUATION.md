# FinRank retrieval evaluation

| Configuration | P@10 | R@10 | F1@10 | MRR | MAP | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1_BM25 | 0.1924 | 1.0000 | 0.3135 | 0.7756 | 0.7273 | 0.7599 | 0.8166 |
| E2_Dense | 0.1924 | 1.0000 | 0.3135 | 0.7785 | 0.7316 | 0.7776 | 0.8181 |
| E3_Hybrid | 0.1924 | 1.0000 | 0.3135 | 0.7965 | 0.7425 | 0.7760 | 0.8281 |
| E4_Hybrid_CE | 0.1924 | 1.0000 | 0.3135 | 0.8558 | 0.8022 | 0.8369 | 0.8705 |

The table ranks each question’s annotated gold passages against its own curated hard negatives. It is not global retrieval or full SEC filing search. Metrics use binary in-record labels; P@k divides by k even when fewer than k candidates exist. Global retrieval is separately reported as known-gold recall and judged coverage in the CSV; unjudged passages are not labeled irrelevant.

Official random test contains 119 records; one duplicated question shared with train was excluded before running the models. No train/dev tuning was performed. See PROGRESS.md and the manifest for provenance, split overlaps, all aliases, model revisions and bootstrap intervals.

## Scope, provenance, and completeness

The official FinRank release at revision `916dc27e77b90e294ea1b3a2cc5b0abcf07392d1` is licensed CC BY-NC 4.0, with attribution to Mansouri and colleagues: https://github.com/datanxt/FinRank . It contains 1,185 manually authored questions over SEC filing passages, 22 tickers, 2,322 gold and 6,021 curated hard-negative occurrences, and 5,230 unique exact-text pooled passages. The official validator reported zero violations; all 8,343 original occurrence hashes match SHA-1 of NFC-normalized, stripped UTF-8 text. The pool is curated and does not represent all SEC filing text.

The official random test split contains 119 records. One question also appears in training and was excluded before scoring; all four methods use the remaining 118 queries. There was no model training or test-set parameter tuning. Gold passage text overlaps between random train/test (96 unique exact pooled texts) and dev/test (32); this limits claims about generalization to unseen filings or companies. Cross-record gold/hard-negative text reuse also exists. Candidate labels are applied only to their own question, and original passage occurrence IDs remain in the local alias-preserving corpus.

In the in-record regime, each method ranks a question's gold passages against its own curated hard negatives. Candidate-set sizes range from 6 to 10 (42 sets of 6, 40 of 7, 29 of 8, six of 9 and one of 10). Consequently every relevant candidate is present by rank 10, so R@10 is 1.0 and P@10/F1@10 are fixed by the number of positives; these metrics do not distinguish the methods here. BM25 uses the full pooled corpus for lexical statistics. Dense in-record scoring uses exact Qdrant search over the explicit candidate set; global dense retrieval uses Qdrant HNSW. Hybrid uses RRF with k=60; the reranker uses the existing BGE cross-encoder and the maximum score across explicit query/passage token windows.

## Global pooled retrieval: partial judgments

| Configuration | Known-gold recall@10 | Judgment coverage@10 |
|---|---:|---:|
| E1 BM25 | 0.3326 | 0.0890 |
| E2 Dense | 0.2620 | 0.0805 |
| E3 Hybrid | 0.3418 | 0.0966 |
| E4 Hybrid + cross-encoder | 0.3743 | 0.1017 |

Known-gold recall is the proportion of this question's annotated positives retrieved; it does not establish full relevance recall. Coverage is the fraction of retrieved passages explicitly judged for that question. Other retrieved passages are unjudged, not assumed irrelevant. Do not read in-record P@10, MRR, MAP or nDCG as global-pool metrics.

Paired bootstrap percentile intervals for in-record MRR differences (2,000 samples, seed 42) were: hybrid minus BM25 +0.0209 [−0.0181, +0.0610], hybrid minus dense +0.0180 [−0.0268, +0.0629], and hybrid+cross-encoder minus hybrid +0.0593 [+0.0007, +0.1199]. These exploratory intervals are not corrected for multiple comparisons and do not establish broad real-world performance.

The GPU was an NVIDIA RTX 4050 Laptop GPU, PyTorch 2.12.0+cu126. The encoder was `BAAI/bge-small-en-v1.5` revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`; the cross-encoder was `BAAI/bge-reranker-base` revision `2cfc18c9415c912f9d8155881c133215df768a70`. The collection was `finrank_passages`, with 5,230 indexed vectors, cosine similarity and HNSW m=16/ef_construct=100. Per-query predictions and aggregates are stored locally in `results/finrank_retrieval_predictions.jsonl`, `results/finrank_per_query_metrics.csv`, `results/finrank_retrieval_metrics.csv`, and `results/finrank_experiment_manifest.json`. The exact evaluated code is preserved in ignored `results/finrank_run_code/`.

Every saved per-query metric was independently recalculated from predictions and original per-question judgments, and aggregates were checked against those rows. No retrieval experiment was rerun during that verification. Financial RAG, numerical correctness and human claim-support validation are separate milestones; generation results from the small development pilot must not be presented as test performance.
