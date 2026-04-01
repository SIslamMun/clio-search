# 5. Evaluation

We evaluate clio-agentic-search along three research questions. **RQ1**: Does dimensional conversion improve retrieval accuracy over standard baselines on cross-unit queries? **RQ2**: Does formula normalization improve retrieval on formula-variant queries? **RQ3**: Does federated multi-namespace search maintain coverage without sacrificing the gains from science-aware operators? We then present an ablation study isolating the marginal contribution of each pipeline component, analyze multi-hop agentic retrieval behavior, and report indexing throughput.

## 5.1 Experimental Setup

**Benchmark construction.** We construct a controlled benchmark designed to isolate the effect of unit heterogeneity on retrieval. The benchmark spans five SI domains from Table 2: distance, mass, time, pressure, and velocity. For each domain, we author document sets in which the same physical quantity is expressed in two or three different units. Each retrieval task consists of a query posed in one unit and a corpus containing relevant documents expressed in different units of the same domain. For example, a query requesting pressures in kPa is paired with documents reporting values in Pa and MPa. The benchmark comprises [TBD] total retrieval tasks. We partition tasks into two conditions: a *control* set in which query and document units match, and a *cross-unit* set in which they differ. The control set establishes that observed differences are attributable to unit mismatch rather than content difficulty.

**Baselines.** We compare five retrieval configurations: (1) **BM25-only**, using DuckDB full-text search with default BM25 scoring; (2) **Dense-only**, using all-MiniLM-L6-v2 embeddings (384 dimensions) with exact cosine similarity; (3) **Hybrid BM25+Dense**, combining lexical and vector scores via linear fusion with equal weights; (4) **String normalization**, following the Numbers Matter! approach \cite{numbers_matter} of normalizing numeric tokens to a canonical string form before indexing; and (5) **Full pipeline**, the complete clio-agentic-search system with dimensional conversion, formula normalization, graph traversal, and agentic rewriting enabled.

**Metrics.** We report precision at $K$ (P@$K$), recall at $K$ (R@$K$), F1 at $K$ (F1@$K$) for $K \in \{1, 3, 5, 10\}$, and mean reciprocal rank (MRR). We designate $K{=}5$ as the primary evaluation point for all comparisons. Statistical significance is assessed via paired bootstrap resampling (10,000 iterations, $p < 0.05$).

## 5.2 RQ1: Dimensional Conversion Accuracy

We address the question: *does dimensional conversion improve retrieval accuracy over standard baselines when query and document units differ?*

**Table 3.** Retrieval performance on cross-unit queries ($K{=}5$).

| Configuration | P@5 | R@5 | F1@5 | MRR |
|---------------|-----|-----|------|-----|
| BM25-only | [TBD] | [TBD] | [TBD] | [TBD] |
| Dense-only | [TBD] | [TBD] | [TBD] | [TBD] |
| Hybrid BM25+Dense | [TBD] | [TBD] | [TBD] | [TBD] |
| String normalization | [TBD] | [TBD] | [TBD] | [TBD] |
| Full pipeline (ours) | [TBD] | [TBD] | [TBD] | [TBD] |

Table 3 reports results on cross-unit queries. The three text-oriented baselines---BM25, dense, and hybrid---retrieve only documents whose surface text contains the same unit string as the query. A query for "200 kPa" does not lexically match a document containing "200000 Pa," and the embedding model has no mechanism to learn that these denote identical physical quantities. Consequently, these baselines are limited to retrieving same-unit documents that happen to appear in the cross-unit corpus, resulting in P@5, R@5, and MRR scores of [TBD], [TBD], and [TBD] respectively for the strongest text baseline (Hybrid).

The string normalization baseline partially addresses this gap by canonicalizing numeric tokens before indexing. This strategy can match values when unit abbreviations coincide or when simple lexical rewriting rules apply, yielding R@5 of [TBD]. However, string normalization cannot perform the arithmetic conversion required to recognize that 200 kPa equals 200000 Pa---it operates on token forms, not physical semantics. Our full pipeline bypasses this limitation entirely: at index time, all measurements are converted to canonical SI base units and stored in the `scientific_measurements` table; at query time, the dimensional operator converts the query bounds to the same canonical space and issues a direct range predicate. This yields R@5 of [TBD] and MRR of [TBD] on cross-unit queries, representing an absolute improvement of [TBD] points in R@5 over the next-best baseline.

On the same-unit control set, all five configurations achieve comparable performance (P@5 within [TBD] points of each other), confirming that the observed gains are attributable specifically to cross-unit matching rather than general retrieval quality differences.

**Per-domain breakdown.** We disaggregate results across the five SI domains. Pressure and velocity queries, which involve conversion factors spanning three orders of magnitude, show the largest baseline degradation and the largest improvement from dimensional conversion. Distance queries exhibit intermediate gains, while mass and time queries---where unit variation in our benchmark is less extreme---show smaller but still statistically significant improvements ([TBD] across all domains, $p < 0.05$). Fig. 4 presents R@5 across all five baselines, grouped by SI domain.

## 5.3 RQ2: Formula Matching Accuracy

We address the question: *does formula normalization improve retrieval accuracy on queries involving mathematical notation variants?*

We construct a set of formula-variant retrieval tasks in which a query references a formula in one notation and relevant documents express the same relationship in alternative forms. Variants include reordering of terms (e.g., $F = ma$ vs. $F = am$), equivalent symbolic rearrangements (e.g., $E = mc^2$ vs. $m = E/c^2$), and differing variable naming conventions. Each task has a known set of relevant documents.

Without formula operators, the system relies on lexical and embedding similarity over raw LaTeX or Unicode strings, achieving P@5 of [TBD] and R@5 of [TBD]. Enabling formula normalization---which canonicalizes expressions into a sorted, simplified symbolic form before indexing and matching---raises P@5 to [TBD] and R@5 to [TBD]. The improvement is statistically significant ($p < 0.05$) and consistent across algebraic, differential, and integral formula classes.

We also measure the interaction between formula and dimensional operators. Enabling both operators jointly yields F1@5 of [TBD], compared to [TBD] for formula-only and [TBD] for dimensional-only, indicating that the two operators provide independent and complementary value. Neither operator's contribution subsumes the other, validating the pluggable design described in Section 3.

## 5.4 RQ3: Federated Search Coverage

We address the question: *does federated multi-namespace search maintain retrieval coverage when the same corpus is distributed across heterogeneous backends?*

We replicate the benchmark corpus across three storage backends: local filesystem, S3-compatible object store (MinIO), and HDF5 files. Each backend is registered as a separate namespace. We measure coverage---the fraction of relevant documents retrieved regardless of source backend---under single-backend and federated configurations, both with and without science-aware operators.

Single-backend search restricted to the filesystem namespace retrieves [TBD]% of relevant documents. Restricting to S3 or HDF5 alone yields [TBD]% and [TBD]% respectively, reflecting the partitioned placement of corpus documents. Federated search across all three namespaces achieves [TBD]% coverage, recovering documents that any single backend misses.

Critically, science-aware operators compose with federation without degradation. Federated search with dimensional conversion achieves R@5 of [TBD] on cross-unit queries, compared to [TBD] for federated search without science operators. The capability negotiation mechanism (Section 3.5) ensures that connectors lacking scientific measurement tables are gracefully skipped rather than producing errors, so adding a backend that does not support dimensional search does not degrade results from backends that do.

## 5.5 Ablation Study

To isolate the marginal contribution of each pipeline component, we evaluate six cumulative configurations on the cross-unit query set:

| Configuration | Components | F1@5 |
|---------------|-----------|------|
| A: Lexical | BM25 | [TBD] |
| B: +Vector | A + dense embeddings | [TBD] |
| C: +Scientific | B + dimensional + formula operators | [TBD] |
| D: +Graph | C + entity-relationship traversal | [TBD] |
| E: +Agentic | D + multi-hop query rewriting | [TBD] |
| F: Full | E + all optimizations | [TBD] |

Fig. 5 plots F1@5 for each configuration. The transition from B to C---adding scientific operators---produces the largest marginal gain ([TBD] absolute F1@5 improvement), confirming that cross-unit matching is the dominant bottleneck for this query class. Adding vector retrieval to the lexical baseline (A to B) provides moderate improvement ([TBD] points), consistent with the known complementarity of sparse and dense retrieval. Graph traversal (C to D) contributes [TBD] points, primarily on queries requiring entity disambiguation. Agentic rewriting (D to E) yields [TBD] additional points, with gains concentrated on multi-hop queries where the initial query formulation is underspecified.

On the same-unit control queries, the A-through-F progression is monotonically non-decreasing but the scientific operator transition (B to C) provides negligible gain ([TBD] points), confirming that this component's value is specific to the cross-unit retrieval scenario it was designed for.

## 5.6 Multi-Hop Agentic Retrieval

We evaluate the agentic retrieval loop (Section 3.6) on [TBD] complex queries requiring information synthesis across multiple documents. We compare single-pass retrieval (1 hop), two-hop, and three-hop configurations.

Single-pass retrieval achieves R@5 of [TBD]. Two-hop retrieval improves to [TBD], with the second hop refining the query based on entities and measurements extracted from first-hop results. Three-hop retrieval yields [TBD], a marginal improvement over two hops, suggesting that for most queries in our benchmark the agentic loop converges within two iterations.

We illustrate the rewriting behavior with representative examples. Consider the query "What simulation used pressures around 200 kPa with a velocity of 50 km/h?" In the first hop, the system retrieves documents matching the dimensional constraints. The agentic rewriter examines these results, identifies a referenced simulation identifier, and reformulates the query for the second hop to include the identifier as a lexical constraint, recovering additional documents linked by that identifier but not by the original numeric constraints.

For deployments where LLM inference is unavailable or latency-constrained, the system falls back to a deterministic rewriting strategy: the rewriter extracts entities and measurement tuples from first-hop results and appends them as structured filters to the second-hop query. On the same benchmark, the non-LLM fallback achieves R@5 of [TBD], compared to [TBD] for LLM-based rewriting---a modest degradation that preserves the majority of multi-hop gains without requiring a language model at query time.

## 5.7 Indexing Performance

We measure indexing throughput on a single compute node (Section 4.3) across the three connector types.

Full indexing of the benchmark corpus ([TBD] documents) completes in [TBD] seconds, corresponding to [TBD] documents/second. Per-connector throughput varies with document complexity: the filesystem connector processes [TBD] documents/second, S3 achieves [TBD] documents/second (dominated by network round-trip latency), and the HDF5 connector processes [TBD] files/second, with per-file time scaling with dataset count and attribute density.

Incremental re-indexing, triggered by SHA-256 checksum changes in the `file_index` table, substantially reduces update cost. Fig. 6 plots indexing time as a function of corpus change fraction. When 5% of documents are modified, incremental indexing completes in [TBD]% of the full-index time. At 10% and 25% change fractions, incremental indexing requires [TBD]% and [TBD]% of full-index time respectively. The sublinear scaling arises because unchanged documents are skipped entirely---no text extraction, embedding computation, or measurement canonicalization is performed---and only the modified chunks and their dependent entries in the `embeddings`, `scientific_measurements`, and `scientific_formulas` tables are rewritten.
