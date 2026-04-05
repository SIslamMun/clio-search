# CLIO Search Evaluation Plan

This document describes the evaluation plan for **CLIO Search**, an
agentic harness for scientific data discovery. The plan is organized as
fourteen experiments that together substantiate the paper's four core
claims:

1. **Agentic efficiency** — CLIO reduces tokens, tool calls, and wall time
   when an LLM agent queries a federated scientific catalog through MCP.
2. **Science-aware correctness** — dimensional-analysis unit conversion
   gives guaranteed correctness across all SI prefixes where learned
   retrievers are probabilistic.
3. **Scalable retrieval** — CLIO's operators scale sub-linearly on a
   single node and near-linearly across distributed nodes for both
   strong and weak scaling regimes.
4. **Metadata-adaptive discovery** — CLIO profiles each corpus before
   querying and selects a different retrieval strategy per dataset,
   including sampling-based recovery when structured metadata is absent.

Each experiment specifies its purpose, method, dataset, metrics, and the
claim it supports. Results are saved as structured JSON under a single
output directory; plots are generated deterministically from those JSONs.

---

## Experiment index

| # | Experiment | Claim supported |
|---|---|---|
| E1 | NDP-MCP vs CLIO+NDP-MCP (agentic end-to-end) | Agentic efficiency |
| E2 | IOWarp CTE BlobQuery scaling 1K → 1M blobs | Scalable retrieval |
| E3 | SI unit cross-prefix correctness | Science-aware correctness |
| E4 | NumConQ benchmark (6,500 numeric-constraint queries) | Science-aware correctness |
| E5 | 100-namespace federation with per-dataset strategy | Metadata-adaptive discovery |
| E6 | Data quality filtering on real weather-station data | Metadata-adaptive discovery |
| E7 | Single-node scaling curves 1K → 100K documents | Scalable retrieval |
| E8 | Cross-corpus diversity across 4 real scientific corpora | Metadata-adaptive discovery |
| E9 | Distributed strong scaling on 2.5M arXiv papers | Scalable retrieval |
| E10 | Distributed weak scaling with data proportional to workers | Scalable retrieval |
| E11 | Distributed indexing throughput on 2.5M papers | Scalable retrieval |
| E12 | Cross-unit precision at 2.5M scale distributed | Science-aware correctness |
| E13 | NumConQ distributed over a 2.5M corpus | Scalable retrieval |
| E14 | 100-namespace federation distributed (10M docs) | Metadata-adaptive discovery |

---

## E1 — NDP-MCP vs CLIO+NDP-MCP (agentic end-to-end)

### Purpose
Quantify the value that CLIO's science-aware harness adds to an LLM agent
querying a real federated scientific catalog through the Model Context
Protocol. The experiment isolates CLIO's contribution by holding the LLM,
the system prompt, the user prompt, and the NDP catalog fixed across the
two modes; the only variable is whether CLIO helper tools are available
alongside the raw NDP-MCP tools.

### Method
A Claude agent is given ten representative scientific queries spanning
cross-unit retrieval (temperature, pressure, wind speed, solar radiation,
precipitation) and semantic retrieval (glacier, humidity, ocean, wildfire,
soil moisture). For each query the agent is run twice with identical
system and user prompts:

- **Mode A**: the agent has access to the real NDP-MCP server tools
  (`search_datasets`, `list_organizations`, `get_dataset_details`).
- **Mode B**: the agent has the same NDP-MCP tools *plus* two CLIO helper
  tools exposed through a local MCP SDK server: a dimensional-unit
  canonicalizer and a science-aware scientific-search pipeline that runs
  NDP discovery, parses CSV resources, infers units from column headers,
  canonicalizes to SI base units, and filters by threshold.

The agent chooses its own tool sequence in both modes. Each run produces
a full trace of assistant turns, tool calls, tool results, and usage
information reported by the underlying API.

### Dataset
The live National Data Platform catalog. No pre-indexing; the agent
discovers data through NDP-MCP calls each time.

### Metrics
- Total tool calls issued by the agent
- Number of assistant turns
- Input tokens, output tokens, cache creation tokens, cache read tokens
- Total tokens (sum of the four above)
- Wall clock time
- Correctness: fraction of ground-truth keywords present in the final
  answer (simple, reviewer-auditable proxy)

Metrics are reported per query and aggregated across the ten queries.

### Why this experiment
This is the most direct end-to-end evidence of the agentic-efficiency
claim. Because both modes use the same LLM and the same catalog, any
reduction in tokens or tool calls observed in Mode B must be attributable
to CLIO's helper tools. The correctness proxy confirms that efficiency
gains do not come at the cost of answer quality.

---

## E2 — IOWarp CTE BlobQuery scaling 1K → 1M blobs

### Purpose
Characterize how CLIO's integration with IOWarp's Context Transfer
Engine scales as the blob count grows by three orders of magnitude. CLIO
depends on CTE's BlobQuery mechanism to resolve document-to-shard
assignments; this experiment establishes the asymptotic behavior of three
distinct query patterns in CTE.

### Method
A Chimaera runtime is started in a container configured with sufficient
shared memory. At each scale N ∈ {1K, 5K, 10K, 50K, 100K, 500K, 1M} the
test:

1. Creates four tags (temperature, pressure, wind, humidity)
2. Puts N synthetic scientific blobs spread evenly across the tags
3. Measures the latency of three BlobQuery patterns, each repeated five
   times with median reported:
   - Tag-filtered query: matches a single tag by exact name, returns all
     blobs under it
   - Specific-regex query: matches all tags but restricts the blob-name
     regex to a single specific blob
   - Full-scan query: matches all tags and all blobs

### Dataset
Synthetic scientific blobs generated deterministically from a seeded
random generator. Each blob contains a single measurement (temperature,
pressure, wind speed, or humidity) in a standard unit.

### Metrics
- PutBlob throughput (blobs per second)
- Median BlobQuery latency for each of the three query patterns, in
  milliseconds
- Result cardinality for each query pattern (sanity check)

### Why this experiment
BlobQuery is the gateway through which CLIO retrieves actual blob
references at scale. Reviewers will ask whether this layer itself scales.
The three query patterns reveal the algorithmic complexity of the
underlying index: tag-filtered queries should scale sub-linearly because
CTE maintains per-tag indices, whereas full-scan queries must traverse
the entire blob space and therefore scale linearly. Establishing this
curve at the 1M-blob ceiling gives a concrete anchor for the paper's
scaling argument.

---

## E3 — SI unit cross-prefix correctness

### Purpose
Demonstrate that CLIO's dimensional-analysis unit registry delivers
deterministic correctness across all SI prefixes for a given physical
quantity, where string-based and embedding-based retrievers fail.

### Method
For each of six physical quantities (pressure, temperature, velocity,
length, mass, energy), construct seven documents that each report the
same underlying canonical value but expressed in a different unit or
prefix variant (e.g. 100 kPa, 100000 Pa, 0.1 MPa, 1 bar, 0.987 atm).
The document text is realistic scientific prose with the measurement
injected via a rotating set of templates.

Then, for each of the seven unit variants used as the query, ask four
retrievers to return the top five documents:

1. BM25 lexical retrieval
2. Dense embedding retrieval (cosine similarity)
3. String normalization (a NumbersMatter-style approach that expands
   unit synonyms in the query string but does not perform arithmetic)
4. CLIO with the numeric-range operator using the query's unit

For each query variant, compute the recall at five — what fraction of
the seven ground-truth documents appear in the top five results.

### Dataset
Generated corpus of 42 documents (6 quantities × 7 variants) using
deterministic seeding and rotating prose templates.

### Metrics
- Recall@5 per (query-unit × document-unit) pair — visualized as a
  heatmap
- Macro-average Recall@5 per method — the headline number
- Precision@5 and MRR as secondary metrics

### Why this experiment
The cross-prefix problem is the simplest case where learned or
string-based retrievers are guaranteed to fail and where arithmetic
canonicalization is guaranteed to succeed. The experiment directly tests
whether dimensional analysis delivers its theoretical guarantee in
practice. It is also the cleanest adversarial demonstration in the paper:
BM25 and dense retrievers are expected to produce macro-average R@5 close
to zero on cross-unit queries, while CLIO should produce R@5 close to
one.

---

## E4 — NumConQ benchmark

### Purpose
Evaluate CLIO on the standard numeric-constraint retrieval benchmark
published alongside NC-Retriever. This establishes a direct comparison
with prior work on the only widely-used benchmark for numeric retrieval.

### Method
Download the NumConQ corpus and query set (6,500 queries across five
domains: finance, medicine, physics, sports, stocks). Index the corpus
into CLIO and run all queries under three configurations:

1. BM25 lexical retrieval over the same corpus
2. Dense embedding retrieval
3. CLIO with the numeric-range operator applied when a numeric constraint
   is present in the query, and lexical fallback otherwise

Metrics are computed against the ground-truth relevance sets provided
with NumConQ.

### Dataset
NumConQ: 6,500 queries × 5 domains with human-annotated relevance.

### Metrics
- Recall@5, Recall@10
- Precision@10
- Mean reciprocal rank (MRR)
- Per-domain breakdown of all metrics

### Why this experiment
NumConQ is the standard benchmark that SIGIR and EMNLP reviewers will
expect any new numeric-retrieval system to report against. Published
baselines hover around 16% Recall@10 on NumConQ; NC-Retriever reports
~82% Recall@10 with a learned model. The purpose of running CLIO on
this benchmark is not to beat NC-Retriever on its home turf — NC-Retriever
is trained end-to-end for this benchmark — but to demonstrate that CLIO's
deterministic arithmetic operators are competitive with learned approaches
while offering correctness guarantees that learned models cannot provide.

---

## E5 — 100-namespace federation with per-dataset strategy

### Purpose
Test CLIO's claim that "no single query strategy works across 100+
heterogeneous datasets" and measure how much retrieval work is avoided by
profiling each dataset before querying.

### Method
Generate 100 synthetic namespaces with labeled characteristics: 30
rich-metadata namespaces with structured scientific measurements, 30
sparse namespaces where measurements are hidden in prose (recoverable
through sampling), 30 pure-text namespaces with no extractable
measurements, 5 formula-heavy namespaces with equations, and 5 empty
namespaces. Each namespace is tagged with a domain (temperature,
pressure, humidity, wind, radiation) so the test can check whether the
planner correctly routes queries to the relevant subset.

Two cross-unit queries are then issued against the full set of 100
namespaces:

1. "Find temperature measurements above 25 degrees Celsius"
2. "Find pressure measurements around 101 kPa"

For each namespace, the planner decides which retrieval branches
(lexical, vector, scientific) to activate based on the corpus profile.
The experiment is run twice: once without sampling-based schema recovery,
and once with sampling enabled.

### Dataset
100 synthetic namespaces generated deterministically, stored in a single
DuckDB instance under different namespace tags.

### Metrics
- Total branches activated vs total branches possible (branch-saving
  rate, the headline number)
- Scientific branch activation count per namespace type
- Routing correctness: true positives, false positives, false negatives
  against the ground-truth labels
- Per-namespace profile time (demonstrates that per-dataset planning is
  cheap enough to be practical)
- Number of namespaces where sampling recovered structure that the
  primary index missed

### Why this experiment
The paper's motivation explicitly invokes 100+ heterogeneous datasets.
This experiment operationalizes that motivation with a labeled corpus
where ground truth is known, so the branch-selection and routing
decisions can be scored directly. The branch-saving rate is a concrete
efficiency metric; the routing correctness is a concrete quality metric.
Together they quantify the value of per-dataset strategy adaptation.

---

## E6 — Data quality filtering on real weather-station data

### Purpose
Demonstrate CLIO's quality filter on real scientific data with real
quality-control flags. This shows that the filter correctly reads
production QC codes and drops unreliable rows before retrieval sees them.

### Method
Download five real weather-station CSVs from the California Irrigation
Management Information System. Each CSV contains approximately 130,000
hourly records with a quality-control column after every measurement,
populated with CIMIS codes:

- blank → good
- Y → questionable
- R → rejected/bad
- M → missing

CLIO's CSV parser reads each file, auto-detects the measurement columns
from header unit suffixes (e.g. "Air Temp (C)"), parses the adjacent QC
column, and canonicalizes each measurement to SI base units with the
parsed quality flag attached.

A single cross-unit query is then issued against the parsed rows:
"find temperature measurements above 30 degrees Celsius" (which
canonicalizes to 303.15 K). The query is run twice:

1. With the default quality filter (accept good and estimated rows only)
2. Without any quality filter (accept all flags including bad and
   missing)

### Dataset
Five CIMIS weather stations (Westlands, Panoche, Arvin-Edison, Fair Oaks,
Belridge) representing approximately 650,000 real measurement rows with
production QC flags.

### Metrics
- Per-column quality flag distribution (good, questionable, bad, missing)
- Rows matching the cross-unit query with the quality filter enabled
- Rows matching the same query with the quality filter disabled
- Rows dropped by the quality filter (the difference)

### Why this experiment
QC flag handling is a standard expectation in scientific data systems
but is rarely supported by retrieval pipelines. This experiment proves
CLIO's quality layer works end-to-end on real production data with real
production QC codes, and shows concretely how many rows the filter
eliminates from a downstream query.

---

## E7 — Single-node scaling curves 1K → 100K documents

### Purpose
Establish the algorithmic scaling behavior of CLIO's profile and query
operations as the corpus size grows, independent of distribution. This
is the reference curve that the distributed experiments extend.

### Method
At each scale N ∈ {1K, 2K, 5K, 10K, 25K, 50K, 100K}, generate N
deterministic synthetic scientific documents (each containing a
temperature, pressure, and wind-speed measurement in prose), index them
into a single DuckDB namespace, and measure:

- Indexing wall time and throughput (documents per second)
- Profile time (corpus-profile aggregation) over ten repeated runs;
  median and standard deviation reported
- Scientific range query latency over ten repeated runs; median and
  standard deviation reported
- Resulting database file size

### Dataset
Deterministically generated synthetic documents with a seeded random
generator for reproducibility.

### Metrics
- Index throughput (documents per second)
- Profile time median, minimum, maximum, and standard deviation (ms)
- Query time median, minimum, maximum, and standard deviation (ms)
- Database size in megabytes
- Scaling ratios (last/first) for each metric, with O(n) reference

### Why this experiment
Reviewers will want to see that CLIO's core operations are sub-linear in
corpus size before they believe distributed scaling claims. This
experiment establishes the single-node baseline curves; repeated runs
provide error bars that make the scaling argument defensible under
statistical scrutiny.

---

## E8 — Cross-corpus diversity across 4 real scientific corpora

### Purpose
Demonstrate that CLIO correctly distinguishes rich-metadata corpora from
sparse ones and picks different strategies accordingly, across real
scientific datasets from different domains and with different metadata
conventions.

### Method
Index four independent corpora in isolation:

1. NOAA GHCN-Daily: 1,728 real weather-station documents with metric
   units
2. DOE Data Explorer: 500 DOE scientific dataset descriptions
3. Controlled v2: 210 synthetic multi-domain scientific documents with
   deliberate unit heterogeneity
4. arXiv subset: 500 real scientific abstracts

For each corpus, run the corpus-profile builder with sampling disabled
and then with sampling enabled, and record the resulting profile
structure.

### Dataset
Four corpora totaling ~2,938 real scientific documents across four
distinct data sources.

### Metrics
- Per-corpus document count, chunk count, measurement count, formula
  count
- Metadata density (narrow scientific metadata fraction)
- Metadata schema richness score (broader schema-based score using
  detected canonical concepts)
- Detected canonical concepts per corpus (temperature, pressure, station
  identifier, etc.)
- Strategy classification (rich / default / sparse)
- Sampling recovery count — concepts or measurements the sampling
  fallback found that the primary index missed

### Why this experiment
The motivation for CLIO is that real scientific corpora vary dramatically
in metadata quality. This experiment demonstrates that variation on real
data and shows that CLIO's strategy selection is not just theoretical —
the same profile-building pipeline produces different classifications for
corpora with genuinely different characteristics.

---

## E9 — Distributed strong scaling on 2.5M arXiv papers

### Purpose
Demonstrate that CLIO's distributed coordinator/worker architecture
scales near-linearly when the corpus size is held fixed and the worker
count is increased.

### Method
The full arXiv metadata dump (~2.5M papers with titles, abstracts,
categories) is sharded deterministically across workers using hash-based
partitioning. Each worker builds its own DuckDB index over its shard on
local high-speed storage. The coordinator is then started on a separate
node and connected to the workers via TCP RPC.

A fixed set of five queries (three cross-unit, two semantic) is issued
against the cluster at three worker counts: 1, 2, and 4 workers. Each
query is repeated ten times; latency percentiles are computed.

### Dataset
2.5 million arXiv paper abstracts with titles and categories, obtained
from the public Kaggle arXiv metadata dump.

### Metrics
- Query latency at the 50th, 95th, and 99th percentiles, per worker
  count
- Speedup ratio compared to the 1-worker baseline
- Comparison to ideal strong-scaling reference (latency_1 / N)
- Coordinator aggregation time vs worker query time breakdown

### Why this experiment
Strong scaling is the primary scaling metric for retrieval systems —
does adding more workers reduce query latency for the same corpus? A
near-linear curve is the positive result the paper needs. A clean
deviation from the ideal curve at high worker counts is also a valid
finding and should be analyzed honestly if observed.

---

## E10 — Distributed weak scaling with data proportional to workers

### Purpose
Demonstrate that CLIO maintains constant per-worker work as both the
corpus and the worker count grow together. This is the scaling regime
that matters for production deployments where data volume grows over
time.

### Method
Three configurations are run, with data scaling proportionally to
workers:

- 1 worker holding 625K arXiv papers
- 2 workers holding 1.25M arXiv papers total (625K each)
- 4 workers holding 2.5M arXiv papers total (625K each)

The same five-query set from E9 is issued against each configuration
with ten repetitions per query. The experiment measures whether per-query
latency stays approximately constant as data and workers scale together.

### Dataset
Subsets of the arXiv metadata dump matched to worker count.

### Metrics
- Query latency at the 50th, 95th, and 99th percentiles for each
  configuration
- Weak-scaling efficiency ratio (latency_N / latency_1); ideal is 1.0
- Coordinator overhead as a function of worker count

### Why this experiment
Weak scaling tests a different property from strong scaling: whether the
system can handle growing workloads without degrading per-query
performance. A flat latency curve across configurations is the positive
result; an upward slope indicates coordinator overhead or communication
bottlenecks.

---

## E11 — Distributed indexing throughput on 2.5M papers

### Purpose
Measure how quickly CLIO can index a large real scientific corpus when
workers process their shards in parallel, and quantify any load imbalance
across workers.

### Method
After the arXiv dataset is sharded deterministically, each worker runs
the CLIO indexing pipeline on its assigned shard in parallel. Wall time
is measured from the moment all workers start until all workers finish.
The experiment records per-worker indexing time so that load imbalance
can be computed.

### Dataset
2.5 million arXiv abstracts sharded across 4 workers (approximately 625K
per worker).

### Metrics
- Per-worker indexing wall time
- Aggregate throughput (total documents per second across all workers)
- Load imbalance (difference between slowest and fastest worker)
- Per-worker database size after indexing

### Why this experiment
Indexing throughput is the foundation for every downstream claim about
scaling query performance. If parallel indexing is efficient, the
distributed architecture is practical; if workers are badly imbalanced,
the coordinator's tail latency will be bounded by the slowest worker.
This experiment makes that tradeoff concrete.

---

## E12 — Cross-unit precision at 2.5M scale distributed

### Purpose
Verify that E3's cross-unit correctness claim holds at 500× larger
corpus scale in a distributed setting.

### Method
Using the distributed CLIO cluster with 4 workers and the 2.5M arXiv
corpus, issue a set of cross-unit probe queries (pressure in various
prefixes, temperature in Celsius / Fahrenheit / Kelvin) and measure how
many relevant results each returns. Because the corpus is real, there is
no ground-truth set of documents, so results are quality-checked by
inspecting the canonical values returned and confirming they fall within
the query range.

### Dataset
2.5M distributed arXiv corpus.

### Metrics
- Number of results returned per query
- Number of shards with at least one match
- Query latency
- Manual validation that canonical values fall within the query range

### Why this experiment
E3 establishes cross-unit correctness on a controlled 42-document
corpus. E12 confirms that the same correctness property holds when the
corpus is three orders of magnitude larger and distributed. This closes
the gap between the correctness argument and the scale argument.

---

## E13 — NumConQ distributed over a 2.5M corpus

### Purpose
Run the NumConQ benchmark against a distributed CLIO cluster with the
arXiv corpus as additional context, measuring how CLIO performs when
NumConQ queries are evaluated over a much larger federated search space
than NumConQ was designed for.

### Method
The NumConQ query set (6,500 queries) is issued against the distributed
cluster that holds both the NumConQ corpus and the arXiv corpus. CLIO's
branch-selection mechanism decides per namespace whether to activate the
scientific branch. Metrics are computed against NumConQ's ground-truth
relevance sets.

### Dataset
NumConQ corpus (original, ~5,000 documents) plus arXiv 2.5M, both
indexed into the distributed cluster.

### Metrics
- Recall@10, Precision@10, MRR (same as E4)
- Effect of the larger federated search space on metrics (does the added
  arXiv distractor corpus hurt precision?)
- Query latency distribution

### Why this experiment
NumConQ queries were designed for small, domain-specific corpora. By
running them over a 500× larger corpus with distractors, the experiment
tests whether CLIO's per-dataset planning correctly focuses the scientific
branch on the relevant namespace without being distracted by the arXiv
corpus. This is a stress test of the branch-selection mechanism at
realistic scale.

---

## E14 — 100-namespace federation distributed (10M documents)

### Purpose
Scale E5 from 100 synthetic namespaces with small corpora to 100
namespaces with 100K documents each (10M documents total), distributed
across 4 workers. Demonstrates that per-dataset strategy adaptation
remains effective at production scale.

### Method
The 100 synthetic namespaces from E5 are regenerated at 100× the size
(100K documents per namespace instead of ~5). The corpus is distributed
across 4 workers with hash sharding. The same branch-saving and routing
correctness metrics from E5 are measured at the larger scale.

### Dataset
10 million synthetic scientific documents across 100 labeled namespaces,
distributed across 4 workers.

### Metrics
- Branch-saving rate (same as E5 but at 1000× data)
- Routing correctness (true positives, false positives, false negatives)
- Per-namespace profile time (should remain sub-10ms)
- Sampling recovery count

### Why this experiment
E5 demonstrates the per-dataset planning mechanism works on small
synthetic data. E14 confirms the mechanism scales: profile time must
remain cheap (well under a second for all 100 namespaces) for the
strategy-selection argument to hold in production. If profile time
grows linearly with namespace size, the whole approach breaks.

---

## Summary: how the experiments map to claims

| Claim | Primary experiments | Supporting |
|---|---|---|
| Agentic efficiency | E1 | — |
| Science-aware correctness | E3, E4 | E12 |
| Scalable retrieval | E2, E7, E9, E10, E11 | E13, E14 |
| Metadata-adaptive discovery | E5, E6, E8 | E14 |

Each claim is supported by at least one controlled experiment on
synthetic data (for correctness) and at least one experiment on real
scientific data (for external validity). The distributed experiments
(E9–E14) establish the scaling argument that a single-node evaluation
cannot.

All experiments are fully scripted, deterministic where possible, and
produce machine-readable JSON output so results can be independently
reproduced and plotted.
