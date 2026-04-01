# 3. System Design

To address the three failures identified in Section 1, we design clio-agentic-search around a central principle: science-aware operators must execute as first-class retrieval branches, not post-processing filters. The architecture comprises a staged pipeline with four parallel branches (Sec. 3.1), dimensional conversion operators that perform explicit SI arithmetic (Sec. 3.2), deterministic formula normalization (Sec. 3.3), a hybrid fusion strategy (Sec. 3.4), federated multi-namespace search (Sec. 3.5), and an LLM-driven multi-hop retrieval loop (Sec. 3.6).

## 3.1 Architecture Overview

The system is a staged pipeline with four parallel retrieval branches (Fig. 1). A query enters through the Query Interface (FastAPI REST endpoint or CLI), which parses the raw query string, extracts typed scientific constraints (numeric ranges, unit filters, formula patterns), and forwards a structured query object to the Namespace Registry. The registry resolves which namespaces---each backed by one of six connector types---are eligible for the query based on capability negotiation (Sec. 3.5).

The Retrieval Coordinator fans the query out to four parallel branches: lexical (BM25), vector (dense embedding similarity), graph (entity-relationship traversal), and scientific (dimensional measurement and formula queries). Each branch executes independently and returns scored candidate chunks. The Merge and Rerank stage deduplicates candidates by chunk identifier, applies hard scientific filters, computes a composite score via heuristic fusion, and emits the final top-$K$ results with source citations.

Six connectors mediate access to heterogeneous backends: local filesystem, S3-compatible object stores, Qdrant vector database, Neo4j graph database, HDF5 (via h5py), and NetCDF (via xarray). Each connector declares a capability set specifying which retrieval branches it supports, enabling the coordinator to skip unsupported branches rather than fail. Every pipeline stage emits structured trace events for observability.

## 3.2 Dimensional Conversion Operators

The core insight behind dimensional conversion is that cross-unit matching is an arithmetic problem, not a similarity problem. When a scientist queries for pressures between 190 and 360 kPa, the system must recognize that a document reporting "250000 Pa" falls within that range. This requires multiplying 190 by 1000 to obtain 190000 Pa and 360 by 1000 to obtain 360000 Pa---a computation that no string normalization or learned embedding can guarantee.

We define a conversion table covering five SI domains and 13 units (Table 2). Each entry maps a source unit to a canonical base unit and a multiplicative conversion factor. At index time, a regex-based measurement extractor identifies numeric-unit pairs in document text and scientific file metadata. Each extracted measurement is stored as a 4-tuple: (raw\_value, raw\_unit, canonical\_value, canonical\_unit). For example, "200 kPa" yields (200.0, "kPa", 200000.0, "Pa"). These tuples populate a `scientific_measurements` table in DuckDB, indexed on canonical\_value and canonical\_unit for efficient range scans.

| Domain   | Source units          | Canonical unit | Conversion factors              |
|----------|-----------------------|----------------|---------------------------------|
| Distance | mm, cm, m, km        | m              | $10^{-3}$, $10^{-2}$, 1, $10^{3}$ |
| Mass     | mg, g, kg            | kg             | $10^{-6}$, $10^{-3}$, 1        |
| Time     | s, min, h            | s              | 1, 60, 3600                     |
| Pressure | Pa, kPa, MPa         | Pa             | 1, $10^{3}$, $10^{6}$          |
| Velocity | m/s, km/h            | m/s            | 1, $\frac{1}{3.6}$             |

**Table 2.** SI conversion table. 5 domains, 13 units. Each source unit maps to a canonical base unit via a deterministic multiplicative factor.

At query time, three operators execute against the canonicalized measurement store. The **numeric-range** operator parses range expressions of the form `low:high:unit`, converts both bounds to canonical units, and issues a DuckDB range predicate: a query "190:360:kPa" becomes a scan for canonical\_value between 190000 and 360000 with canonical\_unit = "Pa". The **unit-match** operator filters chunks containing any measurement in a specified unit domain regardless of value. The **value-match** operator retrieves measurements matching a specific canonical value within a configurable tolerance (default $\pm$1\%).

The correctness guarantee distinguishes this approach from all prior work. Consider matching "200 kPa" against "200000 Pa". Arithmetic conversion computes $200 \times 1000 = 200000$ and succeeds by exact equality. String normalization maps "kilopascal" to one string and "pascal" to another---they never match. Learned embeddings \cite{cone2026} encode the two quantities as vectors in a shared space, but the Numeracy Gap study \cite{numeracygap2026} reports 0.54 mean binary retrieval accuracy across 13 embedding models on numerical content, effectively random against a 0.50 baseline. Weller et al. \cite{embedding_limits2025} prove theoretical upper bounds showing that fixed-dimensional embeddings cannot faithfully preserve all distance relationships, a result that fundamentally undercuts learned approaches to exact quantity matching. Our operators are correct by construction: if the SI conversion factor is correct, the match is correct.

## 3.3 Formula Normalization

Scientific formulas appear in inconsistent surface forms across documents. "F = m * a", "F=ma", "F = m \cdot a", and "ma = F" express the same physical law but produce distinct token sequences that defeat both keyword and embedding retrieval. We normalize formulas through a deterministic six-step pipeline.

First, all whitespace is stripped. Second, all alphabetic characters are lowercased. Third, superscript notations are unified: LaTeX-style `^{N}` and Python-style `**N` are both normalized to `^N`. Fourth, the expression is split on the equality sign into left-hand and right-hand sides. Fifth, within each side, multiplicative factors are sorted lexicographically, so that `m*a` and `a*m` produce the same canonical form. Sixth, the two sides are rejoined with a normalized equality separator. The canonical form of all four variants above is `a*m=f`.

Normalized formulas are stored in a `scientific_formulas` table in DuckDB alongside their source chunk identifiers and the raw original text. At query time, the user's formula expression passes through the same six-step pipeline before lookup. Formula matching integrates with the dimensional conversion operators as a parallel branch in the scientific retrieval stage: a query can simultaneously request formulas containing a specific equation and measurements within a numeric range, and both constraints execute as independent DuckDB queries whose results are intersected at the merge stage.

## 3.4 Hybrid Retrieval Pipeline

The retrieval pipeline executes four branches in parallel and merges their results through a deterministic fusion strategy (Fig. 3).

**Lexical branch.** BM25 with parameters $k_1 = 1.2$ and $b = 0.75$ scores term overlap between query and document chunks, retrieving $\text{top\_k} \times 4$ candidates to provide recall headroom for downstream filtering.

**Vector branch.** Chunks are embedded with MiniLM-L6-v2 (384-dimensional vectors, cosine similarity) and indexed in Qdrant. This branch retrieves $\text{top\_k} \times 4$ candidates.

**Graph branch.** For Neo4j-backed namespaces, breadth-first traversal at depth 1 from query entities retrieves related chunks through typed relationships (e.g., cites, measures, uses\_formula). This branch retrieves $\text{top\_k} \times 2$ candidates, as graph neighborhoods are inherently more targeted.

**Scientific branch.** The dimensional conversion and formula normalization operators (Secs. 3.2--3.3) execute against DuckDB, returning $\text{top\_k} \times 4$ candidates from measurement range queries, unit-match filters, value-match lookups, and formula lookups.

**Merge and rerank.** Candidates from all branches are deduplicated by chunk\_id, retaining the maximum per-branch score. A scientific hard filter then applies: if the query specifies a numeric range or unit constraint, candidates not returned by the scientific branch are discarded---dimensional correctness is enforced, not merely preferred. A metadata filter removes candidates violating explicit metadata constraints (e.g., date range, file type). Surviving candidates are scored by a heuristic weighted sum:

$$\text{score} = w_l \cdot s_{\text{lexical}} + w_v \cdot s_{\text{vector}} + w_m \cdot s_{\text{metadata}} + w_s \cdot s_{\text{scientific}}$$

where the weights $w_l, w_v, w_m, w_s$ are configurable per deployment. The top-$K$ candidates by composite score constitute the final result set. Each result carries provenance: the source namespace, chunk identifier, file path, and the set of branches that contributed to its retrieval.

## 3.5 Federated Multi-Namespace Search

Scientific data at HPC facilities spans heterogeneous backends. A single experimental campaign may deposit raw output on a parallel filesystem, preprocessed arrays in HDF5, derived features in a vector database, and annotations in a knowledge graph. Rather than require data migration, we bring the retrieval operators to the data.

Every storage backend is wrapped by a connector that implements the `NamespaceConnector` protocol and declares a `RetrievalCapabilities` object specifying which retrieval branches it supports. The capability matrix is as follows:

- **Filesystem**: lexical, vector, scientific
- **S3**: lexical, vector, scientific
- **Qdrant**: vector
- **Neo4j**: graph
- **HDF5**: metadata, scientific
- **NetCDF**: metadata, scientific

The Retrieval Coordinator consults this matrix before dispatching. When a query spans multiple namespaces, the coordinator executes the single-namespace pipeline independently for each namespace, activating only the branches that the namespace's connector supports. Results from all namespaces are collected into a global candidate pool, sorted by composite score, deduplicated by chunk\_id (since the same document may appear in multiple backends), and truncated to the final top-$K$.

Incremental indexing avoids redundant reprocessing. Each connector tracks indexed documents by SHA-256 content hash and modification time. On re-index, only documents whose hash or mtime has changed are reprocessed, enabling efficient index maintenance over evolving corpora without full rebuild.

## 3.6 Multi-hop Agentic Retrieval

Single-pass retrieval often fails for complex scientific questions that require iterative refinement. The multi-hop agentic retrieval loop automates this process.

The loop proceeds as follows. The initial query executes the full hybrid pipeline described in Section 3.4 across all federated namespaces. The retrieved results, along with the original query and the retrieval trace, are passed to an LLM (Anthropic API) that evaluates result quality and selects one of four strategies: **expand** (broaden the query to increase recall), **narrow** (add constraints to increase precision), **pivot** (reformulate the query to pursue a different aspect), or **done** (terminate the loop). If the strategy is not done, the LLM generates a rewritten query, which re-enters the pipeline for another full retrieval pass. Results from all hops are merged, deduplicated, and re-ranked. The loop terminates when the LLM selects done, when results stabilize across consecutive hops (no new unique chunk\_ids appear), or when a configurable maximum hop count is reached.

Each hop constitutes a full pipeline execution: all four retrieval branches, all eligible namespaces, merge and rerank. This ensures that every rewritten query benefits from the same dimensional conversion, formula normalization, and federated search capabilities as the original. The LLM's role is strictly query rewriting and termination decision---it does not score or filter results, preserving the deterministic correctness guarantees of the science-aware operators. When the Anthropic API is unavailable, the system falls back to an offline strategy that expands queries with SI unit variants (e.g., appending "Pa" and "MPa" to a query containing "kPa"), providing degraded but functional multi-pass retrieval without LLM dependency.
