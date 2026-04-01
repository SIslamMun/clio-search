# Full Paper Outline — SC2026 IEEE CS

**Title**: Science-Aware Hybrid Retrieval with Dimensional Conversion for HPC Data Discovery
**Format**: IEEE double-column, 10 pages (~7,500 words + references)

---

## Abstract (237 words) — DRAFTED
See: `.planning/drafts/abstract.md`

---

## §1 Introduction (~1,125 words)

### §1.1 AI Agents Entering Scientific Computing (~250 words)

**Argument**: Agents are doing real science — but they can't find data.

Key points:
- Coscientist (Nature 2023): autonomous wet-lab experiments
- AI Scientist v2 (2025): end-to-end paper writing
- ALS beamline (arXiv:2509.17255): 100× prep time reduction
- ORNL autonomous agents (SC'25): cross-facility experiment orchestration
- MCP for HPC (arXiv:2508.18489): standard protocol wrapping Globus, compute, storage
- ScienceAgentBench (NeurIPS 2024): best agent = 32%, failures at **data handling**, not reasoning
- MLAgentBench (ICML 2024): agents break at data preparation
- Agentic AI survey (ICLR 2025): data management explicitly identified as underexplored

**Transition**: "Yet these agents share a fundamental limitation: they treat scientific data as plain text."

### §1.2 Three Compounding Failures (~400 words)

**Failure 1: Dimensional Blindness** (~150 words)
- "200 kPa" and "200000 Pa" = same pressure, never matched
- String normalization (Numbers Matter! 2024): "kilopascal" ≠ "pascal" → miss
- Learned embeddings (CONE 2026): probabilistic, 0.54 accuracy (Numeracy Gap, EACL 2026)
- Neither does arithmetic: 200 × 1000 = 200000

**Failure 2: Formula Opacity** (~100 words)
- "F=ma" vs "F = m · a" vs "ma=F" — no system normalizes
- Math-aware search (Approach0, SSEmb CIKM 2025): LaTeX structure, not physical measurements
- No system combines formula matching with quantity-aware retrieval

**Failure 3: Storage Fragmentation** (~100 words)
- HPC data: parallel filesystems, S3, vector DBs, graph DBs, HDF5/NetCDF
- No unified search across backends with science-aware operators
- Federated RAG (RAGRoute 2025): federates generic text search, not science-aware
- PROV-IO+ (IEEE TPDS 2024): provenance but no search
- HDF Clinic 2025: KG over HDF5 envisioned, not implemented

**Compounding** (~50 words)
- Motivating example: "Find experiments where pressure was between 190–360 kPa across all storage"
- Needs dimensional conversion + federated search + formula context
- No existing system addresses even two simultaneously

### §1.3 Contributions (~350 words)

Numbered list:
1. **Dimensional-conversion measurement retrieval** — canonicalize to base SI via multiplication (kPa × 10³ = Pa). Guaranteed correct by construction. Numeric range queries, unit matching, tolerance-based comparison.
2. **Unified formula normalization** — normalize across whitespace, superscripts, factor ordering. Integrated with dimensional operators as parallel branches in hybrid pipeline.
3. **Federated multi-namespace search** — connector architecture for FS, S3, Qdrant, Neo4j, HDF5/NetCDF. Per-backend capability negotiation. Incremental indexing with SHA256 change detection.
4. **HDF5/NetCDF metadata indexing** — extract and index attributes, dimensions, variables from scientific file formats into unified search.
5. **Multi-hop agentic retrieval** — LLM-driven query rewriting with iterative search refinement over federated scientific data.

### §1.4 Organization (~125 words)
Standard: §2 background, §3 design, §4 implementation, §5 evaluation, §6 conclusion.

**Citations needed**: [Coscientist], [AI Scientist v2], [ALS], [ORNL SC'25], [MCP for HPC], [ScienceAgentBench], [MLAgentBench], [Agentic AI Survey], [Numbers Matter!], [CONE], [Numeracy Gap], [Approach0], [SSEmb], [RAGRoute], [PROV-IO+], [HDF Clinic]

---

## §2 Background and Related Work (~1,125 words)

### §2.1 Hybrid Retrieval Foundations (~175 words)

- BM25 (Robertson & Zaragoza) + dense retrieval (Karpukhin et al.)
- Fusion: convex combination > RRF (Bruch et al., ACM TOIS 2023)
- Learned sparse: SPLADE, Mistral-SPLADE (Doshi et al., 2024)
- BGE-M3 (Chen et al., ACL 2024): multi-granularity hybrid
- A-RAG (Du et al., 2026): hierarchical retrieval interfaces for LLMs
- **Position**: We build ON hybrid retrieval, adding science-aware branches

**Citations**: [BM25], [Bruch], [SPLADE], [BGE-M3], [A-RAG]

### §2.2 The Numerical Reasoning Crisis (~175 words)

- Numeracy Gap (Deng et al., EACL 2026): 0.54 accuracy on numerical content
- NC-Retriever/NumConQ (2025): 16.3% dense retrieval on numeric constraints
- Numbers encoded per-digit, not magnitude (arXiv:2510.08009)
- NumericBench (ACL 2025), Number Cookbook (ICLR 2025)
- GSM-Symbolic (Apple, ICLR 2025): 65% accuracy drop on numeric changes
- **Position**: This isn't just a unit problem — numbers themselves are broken in every retrieval/LLM system

**Citations**: [Numeracy Gap], [NC-Retriever], [Numbers Not Continuous], [NumericBench], [Number Cookbook], [GSM-Symbolic]

### §2.3 Quantity-Aware Retrieval (~250 words)

**String normalization approach** (~100 words):
- QFinder (Almasian & Gertz, SIGIR 2022): Elasticsearch quantity ranking
- CQE (Almasian et al., EMNLP 2023): 531-unit extraction, dependency parsing
- Numbers Matter! (Almasian et al., EMNLP 2024): full retrieval with FinQuant/MedQuant benchmarks
- Limitation: string normalization cannot cross SI prefixes

**Learned embedding approach** (~75 words):
- CONE (Shrestha et al., March 2026): transformer embedding for numbers+units, 25% Recall@10 gain
- NC-Retriever (2025): contrastive learning for numeric constraints
- Limitation: probabilistic, cannot guarantee equivalence

**Math-aware search** (~50 words):
- Approach0: equation structure matching
- SSEmb (CIKM 2025): semantic similarity for equations
- MIRB (2025): math-aware retrieval benchmarks
- Limitation: formula structure only, no integration with measurement retrieval

**Gap statement** (~25 words):
None performs arithmetic conversion. None combines quantity + formula matching. None operates across federated HPC backends.

**Citations**: [QFinder], [CQE], [Numbers Matter!], [CONE], [NC-Retriever], [SciHarvester], [Approach0], [SSEmb], [MIRB]

### §2.4 Scientific Data Discovery in HPC (~275 words)

**Paper search (not data search)** (~75 words):
- OpenScholar (Nature 2026): 45M papers, GPT-4o-level quality
- HiPerRAG (PASC 2025): 3.6M articles across Polaris/Sunspot/Frontier
- Neither searches actual scientific data — HDF5, simulation outputs, experimental logs

**HPC data management** (~75 words):
- Datum (INL, 2024): scientific data catalog, no content search
- BRINDEXER (CCGrid 2020): POSIX metadata indexing on Lustre, 69% improvement
- PROV-IO+ (IEEE TPDS, 2024): HDF5 I/O provenance, <3.5% overhead, no search
- PROV-AGENT (eScience 2025): W3C PROV for AI agents
- HDF Clinic (2025): KG over HDF5, no implementation

**Agentic retrieval** (~75 words):
- Context-1 (Chroma, 2026): 20B agentic search, 88% vs 58% single-pass
- MA-RAG (Nguyen et al., 2025): 4 agent types for multi-hop QA
- HiPRAG (Wu et al., 2025): hierarchical process rewards
- Academy (Pauloski et al., 2025): federated agent middleware
- **All improve orchestration, not understanding** — blind to units, formulas, dimensions

**Gap table** (~50 words):
- Table 1: Comparison across 7 systems × 10 capabilities
- First five rows (our contributions) empty for every prior system
- Key insight: everyone searches papers, nobody searches data

**Citations**: [OpenScholar], [HiPerRAG], [Datum], [BRINDEXER], [PROV-IO+], [PROV-AGENT], [HDF Clinic], [Context-1], [MA-RAG], [HiPRAG], [Academy], [Souza SC'25], [ORNL SC'25]

**Figure**: Table 1 — Capability comparison with prior systems

---

## §3 System Design (~1,650 words)

### §3.1 Architecture Overview (~250 words)

- System-level diagram showing full pipeline
- Query Interface (FastAPI + CLI) → Namespace Registry → Retrieval Coordinator → Parallel Branches → Merge → Citations
- 6 connector types: filesystem, S3, Qdrant, Neo4j, HDF5, NetCDF/xarray
- Capability negotiation: coordinator discovers what each backend supports
- Trace events: every query produces 8-10 stage traces

**Figure**: Fig 1 — System Architecture (block diagram)

### §3.2 Dimensional Conversion Operators (~400 words)

**The core novelty.** This subsection gets the most space.

**SI Conversion Table** (~75 words):
- Table 2: 5 domains × raw units → base SI → conversion factor
- Distance: mm, cm, m, km → m
- Mass: mg, g, kg → kg
- Time: s, min, h → s
- Pressure: Pa, kPa, MPa → Pa
- Velocity: m/s, km/h → m/s

**Measurement Extraction** (~100 words):
- Regex pattern: `[+-]?\d+(\.\d+)?\s*(unit_list)`
- Per match: (raw_value, raw_unit, canonical_value, canonical_unit)
- Stored in DuckDB `scientific_measurements` per chunk
- Index-time canonicalization pipeline (Fig 2)

**Query-Time Operators** (~125 words):
- Numeric range: `--numeric-range "190:360:kPa"` → canonicalize both sides
- Unit match: `--unit-match "5:kg"` → find exact value with tolerance
- Value match: exact canonical comparison
- Walk through motivating example: 3 docs, 3 units, 3 backends → all found

**Correctness Guarantee** (~100 words):
- Arithmetic: 200 × 1000 = 200000 (deterministic)
- vs. string normalization: "kilopascal" ≠ "pascal" (different strings)
- vs. learned embeddings: depends on training data, 0.54 accuracy
- Our approach: guaranteed correct by construction — if the SI conversion factor is correct, the match is correct

**Figure**: Fig 2 — Dimensional Conversion Pipeline
**Table**: Table 2 — SI Conversion Table

### §3.3 Formula Normalization (~175 words)

Normalization pipeline:
1. Strip whitespace → "E=m*c^{2}" → "E=m*c^{2}"
2. Lowercase → "e=m*c^{2}"
3. Normalize superscripts → "e=m*c^2" (^{2}→^2, **2→^2)
4. Split on = → ["e", "m*c^2"]
5. Sort factors per side → ["e", "c^2*m"]
6. Rejoin → "e=c^2*m"

Storage: DuckDB `scientific_formulas` (raw + normalized per chunk)
Scoring: formula match = +1.4 score

Integration: formulas and measurements as parallel branches in the same pipeline — no prior system does this.

### §3.4 Hybrid Retrieval Pipeline (~275 words)

**Parallel Branch Execution**:
1. Lexical (BM25, k1=1.2, b=0.75): tokenize → posting lists → Okapi BM25 → top_k×4
2. Vector (all-MiniLM-L6-v2, 384-dim): embed → cosine similarity → top_k×4
3. Graph (BFS, depth=1): neighbor expansion → top_k×2
4. Scientific (if operators active): DuckDB query → formula/measurement match → top_k×4

**Merge and Rerank**:
- Deduplicate by chunk_id
- Scientific filtering (hard filter: drop chunks failing active operators)
- Metadata filtering (key=value constraints)
- Heuristic reranking: score = lexical + vector + metadata + scientific
- Top-K selection → Citations with snippets + trace

**Figure**: Fig 3 — Hybrid Retrieval Pipeline (4 branches → merge → rerank)

### §3.5 Federated Multi-Namespace Search (~250 words)

**Connector Protocol**:
```
NamespaceConnector: connect(), teardown(), index(), descriptor(), build_citation()
RetrievalCapabilities: search_lexical(), search_vector(), search_graph(), search_scientific()
```

**Capability Negotiation**:
- FilesystemConnector: lexical ✓, vector ✓, scientific ✓
- S3ObjectStoreConnector: lexical ✓, vector ✓, scientific ✓
- QdrantConnector: vector ✓
- Neo4jConnector: graph ✓
- HDF5Connector: metadata ✓, scientific ✓
- NetCDFConnector: metadata ✓, scientific ✓

**Multi-Namespace Query Flow**:
Query → for each namespace → single-namespace pipeline → collect all citations → global sort → deduplicate → top-k

**Incremental Indexing**:
- SHA256 content hash + mtime change detection
- Skip unchanged, re-index changed, remove deleted
- Efficient for evolving corpora

### §3.6 Multi-hop Agentic Retrieval (~200 words)

**Iterative Search Loop**:
1. Initial query → retrieve top-k
2. LLM evaluates relevance of results
3. LLM rewrites/refines query based on retrieved context
4. Re-query with refined query
5. Repeat until convergence or max iterations

**Query Rewriting Strategy**:
- Expand: add related terms discovered in initial results
- Narrow: focus on specific subquery when initial results are too broad
- Pivot: shift to related measurement or formula when direct match fails

**Integration**: Wraps the single-pass hybrid pipeline — each hop is a full pipeline execution with refined query.

---

## §4 Implementation (~750 words)

### §4.1 System Implementation (~350 words)

- **Language**: Python 3.11+
- **API**: FastAPI with async endpoints
- **Storage**: DuckDB (embedded, single-file)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2, 384-dim)
- **ANN**: ExactANNAdapter (sharded, 16 shards) or HnswANNAdapter (hnswlib)
- **CLI**: Click-based (clio query/index/serve/list/seed)

**DuckDB Schema** (8 tables):
| Table | Purpose |
|-------|---------|
| documents | URI, namespace, checksums |
| chunks | Text chunks with document refs |
| embeddings | 384-dim vectors per chunk |
| metadata | Key-value per document |
| file_index | Incremental indexing state |
| lexical_postings | BM25 token→chunk with TF |
| scientific_measurements | Raw + canonical measurements |
| scientific_formulas | Raw + normalized formulas |

**Write Safety**: file-based locking (fcntl on Unix)
**Observability**: OpenTelemetry (opt-in), Prometheus /metrics, trace events in every response
**Testing**: 138 automated tests (136 passing)

### §4.2 HDF5 and NetCDF Connector (~200 words)

**HDF5 (via h5py)**:
- Walk HDF5 groups and datasets
- Extract: dataset names, shapes, dtypes, attributes (units, descriptions)
- Extract measurements from attribute values (e.g., `pressure_unit: "kPa"`, `pressure_value: 250`)
- Chunk attribute metadata as text for hybrid search

**NetCDF (via xarray)**:
- Open datasets, extract CF convention metadata
- Variables: name, units, long_name, dimensions
- Global attributes: title, history, conventions
- Extract measurements from variable metadata
- Same SI canonicalization pipeline as text-based connectors

### §4.3 Experimental Platform (~150 words)

- Hardware: [machine specs]
- OS: Linux
- Python 3.11, DuckDB [version], sentence-transformers [version]
- Model: all-MiniLM-L6-v2 (22.7M parameters, 384-dim)
- Reproducibility: all code open-source, single `uv sync` setup

---

## §5 Evaluation (~1,875 words)

### §5.1 Experimental Setup (~300 words)

**Benchmark Construction**:
- Scientific retrieval tasks with deliberate unit-variation across documents
- 5 SI domains: distance, mass, time, pressure, velocity
- Each task: query in one unit, documents in 2-3 different unit representations
- N tasks total (exact number TBD after benchmark creation)

**Baselines**:
1. BM25-only (lexical)
2. Dense vector-only (all-MiniLM-L6-v2)
3. Hybrid BM25+vector (no science operators)
4. String normalization (Numbers Matter!-style: normalize unit strings, no conversion)
5. clio-agentic-search (full pipeline with science operators)

**Metrics**:
- Precision@K (K=1,3,5,10)
- Recall@K
- F1@K
- MRR (Mean Reciprocal Rank)

### §5.2 Dimensional Conversion Results (~400 words)

**RQ1**: Does dimensional conversion improve over baselines on cross-unit queries?

- Recall@K across 5 SI domains
- Breakdown: same-unit queries (where baselines should be equal) vs cross-unit queries (where our advantage shows)
- Key result: on cross-unit queries, baselines find only documents matching the query unit; our system finds ALL unit variants
- Statistical significance testing

**Figure**: Fig 4 — Bar chart: Recall@5 across baselines on cross-unit queries

### §5.3 Formula Matching Results (~250 words)

**RQ2**: Does formula normalization improve retrieval on formula-variant queries?

- Queries with formula in one notation, documents with formula in other notations
- Precision/Recall with and without formula operators
- Integration effect: formula + dimensional combined vs each alone

### §5.4 Federated Search Results (~300 words)

**RQ3**: Does federated search discover results hidden in other backends?

- Same corpus distributed across filesystem + S3 + HDF5
- Single-backend search vs multi-namespace search
- Coverage: % of relevant documents found by each approach
- Key result: single-backend misses documents in other tiers

### §5.5 Ablation Study (~325 words)

- Contribution of each retrieval branch independently:
  - Lexical only
  - Vector only
  - Lexical + Vector
  - Lexical + Vector + Scientific
  - Lexical + Vector + Scientific + Graph
  - Full pipeline
- Show each component adds independent value
- Scientific operators provide largest marginal gain on cross-unit queries

**Figure**: Fig 5 — Ablation: incremental F1@5 as branches are added

### §5.6 Multi-hop Agentic Retrieval Results (~150 words)

- Single-pass vs multi-hop on complex queries requiring multiple evidence pieces
- Number of hops before convergence
- Query rewriting examples

### §5.7 Indexing Performance (~150 words)

- Full index vs incremental index (5%, 10%, 25% corpus changes)
- Index time as function of corpus size
- Throughput: documents/second

**Figure**: Fig 6 — Incremental vs full indexing time

---

## §6 Conclusion (~600 words)

### §6.1 Summary (~300 words)
- Restated contributions with evaluation evidence
- Science-aware operators as a new class of retrieval primitives
- Dimensional conversion: guaranteed correct by construction, recovers X% false negatives
- Formula + measurement integration: no prior system
- Federated search: unified access across HPC storage

### §6.2 Limitations (~150 words)
- 14 SI units (vs CQE's 531) — coverage limited to 5 domains
- Heuristic reranking (not learned)
- Regex-based extraction (not NLP-based like CQE)
- Not yet tested at HPC scale (petabyte corpora)
- In-memory ANN (not distributed vector search)

### §6.3 Future Work (~150 words)
- Expand SI unit ontology (integrate CQE's 531-unit dictionary)
- Learned reranking (cross-encoder or ColBERT)
- HPC-scale deployment (Polaris, Frontier)
- MCP server wrapper for agent integration
- Integration with provenance systems (PROV-IO+)

---

## Figures Summary

| ID | Description | Section | Status |
|----|-------------|---------|--------|
| Fig 1 | System Architecture | §3.1 | To create |
| Fig 2 | Dimensional Conversion Pipeline | §3.2 | To create |
| Fig 3 | Hybrid Retrieval Pipeline | §3.4 | To create |
| Fig 4 | Recall@5 bar chart (baselines vs ours) | §5.2 | Needs eval |
| Fig 5 | Ablation F1@5 | §5.5 | Needs eval |
| Fig 6 | Indexing time chart | §5.7 | Needs eval |

## Tables Summary

| ID | Description | Section | Status |
|----|-------------|---------|--------|
| Table 1 | Capability comparison (7 systems × 10 caps) | §2.4 | Draft exists |
| Table 2 | SI Conversion Table (5 domains) | §3.2 | Draft exists |
| Table 3 | Evaluation results | §5.2 | Needs eval |

---

## Implementation Work Required (before April 8)

| Task | Priority | Estimated Effort |
|------|----------|-----------------|
| HDF5 connector (h5py) | HIGH | 1 day |
| NetCDF connector (xarray) | HIGH | 1 day |
| Multi-hop agentic loop | HIGH | 1 day |
| LLM query rewriting | HIGH | 0.5 day |
| Real Qdrant integration | MEDIUM | 0.5 day |
| Real Neo4j integration | MEDIUM | 0.5 day |
| Benchmark creation | CRITICAL | 1 day |
| Run all evaluations | CRITICAL | 1 day |
| Write full paper | CRITICAL | 2 days |
| Figures (3 diagrams + 3 charts) | HIGH | 1 day |
