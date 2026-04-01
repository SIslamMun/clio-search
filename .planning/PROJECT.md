# Pluggable Science-Aware Operators for Agentic Retrieval over Federated HPC Data

## What This Is

A research paper for **SC2026** (Data Analytics, Visualization & Storage track) presenting clio-agentic-search, a hybrid retrieval engine that introduces science-aware retrieval operators for AI agents searching HPC data. IEEE double-column format, 10 pages.

## Core Argument

General-purpose retrieval fails on scientific data because it lacks dimensional reasoning, formula understanding, and unified search across heterogeneous HPC storage. We introduce **science-aware retrieval operators** — a new class of retrieval primitives that perform arithmetic SI conversion (guaranteed correct by construction), normalize formulas across notations, and federate search across HPC storage backends — enabling AI agents to find scientific data, not just papers about it.

## Implementation Status (as of 2026-04-01)

### Fully Implemented and Tested
- [x] Dimensional conversion retrieval with SI arithmetic (13 units, 5 domains) — 495 lines
- [x] Formula normalization and matching (whitespace, superscript, side-swap, factor reorder)
- [x] Hybrid 4-branch retrieval (lexical BM25 + vector + graph + scientific)
- [x] Filesystem connector with incremental indexing
- [x] S3 Object Store connector
- [x] HDF5 connector (h5py) — extracts datasets, attributes, units, measurements
- [x] NetCDF connector (xarray) — extracts CF convention metadata, variables, units
- [x] Multi-hop agentic retrieval loop (configurable max_hops, convergence detection)
- [x] LLM query rewriting via Anthropic API (expand/narrow/pivot/done)
- [x] Fallback query rewriting (offline SI unit variant expansion)
- [x] DuckDB storage with 8 tables including scientific_measurements/formulas
- [x] Okapi BM25 in SQL (k1=1.2, b=0.75)
- [x] CLI with --agentic, --max-hops, --llm-rewrite, --numeric-range, --formula flags
- [x] FastAPI with /query, /documents, /jobs/index, /health, /metrics endpoints
- [x] 160+ automated tests (27 new for agentic/HDF5/NetCDF features)

### Stubs (In-Memory Only)
- [ ] Qdrant vector store connector (in-memory mock, no real HTTP client)
- [ ] Neo4j graph store connector (in-memory mock, no real Bolt protocol)
- [ ] Redis KV/log store connector (in-memory mock)

### Evaluation (NOT YET DONE)
- [ ] Evaluation benchmark with unit-variation queries and baselines
- [ ] Comparison against: standard BM25, dense vector, Numbers Matter!-style string normalization
- [ ] Ablation study across retrieval branches
- [ ] Indexing performance benchmarks

## Requirements

### Must Have (for paper)
- [x] Dimensional conversion retrieval with SI arithmetic
- [x] Formula normalization and matching
- [x] Federated multi-namespace search with capability negotiation
- [x] HDF5/NetCDF connector — index scientific file format metadata
- [x] Multi-hop iterative retrieval — agentic search loop with re-querying
- [x] LLM query rewriting — expand/reformulate queries before retrieval
- [ ] Evaluation benchmark with unit-variation queries and baselines
- [ ] Comparison against: standard BM25, dense vector, Numbers Matter!-style
- [ ] Ablation study across retrieval branches
- [ ] System architecture figure
- [ ] Comparison table vs prior work

### Should Have
- [ ] Scalability analysis on larger corpora
- [ ] Cross-unit range query evaluation with multiple SI domains
- [ ] Indexing performance benchmarks

### Out of Scope
- Real Qdrant/Neo4j integration (stubs are sufficient for paper claims)
- Prometheus metrics discussion (exists in code but not central to contribution)

## Target Audience

SC2026 audience across three communities:
- **HPC systems**: Storage architects, I/O researchers — federated search across HPC tiers
- **AI/ML + HPC**: Agent builders, LLM integration researchers — agentic retrieval for science
- **Data management**: FAIR data, provenance, metadata — scientific data discovery

## Constraints

- **Abstract deadline**: April 1, 2026 (TODAY)
- **Full paper deadline**: April 8, 2026
- **Format**: IEEE double-column, 10 pages + references
- **Codebase**: ~10,000 lines Python, 160+ tests, all core features implemented
- **~55 papers** surveyed and verified via web search

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Honest "agentic" claim | Now backed by real multi-hop loop + LLM rewriting | ✓ Implemented |
| Narrow novelty claims | PANGAEA-GPT/LLM-Find search data too — but not with SI conversion | ✓ Documented |
| Drop unverifiable stats | Context-1 "88% recall" unverifiable from public sources | ✓ Removed |
| Fix ScienceAgentBench | Was NeurIPS 2024, actually ICLR 2025 | ✓ Fixed |

---
*Last updated: 2026-04-01 after full implementation and independent literature verification*
