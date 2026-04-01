# Science-Aware Hybrid Retrieval with Dimensional Conversion for HPC Data Discovery

## What This Is

A research paper for **SC2026** (Data Analytics, Visualization & Storage track) presenting clio-agentic-search, a hybrid retrieval engine that introduces science-aware retrieval operators for AI agents searching HPC data. IEEE double-column format, 10 pages.

## Core Argument

General-purpose retrieval fails on scientific data because it lacks dimensional reasoning, formula understanding, and unified search across heterogeneous HPC storage. We introduce **science-aware retrieval operators** — a new class of retrieval primitives that perform arithmetic SI conversion (guaranteed correct by construction), normalize formulas across notations, and federate search across HPC storage backends — enabling AI agents to find scientific data, not just papers about it.

## Requirements

### Must Have

- [ ] Dimensional conversion retrieval with SI arithmetic (kPa×10³=Pa) — implemented and evaluated
- [ ] Formula normalization and matching — implemented and evaluated
- [ ] Federated multi-namespace search with capability negotiation — FS, S3, Qdrant, Neo4j functional
- [ ] HDF5/NetCDF connector — index scientific file format metadata
- [ ] Multi-hop iterative retrieval — agentic search loop with re-querying
- [ ] LLM query rewriting — expand/reformulate queries before retrieval
- [ ] Evaluation benchmark with unit-variation queries and baselines
- [ ] Comparison against: standard BM25, dense vector, Numbers Matter!-style string normalization
- [ ] Ablation study across retrieval branches
- [ ] Indexing performance benchmarks
- [ ] System architecture figure
- [ ] Comparison table vs prior work (Numbers Matter!, CONE, Context-1, HiPerRAG, OpenScholar)

### Should Have

- [ ] Scalability analysis on larger corpora
- [ ] Real Qdrant/Neo4j backend integration (not in-memory mock)
- [ ] Cross-unit range query evaluation with multiple SI domains
- [ ] Prometheus metrics and observability discussion

### Out of Scope

*Nothing explicitly excluded per user direction — all features are in scope for implementation before April 8.*

## Target Audience

Broad SC2026 audience across three communities:
- **HPC systems**: Storage architects, I/O researchers — care about federated search across HPC tiers
- **AI/ML + HPC**: Agent builders, LLM integration researchers — care about agentic retrieval for science
- **Data management**: FAIR data, provenance, metadata — care about scientific data discovery

Assume familiarity with BM25, dense retrieval, and basic HPC storage concepts. Do NOT assume familiarity with quantity-aware retrieval literature (Heidelberg group) or agentic search patterns.

## Constraints

- **Abstract deadline**: April 1, 2026
- **Full paper deadline**: April 8, 2026
- **Format**: IEEE double-column, 10 pages + references
- **Code**: clio-agentic-search exists with core scientific operators working on FS/S3; stubs for Qdrant/Redis/Neo4j; missing HDF5/NetCDF, multi-hop, LLM rewriting
- **Prior work**: ~28 papers surveyed, detailed threat analysis done
- **Professor**: Has not yet seen any drafts — abstract submission will be first contact
- **Author**: Claude Code wrote the implementation; user directed design

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Build all features by April 8 | User wants a complete system paper, not partial claims | — Pending |
| IEEE format (not ACM) | SC2026 Data Analytics track | ✓ Good |
| All features in scope | No explicit exclusions — HDF5, multi-hop, LLM rewriting all claimed | — Pending |
| Build benchmark from scratch | No existing scientific retrieval benchmark with unit-variation queries | — Pending |
| Use existing drafts as context only | Starting fresh with WTF-P, not carrying forward verbatim | ✓ Good |

---
*Last updated: 2026-03-31 after WTF-P initialization*
