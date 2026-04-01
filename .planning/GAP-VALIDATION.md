# Gap Validation — Novelty Audit (Independently Verified)

**Date**: 2026-04-01
**Method**: Full repo analysis + web search verification of all 16 cited papers + search for missed 2026 papers
**Status**: VALIDATED with further narrowed claims after discovering PANGAEA-GPT

---

## Citation Verification Summary

All 16 previously cited papers **verified as real** via web search. Issues found:
- ScienceAgentBench: venue is **ICLR 2025** (not NeurIPS 2024) — outline.md still has the error
- Context-1 "88% recall": **UNVERIFIABLE** from public sources (Chroma research page shows different metrics: 0.94 prune accuracy, comparisons to GPT-5.4)
- "Supercharging Federated Retrieval": actual title is "Supercharging Federated **Intelligence** Retrieval"
- CONE "SIGMOD 2026": venue claim **unverified** (arXiv exists, SIGMOD acceptance not confirmed)

---

## Implementation vs Claims Audit

| Claimed Feature | Actually Implemented? | Evidence |
|---|---|---|
| SI unit conversion (kPa×10³=Pa) | **YES** — 13 units, 5 domains | `indexing/scientific.py` lines 79-85, tested |
| Formula normalization | **YES** — whitespace, superscript, side-swap, factor reorder | `indexing/scientific.py` lines 88-114, tested |
| Hybrid retrieval (4 branches) | **YES** — lexical+vector+graph+scientific | `retrieval/coordinator.py`, tested |
| Filesystem connector | **YES** — full, with incremental indexing | `connectors/filesystem/connector.py` |
| S3 connector | **YES** — full, in-memory S3 client | `connectors/object_store/connector.py` |
| Qdrant connector | **STUB** — in-memory only, no real Qdrant | `connectors/vector_store/connector.py` |
| Neo4j connector | **STUB** — in-memory only, no real Neo4j | `connectors/graph_store/connector.py` |
| HDF5/NetCDF support | **NOT IMPLEMENTED** — zero code | No imports, no handlers |
| Multi-hop agentic loop | **NOT IMPLEMENTED** — zero code | No LLM deps, no loop logic |
| LLM query rewriting | **NOT IMPLEMENTED** — zero code | No anthropic/openai in deps |
| DuckDB storage (8 tables) | **YES** — scientific_measurements + scientific_formulas tables | `storage/duckdb_store.py` |
| Okapi BM25 | **YES** — SQL-native, k1=1.2, b=0.75 | `storage/duckdb_store.py` lines 623-701 |

**Bottom line**: The paper calls itself "agentic" but has zero agentic functionality. The actual contribution is a **science-aware hybrid retrieval engine** with working SI conversion and formula normalization.

---

## Core Novelty (UNIQUE — verified via web search)

### 1. Arithmetic SI Dimensional Conversion in Retrieval
**Claim**: Explicit multiplication (kPa × 10³ = Pa) to canonicalize measurements at index and query time.
**Validation**: Searched "dimensional analysis retrieval", "unit conversion search", "quantity-aware retrieval 2025 2026". **No system found.**
- Numbers Matter! (EMNLP 2024): string normalization — cannot equate "kilopascal" and "pascal"
- CONE (arXiv:2603.04741): learned embeddings — probabilistic, not guaranteed
- Unit Harmonization (arXiv:2505.00810): medical domain, BM25+embeddings — no SI arithmetic
- PANGAEA-GPT (arXiv:2602.21351): validates unit scales **post-hoc** but does NOT convert at retrieval time
- NC-Retriever (2025): learns numeric constraints — no unit conversion
**Result**: ✅ UNIQUE

### 2. Combined Dimensional Conversion + Formula Matching
**Claim**: Formula normalization AND dimensional operators as parallel retrieval branches.
**Validation**: Searched formula retrieval systems. No intersection found.
- Math-aware search (Approach0, SSEmb, MIRB): formula structure only, no unit awareness
- Numbers Matter!: units only, no formula matching
**Result**: ✅ UNIQUE

### 3. Science-Aware Operators as First-Class Retrieval Branches
**Claim**: Unit/formula operators execute alongside BM25+vector as equal retrieval branches.
**Validation**: No retrieval pipeline has science-aware branches.
- A-RAG (2026): hierarchical but domain-agnostic
- CLADD (AAAI 2025): domain KG operators, but for drug discovery, not measurement retrieval
- PANGAEA-GPT: data-type-aware routing, but no measurement operators in retrieval pipeline
**Result**: ✅ UNIQUE

---

## Narrowed Claims (prior work exists — must differentiate)

### 4. "No agentic RAG has domain operators"
**Counter**: CLADD (AAAI 2025) has drug KG operators; PANGAEA-GPT has data-type-aware routing
**Narrowed to**: "No agentic RAG system has science-aware *numerical* retrieval operators (arithmetic dimensional conversion, formula matching)"
**Result**: ⚠️ VALID with narrowed wording

### 5. "No system searches scientific data, only papers" — **SIGNIFICANTLY NARROWED**
**Original**: Retrieval systems search papers, not data
**Counters found**:
- PANGAEA-GPT (arXiv:2602.21351, Feb 2026): searches 400K+ geoscientific datasets with specialist data agents, 8.14/10 vs 2.87/10 baseline, includes unit scale validation
- LLM-Find (arXiv:2407.21024, 2025): LLM agent retrieving actual geospatial data from heterogeneous sources, 80-90% success
- HDF5-FastQuery (2006), MIQS (2019), DAI/JULEA (ICCS 2024): pre-LLM HDF5 query systems
**Narrowed to**: "No retrieval system searches scientific data with arithmetic dimensional conversion and formula matching as retrieval operators"
**Result**: ⚠️ VALID but previous wording was **wrong** — PANGAEA-GPT and LLM-Find DO search scientific data

### 6. "Pluggable domain operators for retrieval"
**Counter**: Elasticsearch plugins (QFinder), LangChain custom retrievers
**Narrowed to**: "The application of pluggable *science-aware* operators (dimensional, formula) as first-class retrieval branches is novel"
**Result**: ⚠️ VALID — novel application, not novel concept

### 7. "No multi-agent by science modality"
**Counter**: SciAgent (2025), PANGAEA-GPT (2026) have specialist science agents
**Narrowed to**: "No multi-agent *retrieval* system uses measurement-aware operators per scientific modality"
**Result**: ⚠️ VALID with narrowed wording

---

## Critical Papers MISSED by Previous Session

| Paper | arXiv | Date | Threat | Impact |
|---|---|---|---|---|
| **PANGAEA-GPT** | 2602.21351 | Feb 2026 | **HIGH** | Searches 400K+ geoscientific datasets, unit scale validation, multi-agent. Directly challenges "nobody searches data" |
| **ScienceClaw/Infinite** | 2603.14312 | Mar 2026 | **MODERATE** | 300+ skill agent swarm, artifact provenance DAG. Strengthens motivation but doesn't do measurement retrieval |
| **MIRB** | 2505.15585 | May 2025 | **MODERATE** | First unified formula retrieval benchmark (12 datasets). Provides evaluation context |
| **Theoretical Limits of Embedding Retrieval** | 2508.21038 | Aug 2025 | **MODERATE** | Proves embeddings have fundamental retrieval limits. **Strengthens our case** |
| **LLM-Find** | 2407.21024 | 2025 | **MODERATE** | LLM agent retrieving actual geospatial data. Challenges "nobody searches data" |
| **FRAG** | 2410.13272 | Oct 2024 | **LOW** | Federated vector DB with encrypted ANN. Privacy-focused, not science-aware |
| **DAI/JULEA** | Springer ICCS 2024 | 2024 | **LOW** | Pre-computation for HDF5 queries, 22,000x speedup. Non-AI predecessor |
| **HPC-FAIR** | OSTI 2504172 | 2025 | **LOW** | FAIR framework for HPC. Background context |

---

## Key Stats Verification

| Stat | Source | Status | Notes |
|------|--------|--------|-------|
| 0.54 accuracy on numbers | Numeracy Gap, EACL 2026 Findings (arXiv:2509.05691) | ✅ Verified | Binary retrieval accuracy across 13 models. Near random (0.50). |
| 32.4% agent success | ScienceAgentBench, **ICLR 2025** (arXiv:2410.05080) | ✅ Verified | 34.3% with expert knowledge + self-debug. Without expert knowledge: 32.4%. |
| 88% answer recall | Context-1, Chroma 2026 | ❓ UNVERIFIABLE | Public page shows different metrics. Use with caution or drop. |
| 16.3% dense retrieval on numeric | NC-Retriever/NumConQ, 2025 | ✅ Verified | |
| 25% Recall@10 improvement | CONE, arXiv:2603.04741 | ✅ Verified | SIGMOD 2026 venue unconfirmed |

---

## Conclusion

**The core gap is real and verified.** But the framing needs honest revision:

### What we CAN claim:
> No existing retrieval system performs arithmetic SI dimensional conversion to canonicalize measurements at index and query time, nor combines this with formula normalization as parallel science-aware retrieval branches.

### What we CANNOT claim (as previously written):
- ~~"No system searches scientific data"~~ → PANGAEA-GPT and LLM-Find do
- ~~"88% recall for agentic search"~~ → Unverifiable stat
- ~~"Agentic retrieval pipeline"~~ → Not implemented in code
- ~~"ScienceAgentBench, NeurIPS 2024"~~ → ICLR 2025

### What should change in the abstract:
1. Drop or qualify "agentic" — the code has no agentic loop. Either implement it before April 8 or reframe as "science-aware hybrid retrieval"
2. Drop Context-1 "88% recall" or qualify as "reported by authors"
3. Cite PANGAEA-GPT as concurrent work and differentiate (they validate units post-hoc, we convert at retrieval time)
4. Strengthen with "Theoretical Limits of Embedding Retrieval" (arXiv:2508.21038) to justify why explicit operators beat embeddings
