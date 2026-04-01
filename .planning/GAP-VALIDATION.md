# Gap Validation — Novelty Audit

**Date**: 2026-03-31
**Status**: VALIDATED with narrowed claims

---

## Core Novelty (UNIQUE — no prior work)

### 1. Arithmetic SI Dimensional Conversion in Retrieval
**Claim**: We perform explicit multiplication (kPa × 10³ = Pa) to canonicalize measurements at index and query time.
**Validation**: No retrieval system does this.
- Numbers Matter! (EMNLP 2024): string normalization only — "kilopascal" ≠ "pascal"
- CONE (SIGMOD 2026): learned embeddings — probabilistic, not guaranteed
- Unit Harmonization (de la Torre, 2025): medical domain, BM25+embeddings for unit matching — no SI arithmetic conversion
- NC-Retriever (2025): learns numeric constraints — no unit conversion
**Result**: ✅ UNIQUE

### 2. Combined Dimensional Conversion + Formula Matching
**Claim**: We normalize formulas AND integrate with dimensional operators as parallel retrieval branches.
**Validation**: No system combines both.
- Math-aware search (Approach0, SSEmb): formula structure only, no unit awareness
- Numbers Matter!: units only, no formula matching
- No intersection found in any paper
**Result**: ✅ UNIQUE

### 3. Science-Aware Operators as Parallel Retrieval Branches
**Claim**: Domain-specific scientific operators (unit, formula) execute alongside BM25+vector as equal retrieval branches.
**Validation**: Retrieval pipelines don't have science-aware branches.
- A-RAG (2026): hierarchical interfaces but domain-agnostic
- Context-1 (2026): agentic search but domain-agnostic
- Collaborative Multi-Agent RAG (2024): agent-per-DB-type, not agent-per-science-modality
**Result**: ✅ UNIQUE

---

## Narrowed Claims (prior work exists in adjacent areas)

### 4. "No agentic RAG has domain operators"
**Original**: No agentic retrieval system has domain-specific operators
**Counter**: CLADD (AAAI 2025) has drug KG operators for biomedical retrieval
**Narrowed to**: "No agentic RAG system has science-aware *numerical* retrieval operators (dimensional conversion, formula matching)"
**Result**: ⚠️ VALID with narrowed wording

### 5. "No system searches scientific data, only papers"
**Original**: Retrieval systems search papers, not data
**Counter**: HDF5-FastQuery (2006), MIQS (2019) search HDF5 data — but pre-LLM
**Narrowed to**: "No *agentic or LLM-based* retrieval system searches scientific data artifacts (HDF5, simulation outputs)"
**Result**: ⚠️ VALID with narrowed wording

### 6. "Pluggable domain operators for retrieval"
**Original**: Pluggable operators are a novel concept
**Counter**: Elasticsearch plugins (QFinder is literally an ES plugin), LangChain custom retrievers
**Narrowed to**: "The application of pluggable *science-aware* operators (dimensional, formula) as first-class retrieval branches is novel"
**Result**: ⚠️ VALID — novel application, not novel concept

### 7. "No multi-agent by science modality"
**Original**: No multi-agent system specializes by science modality
**Counter**: SciAgent (2025) has Chemistry/Physics/Math worker agents
**Narrowed to**: "No multi-agent *retrieval* system specializes by scientific measurement modality"
**Result**: ⚠️ VALID with narrowed wording

---

## Key Stats Verified

| Stat | Source | Verified | Notes |
|------|--------|----------|-------|
| 0.54 accuracy on numbers | Numeracy Gap, EACL 2026 | ✅ | Binary retrieval accuracy (near random 0.50). Don't overstate. |
| 32.4% agent success | ScienceAgentBench, **ICLR 2025** | ✅ | Fix venue (was wrong as NeurIPS 2024). 42.2% with o1+self-debug. |
| 88% answer recall | Context-1, Chroma 2026 | ✅ | vs 58% single-pass |
| 16.3% dense retrieval on numeric | NC-Retriever/NumConQ, 2025 | ✅ | |
| 25% Recall@10 improvement | CONE, SIGMOD 2026 | ✅ | |

---

## Conclusion

**The gap is real and validated.** The narrowed claims are still strong:

> No existing retrieval system combines arithmetic dimensional conversion with formula normalization as parallel science-aware retrieval branches in an agentic, federated pipeline for HPC scientific data.

Every component of this sentence is verified unique.
