# Hypothesis and Motivation

## v0.5 — SC2026: Science-Aware Hybrid Retrieval with Dimensional Conversion for HPC Data Discovery

---

## The Three Failures of Scientific Retrieval

Scientific corpora encode structured knowledge that general-purpose retrieval cannot exploit. We identify three compounding failures:

**Failure 1: Dimensional Blindness**
Retrieval systems treat measurements as strings. "200 kPa" and "200000 Pa" are the same physical quantity but are never matched. Three approaches exist — string normalization (matches synonyms but not cross-prefix), learned embeddings (0.54 accuracy on numbers), and our approach (arithmetic conversion, guaranteed correct).

**Failure 2: Formula Opacity**
"F = ma" appears as "F = m · a," "F=m*a," "ma=F" across documents. No retrieval system normalizes and matches. Math-aware search handles LaTeX equation structure but not the connection between formulas and the physical measurements they describe.

**Failure 3: Storage Fragmentation**
HPC data lives across 5+ storage tiers. No system queries them simultaneously with science-aware operators. Provenance systems track lineage but offer no search. The HDF Group envisions knowledge graphs over HDF5 but no implementation exists.

---

## Research Questions

**RQ1**: Does dimensional-conversion-based retrieval (arithmetic: kPa × 10³ = Pa) improve precision and recall over (a) standard hybrid retrieval, (b) string-based quantity normalization, and (c) learned quantity embeddings on scientific queries with heterogeneous unit representations?

**RQ2**: Does integrating formula normalization with dimensional operators in a unified pipeline improve scientific document discovery compared to using either in isolation?

**RQ3**: Can federated retrieval across heterogeneous HPC storage backends maintain retrieval quality while enabling unified search that single-backend systems cannot achieve?

---

## Hypotheses

**H1 (Dimensional Conversion Superiority)**: Explicit dimensional conversion will achieve higher recall than all three alternatives (standard retrieval, string normalization, learned embeddings) on scientific queries with cross-prefix unit representations.

*Rationale*: String normalization [Almasian et al., 2024] cannot resolve "kPa" vs "Pa" — different strings after normalization. Learned embeddings [CONE, 2026] depend on training data coverage; the Numeracy Gap study [Deng et al., 2026] shows 0.54 accuracy on numbers. Arithmetic conversion (200 × 1000 = 200000) is guaranteed correct by construction.

**H2 (Formula + Measurement Synergy)**: Combining formula matching with dimensional operators will improve F1 by 15-25% over either alone on scientific queries that involve both equations and measurements.

*Rationale*: No prior system combines these. In scientific text, formulas and measurements co-occur (e.g., "PV = nRT where P = 200 kPa"). Retrieving by both modalities yields higher precision.

**H3 (Federated Coverage)**: Federated search across heterogeneous backends will discover relevant results that single-backend search misses entirely, with quality within 5% of centralized retrieval.

*Rationale*: HPC data is distributed by design. Data migration is impractical at petabyte scale. Federated search with per-backend capability negotiation enables unified discovery without centralization.

---

## Motivation: Why This Matters Now

### Converging Trends

1. **AI agents are entering HPC.** ORNL deploys autonomous experiment agents [SC'25]. MCP-based agents query workflow provenance [Souza et al., SC'25]. Federated agent middleware coordinates research infrastructure [Academy, 2025]. These agents need retrieval that understands science.

2. **Retrieval fails on scientific data.** Embedding models get 0.54 on numbers [EACL 2026]. String normalization can't cross SI prefixes. No system searches HDF5/NetCDF metadata. OpenScholar [Nature 2026] searches 45M papers but not scientific *data*.

3. **The gap is confirmed by surveys.** Agentic AI for Scientific Discovery [ICLR 2025] identifies data management as underexplored. The HDF Group envisions knowledge graphs over HDF5 but has no implementation [HDF Clinic, 2025]. PROV-IO+ [IEEE TPDS 2024] tracks HPC I/O provenance but provides no search.

### The Largest Unsolved Gap

> **No system indexes scientific file format metadata (HDF5 attributes, NetCDF CF metadata, Parquet schemas) into a unified search engine that supports natural language queries with dimensional reasoning.**

OpenScholar searches papers. HiPerRAG searches papers. Context-1 searches web documents. Nobody searches the actual scientific *data* on HPC systems.

---

## Why Prior Work Is Insufficient

| Capability | Numbers Matter! [2024] | CONE [2026] | Context-1 [2026] | HiPerRAG [2025] | OpenScholar [2026] | **Ours** |
|---|---|---|---|---|---|---|
| Unit string normalization | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Unit-aware embeddings | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| **Dimensional conversion** | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| **Formula matching** | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| **Numeric range (cross-unit)** | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| BM25 + vector hybrid | ✗ | ✗ | ✓ | ✓ | ✓ | **✓** |
| **Federated multi-backend** | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| HPC scientific domain | ✗ | ✗ | ✗ | ✓ | ✗ | **✓** |
| Agentic multi-hop | ✗ | ✗ | ✓ | ✗ | ✗ | planned |
| Searches papers | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| **Searches scientific data** | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |

The bottom row is the key insight: everyone searches *papers*. We search *data*.
