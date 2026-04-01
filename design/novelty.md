# Novelty Statement — v0.5 FINAL

## Core Thesis

> General-purpose retrieval fails on scientific data because it lacks three capabilities: dimensional reasoning over physical quantities, formula understanding across notations, and unified search across heterogeneous HPC storage. We introduce **science-aware retrieval operators** — a new class of retrieval primitives that address all three.

---

## The Three Unsolved Problems

### Problem 1: Dimensional Quantity Matching

**What exists:**
- String normalization (Heidelberg: QFinder 2022, Numbers Matter! 2024) — "kPa" → "kilopascal"
- Learned embeddings (CONE 2026) — encode units in vector space
- Numeric constraint learning (NC-Retriever 2025) — learn number comparison in embeddings

**What fails:**
```
"200 kPa" vs "200000 Pa"

String normalization: "kilopascal" ≠ "pascal" → NO MATCH
Learned embeddings: maybe matches, 0.54 accuracy on numbers → UNRELIABLE
Our approach: 200 × 1000 = 200000 = 200000 → GUARANTEED MATCH
```

**Our contribution:** Arithmetic dimensional conversion using SI multiplication factors. Guaranteed correct by construction.

### Problem 2: Formula Matching + Measurement Integration

**What exists:**
- Math-aware search (Approach0, SSEmb CIKM 2025, MIRB 2025) — LaTeX equation structure matching
- Quantity extraction (CQE EMNLP 2023, MeasEval 2021) — extract measurements from text

**What nobody does:**
- Match "F = ma" with "F = m * a" with "ma = F" in the same retrieval query
- Combine formula matching WITH dimensional quantity matching in one pipeline
- Use formula context to improve measurement retrieval ("F = ma where F = 200 N")

**Our contribution:** Formula normalization (whitespace, superscripts, factor ordering, side sorting) unified with dimensional operators as parallel retrieval branches.

### Problem 3: Scattered Scientific Data

**What exists:**
- Federated RAG (RAGRoute 2025, DF-RAG 2026) — route queries to relevant backends
- HPC provenance (PROV-IO+ IEEE TPDS 2024, PROV-AGENT 2025) — track data lineage
- HDF5 knowledge graph vision (HDF Clinic 2025) — concept, no implementation

**What nobody does:**
- Query filesystem + S3 + vector DB + graph DB simultaneously WITH science-aware operators
- Index HDF5 attributes / NetCDF metadata into a searchable engine
- Per-backend capability negotiation (only run branches the backend supports)

**Our contribution:** Connector-based federated architecture with dynamic capability negotiation. Incremental indexing with SHA256 change detection.

---

## The Biggest Insight

Everyone searches **papers**. Nobody searches **data**.

- OpenScholar [Nature 2026] → 45M papers
- HiPerRAG [2025] → 3.6M papers
- Context-1 [2026] → web documents
- Numbers Matter! [2024] → finance/medicine documents

**Nobody searches HPC scientific data** — the HDF5 files, simulation outputs, experimental logs, Darshan I/O profiles that contain the actual measurements.

Our system searches the *data*, not just the papers about the data.

---

## Validated Comparison Table

| Capability | Numbers Matter! | CONE | Context-1 | HiPerRAG | OpenScholar | **Ours** |
|---|---|---|---|---|---|---|
| **Dimensional conversion** (kPa×10³=Pa) | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| **Formula matching** | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| **Cross-unit range queries** | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| **Federated multi-backend** | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| **Searches scientific data** (not just papers) | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Unit string normalization | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Unit-aware embeddings | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| BM25 + vector hybrid | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |
| Agentic multi-hop | ✗ | ✗ | ✓ | ✗ | ✗ | planned |
| HPC-scale tested | ✗ | ✗ | ✗ | ✓ | ✗ | needs eval |

Every row in bold is unique to our system. The first five rows are **completely empty** for all prior systems.
