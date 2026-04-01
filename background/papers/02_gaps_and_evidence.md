# Gaps & Supporting Evidence

Paper: **Science-Aware Hybrid Retrieval with Dimensional Conversion for HPC Data Discovery**

---

## The Three Failures (Our Paper's Thesis)

### Failure 1: Numbers Are Broken in Retrieval

| Evidence | Source | Key Stat |
|---|---|---|
| Dense retrieval on numeric constraints | NC-Retriever/NumConQ [2025] | **16.3% accuracy** |
| Embeddings on numerical content | Numeracy Gap [EACL 2026] | **0.54 accuracy** (coin flip) |
| LLMs on number perturbations | GSM-Symbolic [ICLR 2025] | **65% accuracy drop** |
| Number embedding continuity | arXiv:2510.08009 [2025] | Non-continuous, per-digit encoding |
| LLM number understanding | NumericBench [ACL 2025] | Treats numbers as discrete tokens |

**Papers:**
- Numeracy Gap — https://arxiv.org/abs/2509.05691
- NC-Retriever — https://www.sciopen.com/article/10.26599/BDMA.2025.9020047
- GSM-Symbolic — https://arxiv.org/abs/2410.05229
- Numbers Not Continuous — https://arxiv.org/abs/2510.08009
- NumericBench — https://aclanthology.org/2025.findings-acl.1026/
- Number Cookbook [ICLR 2025] — https://openreview.net/forum?id=BWS5gVjgeY

### Failure 2: Nobody Searches Scientific Data (Only Papers)

| System | What it searches | Unit-aware? | Searches data? |
|---|---|---|---|
| OpenScholar [Nature 2026] | 45M papers | No | No |
| HiPerRAG [PASC 2025] | 3.6M papers | No | No |
| Context-1 [Chroma 2026] | Web documents | No | No |
| MCP for HPC [arXiv:2508.18489] | Globus Search (file metadata) | **No** | Metadata only |
| Datum [INL 2024] | File metadata (names, sizes) | No | Metadata only |
| BRINDEXER [CCGrid 2020] | POSIX metadata on Lustre | No | Metadata only |

**Gap**: Zero systems search *inside* HDF5, NetCDF, or Parquet files by measurement content. MCP for HPC is the closest architectural predecessor — it wraps Globus Search via MCP but Globus Search is keyword-only with no dimensional awareness.

**Papers:**
- Datum — https://inlsoftware.inl.gov/product/datum
- BRINDEXER — https://ieeexplore.ieee.org/document/9139660/
- HDF5 Knowledge Graph Vision (no implementation) — https://github.com/HDFGroup/hdf-clinic

### Failure 3: Provenance Captured But Not Used for Search

| System | What it does | Used for retrieval ranking? |
|---|---|---|
| PROV-IO+ [IEEE TPDS 2024] | Tracks HDF5 I/O provenance, <3.5% overhead | No |
| PROV-AGENT [eScience 2025] | W3C PROV for AI agent interactions + MCP | No |

**Gap**: Data lineage exists but is never used as a retrieval signal.

**Papers:**
- PROV-IO+ — https://ieeexplore.ieee.org/document/10472875/
- PROV-AGENT — https://arxiv.org/abs/2508.02866

---

## Supporting Infrastructure Papers

### Hybrid Retrieval Foundations
- **Fusion Functions** [ACM TOIS 2023] — Convex combination outperforms RRF. https://arxiv.org/abs/2210.11934
- **Exp4Fuse** [ACL 2025] — Modified RRF with LLM query expansion. https://aclanthology.org/2025.findings-acl.9/
- **BGE M3-Embedding** [ACL 2024] — Hybrid dense+sparse in single model. https://arxiv.org/abs/2402.03216

### Scientific Extraction (Feeds Our Pipeline)
- **SciEx** [2025] — LLM schema-guided extraction of measurements. https://arxiv.org/abs/2512.10004
- **CQE** [EMNLP 2023] — 531-unit extraction framework. https://arxiv.org/abs/2305.08853
- **Wiki-Quantities** [Scientific Data 2025] — 1.2M annotated quantities. https://www.nature.com/articles/s41597-025-05499-3
- **QUDT Ontology** — 1,300+ units with dimensional vectors. https://www.qudt.org/

### Federated Search
- **RAGRoute** [EuroMLSys 2025] — Dynamic source selection, 77.5% query reduction. https://arxiv.org/abs/2502.19280
- **Federated RAG Survey** [2025] — 18 studies mapped. https://arxiv.org/abs/2505.18906

### Reranking (Future Improvement Path)
- **Rank1** [COLM 2025] — Reasoning-based reranking. https://arxiv.org/abs/2502.18418
- **Rank-K** [2025] — Listwise reranking, +23% over RankZephyr. https://arxiv.org/abs/2505.14432

### Incremental Indexing
- **Quake** [OSDI 2025] — Adaptive vector indexing, 1.5-38x latency reduction. https://www.usenix.org/conference/osdi25/presentation/mohoney

---

## Novelty Validation Summary

| Our Claim | Who else does it? | Verified across 90+ papers |
|---|---|---|
| SI dimensional conversion (arithmetic) | **Nobody** | ✓ |
| Formula matching in retrieval | **Nobody** | ✓ |
| Formula + measurement unified | **Nobody** | ✓ |
| Federated search + science-aware operators | **Nobody** | ✓ |
| Searches scientific data (not papers) | **Nobody** | ✓ |
| BM25 + vector + scientific as parallel branches | **Nobody** | ✓ |
