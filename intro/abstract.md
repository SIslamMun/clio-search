# Abstract

## v0.5 FINAL — SC2026 (250 words)

**Title**: Science-Aware Hybrid Retrieval with Dimensional Conversion for HPC Data Discovery

**Track**: Data Analytics, Visualization, & Storage

---

Scientific corpora encode structured knowledge — dimensional quantities, mathematical formulas, and heterogeneous storage provenance — that general-purpose retrieval systems cannot exploit. AI agents searching HPC data face three compounding failures: retrieval systems cannot match measurements across unit prefixes ("200 kPa" versus "200000 Pa"), cannot match formulas across formatting variations ("F=ma" versus "F = m · a"), and cannot query across the heterogeneous storage backends where scientific data resides. Embedding models achieve only 0.54 accuracy on numerical content, and existing quantity-aware systems normalize unit strings without performing dimensional conversion across SI prefixes.

We present clio-agentic-search, a hybrid retrieval engine that introduces science-aware retrieval operators for HPC data discovery. First, dimensional-conversion-based measurement retrieval canonicalizes quantities to base units using explicit multiplication factors (kPa × 10³ = Pa), enabling cross-prefix numeric comparison guaranteed correct by construction — unlike string normalization or learned embeddings. Second, formula normalization matches mathematical expressions regardless of formatting, whitespace, or factor ordering, unified with measurement operators in a single pipeline. Third, federated multi-namespace search queries heterogeneous HPC storage backends (filesystems, S3, vector databases, graph databases) through a connector architecture with per-backend capability negotiation.

These science-aware operators execute as parallel branches alongside BM25 lexical and dense vector retrieval, with results merged through deduplication and heuristic reranking. Incremental indexing with content-hash change detection enables efficient re-indexing of evolving corpora.

We evaluate on scientific retrieval tasks demonstrating that dimensional conversion recovers results missed by both standard hybrid search and string-based quantity normalization, establishing science-aware operators as a new class of retrieval primitives for AI-driven HPC data discovery.
