# Abstract — SC2026

**Title**: Science-Aware Hybrid Retrieval with Dimensional Conversion for HPC Data Discovery

**Track**: Data Analytics, Visualization, & Storage

---

Scientific corpora encode dimensional quantities, mathematical formulas, and heterogeneous storage provenance that general-purpose retrieval systems cannot exploit. AI agents searching HPC data face three compounding failures: retrieval systems cannot match measurements across unit prefixes ("200 kPa" vs. "200000 Pa"), cannot match formulas across formatting variations ("F=ma" vs. "F = m · a"), and cannot query across the heterogeneous storage backends where scientific data resides. Embedding models achieve only 0.54 accuracy on numerical content, and existing quantity-aware systems normalize unit strings without performing dimensional conversion across SI prefixes.

We present clio-agentic-search, a hybrid retrieval engine that introduces science-aware retrieval operators for HPC data discovery. First, dimensional-conversion measurement retrieval canonicalizes quantities to base SI units via explicit multiplication (kPa × 10³ = Pa), enabling cross-prefix numeric comparison guaranteed correct by construction — unlike string normalization or learned embeddings. Second, formula normalization matches mathematical expressions regardless of formatting, whitespace, or factor ordering, unified with measurement operators in a single retrieval pipeline. Third, federated multi-namespace search queries heterogeneous HPC storage backends — including filesystems, object stores, vector databases, and scientific file formats such as HDF5 and NetCDF — through a connector architecture with per-backend capability negotiation. An agentic retrieval loop with LLM-driven query rewriting enables multi-hop search over federated scientific data.

We evaluate on scientific retrieval tasks with cross-unit queries spanning five SI domains. Dimensional conversion recovers results missed by standard hybrid search, string-based quantity normalization, and learned embeddings, establishing science-aware operators as a new class of retrieval primitives for AI-driven HPC data discovery.

---

**Word count**: 237
