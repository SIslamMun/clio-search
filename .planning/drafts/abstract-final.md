# SC2026 Abstract Submission — FINAL

**Title**: Pluggable Science-Aware Operators for Agentic Retrieval over Federated HPC Data

**Track**: Data Analytics, Visualization, & Storage

**Authors**: S. Islam Munshi, Anthony Kougkas (Illinois Institute of Technology)

---

AI agents are transforming scientific computing — orchestrating experiments, writing papers, and querying workflow provenance across national facilities. Yet these agents consistently fail at data retrieval: the best agent achieves only 32% on scientific tasks, with failures traced to data handling, not reasoning. The root cause is that retrieval systems treat scientific content as plain text. Agentic RAG systems (Context-1, A-RAG) improve search orchestration but remain domain-agnostic. Scientific RAG systems (HiPerRAG, OpenScholar) scale to millions of articles but search papers, not the data artifacts — HDF5 files, simulation outputs, experimental logs — where scientific knowledge resides. No retrieval system combines domain-aware scientific operators with agentic multi-hop search across federated HPC storage.

We present clio-agentic-search, a retrieval engine that introduces pluggable science-aware operators as first-class retrieval primitives alongside standard BM25 and dense vector search. A dimensional-analysis operator canonicalizes physical quantities to base SI units via arithmetic conversion (kPa × 10³ = Pa), guaranteeing cross-prefix matching by construction. A formula-normalization operator matches mathematical expressions across notation variants. A scientific-metadata operator indexes HDF5 attributes and NetCDF variables directly. These operators execute as parallel branches within a federated multi-namespace architecture that searches across filesystems, object stores, vector databases, and scientific file formats through per-backend capability negotiation. An agentic retrieval loop with LLM-driven query rewriting orchestrates multi-hop search, refining queries across federated scientific data.

We evaluate on scientific retrieval tasks spanning five measurement domains. Science-aware operators recover results missed by all domain-agnostic baselines, establishing pluggable domain operators as a generalizable extension to agentic retrieval for HPC data discovery.

---

**Word count**: 249
