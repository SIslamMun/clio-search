# SC2026 Abstract Submission — FINAL (Gap-Validated)

**Title**: Pluggable Science-Aware Operators for Agentic Retrieval over Federated HPC Data

**Track**: Data Analytics, Visualization, & Storage

**Authors**: S. Islam Munshi, Anthony Kougkas (Illinois Institute of Technology)

---

AI agents are transforming scientific computing — orchestrating experiments across national facilities, writing research papers, and querying workflow provenance via MCP. Yet agents consistently fail at data retrieval: ScienceAgentBench (ICLR 2025) shows the best agent achieves only 32% on scientific tasks, with failures traced to data handling rather than reasoning. The root cause: retrieval systems treat scientific content as plain text. Agentic RAG systems improve search orchestration but remain domain-agnostic — none incorporate science-aware numerical retrieval operators. Scientific RAG systems scale to millions of articles but search papers, not the data artifacts where scientific knowledge resides. No retrieval system combines domain-aware scientific operators with agentic multi-hop search across federated HPC storage.

We present clio-agentic-search, a retrieval engine that introduces pluggable science-aware operators as first-class retrieval branches alongside standard BM25 and dense vector search. A dimensional-analysis operator canonicalizes physical quantities to base SI units via explicit arithmetic conversion (kPa × 10³ = Pa), guaranteeing cross-prefix matching by construction — unlike string normalization, which cannot equate "kilopascal" with "pascal," or learned embeddings, which achieve only 0.54 accuracy on numerical content. A formula-normalization operator matches mathematical expressions across notation variants. These operators execute as parallel branches within a federated multi-namespace architecture that queries filesystems, object stores, vector databases, and scientific file formats through per-backend capability negotiation. An agentic loop with LLM-driven query rewriting enables iterative multi-hop search over federated scientific data.

We evaluate on scientific retrieval tasks spanning five measurement domains, demonstrating that science-aware operators recover results missed by all domain-agnostic baselines.

---

**Word count**: 247
