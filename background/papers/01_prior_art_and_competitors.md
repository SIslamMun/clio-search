# Prior Art & Competitors — Who We Must Cite and Differentiate Against

Paper: **Science-Aware Hybrid Retrieval with Dimensional Conversion for HPC Data Discovery**

---

## A. Quantity-Aware Retrieval (CLOSEST PRIOR ART — must cite all)

### 1. QFinder: A Framework for Quantity-centric Ranking
- **Authors**: Satya Almasian, Michael Gertz (University of Heidelberg)
- **Year**: 2022 | **Venue**: SIGIR 2022
- **URL**: https://dl.acm.org/doi/abs/10.1145/3477495.3531672
- **What it does**: Elasticsearch plugin for quantity-aware ranking. Extracts (value, unit) tuples, handles =, >, < conditions on quantities.
- **Limitation**: Normalizes unit *strings* only. Cannot match "200 kPa" with "200000 Pa" — different canonical strings.
- **How we differ**: We do arithmetic SI conversion (200 × 1000 = 200000). They do string matching.
- **Threat level**: HIGH — established prior art, must cite and clearly differentiate.

### 2. CQE: A Comprehensive Quantity Extractor
- **Authors**: Satya Almasian, Vivian Kazakova, Michael Gertz (Heidelberg)
- **Year**: 2023 | **Venue**: EMNLP 2023
- **URL**: https://arxiv.org/abs/2305.08853
- **What it does**: Extraction framework with 531 units, dependency parsing, string normalization. First to detect concepts associated with quantities.
- **Limitation**: Extraction only — not integrated into retrieval ranking. Their unit dictionary (531 units) is more comprehensive than our 14 units.
- **How we differ**: We integrate extraction INTO the retrieval pipeline as a scoring branch. They extract and store.
- **Threat level**: MEDIUM — must acknowledge their extraction is more comprehensive.

### 3. Numbers Matter! Bringing Quantity-awareness to Retrieval Systems
- **Authors**: Satya Almasian, Milena Bruseva, Michael Gertz (Heidelberg)
- **Year**: 2024 | **Venue**: EMNLP 2024 Findings
- **URL**: https://arxiv.org/abs/2407.10283
- **What it does**: Full quantity-aware retrieval. CQE extracts (value, unit) tuples, normalizes unit strings, ranks by textual similarity + numerical proximity. Creates FinQuant and MedQuant benchmarks.
- **Limitation**: String normalization, not SI dimensional conversion. "kilopascal" ≠ "pascal" so cross-prefix matching fails. Finance/medicine domain only.
- **How we differ**: (1) Arithmetic SI conversion, not string normalization. (2) Formula matching. (3) Federated search. (4) HPC scientific domain.
- **Threat level**: HIGH — closest paper to ours. Must cite prominently and differentiate clearly.

### 4. CONE: Embeddings for Complex Numerical Data Preserving Unit and Variable Semantics
- **Authors**: Gyanendra Shrestha, Anna Pyayt, Michael Gubanov (Florida State, USF)
- **Year**: March 2026 | **Venue**: arXiv:2603.04741
- **URL**: https://arxiv.org/html/2603.04741v1
- **What it does**: Hybrid transformer encoder embedding numbers+units into vector space preserving distance. 87.28% F1 on DROP, 25% Recall@10 improvement over SOTA.
- **Limitation**: Learned approach — cannot *guarantee* cross-prefix equivalence. Depends on training data. No formula matching, no federated search.
- **How we differ**: Explicit arithmetic conversion (guaranteed correct by construction) vs learned embeddings (probabilistic). Numeracy Gap [EACL 2026] shows embeddings get 0.54 accuracy on numbers.
- **Threat level**: MODERATE — concurrent 2026 work. Must cite and differentiate on guaranteed vs learned.

### 5. NC-Retriever: Numerical Constraint-Aware Dense Retrieval
- **Authors**: Tongji-KGLLM group
- **Year**: 2025 | **Venue**: Big Data Mining and Analytics (Tsinghua)
- **URL**: https://www.sciopen.com/article/10.26599/BDMA.2025.9020047
- **What it does**: NumConQ benchmark (6,500 queries). Dense retrievers get only 16.3% on numeric constraints. Two-phase contrastive learning achieves 65.84% relative improvement.
- **Limitation**: Learns number representations in embedding space. Does NOT convert between different units of the same dimension.
- **How we differ**: Explicit unit conversion, not learned constraint satisfaction. Different approach entirely.
- **Threat level**: LOW — complementary work, different mechanism.

### 6. SciHarvester: Searching Scientific Documents for Numerical Values
- **Authors**: Maciej Rybinski et al.
- **Year**: 2023 | **Venue**: SIGIR 2023
- **URL**: https://dl.acm.org/doi/10.1145/3539618.3591808
- **What it does**: Numerical value search for agronomic literature (PubAg). Complex queries with numerical restrictions.
- **Limitation**: Domain-specific (agronomy). Different architecture (extraction + neural scoring). No explicit SI canonicalization.
- **Threat level**: LOW — domain-specific, different approach.

---

## B. Agentic Retrieval (Context for why retrieval needs to be science-aware)

### 7. Context-1: Training a Self-Editing Search Agent
- **Authors**: Chroma team
- **Year**: March 2026 | **Venue**: Technical report
- **URL**: https://www.trychroma.com/research/context-1
- **What it does**: 20B agentic search model. Iteratively searches, reads, prunes via tool calls. Hybrid BM25+dense via RRF. 88% answer recall vs 58% single-pass. 94.1% pruning accuracy. 10x faster and 25x cheaper than frontier models.
- **Why it matters for us**: Proves agentic multi-hop retrieval works. But it's domain-agnostic — knows nothing about scientific units or formulas. We add what Context-1 lacks.

### 8. Agentic RAG Survey
- **Authors**: Aditi Singh, Abul Ehtesham, Saket Kumar, Tala Talaei Khoei
- **Year**: 2025 | **Venue**: arXiv:2501.09136
- **URL**: https://arxiv.org/abs/2501.09136
- **What it does**: Comprehensive survey of agentic RAG patterns: reflection, planning, tool use, multi-agent collaboration.
- **Why it matters**: Frames the design space. All surveyed systems are domain-agnostic.

### 9. A-RAG: Hierarchical Retrieval Interfaces
- **Authors**: Mingxuan Du et al.
- **Year**: February 2026 | **Venue**: arXiv:2602.03442
- **URL**: https://arxiv.org/abs/2602.03442
- **What it does**: Exposes keyword/semantic/chunk-read interfaces to LLMs for adaptive retrieval. Outperforms fixed pipelines.
- **Why it matters**: Hierarchical interfaces parallel our capability negotiation. But no science awareness.

### 10. MA-RAG: Multi-Agent RAG
- **Authors**: Thang Nguyen, Peter Chin, Yu-Wing Tai
- **Year**: 2025 | **Venue**: arXiv:2505.20096
- **URL**: https://arxiv.org/abs/2505.20096
- **What it does**: Four agent types (Planner, Extractor, QA) for multi-hop QA. LLaMA3-8B surpasses larger models.

---

## C. Scientific RAG & HPC Systems (Our domain context)

### 11. HiPerRAG: High-Performance RAG for Scientific Insights
- **Authors**: Ozan Gokdemir et al. (25+ authors, Argonne/ORNL)
- **Year**: 2025 | **Venue**: PASC 2025, arXiv:2505.04846
- **URL**: https://arxiv.org/abs/2505.04846v1
- **What it does**: HPC-powered RAG over 3.6M scientific articles. Oreo multimodal parser, ColTrast contrastive encoder. 90% on SciQ. Scales across Polaris, Sunspot, Frontier.
- **Limitation**: Searches *papers*, not scientific *data*. No unit awareness.
- **Why it matters**: Closest HPC-scale scientific retrieval system. We complement it with science-aware operators.

### 12. OpenScholar: Synthesizing Scientific Literature with RAG
- **Authors**: Akari Asai et al.
- **Year**: February 2026 | **Venue**: Nature, Vol. 650, pp. 857-863
- **URL**: https://www.nature.com/articles/s41586-025-10072-4
- **What it does**: RAG over 45M open-access papers. OpenScholar-8B outperforms GPT-4o by 6.1%. Citation accuracy matches human experts.
- **Limitation**: Searches *papers*, not *data*. No unit or formula awareness.

### 13. Academy: Federated Agents for Scientific Workflows
- **Authors**: J. Gregory Pauloski et al.
- **Year**: 2025 | **Venue**: arXiv:2505.05428
- **URL**: https://arxiv.org/html/2505.05428v1
- **What it does**: Middleware for deploying agents across federated research infrastructure. Inter-agent coordination.
- **Why it matters**: Validates our multi-namespace federated approach for HPC.

### 14. LLM Agents for Interactive Workflow Provenance
- **Authors**: Renan Souza et al. (ORNL/Argonne)
- **Year**: 2025 | **Venue**: SC '25 Workshops
- **URL**: https://arxiv.org/abs/2509.13978v2
- **What it does**: MCP-based LLM agents for querying workflow provenance. Works across LLaMA, GPT, Gemini, Claude.
- **Why it matters**: Validates MCP for scientific computing interactivity. Published at SC.

### 15. ORNL Autonomous Experiment Agents
- **Authors**: D. Rosendo, R. F. da Silva et al.
- **Year**: 2025 | **Venue**: SC '25 Workshops
- **URL**: https://dl.acm.org/doi/10.1145/3731599.3767592
- **What it does**: Multi-agent architecture for autonomous cross-facility experiments at ORNL. LLM + programmable facility APIs + provenance.
- **Limitation**: Orchestrates experiments but cannot search existing data.

### 16. Agentic AI for Scientific Discovery Survey
- **Authors**: Large multi-institution team
- **Year**: 2025 | **Venue**: ICLR 2025
- **URL**: https://arxiv.org/abs/2503.08979
- **What it does**: Comprehensive survey covering hypothesis generation, literature review, experimental design. Identifies data management as underexplored.
- **Why it matters**: Confirms our gap — nobody is building agents that search scientific data.
