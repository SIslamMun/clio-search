# Literature Index — VERIFIED 2026-04-01

**Method**: All citations web-searched for existence. Missed papers found via topic searches.

---

## CRITICAL ADDITIONS (missed by previous session)

### HIGH THREAT — Must cite and differentiate

| Key | Citation | Why Critical |
|-----|----------|-------------|
| **PANGAEA-GPT** | arXiv:2602.21351, Feb 2026 | Hierarchical multi-agent system searching 400K+ geoscientific datasets in PANGAEA archive. Data-type-aware routing, specialist agents, sandboxed code execution. Achieves 8.14/10 vs baseline 2.87/10. **Includes unit scale validation.** Directly challenges "nobody searches scientific data" claim. Must differentiate: they validate post-hoc, we convert at retrieval time. |
| **ScienceClaw/Infinite** | Buehler et al. (MIT), arXiv:2603.14312, Mar 2026 | 300+ skill agent swarm for scientific discovery. Immutable artifact DAG with provenance. Plannerless coordination via ArtifactReactor. Real results: peptide binders, ceramics design. Strengthens motivation (agents do science but lack measurement-aware retrieval). |

### MODERATE — Should cite

| Key | Citation | Why |
|-----|----------|-----|
| **MIRB** | Ju & Dong, arXiv:2505.15585, May 2025 | First unified Math Information Retrieval Benchmark. 4 tasks, 12 datasets, 13 models evaluated. Provides evaluation context for formula matching. |
| **Theoretical Limits of Embedding Retrieval** | Weller et al., arXiv:2508.21038, Aug 2025 (rev. Mar 2026) | Proves theoretical upper bounds on embedding-based retrieval. **Strengthens our argument** for explicit arithmetic operators over learned embeddings. |
| **LLM-Find** | Ning et al., arXiv:2407.21024, IJDE 2025 | LLM agent retrieving actual geospatial data from heterogeneous sources via code generation. 80-90% success. Another counter to "nobody searches data." |
| **Accelerating Earth Science Discovery** | Pyayt et al., arXiv:2503.05854, Frontiers in AI 2025 | Perspective on multi-agent LLMs for geosciences. Validates scientific data discovery gap. |
| **FRAG** | Zhao, arXiv:2410.13272, Oct 2024 | Federated vector DB management with encrypted ANN search. Privacy-focused federated retrieval. |
| **DAI/JULEA** | Duwe & Kuhn, Springer ICCS 2024 | Pre-computation for HDF5 queries, 22,000x speedup. Non-AI predecessor for HDF5 search. |
| **HPC-FAIR** | OSTI 2504172, 2025 | FAIR framework for HPC data/model management. Background context. |

---

## Previously Indexed — Verified Real

### Agentic RAG (2026)
| Key | Citation | Verified | Relevance |
|-----|----------|----------|-----------|
| SoK Agentic RAG | Mishra et al., arXiv:2603.07379, Mar 2026 | ✅ Real | First SoK on agentic RAG. **Must cite** |
| JADE | Chen et al., arXiv:2601.21916, Jan 2026 | ✅ Real (full name: "Joint Agentic Dynamic Execution") | **Should cite** |
| A-RAG | Du et al., arXiv:2602.03442, Feb 2026 | ✅ Real | Hierarchical retrieval interfaces. **Must cite** |
| Context-1 | Chroma, Mar 2026 | ✅ Real but **"88% recall" UNVERIFIABLE** — public page shows different metrics (0.94 prune accuracy) | Cite with caution. Drop 88% stat or qualify. |
| HiPRAG | Wu et al., arXiv:2510.07794, Oct 2025 | ✅ Real | **Should cite** |

### Federated RAG (2025-2026)
| Key | Citation | Verified | Relevance |
|-----|----------|----------|-----------|
| FedMosaic | Liang et al., arXiv:2602.05235, Feb 2026 | ✅ Real | **Must cite** |
| Supercharging Fed. Intelligence Retrieval | Stripelis et al., arXiv:2603.25374, Mar 2026 | ✅ Real (title was wrong — "Intelligence" was missing) | **Should cite** |
| HyFedRAG | arXiv:2509.06444, Sep 2025 | Not independently verified | **Should cite** |
| Federated RAG Survey | Chakraborty et al., arXiv:2505.18906, 2025 | Not independently verified | **Must cite** |
| RAGRoute | Guerraoui et al., arXiv:2502.19280, Feb 2025 | Not independently verified | **Must cite** |

### Quantity-Aware / Numerical (2025-2026)
| Key | Citation | Verified | Relevance |
|-----|----------|----------|-----------|
| CONE | Shrestha et al., arXiv:2603.04741, Mar 2026 | ✅ Real. **SIGMOD 2026 venue UNCONFIRMED** | **Must cite** |
| Numeracy Gap | Deng et al., arXiv:2509.05691, EACL 2026 Findings | ✅ Real. 0.54 accuracy confirmed. | **Must cite** |
| Unit Harmonization | de la Torre, arXiv:2505.00810, May 2025 | ✅ Real. Full title: "Scalable Unit Harmonization in Medical Informatics..." | **Must cite** — closest to our work but medical domain, no SI arithmetic |
| Numbers Matter! | Almasian et al., arXiv:2407.10283, EMNLP 2024 Findings | ✅ Real | **Must cite** |

### Scientific Computing + AI (2025-2026)
| Key | Citation | Verified | Relevance |
|-----|----------|----------|-----------|
| ScienceAgentBench | Chen et al., arXiv:2410.05080, **ICLR 2025** | ✅ Real. **Venue: ICLR 2025 (NOT NeurIPS 2024)**. Best: 34.3% with expert+self-debug, 32.4% without expert. | **Must cite — FIX VENUE everywhere** |
| MCP for HPC | arXiv:2508.18489, 2025 | ✅ Real. Full title: "Experiences with Model Context Protocol Servers for Science and HPC" | **Must cite** |
| CLADD | arXiv:2502.17506, AAAI 2025 | ✅ Real. Full title: "RAG-Enhanced Collaborative LLM Agents for Drug Discovery" | **Must cite** |
| SciAgent | arXiv:2511.08151, Nov 2025 | ✅ Real. Hierarchical multi-agent with Coordinator + specialized Worker Systems | **Must cite** |
| OpenScholar | Nature Vol 650, pp 857-863, Feb 2026 | ✅ Real. 45M open-access papers confirmed. | **Must cite** |
| HiPerRAG | arXiv:2505.04846, PASC 2025 | ✅ Real | **Must cite** |
| Coscientist | Boiko et al., Nature 624, 2023 | ✅ Real | **Must cite** |

---

## Corrections Log

| Item | Was | Should Be | Status |
|------|-----|-----------|--------|
| ScienceAgentBench venue | NeurIPS 2024 | **ICLR 2025** | ❌ Still wrong in outline.md |
| ScienceAgentBench best stat | 32% | **34.3%** (with expert+self-debug), **32.4%** (without expert) | Fix |
| Context-1 recall | 88% | **UNVERIFIABLE** — drop or qualify as "reported" | Fix |
| Supercharging title | "Supercharging Federated Retrieval" | "Supercharging Federated **Intelligence** Retrieval" | Fix |
| CONE venue | SIGMOD 2026 | **Unconfirmed** — cite as arXiv only | Fix |
| "Nobody searches data" | Broad claim | **False** — PANGAEA-GPT, LLM-Find search data | Must rewrite |

---

## Gap Validation Status (Updated)

### VALIDATED NOVELTY (verified via web search):
- ✅ Arithmetic SI dimensional conversion in retrieval (no prior system)
- ✅ Combined dimensional conversion + formula matching (no prior system)
- ✅ Science-aware operators as parallel retrieval branches alongside BM25+vector (no prior system)

### NARROWED CLAIMS (must reword):
- ⚠️ "No agentic RAG has domain operators" → CLADD, PANGAEA-GPT have domain operators → Narrow to "no system has *numerical measurement* operators"
- ⚠️ "No system searches scientific data" → PANGAEA-GPT and LLM-Find DO → Narrow to "no system searches data with dimensional conversion"
- ⚠️ "No multi-agent by science modality" → SciAgent, PANGAEA-GPT have specialist agents → Narrow to "no *retrieval* system has measurement-modality operators"
- ⚠️ "Agentic retrieval pipeline" → NOT IMPLEMENTED IN CODE → Either implement by April 8 or reframe paper

### TOTAL VERIFIED PAPERS: ~55
