# Literature Index — UPDATED 2026-03-31

## NEW 2026 Papers Found (add to citations)

### Agentic RAG (2026)
| Key | Citation | Relevance |
|-----|----------|-----------|
| SoK Agentic RAG | Mishra et al., arXiv:2603.07379, Mar 2026 | First SoK on agentic RAG. Formalizes as POMDP. Taxonomy: planning, retrieval orchestration, memory, tools. **Must cite** — positions our work in the taxonomy |
| JADE | Chen et al., arXiv:2601.21916, Jan 2026 | Joint optimization of planning + execution in agentic RAG. Addresses strategic-operational mismatch. **Should cite** — related agentic RAG |
| A-RAG | Du et al., arXiv:2602.03442, Feb 2026 | Hierarchical retrieval interfaces (keyword/semantic/chunk). **Must cite** — already in our list |
| Context-1 | Chroma, Mar 2026 | 20B self-editing search agent, 88% recall. **Must cite** — already in list |
| HiPRAG | Wu et al., arXiv:2510.07794, Oct 2025 | RL-trained process rewards for over/under-search. **Should cite** |
| "Is Agentic RAG Worth It?" | arXiv:2601.07711, Jan 2026 | Experimental comparison of RAG approaches. **Nice to know** |

### Federated RAG (2026)
| Key | Citation | Relevance |
|-----|----------|-----------|
| FedMosaic | Liang et al., arXiv:2602.05235, Feb 2026 | First federated RAG with parametric adapters. 10.9% accuracy gain, 78-86% storage reduction. **Must cite** — concurrent federated RAG work |
| Supercharging Fed. Retrieval | Stripelis et al., arXiv:2603.25374, Mar 2026 | Secure federated RAG with Flower + confidential compute. **Should cite** |
| HyFedRAG | arXiv:2509.06444, Sep 2025 | Federated RAG for heterogeneous + privacy-sensitive data. Edge-cloud collaboration. **Should cite** |
| Federated RAG Survey | Chakraborty et al., arXiv:2505.18906, 2025 | Systematic mapping: 18 federated RAG studies. **Must cite** |
| RAGRoute | Guerraoui et al., arXiv:2502.19280, Feb 2025 | Dynamic source selection, 77.5% query reduction. **Must cite** — already listed |

### Quantity-Aware / Numerical (2025-2026)
| Key | Citation | Relevance |
|-----|----------|-----------|
| CONE | Shrestha et al., arXiv:2603.04741, Mar 2026 | Hybrid transformer for numbers+units. 87.28% F1 on DROP. **To appear at SIGMOD 2026. Must cite** |
| Numeracy Gap | Deng et al., EACL 2026 Findings | 0.54 accuracy on numerical content (binary retrieval). **Must cite** |
| Unit Harmonization (Medical) | de la Torre, arXiv:2505.00810, May 2025 | BM25+embeddings+Bayesian for clinical unit matching. MRR 0.8833. **Must cite** — closest to our unit conversion but medical domain, no SI arithmetic |
| Quantity Retrieval (Financial) | arXiv:2507.08322, 2025 | Description parsing + weak supervision for quantity retrieval. 30.98→64.66% accuracy. **Should cite** |
| BitTokens | arXiv:2510.06824, Oct 2025 | Single-token number embeddings via IEEE 754 binary. **Nice to know** |
| NumericBench | ACL 2025 Findings | Comprehensive numeracy benchmark for LLMs. **Should cite** |
| Numbers Not Continuous | arXiv:2510.08009, 2025 | Numbers encoded per-digit in base-10, not magnitude. **Should cite** |

### Multi-Agent RAG (2024-2025)
| Key | Citation | Relevance |
|-----|----------|-----------|
| Collaborative Multi-Agent RAG | arXiv:2412.05838, Dec 2024 | Agent-per-data-source (SQL, NoSQL, document). **Must cite** — closest multi-agent architecture to ours |
| HM-RAG | arXiv:2504.12330, Apr 2025 | Hierarchical multi-agent multimodal RAG. Routes to vector/graph/web agents. **Should cite** |
| MA-RAG | Nguyen et al., arXiv:2505.20096, May 2025 | Planner/Extractor/QA agents with CoT. **Should cite** |

### Scientific Computing + AI (2025-2026)
| Key | Citation | Relevance |
|-----|----------|-----------|
| ScienceAgentBench | Chen et al., **ICLR 2025** (NOT NeurIPS 2024) | 32.4% best standard agent, 42.2% o1+self-debug. **Must cite — FIX venue** |
| MCP for HPC | arXiv:2508.18489, 2025 | MCP wrapping Globus, Polaris supercomputer. **Must cite** |
| MCP Security Survey | arXiv:2503.23278, Mar 2026 | MCP landscape + security threats. **Nice to know** |
| CLADD | AAAI 2025 | Agentic RAG for drug discovery with domain KG operators. **Must cite** — weakens our "no domain operators" claim |
| SciAgent | arXiv:2511.08151, Nov 2025 | Chemistry/Physics/Math specialized worker agents. **Must cite** — weakens "no science modality agents" claim |
| Math Information Retrieval | ACM Computing Surveys, 2025 | Comprehensive survey of formula retrieval. **Should cite** |
| Advancing Math Formula Search | Springer, 2026 | Leaf-to-Leaf method for formula structure. **Should cite** |

### Formula / Equation Retrieval
| Key | Citation | Relevance |
|-----|----------|-----------|
| Mathematical Information Retrieval (Book) | arXiv:2408.11646, 2024 | Comprehensive overview: MIR search + QA. **Should cite** |
| Approach0 | Ongoing | Equation search engine. **Should cite** |
| SSEmb | CIKM 2025 | Semantic similarity for equations. **Should cite** |

---

## Previously Catalogued (from initial index)

Numbers Matter! (EMNLP 2024), QFinder (SIGIR 2022), CQE (EMNLP 2023), NC-Retriever (2025), HiPerRAG (PASC 2025), OpenScholar (Nature 2026), MLAgentBench (ICML 2024), PROV-IO+ (IEEE TPDS 2024), HDF Clinic (2025), PROV-AGENT (eScience 2025), Coscientist (Nature 2023), AI Scientist v2 (2025), ALS beamline (2025), ORNL SC'25, Academy (2025), Souza SC'25, BGE-M3 (ACL 2024), Bruch fusion (TOIS 2023), BRINDEXER (CCGrid 2020), GSM-Symbolic (ICLR 2025), Datum (INL 2024), SciHarvester (SIGIR 2023), Agentic AI Survey (ICLR 2025)

---

## Corrections Needed

1. **ScienceAgentBench venue**: ICLR 2025, NOT NeurIPS 2024
2. **ScienceAgentBench stat**: 32.4% (best standard agent), 42.2% (o1-preview + self-debug)
3. **CONE venue**: SIGMOD 2026 (not just arXiv)
4. **Numeracy Gap stat**: 0.54 is binary retrieval accuracy (near random 0.50), not general accuracy — don't overstate

## Gap Validation Status

### VALIDATED NOVELTY:
- ✅ Arithmetic SI dimensional conversion in retrieval (no prior system does this)
- ✅ Combined dimensional conversion + formula matching (no prior system)
- ✅ Science-aware operators as parallel retrieval branches alongside BM25+vector

### NARROWED CLAIMS (must reword):
- ⚠️ "No agentic RAG has domain operators" → Narrow to "no agentic RAG has science-aware *numerical* operators" (CLADD has domain KG operators)
- ⚠️ "No system searches scientific data" → Narrow to "no *agentic/LLM* system" (HDF5-FastQuery, MIQS exist as non-AI predecessors)
- ⚠️ "No multi-agent by science modality" → Narrow to "no multi-agent *retrieval* system" (SciAgent has science-modality reasoning agents)
- ⚠️ "Pluggable operators are novel" → The *concept* exists (ES plugins). The *scientific application* is novel.

### TOTAL PAPERS TO CITE: ~45-50
