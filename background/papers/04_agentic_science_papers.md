# Agentic Science Papers — Agents in Scientific Workflows

Paper: **Science-Aware Hybrid Retrieval with Dimensional Conversion for HPC Data Discovery**

---

## The Narrative: Agents Are Doing Science But Retrieval Is Unit-Blind

Scientific AI agents have advanced dramatically in 2024–2026. They now run real experiments, write papers, orchestrate multi-facility workflows, and accelerate discovery by orders of magnitude. But every one of these systems hits the same wall: **they cannot search scientific data by what it measures**. This file documents the evidence that agents are doing science, why retrieval is their bottleneck, and how MCP has become the standard interface — setting up our contribution.

---

## ACT 1: Agents ARE Doing Science (The Capability Is Real)

### 1. The AI Scientist (v1)
- **Authors**: Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, David Ha
- **Year**: 2024 | **Venue**: arXiv:2408.06292
- **URL**: https://arxiv.org/abs/2408.06292
- **What it does**: End-to-end automated scientific discovery — generates hypotheses, writes code, runs experiments, writes papers. First fully automated scientific process.
- **Retrieval gap**: Operates within pre-loaded templates with fixed datasets. Cannot autonomously discover or retrieve new scientific data. "Cannot go to the library."

### 2. AI Scientist-v2
- **Authors**: Yamada et al. (Sakana AI)
- **Year**: 2025 | **Venue**: arXiv:2504.08066
- **URL**: https://arxiv.org/abs/2504.08066
- **What it does**: Experimental iteration — agent proposes experiments, runs them, revises hypotheses in a loop. Works in ML domains. Published peer-reviewed results.
- **Retrieval gap**: Cannot query external data repositories. When agents need prior experimental baselines, they rely on context window — not search.

### 3. ChemCrow
- **Authors**: Andres M. Bran, Sam Cox, Oliver Schilter, Carlo Ignazio Baldassari, Andrew D. White, Philippe Schwaller
- **Year**: 2024 | **Venue**: Nature Machine Intelligence
- **URL**: https://arxiv.org/abs/2304.05376
- **What it does**: LLM + 18 chemistry tools. Solves synthesis, safety, and prediction tasks with expert-level performance.
- **Retrieval gap**: Expert evaluation found failures from not finding the right reference data. Works when data is in a tool API; fails across heterogeneous databases.

### 4. Coscientist
- **Authors**: Daniil A. Boiko, Robert MacKnight, Ben Kline, Gabe Gomes
- **Year**: 2023 | **Venue**: Nature 624, 570–578
- **URL**: https://doi.org/10.1038/s41586-023-06792-0
- **What it does**: Autonomous chemical experiment agent. Designs and executes real-world wet-lab experiments. Demonstrated Suzuki coupling reactions.
- **Retrieval gap**: Retrieval errors propagate into real-world experimental failures. Wrong data = wrong experiments. No unit-aware retrieval exists.

### 5. Agentic AI at the Advanced Light Source
- **Authors**: Synchrotron facility team, LBNL
- **Year**: 2025 | **Venue**: arXiv:2509.17255
- **URL**: https://arxiv.org/abs/2509.17255
- **What it does**: LLM agents deployed at beamline for autonomous X-ray experiments. Reduced pre-experiment preparation time by ~2 orders of magnitude.
- **Retrieval gap**: Uses facility-specific API calls for data access. Cannot search across beamline datasets by measurement value. Each experiment is a silo.

### 6. ORNL Autonomous Cross-Facility Experiment Agents
- **Authors**: D. Rosendo, R. F. da Silva et al.
- **Year**: 2025 | **Venue**: SC '25 Workshops
- **URL**: https://dl.acm.org/doi/10.1145/3731599.3767592
- **What it does**: Multi-agent architecture for autonomous experiment orchestration across ORNL facilities. LLM + programmable facility APIs + provenance.
- **Retrieval gap**: Orchestrates experiments but cannot search existing experimental data by measured quantities.

### 7. Self-Driving Laboratories
- **Authors**: Milad Abolhasani, Eugenia Kumacheva
- **Year**: 2023 | **Venue**: Nature Chemistry
- **URL**: https://doi.org/10.1038/s41557-023-01189-8
- **What it does**: Review of autonomous laboratory systems that design, execute, and analyze experiments with minimal human intervention.
- **Retrieval gap**: Data management identified as top challenge. Each SDL is a data silo. No retrieval system connects SDLs to historical data across facilities.

---

## ACT 2: Agentic Retrieval Is Now Infrastructure (But Domain-Agnostic)

### 8. Context-1: Agentic Search at Scale
- **Authors**: Chroma Team
- **Year**: 2026 | **Venue**: Technical Report, trychroma.com/research/context-1
- **URL**: https://www.trychroma.com/research/context-1
- **What it does**: 20B parameter self-editing agentic search model trained via SFT + CISPO. Iterative search-read-prune via tool calls. 88% answer recall vs 58% single-pass. 10x faster, 25x cheaper than frontier models.
- **Gap for science**: Hybrid BM25+dense via RRF. Domain-agnostic — no unit awareness, no formula matching, no science-specific operators.

### 9. Agentic RAG: Survey
- **Authors**: Aditi Singh, Abul Ehtesham, Saket Kumar, Tala Talaei Khoei
- **Year**: 2025 | **Venue**: arXiv:2501.09136
- **URL**: https://arxiv.org/abs/2501.09136
- **What it does**: Comprehensive taxonomy of agentic RAG patterns: reflection, planning, tool use, multi-agent. Maps the design space.
- **Gap**: All surveyed systems are domain-agnostic. Survey does not address scientific quantities or units as retrieval operators.

### 10. A-RAG: Hierarchical Retrieval Interfaces
- **Authors**: Mingxuan Du et al.
- **Year**: 2026 | **Venue**: arXiv:2602.03442
- **URL**: https://arxiv.org/abs/2602.03442
- **What it does**: Exposes keyword/semantic/chunk-read interfaces to LLMs for adaptive retrieval. Outperforms fixed pipelines on standard benchmarks.
- **Gap**: Hierarchical interfaces improve retrieval flexibility but add no scientific awareness. Units and formulas are still plain text.

### 11. HiPerRAG: High-Performance RAG for Scientific Insights
- **Authors**: Ozan Gokdemir et al. (25+ authors, Argonne/ORNL)
- **Year**: 2025 | **Venue**: PASC 2025, arXiv:2505.04846
- **URL**: https://arxiv.org/abs/2505.04846
- **What it does**: HPC-powered RAG over 3.6M scientific articles. Oreo multimodal parser, ColTrast contrastive encoder. 90% on SciQ. Runs across Polaris, Sunspot, Frontier.
- **Gap**: Searches *papers*, not scientific *data*. No unit or formula awareness. Knowledge boundary is the text, not the measurements.

### 12. OpenScholar: Scientific Literature RAG
- **Authors**: Akari Asai et al.
- **Year**: 2026 | **Venue**: Nature Vol. 650, pp. 857–863
- **URL**: https://www.nature.com/articles/s41586-025-10072-4
- **What it does**: RAG over 45M open-access papers. OpenScholar-8B outperforms GPT-4o by 6.1% on citation accuracy.
- **Gap**: Searches *papers*, not *data*. GPT-4o hallucinates citations 78–90% of the time without grounding — underscoring why retrieval matters.

---

## ACT 3: MCP Has Become the Standard Science Agent Interface — Still Unit-Blind

### 13. MCP Servers for Science and HPC (CRITICAL PAPER)
- **Authors**: Science HPC community
- **Year**: 2025 | **Venue**: arXiv:2508.18489
- **URL**: https://arxiv.org/abs/2508.18489
- **What it does**: Implements MCP servers wrapping HPC resources including **Globus Search** for scientific data discovery. Direct predecessor to our work in the MCP architecture space.
- **Gap**: Globus Search is metadata-only and keyword-based — no unit awareness, no dimensional conversion, no formula matching. This paper proves MCP is the right interface; we add science-aware retrieval operators.
- **Why this matters**: This is the CLOSEST existing work to ours architecturally. Must cite and differentiate clearly.

### 14. LLM Agents for Interactive Workflow Provenance (MCP-Based)
- **Authors**: Renan Souza et al. (ORNL/Argonne)
- **Year**: 2025 | **Venue**: SC '25 Workshops
- **URL**: https://arxiv.org/abs/2509.13978
- **What it does**: MCP-based LLM agents for querying workflow provenance. Works across LLaMA, GPT, Gemini, Claude. Demonstrates MCP at production SC scale.
- **Gap**: Queries provenance (lineage), not experimental measurements. No dimensional conversion.

### 15. Academy: Federated Agents for Scientific Workflows
- **Authors**: J. Gregory Pauloski et al.
- **Year**: 2025 | **Venue**: arXiv:2505.05428
- **URL**: https://arxiv.org/html/2505.05428v1
- **What it does**: Middleware for deploying agents across federated research infrastructure. Manages inter-agent coordination, tool registration, and lifecycle.
- **Gap**: Validates federated agent architecture for HPC. But no retrieval primitives — data discovery is left to the user.

---

## ACT 4: The Bottleneck Is Now Retrieval (Evidence)

### 16. ScienceAgentBench: Best Agent Solves Only 32%
- **Authors**: Ziru Chen et al.
- **Year**: 2024 | **Venue**: NeurIPS 2024 Datasets and Benchmarks
- **URL**: https://arxiv.org/abs/2410.05080
- **What it does**: 44 tasks from 12 scientific disciplines. o1-preview achieves 32%. Failures analyzed.
- **Key finding**: Most failures stem from **incorrect data handling**, wrong file parsing, and inability to find the right data — not reasoning failures.

### 17. MLAgentBench: Agents Break at Data Preparation
- **Authors**: Qian Huang, Jian Vora, Percy Liang, Jure Leskovec
- **Year**: 2024 | **Venue**: ICML 2024
- **URL**: https://arxiv.org/abs/2310.03302
- **What it does**: Benchmark of ML agents on 15 tasks. Systematic failure analysis across task stages.
- **Key finding**: Agents break at data preparation — understanding what data exists, how it's structured, how to load it. "Mundane" data tasks break agents before reasoning.

### 18. Agentic AI for Scientific Discovery: Survey
- **Authors**: Large multi-institution team
- **Year**: 2025 | **Venue**: ICLR 2025
- **URL**: https://arxiv.org/abs/2503.08979
- **What it does**: Comprehensive survey covering hypothesis generation, literature review, experimental design, and data analysis for AI agents.
- **Key finding**: Explicitly identifies **data management as underexplored**. No existing system provides agents with science-aware data search.

### 19. FAIR Principles Are Broken for Agents
- **Year**: 2024–2025 | **Venue**: Scientific Data, Nature Computational Science
- **URL**: https://doi.org/10.1038/s41597-024-03188-x
- **What it does**: Audit of public scientific datasets against machine-actionable FAIR criteria.
- **Key finding**: <20% of datasets are machine-actionable FAIR. Scientific data was designed for human browsing — not agent retrieval.

### 20. RAG Survey: Knowledge Boundary Problem
- **Authors**: Penghao Zhao et al.
- **Year**: 2024 | **Venue**: arXiv:2402.19473
- **URL**: https://arxiv.org/abs/2402.19473
- **What it does**: Comprehensive RAG survey. Identifies retrieval quality as the primary bottleneck.
- **Key finding**: RAG is text-centric. Scientific data (HDF5, NetCDF, Parquet) requires schema-aware, unit-aware retrieval that standard RAG cannot provide.

---

## The 4-Act Story (For Paper Narrative)

```
ACT 1: AGENTS ARE DOING SCIENCE (2023–2026)
  Coscientist → real experiments   │ needs: experimental data search
  AI Scientist → full papers       │ needs: data retrieval (uses templates instead)
  ALS Beamline → 2 OOM speedup    │ needs: cross-dataset measurement search
  ORNL agents → cross-facility     │ needs: historical data discovery

ACT 2: RETRIEVAL IS THE STANDARD PIPELINE STAGE (2025–2026)
  Context-1 → 88% recall          │ agentic search works! but domain-agnostic
  HiPerRAG → 3.6M articles        │ searches papers, not data
  OpenScholar → 45M papers        │ searches papers, not data

ACT 3: MCP IS THE INTERFACE — STILL UNIT-BLIND (2025–2026)
  MCP for HPC → Globus Search     │ metadata/keyword only, no units
  ORNL MCP provenance             │ traces lineage, can't search by value
  Academy → federated agents      │ no retrieval primitives

ACT 4: THE BOTTLENECK IS RETRIEVAL (2024–2025)
  ScienceAgentBench: 32%          │ failure = data handling, not reasoning
  MLAgentBench: breaks at data    │ preparation stage is the wall
  FAIR <20% machine-actionable    │ data not designed for agents
  RAG survey: text-centric        │ can't handle HDF5/NetCDF/Parquet

  ↓
  OUR SYSTEM: Science-aware operators (dimensional conversion + formula matching + federated HPC search)
```

---

## Key Stats for Paper Motivation

| Stat | Source | Where to use |
|---|---|---|
| Best agent: 32% on science tasks | ScienceAgentBench [NeurIPS 2024] | Intro: agents fail at data |
| Failures = data handling, not reasoning | ScienceAgentBench analysis | Intro: specific failure mode |
| Data prep breaks agents | MLAgentBench [ICML 2024] | Intro/motivation |
| <20% datasets machine-actionable FAIR | FAIR audit [2024] | Motivation: infra gap |
| 78–90% citation hallucination (GPT-4o) | OpenScholar [Nature 2026] | Why grounded retrieval matters |
| MCP = production HPC interface | MCP for HPC [arXiv:2508.18489] | Our architecture validation |
| Data management = underexplored | Agentic AI Survey [ICLR 2025] | Our gap is confirmed |
