# The Agentic Gap: Key Evidence Table

Paper: **Science-Aware Hybrid Retrieval with Dimensional Conversion for HPC Data Discovery**

> For full paper details and the 4-act narrative, see `04_agentic_science_papers.md`.

---

## Core Thesis

AI agents are entering science at an unprecedented pace. They can design molecules, run experiments, write papers, and reason about complex hypotheses. But they share a critical weakness: **they cannot find the scientific data they need.** When retrieval fails, agents hallucinate. The bottleneck has shifted from "can AI reason scientifically?" to "can AI find and access the right data?"

---

## Evidence: Agents Fail at Data, Not Reasoning

| Paper | Venue | Key Finding |
|---|---|---|
| ScienceAgentBench [Chen et al.] | NeurIPS 2024 | Best agent (o1-preview): **32%** success. Most failures = data handling, not reasoning |
| MLAgentBench [Huang et al.] | ICML 2024 | Agents break at **data preparation** — before reasoning begins |
| AI Scientist [Lu et al.] | arXiv:2408.06292 | Sidesteps retrieval entirely; operates within pre-loaded templates |
| ChemCrow [Bran et al.] | Nature Machine Intel. | Expert evaluation: failures from not finding right data, not reasoning |
| Coscientist [Boiko et al.] | Nature 624 | Retrieval errors propagate to real-world experimental failures |
| Self-Driving Labs [Abolhasani] | Nature Chemistry | Data management = top challenge; each lab is a data silo |
| Agentic AI Survey | ICLR 2025 | Data management explicitly identified as underexplored capability |

---

## Evidence: Scientific Data Infrastructure Is Not Agent-Ready

| Paper | Venue | Key Finding |
|---|---|---|
| FAIR Audit [2024–2025] | Scientific Data | **<20%** of public datasets are machine-actionable FAIR |
| LLM Agent Survey [Wang et al.] | Frontiers CS 2024 | "Insufficient knowledge acquisition" = top failure mode |
| RAG Survey [Zhao et al.] | arXiv:2402.19473 | RAG is text-centric; cannot handle HDF5/NetCDF/Parquet |
| MCP for HPC [2025] | arXiv:2508.18489 | MCP wraps Globus Search (metadata/keyword only, no unit awareness) |

---

## The Pattern

```
YEAR    SYSTEM                      RETRIEVAL STATUS
2023    Coscientist (experiments)   → Retrieval errors cause experiment failures
2024    AI Scientist (papers)       → Sidesteps retrieval with pre-loaded templates
2024    ChemCrow (18 tools)         → Fails when data isn't in a tool API
2024    SciAgentBench (32%)         → Most failures = data handling, not reasoning
2025    MCP for HPC                 → Metadata/keyword only, no unit awareness
2025    HiPerRAG (3.6M articles)    → Searches papers, not data
2026    OpenScholar (45M papers)    → Searches papers, not data
2026    Context-1 (88% recall)      → Domain-agnostic, no unit awareness
2026    OUR SYSTEM                  → Searches DATA with dimensional conversion
```

---

## Key Stats for the Paper

| Stat | Source | Use |
|---|---|---|
| Best agent: 32% on scientific tasks | ScienceAgentBench | Motivation: agents fail at data |
| <20% datasets are machine-actionable FAIR | FAIR [2024] | Infrastructure gap |
| 20–40% hallucination on measurements | Multiple [2024] | Why retrieval matters |
| 78–90% citation hallucination (GPT-4o) | OpenScholar [Nature 2026] | Grounding necessity |
| 16.3% accuracy on numeric constraints | NC-Retriever [2025] | Numbers are broken in retrieval |
| 0.54 embedding accuracy on numbers | Numeracy Gap [EACL 2026] | Embeddings can't do units |
| 65% accuracy drop on number perturbation | GSM-Symbolic [ICLR 2025] | LLMs can't do math reliably |
