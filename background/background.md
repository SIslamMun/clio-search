# 2. Background and Motivation

## v0.5 — SC2026: Science-Aware Hybrid Retrieval with Dimensional Conversion for HPC Data Discovery

---

### 2.1 Hybrid Retrieval: Lexical + Dense

Modern information retrieval combines two complementary modalities. **Lexical retrieval** (BM25) matches exact terms through inverted indices, excelling at keyword precision but missing semantic paraphrases. **Dense retrieval** encodes queries and documents into embedding spaces (e.g., all-MiniLM-L6-v2, BGE-M3 [Chen et al., 2024]) where cosine similarity captures semantic meaning.

**Hybrid retrieval** fuses both through score combination. Bruch et al. [2023] showed that convex combination consistently outperforms Reciprocal Rank Fusion (RRF). Learned sparse retrieval models like SPLADE and Mistral-SPLADE [Doshi et al., 2024] blur the boundary using LLMs for semantic term expansion. A-RAG [Du et al., 2026] exposes hierarchical retrieval interfaces directly to language models for adaptive multi-granularity search.

These advances establish hybrid retrieval as the dominant paradigm — but all treat document content as domain-agnostic text. Numbers, units, and formulas receive no special handling.

### 2.2 The Numerical Reasoning Crisis

A growing body of evidence demonstrates that retrieval systems fundamentally fail on numerical content:

- **Dense retrieval achieves only 16.3% accuracy** on numerical constraint queries. The NumConQ benchmark [NC-Retriever, 2025] across 6,500 queries shows that even state-of-the-art dense retrievers cannot handle "temperature above 300K" or "price between 500–1000."
- **Embedding models achieve 0.54 accuracy** on numerical content — barely above random [Deng et al., EACL 2026]. "2% growth" and "20% growth" are nearly indistinguishable in embedding space.
- **Numbers are embedded non-continuously** [arXiv 2510.08009, 2025]. While linear reconstruction achieves R² ≥ 0.95, principal components explain only a minor share of variation. Numbers are encoded per-digit in base-10, not as continuous magnitudes.
- **LLMs drop 65% accuracy** when numerical values change in otherwise identical problems [GSM-Symbolic, Apple, ICLR 2025], suggesting pattern matching rather than reasoning.
- **LLMs treat numbers as discrete tokens, not continuous magnitudes** [NumericBench, ACL 2025; Number Cookbook, ICLR 2025].

This crisis is not just about units — numbers themselves are broken in every retrieval system.

### 2.3 Quantity-Aware Retrieval (Closest Prior Art)

Three approaches attempt to address numerical retrieval, none sufficient for scientific computing:

**Approach 1: String Normalization.** QFinder [Almasian and Gertz, SIGIR 2022] introduced quantity-centric ranking as an Elasticsearch plugin. CQE [Almasian et al., EMNLP 2023] provides extraction with 531 units and dependency parsing. "Numbers Matter!" [Almasian et al., EMNLP 2024] combines extraction with ranking by textual similarity + numerical proximity on FinQuant and MedQuant benchmarks. These systems normalize unit *strings* — "km/h" → "kilometer per hour" — but cannot match across SI prefixes. "200 kPa" and "200000 Pa" remain unmatched because "kilopascal" ≠ "pascal."

**Approach 2: Learned Embeddings.** CONE [Shrestha et al., March 2026] encodes numbers with units into a vector space preserving distance, achieving 25% Recall@10 improvement. NC-Retriever [2025] learns numerical constraint satisfaction through contrastive training. However, these approaches cannot *guarantee* cross-prefix equivalence — their accuracy depends on training data coverage, and the Numeracy Gap study confirms 0.54 accuracy on numbers.

**Approach 3: Math-Aware Search.** Approach0, SSEmb [CIKM 2025], and MIRB [2025] match mathematical equation *structure* in LaTeX. These handle formula retrieval but not physical measurement matching — finding "∫f(x)dx" is different from finding "pressure was 200 kPa."

None of these approaches performs arithmetic conversion (200 × 1000 = 200000), combines quantity matching with formula retrieval, or operates across federated storage backends.

### 2.4 AI Agents Are Entering Science — And Hitting the Retrieval Wall

Scientific AI agents have advanced rapidly in 2023–2026. Autonomous systems now execute real wet-lab experiments [Coscientist, Nature 2023], write full research papers [AI Scientist, 2024; AI Scientist-v2, 2025], orchestrate experiments across national facilities [ORNL SC'25], and reduce beamline preparation time by two orders of magnitude [ALS, arXiv:2509.17255]. Self-driving laboratories [Abolhasani & Kumacheva, Nature Chemistry 2023] are proliferating across chemistry, materials, and biology.

Yet agents consistently fail at data retrieval. The ScienceAgentBench [Chen et al., NeurIPS 2024] evaluated 44 tasks across 12 disciplines — the best agent (o1-preview) achieved only 32% success. Failure analysis reveals the cause: agents failed at **data handling**, not reasoning. MLAgentBench [Huang et al., ICML 2024] confirms the pattern — agents break during data preparation, before any reasoning begins. A comprehensive survey of agentic AI for scientific discovery [ICLR 2025] explicitly identifies data management as an underexplored capability.

**MCP has emerged as the standard interface** for connecting AI agents to scientific infrastructure. MCP Servers for Science and HPC [arXiv:2508.18489] wrap HPC resources including Globus Search under the MCP protocol. Souza et al. [SC'25] deploy MCP-based agents for workflow provenance at ORNL. These systems validate MCP as the right interface — but they expose **keyword-based and metadata-only search**. Globus Search has no concept of dimensional units, and MCP servers for science cannot answer "find experiments where pressure exceeded 200 kPa" by searching measurement content.

**Agentic retrieval systems improve orchestration, not understanding.** Context-1 [Chroma, 2026] — a 20B parameter model trained via SFT and CISPO — achieves 88% answer recall versus 58% for single-pass, matching frontier models at 10x speed and 25x lower cost. MA-RAG [Nguyen et al., 2025] orchestrates four agent types for multi-hop QA. HiPRAG [Wu et al., 2025] addresses over-search and under-search through hierarchical process rewards. All improve *how* retrieval is orchestrated without changing *what* retrieval understands — they remain blind to units, formulas, and dimensional relationships.

The bottleneck has shifted. The question is no longer whether AI agents can reason scientifically — they can. The question is whether they can **find the data they need**. Our system addresses this gap directly.

### 2.5 Scientific Data Discovery in HPC

HPC environments present unique retrieval challenges. Data is generated at hundreds of gigabits per second, distributed across parallel filesystems, object stores, vector databases, and graph databases.

**Current systems search papers, not data.** OpenScholar [Nature, 2026] retrieves from 45M open-access papers with citation accuracy matching human experts. HiPerRAG [Gokdemir et al., PASC 2025] indexes 3.6M articles across Polaris, Sunspot, and Frontier supercomputers. But neither searches the actual scientific *data* — the HDF5 files, simulation outputs, and experimental measurements.

**HPC data discovery remains metadata-only.** Datum [INL, 2024] catalogs scientific data with minimal infrastructure but provides no content search inside files. BRINDEXER [CCGrid 2020] indexes POSIX metadata (filenames, sizes, timestamps) on Lustre, improving indexing by 69% — but has no understanding of what is *inside* the files. The HDF Group envisions knowledge graphs over HDF5 [HDF Clinic, 2025], but no implementation exists.

**Provenance is captured but not searched.** PROV-IO+ [IEEE TPDS, 2024] tracks HDF5 I/O provenance with < 3.5% overhead. PROV-AGENT [eScience, 2025] extends W3C PROV for AI agent interactions with MCP support. But provenance data is not used as a retrieval signal — you can trace how data was created but cannot search for "experiments with similar parameters."

**Federated search lacks science-awareness.** RAGRoute [Guerraoui et al., EuroMLSys 2025] dynamically selects data sources, reducing queries by 77.5%. A systematic mapping study [Chakraborty et al., 2025] identifies 18 federated RAG studies. But these federate generic text search — no system federates science-aware search across HPC storage tiers.

**AI agents orchestrate experiments but cannot search data.** MCP Servers for Science and HPC [arXiv:2508.18489] wrap Globus Search under MCP — the closest architectural predecessor to our work — but Globus Search is keyword-only with no unit awareness. ORNL deploys autonomous agents for cross-facility experiments [SC'25], and LLM agents query workflow provenance via MCP [Souza et al., SC'25]. The Agentic AI for Scientific Discovery survey [ICLR 2025] confirms data management is underexplored. No agent searches petabyte-scale scientific data by measured quantity on HPC systems.

### 2.6 The Gap

Table 1 summarizes the landscape across the three failure dimensions:

| Capability | Numbers Matter! [2024] | CONE [2026] | Context-1 [2026] | HiPerRAG [2025] | OpenScholar [2026] | Datum [INL] | **Ours** |
|---|---|---|---|---|---|---|---|
| **Dimensional conversion** (kPa×10³=Pa) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| **Formula matching** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| **Cross-unit range queries** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| **Federated multi-backend** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| **Searches scientific data** | ✗ | ✗ | ✗ | ✗ | ✗ | metadata only | **✓** |
| Unit string normalization | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Unit-aware embeddings | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| BM25 + vector hybrid | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ |
| Agentic multi-hop | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | planned |
| Searches papers | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ |

The first five rows — the bolded capabilities — are empty for every existing system. This is the gap we fill.

### 2.7 Motivating Example

Consider a scientist using an AI agent to search experimental logs across an HPC center's heterogeneous storage:

**Query**: "Find experiments where pressure was between 190 and 360 kPa"

The corpus contains three relevant documents across three storage backends:
- Document A: "The measured pressure was **250 kPa** at steady state" (parallel filesystem)
- Document B: "Chamber pressure reached **200000 Pa** before failure" (S3 archive)
- Document C: "Peak pressure: **0.3 MPa** during phase transition" (Qdrant vector store)

**Standard hybrid retrieval**: BM25 matches Document A on "kPa" token overlap. Vector search weakly associates "pressure" but cannot do dimensional comparison. **Result: Only Document A found.**

**String normalization** [Numbers Matter!]: Normalizes "kPa" → "kilopascal" and "Pa" → "pascal." Different strings. **Result: Only Document A found.**

**Learned embeddings** [CONE]: Might match if training data covered the kPa-Pa relationship. Might not. **Result: Uncertain. 0.54 accuracy on numbers.**

**clio-agentic-search**: Converts all measurements to base SI via multiplication:
- 250 kPa × 1000 = 250000 Pa ✓
- 200000 Pa × 1 = 200000 Pa ✓
- 0.3 MPa × 1000000 = 300000 Pa ✓
- Query range: 190000–360000 Pa

All three match. Federated search queries all backends simultaneously. **Result: All three documents found.**

---

## References

[Abolhasani & Kumacheva, 2023] Self-Driving Laboratories. Nature Chemistry. doi:10.1038/s41557-023-01189-8
[AI Scientist-v2, 2025] Yamada et al. (Sakana AI). arXiv:2504.08066
[ALS Agents, 2025] Agentic AI at Advanced Light Source. arXiv:2509.17255
[Almasian and Gertz, 2022] QFinder. SIGIR 2022.
[Almasian et al., 2023] CQE. EMNLP 2023. arXiv:2305.08853
[Almasian et al., 2024] Numbers Matter! EMNLP 2024 Findings. arXiv:2407.10283
[Bruch et al., 2023] Analysis of Fusion Functions. ACM TOIS 42(1). arXiv:2210.11934
[Chakraborty et al., 2025] Federated RAG Survey. arXiv:2505.18906
[Chen et al., 2024] BGE M3-Embedding. ACL 2024 Findings. arXiv:2402.03216
[Chroma, 2026] Context-1. trychroma.com/research/context-1
[Deng et al., 2026] Numeracy Gap. EACL 2026. arXiv:2509.05691
[Doshi et al., 2024] Mistral-SPLADE. arXiv:2408.11119
[Du et al., 2026] A-RAG. arXiv:2602.03442
[Gokdemir et al., 2025] HiPerRAG. PASC 2025. arXiv:2505.04846
[GSM-Symbolic, 2025] Apple. ICLR 2025. arXiv:2410.05229
[Guerraoui et al., 2025] RAGRoute. EuroMLSys 2025. arXiv:2502.19280
[INL, 2024] Datum. inlsoftware.inl.gov/product/datum
[NC-Retriever, 2025] NumConQ. SciOpen.
[Nguyen et al., 2025] MA-RAG. arXiv:2505.20096
[Numbers Not Continuous, 2025] arXiv:2510.08009
[NumericBench, 2025] ACL 2025. aclanthology.org/2025.findings-acl.1026
[Number Cookbook, 2025] ICLR 2025. openreview.net/forum?id=BWS5gVjgeY
[OpenScholar, 2026] Nature 650. nature.com/articles/s41586-025-10072-4
[ORNL Agents, 2025] SC'25 Workshops. dl.acm.org/doi/10.1145/3731599.3767592
[Paul et al., 2020] BRINDEXER. CCGrid 2020. IEEE 9139660
[Pauloski et al., 2025] Academy. arXiv:2505.05428
[PROV-AGENT, 2025] eScience 2025. arXiv:2508.02866
[PROV-IO+, 2024] IEEE TPDS. IEEE 10472875
[Shin et al., 2025] Agentic AI for Science. arXiv:2509.09915
[Shrestha et al., 2026] CONE. arXiv:2603.04741
[Singh et al., 2025] Agentic RAG Survey. arXiv:2501.09136
[Souza et al., 2025] LLM Workflow Provenance. SC'25. arXiv:2509.13978
[Wu et al., 2025] HiPRAG. arXiv:2510.07794
[Agentic AI Survey, 2025] ICLR 2025. arXiv:2503.08979
[HDF Clinic, 2025] github.com/HDFGroup/hdf-clinic
[Huang et al., 2024] MLAgentBench. ICML 2024. arXiv:2310.03302
[Lu et al., 2024] The AI Scientist. arXiv:2408.06292
[MCP for HPC, 2025] MCP Servers for Science and HPC. arXiv:2508.18489
