# 1. Introduction

Scientific computing generates data at unprecedented scale. A single experimental campaign at a DOE facility may produce terabytes of simulation outputs in HDF5, observational records on parallel filesystems, derived features in object stores, and annotations in knowledge graphs. When a scientist later asks, "which experiments measured pressures between 190 and 360 kPa?"---a question that should be trivial---no existing retrieval system can answer it reliably. The measurements exist, scattered across storage tiers in different unit representations: "200 kPa" on one filesystem, "200000 Pa" in an HDF5 attribute, "0.2 MPa" in an S3-hosted dataset. These describe identical pressure. Every retrieval system treats them as unrelated strings.

This data discovery crisis is intensifying as AI agents enter scientific workflows. Coscientist \cite{coscientist2023} autonomously plans and executes wet-lab synthesis. AI Scientist v2 \cite{aiscientistv2_2025} generates and writes up research end-to-end. Autonomous agents at ORNL orchestrate experiments across national facilities \cite{ornl_agents_sc25}, and the Model Context Protocol now wraps Globus search, compute, and storage under a unified agent interface for HPC \cite{mcp_hpc2025}. The trajectory is clear: agents are transitioning from processing scientific literature to operating scientific infrastructure. Yet these agents consistently fail at data retrieval, not reasoning. ScienceAgentBench \cite{scienceagentbench2025}, a benchmark of 44 tasks drawn from peer-reviewed publications, reports that the best agent achieves only 32.4\% end-to-end success, with failures traced to data handling rather than reasoning. MLAgentBench \cite{mlagentbench2024} observes the same pattern. A comprehensive survey of agentic AI \cite{agentic_ai_survey2025} identifies data management as explicitly underexplored. These agents can reason about science---they cannot find the data they need to reason about.

The root cause is structural: retrieval systems treat scientific content as plain text. Three compounding failures explain why.

\textbf{Dimensional blindness.} Scientific measurements are routinely expressed in different unit prefixes. String-based quantity normalization \cite{numbersmatter2024} maps unit names to canonical strings, but "kilopascal" and "pascal" remain different strings---no string operation bridges the SI prefix gap. Learned embeddings such as CONE \cite{cone2026} encode numbers and units in a shared vector space, but the Numeracy Gap study \cite{numeracygap2026} evaluates 13 embedding models and finds 0.54 binary retrieval accuracy on numerical content---effectively random. NC-Retriever \cite{ncretriever2025} reports 16.3\% dense retrieval accuracy on numeric constraint queries. The fundamental problem is that none of these approaches performs the required arithmetic: $200 \times 1000 = 200{,}000$.

\textbf{Formula opacity.} "$F=ma$," "$F = m \cdot a$," and "$ma=F$" encode the same physical law but produce distinct token sequences. Math-aware search systems \cite{approach0, ssemb2025} match equation structure in LaTeX markup but cannot connect a matched formula to the physical measurements it governs. No retrieval system combines formula normalization with quantity-aware dimensional search.

\textbf{Storage fragmentation.} HPC data spans parallel filesystems, S3 object stores, vector databases, and binary formats such as HDF5 and NetCDF. No system provides unified search with science-aware operators across these backends. RAGRoute \cite{ragroute2025} federates generic text search. PROV-IO+ \cite{provio2024} captures HDF5 provenance with under 3.5\% overhead but exposes no search interface.

These failures compound. The motivating query---"find experiments where pressure was between 190--360 kPa across all storage"---requires dimensional conversion across SI prefixes, federated search across heterogeneous backends, and potentially formula context linking pressure to related variables. No existing system addresses even two of these three requirements simultaneously.

We hypothesize that retrieval systems can overcome these failures through \emph{science-aware operators}: deterministic, pluggable retrieval primitives that encode domain knowledge as explicit arithmetic and symbolic transformations, executing as first-class branches alongside standard BM25 and dense vector search. Unlike string normalization or learned embeddings, these operators make correctness verifiable by inspection---if the SI conversion factor is correct, the match is correct. We test this hypothesis through five contributions:

\begin{enumerate}
\item \textbf{Dimensional-conversion retrieval.} We canonicalize physical quantities to base SI units via arithmetic multiplication (e.g., kPa $\times$ $10^3$ = Pa) across 13 units in 5 domains, with guaranteed correctness by construction.

\item \textbf{Formula normalization.} We normalize mathematical expressions across whitespace, superscript, side-swap, and factor-reordering variants, integrated with dimensional operators as parallel retrieval branches.

\item \textbf{Federated multi-namespace search.} A connector architecture supports unified retrieval across filesystem, S3, Qdrant, Neo4j, HDF5, and NetCDF backends with per-backend capability negotiation and incremental indexing.

\item \textbf{HDF5/NetCDF metadata indexing.} Dedicated connectors extract measurements from scientific file formats and feed them through the same SI canonicalization pipeline, enabling cross-format dimensional search.

\item \textbf{Multi-hop agentic retrieval.} An LLM-driven query rewriting loop iteratively refines searches using expand, narrow, and pivot strategies, converging within 2--3 hops.
\end{enumerate}

The remainder of this paper is organized as follows. Section 2 surveys hybrid retrieval, the numeracy crisis, quantity-aware retrieval, and scientific data discovery. Section 3 presents the system design. Section 4 describes the implementation. Section 5 evaluates across dimensional conversion, formula matching, federated search, and ablation. Section 6 concludes.
