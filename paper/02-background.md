# 2. Background and Related Work

We position our work against four areas: hybrid retrieval foundations we build upon, the numeracy crisis that motivates explicit arithmetic operators, quantity-aware retrieval systems that attempt but fail to solve cross-unit matching, and scientific data discovery in HPC.

## 2.1 Hybrid Retrieval Foundations

Modern retrieval combines lexical and dense signals. BM25 \cite{robertson2009bm25} remains the strongest lexical baseline; dense retrieval via learned bi-encoders \cite{karpukhin2020dpr} captures semantic similarity that keyword matching misses. Fusing these signals is now standard practice. Bruch et al. \cite{bruch2023fusion} show that a learned convex combination consistently outperforms reciprocal rank fusion (RRF) on TREC benchmarks, establishing the analytic framework we adopt. Learned sparse methods---SPLADE \cite{formal2021splade} and its Mistral-SPLADE variant---bridge the gap by producing sparse lexical-weight vectors from dense encoders. BGE-M3 \cite{chen2024bgem3} unifies dense, sparse, and colBERT-style multi-vector retrieval in a single model trained on multilingual data, demonstrating that multi-granularity fusion improves recall across diverse query types. At the agentic frontier, A-RAG \cite{arag2026} introduces hierarchical retrieval interfaces that let LLM agents compose retrieval strategies dynamically.

We build \emph{on} hybrid retrieval, not beside it. Our science-aware operators execute as parallel branches alongside BM25 and dense search within the same fusion pipeline, adding domain-specific recall without replacing any existing component.

## 2.2 The Numerical Reasoning Crisis

Numbers are fundamentally broken in both retrieval systems and large language models. The Numeracy Gap study \cite{numeracygap2026} evaluates 13 embedding models on numerical content and reports 0.54 binary retrieval accuracy---barely above random chance. NC-Retriever \cite{ncretriever2025} finds that dense retrieval achieves only 16.3\% accuracy on queries with numeric constraints, even when the constraint is stated explicitly in both query and document. The root cause is architectural: tokenizers encode numbers per-digit rather than by magnitude, so ``200'' and ``200000'' share no meaningful representation \cite{numbers_not_continuous2025}. NumericBench \cite{numericbench2025} confirms these failures across arithmetic, comparison, and unit conversion tasks in LLMs. The Number Cookbook \cite{numbercookbook2025} systematically catalogs failure modes. GSM-Symbolic \cite{gsmsymbolic2025} delivers the sharpest result: changing only the numeric values in grade-school math problems while preserving problem structure causes a 65\% accuracy drop, proving that LLMs pattern-match on surface numerals rather than reason about magnitude.

This is not merely a unit problem. Numbers themselves---the substrate of all scientific measurement---are broken in every retrieval and language model. Scientific data retrieval, where matching requires comparing magnitudes across units, is especially vulnerable.

## 2.3 Quantity-Aware Retrieval

Three lineages attempt to make retrieval quantity-aware. None succeeds at arithmetic conversion.

**String normalization.** QFinder \cite{qfinder2022} extends Elasticsearch with quantity-aware ranking by extracting and normalizing numeric expressions. CQE \cite{cqe2023} advances this with dependency parsing over a 531-unit dictionary, handling complex quantity expressions. Numbers Matter! \cite{numbersmatter2024} builds a full retrieval pipeline with quantity-aware indexing and evaluation on FinQuant and MedQuant benchmarks. The fundamental limitation persists: string normalization cannot cross SI prefixes. ``kilopascal'' and ``pascal'' remain distinct strings after normalization, so a query for ``200~kPa'' never matches a document containing ``200000~Pa.''

**Learned embeddings.** CONE \cite{cone2026} trains a transformer to embed number--unit pairs into a shared vector space, achieving a 25\% Recall@10 gain on quantity-bearing queries. NC-Retriever \cite{ncretriever2025} applies contrastive learning to numeric constraint matching. These approaches are probabilistic: a model may learn that 200~kPa is \emph{similar} to 200000~Pa, but provides no correctness guarantee. Weller et al. \cite{embedding_limits2025} prove theoretical upper bounds on embedding-based retrieval, showing that fixed-dimensional embeddings cannot faithfully preserve all distance relationships---a result that undercuts learned approaches to exact quantity matching.

**Math-aware search.** Approach0 \cite{approach0} matches equation structure via operator trees. SSEmb \cite{ssemb2025} learns semantic similarity over mathematical expressions. MIRB \cite{mirb2025} provides the first unified math information retrieval benchmark across 12 datasets. These systems match formula \emph{structure} but are disconnected from physical measurements---they cannot link $F = ma$ to a document reporting force in newtons.

**Gap.** No existing retrieval system performs arithmetic SI conversion to canonicalize measurements. None combines quantity matching with formula normalization. None operates across federated HPC backends.

## 2.4 Scientific Data Discovery in HPC

**Paper search.** OpenScholar \cite{openscholar2026} indexes 45 million open-access papers and achieves GPT-4o-level citation quality. HiPerRAG \cite{hiperrag2025} scales RAG to 3.6 million articles across DOE supercomputers Polaris, Sunspot, and Frontier. Both search \emph{papers}---not experimental data, simulation outputs, or HDF5 files.

**Data search.** Recent systems do search scientific data. PANGAEA-GPT \cite{pangaeagpt2026} deploys hierarchical multi-agent search over 400,000+ geoscientific datasets with data-type-aware routing and unit scale validation, achieving 8.14/10 relevance versus a 2.87/10 baseline. LLM-Find \cite{llmfind2025} uses LLM-generated code to retrieve geospatial data from heterogeneous sources with 80--90\% success. The distinction is temporal: PANGAEA-GPT validates unit scales \emph{post-hoc} (after retrieval); we convert \emph{at retrieval time} (during indexing and query processing). LLM-Find generates access code per source; we provide reusable retrieval operators.

**HPC data management.** Datum \cite{datum2024} catalogs scientific data at INL but offers no content search. BRINDEXER \cite{brindexer2020} indexes POSIX metadata on Lustre with 69\% query improvement---metadata only, no content. PROV-IO+ \cite{provio2024} captures HDF5 I/O provenance with under 3.5\% overhead but exposes no search interface. HDF Clinic \cite{hdfclinic2025} proposes a knowledge graph over HDF5 but remains unimplemented.

**Agentic retrieval.** The SoK on Agentic RAG \cite{sok_agentic_rag2026} provides the first systematization of knowledge in this space. Context-1 \cite{context1_2026} demonstrates 20-billion-token agentic search. CLADD \cite{cladd2025} augments collaborative drug-discovery agents with domain knowledge graph operators. SciAgent \cite{sciagent2025} coordinates specialist worker agents under a hierarchical controller. ScienceClaw \cite{scienceclaw2026} deploys a 300+ skill agent swarm with immutable artifact provenance. All improve orchestration. None incorporates science-aware numerical operators---arithmetic dimensional conversion or formula normalization---into the retrieval pipeline itself.

\begin{table*}[t]
\centering
\caption{Capability comparison of retrieval and scientific data systems. Columns 1--5 correspond to our contributions. \ding{51} = supported, \ding{55} = not supported, $\circ$ = partial.}
\label{tab:comparison}
\small
\begin{tabular}{l|ccccc|ccccc}
\toprule
 & \rotatebox{70}{SI conversion} & \rotatebox{70}{Formula norm.} & \rotatebox{70}{Science branches} & \rotatebox{70}{Federated backends} & \rotatebox{70}{HDF5/NetCDF index} & \rotatebox{70}{Agentic multi-hop} & \rotatebox{70}{Quantity extraction} & \rotatebox{70}{Paper search} & \rotatebox{70}{Data search} & \rotatebox{70}{LLM integration} \\
\midrule
\textbf{Ours}        & \ding{51} & \ding{51} & \ding{51} & \ding{51} & \ding{51} & \ding{51} & \ding{51} & \ding{55} & \ding{51} & \ding{51} \\
PANGAEA-GPT          & \ding{55} & \ding{55} & \ding{55} & \ding{55} & \ding{55} & \ding{51} & $\circ$   & \ding{55} & \ding{51} & \ding{51} \\
OpenScholar          & \ding{55} & \ding{55} & \ding{55} & \ding{55} & \ding{55} & \ding{55} & \ding{55} & \ding{51} & \ding{55} & \ding{51} \\
HiPerRAG             & \ding{55} & \ding{55} & \ding{55} & \ding{55} & \ding{55} & \ding{55} & \ding{55} & \ding{51} & \ding{55} & \ding{51} \\
Numbers Matter!      & \ding{55} & \ding{55} & \ding{55} & \ding{55} & \ding{55} & \ding{55} & \ding{51} & \ding{55} & \ding{55} & \ding{55} \\
CONE                 & \ding{55} & \ding{55} & \ding{55} & \ding{55} & \ding{55} & \ding{55} & \ding{51} & \ding{55} & \ding{55} & \ding{55} \\
CLADD                & \ding{55} & \ding{55} & \ding{55} & \ding{55} & \ding{55} & \ding{51} & \ding{55} & \ding{55} & \ding{55} & \ding{51} \\
SciAgent             & \ding{55} & \ding{55} & \ding{55} & \ding{55} & \ding{55} & \ding{51} & \ding{55} & \ding{55} & \ding{55} & \ding{51} \\
Context-1            & \ding{55} & \ding{55} & \ding{55} & \ding{55} & \ding{55} & \ding{51} & \ding{55} & \ding{55} & \ding{55} & \ding{51} \\
\bottomrule
\end{tabular}
\end{table*}

Table~\ref{tab:comparison} summarizes the landscape. No prior system supports any of our first five capabilities: arithmetic SI conversion, formula normalization, science-aware retrieval branches, federated multi-backend search, or HDF5/NetCDF indexing. We honestly mark capabilities we lack---we do not index papers. PANGAEA-GPT receives partial credit for quantity extraction (unit scale validation, but post-hoc rather than at retrieval time).
