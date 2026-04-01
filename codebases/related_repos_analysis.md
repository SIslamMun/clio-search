# Related GitHub Repositories Analysis
## For: "Science-Aware Hybrid Retrieval with Dimensional Conversion for HPC Data Discovery"
## Target Venue: SC2026
## Analysis Date: 2026-03-31

---

## 1. Quantity-Aware Retrieval Systems

---

### 1.1 QuantityAwareRankers — Almasian, Bruseva, Gertz (Heidelberg)

| Field | Details |
|-------|---------|
| **GitHub URL** | https://github.com/satya77/QuantityAwareRankers |
| **Stars** | ~9 |
| **Last Commit** | January 8, 2025 (active) |
| **Paper** | "Numbers Matter! Bringing Quantity-awareness to Retrieval Systems" — arXiv:2407.10283 (July 2024) |
| **Threat Level** | **HIGH** — closest prior art in the quantity-aware retrieval space |

**Key Files/Modules:**
- `data_generation/` — fine-tuning data via concept expansion, unit conversion, value permutation
- `dataset/` — data loaders for query collections
- `models/` — BM25, SPLADE, ColBERT baseline variants + quantity-aware extensions (QBM25, QSPLADE, QColBERT)
- `evaluate/` — pytrec_eval benchmarking with significance testing
- `requirements.txt`

**Technical Approach:**
Introduces two quantity-aware ranking strategies (joint and disjoint) over QBM25, QSPLADE, and QColBERT. The data generation pipeline uses unit conversion and value permutation to produce fine-tuning signal. Benchmarks: **FinQuant** (finance) and **MedQuant** (medicine).

**How Our Work Differs:**
- Their domain is finance/medicine NLP text. Our domain is **HPC scientific datasets** (HDF5, ADIOS2, NetCDF files with dimensional metadata).
- Their unit conversion is a data-augmentation heuristic during training; ours is a **runtime SI dimensional conversion** applied at query time before retrieval (kPa × 1000 = Pa arithmetic on live measurement attributes).
- They do not address **federated backends** (S3, Qdrant, Neo4j, filesystem) or HPC storage at all.
- Our retrieval targets **structured scientific metadata** with typed measurement attributes, not free-text documents.
- No formula normalization in their pipeline.

---

### 1.2 CQE (Comprehensive Quantity Extractor) — Almasian et al. (Heidelberg)

| Field | Details |
|-------|---------|
| **GitHub URL** | https://github.com/vivkaz/CQE |
| **Stars** | ~21 |
| **Last Commit** | January 22, 2026 (actively maintained) |
| **Paper** | "CQE: A Comprehensive Quantity Extractor" — arXiv:2305.08853 (EMNLP 2023) |
| **Threat Level** | **MEDIUM** — relevant for quantity extraction but not retrieval |

**Key Files/Modules:**
- `CQE.py` — main extraction orchestration
- `NumberNormalizer.py` — bounds, numbers, and unit normalization
- `rules.py` — 50+ spaCy DependencyMatcher rules for linguistic quantity detection
- `classes.py` — Bound, Range, Number, Unit, Noun, Quantity data structures
- `unit.json` — 531 normalized units with surfaces, symbols, prefixes, entity, URI, dimensions
- `unit_classifier/` — BERT-based unit disambiguator (spacy-transformers)

**Technical Approach:**
Extracts structured quantity tuples from natural language text using dependency parsing (spaCy). Identifies operator, value, unit symbols, normalized unit, and contextual nouns. First system to detect concepts associated with quantities. Normalizes 531 unit surface forms.

**How Our Work Differs:**
- CQE extracts quantities from **prose text**; we extract quantities from **structured HPC dataset metadata** (HDF5 attributes, ADIOS2 variable metadata, NetCDF coordinate variables).
- Our dimensional conversion is an **SI arithmetic pipeline** for retrieval query rewriting, not an NLP extraction task.
- We integrate quantity normalization directly into the **hybrid retrieval scoring** (BM25 + dense + science-aware), not as a standalone extraction stage.
- CQE has no retrieval component.

---

### 1.3 satya77/CQE — Secondary Repository

| Field | Details |
|-------|---------|
| **GitHub URL** | https://github.com/satya77/CQE |
| **Stars** | 0 |
| **Last Commit** | February 20, 2024 (inactive fork/personal copy) |
| **Threat Level** | **LOW** — same work as vivkaz/CQE above, older copy |

---

### 1.4 NumConQ — Tongji-KGLLM (Tongji University, 2025)

| Field | Details |
|-------|---------|
| **GitHub URL** | https://github.com/Tongji-KGLLM/NumConQ |
| **Stars** | 0 |
| **Last Commit** | July 5, 2025 (active as of 2025) |
| **Paper** | Not yet identified on arXiv; associated with suzhou-22/NumConQ (2025-02-07) |
| **Threat Level** | **MEDIUM** — numerical quantity retrieval focus, partially overlapping |

**Key Files/Modules:**
- `train/ft.sh` — embedding model fine-tuning
- `evaluation/eval.sh` — evaluation pipeline
- `contriever/` — Contriever-based dense retrieval
- `contriever_hybrid/` — hybrid retrieval variant
- `plot_code/` — result visualization

**Technical Approach:**
Trains and evaluates embedding models with numerical quantity awareness. Uses Contriever as the dense retrieval backbone. Includes a hybrid retrieval variant. The exact quantity-handling strategy (whether it performs unit conversion or numerical range matching) is not fully documented in the public README.

**How Our Work Differs:**
- NumConQ targets general NLP quantity retrieval; our focus is **HPC scientific data backends** (HDF5/ADIOS2/NetCDF physical measurements).
- We perform explicit **SI dimensional arithmetic** (kPa × 1000 = Pa) at query time; NumConQ appears to rely on learned representations.
- Our federated multi-backend architecture (filesystem, S3, Qdrant, Neo4j) is absent in NumConQ.
- NumConQ has essentially no adoption (0 stars, minimal documentation).

**Note:** A second repo exists at https://github.com/suzhou-22/NumConQ (also 0 stars, last commit 2025-02-07), likely an earlier personal copy.

---

### 1.5 QFinder — Almasian, Gertz (Heidelberg) — SIGIR 2022

| Field | Details |
|-------|---------|
| **GitHub URL** | **NOT FOUND** — no public repository identified |
| **Stars** | N/A |
| **Last Commit** | N/A |
| **Paper** | SIGIR 2022 workshop or short paper on quantity-aware document retrieval (not on arXiv) |
| **Threat Level** | **MEDIUM** — foundational work by same Heidelberg group, but apparently no public code |

**Note:** The SIGIR 2022 QFinder work appears to be a predecessor to the QuantityAwareRankers (arXiv:2407.10283) paper above. No public GitHub repository was found. The arXiv record for Almasian S. does not include a 2022 paper, suggesting it was a workshop paper or published only through ACM DL. Cite the 2024 QuantityAwareRankers paper as the peer-reviewed equivalent with public code.

---

## 2. Unit and Dimensional Analysis Libraries

---

### 2.1 pint — hgrecco

| Field | Details |
|-------|---------|
| **GitHub URL** | https://github.com/hgrecco/pint |
| **Stars** | ~2,708 |
| **Last Commit** | March 31, 2026 (very actively maintained) |
| **Paper** | No associated paper; production library |
| **Threat Level** | **LOW** — library we may use internally, not a competing system |

**Key Files/Modules:**
- `pint/` — core package with unit registry, quantity arithmetic, dimensional analysis
- `pint/unit.py` — unit definitions
- `pint/registry.py` — extensible unit registry
- `pint/pint-default.txt` — comprehensive unit definition file (prefixes, base units, derived units)
- `docs/` — extensive documentation

**Technical Approach:**
Operates on base units and SI prefixes, building compound units compositionally. Supports full dimensional analysis with `Quantity` objects, NumPy arrays, and uncertainty-aware computations. Recognizes prefixed/pluralized forms without explicit definition. Unit conversions are exact where possible (1 kPa = 1000 Pa by construction).

**How Our Work Builds on This:**
Pint can serve as the **dimensional conversion backend** in our SI arithmetic pipeline. Rather than implementing unit conversion from scratch, our system can leverage pint's registry for `kPa → Pa`, `°C → K`, `mph → m/s` conversions at query time. This is an **adoption relationship**, not a competition.

---

### 2.2 QUDT (Quantities, Units, Dimensions, Types) Ontology

| Field | Details |
|-------|---------|
| **GitHub URL** | https://github.com/qudtlib/qudtlib-java (Java; 19 stars) / https://github.com/egonw/jqudt (Java; 16 stars) |
| **Stars** | 19 / 16 respectively |
| **Last Commit** | March 10, 2026 / March 8, 2026 (active) |
| **Paper** | No single paper; W3C community standard |
| **Threat Level** | **LOW** — ontology standard, not a retrieval system |

**Key Files/Modules (qudtlib-java):**
- `qudtlib-main-rdf/` — RDF/OWL ontology files for 1,700+ units
- `qudtlib-java/` — Java API for unit conversion using QUDT ontology
- Unit coverage: dimensions, conversion multipliers, unit symbols, labels, Wikipedia links

**Technical Approach:**
RDF/OWL ontology defining units with their dimensional vectors and conversion factors. Python wrappers exist (e.g., `linkml-qudt`, `rqudt`) but no authoritative Python QUDT library matches pint's adoption level. QUDT provides semantic URI-based unit identification (e.g., `qudt:Kilopascal`).

**How Our Work Relates:**
QUDT's dimensional vector representation could inform our formula normalization for unit equivalences. For practical runtime conversion, pint is more suitable. We should cite QUDT as the semantic standard our unit taxonomy is compatible with.

---

### 2.3 uom (Units of Measure) — iliekturtles (Rust)

| Field | Details |
|-------|---------|
| **GitHub URL** | https://github.com/iliekturtles/uom |
| **Stars** | ~1,215 |
| **Last Commit** | March 30, 2026 (active) |
| **Paper** | No associated paper |
| **Threat Level** | **LOW** — Rust library, different ecosystem |

**Technical Approach:**
Type-safe zero-cost dimensional analysis in Rust. Units are encoded in the type system, making unit mismatches compile-time errors. Supports SI system with full prefix support.

**How Our Work Relates:**
Not directly applicable to our Python-based pipeline. Relevant as an example of the type-theoretic approach to dimensional safety. Our work takes a runtime arithmetic approach (suitable for Python/query-time conversion) rather than compile-time type checking.

---

## 3. Math and Formula Search

---

### 3.1 Approach0 Math-Aware Search Engine

| Field | Details |
|-------|---------|
| **GitHub URL** | https://github.com/approach0/search-engine |
| **Stars** | ~355 |
| **Last Commit** | March 25, 2026 (active) |
| **Paper** | Multiple SIGIR/CIKM papers on math-aware search (2016–2023) |
| **Threat Level** | **MEDIUM** — formula retrieval prior art, but different domain |

**Key Files/Modules:**
- Core engine in C (80.9% of codebase)
- `math-index/` — mathematical expression indexing
- `tex-parser/` — LaTeX formula parser
- `a0-crawlers/` — web crawlers for formula-containing documents
- Online demo at approach0.xyz; most updated code is **closed source**

**Technical Approach:**
Indexes mathematical formulas from Q&A sites (Math Stack Exchange, AOPS) using operator tree representations. Supports mixed keyword + formula queries. Matches structurally similar formulas by tree edit distance on operator graphs. The master branch is inactive; the research branch contains an early model supporting math-only queries.

**How Our Work Differs:**
- Approach0 targets **mathematical formula retrieval in Q&A text**; we normalize physical/scientific formulas (e.g., `P = nRT/V` vs. `PV = nRT`) in **HPC dataset metadata**.
- Our formula normalization is algebraic rewriting for equivalence detection (e.g., velocity `v = Δx/Δt` matched to `v = (x2-x1)/Δt`), not tree-edit similarity for LaTeX documents.
- We do not index formula-heavy documents; we normalize formula strings in dataset descriptions/attributes.
- Approach0 has no SI unit conversion or HPC backend integration.

---

### 3.2 SSEmb — Ruyin Li, Xiaoyu Chen

| Field | Details |
|-------|---------|
| **GitHub URL** | **NOT FOUND** — no public repository identified |
| **Stars** | N/A |
| **Last Commit** | N/A |
| **Paper** | "SSEmb: A Joint Structural and Semantic Embedding Framework for Mathematical Formula Retrieval" — arXiv:2508.04162 (August 2025) |
| **Threat Level** | **LOW** — formula embedding for math Q&A, no public code, different domain |

**Technical Approach:**
Combines Graph Contrastive Learning (structural encoding of operator graphs) with Sentence-BERT (semantic encoding) for mathematical formula retrieval. Evaluated on ARQMath-3. Achieves >5 pp improvement over embedding-based baselines. Targets math Q&A retrieval, not scientific data discovery.

**How Our Work Differs:**
SSEmb embeds LaTeX formula structure for Q&A search. Our formula normalization is algebraic simplification for equivalence checking in dataset metadata (a much simpler, domain-specific problem). SSEmb has no code, no SI unit conversion, and no HPC backend.

---

### 3.3 mathsteps — Google (socraticorg)

| Field | Details |
|-------|---------|
| **GitHub URL** | https://github.com/google/mathsteps |
| **Stars** | ~2,154 |
| **Last Commit** | March 20, 2026 (maintained under google/ org) |
| **Paper** | No associated paper (educational tool) |
| **Threat Level** | **LOW** — algebraic simplification library, not a retrieval system |

**Key Files/Modules:**
- `lib/` — core library for algebraic transformations
- `index.js` — public API: `simplifyExpression()`, `solveEquation()`
- Each step returns `{from, to, changeType, substeps}` objects

**Technical Approach:**
Node.js library providing step-by-step algebraic simplification and equation solving. Applies sequential transformation rules (combining like terms, simplifying fractions, etc.). Returns intermediate transformation steps with change type labels.

**How Our Work Builds on This:**
mathsteps provides the kind of algebraic normalization we need for formula matching (e.g., rewriting `P*V = n*R*T` to canonical form). However, it is JavaScript-only and oriented toward educational use. Our formula normalization uses similar step-by-step algebraic rewriting but implemented in Python and targeted at scientific notation variants in dataset descriptions. We can **cite mathsteps** as a related algebraic normalization approach.

---

## 4. Scientific RAG and Hybrid Retrieval Systems

---

### 4.1 SPLADE — NAVER Labs Europe

| Field | Details |
|-------|---------|
| **GitHub URL** | https://github.com/naver/splade |
| **Stars** | ~985 |
| **Last Commit** | March 30, 2026 (active) |
| **Paper** | "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking" — SIGIR 2021, SIGIR 2022 |
| **Threat Level** | **MEDIUM** — learned sparse retrieval component we may use/compare against |

**Key Files/Modules:**
- `train.py` — SPLADE model training
- `index.py` — inverted index construction from SPLADE representations
- `retrieve.py` — retrieval over SPLADE index
- `all.py` — end-to-end pipeline
- `conf/` — Hydra configuration for training and retrieval
- `inference_splade.ipynb` — inference demonstration

**Technical Approach:**
Learns sparse query/document representations via BERT MLM head with L1 regularization (FLOPS regularization). Generates interpretable bag-of-words-style representations that work with standard inverted indexes. Achieves near-dense retrieval effectiveness while maintaining BM25-like efficiency. SPLADE++ (SIGIR 2022) adds ensemble distillation and improved regularization.

**How Our Work Builds on This:**
SPLADE is a potential component of our learned sparse retrieval stage. Our key addition is the **science-aware scoring layer** on top of standard SPLADE/BM25 that performs SI dimensional matching on measurement attributes. SPLADE alone cannot handle `kPa` matching `Pa` ranges without our conversion layer.

---

### 4.2 ColBERT — Stanford Future Data Systems

| Field | Details |
|-------|---------|
| **GitHub URL** | https://github.com/stanford-futuredata/ColBERT |
| **Stars** | ~3,817 |
| **Last Commit** | March 30, 2026 (very active) |
| **Paper** | ColBERT (SIGIR'20), ColBERTv2 (NAACL'22), PLAID (CIKM'22) |
| **Threat Level** | **LOW-MEDIUM** — dense retrieval component we compare against |

**Key Files/Modules:**
- `colbert/` — main package
  - `colbert/indexer.py` — passage preprocessing and FAISS index construction
  - `colbert/searcher.py` — multi-vector MaxSim retrieval
  - `colbert/trainer.py` — fine-tuning ColBERT models
- `baleen/` — multi-hop QA extensions
- Pre-trained ColBERTv2 checkpoint on MS MARCO available

**Technical Approach:**
Late-interaction retrieval: encodes each passage into a matrix of token-level embeddings (not a single vector). At query time, uses MaxSim operator (cosine similarity for each query token against all passage tokens, take max, then sum) for rich semantic matching. PLAID adds efficient approximate MaxSim with centroid-based compression.

**How Our Work Differs:**
ColBERT is a general-purpose dense retrieval model trained on web text (MS MARCO). It has no concept of SI units, dimensional conversion, or scientific measurement semantics. Our hybrid pipeline uses a ColBERT-class dense encoder as one component, but the **science-aware dimensional scoring** is our novel contribution layered on top.

---

### 4.3 BGE-M3 / FlagEmbedding — BAAI / FlagOpen

| Field | Details |
|-------|---------|
| **GitHub URL** | https://github.com/FlagOpen/FlagEmbedding |
| **Stars** | ~11,480 |
| **Last Commit** | March 31, 2026 (very active) |
| **Paper** | "M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation" — arXiv:2402.03216 (SIGIR 2024) |
| **Threat Level** | **LOW** — foundational embedding model we use as a component |

**Key Files/Modules:**
- `FlagEmbedding/` — main package
  - `inference/` — Embedder and Reranker inference modules
  - `finetune/` — fine-tuning scripts for embedder and reranker
  - `evaluation/` — MTEB-style evaluation
- Models on HuggingFace: `BAAI/bge-m3`

**Technical Approach:**
BGE-M3 unifies three retrieval modalities in one model: (1) dense retrieval (single-vector), (2) lexical sparse retrieval (MLM head, similar to SPLADE), (3) multi-vector retrieval (ColBERT-style late interaction). Supports 100+ languages and up to 8,192 token inputs. Training uses self-knowledge distillation across the three modalities.

**How Our Work Builds on This:**
BGE-M3 is the state-of-the-art embedding backbone for our dense retrieval stage. Our contribution is the **science-aware scoring layer** that post-processes BGE-M3 similarity scores with SI dimensional matching for measurement queries. BGE-M3 alone cannot match "190 kPa" against records labeled "190,000 Pa" without our conversion layer.

---

### 4.4 HiPerRAG — Argonne National Laboratory / ALCF

| Field | Details |
|-------|---------|
| **GitHub URL** | **NOT PUBLICLY AVAILABLE** — no open-source release found |
| **Stars** | N/A |
| **Last Commit** | N/A |
| **Paper** | "HiPerRAG: High-Performance Retrieval Augmented Generation for Scientific Insights" — arXiv:2505.04846 (May 2025) |
| **Threat Level** | **MEDIUM** — scientific RAG at scale, different focus from ours |

**Technical Approach:**
Two components: **Oreo** (multimodal document parser for scientific PDFs) and **ColTrast** (ColBERT fine-tuning via contrastive learning). Scales across supercomputers (Polaris, Sunspot, Frontier) to index 3.6M+ scientific articles. Achieves 90% on SciQ, 76% on PubMedQA. Targets **scientific literature** (PDFs, journal papers), not **HPC datasets** (HDF5, ADIOS2, etc.).

**How Our Work Differs:**
- HiPerRAG retrieves from scientific **papers** (prose text); we retrieve from scientific **datasets** (structured metadata with typed measurements).
- HiPerRAG has no SI dimensional conversion or formula normalization.
- Our federated backend (HDF5, S3, Qdrant, Neo4j) vs. their centralized index on HPC filesystems.
- Code is not open-source; our system is designed to be the open-source equivalent for HPC data discovery.

---

## 5. HPC Data Discovery Systems

---

### 5.1 BRINDEXER — (CCGrid 2020)

| Field | Details |
|-------|---------|
| **GitHub URL** | **NOT FOUND** — no public repository identified after exhaustive search |
| **Stars** | N/A |
| **Last Commit** | N/A |
| **Paper** | Appeared at CCGrid 2020; paper not on arXiv; not found in GitHub search |
| **Threat Level** | **LOW** — appears to be internal/proprietary, no code to compare against |

**Note:** BRINDEXER (possibly "Burst-buffer/Reactive Indexer" or similar acronym) from CCGrid 2020 does not appear to have a public GitHub repository. The system likely indexes HPC simulation output for discovery, but without code access it cannot be directly compared. We cite it as non-reproducible prior art.

---

### 5.2 Globus Search / Globus SDK — Globus

| Field | Details |
|-------|---------|
| **GitHub URLs** | https://github.com/globus/globus-sdk-python (74 stars) / https://github.com/globus/globus-search-cli (2 stars) / https://github.com/globus/django-globus-portal-framework (15 stars) |
| **Stars** | 74 / 2 / 15 respectively |
| **Last Commit** | March 30, 2026 / December 4, 2024 / March 2, 2026 |
| **Threat Level** | **LOW** — infrastructure/middleware, not a retrieval algorithm |

**Key Files/Modules (globus-sdk-python):**
- `src/globus_sdk/` — Python SDK for all Globus services
- `src/globus_sdk/services/search/` — Globus Search service client
  - GlobusSearchClient for index creation, ingest, and query
- Supports boolean, range, and text queries over federated datasets

**Technical Approach:**
Globus Search provides managed search-as-a-service over scientific datasets registered in Globus indexes. Uses Elasticsearch under the hood. The `globus-search-cli` provides a CLI wrapper. `django-globus-portal-framework` enables portal UIs that aggregate Globus Search across endpoints.

**How Our Work Relates:**
Globus Search is a **federated backend** that our system could integrate alongside Qdrant, Neo4j, and filesystem indexers. It handles administrative metadata (dataset location, access controls, provenance) while we add **science-aware measurement retrieval** on top of or alongside Globus Search indexes. Our dimensional conversion and formula matching are not present in Globus Search, which relies on standard Elasticsearch keyword/range queries without SI normalization.

---

### 5.3 MCP Servers for Science and HPC — Argonne/ANL

| Field | Details |
|-------|---------|
| **GitHub URL** | **NOT PUBLICLY FOUND** — no open repository in the paper or on GitHub |
| **Stars** | N/A |
| **Last Commit** | N/A |
| **Paper** | "Experiences with Model Context Protocol Servers for Science and High Performance Computing" — arXiv:2508.18489 (August 2025); 15 authors including Ryan Chard, Ian Foster et al. (Argonne) |
| **Threat Level** | **MEDIUM** — adjacent work on MCP+HPC, but not a retrieval system |

**Technical Approach:**
Implements MCP (Model Context Protocol) servers as thin wrappers over existing services: Globus Transfer, Globus Compute, Globus Search, facility status APIs, Octopus event fabric, Garden, and Galaxy. Demonstrates agent-driven scientific workflows in chemistry, bioinformatics, quantum chemistry, and filesystem monitoring.

**How Our Work Relates:**
This paper establishes MCP as a viable interface for HPC scientific services (same architecture as CLIO Kit). Our clio-agentic-search component provides the **data discovery and retrieval capabilities** that MCP servers for science would need but do not implement: science-aware hybrid retrieval with dimensional conversion. We are building the discovery layer that MCP servers like Globus Search wrap, but with science-aware measurement semantics.

---

## 6. Summary Threat Matrix

| Repository | Threat Level | Key Overlap | Our Differentiator |
|------------|-------------|-------------|-------------------|
| satya77/QuantityAwareRankers | **HIGH** | Quantity-aware BM25/SPLADE/ColBERT with unit-aware training | Runtime SI arithmetic on HPC metadata; federated backends; structured measurements not NLP text |
| vivkaz/CQE | **MEDIUM** | Unit normalization (531 units), quantity extraction from text | We target structured HPC metadata, not prose; integrated into retrieval pipeline not standalone extraction |
| Tongji-KGLLM/NumConQ | **MEDIUM** | Numerical quantity-aware embedding training | 0 adoption; no SI conversion; no HPC backends; not science-data focused |
| approach0/search-engine | **MEDIUM** | Formula/math-aware retrieval, operator tree matching | Our formula normalization is algebraic rewriting on metadata strings, not full MathML document indexing |
| naver/splade | **MEDIUM** | Learned sparse retrieval component | SPLADE is a building block; our contribution is the science-aware layer on top |
| stanford-futuredata/ColBERT | **LOW-MED** | Dense multi-vector retrieval component | ColBERT is a building block with no dimensional conversion or HPC specificity |
| FlagOpen/FlagEmbedding (BGE-M3) | **LOW** | Dense embedding backbone | BGE-M3 is our embedding component; we add dimensional conversion above it |
| HiPerRAG (arXiv:2505.04846) | **MEDIUM** | Scientific RAG at HPC scale | Targets scientific papers not datasets; no unit conversion; closed source |
| hgrecco/pint | **LOW** | SI unit conversion library | We use pint internally; it is a dependency, not a competing system |
| QUDT Ontology | **LOW** | Unit/dimension ontology | Semantic standard; we are compatible with QUDT vocabulary |
| iliekturtles/uom | **LOW** | Type-safe dimensional analysis (Rust) | Different language/paradigm; not competing |
| SSEmb (arXiv:2508.04162) | **LOW** | Structural+semantic formula embedding | No code; targets math Q&A; different problem domain |
| google/mathsteps | **LOW** | Step-by-step algebraic simplification | Educational JS library; not a retrieval system; potential inspiration for formula normalization |
| globus/globus-sdk-python | **LOW** | Federated scientific data search infrastructure | Infrastructure layer; lacks science-aware measurement retrieval semantics |
| MCP for HPC (arXiv:2508.18489) | **MEDIUM** | MCP+HPC architecture, Globus Search | Not a retrieval algorithm; our system is the missing discovery layer they wrap |
| BRINDEXER (CCGrid 2020) | **LOW** | HPC data indexing | No public code; no retrieval algorithm; not reproducible |
| QFinder (SIGIR 2022) | **MEDIUM** | Early Heidelberg quantity retrieval | No public code; superseded by QuantityAwareRankers which is HIGH threat |

---

## 7. Research Gaps This Paper Fills

Based on the repository landscape above, no existing open-source system simultaneously addresses:

1. **SI dimensional conversion at retrieval time** (kPa ↔ Pa, °C ↔ K arithmetic) — QuantityAwareRankers uses unit conversion in training data augmentation only, not at query time.
2. **Formula normalization for scientific notation variants** — Approach0 handles mathematical Q&A formulas; no system handles physical law variant matching in dataset metadata.
3. **Federated retrieval across heterogeneous HPC storage backends** — filesystem, S3, Qdrant (vector), Neo4j (graph) in a single retrieval pipeline.
4. **BM25 + dense + science-aware three-stage hybrid pipeline** specifically designed for HPC dataset discovery, not general web or paper retrieval.
5. **Structured scientific measurement metadata** as the retrieval corpus (HDF5/ADIOS2/NetCDF typed attributes) rather than free-text documents.

The combination of (1) + (2) + (3) + (4) on target domain (5) is the novel contribution of this paper with no direct prior art.

---

## 8. Recommended Citations

```bibtex
@article{almasian2024numbers,
  title={{Numbers Matter! Bringing Quantity-awareness to Retrieval Systems}},
  author={Almasian, Satya and Bruseva, Milena and Gertz, Michael},
  journal={arXiv:2407.10283},
  year={2024},
  url={https://github.com/satya77/QuantityAwareRankers}
}

@inproceedings{almasian2023cqe,
  title={{CQE: A Comprehensive Quantity Extractor}},
  author={Almasian, Satya and Kazakova, Vivian and G{\"o}ldner, Philip and Gertz, Michael},
  booktitle={EMNLP},
  year={2023},
  url={https://github.com/vivkaz/CQE}
}

@inproceedings{formal2021splade,
  title={{SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking}},
  author={Formal, Thibault and Piwowarski, Benjamin and Clinchant, Stephane},
  booktitle={SIGIR},
  year={2021},
  url={https://github.com/naver/splade}
}

@inproceedings{santhanam2022colbertv2,
  title={{ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction}},
  author={Santhanam, Keshav and Khattab, Omar and Saad-Falcon, Jon and Potts, Christopher and Zaharia, Matei},
  booktitle={NAACL},
  year={2022},
  url={https://github.com/stanford-futuredata/ColBERT}
}

@inproceedings{chen2024bge,
  title={{M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation}},
  author={Chen, Jianlv and Xiao, Shitao and Zhang, Peitian and Luo, Kun and Lian, Defu and Liu, Zheng},
  booktitle={SIGIR},
  year={2024},
  url={https://github.com/FlagOpen/FlagEmbedding}
}

@misc{pan2025mcp,
  title={{Experiences with Model Context Protocol Servers for Science and High Performance Computing}},
  author={Pan, Haochen and Chard, Ryan and Mello, Reid and others},
  journal={arXiv:2508.18489},
  year={2025}
}

@misc{hiperrag2025,
  title={{HiPerRAG: High-Performance Retrieval Augmented Generation for Scientific Insights}},
  journal={arXiv:2505.04846},
  year={2025}
}
```

---

*Analysis prepared for SC2026 submission. All star counts and last-commit dates as of 2026-03-31.*
