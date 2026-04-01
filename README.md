# clio-search

**Science-Aware Retrieval Operators for AI-Driven HPC Data Discovery**

Research implementation and paper for **SC2026** (Data Analytics, Visualization & Storage track).

---

## The Problem

When a scientist asks an AI agent *"Find experiments where pressure was between 190 and 360 kPa"*, no retrieval system can answer reliably. The measurements exist across storage tiers in different unit representations — "200 kPa" on a filesystem, "200000 Pa" in an HDF5 attribute, "0.2 MPa" in an S3 dataset. These describe identical pressure. Every retrieval system treats them as unrelated strings. Embedding models achieve only 0.54 accuracy on numerical content (Deng et al., EACL 2026). String normalization cannot bridge SI prefixes. Learned embeddings cannot guarantee equivalence.

## Our Approach

**clio-agentic-search** introduces *science-aware retrieval operators* — deterministic, pluggable primitives that execute as first-class branches alongside standard BM25 and dense vector search:

1. **Dimensional-conversion retrieval** — Canonicalizes quantities to base SI units via arithmetic multiplication (kPa x 10^3 = Pa). Correctness guaranteed by construction across 13 units in 5 SI domains.
2. **Formula normalization** — Matches mathematical expressions across whitespace, superscript, side-swap, and factor-reordering variants.
3. **Federated multi-namespace search** — Unified retrieval across filesystem, S3, Qdrant, Neo4j, HDF5, and NetCDF backends with per-backend capability negotiation.
4. **HDF5/NetCDF metadata indexing** — Extracts measurements from scientific file formats through the same SI canonicalization pipeline.
5. **Multi-hop agentic retrieval** — LLM-driven query rewriting (expand/narrow/pivot) with iterative search refinement, converging in 2-3 hops.

---

## Motivating Example

**Query**: *"Find experiments where pressure was between 190 and 360 kPa"*

| Document | Value | Backend | Standard Retrieval | clio-agentic-search |
|---|---|---|---|---|
| Doc A | 250 kPa | Filesystem | Found (string match) | Found (250000 Pa in range) |
| Doc B | 200000 Pa | S3 | Missed | Found (200000 Pa in range) |
| Doc C | 0.3 MPa | HDF5 | Missed | Found (300000 Pa in range) |

clio-agentic-search converts all values to canonical Pa, range-checks against 190000-360000, and finds **all three** across all backends.

---

## Repository Structure

```
clio-search/
├── paper/                          # SC2026 paper (LaTeX + PDF)
│   ├── paper.tex                   # Full paper (11 pages, IEEE CS format)
│   ├── paper.pdf                   # Compiled PDF
│   ├── references.bib              # 44 BibTeX entries
│   └── figures/
│       └── architecture.tex        # TikZ architecture diagram (Fig. 1)
├── code/                           # clio-agentic-search implementation
│   ├── src/clio_agentic_search/
│   │   ├── api/                    # FastAPI server (/query, /documents, /jobs, /health)
│   │   ├── cli/                    # CLI (clio query/index/serve/list/seed)
│   │   ├── connectors/
│   │   │   ├── filesystem/         # Full — lexical, vector, scientific
│   │   │   ├── object_store/       # Full — S3-compatible (MinIO, AWS)
│   │   │   ├── hdf5/              # Full — h5py, attribute extraction, SI canonicalization
│   │   │   ├── netcdf/            # Full — xarray, CF conventions, SI canonicalization
│   │   │   ├── vector_store/      # Stub — in-memory Qdrant protocol
│   │   │   └── graph_store/       # Stub — in-memory Neo4j protocol
│   │   ├── retrieval/
│   │   │   ├── coordinator.py     # 4-branch hybrid retrieval pipeline
│   │   │   ├── agentic.py         # Multi-hop iterative retrieval loop
│   │   │   ├── query_rewriter.py  # LLM + fallback query rewriting
│   │   │   ├── scientific.py      # Science-aware scoring operators
│   │   │   └── capabilities.py    # Protocol-based capability negotiation
│   │   ├── indexing/
│   │   │   └── scientific.py      # SI conversion (13 units), formula normalization
│   │   └── storage/
│   │       └── duckdb_store.py    # 8-table schema, SQL-native BM25
│   └── tests/                     # 169 automated tests
├── .planning/                     # WTF-P paper planning infrastructure
│   ├── PROJECT.md                 # Project brief and requirements
│   ├── ROADMAP.md                 # 6 sections, 11 writing plans, 3 waves
│   ├── STATE.md                   # Current writing state
│   ├── GAP-VALIDATION.md          # Independent novelty audit (55 papers)
│   └── sources/literature.md      # Verified literature index
└── background/                    # Research notes and paper analyses
```

---

## Quick Start

```bash
cd code
uv sync --all-extras --dev
```

### Query

```bash
# Dimensional conversion (cross-unit range query)
uv run clio query --q "pressure between 190 and 360 kPa" --numeric-range "190:360:kPa"

# Formula query
uv run clio query --q "F = ma" --formula "F=ma"

# Multi-hop agentic retrieval with LLM rewriting
uv run clio query --q "high pressure turbulence simulations" --agentic --max-hops 3

# Agentic with LLM query rewriting (requires ANTHROPIC_API_KEY)
uv run clio query --q "find velocity data above 100 km/h" --agentic --llm-rewrite

# Multi-namespace federated search
uv run clio query --namespaces "local_fs,object_s3,hdf5_data" --q "temperature measurements"
```

### Index and Serve

```bash
uv run clio index --namespace local_fs          # Index local filesystem
uv run clio index --namespace hdf5_data         # Index HDF5 files
uv run clio index --namespace netcdf_data       # Index NetCDF files
uv run clio serve                               # Start FastAPI at localhost:8000
```

### API Endpoints

```
POST /query          — Hybrid retrieval with science-aware operators
GET  /documents      — List indexed documents
POST /jobs/index     — Async indexing job
GET  /health         — Liveness probe
GET  /metrics        — Prometheus metrics
```

### Tests

```bash
uv run pytest tests/ -v                         # All 169 tests
uv run pytest tests/unit/ -v                    # Unit tests only
uv run pytest tests/scientific/ -v              # Scientific retrieval tests
```

---

## Paper Status

| Milestone | Date | Status |
|---|---|---|
| Abstract submission | April 1, 2026 | Submitted |
| Full paper draft | April 1, 2026 | Complete (eval [TBD]) |
| Full paper submission | April 8, 2026 | In progress |
| Venue | SC2026 — Data Analytics, Visualization & Storage | — |

**Paper**: 11 pages, IEEE CS double-column, 44 references, architecture figure. Evaluation results pending benchmark execution.

---

## Capability Comparison

| Capability | Numbers Matter! | CONE | PANGAEA-GPT | HiPerRAG | OpenScholar | **Ours** |
|---|---|---|---|---|---|---|
| SI dimensional conversion | - | - | - | - | - | **Yes** |
| Formula normalization | - | - | - | - | - | **Yes** |
| Science-aware retrieval branches | - | - | - | - | - | **Yes** |
| Federated multi-backend | - | - | - | - | - | **Yes** |
| HDF5/NetCDF indexing | - | - | - | - | - | **Yes** |
| Agentic multi-hop | - | - | Yes | - | - | **Yes** |
| Quantity extraction | Yes | Yes | Partial | - | - | **Yes** |
| Data search (not papers) | - | - | Yes | - | - | **Yes** |
| Paper search | - | - | - | Yes | Yes | - |

---

## Key Citations

- **Numeracy Gap** (Deng et al., EACL 2026): 0.54 embedding accuracy on numerical content
- **ScienceAgentBench** (Chen et al., ICLR 2025): Best agent achieves 32.4% on scientific tasks
- **Numbers Matter!** (Almasian et al., EMNLP 2024): Quantity-aware retrieval via string normalization
- **CONE** (Shrestha et al., 2026): Learned embeddings for numbers+units
- **PANGAEA-GPT** (2026): Multi-agent geoscientific data search with post-hoc unit validation
- **Theoretical Limits of Embedding Retrieval** (Weller et al., 2025): Formal bounds on embedding fidelity

Full literature index: [`.planning/sources/literature.md`](.planning/sources/literature.md) (55 papers verified)

---

## Part of CLIO Kit

This work is part of [**CLIO Kit**](https://github.com/iowarp/clio-kit), the IoWarp platform's tooling layer for AI agents in scientific computing.
