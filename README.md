# CLIO Search

**Agentic Search for Scientific Data**

SC2026 submission — research implementation and paper.

---

## The Problem

Scientific data discovery requires search that thinks before it acts. A search system must profile each corpus to understand what metadata exist, decide which strategies apply, execute domain-specific reasoning (such as recognizing that 200 kPa = 200000 Pa = 0.2 MPa), and refine when results are insufficient. This observe-decide-act-evaluate cycle is what distinguishes agentic search from conventional retrieval. No current system performs these operations in combination.

## CLIO Search

CLIO implements a four-stage agentic pipeline:

1. **Discover** — Profile each corpus to learn what data and metadata exist (14–22 ms)
2. **Reason** — Select viable retrieval branches based on corpus characteristics
3. **Transform** — Execute dimensional conversion and formula normalization as deterministic database operators alongside BM25 and dense vector search
4. **Execute** — Federate across storage backends with iterative refinement (LLM rewriter or deterministic fallback achieving 96% of LLM quality)

### Key Results

| Metric | Value |
|--------|-------|
| Cross-unit P@5 | **0.99** (BM25: 0.40, NC-Retriever: 0.16) |
| Same-unit controls | 0.30–0.45 (all methods comparable) |
| Real-world NOAA | **+87%** precision over hybrid baselines |
| Distributed scale | 2.5M arXiv, 4 GH200 nodes, sub-50 ms, 0.94 weak scaling |
| IOWarp integration | 0.5% corpus inspection (native API: 100%) |
| Agentic efficiency | 52% token reduction, 91% time reduction vs native agent |

---

## Repository Structure

```
clio-search/
├── paper/                          # SC2026 paper
│   ├── paper.tex                   # Main driver
│   ├── paper.pdf                   # Compiled PDF (11 pages)
│   ├── sections/                   # Per-section LaTeX files
│   │   ├── abstract.tex
│   │   ├── introduction.tex
│   │   ├── background.tex
│   │   ├── design.tex
│   │   ├── implementation.tex
│   │   ├── evaluation.tex
│   │   └── conclusion.tex
│   ├── figures/                    # Architecture diagram + plots
│   └── references.bib             # 52 BibTeX entries
│
├── code/                           # Implementation (13.1K lines, 283 tests)
│   ├── src/clio_agentic_search/
│   │   ├── connectors/             # 9 storage backends
│   │   │   ├── filesystem/         # Full — lexical, vector, scientific, profile
│   │   │   ├── object_store/       # Full — S3-compatible
│   │   │   ├── hdf5/              # Full — h5py, SI canonicalization
│   │   │   ├── netcdf/            # Full — xarray, CF conventions
│   │   │   ├── iowarp/            # Full — CTE blob storage
│   │   │   ├── ndp/               # Full — NDP catalog via MCP
│   │   │   ├── kv_log_store/      # Lexical — key-value/log data
│   │   │   ├── vector_store/      # Stub — Qdrant protocol
│   │   │   └── graph_store/       # Stub — Neo4j protocol
│   │   ├── retrieval/              # 4-branch pipeline + agentic orchestration
│   │   ├── indexing/               # SI conversion (46 units, 12 domains), formulas
│   │   └── storage/                # DuckDB (8-table schema)
│   └── tests/                      # 283 automated tests
│
├── eval/                           # Evaluation scripts + results
│   └── eval_final/
│       ├── code/laptop/            # L1–L8 experiments
│       ├── code/delta/             # D1–D5 distributed experiments
│       ├── outputs/                # JSON results
│       └── plots/                  # Generated figures
│
└── background/                     # Literature review notes
```

---

## Quick Start

```bash
cd code
uv sync --all-extras --dev
```

### Query

```bash
# Cross-unit range query
uv run clio query --q "pressure between 190 and 360 kPa" --numeric-range "190:360:kPa"

# Formula query
uv run clio query --q "F = ma" --formula "F=ma"

# Multi-hop agentic retrieval
uv run clio query --q "high pressure turbulence" --agentic --max-hops 3

# Federated multi-namespace search
uv run clio query --namespaces "local_fs,hdf5_data" --q "temperature measurements"
```

### Index and Serve

```bash
uv run clio index --namespace local_fs
uv run clio serve                        # FastAPI at localhost:8000
```

### Tests

```bash
uv run pytest tests/ -v                  # All 283 tests
uv run ruff check src/ tests/            # Lint
uv run mypy                              # Type check
```

---

## Competitor Landscape

| System | SI Conv. | Formula | Federated | Adaptive | Data Search |
|--------|:--------:|:-------:|:---------:|:--------:|:-----------:|
| **CLIO (ours)** | **✓** | **✓** | **✓** | **✓** | **✓** |
| PANGAEA-GPT | ✗ (post-hoc) | ✗ | ✗ | ✓ | ✓ |
| LLM-Find | ✗ | ✗ | partial | ✓ | ✓ |
| NC-Retriever | ✗ (learned) | ✗ | ✗ | ✗ | ✗ |
| Numbers Matter! | ✗ (string) | ✗ | ✗ | ✗ | ✗ |
| OpenScholar | ✗ | ✗ | ✗ | ✗ | papers only |
| HiPerRAG | ✗ | ✗ | ✗ | ✗ | papers only |

---

## Paper Status

| Milestone | Status |
|-----------|--------|
| Abstract + Introduction | Revised (4 AI reviewer passes) |
| Background + Related Work | Complete (11 systems compared) |
| System Design | Complete (9 connectors, 4-stage pipeline) |
| Implementation | Complete (13.1K lines, 283 tests) |
| Evaluation | Complete (L1–L8 laptop + D1–D5 DeltaAI) |
| Submission target | SC2026 |

---

## Part of CLIO Kit

This work is part of [**CLIO Kit**](https://github.com/iowarp/clio-kit), the IOWarp platform's tooling layer for AI agents in scientific computing.
