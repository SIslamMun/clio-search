# CLIO Search: Agentic Search for Scientific Data

SC2026 submission — implementation, evaluation, and paper.

## Overview

Scientific data search requires human reasoning that does not scale: inspecting corpora, adapting to metadata quality, converting between unit conventions, refining iteratively. CLIO Search automates this through an observe-decide-act-evaluate cycle. Science-aware operators execute deterministic SI conversion and formula normalization alongside BM25 and dense search. CLIO inspects what metadata each corpus provides and adapts accordingly. Federated search spans heterogeneous HPC backends with agentic refinement.

### Key Results

| Metric | Value |
|--------|-------|
| Cross-unit P@5 | **0.99** (strongest baseline: 0.40) |
| Real-world NOAA validation | **+87%** precision over hybrid baseline |
| Distributed scale | 2.5M arXiv, 4 GH200 nodes, p50 < 42 ms, 0.94 weak scaling |
| Agentic efficiency | **52% token reduction, 91% time reduction** vs native agent (10/10 correct) |
| IOWarp scaling | 17 ms profiling at 1K → 90 ms at 500K blobs |

## Repository Structure

```
clio-search/
├── code/                           # Implementation (13.1K lines, 60 modules, 275 tests)
│   ├── src/clio_agentic_search/
│   │   ├── connectors/             # 9 storage backends (filesystem, S3, HDF5,
│   │   │                           #   NetCDF, IOWarp, NDP, KV, Qdrant, Neo4j)
│   │   ├── retrieval/              # 4-branch pipeline + agentic orchestration
│   │   ├── indexing/               # SI conversion (58 units, 12 domains), formulas
│   │   └── storage/                # DuckDB (8-table schema)
│   ├── tests/                      # 275 automated tests
│   └── benchmarks/                 # Benchmark scripts
│
├── paper/                          # SC2026 paper (10 pages, double-anonymous)
│   ├── paper.tex                   # Main driver
│   ├── paper.pdf                   # Compiled PDF
│   ├── sections/                   # Per-section LaTeX files
│   ├── figures/                    # Architecture diagram + plots
│   ├── references.bib              # 52 BibTeX entries
│   ├── ad_appendix/                # SC26 Artifact Description appendix
│   └── gcasr_poster/               # GCASR poster abstract
│
├── eval/eval_final/                # Evaluation data
│   ├── code/laptop/                # L1–L8 experiments
│   ├── code/delta/                 # Distributed experiments (DeltaAI GH200)
│   ├── outputs/                    # JSON results (all paper tables + figures)
│   └── plots/                      # Generated figures
│
└── background/                     # Literature review notes
```

## Quick Start

```bash
cd code
uv sync --all-extras --dev

# Run tests
uv run pytest tests/ -v

# Query
uv run clio query --q "pressure 200 kPa" --numeric-range "190:360:kPa"
uv run clio query --q "F=ma" --formula "F=ma"
uv run clio query --q "turbulence" --agentic --max-hops 3

# Index and serve
uv run clio index --namespace local_fs
uv run clio serve
```

## Paper Compilation

```bash
cd paper
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

## Evaluation

All paper results live in `eval/eval_final/outputs/`. See `eval/eval_final/README.md` for the full mapping of result files to paper tables and figures.

```bash
# Reproduce plots
cd code
python3 ../eval/eval_final/code/generate_plots.py
```

## Part of CLIO Kit

This work is part of [**CLIO Kit**](https://github.com/iowarp/clio-kit), the IOWarp platform's tooling layer for AI agents in scientific computing.
