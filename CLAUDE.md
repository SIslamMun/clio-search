# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SC2026 research paper + implementation for **clio-agentic-search**: a retrieval engine with science-aware operators (arithmetic SI unit conversion, formula normalization) that execute as first-class branches alongside BM25 and dense vector search across federated HPC storage backends.

## Build & Development Commands

All commands run from `code/` directory:

```bash
cd code
uv sync --all-extras --dev    # Install all dependencies
uv run pytest tests/ -v       # Run all 283 tests
uv run pytest tests/unit/ -v  # Unit tests only
uv run pytest tests/unit/test_formula_normalization.py -v  # Single test file
uv run pytest -k "test_measurement_range" -v  # Single test by name
uv run ruff check src/ tests/  # Lint
uv run ruff format src/ tests/ # Format
uv run mypy                    # Type check (strict mode)
```

### CLI

```bash
uv run clio query --q "pressure 200 kPa" --numeric-range "190:360:kPa"
uv run clio query --q "F=ma" --formula "F=ma"
uv run clio query --q "turbulence" --agentic --max-hops 3
uv run clio index --namespace local_fs
uv run clio serve  # FastAPI at localhost:8000
```

### Benchmarks

```bash
cd code
python3 benchmarks/evaluate.py       # v1 baseline benchmark (40 docs)
python3 benchmarks/evaluate_v2.py    # v2 benchmark (210 docs, 80 queries)
python3 benchmarks/llm_benchmark.py  # Multi-provider LLM benchmark
```

### Paper compilation

```bash
cd paper
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

## Architecture

### Retrieval Pipeline

```
Query → Namespace Registry → Retrieval Coordinator → 4 parallel branches:
  ├── Lexical (BM25, k1=1.2, b=0.75)     [storage/duckdb_store.py]
  ├── Vector (MiniLM-L6-v2, 384-dim)      [retrieval/ann.py]
  ├── Graph (BFS depth=1)                  [connectors/graph_store/]
  └── Scientific (SI conversion + formula) [retrieval/scientific.py]
→ Merge + Rerank → Citations + Traces
```

The **AgenticRetriever** (`retrieval/agentic.py`) wraps this pipeline in a multi-hop loop with LLM query rewriting (`retrieval/query_rewriter.py`).

### Key Design Patterns

- **Protocol-based capabilities**: Connectors declare what they support via `LexicalSearchCapable`, `VectorSearchCapable`, `ScientificSearchCapable`, `GraphSearchCapable` protocols in `retrieval/capabilities.py`. The coordinator checks `isinstance()` before dispatching.
- **Science-aware operators**: `indexing/scientific.py` contains the SI conversion table (`_UNIT_CANONICALIZATION`), measurement extraction regex, and formula normalization. These produce metadata stored in DuckDB `scientific_measurements` and `scientific_formulas` tables.
- **Connector protocol**: All connectors implement `NamespaceConnector` from `core/connectors.py` — `descriptor()`, `connect()`, `teardown()`, `index()`, `build_citation()`. The `FilesystemConnector` is the reference implementation.

### Connectors (6 types)

| Connector | Status | Location |
|---|---|---|
| Filesystem | Full | `connectors/filesystem/connector.py` |
| S3 Object Store | Full | `connectors/object_store/connector.py` |
| HDF5 (h5py) | Full | `connectors/hdf5/connector.py` |
| NetCDF (xarray) | Full | `connectors/netcdf/connector.py` |
| Qdrant Vector | Stub (in-memory) | `connectors/vector_store/connector.py` |
| Neo4j Graph | Stub (in-memory) | `connectors/graph_store/connector.py` |

### Storage

Single DuckDB file with 8 tables. Schema in `storage/duckdb_store.py`. The two science-specific tables (`scientific_measurements`, `scientific_formulas`) enable dimensional range queries and formula lookups via SQL.

### LLM Providers

`benchmarks/llm_providers.py` has a pluggable provider system. `OllamaProvider`, `GeminiProvider`, `ClaudeAgentProvider`, `OpenAICompatibleProvider` (base for LM Studio, vLLM, OpenAI, Together, Groq), `FallbackProvider` (no LLM). Discovery via `discover_providers()`.

## Code Style

- Python 3.11+, ruff line-length=100, double quotes
- `dataclass(frozen=True, slots=True)` for value types, `dataclass(slots=True)` for mutable
- Type annotations everywhere, mypy strict mode
- Optional imports guarded with `try/except` + `HAS_X` flag pattern
- ruff lint rules: E, F, I, B, UP

## Repository Layout

- `code/` — Python implementation (~13K lines, 283 tests)
- `paper/` — LaTeX paper (paper.tex, references.bib, paper.pdf, figures/)
- `eval/` — Saved benchmark results (JSON)
- `.planning/` — WTF-P paper planning infrastructure
- `background/` — Literature review notes
