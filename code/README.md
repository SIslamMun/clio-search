# CLIO Search — Implementation

Science-aware retrieval engine with an observe-decide-act-evaluate cycle for scientific data discovery. 13,100 lines of Python across 60 modules, with 275 tests.

## Architecture

```
Query → Namespace Registry → Retrieval Coordinator → 4 parallel branches:
  ├── Lexical (BM25, k1=1.2, b=0.75)
  ├── Vector (MiniLM-L6-v2, 384-dim, cosine similarity)
  ├── Graph (BFS depth=1, protocol-validated)
  └── Scientific (SI conversion + formula normalization)
→ Merge + Rerank (scientific hard filter) → Citations + Traces
```

The **AgenticRetriever** wraps this pipeline in a multi-hop loop with LLM query rewriting (expand, narrow, pivot, done).

## Connectors (9 backends)

| Connector | Capabilities | Location |
|---|---|---|
| Filesystem | Lexical, vector, scientific, profile | `connectors/filesystem/` |
| S3 Object Store | Lexical, vector, scientific, profile | `connectors/object_store/` |
| HDF5 (h5py) | Metadata, scientific | `connectors/hdf5/` |
| NetCDF (xarray) | Metadata, scientific | `connectors/netcdf/` |
| IOWarp CTE | Lexical, vector, scientific, profile | `connectors/iowarp/` |
| NDP (CKAN API) | Lexical, scientific | `connectors/ndp/` |
| KV Log Store | Lexical | `connectors/kv_log_store/` |
| Qdrant Vector | Vector (stub) | `connectors/vector_store/` |
| Neo4j Graph | Graph (stub) | `connectors/graph_store/` |

## Science-aware operators

- **SI unit conversion**: 58-unit registry across 12 physical domains. Converts measurements to SI base units via dimension vectors + scale factors. Correctness guaranteed by construction.
- **Formula normalization**: 6-step deterministic pipeline (strip whitespace, lowercase, unify superscripts, split on equality, sort factors, rejoin).
- **Corpus profiling**: SQL aggregations over indexed metadata — document count, measurement count, metadata density, unit domains — to decide which branches to activate.

## Quick start

```bash
uv sync --all-extras --dev
uv run pytest tests/ -v          # 275 tests
uv run ruff check src/ tests/    # Lint
uv run mypy                      # Type check (strict)
```

## CLI

```bash
uv run clio query --q "pressure 200 kPa" --numeric-range "190:360:kPa"
uv run clio query --q "F=ma" --formula "F=ma"
uv run clio query --q "turbulence" --agentic --max-hops 3
uv run clio index --namespace local_fs
uv run clio serve                # FastAPI at localhost:8000
```

## Source layout

```
src/clio_agentic_search/
├── api/              FastAPI server + routes
├── cli/              CLI (query, index, serve, list, seed)
├── connectors/       9 storage backend connectors
├── core/             NamespaceConnector protocol, registry
├── indexing/         SI conversion, measurement extraction, formula normalization
├── models/           Data contracts (ChunkRecord, CitationRecord, etc.)
├── retrieval/        4-branch pipeline, agentic orchestrator, query rewriter
├── storage/          DuckDB adapter (8-table schema)
├── evals/            Quality gate
├── telemetry/        OpenTelemetry + Prometheus
├── jobs.py           Async indexing job queue
└── retry.py          Retry/backoff wrappers
```

## Code style

- Python 3.11+, ruff (line-length=100, double quotes)
- `dataclass(frozen=True, slots=True)` for value types
- Type annotations everywhere, mypy strict mode
- Protocol-based capabilities (`LexicalSearchCapable`, `VectorSearchCapable`, etc.)
