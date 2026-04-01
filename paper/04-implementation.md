# 4. Implementation

clio-agentic-search is implemented as a single-binary Python system with no external service dependencies beyond an optional LLM endpoint. We cover the software stack and storage schema (Sec. 4.1), connectors for binary scientific formats (Sec. 4.2), and the experimental platform (Sec. 4.3).

## 4.1 System Implementation

The system is implemented in Python 3.11+ and exposes both a CLI and an async HTTP API via FastAPI. The CLI provides five primary commands---`clio query`, `clio index`, `clio serve`, `clio list`, and `clio seed`---with flags that control retrieval behavior: `--agentic` and `--max-hops` enable multi-hop retrieval, `--llm-rewrite` activates LLM query reformulation, and `--numeric-range` and `--formula` engage the science-aware operator branches. The HTTP API mirrors the CLI surface and serves as the entry point for programmatic and agent-driven access.

Dense retrieval uses sentence-transformers with the all-MiniLM-L6-v2 model (22.7M parameters, 384-dimensional embeddings, Apache 2.0 license). Two ANN backends are supported behind a common `ANNAdapter` protocol: `ExactANNAdapter`, which partitions vectors across 16 shards and computes exact cosine similarity, and `HnswANNAdapter`, which delegates to hnswlib for approximate search on larger corpora. Write safety on shared filesystems is enforced via `fcntl` file locking. Observability is provided by OpenTelemetry instrumentation with Prometheus-compatible `/metrics` endpoints; all retrieval branches and agentic hops emit structured trace events.

All persistent state resides in a single DuckDB database file. DuckDB was selected for three reasons: it is embedded (no external service to deploy on compute nodes), it supports SQL-native full-text search sufficient for BM25 scoring, and its single-file storage model is compatible with POSIX parallel filesystems common in HPC environments. The schema comprises eight tables:

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `documents` | Corpus membership | URI, namespace, SHA-256 checksum |
| `chunks` | Text segments | text, document FK, position |
| `embeddings` | Dense vectors | 384-dim float vector, chunk FK |
| `metadata` | Flexible annotations | key-value pairs per document or chunk |
| `file_index` | Change detection | path, SHA-256, mtime |
| `lexical_postings` | BM25 retrieval | token to chunk mapping, term frequency |
| `scientific_measurements` | Dimensional search | raw\_value, raw\_unit, canonical\_value, canonical\_unit |
| `scientific_formulas` | Formula matching | raw expression, normalized form |

The `scientific_measurements` and `scientific_formulas` tables are the storage-level counterpart of the science-aware operators described in Section 3. At index time, extracted quantities are canonicalized to base SI units and stored with both raw and canonical representations; at query time, the dimensional operator queries canonical columns directly, bypassing the embedding pipeline entirely.

The system is validated by 169 automated tests spanning unit, integration, and benchmark suites, executed via pytest with no external service dependencies.

## 4.2 HDF5 and NetCDF Connectors

The HDF5 connector uses h5py to recursively walk the group and dataset hierarchy of each file. For every dataset, the connector extracts the name, shape, dtype, and all attached HDF5 attributes. Attributes carrying physical semantics---`units`, `long_name`, `description`---are identified by convention and routed through the SI canonicalization pipeline described in Section 3.2, producing entries in the `scientific_measurements` table. Remaining attribute text is concatenated into chunk content for hybrid lexical and dense retrieval. Dataset structural metadata (shape, dtype, group path) is stored as key-value pairs in the `metadata` table, enabling queries such as "find all 3D datasets with a pressure variable." Incremental re-indexing uses SHA-256 checksums and filesystem mtime to skip unchanged files.

The NetCDF connector uses xarray and is aware of Climate and Forecast (CF) metadata conventions \cite{cfconventions}. For each file, the connector extracts all variables with their names, units, `long_name` attributes, dimensions, and shapes, as well as global attributes including `title`, `history`, and `Conventions`. Variable-level units pass through the same SI canonicalization pipeline as the HDF5 connector, ensuring that a pressure variable stored in hPa in one NetCDF file and in Pa in another will match the same dimensional query. Global attributes are indexed as document-level metadata. Both connectors share the same base architecture---threading, change detection, lexical ingestion, and scientific extraction---differing only in the format-specific traversal logic.

## 4.3 Experimental Platform

All experiments reported in Section 5 were conducted on [TBD --- hardware specifications to be added after benchmarking]. The software environment consists of Linux, Python 3.11, DuckDB 1.1, and sentence-transformers 3.3 with the all-MiniLM-L6-v2 embedding model (22.7M parameters, 384 dimensions). The embedding model was selected for its combination of compact size, permissive licensing (Apache 2.0), and demonstrated effectiveness on semantic textual similarity benchmarks \cite{reimers2019sbert}.

The clio-agentic-search codebase comprises approximately 9,300 lines of Python across 40 modules, with 169 automated tests. The project uses uv for dependency resolution and reproducible environment setup; `uv sync` installs all dependencies from a locked manifest. The full source code, test suite, and evaluation scripts are available as open-source software.

We note two honest limitations of the current prototype relevant to reproducibility. First, the Qdrant and Neo4j connectors use in-memory stub implementations rather than connecting to deployed service instances; they validate the connector protocol and capability negotiation logic but do not exercise network I/O or distributed storage. Second, all experiments in this paper run on a single node. Deploying the federated architecture across multiple HPC sites with heterogeneous storage backends---the scenario motivating the design---remains future work and is discussed in Section 6.
