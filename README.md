# clio-search

**Science-Aware Hybrid Retrieval with Dimensional Conversion for HPC Data Discovery**

Research workspace and implementation for a paper submitted to **SC2026** (Data Analytics, Visualization & Storage track).

---

## Abstract

Scientific corpora encode structured knowledge — dimensional quantities, mathematical formulas, and heterogeneous storage provenance — that general-purpose retrieval systems cannot exploit. AI agents searching HPC data face three compounding failures: retrieval systems cannot match measurements across unit prefixes ("200 kPa" vs "200000 Pa"), cannot match formulas across formatting variations ("F=ma" vs "F = m · a"), and cannot query across the heterogeneous storage backends where scientific data resides. Embedding models achieve only 0.54 accuracy on numerical content, and existing quantity-aware systems normalize unit strings without performing dimensional conversion across SI prefixes.

We present **clio-agentic-search**, a hybrid retrieval engine that introduces *science-aware retrieval operators* for HPC data discovery:

1. **Dimensional-conversion measurement retrieval** — canonicalizes quantities to base SI units via explicit multiplication (kPa × 10³ = Pa), enabling cross-prefix numeric comparison guaranteed correct by construction.
2. **Formula normalization** — matches mathematical expressions regardless of formatting, whitespace, or factor ordering, unified with measurement operators in a single pipeline.
3. **Federated multi-namespace search** — queries heterogeneous HPC storage backends (filesystems, S3, vector databases, graph databases) through a connector architecture with per-backend capability negotiation.

---

## Key Differentiator

| Capability | Numbers Matter! [2024] | CONE [2026] | Context-1 [2026] | HiPerRAG [2025] | **Ours** |
|---|---|---|---|---|---|
| Dimensional conversion (kPa×10³=Pa) | ✗ | ✗ | ✗ | ✗ | **✓** |
| Formula matching | ✗ | ✗ | ✗ | ✗ | **✓** |
| Cross-unit range queries | ✗ | ✗ | ✗ | ✗ | **✓** |
| Federated multi-backend | ✗ | ✗ | ✗ | ✗ | **✓** |
| Searches scientific data (not papers) | ✗ | ✗ | ✗ | ✗ | **✓** |
| BM25 + vector hybrid | ✗ | ✗ | ✓ | ✓ | ✓ |

---

## Motivating Example

**Query**: *"Find experiments where pressure was between 190 and 360 kPa"*

Corpus spread across filesystem, S3, and vector DB:
- Doc A: "measured pressure was **250 kPa**" (filesystem)
- Doc B: "chamber pressure reached **200000 Pa**" (S3 archive)
- Doc C: "peak pressure: **0.3 MPa**" (Qdrant)

**BM25 / string normalization**: finds only Doc A (kPa token match).
**Learned embeddings**: uncertain — 0.54 accuracy on numbers.
**clio-agentic-search**: converts all to Pa (250000, 200000, 300000), range-checks against 190000–360000 Pa, finds **all three** across all backends.

---

## Repository Structure

```
clio-search/
├── intro/
│   ├── abstract.md            # SC2026 abstract (v0.5, 250 words)
│   └── introduction.md        # Paper introduction (v0.5)
├── background/
│   ├── background.md          # Full background section (hybrid retrieval, numerical crisis,
│   │                          #   quantity-aware retrieval, agentic science narrative)
│   └── papers/
│       ├── 01_prior_art_and_competitors.md   # 16 papers, threat levels, differentiation
│       ├── 02_gaps_and_evidence.md           # Three failures + supporting infrastructure papers
│       ├── 03_agentic_gaps_evidence.md       # Evidence agents fail at data (stats table)
│       └── 04_agentic_science_papers.md      # 4-act narrative: agents doing science
├── design/
│   ├── architecture.md        # System design with ASCII diagrams, BM25 formula, SI conversion
│   ├── hypothesis.md          # 3 RQs, 3 hypotheses, comparison table
│   ├── novelty.md             # Core novelty argument, gap validation
│   └── figures.md             # Figure descriptions and captions
├── codebases/
│   └── related_repos_analysis.md   # 17 GitHub repos analyzed: threat matrix, differentiation
└── code/                      # clio-agentic-search implementation
    ├── src/clio_agentic_search/
    │   ├── api/               # FastAPI server
    │   ├── cli/               # CLI (clio query/index/serve)
    │   ├── connectors/        # filesystem, S3, Qdrant, Neo4j, Redis
    │   ├── retrieval/         # coordinator, BM25, vector, scientific operators
    │   ├── storage/           # DuckDB store (Okapi BM25 SQL)
    │   └── indexing/          # lexical, vector, scientific indexing
    └── tests/                 # 136/138 tests passing
```

---

## Running the Code

### Setup

```bash
cd code
uv sync --all-extras --dev
```

### Query

```bash
# Dimensional conversion query (cross-unit)
uv run clio query --namespace local_fs --q "pressure between 190 and 360 kPa"

# Formula query
uv run clio query --namespace local_fs --q "F = ma"

# Standard semantic query
uv run clio query --namespace local_fs --q "turbulence simulation results"
```

### Index and Serve

```bash
uv run clio index --namespace local_fs      # Index local filesystem
uv run clio serve                           # Start FastAPI at localhost:8000
```

### API

```
POST /query          — run retrieval, returns citations + trace
GET  /documents      — list indexed documents
POST /jobs/index     — async indexing job
GET  /health         — liveness probe
GET  /metrics        — Prometheus metrics
```

### Tests

```bash
uv run pytest --ignore=tests/benchmarks -v
uv run python -m clio_agentic_search.evals.quality_gate   # full quality gate
```

---

## Paper Status

| Milestone | Date | Status |
|---|---|---|
| Abstract submission | April 1, 2026 | Target |
| Full paper submission | April 8, 2026 | Target |
| Venue | SC2026 — Data Analytics, Visualization & Storage | — |

---

## Related Work Summary

~54 unique papers surveyed across:
- Quantity-aware retrieval: QFinder, CQE, Numbers Matter!, CONE, NC-Retriever
- Hybrid retrieval foundations: BM25 fusion, BGE-M3, SPLADE, Rank1
- Scientific RAG: HiPerRAG, OpenScholar, MCP for HPC (arXiv:2508.18489)
- Agentic science: AI Scientist v1/v2, Coscientist, ChemCrow, ALS beamline agents
- Benchmarks: ScienceAgentBench (32%), MLAgentBench, Numeracy Gap (0.54)

Full analysis in [`background/papers/`](background/papers/) and [`codebases/related_repos_analysis.md`](codebases/related_repos_analysis.md).

---

## Part of CLIO Kit

This work is part of [**CLIO Kit**](https://github.com/iowarp/clio-kit), the IoWarp platform's tooling layer for AI agents in scientific computing.
