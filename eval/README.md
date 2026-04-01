# Evaluation Data and Benchmarks

This directory contains the benchmark queries and results used in the
clio-agentic-search SC2026 paper evaluation (Section 5).

## Files

### queries.json

The controlled benchmark query set spanning five SI domains (pressure, distance,
mass, time, velocity). Each entry contains:

- `id` -- unique query identifier (e.g., `pressure_01`)
- `query` -- natural-language retrieval query
- `domain` -- SI domain (pressure, distance, mass, time, velocity)
- `numeric_range` -- structured range in `low:high:unit` format
- `relevant_docs` -- ground-truth relevant document paths
- `type` -- query condition: `cross_unit` (query and document units differ),
  `same_unit` (units match), or `formula` (formula-variant queries)

The benchmark comprises 35 total retrieval tasks partitioned into cross-unit,
same-unit control, and formula-variant conditions.

### baseline_results.json

Full retrieval results for all baseline configurations:

- **bm25_only** -- DuckDB full-text search with default BM25 scoring
- **dense_only** -- all-MiniLM-L6-v2 embeddings (384-dim) with exact cosine similarity
- **hybrid** -- linear fusion of BM25 and dense scores with equal weights
- **string_normalization** -- Numbers Matter! approach (canonical string form)
- **full_pipeline** -- complete clio-agentic-search with dimensional conversion,
  formula normalization, graph traversal, and agentic rewriting

Each baseline reports P@K, R@K, F1@K (K = 1, 3, 5, 10) and MRR, broken down by
query type (overall, cross_unit, same_unit, formula) and per-query.

### llm_provider_results.json

Results from testing the agentic multi-hop query rewriting loop across four LLM
providers:

- `ollama/functiongemma-v6:latest` -- local open-weight model
- `gemini/gemini-1.5-flash` -- Google Gemini API
- `claude-agent-sdk/sonnet` -- Anthropic Claude via agent SDK
- `fallback/si-expansion` -- deterministic SI-expansion heuristic (no LLM)

Reports per-provider availability, average latency, token usage, cost, and
per-query rewriting behavior (strategy selected, rewritten query, reasoning).

## Reproducing the Benchmarks

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Seed the benchmark corpus:
   ```bash
   clio seed --benchmark
   ```

3. Index the benchmark documents:
   ```bash
   clio index --namespace benchmark
   ```

4. Run the baseline evaluation:
   ```bash
   python -m pytest tests/ -k benchmark --tb=short
   ```

5. Run cross-unit evaluation with the full pipeline:
   ```bash
   clio query --numeric-range "200:300:kPa" --namespace benchmark
   ```

6. Run agentic multi-hop evaluation:
   ```bash
   clio query "experiments with pressure around 250 kPa" \
     --agentic --max-hops 3 --llm-rewrite
   ```

Results are written to this directory. The evaluation scripts compute P@K, R@K,
F1@K, and MRR at K = 1, 3, 5, 10 using the ground-truth relevance judgments
from `queries.json`.

## Key Results Summary

| Configuration        | P@5  | R@5  | F1@5 | MRR  |
|----------------------|------|------|------|------|
| BM25-only            | 0.35 | 0.60 | 0.42 | 0.73 |
| Dense-only           | 0.36 | 0.66 | 0.44 | 0.66 |
| Hybrid BM25+Dense    | 0.37 | 0.68 | 0.45 | 0.69 |
| String normalization | 0.34 | 0.79 | 0.46 | 0.70 |
| **Full pipeline**    | **0.97** | **1.00** | **0.98** | **1.00** |

On cross-unit queries, the dimensional-conversion operator achieves R@5 = 1.00
where the strongest text baseline (string normalization) scores 0.79. The
ablation study confirms the B-to-C transition (adding science-aware operators)
provides the largest marginal gain (+0.53 absolute F1@5). Formula normalization
adds 0.62 F1@5 over text baselines on notation-variant queries. Federated search
recovers 93% of relevant documents across filesystem, S3, and HDF5 backends.
