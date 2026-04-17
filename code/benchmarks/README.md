# Benchmarks

Scripts for generating corpora, running retrieval benchmarks, and evaluating CLIO Search.

## Core benchmarks

| Script | Description |
|--------|-------------|
| `evaluate.py` | v1 baseline benchmark (40 docs) |
| `evaluate_v2.py` | v2 controlled benchmark (210 docs, 80 queries, 7 domains) |
| `evaluate_comparison.py` | Side-by-side method comparison (BM25, Dense, Hybrid, String norm., CLIO) |
| `evaluate_hard.py` | Hard cross-unit queries |
| `evaluate_real.py` | Real-world NOAA GHCN-Daily evaluation |
| `evaluate_agentic.py` | Agentic multi-hop retrieval benchmark |
| `evaluate_mcp.py` | NDP-MCP baseline comparison |
| `evaluate_metadata_adaptive.py` | Corpus-adaptive strategy benchmark |
| `evaluate_scale_projection.py` | Single-node scaling projection |
| `evaluate_all.py` | Run all benchmarks sequentially |

## Corpus generation

| Script | Description |
|--------|-------------|
| `generate_corpus_v2.py` | Generate 210-doc controlled benchmark corpus |
| `generate_queries_v2.py` | Generate 80 cross-unit queries across 7 domains |
| `build_real_corpus.py` | Build NOAA GHCN-Daily corpus from raw .dly files |
| `build_external_corpora.py` | Build DOE, PANGAEA corpora |

## LLM evaluation

| Script | Description |
|--------|-------------|
| `llm_benchmark.py` | Multi-provider LLM benchmark for agentic query rewriting |
| `llm_providers.py` | Pluggable provider system (Ollama, Gemini, Claude, OpenAI, vLLM, etc.) |

## Other

| Script | Description |
|--------|-------------|
| `generate_figures.py` | Generate paper figures from benchmark results |
| `real_test_ndp.py` | NDP live catalog test |
| `real_test_agentic.py` | Agentic retrieval end-to-end test |
| `statistical_tests.py` | Statistical significance tests on benchmark results |

## Corpora directories

| Directory | Description |
|-----------|-------------|
| `corpus/` | v1 baseline corpus (40 docs) |
| `corpus_v2/` | v2 controlled corpus (210 docs, generated) |
| `corpus_doe/` | DOE dataset corpus |
| `corpus_numconq/` | NumConQ numeric query corpus |
| `corpus_pangaea/` | PANGAEA earth science corpus |
| `corpus_real/` | Real-world NOAA corpus |

## Usage

```bash
cd code
uv sync --all-extras --dev

# Generate v2 corpus and run benchmark
python3 benchmarks/generate_corpus_v2.py
python3 benchmarks/evaluate_v2.py

# Run all benchmarks
python3 benchmarks/evaluate_all.py

# LLM benchmark (requires API keys)
python3 benchmarks/llm_benchmark.py
```
