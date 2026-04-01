# Evaluation Guide: Running on Delta AI / HPC Nodes

## Quick Start (All Benchmarks)

```bash
# 1. Clone and setup
git clone git@github.com:SIslamMun/clio-search.git
cd clio-search/code
uv sync --all-extras --dev

# 2. Download external datasets
git clone --depth=1 https://github.com/Tongji-KGLLM/NumConQ.git /tmp/NumConQ
curl -s "https://ws.pangaea.de/es/pangaea/panmd/_search?q=temperature+pressure+climate&size=500" \
  -o ../data/raw/pangaea_500.json
curl -s "https://www.osti.gov/dataexplorer/api/v1/records?q=temperature+pressure&rows=500" \
  -o ../data/raw/doe_500.json

# 3. Build all corpora
python3 benchmarks/build_real_corpus.py        # NOAA GHCN-Daily (288 docs)
python3 benchmarks/build_external_corpora.py   # NumConQ + PANGAEA + DOE

# 4. Run all evaluations
python3 benchmarks/evaluate_v2.py              # Controlled benchmark
python3 benchmarks/evaluate_all.py             # All external datasets

# 5. Run LLM provider benchmark (needs Ollama / API keys)
python3 benchmarks/llm_benchmark.py

# 6. Generate figures
python3 benchmarks/generate_figures.py

# 7. Compile paper
cd ../paper
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

---

## Delta AI SLURM Job Script

Save as `run_eval.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=clio-eval
#SBATCH --partition=gpuA100x4    # or cpu, gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --output=clio-eval-%j.out
#SBATCH --error=clio-eval-%j.err

# Load modules (adjust for your Delta setup)
module load python/3.11
module load cuda/12.0   # if using GPU for embeddings

# Setup
cd $HOME/clio-search/code
source .venv/bin/activate  # or: uv sync --all-extras --dev

# ============================================================
# Phase 1: Build corpora (if not already built)
# ============================================================
echo "=== Phase 1: Building corpora ==="

# NOAA (already in repo, skip if exists)
if [ ! -d benchmarks/corpus_real ]; then
    python3 benchmarks/build_real_corpus.py
fi

# External datasets
if [ ! -d benchmarks/corpus_numconq ]; then
    git clone --depth=1 https://github.com/Tongji-KGLLM/NumConQ.git /tmp/NumConQ
    curl -s "https://ws.pangaea.de/es/pangaea/panmd/_search?q=temperature+pressure+climate&size=500" \
      -o ../data/raw/pangaea_500.json
    curl -s "https://www.osti.gov/dataexplorer/api/v1/records?q=temperature+pressure&rows=500" \
      -o ../data/raw/doe_500.json
    python3 benchmarks/build_external_corpora.py
fi

# ============================================================
# Phase 2: Run controlled benchmark (full: baselines + ablation + federated)
# ============================================================
echo "=== Phase 2: Controlled benchmark ==="
python3 benchmarks/evaluate_v2.py

# ============================================================
# Phase 3: Run all external benchmarks
# ============================================================
echo "=== Phase 3: External benchmarks ==="
python3 benchmarks/evaluate_all.py --dataset noaa,numconq,pangaea,doe --max-queries 500

# ============================================================
# Phase 4: LLM provider benchmark (optional — needs network/API keys)
# ============================================================
echo "=== Phase 4: LLM benchmark ==="

# For Ollama (local): must be running on the node
# Start in background if available:
# ollama serve &
# sleep 5
# ollama pull llama3.2
# ollama pull qwen2.5:14b
# ollama pull mistral

# Set API keys if available
# export GOOGLE_API_KEY="your-gemini-key"
# export ANTHROPIC_API_KEY="your-claude-key"
# export OPENAI_API_KEY="your-openai-key"

python3 benchmarks/llm_benchmark.py

# ============================================================
# Phase 5: Generate figures and compile paper
# ============================================================
echo "=== Phase 5: Figures and paper ==="
python3 benchmarks/generate_figures.py
cd ../paper
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex

echo "=== Evaluation complete ==="
echo "Results in: eval/"
ls -la ../eval/*.json
```

Submit with: `sbatch run_eval.sh`

---

## Dataset Summary

| Dataset | Docs | Queries | Source | Units | Type |
|---------|------|---------|--------|-------|------|
| **Controlled** | 210 | 80 | Synthetic (7 domains) | Pa/kPa/MPa, mm/m, m/s | Cross-unit, formula, multi-constraint |
| **NOAA GHCN-Daily** | 288 | 8 | Real observatory data | degC, mm, m/s vs °F, inches, mph | Cross-unit (metric→imperial) |
| **NumConQ** | 5,362 | 6,577 | NC-Retriever (2025) | Mixed (5 domains) | Numeric constraints with qrels |
| **PANGAEA** | 500 | 1-7 | Geoscience catalog | hPa, °C, m, PSU | Unit matching |
| **DOE Data Explorer** | 500 | 5 | DOE/OSTI datasets | psi, °F, feet, m | Measurement retrieval |
| **Total** | **7,360** | **6,670+** | | | |

## Baselines (5 methods, all datasets)

| ID | Method | Description |
|----|--------|-------------|
| B1 | BM25 Only | Okapi BM25 (k1=1.2, b=0.75) |
| B2 | Dense Only | MiniLM-L6-v2 (384-dim) |
| B3 | Hybrid | BM25 + Vector (learned fusion) |
| B4 | String Norm | Hybrid + unit alias expansion |
| B5 | **Full Pipeline** | Hybrid + SI operators + formula normalization |

## Metrics

| Metric | Description | Primary? |
|--------|-------------|----------|
| **P@5** | Precision at 5 | Yes |
| **MRR** | Mean Reciprocal Rank | Yes |
| P@1, P@10, P@20 | Precision at other cutoffs | No |
| R@5, R@10, R@20 | Recall at cutoffs | No |
| F1@5 | Harmonic mean P@5/R@5 | No |
| nDCG@10 | Normalized Discounted Cumulative Gain | No |

## LLM Providers (for agentic rewriting benchmark)

| Provider | Model | Inference | Env Variable |
|----------|-------|-----------|-------------|
| Ollama | llama3.2, qwen2.5:14b, mistral | Local CPU | `ollama serve` must be running |
| Gemini | 2.0-flash, 1.5-flash, 1.5-pro | Cloud API | `GOOGLE_API_KEY` |
| Claude | sonnet, opus | Cloud API | `ANTHROPIC_API_KEY` |
| OpenAI | gpt-4o-mini | Cloud API | `OPENAI_API_KEY` |
| Together | llama-3.1-70b | Cloud API | `TOGETHER_API_KEY` |
| Groq | llama-3.1-8b | Cloud API | `GROQ_API_KEY` |
| Fallback | SI expansion | Deterministic | None needed |

## Output Files

All results saved as JSON in `eval/`:

```
eval/
├── benchmark_v2_results.json          # Controlled (210 docs)
├── real_benchmark_results.json        # NOAA GHCN-Daily (288 docs)
├── numconq_benchmark_results.json     # NumConQ (5362 docs)
├── pangaea_benchmark_results.json     # PANGAEA (500 docs)
├── doe_benchmark_results.json         # DOE (500 docs)
├── all_benchmark_results.json         # Combined results
├── llm_provider_results.json          # LLM provider comparison
├── EVALUATION_PLAN.md                 # Full evaluation plan
└── README_EVALUATION.md               # This file
```

## Running Individual Datasets

```bash
# Just NOAA
python3 benchmarks/evaluate_all.py --dataset noaa

# Just NumConQ (large — limit queries)
python3 benchmarks/evaluate_all.py --dataset numconq --max-queries 200

# NOAA + DOE
python3 benchmarks/evaluate_all.py --dataset noaa,doe

# Everything
python3 benchmarks/evaluate_all.py --dataset all
```

## Verifying Results

```bash
# Run all tests (should be 173 passed)
uv run pytest tests/ -v

# Quick smoke test
python3 -c "
import json
for f in ['benchmark_v2_results.json', 'real_benchmark_results.json']:
    with open(f'../eval/{f}') as fh:
        d = json.load(fh)
    print(f'{f}: loaded OK')
"
```

## Troubleshooting

**"Module not found" errors**: Run `uv sync --all-extras --dev` from the `code/` directory.

**NumConQ not found**: Clone it first: `git clone https://github.com/Tongji-KGLLM/NumConQ.git /tmp/NumConQ`

**Ollama not available**: The LLM benchmark will skip Ollama providers. Set `GOOGLE_API_KEY` for Gemini as a cloud alternative.

**Indexing slow**: First run builds vector embeddings (~100s for 288 docs on CPU). Subsequent runs reuse the index.

**Delta module issues**: Try `module load anaconda3` then create a conda env, or use `uv` directly if available.
