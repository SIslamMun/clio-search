# Clio-Agentic-Search: Full Evaluation Plan

## 1. Research Questions

| RQ | Question | Section |
|----|----------|---------|
| RQ1 | Does dimensional conversion improve retrieval when query and document units differ? | §5.1 |
| RQ2 | Does formula normalization improve retrieval on notation variants? | §5.2 |
| RQ3 | Does federated search increase coverage across namespaces? | §5.3 |
| RQ4 | Which pipeline component contributes the most gain? (Ablation) | §5.4 |
| RQ5 | Does multi-hop agentic retrieval improve over single-pass? | §5.5 |
| RQ6 | How do LLM providers compare for query rewriting? | §5.5 |
| RQ7 | Do results generalize to real-world observatory data? | §5.7 |

---

## 2. Datasets

### 2.1 Controlled Benchmark (Existing)
- **Corpus**: 210 documents, 7 scientific domains
- **Source**: Synthetically generated with parametric unit variation
- **Domains**: atmospheric_science (40), chemistry (30), fluid_dynamics (30), hpc_simulation (25), materials_science (30), mixed (45), negatives (10)
- **Unit variation**: Same quantities expressed in different SI prefixes (Pa/kPa/MPa, mm/cm/m, etc.)
- **Queries**: 80 total — 40 cross-unit, 20 same-unit (control), 10 formula, 10 multi-constraint
- **Ground truth**: Automatically generated from document metadata
- **Location**: `code/benchmarks/corpus_v2/`, `code/benchmarks/queries_v2.json`
- **Script**: `python3 benchmarks/evaluate_v2.py`

### 2.2 Real-World NOAA GHCN-Daily (Existing)
- **Corpus**: 288 documents (8 stations × 36 months)
- **Source**: NOAA Global Historical Climatology Network - Daily (Menne et al., 2012)
- **Stations**: Boston Logan, NYC Central Park, LAX, Miami Intl, Atlanta Hartsfield, Anchorage Intl, Boulder CO, Portland Intl
- **Years**: 2022, 2023, 2024
- **Document units**: Metric only (degC, mm, m/s) — no imperial in documents
- **Queries**: 8 cross-unit queries in imperial (°F, inches, mph)
- **Ground truth**: Threshold arithmetic on raw station measurements
- **Location**: `code/benchmarks/corpus_real/`, `code/benchmarks/real_queries.json`
- **Raw data**: `data/raw/*.dly` (NOAA fixed-width station files)
- **Script**: `python3 benchmarks/evaluate_real.py`

### 2.3 Extended Real-World (TODO — for full paper)

#### A. PANGAEA Geoscience Records
- **Source**: PANGAEA Data Publisher (https://www.pangaea.de)
- **Access**: REST API `https://ws.pangaea.de/es/pangaea/panmd/_search`
- **Target**: 500 documents from ocean/climate science with explicit unit metadata
- **Unit types**: Temperature (°C), salinity (PSU), depth (m/km), pressure (dbar/hPa)
- **Queries**: 20 cross-unit + 10 same-unit control
- **Why**: Independent geoscience source, peer-reviewed metadata, different domain from NOAA

#### B. FinQuant / MedQuant (Numbers Matter!, EMNLP 2024)
- **Source**: https://drive.google.com/file/d/1JD2a2BRU8-gf5arLufpZw-51nIM6_kJm
- **Content**: Finance (stock prices, market caps) and medical (dosages, lab values) quantity-aware queries
- **Why**: Published benchmark from peer-reviewed work, tests quantity matching outside physical sciences
- **Note**: Not SI units — tests the system's graceful degradation on unsupported unit types

#### C. ORNL DAAC Climate Data (Optional)
- **Source**: https://daac.ornl.gov (NASA Distributed Active Archive Center)
- **Target**: 200 climate dataset metadata records
- **Why**: DOE-affiliated, SC-relevant, real HPC data descriptions

---

## 3. System Configuration

### 3.1 Hardware
- **CPU**: AMD/Intel (specify exact model from test machine)
- **RAM**: (specify)
- **GPU**: None (CPU-only inference for Ollama)
- **Storage**: Local SSD
- **OS**: Linux 6.17.0

### 3.2 Software Stack
- **Python**: 3.11+
- **DuckDB**: Storage backend (8 tables)
- **Sentence-Transformers**: MiniLM-L6-v2 (384-dim embeddings)
- **BM25**: Okapi BM25 (k1=1.2, b=0.75)
- **Framework**: uv + pytest

### 3.3 Indexing Configuration
- **Chunk size**: Structure-aware (sections, captions, equations, tables)
- **SI conversion table**: 16 units, 6 domains
- **Measurement extraction**: Regex-based pattern matching
- **Formula normalization**: 6-step deterministic pipeline

---

## 4. LLM Providers

### 4.1 Provider Matrix

| Provider | Model | Parameters | Inference | Cost/query | Temperature |
|----------|-------|------------|-----------|------------|-------------|
| Ollama | Llama 3.2 | 3B | Local CPU | $0 | 0.0 |
| Ollama | Qwen 2.5 | 14B | Local CPU | $0 | 0.0 |
| Ollama | Mistral | 7B | Local CPU | $0 | 0.0 |
| Gemini | 2.0 Flash | — | Cloud API | ~$0 (free tier) | 0.0 |
| Gemini | 1.5 Flash | — | Cloud API | ~$0 (free tier) | 0.0 |
| Gemini | 1.5 Pro | — | Cloud API | $0.30/M tokens | 0.0 |
| Claude | Sonnet | — | Cloud API | $3/$15 per M | 0.0 |
| Claude | Opus | — | Cloud API | $15/$75 per M | 0.0 |
| OpenAI | GPT-4o-mini | — | Cloud API | $0.15/$0.60 per M | 0.0 |
| Together | Llama 3.1 70B | 70B | Cloud API | $0.20/M tokens | 0.0 |
| Groq | Llama 3.1 8B | 8B | Cloud API | $0.05/M tokens | 0.0 |
| — | Fallback (no LLM) | 0 | Deterministic | $0 | N/A |

### 4.2 LLM Configuration
- **Temperature**: 0.0 (deterministic, reproducible)
- **Max tokens**: 256 (query rewriting only)
- **System prompt**: Scientific query optimization (expand/narrow/pivot/done strategies)
- **Retry policy**: 3 retries with exponential backoff
- **Timeout**: 30s per query

### 4.3 What LLMs Do in Our System
- LLMs are used ONLY for **query rewriting** in the agentic loop
- They do NOT do retrieval, ranking, or unit conversion
- The scientific operators (SI conversion, formula normalization) are **deterministic** — no LLM involved
- This is why the no-LLM fallback achieves 96% of LLM quality

---

## 5. Baselines

### 5.1 Retrieval Baselines (All Benchmarks)

| ID | Method | Components | What It Tests |
|----|--------|-----------|---------------|
| B1 | BM25 Only | Lexical search (Okapi BM25) | Pure keyword matching |
| B2 | Dense Only | Vector search (MiniLM-L6-v2) | Semantic similarity |
| B3 | Hybrid | BM25 + Vector (learned fusion) | Standard hybrid retrieval |
| B4 | String Norm. | Hybrid + unit alias expansion | Can string ops bridge units? |
| B5 | **Full Pipeline** | Hybrid + SI operators + formula | Our complete system |

### 5.2 Ablation Configurations (Controlled Benchmark Only)

| Config | Components | Purpose |
|--------|-----------|---------|
| A | BM25 only | Lexical baseline |
| B | A + dense vectors | + semantic similarity |
| C | B + scientific operators | + SI conversion + formula |
| D | C + graph + agentic | Full pipeline |

### 5.3 Agentic Configurations

| Config | Hops | Rewriter | Purpose |
|--------|------|----------|---------|
| 1-hop | 1 | Single-pass | Baseline (no iteration) |
| 2-hop | 2 | LLM rewrite | One refinement |
| 3-hop | 3 | LLM rewrite | Two refinements |

---

## 6. Metrics

### 6.1 Retrieval Quality
| Metric | Formula | Primary? | Why |
|--------|---------|----------|-----|
| **P@5** | Precision at K=5 | **Yes** | How many of top-5 are relevant |
| **MRR** | Mean Reciprocal Rank | **Yes** | How quickly first relevant appears |
| P@1 | Precision at 1 | No | Exact first-result accuracy |
| P@10 | Precision at 10 | No | Broader precision |
| R@5 | Recall at 5 | No | Fraction of relevant docs found (bounded by K/|relevant|) |
| R@10 | Recall at 10 | No | Broader recall |
| F1@5 | Harmonic mean of P@5, R@5 | No | Balance metric |

### 6.2 Efficiency Metrics
| Metric | What | Unit |
|--------|------|------|
| Rewrite latency | LLM query rewriting time | seconds |
| Query latency | End-to-end retrieval time | seconds |
| Indexing throughput | Full index build time | docs/second |
| Incremental index | Re-index time at 10% change | seconds |

### 6.3 Coverage Metrics (Federated)
| Metric | What |
|--------|------|
| Single-namespace coverage | % of queries with ≥1 relevant result from one namespace |
| Multi-namespace coverage | % of queries with ≥1 relevant result across all namespaces |

### 6.4 Cost Metrics (LLM)
| Metric | What | Unit |
|--------|------|------|
| Cost per query | API cost for rewriting | USD |
| Token usage | Prompt + completion tokens | count |

### 6.5 Statistical Significance
- **Method**: Paired bootstrap resampling
- **Iterations**: 10,000
- **Threshold**: p < 0.05
- **Applied to**: P@5 and MRR comparisons between Full Pipeline and each baseline

---

## 7. Evaluation Harness

### 7.1 Scripts

| Script | Input | Output | What It Runs |
|--------|-------|--------|-------------|
| `evaluate_v2.py` | `corpus_v2/`, `queries_v2.json` | `eval/benchmark_v2_results.json` | 5 baselines + ablation + federated + agentic + indexing |
| `evaluate_real.py` | `corpus_real/`, `real_queries.json` | `eval/real_benchmark_results.json` | 5 baselines on NOAA data |
| `llm_benchmark.py` | `corpus_v2/`, 10 queries | `eval/llm_provider_results.json` | All LLM providers: rewriting + retrieval + multi-hop |
| `generate_figures.py` | `eval/*.json` | `paper/figures/fig4-8.pdf` | All publication figures |

### 7.2 Execution Order

```
# Step 1: Build corpora
cd code
python3 benchmarks/generate_corpus_v2.py    # 210 synthetic docs
python3 benchmarks/generate_queries_v2.py   # 80 queries
python3 benchmarks/build_real_corpus.py     # 288 NOAA docs + 8 queries

# Step 2: Run controlled benchmark (RQ1-RQ5)
python3 benchmarks/evaluate_v2.py           # ~5 min

# Step 3: Run real-world benchmark (RQ7)
python3 benchmarks/evaluate_real.py         # ~3 min

# Step 4: Run LLM provider benchmark (RQ6)
python3 benchmarks/llm_benchmark.py         # ~30 min (depends on providers)

# Step 5: Generate figures
python3 benchmarks/generate_figures.py      # ~10 sec

# Step 6: Compile paper
cd ../paper
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

### 7.3 Reproducibility
- All eval results saved as JSON in `eval/`
- Random seed: Not applicable (deterministic operators, temperature=0.0 for LLMs)
- Corpus generation is deterministic (seeded)
- Raw NOAA data included in `data/raw/` for full reproducibility

---

## 8. Expected Results Matrix

### 8.1 Controlled Benchmark (Current)

| Method | P@5 | MRR | Status |
|--------|-----|-----|--------|
| BM25 | 0.40 | 0.45 | Done ✓ |
| Dense | 0.36 | 0.50 | Done ✓ |
| Hybrid | 0.36 | 0.47 | Done ✓ |
| String Norm | 0.32 | 0.39 | Done ✓ |
| **Full Pipeline** | **0.99** | **0.99** | Done ✓ |

### 8.2 Real-World NOAA (Current)

| Method | P@5 | MRR | Status |
|--------|-----|-----|--------|
| BM25 | 0.375 | 0.562 | Done ✓ |
| Dense | 0.250 | 0.515 | Done ✓ |
| Hybrid | 0.375 | 0.708 | Done ✓ |
| String Norm | 0.375 | 0.708 | Done ✓ |
| **Full Pipeline** | **0.700** | **0.760** | Done ✓ |

### 8.3 LLM Providers (Current — 4 verified, 7 from raw log)

| Provider | Latency | P@5 | MRR | Status |
|----------|---------|-----|-----|--------|
| Ollama Llama 3.2 | 4.93s | 0.90 | 0.95 | Raw log |
| Ollama Qwen 2.5 | 18.08s | 0.90 | 1.00 | Raw log |
| Ollama Mistral | 10.52s | 0.90 | 0.95 | Raw log |
| Gemini 2.0 Flash | 0.09s | 0.90 | 1.00 | Raw log |
| Gemini 1.5 Pro | 0.05s | 0.90 | 1.00 | Raw log |
| Claude Sonnet | 8.24s | 0.86 | 0.88 | JSON ✓ |
| Fallback | 0.00s | 0.86 | 0.88 | JSON ✓ |

### 8.4 Ablation (Current)

| Config | P@5 | MRR | Delta P@5 | Status |
|--------|-----|-----|-----------|--------|
| A: BM25 | 0.40 | 0.45 | — | Done ✓ |
| B: +Vector | 0.36 | 0.47 | -0.04 | Done ✓ |
| C: +Scientific | 0.99 | 0.99 | **+0.63** | Done ✓ |
| D: Full | 0.99 | 0.99 | +0.00 | Done ✓ |

---

## 9. TODO for Full Paper Deadline

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| **P0** | Re-run LLM benchmark with all available providers (save structured JSON) | 1 hour | Verifies LLM table |
| **P0** | Add PANGAEA dataset (500 docs, 30 queries) | 2 hours | Third independent dataset |
| **P1** | Add statistical significance tests (bootstrap) | 1 hour | Reviewer requirement |
| **P1** | Record exact hardware specs (CPU model, RAM) | 5 min | Reproducibility |
| **P2** | Try FinQuant/MedQuant for graceful degradation test | 2 hours | Shows system handles unsupported units |
| **P2** | Add OpenAI/Together/Groq to LLM benchmark | 1 hour | More providers in table |
| **P3** | Scale test: 10K docs indexing throughput | 1 hour | Scalability evidence |
