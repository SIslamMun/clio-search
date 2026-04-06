# eval_final — SC26 Main Track Evaluation

Full evaluation pipeline for the SC26 main-track submission of
**CLIO Search: Agentic Scientific Data Discovery with Intelligent
Orchestration**.

## Architecture

CLIO is an agentic retrieval service (not an MCP server, not an LLM agent).
It connects to data sources via connectors (NDP-MCP, filesystem, S3, HDF5,
etc.), indexes into DuckDB with science-aware operators (SI unit
canonicalization, quality filtering, formula normalization), and runs
multi-hop retrieval via `AgenticRetriever`.

The evaluation compares two paths for the same query:
```
Path 1 (baseline):  LLM agent → NDP-MCP → NDP API  (agent does all reasoning)
Path 2 (CLIO):      LLM agent → CLIO → NDP-MCP → NDP API → DuckDB → agentic search
                    (CLIO does the reasoning, agent just reads compact results)
```

## Directory layout

```
eval/eval_final/
├── README.md                        ← this file
├── code/
│   ├── laptop/                      ← L1–L8 experiments (run locally)
│   │   ├── L1_single.py             ← L1: LLM+NDP-MCP vs LLM+CLIO (10 queries)
│   │   ├── L2_iowarp_cte_scaling.py
│   │   ├── L3_si_unit_cross_prefix.py
│   │   ├── L4_numconq_benchmark.py
│   │   ├── L5_federation_100_namespaces.py
│   │   ├── L6_cimis_quality_filter.py
│   │   ├── L7_scaling_curves.py
│   │   └── L8_cross_corpus_diversity.py
│   │
│   ├── delta/                       ← D1–D6 experiments (run on DeltaAI)
│   │   ├── distributed_clio.py      ← coordinator/worker TCP binary
│   │   ├── D1_D6_experiments.py     ← experiment driver
│   │   ├── download_arxiv.py        ← data prep (2.5M arXiv abstracts)
│   │   ├── index_shard.py           ← per-worker DuckDB indexer
│   │   ├── prepare_weak_data.sh     ← weak scaling data setup
│   │   ├── slurm_strong_scaling.sh  ← Slurm batch for D1 (strong scaling)
│   │   └── slurm_weak_scaling.sh    ← Slurm batch for D2 (weak scaling)
│   │
│   └── generate_plots.py            ← generates all paper figures from JSONs
│
├── outputs/                         ← JSON results (one per experiment)
└── plots/                           ← PNG figures for the paper
```

---

## Results (laptop tier, completed)

| Experiment | Key claim | Result |
|---|---|---|
| **L1** LLM+NDP-MCP vs LLM+CLIO (10 queries) | CLIO reduces token consumption for LLM agents | **70.1% token reduction**, 78% fewer tool calls, 74% faster |
| **L2** IOWarp CTE scaling (1K→50K blobs) | Tag-filtered BlobQuery scales sub-linearly | 50× data → 5.76× query time (sub-linear) |
| **L3** SI unit cross-prefix (42 synthetic docs) | Dimensional analysis R@5 where baselines fail | CLIO 0.69 vs Dense 0.00 |
| **L4** NumConQ (2,614 queries, 2 domains) | Competitive on numeric benchmark | CLIO R@5 = 0.36 ≈ BM25 |
| **L5** 100-namespace federation | Strategy adaptation saves retrieval work | 52.3% branches saved |
| **L6** CIMIS quality filter (5 stations, 657K rows) | Quality filter drops bad rows at SQL level | 12,796 rows rejected |
| **L7** Scaling curves (1K→100K) | Sub-linear profile + query time | 100× data → 4.44× profile, 2.82× query |
| **L8** Cross-corpus diversity (NOAA+DOE+controlled) | Metadata-adaptive strategy classification | Rich/sparse/default correctly identified |

---

## How to run

### Laptop experiments

```bash
cd clio-search/code
uv sync --all-extras --dev

# L1: requires NDP-MCP server + Claude Agent SDK
uv add claude-agent-sdk aiohttp
python3 ../eval/eval_final/code/laptop/L1_single.py

# L2: requires Docker + iowarp wheel
python3 ../eval/eval_final/code/laptop/L2_iowarp_cte_scaling.py

# L3–L8: no special dependencies
python3 ../eval/eval_final/code/laptop/L3_si_unit_cross_prefix.py
python3 ../eval/eval_final/code/laptop/L4_numconq_benchmark.py
python3 ../eval/eval_final/code/laptop/L5_federation_100_namespaces.py
python3 ../eval/eval_final/code/laptop/L6_cimis_quality_filter.py
python3 ../eval/eval_final/code/laptop/L7_scaling_curves.py
python3 ../eval/eval_final/code/laptop/L8_cross_corpus_diversity.py

# Generate plots
python3 ../eval/eval_final/code/generate_plots.py
```

### DeltaAI experiments

```bash
# 1. Clone repo on DeltaAI
ssh delta-ai
git clone git@github.com:SIslamMun/clio-search.git
cd clio-search

# 2. Set up Python
module load python
python3 -m venv $HOME/clio-venv
source $HOME/clio-venv/bin/activate
cd code && pip install -e . && pip install aiohttp
cd ..

# 3. Download arXiv data (2.5M abstracts, ~5 GB)
python3 eval/eval_final/code/delta/download_arxiv.py \
    --output-dir /scratch/$USER/arxiv --shards 4

# 4. Build DuckDB indices (4 parallel jobs, ~30 min)
for i in 0 1 2 3; do
    python3 eval/eval_final/code/delta/index_shard.py \
        --shard-jsonl /scratch/$USER/arxiv/arxiv_shard_$i.jsonl \
        --db-path /scratch/$USER/clio_shard_$i.duckdb \
        --namespace distributed_clio &
done
wait

# 5. Prepare weak scaling data (symlinks phase directories)
bash eval/eval_final/code/delta/prepare_weak_data.sh

# 6. IMPORTANT: Edit Slurm scripts — change YOUR_ACCOUNT
vim eval/eval_final/code/delta/slurm_strong_scaling.sh
vim eval/eval_final/code/delta/slurm_weak_scaling.sh

# 7. Submit jobs
sbatch eval/eval_final/code/delta/slurm_strong_scaling.sh   # D1 strong scaling
sbatch eval/eval_final/code/delta/slurm_weak_scaling.sh     # D2 weak scaling

# 8. Monitor
squeue -u $USER

# 9. When done — generate plots
python3 eval/eval_final/code/generate_plots.py
```

---

## How L1 works (the headline experiment)

**Query**: "Find datasets with temperature above 30°C from weather stations"

### Path 1: LLM agent + NDP-MCP (baseline)
```
Agent calls search_datasets("temperature") → NDP returns 25 dataset JSONs
Agent reads all 25 descriptions (raw catalog text enters LLM context)
Agent calls search_datasets("weather") → 25 more
Agent calls get_dataset_details() for 4 promising ones
Agent reasons, filters, writes prose answer
→ 12 tool calls, 131K tokens, 58 seconds
```

### Path 2: LLM agent + CLIO
```
Agent calls clio_search(query="...", numeric_constraint={unit:"degC", min:30})
  └─ CLIO internally:
     1. NDPMCPClient connects to NDP-MCP via MCP protocol
     2. Discovers 50 datasets
     3. Indexes into DuckDB (chunks, embeddings, measurements)
     4. Downloads CSVs → parse_scientific_csv → extract measurements
        → canonicalize_measurement(30, "degC") → 303.15 K
        → derive_flag_from_value() → quality="good"
     5. AgenticRetriever runs multi-hop:
        BM25 + vector + scientific branches in parallel
        FallbackQueryRewriter expands unit variants
     6. Returns 10 ranked citations (~2K tokens)
Agent reads compact citations, writes summary
→ 2 tool calls, 39K tokens, 19 seconds
```

**Result: 70% fewer tokens because CLIO kept 100KB of catalog text
out of the LLM context.**

---

## Delta experiments (D1–D6)

| Experiment | What it tests | Setup |
|---|---|---|
| **D1** Strong scaling | Query latency with 1/2/4 workers on fixed 2.5M corpus | 5 nodes |
| **D2** Weak scaling | Per-worker latency stays flat as data grows with workers | 5 nodes |
| **D3** Indexing throughput | Parallel sharded indexing across 4 workers | 4 nodes |
| **D4** Cross-unit at scale | Unit canonicalization correctness on 2.5M corpus | 4 nodes |
| **D5** Numeric queries at scale | 20 scientific queries against 2.5M distributed corpus | 4 nodes |
| **D6** Federation at scale | 100-namespace routing through distributed coordinator | 4 nodes |

### How distributed CLIO works
```
Coordinator (1 node, port 9200)
  ├── Worker 0 (node 1, port 9201) → DuckDB shard 0 (625K docs)
  ├── Worker 1 (node 2, port 9201) → DuckDB shard 1 (625K docs)
  ├── Worker 2 (node 3, port 9201) → DuckDB shard 2 (625K docs)
  └── Worker 3 (node 4, port 9201) → DuckDB shard 3 (625K docs)

Query flow: Coordinator fans out to all workers → each worker
queries its local DuckDB shard → results merged + reranked
```

Workers are stateless HTTP servers (aiohttp). Coordinator aggregates
results by score. Communication is TCP/JSON — no MPI needed.

---

## Paper claims → experiments

| Claim | Evidence |
|---|---|
| CLIO reduces LLM token consumption for scientific discovery | L1 (70.1% reduction over 10 queries) |
| Dimensional analysis gives correct cross-unit retrieval | L3 (CLIO R@5 vs baselines) |
| Quality filter drops bad data at SQL level | L6 (12K rows rejected from CIMIS) |
| Metadata-adaptive strategy saves retrieval work | L5 (52% branches saved) |
| Sub-linear single-node scaling | L7 (100× data → 4.4× profile time) |
| Near-linear distributed scaling | D1, D2 (on DeltaAI) |
| CLIO works across data source types | L8 (NOAA, DOE, controlled corpora) |
