# eval_final — SC26 Evaluation Data

Evaluation results and scripts for **CLIO Search: Agentic Search for Scientific Data** (SC26 submission).

## Directory layout

```
eval/eval_final/
├── README.md
├── code/
│   ├── laptop/              ← L1–L8 experiments (run locally)
│   ├── delta/               ← Distributed experiments (run on DeltaAI GH200)
│   └── generate_plots.py    ← Generates all paper figures from JSONs
├── outputs/                 ← JSON results (one per experiment)
├── plots/                   ← PNG/PDF figures for the paper
└── data/                    ← NOAA GHCN-Daily and NumConQ datasets
```

## Paper tables and figures → result files

| Paper Element | Description | Result File(s) |
|---|---|---|
| **Table V** | Agentic efficiency (3-path comparison) | `L1_three_path.json`, `L1_q02_q10_fixed.json` |
| **Table VI** | Cross-unit retrieval P@5 (210 docs, 80 queries) | `L4_numconq_benchmark.json` |
| **Table VII** | NOAA GHCN-Daily real-world validation | `L6_cimis_quality_filter.json` |
| **Table VIII** | Ablation (science operators contribution) | `L3_si_unit_cross_prefix.json` |
| **Table IX** | IOWarp CTE scaling (1K–500K blobs) | `L2_delta_mem.json`, `L2_delta_mem_large.json`, `L2C_delta_mem.json` |
| **Table X** | Distributed scaling (2.5M arXiv, 1–4 GH200 nodes) | `D_weak_*.json`, `D_strong_*.json` |
| **Fig 2** | Agentic efficiency bar chart (tokens + time) | `L1_three_path.json` |
| **Fig 3** | Cross-unit P@5 bar chart | `L4_numconq_benchmark.json` |
| **Fig 4** | IOWarp scaling dual-axis plot | `L2_delta_mem.json`, `L2C_delta_mem.json` |
| **Fig 5** | Distributed cross-unit correctness | `D_cross_unit.json` |
| **Sec 5 prose** | Federation routing (100 namespaces) | `federation_100_namespaces.json` |

## Key results (final, as submitted)

| Experiment | Paper Claim | Result |
|---|---|---|
| **L1** Three-path comparison (10 queries, NDP 341 datasets) | CLIO reduces agentic search cost | 52% token reduction, 91% time reduction, 10/10 correct |
| **L3** SI unit cross-prefix ablation | Science operators drive accuracy | CLIO 0.99 P@5 vs BM25 0.40 |
| **L6** NOAA GHCN-Daily validation | Works on real observatory data | CLIO 0.700 P@5 vs Hybrid 0.375 |
| **L2** IOWarp CTE scaling | Sub-linear profiling | 17ms at 1K → 90ms at 500K (rich) |
| **D** Distributed scaling (DeltaAI GH200) | Near-linear weak scaling | 0.94 efficiency at 4 workers, p50 < 42ms |

## How to run

### Laptop experiments (L1–L8)

```bash
cd clio-search/code
uv sync --all-extras --dev

# L1: requires Claude Agent SDK + API key
python3 ../eval/eval_final/code/laptop/L1_single.py

# L3–L8: no special dependencies
python3 ../eval/eval_final/code/laptop/L3_si_unit_cross_prefix.py
python3 ../eval/eval_final/code/laptop/L4_numconq_benchmark.py
python3 ../eval/eval_final/code/laptop/L5_federation_100_namespaces.py
python3 ../eval/eval_final/code/laptop/L6_cimis_quality_filter.py

# Generate plots
python3 ../eval/eval_final/code/generate_plots.py
```

### DeltaAI experiments (distributed + IOWarp)

```bash
ssh delta-ai
cd clio-search

# IOWarp scaling
sbatch eval/eval_final/code/delta/slurm_l2_mem.sh
sbatch eval/eval_final/code/delta/slurm_l2_mem_large.sh

# Distributed scaling (2.5M arXiv, 1–4 GH200 nodes)
python3 eval/eval_final/code/delta/download_arxiv.py --output-dir /scratch/$USER/arxiv --shards 4
sbatch eval/eval_final/code/delta/slurm_strong_scaling.sh
sbatch eval/eval_final/code/delta/slurm_weak_scaling.sh
```

## Artifacts (AD Appendix)

- **A1** (`clio-search` repo): Supports C1 (science-aware operators), C2 (corpus-adaptive orchestration) → Tables IV–VIII, Figs 2–3
- **A2** (`eval/eval_final/`): Supports C1, C2, C3 (federated search) → Tables IX–X, Figs 4–5
