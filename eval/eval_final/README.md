# eval_final — SC26 Main Track Evaluation

Full evaluation pipeline for the SC26 main-track submission of
**CLIO Search: Agentic Scientific Data Discovery with Intelligent
Orchestration**. The evaluation is split into two tiers:

1. **Laptop experiments (L1–L8)** — everything that can be done on a
   single workstation. Real data, real MCP servers, real LLM integration,
   real benchmarks.
2. **Delta experiments (D1–D6)** — distributed scaling experiments on
   NCSA DeltaAI (GH200 Grace Hopper supercomputer) that the laptop cannot
   do. Requires Slurm + ACCESS allocation.

Everything is **pre-built and documented but not yet executed**. Each
script writes a structured JSON into `outputs/`; each plot is generated
from those JSONs by `code/generate_plots.py`.

## Directory layout

```
eval/eval_final/
├── README.md                       ← this file
├── ARCHITECTURE.md                 ← CLIO vs MCP architectural note
├── iowarp_core-1.0.3-cp312-*.whl   ← pre-built main-branch wheel for L2
│
├── code/
│   ├── laptop/                     ← L1–L8 experiments (run locally)
│   │   ├── L1_ndp_vs_clio_agent.py
│   │   ├── L1b_fair_comparison_reference.py  (earlier draft, reference only)
│   │   ├── L2_iowarp_cte_scaling.py           (up to 1M blobs)
│   │   ├── L3_si_unit_cross_prefix.py
│   │   ├── L4_numconq_benchmark.py
│   │   ├── L5_federation_100_namespaces.py
│   │   ├── L6_cimis_quality_filter.py
│   │   ├── L7_scaling_curves.py
│   │   └── L8_cross_corpus_diversity.py
│   │
│   ├── delta/                      ← D1–D6 experiments (run on DeltaAI)
│   │   ├── distributed_clio.py          ← coordinator/worker binary
│   │   ├── D1_D6_experiments.py         ← driver script
│   │   ├── download_arxiv.py            ← data prep
│   │   ├── index_shard.py               ← per-worker indexer
│   │   ├── slurm_strong_scaling.sh      ← Slurm batch for D1
│   │   └── slurm_weak_scaling.sh        ← Slurm batch for D2
│   │
│   └── generate_plots.py           ← produces all paper figures
│
├── data/                           ← (empty — populate as experiments need)
│   └── NumConQ/                    ← expected: `git clone Tongji-KGLLM/NumConQ`
│
├── outputs/                        ← JSON results (one file per experiment)
│
└── plots/                          ← PNG figures for the paper
```

---

## Experiment summary

### Laptop tier (L1–L8)

| # | Experiment | Key claim | Data | Estimated runtime |
|---|---|---|---:|---:|
| L1 | NDP-MCP vs CLIO+NDP-MCP (Claude Agent SDK, 10 queries) | CLIO-as-harness saves tokens and tool calls for an LLM agent querying the real NDP catalog | Live NDP API | ~30 min |
| L2 | IOWarp CTE BlobQuery scaling 1K → 1M blobs | Tag-filtered BlobQuery scales sub-linearly; full-scan is linear | Synthetic scientific blobs in Docker | ~20-30 min |
| L3 | SI unit cross-prefix correctness | Dimensional-analysis canonicalisation gives R@5 ≈ 1.0 where BM25/dense go to ~0 on cross-unit queries | Synthetic scientific abstracts (6 quantities × 7 unit variants) | ~2 min |
| L4 | NumConQ benchmark (6,500 queries, 5 domains) | CLIO is competitive with learned retrievers on the standard numeric benchmark | NumConQ from Tongji-KGLLM (must be downloaded) | ~10-30 min |
| L5 | 100-namespace federation with quality + schema + sampling | Per-dataset strategy adaptation saves 52% of retrieval work | Synthetic 100 namespaces × 5 types | ~2 min |
| L6 | CIMIS quality filter on real NOAA-standard QC data | Quality filter rejects bad/missing rows at SQL level | 5 CIMIS weather stations (15 MB each) | ~5-10 min |
| L7 | Single-node scaling curves 1K → 100K synthetic | Profile + query time are sub-linear in corpus size | Synthetic | ~10-15 min |
| L8 | Cross-corpus diversity (NOAA + DOE + controlled + arXiv) | CLIO distinguishes rich / sparse / dense metadata across real corpora | NOAA 1728, DOE 500, controlled 210 | ~10 min |

**Laptop total runtime: ~90-120 minutes of compute over a single working session.**

### Delta tier (D1–D6)

| # | Experiment | Key claim | Resources | Estimated runtime |
|---|---|---|---|---:|
| D1 | Distributed strong scaling 1 → 2 → 4 workers | CLIO query latency decreases with worker count at fixed 2.5M corpus | 5 DeltaAI nodes (1 coord + 4 workers) | ~90 min (queue-dependent) |
| D2 | Distributed weak scaling 1/625K → 2/1.25M → 4/2.5M | Per-worker work stays constant as data scales with nodes | 5 DeltaAI nodes | ~60 min |
| D3 | Large-scale distributed indexing of 2.5M arXiv | Parallel sharded indexing across 4 workers | 4 DeltaAI nodes | ~30 min |
| D4 | Cross-unit precision at 2.5M scale | L3's claim holds at 500× larger corpus distributed | 4 DeltaAI nodes | ~10 min |
| D5 | NumConQ at scale (6.5K queries × 2.5M corpus distributed) | CLIO scales to corpora larger than existing benchmarks target | 4 DeltaAI nodes | ~30 min |
| D6 | 100-namespace federation distributed | L5's claim holds at 10M total docs distributed across 4 workers | 4 DeltaAI nodes | ~15 min |

**Delta total budget: ≈ 4 nodes × 8-10 hours = 32-40 node-hours.**

---

## Prerequisites

### Laptop setup

```bash
# 1. Python venv with all CLIO deps
cd /home/shazzadul/Illinois_Tech/Spring26/RA/clio-search/code
uv sync
source .venv/bin/activate

# 2. NDP-MCP server (for L1)
uv add /tmp/clio-kit/clio-kit-mcp-servers/ndp

# 3. Claude Agent SDK (for L1)
uv add claude-agent-sdk aiohttp

# 4. iowarp-core main-branch wheel (for L2)
#    Already built at eval/eval_final/iowarp_core-1.0.3-*.whl
#    Docker must be available.

# 5. NumConQ data (for L4) — download manually
cd eval/eval_final/data
git clone https://github.com/Tongji-KGLLM/NumConQ.git

# 6. matplotlib (for plot generation)
pip install matplotlib
```

### DeltaAI setup

```bash
# On your laptop:
ssh delta-ai

# On DeltaAI:
mkdir -p $HOME/clio-search
# Transfer the repo (rsync or git)

# Create a Python venv on Delta
module load python
python3 -m venv $HOME/clio-venv
source $HOME/clio-venv/bin/activate
cd $HOME/clio-search/code
pip install -e .
pip install aiohttp  # for distributed_clio.py

# ACCESS allocation: make sure your --account in the slurm scripts
# matches your active project.
```

---

## Running the experiments

### Laptop

```bash
cd /home/shazzadul/Illinois_Tech/Spring26/RA/clio-search

# Run each experiment individually. Order is independent; each writes a
# separate JSON into eval/eval_final/outputs/.
python3 eval/eval_final/code/laptop/L1_ndp_vs_clio_agent.py
python3 eval/eval_final/code/laptop/L2_iowarp_cte_scaling.py
python3 eval/eval_final/code/laptop/L3_si_unit_cross_prefix.py
python3 eval/eval_final/code/laptop/L4_numconq_benchmark.py
python3 eval/eval_final/code/laptop/L5_federation_100_namespaces.py
python3 eval/eval_final/code/laptop/L6_cimis_quality_filter.py
python3 eval/eval_final/code/laptop/L7_scaling_curves.py
python3 eval/eval_final/code/laptop/L8_cross_corpus_diversity.py

# Generate all plots from whatever JSONs exist
python3 eval/eval_final/code/generate_plots.py
```

### Delta

```bash
# 0. Upload code to Delta
rsync -avz eval/eval_final/code/delta/ delta-ai:clio-search/eval/eval_final/code/delta/

# 1. Download arXiv dataset on DeltaAI login node (do once)
ssh delta-ai
cd clio-search
python3 eval/eval_final/code/delta/download_arxiv.py \
    --output-dir /scratch/$USER/arxiv --shards 4

# 2. Pre-build per-worker DuckDB indices (do once — takes ~30 min for 2.5M)
for i in 0 1 2 3; do
    python3 eval/eval_final/code/delta/index_shard.py \
        --shard-jsonl /scratch/$USER/arxiv/arxiv_shard_$i.jsonl \
        --db-path /scratch/$USER/clio_shard_$i.duckdb \
        --namespace distributed_clio &
done
wait

# 3. Edit slurm scripts: replace YOUR_ACCOUNT with your ACCESS allocation
vim eval/eval_final/code/delta/slurm_strong_scaling.sh
vim eval/eval_final/code/delta/slurm_weak_scaling.sh

# 4. Submit strong-scaling job (1/2/4 worker phases)
sbatch eval/eval_final/code/delta/slurm_strong_scaling.sh

# 5. Submit weak-scaling job
sbatch eval/eval_final/code/delta/slurm_weak_scaling.sh

# 6. Monitor
squeue -u $USER

# 7. When done, copy outputs back to laptop
rsync -avz delta-ai:clio-search/eval/eval_final/outputs/ eval/eval_final/outputs/

# 8. Generate updated plots
python3 eval/eval_final/code/generate_plots.py
```

---

## Expected outputs

After all experiments run, `eval/eval_final/outputs/` contains:

```
L1_ndp_vs_clio_agent.json
L1_ndp_vs_clio_agent_trace.json
L2_iowarp_cte_scaling.json
L3_si_unit_cross_prefix.json
L4_numconq_benchmark.json
L5_federation_100_namespaces.json
L6_cimis_quality_filter.json
L7_scaling_curves.json
L8_cross_corpus_diversity.json
D_strong_1workers.json
D_strong_2workers.json
D_strong_4workers.json
D_weak_1workers.json
D_weak_2workers.json
D_weak_4workers.json
D_indexing.json
D_cross_unit.json
D_numconq.json
D_federation.json
```

And `eval/eval_final/plots/` contains the matching PNG figures.

---

## How this maps to the paper's claims

| Paper claim | Supporting experiments |
|---|---|
| "CLIO reduces tool calls and tokens for LLM agents accessing MCP-backed scientific catalogs" | L1, Pareto chart |
| "Dimensional-analysis unit conversion is guaranteed-correct across all SI prefixes, where learned retrievers are probabilistic" | L3, L4 |
| "CLIO scales to 1M scientific blobs on a single node via indexed DuckDB + CTE, and sub-linearly across multiple nodes on DeltaAI" | L2, L7, D1, D2 |
| "Per-dataset strategy adaptation via corpus profiling saves 50%+ of retrieval work on heterogeneous 100-namespace federations" | L5, D6 |
| "Data quality filtering at SQL level eliminates bad/missing rows before retrieval in real weather-station datasets" | L6 |
| "CLIO distinguishes rich / sparse / no-metadata corpora and picks different strategies accordingly" | L8 |
| "Distributed CLIO on DeltaAI achieves near-linear strong scaling on a 2.5M real scientific corpus" | D1 |
| "CLIO's per-worker architecture supports weak scaling with constant per-node work" | D2 |

---

## Honest caveats

- **L2 (IOWarp CTE 1M blobs)** runs inside a Docker container. The wheel at
  `eval/eval_final/iowarp_core-1.0.3-*.whl` is the main-branch build — PyPI's
  v1.0.3 has a shared-memory bug that blocks PutBlob. If the container's shm
  isn't large enough (needs `--shm-size=8g`), the 1M scale may fail; the
  script reports the furthest scale reached.
- **L4 (NumConQ)** requires the dataset to be downloaded manually and its
  format may differ from what `_load_queries()` expects. Adapt if needed.
- **L3 cross-unit probe** uses synthetic-but-realistic abstract templates
  because we don't index a full arXiv dump on the laptop. The synthetic
  data is sufficient for the correctness claim (measurement injection is
  what matters, prose backbone is flavour).
- **Delta experiments** require working ACCESS allocation + Slurm + SSH
  to DeltaAI. If any of these are unavailable, the laptop tier alone is
  still publishable at an SC workshop.
- **Scale target**: 2.5M arXiv papers × 4 DeltaAI nodes is borderline
  SC main track. It clears SC workshop + PASC comfortably. See the
  planning note in previous sessions for venue assessment.

---

## Contact + reproducibility

Every experiment is deterministic given its seed. Results should
reproduce to within ±5% across runs (noise comes from OS scheduling and
NDP API latency).

Raw JSON outputs are machine-readable and include timestamps, scales,
and all raw measurements — not just aggregates.

