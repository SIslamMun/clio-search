#!/bin/bash
#PBS -N clio_dist_q
#PBS -l select=10:ngpus=6
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:flare
#PBS -q gpu_hack
#PBS -A gpu_hack
#PBS -j oe
set -u
module load python/3.12.12 apptainer/1.2.5 2>/dev/null

NUM_NODES=$(wc -l < $PBS_NODEFILE)
SCRIPT=/lus/flare/projects/gpu_hack/sislam6/clio_eval/scripts/distributed_mega_query.py
echo "Distributed mega query: $NUM_NODES ranks across 20 mega shards"
echo "  each rank handles ~$((20 / NUM_NODES)) shards"

# Cleanup any prior per-rank results
rm -f /lus/flare/projects/gpu_hack/sislam6/clio_eval/mega_query_dist/rank_*.json 2>/dev/null

T0=$(date +%s)
mpiexec -n $NUM_NODES -ppn 1 \
    apptainer exec --bind /lus:/lus,/tmp:/tmp \
        /home/sislam6/jarvis-work/sif/iowarp_deploy_cpu.sif \
        bash -c "source /home/sislam6/jarvis-work/iowarp-venv/bin/activate && WORLD_SIZE=$NUM_NODES python3 $SCRIPT"
T1=$(date +%s)
echo
echo "All ranks done in $((T1-T0))s"
echo
echo "===== aggregate (head, post-mpiexec) ====="
python3 << PYEOF
import json, glob
from collections import defaultdict
agg = defaultdict(lambda: {"hits": 0, "shards": 0, "rank_walls": []})
total_io = {"rchar": 0, "syscr": 0, "read_bytes": 0}
walls = []
for f in sorted(glob.glob('/lus/flare/projects/gpu_hack/sislam6/clio_eval/mega_query_dist/rank_*.json')):
    d = json.load(open(f))
    walls.append(d["total_wall_s"])
    for k in total_io:
        total_io[k] += d["io_delta"].get(k, 0)
    for r in d["results"]:
        a = agg[r["prompt_name"]]
        a["hits"] += r["local_hits"]
        a["shards"] += r["local_shards_with_hits"]
        a["rank_walls"].append(r["local_wall_s"])

print(f"n_ranks={len(walls)}  rank_wall p50={sorted(walls)[len(walls)//2]:.1f}s  max={max(walls):.1f}s")
print(f"aggregate IO across all ranks: rchar={total_io['rchar']:,} syscr={total_io['syscr']:,} read_bytes={total_io['read_bytes']:,}")
print()
print(f"{'prompt':<28} {'hits':>8} {'shards_w_hits':>16}")
for name, a in sorted(agg.items()):
    print(f"{name:<28} {a['hits']:>8} {a['shards']:>16}")
PYEOF
