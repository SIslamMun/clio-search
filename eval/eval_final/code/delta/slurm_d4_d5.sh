#!/bin/bash
# ---------------------------------------------------------------------------
# D4 + D5: Cross-unit precision at scale + NumConQ distributed
#
# D4: Same pressure/temperature cross-unit queries as L3 but against the
#     full 2.5M arXiv corpus distributed across 4 workers. Proves SI
#     canonicalization correctness holds at scale.
#
# D5: 20 NumConQ-style numeric-range queries (Higgs, CMB, LHC, etc.)
#     against the distributed index. Reports hit rate and latency.
#
# Requires: shards already built by slurm_setup.sh
# Produces: eval/eval_final/outputs/D_cross_unit.json
#           eval/eval_final/outputs/D_numconq.json
#
# Usage:  sbatch slurm_d4_d5.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=clio_d4_d5
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/clio_d4_d5_%j.out
#SBATCH --error=logs/clio_d4_d5_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs

WORK="/work/nvme/bekn/sislam3"
module load python/3.11.9
source "${WORK}/clio-venv/bin/activate"

REPO="/u/sislam3/clio-search"
DIST_SCRIPT="$REPO/eval/eval_final/code/delta/distributed_clio.py"
DRIVER_SCRIPT="$REPO/eval/eval_final/code/delta/D1_D6_experiments.py"

NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
COORDINATOR="${NODES[0]}"
WORKERS=("${NODES[@]:1}")

echo "========================================"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Coordinator: $COORDINATOR"
echo "Workers: ${WORKERS[@]}"
echo "========================================"

# Start 4 workers
worker_urls=""
for i in 0 1 2 3; do
    w="${WORKERS[$i]}"
    echo "Launching worker $i on $w (shard $i)"
    srun --nodes=1 --ntasks=1 --nodelist="$w" \
        python3 "$DIST_SCRIPT" worker \
            --port 9201 \
            --shard-id "$i" \
            --total-shards 4 \
            --db-path "${WORK}/clio_shard_${i}.duckdb" \
            > "logs/d4d5_worker_${i}_${SLURM_JOB_ID}.log" 2>&1 &
    worker_urls="${worker_urls}${worker_urls:+,}http://${w}:9201"
done

# Wait for workers
echo "Waiting for workers..."
for i in 0 1 2 3; do
    w="${WORKERS[$i]}"
    for attempt in $(seq 1 30); do
        if curl -sf "http://${w}:9201/ping" > /dev/null 2>&1; then
            echo "  worker $i ($w) healthy"
            break
        fi
        [ "$attempt" -eq 30 ] && { echo "ERROR: worker $i timed out"; exit 1; }
        sleep 2
    done
done

# Start coordinator
echo "Launching coordinator on $COORDINATOR"
srun --nodes=1 --ntasks=1 --nodelist="$COORDINATOR" \
    python3 "$DIST_SCRIPT" coordinator \
        --port 9200 \
        --workers "$worker_urls" \
        > "logs/d4d5_coordinator_${SLURM_JOB_ID}.log" 2>&1 &

# Wait for coordinator
echo "Waiting for coordinator..."
for attempt in $(seq 1 30); do
    if curl -sf "http://${COORDINATOR}:9200/profile" > /dev/null 2>&1; then
        echo "  coordinator healthy"
        break
    fi
    [ "$attempt" -eq 30 ] && { echo "ERROR: coordinator timed out"; exit 1; }
    sleep 2
done

export CLIO_COORDINATOR="http://${COORDINATOR}:9200"

# Run D4
echo ""
echo "=========================================="
echo "Running D4 (cross-unit precision at scale)"
echo "=========================================="
python3 "$DRIVER_SCRIPT" --experiment cross_unit

# Run D5
echo ""
echo "=========================================="
echo "Running D5 (NumConQ distributed)"
echo "=========================================="
python3 "$DRIVER_SCRIPT" --experiment numconq

# Tear down
pkill -P $$ -f distributed_clio.py || true

echo ""
echo "D4 + D5 complete."
echo "  eval/eval_final/outputs/D_cross_unit.json"
echo "  eval/eval_final/outputs/D_numconq.json"
