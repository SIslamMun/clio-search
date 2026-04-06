#!/bin/bash
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# IMPORTANT: You MUST edit --account below to your ACCESS/DeltaAI allocation
# before submitting this script.  E.g.: #SBATCH --account=bbka-delta-gpu
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#SBATCH --job-name=clio_weak
#SBATCH --account=YOUR_ACCOUNT         # <-- EDIT THIS to your allocation
#SBATCH --partition=ghx4
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/clio_weak_%j.out
#SBATCH --error=logs/clio_weak_%j.err

# ---------------------------------------------------------------------------
# Weak scaling experiment on DeltaAI
#
# Each phase scales data AND workers together:
#   1 worker  →  625K docs
#   2 workers → 1.25M docs
#   4 workers → 2.50M docs
#
# Each worker's shard is the same size (625K) so per-worker work is held
# constant. Efficiency = reference_latency / phase_latency.
#
# Before running, execute prepare_weak_data.sh to create the phase-specific
# directories from the base shards:
#   bash prepare_weak_data.sh
#
# This creates:
#   /scratch/$USER/weak_phase_1/clio_shard_0.duckdb  (shard 0 only)
#   /scratch/$USER/weak_phase_2/clio_shard_{0,1}.duckdb
#   /scratch/$USER/weak_phase_4/clio_shard_{0,1,2,3}.duckdb
# ---------------------------------------------------------------------------

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs

source "$HOME/clio-venv/bin/activate"

REPO="$HOME/clio-search"
DIST_SCRIPT="$REPO/eval/eval_final/code/delta/distributed_clio.py"
DRIVER_SCRIPT="$REPO/eval/eval_final/code/delta/D1_D6_experiments.py"

NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
COORDINATOR="${NODES[0]}"
WORKERS=("${NODES[@]:1}")

run_phase() {
    local n_workers="$1"
    local phase_workers=("${WORKERS[@]:0:$n_workers}")
    local worker_urls=""

    echo ""
    echo "=========================================="
    echo "WEAK PHASE: $n_workers workers"
    echo "=========================================="

    for i in $(seq 0 $((n_workers - 1))); do
        w="${phase_workers[$i]}"
        srun --nodes=1 --ntasks=1 --nodelist="$w" \
            python3 "$DIST_SCRIPT" worker \
                --port 9201 \
                --shard-id "$i" \
                --total-shards "$n_workers" \
                --db-path "/scratch/$USER/weak_phase_${n_workers}/clio_shard_${i}.duckdb" \
                > "logs/weak_worker_${i}_${n_workers}_${SLURM_JOB_ID}.log" 2>&1 &
        worker_urls="${worker_urls}${worker_urls:+,}http://${w}:9201"
    done

    # Wait for all workers to be healthy before launching coordinator
    echo "Waiting for workers to become healthy..."
    for i in $(seq 0 $((n_workers - 1))); do
        w="${phase_workers[$i]}"
        for attempt in $(seq 1 30); do
            if curl -sf "http://${w}:9201/ping" > /dev/null 2>&1; then
                echo "  worker $i ($w) is healthy"
                break
            fi
            if [ "$attempt" -eq 30 ]; then
                echo "ERROR: worker $i ($w) did not start after 60s"
                pkill -P $$ -f distributed_clio.py || true
                return 1
            fi
            sleep 2
        done
    done

    srun --nodes=1 --ntasks=1 --nodelist="$COORDINATOR" \
        python3 "$DIST_SCRIPT" coordinator \
            --port 9200 \
            --workers "$worker_urls" \
            > "logs/weak_coord_${n_workers}_${SLURM_JOB_ID}.log" 2>&1 &

    # Wait for coordinator to be healthy
    echo "Waiting for coordinator to become healthy..."
    for attempt in $(seq 1 30); do
        if curl -sf "http://${COORDINATOR}:9200/profile" > /dev/null 2>&1; then
            echo "  coordinator is healthy"
            break
        fi
        if [ "$attempt" -eq 30 ]; then
            echo "ERROR: coordinator did not start after 60s"
            pkill -P $$ -f distributed_clio.py || true
            return 1
        fi
        sleep 2
    done

    export CLIO_COORDINATOR="http://${COORDINATOR}:9200"
    python3 "$DRIVER_SCRIPT" \
        --experiment weak \
        --output-suffix "_${n_workers}workers"

    pkill -P $$ -f distributed_clio.py || true
    sleep 5
}

run_phase 1
run_phase 2
run_phase 4

echo ""
echo "Weak scaling complete. See eval/eval_final/outputs/D_weak_*.json"
