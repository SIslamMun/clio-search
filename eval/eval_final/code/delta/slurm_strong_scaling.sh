#!/bin/bash
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# IMPORTANT: You MUST edit --account below to your ACCESS/DeltaAI allocation
# before submitting this script.  E.g.: #SBATCH --account=bbka-delta-gpu
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#SBATCH --job-name=clio_strong
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4               # DeltaAI GH200 partition
#SBATCH --nodes=5                      # 1 coordinator + 4 workers
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1                   # one GH200 per node
#SBATCH --time=02:00:00
#SBATCH --output=logs/clio_strong_%j.out
#SBATCH --error=logs/clio_strong_%j.err

# ---------------------------------------------------------------------------
# Strong scaling experiment on DeltaAI
#
# This script runs the distributed CLIO cluster with 1, 2, and 4 workers
# against a fixed 2.5M arXiv corpus and records query latency + throughput
# for each configuration.
#
# Strong scaling means ALL phases use the SAME 4 pre-built indices —
# we just vary how many workers are active (1, 2, 4). The indices are
# named clio_shard_0.duckdb ... clio_shard_3.duckdb.
#
# Assumes:
#   - Python venv at $HOME/clio-venv with all CLIO deps installed
#   - iowarp-core wheel built at $HOME/iowarp_core.whl
#   - arXiv corpus pre-sharded into 4 files at ${WORK}/arxiv/arxiv_shard_*.jsonl
#   - Each worker's DuckDB index pre-built at ${WORK}/clio_shard_*.duckdb
#
# Edit the ACCOUNT line before submission. Run as:
#   sbatch slurm_strong_scaling.sh
# ---------------------------------------------------------------------------

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs

WORK="/work/nvme/bekn/sislam3"
module load python/3.11.9
source "${WORK}/clio-venv/bin/activate"

REPO="/u/sislam3/clio-search"
DIST_SCRIPT="$REPO/eval/eval_final/code/delta/distributed_clio.py"
DRIVER_SCRIPT="$REPO/eval/eval_final/code/delta/D1_D6_experiments.py"

# Parse Slurm nodelist
NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
COORDINATOR="${NODES[0]}"
WORKERS=("${NODES[@]:1}")
echo "Coordinator: $COORDINATOR"
echo "Workers: ${WORKERS[@]}"

run_phase() {
    local n_workers="$1"
    local phase_workers=("${WORKERS[@]:0:$n_workers}")
    local worker_urls=""

    echo ""
    echo "=========================================="
    echo "PHASE: $n_workers workers"
    echo "=========================================="

    # Launch workers on their respective nodes.
    # Strong scaling: all phases use the same 4 pre-built indices
    # (clio_shard_0.duckdb .. clio_shard_3.duckdb). We just vary how many
    # workers are active.
    for i in $(seq 0 $((n_workers - 1))); do
        w="${phase_workers[$i]}"
        echo "Launching worker $i on $w"
        srun --nodes=1 --ntasks=1 --nodelist="$w" \
            python3 "$DIST_SCRIPT" worker \
                --port 9201 \
                --shard-id "$i" \
                --total-shards "$n_workers" \
                --db-path "${WORK}/clio_shard_${i}.duckdb" \
                > "logs/worker_${i}_of_${n_workers}_${SLURM_JOB_ID}.log" 2>&1 &
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

    # Launch coordinator on node 0
    echo "Launching coordinator on $COORDINATOR with workers: $worker_urls"
    srun --nodes=1 --ntasks=1 --nodelist="$COORDINATOR" \
        python3 "$DIST_SCRIPT" coordinator \
            --port 9200 \
            --workers "$worker_urls" \
            > "logs/coordinator_${n_workers}_${SLURM_JOB_ID}.log" 2>&1 &

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

    # Run the experiment from the submit node, talking to the coordinator
    export CLIO_COORDINATOR="http://${COORDINATOR}:9200"
    python3 "$DRIVER_SCRIPT" \
        --experiment strong \
        --output-suffix "_${n_workers}workers"

    # D3: distributed indexing throughput — run once when all 4 workers are up
    if [ "$n_workers" -eq 4 ]; then
        echo "Running D3 (distributed indexing throughput)..."
        python3 "$DRIVER_SCRIPT" --experiment indexing
    fi

    # Tear down
    pkill -P $$ -f distributed_clio.py || true
    sleep 5
}

# Run 1-worker, 2-worker, 4-worker phases
run_phase 1
run_phase 2
run_phase 4

echo ""
echo "All phases complete. See eval/eval_final/outputs/D_strong_*.json"
