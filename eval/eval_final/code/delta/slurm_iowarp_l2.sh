#!/bin/bash
# ---------------------------------------------------------------------------
# L2-B: CLIO + IOWarp CTE Integration — DeltaAI job
#
# Runs the CLIO+IOWarp integration test (cross-unit search on IOWarp blobs)
# at scales 10K, 50K, 100K using Apptainer on a single GH200 node.
#
# Produces: eval/eval_final/outputs/L2_delta_iowarp.json
#
# Usage:  sbatch slurm_iowarp_l2.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=clio_l2_iowarp
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/clio_l2_iowarp_%j.out
#SBATCH --error=logs/clio_l2_iowarp_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

REPO="/u/sislam3/clio-search"
WORK="/work/nvme/bekn/sislam3"
WHEEL="${REPO}/eval/eval_eve/iowarp_core-1.0.3-cp312-cp312-linux_x86_64.whl"
INNER_SCRIPT="${REPO}/eval/eval_final/code/delta/l2_iowarp_inner.py"
SIF="${WORK}/iowarp-deps-cpu.sif"
DB_DIR="${WORK}/l2_iowarp_db"
SCALES_JSON='[1000, 5000, 10000, 50000, 100000, 500000, 1000000]'

echo "========================================"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Scales: $SCALES_JSON"
echo "========================================"

# ---------- Step 1: Apptainer SIF ----------
module load apptainer 2>/dev/null || true

if [ ! -f "$SIF" ]; then
    echo "Pulling iowarp/deps-cpu:latest -> $SIF (this may take ~5 min)..."
    apptainer pull "$SIF" docker://iowarp/deps-cpu:latest
else
    echo "SIF already exists: $SIF"
fi

mkdir -p "$DB_DIR"

# ---------- Step 2: Run inside Apptainer ----------
echo ""
echo "Running L2-B inside Apptainer..."
echo "  Wheel:  $WHEEL"
echo "  Inner:  $INNER_SCRIPT"
echo "  DB dir: $DB_DIR"
echo ""

WHEEL_NAME=$(basename "$WHEEL")

apptainer exec \
    --writable-tmpfs \
    --env "SCALES_JSON=${SCALES_JSON}" \
    --env "DB_DIR=/dbdir" \
    --bind "${WHEEL}:/wheels/${WHEEL_NAME}:ro" \
    --bind "${INNER_SCRIPT}:/test.py:ro" \
    --bind "${REPO}/code/src:/clio/src:ro" \
    --bind "${DB_DIR}:/dbdir" \
    "$SIF" \
    bash -c "
        export PATH=/home/iowarp/venv/bin:/home/iowarp/.local/bin:/usr/local/bin:/opt/conda/bin:\$PATH
        pip install /wheels/${WHEEL_NAME} --force-reinstall -q 2>&1 | tail -3
        pip install duckdb -q 2>&1 | tail -1
        python3 /test.py
    " | tee /tmp/l2_raw_output.txt

# ---------- Step 3: Extract + save JSON ----------
echo ""
echo "Extracting results..."

python3 - <<'PYEOF'
import json, sys, time
from pathlib import Path

raw = Path("/tmp/l2_raw_output.txt").read_text()
begin = raw.find("===RESULT_JSON_BEGIN===")
end   = raw.find("===RESULT_JSON_END===")

if begin == -1 or end == -1:
    print("ERROR: result JSON markers not found in output")
    sys.exit(1)

result = json.loads(raw[begin + len("===RESULT_JSON_BEGIN==="):end].strip())

output = {
    "experiment": "L2-B: CLIO + IOWarp CTE Integration (DeltaAI)",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "iowarp_version": "1.0.3",
    "environment": "Apptainer iowarp/deps-cpu:latest on DeltaAI GH200",
    **result,
}

out_path = Path("/u/sislam3/clio-search/eval/eval_final/outputs/L2_delta_iowarp.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(output, indent=2))
print(f"Saved: {out_path}")

# Summary
for s in output["scales"]:
    N = s["N_blobs"]
    print(f"\n  {N:,} blobs:")
    print(f"    CTE write:  {s['put_time_s']:.2f}s  ({s['put_throughput_blobs_per_s']:,}/s)")
    print(f"    CLIO index: {s['clio_index_time_s']:.2f}s  ({s['clio_index_throughput_blobs_per_s']:,}/s)")
    print(f"    Inspection: {s['inspection_rate_pct']:.4f}% of corpus per query")
    for q in s["queries"]:
        cross = "YES" if q["clio_finds_cross_unit"] else "NO"
        print(f"    [{cross}] {q['query']}: {q['clio_scientific_results']} CLIO / {q['raw_blobquery_results']} BlobQuery")
PYEOF

echo ""
echo "========================================"
echo "L2-B complete."
echo "========================================"
