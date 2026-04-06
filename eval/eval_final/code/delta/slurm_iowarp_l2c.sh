#!/bin/bash
# ---------------------------------------------------------------------------
# L2-C: CLIO + IOWarp — NO METADATA case
#
# Proves CLIO can search blob content when there is NO external metadata.
# CLIO reads blobs from CTE via GetBlob, parses JSON content, extracts
# scientific measurements, and answers cross-unit queries — all without
# any metadata being passed at write time.
#
# Scales: 1K → 5K → 10K → 50K → 100K → 500K → 1M
# Produces: eval/eval_final/outputs/L2C_delta_iowarp_no_metadata.json
#
# Usage:  sbatch slurm_iowarp_l2c.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=clio_l2c_nm
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/clio_l2c_nm_%j.out
#SBATCH --error=logs/clio_l2c_nm_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

REPO="/u/sislam3/clio-search"
WORK="/work/nvme/bekn/sislam3"
WHEEL="${REPO}/eval/eval_eve/iowarp_core-1.0.3-cp312-cp312-linux_x86_64.whl"
INNER_SCRIPT="${REPO}/eval/eval_final/code/delta/l2c_iowarp_no_metadata_inner.py"
SIF="${WORK}/iowarp-deps-cpu.sif"
DB_DIR="${WORK}/l2c_iowarp_db"
SCALES_JSON='[1000, 5000, 10000, 50000, 100000, 500000, 1000000]'

echo "========================================"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Job: L2-C (no-metadata blob search)"
echo "Scales: $SCALES_JSON"
echo "========================================"

# ---------- Step 1: Apptainer SIF ----------
module load apptainer 2>/dev/null || true

if [ ! -f "$SIF" ]; then
    echo "Pulling iowarp/deps-cpu:latest -> $SIF (may take ~5 min)..."
    apptainer pull "$SIF" docker://iowarp/deps-cpu:latest
else
    echo "SIF already exists: $SIF"
fi

mkdir -p "$DB_DIR"

# ---------- Step 2: Run inside Apptainer ----------
echo ""
echo "Running L2-C inside Apptainer..."
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
        export PATH=/home/iowarp/venv/bin:\$PATH
        pip install /wheels/${WHEEL_NAME} --force-reinstall -q 2>&1 | tail -3
        pip install duckdb -q 2>&1 | tail -1
        python3 /test.py
    " | tee /tmp/l2c_raw_output.txt

# ---------- Step 3: Extract + save JSON ----------
echo ""
echo "Extracting results..."

python3 - <<'PYEOF'
import json, sys, time
from pathlib import Path

raw = Path("/tmp/l2c_raw_output.txt").read_text()
begin = raw.find("===RESULT_JSON_BEGIN===")
end   = raw.find("===RESULT_JSON_END===")

if begin == -1 or end == -1:
    print("ERROR: result JSON markers not found in output")
    print("\nLast 50 lines of output:")
    for line in raw.splitlines()[-50:]:
        print(f"  {line}")
    sys.exit(1)

result = json.loads(raw[begin + len("===RESULT_JSON_BEGIN==="):end].strip())

output = {
    "experiment": "L2-C: CLIO + IOWarp No-Metadata (DeltaAI)",
    "description": "CLIO reads blob content via GetBlob, parses JSON, extracts measurements — no external metadata",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "iowarp_version": "1.0.3",
    "environment": "Apptainer iowarp/deps-cpu:latest on DeltaAI GH200",
    **result,
}

out_path = Path("/u/sislam3/clio-search/eval/eval_final/outputs/L2C_delta_iowarp_no_metadata.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(output, indent=2))
print(f"Saved: {out_path}")

print("\n" + "="*70)
print("L2-C RESULTS SUMMARY")
print("="*70)
for s in output["scales"]:
    N = s["N_blobs"]
    print(f"\n  {N:,} blobs (read from CTE, no metadata):")
    print(f"    CTE write:    {s['put_time_s']:.2f}s  ({s['put_throughput_blobs_per_s']:,}/s)")
    print(f"    CLIO index:   {s['clio_index_time_s']:.2f}s  ({s['clio_index_throughput_blobs_per_s']:,}/s)")
    print(f"    Measurements: {s['measurement_count']:,} extracted from raw blob JSON")
    print(f"    Units found:  {s['distinct_units']}")
    print(f"    Inspection:   {s['inspection_rate_pct']:.4f}% of corpus per query")
    for q in s["queries"]:
        ok = "YES" if q["clio_finds_cross_unit"] else "NO "
        print(
            f"    [{ok}] {q['query']}: "
            f"GT={q['ground_truth_matches']}  CLIO={q['clio_scientific_results']}  "
            f"BlobQuery={q['raw_blobquery_results']} (unfiltered)"
        )

# Key finding
if output["scales"]:
    last = output["scales"][-1]
    all_found = all(q["clio_finds_cross_unit"] for q in last["queries"])
    print(f"\nKEY FINDING at {last['N_blobs']:,} blobs:")
    print(f"  CLIO parsed {last['measurement_count']:,} measurements from raw JSON blobs")
    print(f"  Cross-unit queries succeeded: {all_found}")
    print(f"  BlobQuery returns ALL blobs (no filtering) — CLIO filters to {last['inspection_rate_pct']:.4f}%")
PYEOF

echo ""
echo "========================================"
echo "L2-C complete."
echo "========================================"
