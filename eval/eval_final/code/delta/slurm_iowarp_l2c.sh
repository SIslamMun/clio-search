#!/bin/bash
# ---------------------------------------------------------------------------
# L2-C: CLIO + IOWarp — NO METADATA case (native build)
#
# Runs each scale in a SEPARATE Python process to avoid Chimaera SHM task
# queue carryover between scales.
#
# Scales: 1K → 5K → 10K → 50K → 100K
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
#SBATCH --time=16:00:00
#SBATCH --output=logs/clio_l2c_nm_%j.out
#SBATCH --error=logs/clio_l2c_nm_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

REPO="/u/sislam3/clio-search"
WORK="/work/nvme/bekn/sislam3"
INNER_SCRIPT="${REPO}/eval/eval_final/code/delta/l2c_iowarp_no_metadata_inner.py"
DB_DIR="${WORK}/l2c_iowarp_db"
ALL_RESULTS_FILE="/tmp/l2c_all_results_${SLURM_JOB_ID}.json"
SCALES="1000 5000 10000 50000 100000"

IOWARP_VENV="${WORK}/iowarp-venv"
IOWARP_LIB="${IOWARP_VENV}/lib/python3.11/site-packages/lib"
IOWARP_PYSITE="${IOWARP_VENV}/lib/python3.11/site-packages"
PYTHON="${IOWARP_VENV}/bin/python3"
CHI_CONF="/u/sislam3/.chimaera/chimaera.yaml"

echo "========================================"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Job: L2-C (1K–100K, no metadata, one process per scale)"
echo "Scales: $SCALES"
echo "========================================"

mkdir -p "$DB_DIR"
echo '{"scales":[]}' > "$ALL_RESULTS_FILE"

chimaera_cleanup() {
    echo "  [cleanup] killing lingering processes..."
    pkill -9 -f "l2_iowarp_inner\|l2c_iowarp_no_metadata_inner\|chimaera_start_runtime" 2>/dev/null || true
    sleep 15
    echo "  [cleanup] removing stale Chimaera SHM segments..."
    rm -f /dev/shm/chi_*_sislam3 /dev/shm/hermes_shm_*_sislam3 2>/dev/null || true
    sleep 15
    echo "  [cleanup] done."
}

echo "Initial cleanup on $(hostname)..."
chimaera_cleanup

for N in $SCALES; do
    chimaera_cleanup
    RAW_FILE="/tmp/l2c_scale_${N}_${SLURM_JOB_ID}.txt"
    echo ""
    echo "========================================"
    echo "Running scale $N  ($(date))"
    echo "========================================"

    CHI_SERVER_CONF="${CHI_CONF}" \
    LD_LIBRARY_PATH="${IOWARP_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    LD_PRELOAD=/usr/lib64/libstdc++.so.6 \
    SCALES_JSON="[$N]" \
    DB_DIR="${DB_DIR}" \
    PYTHONPATH="${IOWARP_PYSITE}:${REPO}/code/src" \
    "${PYTHON}" "${INNER_SCRIPT}" | tee "$RAW_FILE" || {
        echo "WARNING: scale $N exited non-zero, continuing..."
    }

    ALL_RESULTS_FILE="$ALL_RESULTS_FILE" RAW_FILE="$RAW_FILE" N="$N" python3 - <<'PYEOF'
import json, os, sys
from pathlib import Path

raw = Path(os.environ["RAW_FILE"]).read_text()
all_f = Path(os.environ["ALL_RESULTS_FILE"])
N = int(os.environ["N"])

begin = raw.find("===RESULT_JSON_BEGIN===")
end   = raw.find("===RESULT_JSON_END===")
if begin == -1 or end == -1:
    print(f"  WARNING: scale {N:,} produced no result JSON — skipping")
    sys.exit(0)

scale_data = json.loads(raw[begin + len("===RESULT_JSON_BEGIN==="):end].strip())
all_results = json.loads(all_f.read_text())
all_results["scales"].extend(scale_data.get("scales", []))
all_f.write_text(json.dumps(all_results))
print(f"  Merged scale {N:,} → {len(all_results['scales'])} total scale(s) accumulated.")
PYEOF

done

echo ""
echo "All scales done. Saving results..."

ALL_RESULTS_FILE="$ALL_RESULTS_FILE" python3 - <<'PYEOF'
import json, time, os
from pathlib import Path

result = json.loads(Path(os.environ["ALL_RESULTS_FILE"]).read_text())

output = {
    "experiment": "L2-C: CLIO + IOWarp No-Metadata (DeltaAI native)",
    "description": "CLIO reads blob content via GetBlob, parses JSON, extracts measurements — no external metadata",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "iowarp_version": "0.6.4",
    "environment": "native iowarp_core 0.6.4 on DeltaAI GH200 aarch64",
    **result,
}

out_path = Path("/u/sislam3/clio-search/eval/eval_final/outputs/L2C_delta_iowarp_no_metadata.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(output, indent=2))
print(f"Saved: {out_path}")

for s in output["scales"]:
    N = s["N_blobs"]
    print(f"\n  {N:,} blobs (read from CTE, no metadata):")
    print(f"    CTE write:    {s['put_time_s']:.2f}s  ({s['put_throughput_blobs_per_s']:,}/s)")
    print(f"    GetBlob read: {s['getblob_read_time_s']:.2f}s  ({s['getblob_read_throughput_blobs_per_s']:,}/s)")
    print(f"    CLIO index:   {s['clio_index_time_s']:.2f}s  ({s['clio_index_throughput_blobs_per_s']:,}/s)")
    print(f"    Measurements: {s['measurement_count']:,} extracted from raw blob JSON")
    print(f"    Inspection:   {s['inspection_rate_pct']:.4f}% of corpus per query")
    for q in s["queries"]:
        ok = "YES" if q["clio_finds_cross_unit"] else "NO "
        print(f"    [{ok}] {q['query']}: GT={q['ground_truth_matches']}  CLIO={q['clio_scientific_results']}")
PYEOF

echo ""
echo "========================================"
echo "L2-C complete."
echo "========================================"
