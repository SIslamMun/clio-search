#!/bin/bash
# ---------------------------------------------------------------------------
# L2-B LARGE: CLIO + IOWarp — 500K and 1M scales only (native build)
#
# Run after slurm_iowarp_l2.sh (1K–100K) completes.
# Produces: eval/eval_final/outputs/L2_delta_iowarp_large.json
#
# Usage:  sbatch slurm_iowarp_l2_large.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=clio_l2_large
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/clio_l2_large_%j.out
#SBATCH --error=logs/clio_l2_large_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

REPO="/u/sislam3/clio-search"
WORK="/work/nvme/bekn/sislam3"
INNER_SCRIPT="${REPO}/eval/eval_final/code/delta/l2_iowarp_inner.py"
DB_DIR="${WORK}/l2_iowarp_db"
CHECKPOINT_FILE="${WORK}/l2_iowarp_large_checkpoint.json"
SCALES_JSON='[500000, 1000000]'

IOWARP_VENV="${WORK}/iowarp-venv"
IOWARP_LIB="${IOWARP_VENV}/lib/python3.11/site-packages/lib"
IOWARP_PYSITE="${IOWARP_VENV}/lib/python3.11/site-packages"
PYTHON="${IOWARP_VENV}/bin/python3"
CHI_CONF="/u/sislam3/.chimaera/chimaera.yaml"

echo "========================================"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Job: L2-B large scales (500K, 1M)"
echo "Scales: $SCALES_JSON"
echo "========================================"

mkdir -p "$DB_DIR"

echo ""
echo "Running L2-B large scales natively (iowarp_core 0.6.4)..."
echo "  Inner:   $INNER_SCRIPT"
echo "  DB dir:  $DB_DIR"
echo "  Chi conf: $CHI_CONF"
echo ""

CHI_SERVER_CONF="${CHI_CONF}" \
LD_LIBRARY_PATH="${IOWARP_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
LD_PRELOAD=/usr/lib64/libstdc++.so.6 \
SCALES_JSON="${SCALES_JSON}" \
DB_DIR="${DB_DIR}" \
CHECKPOINT_FILE="${CHECKPOINT_FILE}" \
PYTHONPATH="${IOWARP_PYSITE}:${REPO}/code/src" \
"${PYTHON}" "${INNER_SCRIPT}" | tee /tmp/l2_large_raw_output.txt

echo ""
echo "Extracting results..."

CHECKPOINT_FILE="${CHECKPOINT_FILE}" python3 - <<'PYEOF'
import json, sys, time, os
from pathlib import Path

raw = Path("/tmp/l2_large_raw_output.txt").read_text()
begin = raw.find("===RESULT_JSON_BEGIN===")
end   = raw.find("===RESULT_JSON_END===")

if begin != -1 and end != -1:
    result = json.loads(raw[begin + len("===RESULT_JSON_BEGIN==="):end].strip())
    print("Results extracted from stdout markers.")
else:
    chk = os.environ.get("CHECKPOINT_FILE", "")
    if chk and Path(chk).exists():
        result = json.loads(Path(chk).read_text())
        n = len(result.get("scales", []))
        print(f"WARNING: stdout markers missing — loaded checkpoint ({n} scales completed).")
    else:
        print("ERROR: result JSON markers not found and no checkpoint available")
        print("\nLast 50 lines of output:")
        for line in raw.splitlines()[-50:]:
            print(f"  {line}")
        sys.exit(1)

output = {
    "experiment": "L2-B: CLIO + IOWarp CTE Integration — large scales (DeltaAI native)",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "iowarp_version": "0.6.4",
    "environment": "native iowarp_core 0.6.4 on DeltaAI GH200 aarch64",
    **result,
}

out_path = Path("/u/sislam3/clio-search/eval/eval_final/outputs/L2_delta_iowarp_large.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(output, indent=2))
print(f"Saved: {out_path}")

for s in output["scales"]:
    N = s["N_blobs"]
    print(f"\n  {N:,} blobs:")
    print(f"    CTE write:  {s['put_time_s']:.2f}s  ({s['put_throughput_blobs_per_s']:,}/s)")
    print(f"    CLIO index: {s['clio_index_time_s']:.2f}s  ({s['clio_index_throughput_blobs_per_s']:,}/s)")
    print(f"    Inspection: {s['inspection_rate_pct']:.4f}% of corpus per query")
    for q in s["queries"]:
        cross = "YES" if q["clio_finds_cross_unit"] else "NO"
        print(f"    [{cross}] {q['query']}: {q['clio_scientific_results']} CLIO / {q['raw_blobquery_results']} raw")
PYEOF

echo ""
echo "========================================"
echo "L2-B large scales complete."
echo "========================================"
