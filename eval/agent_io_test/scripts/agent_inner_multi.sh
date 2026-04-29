#!/bin/bash
# Per-rank agent runner (multi-dataset) — parameterized version of agent_inner_v2.sh
# Selects argo or era5 paths/prompts via the DATASET env var (default: argo).
set -u

RANK=${PALS_RANKID:-${PMI_RANK:-0}}
HOST=$(hostname)

export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export PATH="$HOME/.opencode/bin:$PATH"

DATASET=${DATASET:-argo}

# --- Dataset-specific paths ---
case "$DATASET" in
    argo)
        DATA_DIR=$BASE/workspaces/arm_a/data
        TEST_INDEX=$BASE/test_index.duckdb
        BLOBS_JSON=$BASE/blobs.json
        GROUNDTRUTH_JSON=$BASE/groundtruth.json
        CLIO_USAGE_SRC=$BASE/workspaces/arm_b/CLIO_USAGE.md
        ;;
    era5)
        DATA_DIR=/home/sislam6/jarvis-work/era5_data/raw
        TEST_INDEX=$BASE/era5_test_index.duckdb
        BLOBS_JSON=$BASE/era5_blobs.json
        GROUNDTRUTH_JSON=$BASE/era5_groundtruth.json
        CLIO_USAGE_SRC=$BASE/workspaces/arm_b_era5/CLIO_USAGE.md
        ;;
    *)
        echo "[rank $RANK] ERROR: unknown DATASET=$DATASET (expected argo or era5)" >&2
        exit 2
        ;;
esac

WORKSPACE=$RUN_DIR/arm_${ARM}/rank_${RANK}
mkdir -p $WORKSPACE/data
for f in $DATA_DIR/*.nc; do
    ln -sf "$f" $WORKSPACE/data/$(basename $f)
done

# --- Per-rank HOME isolation (avoid SQLite session DB collision on Lustre) ---
RANK_HOME=$WORKSPACE/.rank_home
mkdir -p $RANK_HOME/.config/opencode
cp -f $HOME/.config/opencode/opencode.json $RANK_HOME/.config/opencode/ 2>/dev/null
ln -sf $HOME/.opencode/bin $RANK_HOME/.opencode-bin 2>/dev/null
mkdir -p $RANK_HOME/.opencode
export HOME=$RANK_HOME
export PATH="$RANK_HOME/.opencode-bin:$PATH"

# Arm B: copy dataset-specific CLIO_USAGE.md
if [ "$ARM" = "b" ]; then
    cp $CLIO_USAGE_SRC $WORKSPACE/CLIO_USAGE.md
fi

LOG=$WORKSPACE/agent.log
RESULT=$WORKSPACE/result.json
SIF=/home/sislam6/jarvis-work/sif/iowarp_deploy_cpu.sif
VENV=/home/sislam6/jarvis-work/iowarp-venv

CHI_PID=""

# --- Arm B ONLY: start chimaera daemon + populate iowarp ---
if [ "$ARM" = "b" ]; then
    echo "[rank $RANK] starting chimaera daemon for iowarp content store (dataset=$DATASET)" > $LOG

    # Each rank uses a unique chimaera state dir to avoid contention
    export CHI_STATE_DIR=/tmp/chi_rank_${RANK}_$$
    rm -rf $CHI_STATE_DIR; mkdir -p $CHI_STATE_DIR

    # Start daemon (kServer) in apptainer, in background
    apptainer exec $SIF bash -c "
        source $VENV/bin/activate
        chimaera runtime start
    " > $WORKSPACE/chi.log 2>&1 &
    CHI_PID=$!

    # Wait for daemon to be ready
    for i in $(seq 1 30); do
        sleep 1
        if grep -q "Successfully started local server" $WORKSPACE/chi.log 2>/dev/null; then
            echo "[rank $RANK] chimaera up after ${i}s" >> $LOG
            break
        fi
    done

    # Populate iowarp with blob content (kClient connects to running daemon)
    apptainer exec $SIF bash -c "
        source $VENV/bin/activate
        python3 - <<PYEOF
import json
from iowarp_core import wrp_cte_core_ext as cte
cte.chimaera_init(cte.ChimaeraMode.kClient)
cte.initialize_cte('', cte.PoolQuery.Dynamic())
blobs = json.load(open('$BLOBS_JSON'))
n = 0
for uri, text in blobs.items():
    rest = uri[len('cte://'):]
    tag_name, blob_name = rest.split('/', 1)
    cte.Tag(tag_name).PutBlob(blob_name, text.encode())
    n += 1
print(f'[rank $RANK] populated {n} blobs into iowarp')
PYEOF
    " >> $LOG 2>&1
fi

# --- Dataset-specific task prompts ---
TASK_A_ARGO="Count how many Argo profile NetCDF files in the data/ directory have a surface temperature above 30 degrees Celsius. Each .nc file is one profile (NetCDF-3 classic format). Use Python with the netCDF4 module. Print only the final count number."

TASK_B_ARGO="Your current working directory contains a file named CLIO_USAGE.md (use 'cat CLIO_USAGE.md' to read it). It explains how to use CLIO+IOWarp from Python in this workspace. The summary: write a Python script that connects to the running chimaera daemon via cte.chimaera_init(cte.ChimaeraMode.kClient), opens DuckDBStorage at ${TEST_INDEX}, instantiates IOWarpConnector(namespace='agent_test', tag_pattern='argo_.*', blob_pattern='.*'), then calls conn.search_scientific_with_content(query='temperature above 86 degF', top_k=1000, operators=ScientificQueryOperators(numeric_range=NumericRangeOperator(minimum=86.0, maximum=None, unit='degF'))). Run that Python via 'apptainer exec /home/sislam6/jarvis-work/sif/iowarp_deploy_cpu.sif bash -c \"source /home/sislam6/jarvis-work/iowarp-venv/bin/activate && python3 your_script.py\"'. Print only the count of matches as a single integer on the last line."

TASK_A_ERA5="Count how many ERA5 grid cells in the NetCDF files in the data/ directory have a 2 metre temperature mean exceeding 30 degrees Celsius. The files are ERA5 single-level reanalysis (one per month) and contain the variable 't2m' (units: kelvin). For each file, compute the mean of t2m over the time dimension at every (latitude, longitude) cell, then count cells whose mean exceeds 303.15 K (== 30 degC). Sum that count across all files and print only the final integer count. Use Python with the netCDF4 module."

TASK_B_ERA5="Your current working directory contains a file named CLIO_USAGE.md (use 'cat CLIO_USAGE.md' to read it). It explains how to use CLIO+IOWarp from Python in this workspace. The summary: write a Python script that connects to the running chimaera daemon via cte.chimaera_init(cte.ChimaeraMode.kClient), opens DuckDBStorage at ${TEST_INDEX}, instantiates IOWarpConnector(namespace='agent_test', tag_pattern='era5_.*', blob_pattern='.*'), then calls conn.search_scientific_with_content(query='temperature above 86 degF', top_k=1000, operators=ScientificQueryOperators(numeric_range=NumericRangeOperator(minimum=86.0, maximum=None, unit='degF'))). Run that Python via 'apptainer exec /home/sislam6/jarvis-work/sif/iowarp_deploy_cpu.sif bash -c \"source /home/sislam6/jarvis-work/iowarp-venv/bin/activate && python3 your_script.py\"'. Print only the count of matches as a single integer on the last line."

if [ "$DATASET" = "argo" ]; then
    TASK_A="$TASK_A_ARGO"; TASK_B="$TASK_B_ARGO"
else
    TASK_A="$TASK_A_ERA5"; TASK_B="$TASK_B_ERA5"
fi

if [ "$ARM" = "a" ]; then TASK="$TASK_A"; else TASK="$TASK_B"; fi

cd $WORKSPACE
T0=$(date +%s.%N)
echo "[rank $RANK on $HOST arm=$ARM dataset=$DATASET] start $(date -Iseconds)" >> $LOG

# Run opencode
timeout 300 opencode run -m alcf_sophia/openai/gpt-oss-120b "$TASK" >> $LOG 2>&1
RC=$?

T1=$(date +%s.%N)
WALL=$(awk "BEGIN {print $T1 - $T0}")

ANSWER=$(grep -oE '\b[0-9]+\b' $LOG | tail -1)
DATA_BYTES=$(du -sb $WORKSPACE/data 2>/dev/null | awk '{print $1}')

# Did the agent actually call iowarp? Look for "fetch_blob_content" or "search_scientific_with_content" in the log
IOWARP_CALLS=$(grep -cE "fetch_blob_content|search_scientific_with_content|Tag\.GetBlob|cte\.Tag" $LOG 2>/dev/null || echo "0")

cat > $RESULT <<EOF
{
  "rank": $RANK,
  "host": "$HOST",
  "arm": "$ARM",
  "dataset": "$DATASET",
  "wall_seconds": $WALL,
  "exit_code": $RC,
  "answer": "${ANSWER:-null}",
  "data_dir_bytes": ${DATA_BYTES:-0},
  "iowarp_call_evidence_lines": ${IOWARP_CALLS:-0}
}
EOF

# Cleanup chimaera daemon if Arm B
if [ -n "$CHI_PID" ]; then
    kill -TERM $CHI_PID 2>/dev/null
    wait $CHI_PID 2>/dev/null
    rm -rf $CHI_STATE_DIR 2>/dev/null
fi
rm -f /tmp/chimaera_*.ipc /tmp/chi_main_segment_* 2>/dev/null

echo "[rank $RANK] done wall=${WALL}s answer=${ANSWER} iowarp_evidence=${IOWARP_CALLS} dataset=${DATASET}" >> $LOG
exit $RC
