#!/bin/bash
# Multi-prompt agent runner v3 — same infra as v2 but each rank gets a
# different scientific question from the 3-prompt grid (prompts_grid.json).
# Arms A/B/C are multi-prompt; arm D keeps its single TASK_D for now.
set -u

RANK=${PALS_RANKID:-${PMI_RANK:-0}}
HOST=$(hostname)

export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export PATH="$HOME/.opencode/bin:$PATH"

WORKSPACE=$RUN_DIR/arm_${ARM}/rank_${RANK}
mkdir -p $WORKSPACE
lfs setstripe -p gecko.ddn_hdd -E -1 -c 2 $WORKSPACE 2>/dev/null || true
mkdir -p $WORKSPACE/data
for f in $DATA_DIR/*.nc; do
    ln -sf "$f" $WORKSPACE/data/$(basename $f)
done

# Per-rank fresh HOME on ddn_hdd
RANK_HOME=$WORKSPACE/.rank_home
mkdir -p $RANK_HOME
lfs setstripe -p gecko.ddn_hdd -E -1 -c 2 $RANK_HOME 2>/dev/null || true
mkdir -p $RANK_HOME/.config/opencode
cp -f $HOME/.config/opencode/opencode.json $RANK_HOME/.config/opencode/ 2>/dev/null
ln -sf $HOME/.opencode/bin $RANK_HOME/.opencode-bin 2>/dev/null
mkdir -p $RANK_HOME/.opencode
export HOME=$RANK_HOME
export PATH="$RANK_HOME/.opencode-bin:$PATH"

if [ "$ARM" = "b" ] || [ "$ARM" = "c" ]; then
    cp $BASE/workspaces/arm_b/CLIO_USAGE.md $WORKSPACE/CLIO_USAGE.md 2>/dev/null
fi

LOG=$WORKSPACE/agent.log
RESULT=$WORKSPACE/result.json
SIF=/home/sislam6/jarvis-work/sif/iowarp_deploy_cpu.sif
VENV=/home/sislam6/jarvis-work/iowarp-venv
BLOBS_JSON=$BASE/blobs_unified.json

# ----- Prompt grid (parallel arrays) -----
# Same content as $BASE/prompts_grid.json but inlined to avoid jq dependency
# and shell-escaping issues with complex nested quoting.
PROMPT_NAMES=(
    "surface_temp_gt_30"
    "surface_temp_gt_25"
    "any_depth_temp_gt_25"
)
PROMPT_NL=(
    "Count how many Argo profile NetCDF files in the data/ directory have a surface temperature above 30 degrees Celsius. Each .nc file is one profile (NetCDF-3 classic format). Use Python with the netCDF4 module. Print only the final count number."
    "Count how many Argo profile NetCDF files in the data/ directory have a surface temperature above 25 degrees Celsius. Each .nc file is one profile (NetCDF-3 classic format). Use Python with the netCDF4 module. Print only the final count number."
    "Count how many Argo profile NetCDF files have a temperature above 25 degrees Celsius at any depth in the profile. Each .nc file is one profile (NetCDF-3 classic format). Use Python with the netCDF4 module. Print only the final count number."
)
# CLIO query operators per prompt (as numeric strings for shell substitution)
PROMPT_MIN=(  "30.0" "25.0" "25.0" )
PROMPT_UNIT=( "degC" "degC" "degC" )
PROMPT_QTEXT=(
    "temperature above 30 degC"
    "temperature above 25 degC"
    "temperature above 25 degC any depth"
)

NUM_PROMPTS=${#PROMPT_NAMES[@]}
PROMPT_ID=$(( RANK % NUM_PROMPTS ))
PROMPT_NAME="${PROMPT_NAMES[$PROMPT_ID]}"
PROMPT_NL_VAL="${PROMPT_NL[$PROMPT_ID]}"
PROMPT_MIN_VAL="${PROMPT_MIN[$PROMPT_ID]}"
PROMPT_UNIT_VAL="${PROMPT_UNIT[$PROMPT_ID]}"
PROMPT_QTEXT_VAL="${PROMPT_QTEXT[$PROMPT_ID]}"

CHI_PID=""

# Arm C — chimaera daemon (unchanged from v2)
if [ "$ARM" = "c" ]; then
    echo "[rank $RANK] starting chimaera daemon" > $LOG
    export CHI_STATE_DIR=/tmp/chi_rank_${RANK}_$$
    rm -rf $CHI_STATE_DIR; mkdir -p $CHI_STATE_DIR
    apptainer exec $SIF bash -c "
        source $VENV/bin/activate
        chimaera runtime start
    " > $WORKSPACE/chi.log 2>&1 &
    CHI_PID=$!
    for i in $(seq 1 30); do
        sleep 1
        if grep -q "Successfully started local server" $WORKSPACE/chi.log 2>/dev/null; then
            echo "[rank $RANK] chimaera up after ${i}s" >> $LOG
            break
        fi
    done
    apptainer exec $SIF bash -c "
        source $VENV/bin/activate
        python3 - <<PYEOF
import json, os
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
os._exit(0)
PYEOF
    " >> $LOG 2>&1
fi

# ----- Build TASK per (arm, prompt) -----
TASK_A="$PROMPT_NL_VAL"

TASK_B="You MUST do TWO things in order: (1) Write a Python script ./q.py with the code below. (2) IMMEDIATELY EXECUTE it via the apptainer command and report the integer it prints. Both steps are required — do not stop after writing the file. \
\
The exact code for ./q.py: \
import sys, shutil, tempfile; sys.path.insert(0, '/home/sislam6/clio-search/code/src'); \
from pathlib import Path; \
from clio_agentic_search.connectors.iowarp.connector import IOWarpConnector; \
from clio_agentic_search.storage.duckdb_store import DuckDBStorage; \
from clio_agentic_search.retrieval.scientific import ScientificQueryOperators, NumericRangeOperator; \
src = Path('/home/sislam6/clio-search/eval/agent_io_test/test_index_unified.duckdb'); \
tmp = Path(tempfile.gettempdir()) / f'idx_{Path.cwd().name}.duckdb'; \
shutil.copyfile(src, tmp); \
store = DuckDBStorage(tmp); \
conn = IOWarpConnector(namespace='agent_test', storage=store, tag_pattern='argo_.*', blob_pattern='.*'); \
conn.connect(); \
op = ScientificQueryOperators(numeric_range=NumericRangeOperator(minimum=${PROMPT_MIN_VAL}, maximum=None, unit='${PROMPT_UNIT_VAL}')); \
hits = conn.search_scientific(query='${PROMPT_QTEXT_VAL}', top_k=10000, operators=op); \
print(len(hits)) \
\
DO NOT call cte.chimaera_init or initialize_cte — this arm uses CLIO's DuckDB index only. \
\
After writing the file, RUN this exact command and read the output: \
apptainer exec /home/sislam6/jarvis-work/sif/iowarp_deploy_cpu.sif bash -c 'source /home/sislam6/jarvis-work/iowarp-venv/bin/activate && python3 ./q.py' \
\
Print only the integer it prints, on the very last line of your final response."

TASK_C="You MUST do TWO things in order: (1) Write a Python script ./q.py with the code below. (2) IMMEDIATELY EXECUTE it via the apptainer command and report the integer it prints. Both steps are required — do not stop after writing the file. \
\
The exact code for ./q.py: \
import sys, shutil, tempfile; sys.path.insert(0, '/home/sislam6/clio-search/code/src'); \
from pathlib import Path; \
from clio_agentic_search.connectors.iowarp.connector import IOWarpConnector; \
from clio_agentic_search.storage.duckdb_store import DuckDBStorage; \
from clio_agentic_search.retrieval.scientific import ScientificQueryOperators, NumericRangeOperator; \
src = Path('/home/sislam6/clio-search/eval/agent_io_test/test_index_unified.duckdb'); \
tmp = Path(tempfile.gettempdir()) / f'idx_{Path.cwd().name}.duckdb'; \
shutil.copyfile(src, tmp); \
store = DuckDBStorage(tmp); \
conn = IOWarpConnector(namespace='agent_test', storage=store, tag_pattern='argo_.*', blob_pattern='.*'); \
conn.connect(); \
op = ScientificQueryOperators(numeric_range=NumericRangeOperator(minimum=${PROMPT_MIN_VAL}, maximum=None, unit='${PROMPT_UNIT_VAL}')); \
res = conn.search_scientific_with_content(query='${PROMPT_QTEXT_VAL}', top_k=10000, operators=op); \
print(len(res)) \
\
DO NOT call cte.chimaera_init or initialize_cte directly — search_scientific_with_content fetches IOWarp content via an internal subprocess. \
\
After writing the file, RUN this exact command and read the output: \
apptainer exec /home/sislam6/jarvis-work/sif/iowarp_deploy_cpu.sif bash -c 'source /home/sislam6/jarvis-work/iowarp-venv/bin/activate && python3 ./q.py' \
\
Print only the integer it prints, on the very last line of your final response."

# Arm D: unchanged single-prompt federated (post-fix, output silenced)
TASK_D="Run this exact command and report only the integer it prints: \
apptainer exec /home/sislam6/jarvis-work/sif/iowarp_deploy_cpu.sif bash -c 'source /home/sislam6/jarvis-work/iowarp-venv/bin/activate && python3 /home/sislam6/jarvis-work/federated_query.py --shards /home/sislam6/jarvis-work/shards/argo_20260428_183628 --dataset argo --out ./fed.json > /dev/null 2>&1 && python3 -c \"import json; print(json.load(open(\\\"./fed.json\\\"))[\\\"queries\\\"][0][\\\"global_top_k_count\\\"])\"' \
Print only the integer it prints on the very last line of your final response."

case "$ARM" in
  a) TASK="$TASK_A" ;;
  b) TASK="$TASK_B" ;;
  c) TASK="$TASK_C" ;;
  d) TASK="$TASK_D"; PROMPT_NAME="single_federated" ;;
  *) echo "Unknown ARM=$ARM" >&2; exit 2 ;;
esac

cd $WORKSPACE
printf '%s\n' "$TASK" > $WORKSPACE/prompt.txt

cat > $WORKSPACE/meta.json <<META
{
  "rank": $RANK,
  "host": "$HOST",
  "arm": "$ARM",
  "prompt_id": $PROMPT_ID,
  "prompt_name": "$PROMPT_NAME",
  "model": "alcf_sophia/openai/gpt-oss-120b",
  "agent_framework": "opencode",
  "timeout_s": 300,
  "data_dir": "$DATA_DIR",
  "data_dir_files": $(ls $DATA_DIR | wc -l),
  "workspace": "$WORKSPACE",
  "started_at": "$(date -Iseconds)"
}
META

T0=$(date +%s.%N)
echo "[rank $RANK on $HOST arm=$ARM prompt=$PROMPT_NAME] start $(date -Iseconds)" >> $LOG
echo "===== BEGIN OPENCODE RAW OUTPUT =====" >> $LOG

sleep $(( RANK * 2 ))

RC=0
for attempt in 1 2 3; do
    ATTEMPT_LOG=$WORKSPACE/.attempt_${attempt}.log
    timeout 300 opencode run -m alcf_sophia/openai/gpt-oss-120b "$TASK" \
        > $ATTEMPT_LOG 2>&1
    RC=$?
    cat $ATTEMPT_LOG >> $LOG
    if [ $RC -ne 0 ] || grep -qE "Expected 'id' to be a string|StreamingTimeoutError|\"code\":504|Unknown role: commentary" $ATTEMPT_LOG; then
        echo "[rank $RANK] attempt $attempt failed (exit=$RC); backing off" >> $LOG
        rm -f $ATTEMPT_LOG
        if [ $attempt -lt 3 ]; then
            sleep $(( attempt * 5 ))
            continue
        fi
    fi
    rm -f $ATTEMPT_LOG
    break
done

echo "===== END OPENCODE RAW OUTPUT (exit=$RC) =====" >> $LOG

T1=$(date +%s.%N)
WALL=$(awk "BEGIN {print $T1 - $T0}")

ANSWER=$(awk '/===== BEGIN OPENCODE RAW OUTPUT =====/{f=1; next} /===== END OPENCODE RAW OUTPUT/{f=0} f' $LOG \
  | sed 's/\x1b\[[0-9;]*m//g' \
  | grep -E '^[[:space:]]*[0-9]+[[:space:]]*$' \
  | tail -1 \
  | tr -d '[:space:]')
DATA_BYTES=$(du -sb $WORKSPACE/data 2>/dev/null | awk '{print $1}')

tail -100 $LOG > $WORKSPACE/final_output.txt

IOWARP_CALLS=$(grep -cE "fetch_blob_content|search_scientific_with_content|Tag\.GetBlob|cte\.Tag" $LOG 2>/dev/null | head -1)
IOWARP_CALLS=${IOWARP_CALLS:-0}

cat > $RESULT <<EOF
{
  "rank": $RANK,
  "host": "$HOST",
  "arm": "$ARM",
  "prompt_id": $PROMPT_ID,
  "prompt_name": "$PROMPT_NAME",
  "wall_seconds": $WALL,
  "exit_code": $RC,
  "answer": "${ANSWER:-null}",
  "data_dir_bytes": ${DATA_BYTES:-0},
  "iowarp_call_evidence_lines": ${IOWARP_CALLS:-0}
}
EOF

if [ -n "$CHI_PID" ]; then
    kill -TERM $CHI_PID 2>/dev/null
    wait $CHI_PID 2>/dev/null
    rm -rf $CHI_STATE_DIR 2>/dev/null
fi
rm -f /tmp/chimaera_*.ipc /tmp/chi_main_segment_* 2>/dev/null

echo "[rank $RANK] done wall=${WALL}s answer=${ANSWER} prompt=${PROMPT_NAME} iowarp_evidence=${IOWARP_CALLS}" >> $LOG
exit 0
