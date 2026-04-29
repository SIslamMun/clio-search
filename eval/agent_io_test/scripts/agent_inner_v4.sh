#!/bin/bash
# Multi-prompt + multi-dataset agent runner v4.
#
# Selects:
#   DATASET (env): argo | era5  -- which corpus this run uses
#   ARM (env):     a | b | c | d
#
# Prompts: 3 per dataset, picked round-robin by rank index.
#
# Differences from v3:
#   * Arm A's prompt is now in "MUST execute" form (forces opencode's bash
#     tool to run the script, removing the failure mode where the LLM writes
#     a tutorial-style markdown response and never invokes anything).
#     The surface-vs-any-depth interpretation is left to the agent — that's
#     the variable we're measuring.
#   * DATASET=era5 swaps the index path, namespace, tag pattern, prompt list,
#     and data dir; arm A is currently disabled for ERA5 (different scaling
#     concern; left for a future iteration).

set -u

DATASET="${DATASET:-argo}"
RANK=${PALS_RANKID:-${PMI_RANK:-0}}
HOST=$(hostname)

if [ "$DATASET" = "era5" ] && [ "$ARM" = "a" ]; then
    echo "[rank $RANK] arm=A on ERA5 not implemented in this iteration; exiting cleanly" >&2
    exit 0
fi

export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export PATH="$HOME/.opencode/bin:$PATH"

# ----- per-dataset config -----
case "$DATASET" in
    argo)
        DATA_DIR_OVERRIDE="$BASE/workspaces/arm_a/data"     # 200 Argo NetCDFs
        INDEX_PATH="/home/sislam6/clio-search/eval/agent_io_test/test_index_unified.duckdb"
        BLOBS_JSON="$BASE/blobs_unified.json"
        NS="agent_test"
        TAG_PATTERN="argo_.*"
        PROMPT_NAMES=( "surface_temp_gt_30" "surface_temp_gt_25" "any_depth_temp_gt_25" )
        PROMPT_NL_A=(  # naive natural-language version (with force-execute wrapper appended in TASK_A construction)
            "how many Argo profile NetCDF files in the data/ directory have a surface temperature above 30 degrees Celsius"
            "how many Argo profile NetCDF files in the data/ directory have a surface temperature above 25 degrees Celsius"
            "how many Argo profile NetCDF files have a temperature above 25 degrees Celsius at any depth in the profile"
        )
        PROMPT_MIN=(  "30.0" "25.0" "25.0" )
        PROMPT_MAX=(  "None" "None" "None" )
        PROMPT_UNIT=( "degC" "degC" "degC" )
        PROMPT_QTEXT=( "temperature above 30 degC" "temperature above 25 degC" "temperature above 25 degC any depth" )
        ;;
    era5)
        # Arm A on ERA5 not implemented; only B/C use these.
        DATA_DIR_OVERRIDE="$DATA_DIR"  # whatever the launcher set
        INDEX_PATH="/home/sislam6/clio-search/eval/agent_io_test/era5_test_index_unified.duckdb"
        BLOBS_JSON="/tmp/era5_blobs_unified.json"  # only at /tmp, not eval (EDQUOT)
        NS="agent_test_era5"
        TAG_PATTERN="era5_.*"
        PROMPT_NAMES=( "any_t2m_above_30C" "any_t2m_above_35C" "low_pressure_lt_95000Pa" )
        PROMPT_NL_A=( "" "" "" )  # no naive arm for ERA5 in this iteration
        PROMPT_MIN=(  "30.0" "35.0" "None" )
        PROMPT_MAX=(  "None" "None" "95000.0" )
        PROMPT_UNIT=( "degC" "degC" "Pa" )
        PROMPT_QTEXT=( "temperature above 30 degC" "temperature above 35 degC" "surface pressure below 95000 Pa" )
        ;;
    *)
        echo "Unknown DATASET=$DATASET" >&2; exit 2 ;;
esac

NUM_PROMPTS=${#PROMPT_NAMES[@]}
PROMPT_ID=$(( RANK % NUM_PROMPTS ))
PROMPT_NAME="${PROMPT_NAMES[$PROMPT_ID]}"
PROMPT_NL_A_VAL="${PROMPT_NL_A[$PROMPT_ID]}"
PROMPT_MIN_VAL="${PROMPT_MIN[$PROMPT_ID]}"
PROMPT_MAX_VAL="${PROMPT_MAX[$PROMPT_ID]}"
PROMPT_UNIT_VAL="${PROMPT_UNIT[$PROMPT_ID]}"
PROMPT_QTEXT_VAL="${PROMPT_QTEXT[$PROMPT_ID]}"

# ----- workspace + isolation -----
WORKSPACE=$RUN_DIR/arm_${ARM}/rank_${RANK}
mkdir -p $WORKSPACE
lfs setstripe -p gecko.ddn_hdd -E -1 -c 2 $WORKSPACE 2>/dev/null || true
mkdir -p $WORKSPACE/data
for f in $DATA_DIR_OVERRIDE/*.nc; do
    ln -sf "$f" $WORKSPACE/data/$(basename $f) 2>/dev/null
done

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

CHI_PID=""

# Arm C — start chimaera daemon + populate iowarp content store
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

# Arm A — TIGHTENED: force-execute framing matching B/C, but the agent still
# has to pick how to interpret 'surface' / 'any depth' from the NL question.
TASK_A="You MUST do TWO things in order: (1) Write a Python script ./scan.py that reads the NetCDF files in data/ using the netCDF4 module and counts ${PROMPT_NL_A_VAL}. (2) IMMEDIATELY EXECUTE the script via the apptainer command and report the integer it prints. Both steps required — do not stop after writing the file. \
\
After writing the file, RUN this exact command and read the output: \
apptainer exec /home/sislam6/jarvis-work/sif/iowarp_deploy_cpu.sif bash -c 'source /home/sislam6/jarvis-work/iowarp-venv/bin/activate && python3 ./scan.py' \
\
Print only the integer it prints, on the very last line of your final response."

# Arm B — CLIO + DuckDB only
TASK_B="You MUST do TWO things in order: (1) Write a Python script ./q.py with the code below. (2) IMMEDIATELY EXECUTE it via the apptainer command and report the integer it prints. Both steps are required — do not stop after writing the file. \
\
The exact code for ./q.py: \
import sys, shutil, tempfile; sys.path.insert(0, '/home/sislam6/clio-search/code/src'); \
from pathlib import Path; \
from clio_agentic_search.connectors.iowarp.connector import IOWarpConnector; \
from clio_agentic_search.storage.duckdb_store import DuckDBStorage; \
from clio_agentic_search.retrieval.scientific import ScientificQueryOperators, NumericRangeOperator; \
src = Path('${INDEX_PATH}'); \
tmp = Path(tempfile.gettempdir()) / f'idx_{Path.cwd().name}.duckdb'; \
shutil.copyfile(src, tmp); \
store = DuckDBStorage(tmp); \
conn = IOWarpConnector(namespace='${NS}', storage=store, tag_pattern='${TAG_PATTERN}', blob_pattern='.*'); \
conn.connect(); \
op = ScientificQueryOperators(numeric_range=NumericRangeOperator(minimum=${PROMPT_MIN_VAL}, maximum=${PROMPT_MAX_VAL}, unit='${PROMPT_UNIT_VAL}')); \
hits = conn.search_scientific(query='${PROMPT_QTEXT_VAL}', top_k=10000, operators=op); \
print(len(hits)) \
\
DO NOT call cte.chimaera_init or initialize_cte — this arm uses CLIO's DuckDB index only. \
\
After writing the file, RUN this exact command and read the output: \
apptainer exec /home/sislam6/jarvis-work/sif/iowarp_deploy_cpu.sif bash -c 'source /home/sislam6/jarvis-work/iowarp-venv/bin/activate && python3 ./q.py' \
\
Print only the integer it prints, on the very last line of your final response."

# Arm C — CLIO + IOWarp at query time
TASK_C="You MUST do TWO things in order: (1) Write a Python script ./q.py with the code below. (2) IMMEDIATELY EXECUTE it via the apptainer command and report the integer it prints. Both steps are required — do not stop after writing the file. \
\
The exact code for ./q.py: \
import sys, shutil, tempfile; sys.path.insert(0, '/home/sislam6/clio-search/code/src'); \
from pathlib import Path; \
from clio_agentic_search.connectors.iowarp.connector import IOWarpConnector; \
from clio_agentic_search.storage.duckdb_store import DuckDBStorage; \
from clio_agentic_search.retrieval.scientific import ScientificQueryOperators, NumericRangeOperator; \
src = Path('${INDEX_PATH}'); \
tmp = Path(tempfile.gettempdir()) / f'idx_{Path.cwd().name}.duckdb'; \
shutil.copyfile(src, tmp); \
store = DuckDBStorage(tmp); \
conn = IOWarpConnector(namespace='${NS}', storage=store, tag_pattern='${TAG_PATTERN}', blob_pattern='.*'); \
conn.connect(); \
op = ScientificQueryOperators(numeric_range=NumericRangeOperator(minimum=${PROMPT_MIN_VAL}, maximum=${PROMPT_MAX_VAL}, unit='${PROMPT_UNIT_VAL}')); \
res = conn.search_scientific_with_content(query='${PROMPT_QTEXT_VAL}', top_k=10000, operators=op); \
print(len(res)) \
\
DO NOT call cte.chimaera_init or initialize_cte directly — search_scientific_with_content fetches IOWarp content via an internal subprocess. \
\
After writing the file, RUN this exact command and read the output: \
apptainer exec /home/sislam6/jarvis-work/sif/iowarp_deploy_cpu.sif bash -c 'source /home/sislam6/jarvis-work/iowarp-venv/bin/activate && python3 ./q.py' \
\
Print only the integer it prints, on the very last line of your final response."

# Arm D — federated only meaningful for argo
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
  "dataset": "$DATASET",
  "prompt_id": $PROMPT_ID,
  "prompt_name": "$PROMPT_NAME",
  "model": "alcf_sophia/openai/gpt-oss-120b",
  "agent_framework": "opencode",
  "timeout_s": 300,
  "data_dir": "$DATA_DIR_OVERRIDE",
  "data_dir_files": $(ls $DATA_DIR_OVERRIDE 2>/dev/null | wc -l),
  "workspace": "$WORKSPACE",
  "started_at": "$(date -Iseconds)"
}
META

T0=$(date +%s.%N)
echo "[rank $RANK on $HOST dataset=$DATASET arm=$ARM prompt=$PROMPT_NAME] start $(date -Iseconds)" >> $LOG
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
  "dataset": "$DATASET",
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

echo "[rank $RANK] done dataset=${DATASET} arm=${ARM} prompt=${PROMPT_NAME} wall=${WALL}s answer=${ANSWER}" >> $LOG
exit 0
