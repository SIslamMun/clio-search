#!/bin/bash
# v8 — distributed agent benchmark over the 20 mega shards.
#
# Each rank handles ONE mega shard (mega_shard_${RANK:03d}.duckdb) and
# queries one prompt (RANK % 3). The agent's q.py points at this rank's
# specific shard. Aggregator post-process sums per-prompt hits across all
# 20 ranks.
#
# Arms supported: b (CLIO+DuckDB) and c (CLIO+IOWarp). Arm A is skipped
# because naive scanning 5 GB of raw NetCDFs through opencode's 300 s
# per-attempt timeout is infeasible.

set -u

RANK=${PALS_RANKID:-${PMI_RANK:-0}}
HOST=$(hostname)

if [ "$ARM" = "d" ]; then
    echo "[rank $RANK] arm=d not supported on mega corpus" >&2
    exit 0
fi

export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export PATH="$HOME/.opencode/bin:$PATH"

# Per-rank shard
SHARD_PATH="/lus/flare/projects/gpu_hack/sislam6/clio_eval/mega_shards/mega_shard_$(printf '%03d' $RANK).duckdb"
NS="mega_shard_${RANK}"

# Prompt grid (same 3 as before but threshold tuned for mega scale)
PROMPT_NAMES=(
    "mega_temp_above_30C"
    "mega_temp_above_25C"
    "mega_temp_above_35C"
)
PROMPT_MIN=(  "30.0" "25.0" "35.0" )
PROMPT_UNIT=( "degC" "degC" "degC" )
PROMPT_QTEXT=(
    "temperature above 30 degC"
    "temperature above 25 degC"
    "temperature above 35 degC"
)

NUM_PROMPTS=${#PROMPT_NAMES[@]}
PROMPT_ID=$(( RANK % NUM_PROMPTS ))
PROMPT_NAME="${PROMPT_NAMES[$PROMPT_ID]}"
PROMPT_MIN_VAL="${PROMPT_MIN[$PROMPT_ID]}"
PROMPT_UNIT_VAL="${PROMPT_UNIT[$PROMPT_ID]}"
PROMPT_QTEXT_VAL="${PROMPT_QTEXT[$PROMPT_ID]}"

# Workspace + isolation on /lus/flare
WORKSPACE=$RUN_DIR/arm_${ARM}/rank_${RANK}
mkdir -p $WORKSPACE

RANK_HOME=$WORKSPACE/.rank_home
mkdir -p $RANK_HOME/.config/opencode $RANK_HOME/.opencode
cp -f $HOME/.config/opencode/opencode.json $RANK_HOME/.config/opencode/ 2>/dev/null
ln -sf $HOME/.opencode/bin $RANK_HOME/.opencode-bin 2>/dev/null
export HOME=$RANK_HOME
export PATH="$RANK_HOME/.opencode-bin:$PATH"

# Arm A: symlink the FULL 5000 Argo NetCDFs into rank workspace data/
# (5000 files / 488 MB — naive scan target). One per rank — same data for each
# rank's scan. Demonstrates "naive scaling" at non-trivial size.
if [ "$ARM" = "a" ]; then
    mkdir -p $WORKSPACE/data
    for f in /lus/flare/projects/gpu_hack/sislam6/clio_eval/big_corpus/data/*.nc; do
        ln -sf "$f" $WORKSPACE/data/$(basename $f)
    done
fi

LOG=$WORKSPACE/agent.log
RESULT=$WORKSPACE/result.json
SIF=/home/sislam6/jarvis-work/sif/iowarp_deploy_cpu.sif
VENV=/home/sislam6/jarvis-work/iowarp-venv

CHI_PID=""
if [ "$ARM" = "c" ]; then
    echo "[rank $RANK] starting chimaera daemon" > $LOG
    export CHI_STATE_DIR=/tmp/chi_rank_${RANK}_$$
    rm -rf $CHI_STATE_DIR; mkdir -p $CHI_STATE_DIR
    apptainer exec --bind /lus:/lus,/tmp:/tmp $SIF bash -c "
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
fi

# Build per-prompt natural-language description for arm A
case "$PROMPT_ID" in
  0) NL_A="how many of those Argo profile NetCDF files have a temperature above 30 degrees Celsius" ;;
  1) NL_A="how many of those Argo profile NetCDF files have a temperature above 25 degrees Celsius" ;;
  2) NL_A="how many of those Argo profile NetCDF files have a temperature above 35 degrees Celsius" ;;
esac

# TASK_A: naive scan of 5000 Argo NetCDFs in data/ (488 MB). Force-execute framing.
TASK_A="You MUST do TWO things in order: (1) Write a Python script ./scan.py that reads ALL the NetCDF files in the data/ directory using the netCDF4 module and counts ${NL_A_VAL:-$NL_A}. There are 5000 files in data/. (2) IMMEDIATELY EXECUTE the script via the apptainer command and report the integer it prints. Both steps required. \
\
After writing the file, RUN this exact command: \
apptainer exec --bind /lus:/lus,/tmp:/tmp /home/sislam6/jarvis-work/sif/iowarp_deploy_cpu.sif bash -c 'source /home/sislam6/jarvis-work/iowarp-venv/bin/activate && python3 /home/sislam6/jarvis-work/io_wrap.py ./scan.py' \
\
Print only the integer it prints, on the very last line of your final response."

# TASK_B: CLIO DuckDB-only against rank's shard
TASK_B="You MUST do TWO things in order: (1) Write a Python script ./q.py with the code below. (2) IMMEDIATELY EXECUTE it via the apptainer command and report the integer it prints. Both steps are required — do not stop after writing the file. \
\
The exact code for ./q.py: \
import sys, shutil, tempfile; sys.path.insert(0, '/home/sislam6/clio-search/code/src'); \
from pathlib import Path; \
from clio_agentic_search.connectors.iowarp.connector import IOWarpConnector; \
from clio_agentic_search.storage.duckdb_store import DuckDBStorage; \
from clio_agentic_search.retrieval.scientific import ScientificQueryOperators, NumericRangeOperator; \
src = Path('${SHARD_PATH}'); \
tmp = Path(tempfile.gettempdir()) / f'mega_shard_${RANK}.duckdb'; \
shutil.copyfile(src, tmp); \
store = DuckDBStorage(tmp); \
conn = IOWarpConnector(namespace='${NS}', storage=store, tag_pattern='.*', blob_pattern='.*'); \
conn.connect(); \
op = ScientificQueryOperators(numeric_range=NumericRangeOperator(minimum=${PROMPT_MIN_VAL}, maximum=None, unit='${PROMPT_UNIT_VAL}')); \
hits = conn.search_scientific(query='${PROMPT_QTEXT_VAL}', top_k=1000000, operators=op); \
print(len(hits)) \
\
After writing the file, RUN this exact command and read the output: \
apptainer exec --bind /lus:/lus,/tmp:/tmp /home/sislam6/jarvis-work/sif/iowarp_deploy_cpu.sif bash -c 'source /home/sislam6/jarvis-work/iowarp-venv/bin/activate && python3 /home/sislam6/jarvis-work/io_wrap.py ./q.py' \
\
Print only the integer it prints, on the very last line of your final response."

TASK_C="You MUST do TWO things in order: (1) Write a Python script ./q.py with the code below. (2) IMMEDIATELY EXECUTE it via the apptainer command and report the integer it prints. \
\
The exact code for ./q.py: \
import sys, shutil, tempfile; sys.path.insert(0, '/home/sislam6/clio-search/code/src'); \
from pathlib import Path; \
from clio_agentic_search.connectors.iowarp.connector import IOWarpConnector; \
from clio_agentic_search.storage.duckdb_store import DuckDBStorage; \
from clio_agentic_search.retrieval.scientific import ScientificQueryOperators, NumericRangeOperator; \
src = Path('${SHARD_PATH}'); \
tmp = Path(tempfile.gettempdir()) / f'mega_shard_${RANK}.duckdb'; \
shutil.copyfile(src, tmp); \
store = DuckDBStorage(tmp); \
conn = IOWarpConnector(namespace='${NS}', storage=store, tag_pattern='.*', blob_pattern='.*'); \
conn.connect(); \
op = ScientificQueryOperators(numeric_range=NumericRangeOperator(minimum=${PROMPT_MIN_VAL}, maximum=None, unit='${PROMPT_UNIT_VAL}')); \
res = conn.search_scientific_with_content(query='${PROMPT_QTEXT_VAL}', top_k=1000000, operators=op); \
print(len(res)) \
\
After writing the file, RUN this exact command: \
apptainer exec --bind /lus:/lus,/tmp:/tmp /home/sislam6/jarvis-work/sif/iowarp_deploy_cpu.sif bash -c 'source /home/sislam6/jarvis-work/iowarp-venv/bin/activate && python3 /home/sislam6/jarvis-work/io_wrap.py ./q.py' \
\
Print only the integer it prints, on the very last line of your final response."

case "$ARM" in
  a) TASK="$TASK_A" ;;
  b) TASK="$TASK_B" ;;
  c) TASK="$TASK_C" ;;
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
  "shard_path": "$SHARD_PATH",
  "started_at": "$(date -Iseconds)"
}
META

T0=$(date +%s.%N)
echo "[rank $RANK on $HOST arm=$ARM prompt=$PROMPT_NAME shard=mega_$(printf '%03d' $RANK)] start" >> $LOG
echo "===== BEGIN OPENCODE RAW OUTPUT =====" >> $LOG

sleep $(( RANK * 1 ))  # gentle stagger

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

# Extract io_delta line from agent.log (printed by io_wrap.py)
IO_LINE=$(grep -aE 'IO_DELTA: \{' $LOG | tail -1 | sed 's/^.*IO_DELTA: //')
[ -z "$IO_LINE" ] && IO_LINE="null"

cat > $RESULT <<EOF
{
  "rank": $RANK,
  "host": "$HOST",
  "arm": "$ARM",
  "prompt_id": $PROMPT_ID,
  "prompt_name": "$PROMPT_NAME",
  "shard_path": "$SHARD_PATH",
  "wall_seconds": $WALL,
  "exit_code": $RC,
  "answer": "${ANSWER:-null}",
  "io_delta": $IO_LINE
}
EOF

if [ -n "$CHI_PID" ]; then
    kill -TERM $CHI_PID 2>/dev/null
    wait $CHI_PID 2>/dev/null
    rm -rf $CHI_STATE_DIR 2>/dev/null
fi
rm -f /tmp/chimaera_*.ipc /tmp/chi_main_segment_* /tmp/mega_shard_${RANK}.duckdb* 2>/dev/null

echo "[rank $RANK] done arm=${ARM} prompt=${PROMPT_NAME} wall=${WALL}s answer=${ANSWER}" >> $LOG
exit 0
