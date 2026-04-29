#!/bin/bash
# Per-rank agent runner v2 — same as v1 but Arm B starts a chimaera daemon
# AND populates iowarp with full blob content, so the agent's CLIO call
# (search_scientific_with_content) genuinely uses IOWarp at query time.
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

# --- Per-rank HOME isolation (avoid SQLite session DB collision on Lustre) ---
# Lustre default striping puts files on the ddn_ssd OST pool, which is at
# per-OST quota — opencode's SQLite migration touches >1 MB and dies with
# "disk I/O error" (EDQUOT). Stripe RANK_HOME onto ddn_hdd before opencode
# creates anything inside it.
RANK_HOME=$WORKSPACE/.rank_home
mkdir -p $RANK_HOME
lfs setstripe -p gecko.ddn_hdd -E -1 -c 2 $RANK_HOME 2>/dev/null || true
mkdir -p $RANK_HOME/.config/opencode
cp -f $HOME/.config/opencode/opencode.json $RANK_HOME/.config/opencode/ 2>/dev/null
ln -sf $HOME/.opencode/bin $RANK_HOME/.opencode-bin 2>/dev/null
mkdir -p $RANK_HOME/.opencode
export HOME=$RANK_HOME
export PATH="$RANK_HOME/.opencode-bin:$PATH"

# Arms B and C: copy CLIO_USAGE.md
if [ "$ARM" = "b" ] || [ "$ARM" = "c" ]; then
    cp $BASE/workspaces/arm_b/CLIO_USAGE.md $WORKSPACE/CLIO_USAGE.md 2>/dev/null
fi

LOG=$WORKSPACE/agent.log
RESULT=$WORKSPACE/result.json
SIF=/home/sislam6/jarvis-work/sif/iowarp_deploy_cpu.sif
VENV=/home/sislam6/jarvis-work/iowarp-venv
BLOBS_JSON=$BASE/blobs.json

CHI_PID=""

# --- Arm C ONLY: start chimaera daemon + populate iowarp ---
# (Arm B uses CLIO/DuckDB only, no iowarp at query time, so no daemon needed)
if [ "$ARM" = "c" ]; then
    echo "[rank $RANK] starting chimaera daemon for iowarp content store" > $LOG

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

    # Populate iowarp with blob content (kClient connects to running daemon).
    # Use os._exit(0) to skip iowarp_core's benign-but-noisy interpreter
    # shutdown segfault.
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

# Task prompts — three arms
TASK_A="Count how many Argo profile NetCDF files in the data/ directory have a surface temperature above 30 degrees Celsius. Each .nc file is one profile (NetCDF-3 classic format). Use Python with the netCDF4 module. Print only the final count number."

# Arm B: CLIO + DuckDB only (no IOWarp at query time — yesterday's baseline).
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
op = ScientificQueryOperators(numeric_range=NumericRangeOperator(minimum=86.0, maximum=None, unit='degF')); \
hits = conn.search_scientific(query='temperature above 86 degF', top_k=1000, operators=op); \
print(len(hits)) \
\
DO NOT call cte.chimaera_init or initialize_cte — this arm uses CLIO's DuckDB index only. \
\
After writing the file, RUN this exact command and read the output: \
apptainer exec /home/sislam6/jarvis-work/sif/iowarp_deploy_cpu.sif bash -c 'source /home/sislam6/jarvis-work/iowarp-venv/bin/activate && python3 ./q.py' \
\
Print only the integer it prints, on the very last line of your final response."

# Arm C: CLIO + IOWarp at query time (uses the new search_scientific_with_content
# path which fetches blob content from iowarp via a subprocess).
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
op = ScientificQueryOperators(numeric_range=NumericRangeOperator(minimum=86.0, maximum=None, unit='degF')); \
res = conn.search_scientific_with_content(query='temperature above 86 degF', top_k=1000, operators=op); \
print(len(res)) \
\
DO NOT call cte.chimaera_init or initialize_cte directly — search_scientific_with_content fetches IOWarp content via an internal subprocess. \
\
After writing the file, RUN this exact command and read the output: \
apptainer exec /home/sislam6/jarvis-work/sif/iowarp_deploy_cpu.sif bash -c 'source /home/sislam6/jarvis-work/iowarp-venv/bin/activate && python3 ./q.py' \
\
Print only the integer it prints, on the very last line of your final response."

# Arm D: federated CLIO query across yesterday's 100 pre-built Argo shards
# (true distributed CLIO at the agent layer — RetrievalCoordinator scatters
# one query across all 100 DuckDB shards on Lustre and merges by score).
#
# IMPORTANT: federated_query.py emits ~hundreds of lines of chimaera/IOWarp
# log output to stdout. When that stream is consumed by opencode's bash-tool
# output handler (Bun runtime), opencode hard-crashes with SIGTRAP (exit=133)
# 10-30s into the bash subprocess — reproduced 15/15 times in run
# v2_20260429_171429. Verified by 4-rank sanity test (8457077) that opencode
# itself works fine on trivial prompts at this scale; the crash is specific
# to TASK_D's bash-subprocess output volume.
#
# Workaround: silence federated_query's stdout/stderr (still writes the JSON
# output to ./fed.json) and have the bash command emit only the integer
# answer via a one-liner json read. opencode's bash tool then receives a
# single clean line — same I/O shape as arms B and C.
TASK_D="Run this exact command and report only the integer it prints: \
apptainer exec /home/sislam6/jarvis-work/sif/iowarp_deploy_cpu.sif bash -c 'source /home/sislam6/jarvis-work/iowarp-venv/bin/activate && python3 /home/sislam6/jarvis-work/federated_query.py --shards /home/sislam6/jarvis-work/shards/argo_20260428_183628 --dataset argo --out ./fed.json > /dev/null 2>&1 && python3 -c \"import json; print(json.load(open(\\\"./fed.json\\\"))[\\\"queries\\\"][0][\\\"global_top_k_count\\\"])\"' \
Print only the integer it prints on the very last line of your final response."

case "$ARM" in
  a) TASK="$TASK_A" ;;
  b) TASK="$TASK_B" ;;
  c) TASK="$TASK_C" ;;
  d) TASK="$TASK_D" ;;
  *) echo "Unknown ARM=$ARM" >&2; exit 2 ;;
esac

cd $WORKSPACE

# Save the exact prompt sent to the agent for later validation
printf '%s\n' "$TASK" > $WORKSPACE/prompt.txt

# Save invocation metadata
cat > $WORKSPACE/meta.json <<META
{
  "rank": $RANK,
  "host": "$HOST",
  "arm": "$ARM",
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
echo "[rank $RANK on $HOST arm=$ARM] start $(date -Iseconds)" >> $LOG
echo "[rank $RANK] prompt saved to prompt.txt; raw agent output below" >> $LOG
echo "===== BEGIN OPENCODE RAW OUTPUT =====" >> $LOG

# Stagger rank starts to spread connections to the model endpoint.
# (Prevents the simultaneous-burst that produced StreamingTimeout/'Expected id'
# errors in the 80-rank run on 2026-04-29.)
sleep $(( RANK * 2 ))

# Retry opencode on transient model-service errors. The Sophia gpt-oss-120b
# endpoint intermittently returns parse errors ('Expected 'id' to be a string')
# or HTTP 504 streaming timeouts under load — both are transient and retrying
# typically succeeds. We retry up to 3 times.
RC=0
for attempt in 1 2 3; do
    ATTEMPT_LOG=$WORKSPACE/.attempt_${attempt}.log
    timeout 300 opencode run -m alcf_sophia/openai/gpt-oss-120b "$TASK" \
        > $ATTEMPT_LOG 2>&1
    RC=$?
    cat $ATTEMPT_LOG >> $LOG
    # Retry on:
    #  - the known transient model errors,
    #  - any non-zero exit (covers opencode crashes like exit=133 on TASK_D
    #    that don't print a recognizable error string).
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

# Extract answer: only an integer that appears ON ITS OWN LINE between the
# BEGIN/END markers. This deliberately excludes integers embedded in error
# JSON (e.g. "code":504, "after 30 seconds", "[Errno 30] Read-only ...") that
# the previous version pulled with `grep -oE '\b[0-9]+\b' | tail -1`.
ANSWER=$(awk '/===== BEGIN OPENCODE RAW OUTPUT =====/{f=1; next} /===== END OPENCODE RAW OUTPUT/{f=0} f' $LOG \
  | sed 's/\x1b\[[0-9;]*m//g' \
  | grep -E '^[[:space:]]*[0-9]+[[:space:]]*$' \
  | tail -1 \
  | tr -d '[:space:]')
DATA_BYTES=$(du -sb $WORKSPACE/data 2>/dev/null | awk '{print $1}')

# Save the agent's last ~100 lines of output for quick validation
tail -100 $LOG > $WORKSPACE/final_output.txt

# Did the agent actually call iowarp? Look for "fetch_blob_content" or "search_scientific_with_content" in the log
IOWARP_CALLS=$(grep -cE "fetch_blob_content|search_scientific_with_content|Tag\.GetBlob|cte\.Tag" $LOG 2>/dev/null | head -1)
IOWARP_CALLS=${IOWARP_CALLS:-0}

cat > $RESULT <<EOF
{
  "rank": $RANK,
  "host": "$HOST",
  "arm": "$ARM",
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

echo "[rank $RANK] done wall=${WALL}s answer=${ANSWER} iowarp_evidence=${IOWARP_CALLS}" >> $LOG

# IMPORTANT: always exit 0 from the per-rank script. PALS mpiexec on Aurora
# treats any rank's non-zero exit as an abort signal and SIGTERMs the other
# ranks mid-flight — that destroyed arm D ranks 1-4 in run v2_20260429_170626.
# The per-rank result.json already records the real opencode exit code, so
# returning 0 here doesn't lose information.
exit 0
