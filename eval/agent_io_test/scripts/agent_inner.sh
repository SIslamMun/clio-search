#!/bin/bash
# Per-rank agent runner.
# Reads:  RUN_DIR (Lustre-shared root), ARM ("a" or "b"), DATA_DIR, CLIO_DOC (only for arm b)
#         PALS_RANKID / PMI_RANK (set by mpiexec)
# Each rank creates its own workspace (symlinks to common data) and runs opencode.
set -u

RANK=${PALS_RANKID:-${PMI_RANK:-0}}
HOST=$(hostname)

# Compute-node proxy for ALCF endpoint
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export PATH="$HOME/.opencode/bin:$PATH"

WORKSPACE=$RUN_DIR/arm_${ARM}/rank_${RANK}
mkdir -p $WORKSPACE/data
# symlink data files (so this is per-rank but doesn't duplicate data)
for f in $DATA_DIR/*.nc; do
    ln -sf "$f" $WORKSPACE/data/$(basename $f)
done

# Arm B also gets the CLIO usage doc
if [ "$ARM" = "b" ] && [ -n "${CLIO_DOC:-}" ]; then
    cp "$CLIO_DOC" $WORKSPACE/CLIO_USAGE.md
fi

LOG=$WORKSPACE/agent.log
RESULT=$WORKSPACE/result.json
TASK_A="Count how many Argo profile NetCDF files in the data/ directory have a surface temperature above 30 degrees Celsius. Each .nc file is one profile (NetCDF-3 classic format). Use Python with the netCDF4 module. Print only the final count number."
TASK_B="Read CLIO_USAGE.md first to learn how to use the CLIO retrieval primitive available in this workspace. Then count how many Argo profiles in data/ have a surface temperature above 30 degrees Celsius. Use CLIO instead of scanning the NetCDF files directly. Print only the final count number."

if [ "$ARM" = "a" ]; then TASK="$TASK_A"; else TASK="$TASK_B"; fi

cd $WORKSPACE
T0=$(date +%s.%N)
echo "[rank $RANK on $HOST arm=$ARM] start $(date -Iseconds)" > $LOG

# Run opencode with the gpt-oss-120b model on Sophia
timeout 300 opencode run -m alcf_sophia/openai/gpt-oss-120b "$TASK" >> $LOG 2>&1
RC=$?

T1=$(date +%s.%N)
WALL=$(awk "BEGIN {print $T1 - $T0}")

# Heuristic: extract last number printed in the log (most likely the count)
ANSWER=$(grep -oE '\b[0-9]+\b' $LOG | tail -1)

# Approximate I/O bytes:
#   Arm A: counts data dir size (the agent likely scanned all of it)
#   Arm B: tiny DuckDB read (we approximate as 100 KB; actual depends on query)
DATA_BYTES=$(du -sb $WORKSPACE/data 2>/dev/null | awk '{print $1}')

cat > $RESULT <<EOF
{
  "rank": $RANK,
  "host": "$HOST",
  "arm": "$ARM",
  "wall_seconds": $WALL,
  "exit_code": $RC,
  "answer": "${ANSWER:-null}",
  "data_dir_bytes": ${DATA_BYTES:-0}
}
EOF

echo "[rank $RANK] done wall=${WALL}s answer=${ANSWER}" >> $LOG
exit $RC
