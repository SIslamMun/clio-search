#!/bin/bash
# v2 100-vs-100 agent fleet, but on the small `gpu_hack` queue (max 8 nodes).
# Submit twice with -v ARM=a / ARM=b.
#
#   qsub -v ARM=a ~/clio-search/eval/agent_io_test/scripts/pbs_agent_run_v2.sh
#   qsub -v ARM=b ~/clio-search/eval/agent_io_test/scripts/pbs_agent_run_v2.sh
#
#PBS -N clio_agent_v2
#PBS -l select=5:ngpus=6
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q gpu_hack
#PBS -A gpu_hack
#PBS -j oe

set -u
module load python/3.12.12 apptainer/1.2.5 2>/dev/null
export PATH="$HOME/.opencode/bin:$PATH"

ARM=${ARM:?must set ARM=a or ARM=b via qsub -v}
BASE=/home/sislam6/clio-search/eval/agent_io_test
DATA_DIR=$BASE/workspaces/arm_a/data
RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$BASE/results/v2_${RUN_TS}
mkdir -p $RUN_DIR
# Lustre default striping (ddn_ssd pool) is at per-OST quota — large writes
# under $RUN_DIR (opencode SQLite, agent.log) fail with EDQUOT. Force the
# ddn_hdd pool which has headroom; child dirs/files inherit this layout.
lfs setstripe -p gecko.ddn_hdd -E -1 -c 2 $RUN_DIR 2>/dev/null || true

NUM_NODES=$(wc -l < $PBS_NODEFILE)

echo "============================================================"
echo "Agent fleet PBS run (v2 — IOWarp at query time, kClient daemon)"
echo "  job:        $PBS_JOBID"
echo "  arm:        $ARM"
echo "  num nodes:  $NUM_NODES"
echo "  data dir:   $DATA_DIR  ($(ls $DATA_DIR | wc -l) files)"
echo "  run dir:    $RUN_DIR"
echo "  start:      $(date -Iseconds)"
echo "============================================================"

T0=$(date +%s)
export RUN_DIR ARM DATA_DIR BASE
mpiexec -n $NUM_NODES -ppn 1 $BASE/scripts/agent_inner_v2.sh
T1=$(date +%s)
WALL=$((T1 - T0))

echo
echo "[head] All ranks done in ${WALL}s. Aggregating..."

$BASE/scripts/aggregate_agents.py $RUN_DIR/arm_${ARM} $WALL $ARM > $RUN_DIR/summary_${ARM}.json
cat $RUN_DIR/summary_${ARM}.json
echo
echo "[head] saved: $RUN_DIR/summary_${ARM}.json"
