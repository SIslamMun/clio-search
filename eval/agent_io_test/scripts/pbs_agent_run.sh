#!/bin/bash
# 100-agent PBS run for the CLIO+iowarp I/O benchmark.
# Submit twice with -v ARM=a and -v ARM=b for the two arms.
#
#   qsub -v ARM=a ~/clio-search/eval/agent_io_test/scripts/pbs_agent_run.sh
#   qsub -v ARM=b ~/clio-search/eval/agent_io_test/scripts/pbs_agent_run.sh
#
#PBS -N clio_agent_io
#PBS -l select=100:ncpus=64
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
CLIO_DOC=$BASE/workspaces/arm_b/CLIO_USAGE.md
RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$BASE/results/run_${RUN_TS}
mkdir -p $RUN_DIR

NUM_NODES=$(wc -l < $PBS_NODEFILE)

echo "============================================================"
echo "Agent fleet PBS run"
echo "  job:        $PBS_JOBID"
echo "  arm:        $ARM"
echo "  num nodes:  $NUM_NODES"
echo "  data dir:   $DATA_DIR  ($(ls $DATA_DIR | wc -l) files)"
echo "  run dir:    $RUN_DIR"
echo "  start:      $(date -Iseconds)"
echo "============================================================"

T0=$(date +%s)
export RUN_DIR ARM DATA_DIR CLIO_DOC
mpiexec -n $NUM_NODES -ppn 1 $BASE/scripts/agent_inner.sh
T1=$(date +%s)
WALL=$((T1 - T0))

echo
echo "[head] All ranks done in ${WALL}s. Aggregating..."

# Aggregate per-rank result.json files
$BASE/scripts/aggregate_agents.py $RUN_DIR/arm_${ARM} $WALL $ARM > $RUN_DIR/summary_${ARM}.json
cat $RUN_DIR/summary_${ARM}.json
echo
echo "[head] saved: $RUN_DIR/summary_${ARM}.json"
