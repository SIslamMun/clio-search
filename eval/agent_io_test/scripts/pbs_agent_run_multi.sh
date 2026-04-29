#!/bin/bash
# Multi-dataset agent fleet PBS submitter (parameterized over argo/era5).
# Submit with -v ARM=a|b and DATASET=argo|era5.
#
#   qsub -v ARM=a,DATASET=argo ~/clio-search/eval/agent_io_test/scripts/pbs_agent_run_multi.sh
#   qsub -v ARM=b,DATASET=argo ~/clio-search/eval/agent_io_test/scripts/pbs_agent_run_multi.sh
#   qsub -v ARM=a,DATASET=era5 ~/clio-search/eval/agent_io_test/scripts/pbs_agent_run_multi.sh
#   qsub -v ARM=b,DATASET=era5 ~/clio-search/eval/agent_io_test/scripts/pbs_agent_run_multi.sh
#
#PBS -N clio_agent_multi
#PBS -l select=2:ngpus=6:ncpus=208
#PBS -l walltime=01:00:00
#PBS -l filesystems=home
#PBS -q gpu_hack
#PBS -A gpu_hack
#PBS -j oe

set -u
module load python/3.12.12 apptainer/1.2.5 2>/dev/null
export PATH="$HOME/.opencode/bin:$PATH"

ARM=${ARM:?must set ARM=a or ARM=b via qsub -v}
DATASET=${DATASET:-argo}
BASE=/home/sislam6/clio-search/eval/agent_io_test

# Resolve dataset-specific data dir for the head-node banner only
case "$DATASET" in
    argo)
        DATA_DIR=$BASE/workspaces/arm_a/data
        ;;
    era5)
        DATA_DIR=/home/sislam6/jarvis-work/era5_data/raw
        ;;
    *)
        echo "ERROR: unknown DATASET=$DATASET (expected argo or era5)" >&2
        exit 2
        ;;
esac

RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$BASE/results/multi_${DATASET}_${RUN_TS}
mkdir -p $RUN_DIR

NUM_NODES=$(wc -l < $PBS_NODEFILE)

echo "============================================================"
echo "Agent fleet PBS run (multi — IOWarp at query time, kClient daemon)"
echo "  job:        $PBS_JOBID"
echo "  arm:        $ARM"
echo "  dataset:    $DATASET"
echo "  num nodes:  $NUM_NODES"
echo "  data dir:   $DATA_DIR  ($(ls $DATA_DIR 2>/dev/null | wc -l) files)"
echo "  run dir:    $RUN_DIR"
echo "  start:      $(date -Iseconds)"
echo "============================================================"

T0=$(date +%s)
export RUN_DIR ARM DATASET BASE
mpiexec -n $NUM_NODES -ppn 1 $BASE/scripts/agent_inner_multi.sh
T1=$(date +%s)
WALL=$((T1 - T0))

echo
echo "[head] All ranks done in ${WALL}s. Aggregating..."

$BASE/scripts/aggregate_agents.py $RUN_DIR/arm_${ARM} $WALL $ARM > $RUN_DIR/summary_${ARM}.json
cat $RUN_DIR/summary_${ARM}.json
echo
echo "[head] saved: $RUN_DIR/summary_${ARM}.json"
