#!/bin/bash
#PBS -N clio_agent_v6
#PBS -l select=5:ngpus=6
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q gpu_hack
#PBS -A gpu_hack
#PBS -j oe
set -u
module load python/3.12.12 apptainer/1.2.5 2>/dev/null
export PATH="$HOME/.opencode/bin:$PATH"

ARM=${ARM:?must set ARM=a/b/c/d via qsub -v}
DATASET=${DATASET:-argo}
BASE=/home/sislam6/clio-search/eval/agent_io_test
SCRIPTS=/lus/flare/projects/gpu_hack/sislam6/clio_eval/scripts

case "$DATASET" in
    argo) DATA_DIR=$BASE/workspaces/arm_a/data ;;
    era5) DATA_DIR=/home/sislam6/jarvis-work/era5_data/raw ;;
    *)    echo "Unknown DATASET=$DATASET" >&2; exit 2 ;;
esac

RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=/lus/flare/projects/gpu_hack/sislam6/clio_eval/results/v6_${DATASET}_${ARM}_${RUN_TS}
mkdir -p $RUN_DIR

NUM_NODES=$(wc -l < $PBS_NODEFILE)
echo "v6 (io_wrap inside agent process) dataset=$DATASET arm=$ARM nodes=$NUM_NODES run_dir=$RUN_DIR"

T0=$(date +%s)
export RUN_DIR ARM DATASET DATA_DIR BASE
mpiexec -n $NUM_NODES -ppn 1 $SCRIPTS/agent_inner_v6.sh
T1=$(date +%s)
echo "[head] dataset=$DATASET arm=$ARM done in $((T1-T0))s"
