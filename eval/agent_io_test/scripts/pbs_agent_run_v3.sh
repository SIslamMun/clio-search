#!/bin/bash
#PBS -N clio_agent_v3
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
BASE=/home/sislam6/clio-search/eval/agent_io_test
DATA_DIR=$BASE/workspaces/arm_a/data
RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$BASE/results/v3_${RUN_TS}
mkdir -p $RUN_DIR
lfs setstripe -p gecko.ddn_hdd -E -1 -c 2 $RUN_DIR 2>/dev/null

NUM_NODES=$(wc -l < $PBS_NODEFILE)
echo "v3 multi-prompt run: arm=$ARM nodes=$NUM_NODES run_dir=$RUN_DIR"

T0=$(date +%s)
export RUN_DIR ARM DATA_DIR BASE
mpiexec -n $NUM_NODES -ppn 1 $BASE/scripts/agent_inner_v3.sh
T1=$(date +%s)
echo "[head] arm=$ARM done in $((T1-T0))s"
