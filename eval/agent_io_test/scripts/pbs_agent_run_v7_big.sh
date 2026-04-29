#!/bin/bash
#PBS -N clio_v7_big
#PBS -l select=5:ngpus=6
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q gpu_hack
#PBS -A gpu_hack
#PBS -j oe
set -u
module load python/3.12.12 apptainer/1.2.5 2>/dev/null
export PATH="$HOME/.opencode/bin:$PATH"

ARM=${ARM:?must set ARM=a/b/c via qsub -v}
DATASET=argo  # v7_big repurposes argo case to point at big corpus
BASE=/home/sislam6/clio-search/eval/agent_io_test
SCRIPTS=/lus/flare/projects/gpu_hack/sislam6/clio_eval/scripts

DATA_DIR=/lus/flare/projects/gpu_hack/sislam6/clio_eval/big_corpus/data

RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=/lus/flare/projects/gpu_hack/sislam6/clio_eval/results/v7big_${ARM}_${RUN_TS}
mkdir -p $RUN_DIR

NUM_NODES=$(wc -l < $PBS_NODEFILE)
echo "v7 BIG (5000 NetCDFs / 4928-blob index) arm=$ARM nodes=$NUM_NODES run_dir=$RUN_DIR"

T0=$(date +%s)
export RUN_DIR ARM DATASET DATA_DIR BASE
mpiexec -n $NUM_NODES -ppn 1 $SCRIPTS/agent_inner_v7_big.sh
T1=$(date +%s)
echo "[head] arm=$ARM done in $((T1-T0))s"
