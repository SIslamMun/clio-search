#!/bin/bash
#PBS -N clio_v8_mega
#PBS -l select=20:ngpus=6
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:flare
#PBS -q gpu_hack
#PBS -A gpu_hack
#PBS -j oe
set -u
module load python/3.12.12 apptainer/1.2.5 2>/dev/null
export PATH="$HOME/.opencode/bin:$PATH"

ARM=${ARM:?must set ARM=b/c via qsub -v}
SCRIPTS=/lus/flare/projects/gpu_hack/sislam6/clio_eval/scripts

RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=/lus/flare/projects/gpu_hack/sislam6/clio_eval/results/v8mega_${ARM}_${RUN_TS}
mkdir -p $RUN_DIR

NUM_NODES=$(wc -l < $PBS_NODEFILE)
echo "v8_mega arm=$ARM nodes=$NUM_NODES (each rank queries 1 shard from 20 mega shards / 320k blobs)"

T0=$(date +%s)
export RUN_DIR ARM
mpiexec -n $NUM_NODES -ppn 1 $SCRIPTS/agent_inner_v8_mega.sh
T1=$(date +%s)
echo "[head] arm=$ARM done in $((T1-T0))s"
