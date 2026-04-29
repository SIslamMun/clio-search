#!/bin/bash
#PBS -N clio_shard_bld
#PBS -l select=20:ngpus=6
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q gpu_hack
#PBS -A gpu_hack
#PBS -j oe
set -u
module load python/3.12.12 apptainer/1.2.5 2>/dev/null

NUM_NODES=$(wc -l < $PBS_NODEFILE)
SCRIPT=/lus/flare/projects/gpu_hack/sislam6/clio_eval/scripts/parallel_shard_build.py
echo "Parallel shard build: $NUM_NODES ranks"

T0=$(date +%s)
mpiexec -n $NUM_NODES -ppn 1 \
    apptainer exec --bind /lus:/lus,/tmp:/tmp \
        /home/sislam6/jarvis-work/sif/iowarp_deploy_cpu.sif \
        bash -c "source /home/sislam6/jarvis-work/iowarp-venv/bin/activate && WORLD_SIZE=$NUM_NODES python3 $SCRIPT"
T1=$(date +%s)
echo "All shards built in $((T1-T0))s"
echo
echo "===== shards produced ====="
ls -la /lus/flare/projects/gpu_hack/sislam6/clio_eval/mega_shards/ 2>&1 | head -25
echo
echo "===== total index size ====="
du -sh /lus/flare/projects/gpu_hack/sislam6/clio_eval/mega_shards/ 2>&1
