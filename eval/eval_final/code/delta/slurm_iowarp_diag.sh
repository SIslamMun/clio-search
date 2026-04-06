#!/bin/bash
#SBATCH --job-name=clio_diag
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/clio_diag_%j.out
#SBATCH --error=logs/clio_diag_%j.err

WORK="/work/nvme/bekn/sislam3"
REPO="/u/sislam3/clio-search"
WHEEL="${REPO}/eval/eval_eve/iowarp_core-1.0.3-cp312-cp312-linux_x86_64.whl"
SIF="${WORK}/iowarp-deps-cpu.sif"

mkdir -p logs
module load apptainer 2>/dev/null || true

apptainer exec --writable-tmpfs "$SIF" bash -c "
    echo '=== PATH ==='
    echo \$PATH
    echo '=== find chimaera ==='
    find / -name 'chimaera*' -type f 2>/dev/null | head -20
    echo '=== find jarvis ==='
    find / -name 'jarvis*' -type f 2>/dev/null | head -10
    echo '=== after pip install ==='
    pip install /wheels/$(basename $WHEEL) -q --force-reinstall 2>/dev/null
    find /home/iowarp/venv/bin/ -type f | sort
" --bind "${WHEEL}:/wheels/$(basename $WHEEL):ro"
