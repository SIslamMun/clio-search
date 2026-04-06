#!/bin/bash
# Diagnostic: check iowarp_core availability in Apptainer on GH200 (aarch64)
#SBATCH --job-name=clio_iowarp_diag
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/clio_iowarp_diag_%j.out
#SBATCH --error=logs/clio_iowarp_diag_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

SIF="/work/nvme/bekn/sislam3/iowarp-deps-cpu.sif"
module load apptainer 2>/dev/null || true

echo "=== HOST ARCH ==="
uname -m

echo "=== Apptainer arch ==="
apptainer exec "$SIF" uname -m

echo "=== Python version in container ==="
apptainer exec "$SIF" python3 --version

echo "=== iowarp_core pre-installed? ==="
apptainer exec "$SIF" bash -c "
    source /home/iowarp/venv/bin/activate 2>/dev/null || true
    python3 -c 'import iowarp_core; print(\"YES - version:\", iowarp_core.__version__)' 2>&1 || echo 'NOT installed'
"

echo "=== pip show iowarp_core ==="
apptainer exec "$SIF" bash -c "
    source /home/iowarp/venv/bin/activate 2>/dev/null || true
    pip show iowarp_core 2>&1 || echo 'not in pip'
"

echo "=== pip index versions iowarp_core ==="
apptainer exec "$SIF" bash -c "
    pip index versions iowarp_core 2>&1 | head -5 || echo 'pip index unavailable'
"

echo "=== Try pip install iowarp_core from PyPI ==="
apptainer exec --writable-tmpfs "$SIF" bash -c "
    source /home/iowarp/venv/bin/activate 2>/dev/null || true
    pip install iowarp_core -q 2>&1 | tail -5
    python3 -c 'import iowarp_core; print(\"SUCCESS:\", iowarp_core.__version__)' 2>&1 || echo 'FAILED to import after install'
"

echo "=== DONE ==="
