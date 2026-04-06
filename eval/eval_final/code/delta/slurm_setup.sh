#!/bin/bash
# ---------------------------------------------------------------------------
# Step 0: Setup — create venv, download arXiv data, build DuckDB indices.
# Runs on a SINGLE compute node. Must complete before strong/weak scaling.
#
# Usage:  sbatch slurm_setup.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=clio_setup
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/clio_setup_%j.out
#SBATCH --error=logs/clio_setup_%j.err

set -euo pipefail

REPO="/u/sislam3/clio-search"
WORK="/work/nvme/bekn/sislam3"
ARXIV_DIR="${WORK}/arxiv"
VENV="${WORK}/clio-venv"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs "$WORK"

echo "========================================"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "========================================"

# ---------- Step 1: Python environment ----------
module load python/3.11.9

if [ ! -d "$VENV" ]; then
    echo "Creating Python venv at $VENV"
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

echo "Python: $(which python3) — $(python3 --version)"

# Install CLIO and dependencies
cd "$REPO/code"
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
pip install aiohttp kaggle

echo "CLIO installed successfully."

# ---------- Step 2: Download arXiv data ----------
mkdir -p "$ARXIV_DIR"

if [ -f "${ARXIV_DIR}/arxiv_shard_0.jsonl" ]; then
    echo "arXiv shards already exist, skipping download."
else
    echo "Downloading arXiv metadata..."
    python3 "$REPO/eval/eval_final/code/delta/download_arxiv.py" \
        --output-dir "$ARXIV_DIR" --shards 4
fi

echo "arXiv shards:"
wc -l "${ARXIV_DIR}"/arxiv_shard_*.jsonl || true

# ---------- Step 3: Build DuckDB indices ----------
echo "Building DuckDB indices (4 shards, parallel)..."
for i in 0 1 2 3; do
    DB="${WORK}/clio_shard_${i}.duckdb"
    SHARD="${ARXIV_DIR}/arxiv_shard_${i}.jsonl"
    if [ -f "$DB" ]; then
        echo "  shard $i: DB already exists ($DB), skipping."
    else
        echo "  shard $i: indexing $SHARD -> $DB"
        python3 "$REPO/eval/eval_final/code/delta/index_shard.py" \
            --shard-jsonl "$SHARD" \
            --db-path "$DB" \
            --namespace distributed_clio &
    fi
done
wait
echo "All shards indexed."

# ---------- Step 4: Prepare weak scaling data ----------
echo "Preparing weak scaling phase directories..."
export BASE_DIR="$WORK"
for n_workers in 1 2 4; do
    phase_dir="${WORK}/weak_phase_${n_workers}"
    mkdir -p "$phase_dir"
    case $n_workers in
        1) indices="0" ;;
        2) indices="0 1" ;;
        4) indices="0 1 2 3" ;;
    esac
    for idx in $indices; do
        src="${WORK}/clio_shard_${idx}.duckdb"
        dst="${phase_dir}/clio_shard_${idx}.duckdb"
        if [ -e "$dst" ]; then
            echo "  skip (exists): $dst"
        else
            echo "  link: $src -> $dst"
            ln -s "$src" "$dst"
        fi
    done
    echo "  Phase $n_workers ready."
done

# ---------- Summary ----------
echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo "Venv:     $VENV"
echo "arXiv:    $ARXIV_DIR"
echo "Shards:   ${WORK}/clio_shard_{0,1,2,3}.duckdb"
echo ""
echo "Next steps:"
echo "  sbatch slurm_strong_scaling.sh"
echo "  sbatch slurm_weak_scaling.sh"
ls -lh "${WORK}"/clio_shard_*.duckdb 2>/dev/null || echo "(no DB files yet)"
echo "Total NVMe usage:"
du -sh "$WORK" 2>/dev/null || true
