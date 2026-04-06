#!/bin/bash
# ---------------------------------------------------------------------------
# Prepare weak-scaling phase directories from base shards.
#
# The base shards (clio_shard_0..3.duckdb) are produced by index_shard.py.
# This script creates per-phase directories by symlinking (or copying)
# the appropriate subset of shards:
#
#   weak_phase_1/  -> shard 0 only          (625K docs)
#   weak_phase_2/  -> shards 0, 1           (1.25M docs)
#   weak_phase_4/  -> shards 0, 1, 2, 3     (2.5M docs)
#
# Usage:
#   bash prepare_weak_data.sh [--copy]
#
# By default uses symlinks. Pass --copy to make full copies instead
# (needed if workers modify the DB files).
# ---------------------------------------------------------------------------

set -euo pipefail

BASE_DIR="/scratch/$USER"
USE_COPY=false

if [[ "${1:-}" == "--copy" ]]; then
    USE_COPY=true
fi

# Phases: number of workers -> which shard indices to include
declare -A PHASES
PHASES[1]="0"
PHASES[2]="0 1"
PHASES[4]="0 1 2 3"

for n_workers in 1 2 4; do
    phase_dir="${BASE_DIR}/weak_phase_${n_workers}"
    mkdir -p "$phase_dir"

    for i in ${PHASES[$n_workers]}; do
        src="${BASE_DIR}/clio_shard_${i}.duckdb"
        dst="${phase_dir}/clio_shard_${i}.duckdb"

        if [ ! -f "$src" ]; then
            echo "ERROR: base shard not found: $src"
            echo "Run index_shard.py first to create the base shards."
            exit 1
        fi

        if [ -e "$dst" ]; then
            echo "  skip (exists): $dst"
            continue
        fi

        if $USE_COPY; then
            echo "  copy: $src -> $dst"
            cp "$src" "$dst"
        else
            echo "  link: $src -> $dst"
            ln -s "$src" "$dst"
        fi
    done

    echo "Phase $n_workers: $phase_dir ready ($(ls "$phase_dir"/*.duckdb 2>/dev/null | wc -l) shards)"
done

echo ""
echo "All weak-scaling phase directories prepared."
echo "You can now submit: sbatch slurm_weak_scaling.sh"
