"""Per-rank shard build — runs once per rank under mpiexec.

Each rank takes 1/WORLD slice of (Argo + ERA5 blobs) and builds its own
DuckDB shard at /tmp first, then copies to /lus/flare. Parallel across
mpiexec ranks. Output: 50 shards in /lus/flare/.../mega_shards/.
"""
import json, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, "/home/sislam6/clio-search/code/src")
from clio_agentic_search.connectors.iowarp.connector import IOWarpConnector
from clio_agentic_search.storage.duckdb_store import DuckDBStorage

ARGO_BLOBS = Path("/lus/flare/projects/gpu_hack/sislam6/clio_eval/big_corpus/argo_big_blobs.json")
ERA5_BLOBS = Path("/home/sislam6/clio-search/eval/agent_io_test/era5_blobs.json")
OUT_DIR    = Path("/lus/flare/projects/gpu_hack/sislam6/clio_eval/mega_shards")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANK = int(os.environ.get("PALS_RANKID", os.environ.get("PMI_RANK", "0")))
WORLD = int(os.environ.get("PMI_SIZE", os.environ.get("PALS_LOCAL_SIZE", "1")))
# PALS_LOCAL_SIZE is per-node; total ranks = NUM_NODES from launcher
WORLD = int(os.environ.get("WORLD_SIZE", WORLD))

print(f"[rank {RANK}/{WORLD}] start", flush=True)

# Each rank loads ALL blobs (yes wasteful, but ERA5 is only 94MB).
# Then takes a deterministic slice.
t0 = time.time()
print(f"[rank {RANK}] loading argo + era5 blobs...", flush=True)
argo = json.loads(ARGO_BLOBS.read_text())
era5 = json.loads(ERA5_BLOBS.read_text())
all_keys = sorted(argo.keys()) + sorted(era5.keys())
all_blobs = {**argo, **era5}
total = len(all_keys)
print(f"[rank {RANK}] total blobs: {total}, load time {time.time()-t0:.1f}s", flush=True)

# Per-rank slice
chunk = (total + WORLD - 1) // WORLD
start = RANK * chunk
end = min(start + chunk, total)
my_keys = all_keys[start:end]
my_blobs = {k: all_blobs[k] for k in my_keys}
print(f"[rank {RANK}] slice [{start}:{end}] = {len(my_blobs)} blobs", flush=True)

# Build at /tmp (fast)
TMP_SHARD = Path(f"/tmp/mega_shard_{RANK}.duckdb")
for ext in ("", ".wal", ".lock"):
    p = TMP_SHARD.with_suffix(TMP_SHARD.suffix + ext) if ext else TMP_SHARD
    if p.exists(): p.unlink()

NS = f"mega_shard_{RANK}"
store = DuckDBStorage(TMP_SHARD)
conn = IOWarpConnector(
    namespace=NS, storage=store,
    tag_pattern=".*", blob_pattern=".*",  # accept argo_* AND era5_*
)
conn.connect()

t1 = time.time()
print(f"[rank {RANK}] indexing {len(my_blobs)} blobs...", flush=True)
report = conn.index_from_texts(my_blobs, full_rebuild=True)
print(f"[rank {RANK}] indexed={report.indexed_files} skipped={report.skipped_files} elapsed={time.time()-t1:.1f}s", flush=True)
conn.teardown()

# Copy to /lus/flare
TARGET = OUT_DIR / f"mega_shard_{RANK:03d}.duckdb"
print(f"[rank {RANK}] copying {TMP_SHARD} ({TMP_SHARD.stat().st_size:,}b) -> {TARGET}", flush=True)
with TMP_SHARD.open("rb") as src, TARGET.open("wb") as dst:
    while True:
        b = src.read(8 * 1024 * 1024)
        if not b: break
        dst.write(b)
print(f"[rank {RANK}] DONE in {time.time()-t0:.1f}s total", flush=True)
