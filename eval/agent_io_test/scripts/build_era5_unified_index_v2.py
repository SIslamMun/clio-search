"""ERA5 unified index — sample 2000 blobs, build index at /tmp,
copy into /home with explicit ddn_hdd stripe.

This version skips writing the sampled-blobs JSON to /home (not needed for
the eval; only the index file matters; sample blobs kept at /tmp for reproducibility).
"""
import json, random, subprocess, sys
from pathlib import Path

random.seed(42)
N = 2000

sys.path.insert(0, "/home/sislam6/clio-search/code/src")
from clio_agentic_search.connectors.iowarp.connector import IOWarpConnector
from clio_agentic_search.storage.duckdb_store import DuckDBStorage

BLOBS_FULL = Path("/home/sislam6/clio-search/eval/agent_io_test/era5_blobs.json")
TARGET     = Path("/home/sislam6/clio-search/eval/agent_io_test/era5_test_index_unified.duckdb")
TMP        = Path("/tmp/era5_test_index_unified.duckdb")
TMP_SAMPLE = Path("/tmp/era5_blobs_unified.json")

for ext in ("", ".wal", ".lock"):
    p = TMP.with_suffix(TMP.suffix + ext) if ext else TMP
    if p.exists():
        p.unlink()

print(f"Loading {BLOBS_FULL} (94MB)...", flush=True)
all_blobs = json.loads(BLOBS_FULL.read_text())
print(f"  {len(all_blobs)} total blobs", flush=True)

keys = list(all_blobs.keys())
random.shuffle(keys)
chosen = keys[:N]
sample = {k: all_blobs[k] for k in chosen}
print(f"  sampled {len(sample)}", flush=True)

# Save sample only at /tmp (skip /home — EDQUOT)
TMP_SAMPLE.write_text(json.dumps(sample))
print(f"  sample written: {TMP_SAMPLE} ({TMP_SAMPLE.stat().st_size} bytes)", flush=True)

print(f"Opening DuckDB at {TMP}...", flush=True)
store = DuckDBStorage(TMP)
conn = IOWarpConnector(
    namespace="agent_test_era5", storage=store,
    tag_pattern="era5_.*", blob_pattern=".*",
)
conn.connect()

print(f"Indexing {len(sample)} blobs via index_from_texts (slowest step)...", flush=True)
report = conn.index_from_texts(sample, full_rebuild=True)
print(f"  indexed={report.indexed_files} skipped={report.skipped_files}", flush=True)
conn.teardown()

# Cleanup any pre-existing target
for ext in ("", ".wal", ".lock"):
    p = TARGET.with_suffix(TARGET.suffix + ext) if ext else TARGET
    if p.exists():
        p.unlink()

# Copy /tmp index to /home with explicit ddn_hdd stripe
print(f"Copying {TMP} ({TMP.stat().st_size} bytes) -> {TARGET}...", flush=True)
subprocess.run(["lfs", "setstripe", "-p", "gecko.ddn_hdd", "-E", "-1", "-c", "2", str(TARGET)],
               check=False)
with TMP.open("rb") as src, TARGET.open("wb") as dst:
    while True:
        b = src.read(8 * 1024 * 1024)
        if not b: break
        dst.write(b)
print(f"  -> {TARGET} ({TARGET.stat().st_size} bytes)", flush=True)

tmp_wal = TMP.with_suffix(TMP.suffix + ".wal")
if tmp_wal.exists():
    tgt_wal = TARGET.with_suffix(TARGET.suffix + ".wal")
    subprocess.run(["lfs", "setstripe", "-p", "gecko.ddn_hdd", "-E", "-1", "-c", "2", str(tgt_wal)],
                   check=False)
    with tmp_wal.open("rb") as src, tgt_wal.open("wb") as dst:
        while True:
            b = src.read(8 * 1024 * 1024)
            if not b: break
            dst.write(b)
    print(f"  -> {tgt_wal} ({tgt_wal.stat().st_size} bytes)", flush=True)

print("DONE")
