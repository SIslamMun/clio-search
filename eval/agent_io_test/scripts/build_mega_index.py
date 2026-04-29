"""Mega index — combine ALL Argo blobs + ALL ERA5 blobs into one corpus.

Two namespaces in one index:
  - Argo blobs (~4928, tag pattern argo_*)
  - ERA5 blobs (~315648, tag pattern era5_*)

Total expected: ~320,576 blobs.

Builds at /tmp then moves to /lus/flare. Long-running (could be 5-15 minutes).
"""
import json, time, sys, shutil, subprocess
from pathlib import Path

sys.path.insert(0, "/home/sislam6/clio-search/code/src")
from clio_agentic_search.connectors.iowarp.connector import IOWarpConnector
from clio_agentic_search.storage.duckdb_store import DuckDBStorage

ARGO_BLOBS = Path("/lus/flare/projects/gpu_hack/sislam6/clio_eval/big_corpus/argo_big_blobs.json")
ERA5_BLOBS = Path("/home/sislam6/clio-search/eval/agent_io_test/era5_blobs.json")
OUT_DIR    = Path("/lus/flare/projects/gpu_hack/sislam6/clio_eval/mega_corpus")
OUT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_OUT  = OUT_DIR / "mega_index.duckdb"
TMP_INDEX  = Path("/tmp/mega_index.duckdb")

# Cleanup
for ext in ("", ".wal", ".lock"):
    p = TMP_INDEX.with_suffix(TMP_INDEX.suffix + ext) if ext else TMP_INDEX
    if p.exists(): p.unlink()

# ---- Load both blob sets ----
print(f"Loading argo blobs from {ARGO_BLOBS}...", flush=True)
argo = json.loads(ARGO_BLOBS.read_text())
print(f"  {len(argo)} argo blobs", flush=True)

print(f"Loading era5 blobs from {ERA5_BLOBS} (94MB, may be slow)...", flush=True)
era5 = json.loads(ERA5_BLOBS.read_text())
print(f"  {len(era5)} era5 blobs", flush=True)

total = len(argo) + len(era5)
print(f"  combined: {total} blobs", flush=True)

# ---- Index argo namespace first ----
print(f"Building DuckDB index at {TMP_INDEX}...", flush=True)
store = DuckDBStorage(TMP_INDEX)
conn_argo = IOWarpConnector(
    namespace="mega_argo", storage=store,
    tag_pattern="argo_.*", blob_pattern=".*",
)
conn_argo.connect()

t0 = time.time()
print(f"  [argo] indexing {len(argo)} blobs...", flush=True)
report = conn_argo.index_from_texts(argo, full_rebuild=True)
print(f"  [argo] indexed={report.indexed_files} skipped={report.skipped_files}  elapsed={time.time()-t0:.1f}s", flush=True)
conn_argo.teardown()

# ---- Now index era5 namespace into same DB ----
conn_era5 = IOWarpConnector(
    namespace="mega_era5", storage=store,
    tag_pattern="era5_.*", blob_pattern=".*",
)
conn_era5.connect()

t0 = time.time()
print(f"  [era5] indexing {len(era5)} blobs (this is the heavy step)...", flush=True)
report = conn_era5.index_from_texts(era5, full_rebuild=False)  # don't clear argo
print(f"  [era5] indexed={report.indexed_files} skipped={report.skipped_files}  elapsed={time.time()-t0:.1f}s", flush=True)
conn_era5.teardown()

# Verify counts
import duckdb
c = duckdb.connect(str(TMP_INDEX), read_only=True)
chunks = c.execute("SELECT namespace, COUNT(*) FROM chunks GROUP BY namespace").fetchall()
print(f"Index verification:")
for ns, n in chunks:
    print(f"  {ns}: {n} chunks", flush=True)
c.close()

# Move to flare
for ext in ("", ".wal", ".lock"):
    p = INDEX_OUT.with_suffix(INDEX_OUT.suffix + ext) if ext else INDEX_OUT
    if p.exists(): p.unlink()
print(f"Copying {TMP_INDEX} ({TMP_INDEX.stat().st_size:,} bytes) -> {INDEX_OUT}", flush=True)
with TMP_INDEX.open("rb") as src, INDEX_OUT.open("wb") as dst:
    while True:
        b = src.read(16 * 1024 * 1024)
        if not b: break
        dst.write(b)
print(f"  -> {INDEX_OUT} ({INDEX_OUT.stat().st_size:,} bytes)", flush=True)
print("DONE")
