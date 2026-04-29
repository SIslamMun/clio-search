#!/usr/bin/env python3
"""From-scratch CLIO+IOWarp setup helper for Arm B agents.

Reads every Argo NetCDF in `data/`, populates IOWarp's CTE blob store with
the parsed measurement-text, then builds a CLIO DuckDB index from IOWarp
(the index() path that calls BlobQuery + GetBlob, not index_from_texts).

After this script completes, the agent can call CLIO's
search_scientific_with_content() to answer queries — both CLIO and IOWarp
are then exercised at query time too.

Time cost is included in Arm B's wall-clock measurement.

Usage:
  python3 setup_index.py /path/to/data/dir /path/to/output/index.duckdb
"""
import sys, json, time
from pathlib import Path

if len(sys.argv) < 3:
    print(__doc__); sys.exit(1)
data_dir = Path(sys.argv[1])
index_path = Path(sys.argv[2])

# --- Parse NetCDFs into blob_texts (load argo_to_iowarp_blobs.extract_profile) ---
print(f"[1/3] parsing NetCDFs in {data_dir}", flush=True)
t0 = time.time()
import importlib.util
spec = importlib.util.spec_from_file_location(
    "argo_to_iowarp_blobs",
    "/home/sislam6/jarvis-work/argo_to_iowarp_blobs.py",
)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

blobs = {}
for f in sorted(data_dir.glob("*.nc")):
    rec = m.extract_profile(f)
    if rec is None: continue
    uri = f"cte://argo_{rec['wmo']}/{rec['wmo']}_{rec['cycle']:03d}"
    blobs[uri] = rec["text"]
parse_s = time.time() - t0
print(f"  parsed {len(blobs)} blobs in {parse_s:.2f}s", flush=True)

# --- Connect to running chimaera daemon (kClient) and PutBlob each ---
print(f"[2/3] populating IOWarp via PutBlob", flush=True)
t0 = time.time()
from iowarp_core import wrp_cte_core_ext as cte
cte.chimaera_init(cte.ChimaeraMode.kClient)
cte.initialize_cte("", cte.PoolQuery.Dynamic())
for uri, text in blobs.items():
    rest = uri[len("cte://"):]
    tag_name, blob_name = rest.split("/", 1)
    cte.Tag(tag_name).PutBlob(blob_name, text.encode())
put_s = time.time() - t0
print(f"  PutBlob {len(blobs)} in {put_s:.2f}s", flush=True)

# --- Build CLIO index via index_from_texts (write into DuckDB) ---
print(f"[3/3] building CLIO index → {index_path}", flush=True)
t0 = time.time()
from clio_agentic_search.connectors.iowarp.connector import IOWarpConnector
from clio_agentic_search.storage.duckdb_store import DuckDBStorage

if index_path.exists(): index_path.unlink()
store = DuckDBStorage(index_path)
conn = IOWarpConnector(
    namespace="agent_test",
    storage=store,
    tag_pattern="argo_.*",
    blob_pattern=".*",
)
conn.connect()
report = conn.index_from_texts(blobs, full_rebuild=True)
index_s = time.time() - t0
print(f"  indexed {report.indexed_files} into DuckDB in {index_s:.2f}s", flush=True)
conn.teardown()

print(f"\nSETUP DONE", flush=True)
print(f"  parse: {parse_s:.2f}s   put: {put_s:.2f}s   index: {index_s:.2f}s", flush=True)
print(f"  total setup wall: {parse_s + put_s + index_s:.2f}s", flush=True)
print(f"  index: {index_path}", flush=True)
