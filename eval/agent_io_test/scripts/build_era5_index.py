#!/usr/bin/env python3
"""Build a CLIO index over the ERA5 measurement-text blobs (~315K), plus
compute ground truth: how many of those blobs have 2m mean temp > 30 degC.

Mirrors build_test_index.py (the Argo build script) but uses pre-parsed
era5_blobs.json from era5_to_iowarp_blobs.py instead of re-parsing NetCDFs.
"""
import sys, json, time, re
from pathlib import Path

BLOBS_PATH = Path("/home/sislam6/clio-search/eval/agent_io_test/era5_blobs.json")
INDEX_PATH = Path("/home/sislam6/clio-search/eval/agent_io_test/era5_test_index.duckdb")
GT_PATH = Path("/home/sislam6/clio-search/eval/agent_io_test/era5_groundtruth.json")

t0 = time.time()

# Step 1: load pre-parsed ERA5 blob texts
print("[1/3] loading ERA5 blob texts...", flush=True)
blobs = json.loads(BLOBS_PATH.read_text())
print(f"  loaded {len(blobs):,} blobs from {BLOBS_PATH}", flush=True)

# Step 2: build CLIO index over those blobs (using kServer mode)
print("[2/3] building CLIO index in DuckDB...", flush=True)
from iowarp_core import wrp_cte_core_ext as cte
cte.chimaera_init(cte.ChimaeraMode.kServer)
cte.initialize_cte("", cte.PoolQuery.Dynamic())

from clio_agentic_search.connectors.iowarp.connector import IOWarpConnector
from clio_agentic_search.storage.duckdb_store import DuckDBStorage

if INDEX_PATH.exists():
    INDEX_PATH.unlink()
store = DuckDBStorage(INDEX_PATH)
conn = IOWarpConnector(
    namespace="agent_test_era5",
    storage=store,
    tag_pattern="era5_.*",
    blob_pattern=".*",
)
conn.connect()
report = conn.index_from_texts(blobs, full_rebuild=True)
print(f"  indexed {report.indexed_files} into {INDEX_PATH}", flush=True)

# Step 3: compute ground truth -- how many ERA5 blobs report 2m mean temp > 30 degC?
print("[3/3] computing ground truth...", flush=True)
# Match: "2m temperature mean <K> K (<C> degC)" -- extract the degC value.
TEMP_C = re.compile(
    r"2m\s+temperature\s+mean\s+-?\d+\.?\d*\s*K\s*\(\s*(-?\d+\.?\d*)\s*degC\s*\)",
    re.IGNORECASE,
)
gt_count = 0
gt_uris = []
for uri, text in blobs.items():
    m = TEMP_C.search(text)
    if m and float(m.group(1)) > 30.0:
        gt_count += 1
        gt_uris.append(uri)
print(f"  ground truth: {gt_count:,} of {len(blobs):,} have 2m mean temp > 30 degC", flush=True)
GT_PATH.write_text(json.dumps({"count": gt_count, "total": len(blobs), "uris": gt_uris}))

elapsed = time.time() - t0
print("\nDONE")
print(f"  index: {INDEX_PATH}")
print(f"  blobs JSON: {BLOBS_PATH}")
print(f"  ground truth: {GT_PATH}")
print(f"  ground-truth count = {gt_count}")
print(f"  total wall time = {elapsed:.1f} sec")
