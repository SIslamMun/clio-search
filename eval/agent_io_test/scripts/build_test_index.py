#!/usr/bin/env python3
"""Build a tiny CLIO index over the 200 NetCDFs in the test workspaces, plus
compute ground truth: how many of those 200 profiles have surface temp > 30 degC.
"""
import sys, json, subprocess
from pathlib import Path

WORK_A = Path("/tmp/agent_arm_a/data")
WORK_B = Path("/tmp/agent_arm_b/data")
INDEX_PATH = Path("/tmp/agent_test_index.duckdb")
BLOBS_PATH = Path("/tmp/agent_test_blobs.json")
GT_PATH = Path("/tmp/agent_test_groundtruth.json")

# Step 1: parse the 200 NetCDFs into blob_texts using existing loader
print("[1/3] parsing 200 NetCDFs to measurement text...", flush=True)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "argo_to_iowarp_blobs",
    "/home/sislam6/jarvis-work/argo_to_iowarp_blobs.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

blobs = {}
for f in sorted(WORK_A.glob("*.nc")):
    rec = m.extract_profile(f)
    if rec is None:
        continue
    uri = f"cte://argo_{rec['wmo']}/{rec['wmo']}_{rec['cycle']:03d}"
    blobs[uri] = rec["text"]
print(f"  parsed {len(blobs)} blobs (out of 200)", flush=True)
BLOBS_PATH.write_text(json.dumps(blobs))

# Step 2: build CLIO index over those blobs (using kClient mode -- chimaera daemon must be running)
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
    namespace="agent_test",
    storage=store,
    tag_pattern="argo_.*",
    blob_pattern=".*",
)
conn.connect()
report = conn.index_from_texts(blobs, full_rebuild=True)
print(f"  indexed {report.indexed_files} into {INDEX_PATH}", flush=True)

# Step 3: compute ground truth — how many of these 200 profiles have surface temp > 30 degC?
print("[3/3] computing ground truth...", flush=True)
import re
TEMP_C = re.compile(r"temperature\s+(-?\d+\.?\d*)\s*degC", re.IGNORECASE)
gt_count = 0
gt_uris = []
for uri, text in blobs.items():
    m = TEMP_C.search(text)
    if m and float(m.group(1)) > 30.0:
        gt_count += 1
        gt_uris.append(uri)
print(f"  ground truth: {gt_count} of {len(blobs)} have surface temp > 30 degC", flush=True)
GT_PATH.write_text(json.dumps({"count": gt_count, "total": len(blobs), "uris": gt_uris}))

print("\nDONE")
print(f"  index: {INDEX_PATH}")
print(f"  blobs JSON: {BLOBS_PATH}")
print(f"  ground truth: {GT_PATH}")
print(f"  ground-truth count = {gt_count}")
