#!/usr/bin/env python3
"""Build the v3 CLIO+IOWarp test corpus over 1000 Argo NetCDFs + GT for 5 queries."""
import sys, json, time
from pathlib import Path

V3_DIR = Path("/home/sislam6/clio-search/eval/agent_io_test/v3")
V3_DIR.mkdir(parents=True, exist_ok=True)
WORKSPACE_DATA_A = V3_DIR / "workspaces" / "arm_a" / "data"
WORKSPACE_DATA_A.mkdir(parents=True, exist_ok=True)
WORKSPACE_DATA_B = V3_DIR / "workspaces" / "arm_b" / "data"
WORKSPACE_DATA_B.mkdir(parents=True, exist_ok=True)

ARGO_RAW = Path("/home/sislam6/jarvis-work/argo_data/raw")
N_FILES = 1000

print(f"[1/4] staging {N_FILES} NetCDFs in workspaces", flush=True)
ncs = sorted(p for p in ARGO_RAW.rglob("*.nc"))[:N_FILES]
for p in ncs:
    (WORKSPACE_DATA_A / p.name).unlink(missing_ok=True)
    (WORKSPACE_DATA_A / p.name).symlink_to(p)
    (WORKSPACE_DATA_B / p.name).unlink(missing_ok=True)
    (WORKSPACE_DATA_B / p.name).symlink_to(p)
print(f"  staged {len(ncs)} files", flush=True)

print(f"[2/4] parsing NetCDFs", flush=True)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "argo_to_iowarp_blobs", "/home/sislam6/jarvis-work/argo_to_iowarp_blobs.py"
)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
blobs = {}
t0 = time.time()
for nc_path in ncs:
    rec = m.extract_profile(nc_path)
    if rec is None: continue
    uri = f"cte://argo_{rec['wmo']}/{rec['wmo']}_{rec['cycle']:03d}"
    blobs[uri] = rec["text"]
print(f"  parsed {len(blobs)} blobs in {time.time()-t0:.1f}s", flush=True)
(V3_DIR / "blobs.json").write_text(json.dumps(blobs))

print(f"[3/4] building CLIO index", flush=True)
from iowarp_core import wrp_cte_core_ext as cte
cte.chimaera_init(cte.ChimaeraMode.kServer)
cte.initialize_cte("", cte.PoolQuery.Dynamic())
from clio_agentic_search.connectors.iowarp.connector import IOWarpConnector
from clio_agentic_search.storage.duckdb_store import DuckDBStorage
idx = V3_DIR / "test_index.duckdb"
if idx.exists(): idx.unlink()
store = DuckDBStorage(idx)
conn = IOWarpConnector(namespace="agent_test_v3", storage=store,
                       tag_pattern="argo_.*", blob_pattern=".*")
conn.connect()
report = conn.index_from_texts(blobs, full_rebuild=True)
print(f"  indexed {report.indexed_files} blobs into {idx}", flush=True)

print(f"[4/4] ground truth for 5 queries", flush=True)
import re
TEMP_C = re.compile(r"temperature\s+(-?\d+\.?\d*)\s*degC", re.IGNORECASE)
PRES_DBAR = re.compile(r"pressure\s+(-?\d+\.?\d*)\s*dbar", re.IGNORECASE)
SAL = re.compile(r"salinity\s+(-?\d+\.?\d*)\s*PSU", re.IGNORECASE)
def parse(t):
    o = {}
    m = TEMP_C.search(t); o["temp_c"] = float(m.group(1)) if m else None
    m = PRES_DBAR.search(t); o["pres_dbar"] = float(m.group(1)) if m else None
    m = SAL.search(t); o["sal_psu"] = float(m.group(1)) if m else None
    return o
parsed = {u: parse(t) for u, t in blobs.items()}
QUERIES = [
    ("Q1_temp_above_30C",   lambda v: v["temp_c"] is not None and v["temp_c"] > 30),
    ("Q2_temp_0_to_10C",    lambda v: v["temp_c"] is not None and 0 <= v["temp_c"] <= 10),
    ("Q3_pres_9_to_10dbar", lambda v: v["pres_dbar"] is not None and 9 <= v["pres_dbar"] <= 10),
    ("Q4_sal_33_to_37PSU",  lambda v: v["sal_psu"] is not None and 33 <= v["sal_psu"] <= 37),
    ("Q5_sal_above_35PSU",  lambda v: v["sal_psu"] is not None and v["sal_psu"] > 35),
]
gt = {}
for name, pred in QUERIES:
    matching = [u for u, v in parsed.items() if pred(v)]
    gt[name] = {"count": len(matching), "uris": matching}
    print(f"  {name}: {len(matching)} of {len(blobs)}", flush=True)
(V3_DIR / "groundtruth.json").write_text(json.dumps(gt, indent=2))

print(f"\nDONE")
print(f"  blobs:       {V3_DIR / 'blobs.json'} ({len(blobs)} entries)")
print(f"  index:       {V3_DIR / 'test_index.duckdb'}")
print(f"  groundtruth: {V3_DIR / 'groundtruth.json'}")
