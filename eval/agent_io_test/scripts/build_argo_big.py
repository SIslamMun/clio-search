"""Build the BIG Argo corpus and index — all 5000 NetCDFs.

Outputs:
  /lus/flare/.../argo_big_blobs.json     — extracted blobs from all 5000 files
  /lus/flare/.../argo_big_index.duckdb   — CLIO index over those blobs
  /lus/flare/.../argo_big_prompts.json   — pre-computed GT for the 3 prompts
"""
import json, re, sys, time, shutil, subprocess
from pathlib import Path

sys.path.insert(0, "/home/sislam6/jarvis-work")
sys.path.insert(0, "/home/sislam6/clio-search/code/src")

from argo_to_iowarp_blobs import extract_profile
from clio_agentic_search.connectors.iowarp.connector import IOWarpConnector
from clio_agentic_search.storage.duckdb_store import DuckDBStorage

ROOT = Path("/home/sislam6/jarvis-work/argo_data/raw")
OUT_DIR = Path("/lus/flare/projects/gpu_hack/sislam6/clio_eval/big_corpus")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BLOBS_OUT  = OUT_DIR / "argo_big_blobs.json"
INDEX_OUT  = OUT_DIR / "argo_big_index.duckdb"
TMP_INDEX  = Path("/tmp/argo_big_index.duckdb")
GT_OUT     = OUT_DIR / "argo_big_prompts.json"
DATA_LINKS = OUT_DIR / "data_links"

# ---- 1) Extract blobs from all 5000 NetCDFs ----
print(f"[1/4] Scanning {ROOT}...", flush=True)
files = sorted(ROOT.rglob("*.nc"))
print(f"  found {len(files)} files", flush=True)

blobs = {}
n_skipped = 0
t0 = time.time()
for i, f in enumerate(files):
    try:
        prof = extract_profile(f)
    except Exception:
        prof = None
    if prof is None:
        n_skipped += 1
        continue
    wmo = prof["wmo"]; cyc = prof["cycle"]
    uri = f"cte://argo_{wmo}/{wmo}_{cyc:03d}"
    blobs[uri] = prof["text"]
    if (i + 1) % 500 == 0:
        print(f"  parsed {i+1}/{len(files)} ({len(blobs)} ok)  elapsed={time.time()-t0:.1f}s", flush=True)

print(f"  total: {len(blobs)} ok, {n_skipped} skipped", flush=True)

# ---- 2) Persist blobs JSON (for the naive arm + reproducibility) ----
print(f"[2/4] Saving blobs JSON to {BLOBS_OUT}", flush=True)
BLOBS_OUT.write_text(json.dumps(blobs))
print(f"  {BLOBS_OUT.stat().st_size:,} bytes", flush=True)

# ---- 3) Build CLIO index ----
for ext in ("", ".wal", ".lock"):
    p = TMP_INDEX.with_suffix(TMP_INDEX.suffix + ext) if ext else TMP_INDEX
    if p.exists(): p.unlink()

print(f"[3/4] Building CLIO index at {TMP_INDEX}", flush=True)
store = DuckDBStorage(TMP_INDEX)
conn = IOWarpConnector(
    namespace="agent_test_argo_big", storage=store,
    tag_pattern="argo_.*", blob_pattern=".*",
)
conn.connect()
report = conn.index_from_texts(blobs, full_rebuild=True)
print(f"  indexed={report.indexed_files} skipped={report.skipped_files}", flush=True)
conn.teardown()

# Move index to /lus/flare
for ext in ("", ".wal", ".lock"):
    p = INDEX_OUT.with_suffix(INDEX_OUT.suffix + ext) if ext else INDEX_OUT
    if p.exists(): p.unlink()
print(f"  copying {TMP_INDEX} -> {INDEX_OUT}", flush=True)
with TMP_INDEX.open("rb") as src, INDEX_OUT.open("wb") as dst:
    while True:
        b = src.read(8 * 1024 * 1024)
        if not b: break
        dst.write(b)
print(f"  -> {INDEX_OUT} ({INDEX_OUT.stat().st_size:,} bytes)", flush=True)

# ---- 4) Compute GT for the 3 prompts ----
print(f"[4/4] Computing GT for the 3 prompts...", flush=True)

# naive: scan blob text for surface/mid/deep field values
SCOPE_RE = {
    "surface": re.compile(r"surface:\s*pressure\s+([\d.]+)\s*dbar,\s*temperature\s+(-?[\d.]+)\s*degC"),
    "mid":     re.compile(r"mid:\s*pressure\s+([\d.]+)\s*dbar,\s*temperature\s+(-?[\d.]+)\s*degC"),
    "deep":    re.compile(r"deep:\s*pressure\s+([\d.]+)\s*dbar,\s*temperature\s+(-?[\d.]+)\s*degC"),
}

PROMPTS = [
    {"id": 0, "name": "surface_temp_gt_30",
     "scope": "surface", "thresh": 30.0,
     "clio_min": 30.0, "clio_unit": "degC", "qtext": "temperature above 30 degC"},
    {"id": 1, "name": "surface_temp_gt_25",
     "scope": "surface", "thresh": 25.0,
     "clio_min": 25.0, "clio_unit": "degC", "qtext": "temperature above 25 degC"},
    {"id": 2, "name": "any_depth_temp_gt_25",
     "scope": "any", "thresh": 25.0,
     "clio_min": 25.0, "clio_unit": "degC", "qtext": "temperature above 25 degC any depth"},
]

# Naive count
naive_counts = {p["id"]: 0 for p in PROMPTS}
for uri, text in blobs.items():
    for p in PROMPTS:
        scopes = ("surface", "mid", "deep") if p["scope"] == "any" else (p["scope"],)
        hit = False
        for s in scopes:
            m = SCOPE_RE[s].search(text)
            if m and float(m.group(2)) > p["thresh"]:
                hit = True; break
        if hit:
            naive_counts[p["id"]] += 1

# CLIO count
from clio_agentic_search.retrieval.scientific import (
    ScientificQueryOperators, NumericRangeOperator,
)
clio_counts = {}
for p in PROMPTS:
    tmp = Path("/tmp") / f"big_gt_{p['id']}.duckdb"
    if tmp.exists(): tmp.unlink()
    shutil.copyfile(TMP_INDEX, tmp)
    store = DuckDBStorage(tmp, read_only=True)
    conn = IOWarpConnector(
        namespace="agent_test_argo_big", storage=store,
        tag_pattern="argo_.*", blob_pattern=".*",
    )
    conn.connect()
    op = ScientificQueryOperators(numeric_range=NumericRangeOperator(
        minimum=p["clio_min"], maximum=None, unit=p["clio_unit"],
    ))
    hits = conn.search_scientific(query=p["qtext"], top_k=1000000, operators=op)
    clio_counts[p["id"]] = len(hits)
    conn.teardown()
    tmp.unlink()

print()
print(f"{'id':>3} {'name':<28} {'naive':>7} {'clio':>7}")
out = []
for p in PROMPTS:
    print(f"{p['id']:>3} {p['name']:<28} {naive_counts[p['id']]:>7} {clio_counts[p['id']]:>7}")
    out.append({**p, "gt_naive": naive_counts[p["id"]], "gt_clio": clio_counts[p["id"]]})

GT_OUT.write_text(json.dumps(out, indent=2))
print(f"GT saved: {GT_OUT}")
print()
print("DONE")
