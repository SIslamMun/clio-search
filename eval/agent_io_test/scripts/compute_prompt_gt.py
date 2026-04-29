"""Pre-compute ground truth for 5 multi-prompt grid prompts on the unified
200-blob Argo corpus.

Two answer columns per prompt:
  - 'naive_surface': the literal answer to the natural-language question
                     using only the surface-most measurement of each profile.
                     This is what arm A *should* return if the LLM correctly
                     interprets "surface" as "shallowest pressure point".
  - 'clio_any_depth': what CLIO's search_scientific returns when given the
                      matching NumericRangeOperator on the chunked index —
                      i.e. matches if ANY measurement in the profile satisfies.
                      This is what arm B and arm C return.

We expect them to differ for prompts where the depth scope matters.
"""
import json, re, sys
from pathlib import Path

sys.path.insert(0, "/home/sislam6/jarvis-work")
sys.path.insert(0, "/home/sislam6/clio-search/code/src")

from argo_to_iowarp_blobs import extract_profile

DATA_DIR = Path("/home/sislam6/clio-search/eval/agent_io_test/workspaces/arm_a/data")
INDEX = Path("/home/sislam6/clio-search/eval/agent_io_test/test_index_unified.duckdb")

PROMPTS = [
    {"id": 0, "name": "surface_temp_gt_30",
     "nl": "Count how many Argo profile NetCDF files in the data/ directory have a surface temperature above 30 degrees Celsius.",
     "clio_min_si": 30.0, "clio_unit": "degC", "field": "temperature", "scope": "surface"},
    {"id": 1, "name": "surface_temp_gt_25",
     "nl": "Count how many Argo profile NetCDF files in the data/ directory have a surface temperature above 25 degrees Celsius.",
     "clio_min_si": 25.0, "clio_unit": "degC", "field": "temperature", "scope": "surface"},
    {"id": 2, "name": "any_depth_temp_gt_25",
     "nl": "Count how many Argo profile NetCDF files have temperature above 25 degrees Celsius at any depth.",
     "clio_min_si": 25.0, "clio_unit": "degC", "field": "temperature", "scope": "any"},
    {"id": 3, "name": "any_depth_salinity_gt_36",
     "nl": "Count how many Argo profile NetCDF files have salinity above 36 PSU at any depth.",
     "clio_min_si": 36.0, "clio_unit": "PSU", "field": "salinity", "scope": "any"},
    {"id": 4, "name": "deep_pressure_gt_1500",
     "nl": "Count how many Argo profile NetCDF files have a deep pressure above 1500 dbar.",
     "clio_min_si": 1500.0, "clio_unit": "dbar", "field": "pressure", "scope": "deep"},
]


# ---- naive surface-style scan -----------------------------------------------
# Use the same blob-text format the unified corpus was built from. Each profile
# blob has "surface: pressure X dbar, temperature Y degC, salinity Z PSU."
# Plus "mid:" and "deep:" sections.

SCOPE_RE = {
    "surface": re.compile(r"surface:\s*pressure\s+([\d.]+)\s*dbar,\s*temperature\s+(-?[\d.]+)\s*degC,\s*salinity\s+([\d.]+)\s*PSU"),
    "mid":     re.compile(r"mid:\s*pressure\s+([\d.]+)\s*dbar,\s*temperature\s+(-?[\d.]+)\s*degC,\s*salinity\s+([\d.]+)\s*PSU"),
    "deep":    re.compile(r"deep:\s*pressure\s+([\d.]+)\s*dbar,\s*temperature\s+(-?[\d.]+)\s*degC,\s*salinity\s+([\d.]+)\s*PSU"),
}
FIELD_INDEX = {"pressure": 0, "temperature": 1, "salinity": 2}


def naive_count(prompts, files):
    """Count profiles satisfying the literal scope+field+threshold predicate."""
    counts = {p["id"]: 0 for p in prompts}
    n_scanned = 0
    for f in files:
        try:
            prof = extract_profile(f)
        except Exception:
            continue
        if prof is None:
            continue
        n_scanned += 1
        text = prof["text"]
        for p in prompts:
            scope = p["scope"] if p["scope"] in SCOPE_RE else "surface"
            # for "any" scope, check all 3 layers
            scopes_to_check = ("surface", "mid", "deep") if p["scope"] == "any" else (scope,)
            f_idx = FIELD_INDEX[p["field"]]
            hit = False
            for s in scopes_to_check:
                m = SCOPE_RE[s].search(text)
                if m:
                    val = float(m.group(f_idx + 1))
                    if val > p["clio_min_si"]:
                        hit = True
                        break
            if hit:
                counts[p["id"]] += 1
    return counts, n_scanned


# ---- CLIO any-depth structured query -----------------------------------------

def clio_count(prompts, index_path):
    """For each prompt, run CLIO search_scientific on the unified index."""
    import shutil, tempfile
    from clio_agentic_search.connectors.iowarp.connector import IOWarpConnector
    from clio_agentic_search.storage.duckdb_store import DuckDBStorage
    from clio_agentic_search.retrieval.scientific import (
        ScientificQueryOperators, NumericRangeOperator,
    )

    counts = {}
    for p in prompts:
        tmp = Path(tempfile.gettempdir()) / f"idx_gt_{p['id']}.duckdb"
        if tmp.exists():
            tmp.unlink()
        shutil.copyfile(index_path, tmp)
        store = DuckDBStorage(tmp, read_only=True)
        conn = IOWarpConnector(
            namespace="agent_test", storage=store,
            tag_pattern="argo_.*", blob_pattern=".*",
        )
        conn.connect()
        op = ScientificQueryOperators(numeric_range=NumericRangeOperator(
            minimum=p["clio_min_si"], maximum=None, unit=p["clio_unit"],
        ))
        hits = conn.search_scientific(query=p["nl"], top_k=10000, operators=op)
        # Each profile has one chunk in the unified index, so chunk count = profile count.
        counts[p["id"]] = len(hits)
        conn.teardown()
    return counts


def main():
    print("scanning data/ for naive ground truth...", flush=True)
    files = sorted(DATA_DIR.glob("*.nc"))
    naive, n = naive_count(PROMPTS, files)
    print(f"  scanned {n}/{len(files)} files", flush=True)

    print("running CLIO search on unified index...", flush=True)
    clio = clio_count(PROMPTS, INDEX)

    print()
    print(f"{'id':>3} {'name':<28} {'naive':>7} {'clio':>7}  -- prompt --")
    out = []
    for p in PROMPTS:
        i = p["id"]
        nv, cv = naive[i], clio[i]
        out.append({**p, "gt_naive": nv, "gt_clio": cv})
        print(f"{i:>3} {p['name']:<28} {nv:>7} {cv:>7}  '{p['nl']}'")

    json.dump(out, open("/tmp/prompts_with_gt.json", "w"), indent=2)
    print()
    print("Saved /tmp/prompts_with_gt.json")


if __name__ == "__main__":
    main()
