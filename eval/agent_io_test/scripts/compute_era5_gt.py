"""Pre-compute GT for ERA5 prompt grid using the new 2000-blob unified index.

Naive surface-style scan = scan the 2000 sampled blobs textually for the field's
mean/max/min value, count those satisfying the predicate. This mirrors what an
LLM with raw blob/text access would do.

CLIO any-depth = run search_scientific against the unified ERA5 index.
"""
import json, re, sys
from pathlib import Path

sys.path.insert(0, "/home/sislam6/clio-search/code/src")

INDEX = Path("/home/sislam6/clio-search/eval/agent_io_test/era5_test_index_unified.duckdb")
SAMPLE = Path("/tmp/era5_blobs_unified.json")  # the 2000-blob sample we just indexed

PROMPTS = [
    # (id, name, NL, scope, field, threshold, op, clio_min, clio_unit)
    {"id": 0, "name": "any_t2m_above_30C",
     "nl": "Count how many ERA5 cells in the unified blob set have a 2m temperature above 30 degrees Celsius. Print only the final count.",
     "field": "t2m_max", "thresh": 303.15, "op": ">",
     "clio_min": 30.0, "clio_unit": "degC", "qtext": "temperature above 30 degC"},
    {"id": 1, "name": "any_t2m_above_35C",
     "nl": "Count how many ERA5 cells in the unified blob set have a 2m temperature above 35 degrees Celsius. Print only the final count.",
     "field": "t2m_max", "thresh": 308.15, "op": ">",
     "clio_min": 35.0, "clio_unit": "degC", "qtext": "temperature above 35 degC"},
    {"id": 2, "name": "low_pressure_lt_95000Pa",
     "nl": "Count how many ERA5 cells in the unified blob set have a surface pressure mean below 95000 Pascals. Print only the final count.",
     "field": "sp_mean", "thresh": 95000.0, "op": "<",
     "clio_min": None, "clio_max": 95000.0, "clio_unit": "Pa", "qtext": "surface pressure below 95000 Pa"},
]

# Naive scan: blob text format is e.g.
#   "ERA5 reanalysis cell ... on 2022-01-01:
#    2m temperature mean 268.69 K (-4.46 degC), min 252.19 K (-20.96 degC), max 281.45 K (8.30 degC).
#    2m dewpoint mean 266.75 K ...
#    Surface pressure mean 92988.05 Pa ..."
RE_T2M_MAX  = re.compile(r"2m temperature.*?max\s+([\d.]+)\s*K", re.S)
RE_T2M_MIN  = re.compile(r"2m temperature.*?min\s+([\d.]+)\s*K", re.S)
RE_T2M_MEAN = re.compile(r"2m temperature mean\s+([\d.]+)\s*K", re.S)
RE_SP_MEAN  = re.compile(r"Surface pressure mean\s+([\d.]+)\s*Pa", re.S)

FIELDRE = {
    "t2m_max": RE_T2M_MAX,
    "t2m_min": RE_T2M_MIN,
    "t2m_mean": RE_T2M_MEAN,
    "sp_mean":  RE_SP_MEAN,
}


def naive_count(prompts, sample_blobs):
    counts = {p["id"]: 0 for p in prompts}
    for uri, text in sample_blobs.items():
        for p in prompts:
            r = FIELDRE[p["field"]]
            m = r.search(text)
            if m:
                v = float(m.group(1))
                if p["op"] == ">" and v > p["thresh"]:
                    counts[p["id"]] += 1
                elif p["op"] == "<" and v < p["thresh"]:
                    counts[p["id"]] += 1
    return counts


def clio_count(prompts, index_path):
    import shutil, tempfile
    from clio_agentic_search.connectors.iowarp.connector import IOWarpConnector
    from clio_agentic_search.storage.duckdb_store import DuckDBStorage
    from clio_agentic_search.retrieval.scientific import (
        ScientificQueryOperators, NumericRangeOperator,
    )

    counts = {}
    for p in prompts:
        tmp = Path(tempfile.gettempdir()) / f"era5_idx_gt_{p['id']}.duckdb"
        if tmp.exists(): tmp.unlink()
        shutil.copyfile(index_path, tmp)
        store = DuckDBStorage(tmp, read_only=True)
        conn = IOWarpConnector(
            namespace="agent_test_era5", storage=store,
            tag_pattern="era5_.*", blob_pattern=".*",
        )
        conn.connect()
        op = ScientificQueryOperators(numeric_range=NumericRangeOperator(
            minimum=p.get("clio_min"),
            maximum=p.get("clio_max"),
            unit=p["clio_unit"],
        ))
        hits = conn.search_scientific(query=p["qtext"], top_k=10000, operators=op)
        # number of distinct chunks satisfying predicate. Each ERA5 blob = 1 chunk.
        counts[p["id"]] = len(hits)
        conn.teardown()
    return counts


def main():
    sample_blobs = json.load(open(SAMPLE))
    print(f"naive scan over {len(sample_blobs)} sampled blobs", flush=True)
    naive = naive_count(PROMPTS, sample_blobs)
    print("CLIO query against index", flush=True)
    clio = clio_count(PROMPTS, INDEX)

    print()
    print(f"{'id':>3} {'name':<30} {'naive':>7} {'clio':>7}")
    out = []
    for p in PROMPTS:
        out.append({**p, "gt_naive": naive[p["id"]], "gt_clio": clio[p["id"]]})
        print(f"{p['id']:>3} {p['name']:<30} {naive[p['id']]:>7} {clio[p['id']]:>7}")

    json.dump(out, open("/tmp/era5_prompts_with_gt.json", "w"), indent=2)
    print("\n/tmp/era5_prompts_with_gt.json")


if __name__ == "__main__":
    main()
