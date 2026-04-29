"""Per-rank distributed federated query.

Each rank opens its assigned shards (round-robin: rank R handles shards R, R+W, R+2W,...)
and runs the 3 prompts. Saves per-rank results to a JSON file.
A post-process aggregator (run on the head) sums per-prompt hits.

The DISTRIBUTED part: N ranks query N×M shards in parallel.
Total wall ≈ (M shards per rank × per-shard-time), not (total shards × per-shard-time).
"""
import json, os, sys, time
from pathlib import Path

sys.path.insert(0, "/home/sislam6/clio-search/code/src")
from clio_agentic_search.connectors.iowarp.connector import IOWarpConnector
from clio_agentic_search.storage.duckdb_store import DuckDBStorage
from clio_agentic_search.retrieval.scientific import (
    ScientificQueryOperators, NumericRangeOperator,
)

SHARDS_DIR = Path("/lus/flare/projects/gpu_hack/sislam6/clio_eval/mega_shards")
OUT_DIR    = Path("/lus/flare/projects/gpu_hack/sislam6/clio_eval/mega_query_dist")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANK = int(os.environ.get("PALS_RANKID", os.environ.get("PMI_RANK", "0")))
WORLD = int(os.environ.get("WORLD_SIZE",
            os.environ.get("PMI_SIZE",
            os.environ.get("PALS_LOCAL_SIZE", "1"))))

PROMPTS = [
    {"id": 0, "name": "argo_temp_above_30C", "min": 30.0, "unit": "degC", "qtext": "temperature above 30 degC"},
    {"id": 1, "name": "argo_temp_above_25C", "min": 25.0, "unit": "degC", "qtext": "temperature above 25 degC"},
    {"id": 2, "name": "any_temp_above_35C",  "min": 35.0, "unit": "degC", "qtext": "temperature above 35 degC"},
]


def read_proc_io():
    out = {}
    for line in open("/proc/self/io"):
        k, v = line.split(":", 1)
        out[k.strip()] = int(v.strip())
    return out


# Identify shards assigned to this rank (round-robin)
all_shards = sorted(SHARDS_DIR.glob("mega_shard_*.duckdb"))
my_shards = [all_shards[i] for i in range(RANK, len(all_shards), WORLD)]
print(f"[rank {RANK}/{WORLD}] my shards: {[s.name for s in my_shards]}", flush=True)

io_total_start = read_proc_io()
t_total = time.time()

# Open assigned shards
t0 = time.time()
connectors = []
for sp in my_shards:
    rank_n = int(sp.stem.split("_")[-1])
    store = DuckDBStorage(sp, read_only=True)
    conn = IOWarpConnector(
        namespace=f"mega_shard_{rank_n}", storage=store,
        tag_pattern=".*", blob_pattern=".*",
    )
    conn.connect()
    connectors.append(conn)
setup_s = time.time() - t0
print(f"[rank {RANK}] opened {len(connectors)} shards in {setup_s:.2f}s", flush=True)

# Per-prompt query
my_results = []
for p in PROMPTS:
    op = ScientificQueryOperators(numeric_range=NumericRangeOperator(
        minimum=p["min"], maximum=None, unit=p["unit"],
    ))
    t1 = time.time()
    n_hits = 0
    n_shards_with_hits = 0
    for c in connectors:
        try:
            hits = c.search_scientific(query=p["qtext"], top_k=1000000, operators=op)
            if hits:
                n_hits += len(hits)
                n_shards_with_hits += 1
        except Exception as e:
            print(f"[rank {RANK}] shard error: {e}", flush=True)
    q_wall = time.time() - t1
    my_results.append({
        "prompt_id": p["id"], "prompt_name": p["name"],
        "local_hits": n_hits, "local_shards_with_hits": n_shards_with_hits,
        "local_wall_s": q_wall,
    })
    print(f"[rank {RANK}] {p['name']}: hits={n_hits} shards_with_hits={n_shards_with_hits} wall={q_wall:.2f}s", flush=True)

for c in connectors:
    try: c.teardown()
    except Exception: pass

io_total_end = read_proc_io()
total_io = {k: io_total_end[k] - io_total_start[k] for k in io_total_start}
total_wall = time.time() - t_total

out = {
    "rank": RANK, "world": WORLD,
    "my_shards": [s.name for s in my_shards],
    "n_shards": len(my_shards),
    "setup_s": setup_s,
    "total_wall_s": total_wall,
    "io_delta": total_io,
    "results": my_results,
}
out_file = OUT_DIR / f"rank_{RANK:03d}_results.json"
out_file.write_text(json.dumps(out, indent=2))
print(f"[rank {RANK}] DONE total_wall={total_wall:.2f}s saved={out_file}", flush=True)
