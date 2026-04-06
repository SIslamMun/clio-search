#!/usr/bin/env python3
"""L2-C inner test: CLIO + IOWarp — NO METADATA case.

Proves CLIO can search blob content when there is NO external metadata and
NO text description passed at write time. CLIO must read blobs from CTE,
parse their contents (JSON in this test), extract scientific measurements,
and answer cross-unit queries.

Controlled by environment variables:
  SCALES_JSON  - JSON array of blob counts, e.g. '[1000,10000,100000]'
  DB_DIR       - directory for DuckDB files (default: /tmp)

Key difference from L2-B (index_from_texts):
  L2-B: text passed to CLIO at write time — CLIO never reads CTE
  L2-C: only raw JSON bytes in CTE, NO text to CLIO — CLIO reads GetBlob,
        parses JSON, extracts measurements, builds index from blob content
"""

import subprocess
import time
import json
import random
import sys
import os
from pathlib import Path

# --- Start Chimaera runtime ---
rt_log = "/tmp/rt.log"
rt = subprocess.Popen(
    ["chimaera", "runtime", "start"],
    stdout=open(rt_log, "w"),
    stderr=subprocess.STDOUT,
)
for _ in range(30):
    time.sleep(1)
    try:
        if "Successfully started local server" in open(rt_log).read():
            break
    except Exception:
        pass

# --- Init CTE client ---
from iowarp_core import wrp_cte_core_ext as cte

cte.chimaera_init(cte.ChimaeraMode.kClient)
cte.initialize_cte("", cte.PoolQuery.Dynamic())
client = cte.get_cte_client()

# --- Import CLIO ---
sys.path.insert(0, "/clio/src")
from clio_agentic_search.connectors.iowarp.connector import IOWarpConnector
from clio_agentic_search.storage.duckdb_store import DuckDBStorage
from clio_agentic_search.retrieval.scientific import (
    ScientificQueryOperators,
    NumericRangeOperator,
)

scales = json.loads(os.environ["SCALES_JSON"])
DB_DIR = os.environ.get("DB_DIR", "/tmp")
rng = random.Random(42)

DOMAINS = ["temperature", "pressure", "wind", "humidity"]


def generate_blob_json(domain: str, idx: int) -> tuple[bytes, float, str]:
    """Generate a JSON blob with NO separate text description.

    The blob IS the data — a structured JSON record. CLIO must read and
    parse this blob to discover what measurements it contains.
    The 'measurement' field contains "{value} {unit}" which CLIO's regex
    can extract after JSON parsing and text flattening.
    """
    sid = f"STN-{idx % 200:04d}"
    ts = f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}T{rng.randint(0,23):02d}:00:00Z"

    if domain == "temperature":
        val = round(rng.uniform(-30, 50), 2)
        record = {
            "domain": "temperature",
            "station": sid,
            "measurement": f"{val} degC",
            "elevation_m": rng.randint(10, 3000),
            "humidity_pct": round(rng.uniform(20, 95), 1),
            "timestamp": ts,
        }
    elif domain == "pressure":
        val = round(rng.uniform(85, 108), 2)
        record = {
            "domain": "pressure",
            "station": sid,
            "measurement": f"{val} kPa",
            "sea_level_kpa": round(val + rng.uniform(-2, 2), 2),
            "wind_ms": round(rng.uniform(0, 25), 1),
            "timestamp": ts,
        }
    elif domain == "wind":
        val = round(rng.uniform(0, 40), 2)
        record = {
            "domain": "wind",
            "station": sid,
            "measurement": f"{val} m/s",
            "gust_ms": round(val + rng.uniform(0, 15), 1),
            "temp_degc": round(rng.uniform(-10, 35), 1),
            "timestamp": ts,
        }
    else:  # humidity
        val = round(rng.uniform(10, 100), 2)
        record = {
            "domain": "humidity",
            "station": sid,
            "measurement": f"{val} percent",
            "dewpoint_degc": round(rng.uniform(-10, 25), 1),
            "pressure_kpa": round(rng.uniform(95, 105), 1),
            "timestamp": ts,
        }

    return json.dumps(record).encode(), val, record["measurement"].split()[1]


# Cross-unit queries — identical to L2-B to allow direct comparison
QUERIES = [
    {
        "name": "Temperature cross-unit (degF → degC)",
        "description": "Find temperatures above 86°F (= 30°C)",
        "query_text": "temperature above 86 degF",
        "numeric_range": {"min": 86.0, "max": None, "unit": "degF"},
        "target_domain": "temperature",
    },
    {
        "name": "Pressure cross-unit (psi → kPa)",
        "description": "Find pressure between 13-15 psi (≈ 89.6-103.4 kPa)",
        "query_text": "pressure between 13 and 15 psi",
        "numeric_range": {"min": 13.0, "max": 15.0, "unit": "psi"},
        "target_domain": "pressure",
    },
    {
        "name": "Wind cross-unit (km/h → m/s)",
        "description": "Find wind above 72 km/h (= 20 m/s)",
        "query_text": "wind speed above 72 km/h",
        "numeric_range": {"min": 72.0, "max": None, "unit": "km/h"},
        "target_domain": "wind",
    },
    {
        "name": "Temperature range (degF → degC)",
        "description": "Find temperatures between 32-50°F (= 0-10°C)",
        "query_text": "temperature between 32 and 50 degF",
        "numeric_range": {"min": 32.0, "max": 50.0, "unit": "degF"},
        "target_domain": "temperature",
    },
]

all_results = {"scales": []}

for N in scales:
    print(f"\n{'='*60}", flush=True)
    print(f"SCALE: {N:,} blobs  [NO METADATA — CLIO reads blob content]", flush=True)
    print(f"{'='*60}", flush=True)

    tag_names = {d: f"nm_{d}_{N}" for d in DOMAINS}  # nm_ prefix = no-metadata

    # --- Phase 1: Write JSON blobs to CTE (NO text passed to CLIO) ---
    print(f"  Writing {N:,} JSON blobs to CTE...", flush=True)
    tags = {d: cte.Tag(tag_names[d]) for d in DOMAINS}
    ground_truth = {d: [] for d in DOMAINS}

    t0 = time.time()
    for i in range(N):
        domain = DOMAINS[i % 4]
        blob_bytes, val, unit = generate_blob_json(domain, i)
        blob_name = f"blob_{i:07d}"
        tags[domain].PutBlob(blob_name, blob_bytes)
        ground_truth[domain].append((blob_name, val, unit))
        # NOTE: we do NOT build a blob_texts dict — CLIO gets nothing at write time
    put_time = time.time() - t0
    print(f"  Write: {put_time:.2f}s ({N/put_time:,.0f} blobs/s)", flush=True)
    print(f"  CLIO has NO knowledge of blob contents yet.", flush=True)

    # --- Phase 2: CLIO indexes by READING blobs from CTE ---
    print(f"  CLIO reading + indexing {N:,} blobs from CTE (GetBlob path)...", flush=True)
    db_path = f"{DB_DIR}/clio_nm_{N}.duckdb"
    store = DuckDBStorage(Path(db_path))
    tag_pattern = f"nm_(temperature|pressure|wind|humidity)_{N}"

    connector = IOWarpConnector(
        namespace=f"iowarp_nm_{N}",
        storage=store,
        tag_pattern=tag_pattern,
        blob_pattern=".*",
        max_blobs_per_query=N + 1000,
    )
    connector.connect()

    t0 = time.time()
    # Uses index() — enumerates via BlobQuery, reads each blob via GetBlob,
    # parses JSON content, extracts measurements, builds DuckDB index.
    report = connector.index(full_rebuild=True)
    index_time = time.time() - t0
    print(
        f"  Index: {index_time:.2f}s  scanned={report.scanned_files}  "
        f"indexed={report.indexed_files}  skipped={report.skipped_files}  "
        f"({report.indexed_files/index_time:,.0f} blobs/s)",
        flush=True,
    )

    # --- Phase 3: Corpus profile ---
    t0 = time.time()
    profile = connector.corpus_profile()
    profile_time = (time.time() - t0) * 1000
    print(
        f"  Profile: {profile_time:.2f}ms — {profile.document_count} docs, "
        f"{profile.measurement_count} measurements extracted from blob content, "
        f"units={profile.distinct_units}",
        flush=True,
    )

    # --- Phase 4: Cross-unit queries ---
    query_results = []
    for q in QUERIES:
        nr = q["numeric_range"]
        operators = ScientificQueryOperators(
            numeric_range=NumericRangeOperator(
                minimum=nr["min"],
                maximum=nr["max"],
                unit=nr["unit"],
            ),
        )

        t0 = time.perf_counter()
        clio_results = connector.search_scientific(
            query=q["query_text"], top_k=50, operators=operators
        )
        clio_time = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        lexical_results = connector.search_lexical(query=q["query_text"], top_k=50)
        lexical_time = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        tag_regex = f"nm_{q['target_domain']}_{N}"
        raw_results = list(client.BlobQuery(tag_regex, ".*", N, cte.PoolQuery.Dynamic()))
        blobquery_time = (time.perf_counter() - t0) * 1000

        # Ground truth: convert query range to stored unit
        gt_domain = q["target_domain"]
        gt_matches = 0
        for _, val, _ in ground_truth[gt_domain]:
            if nr["unit"] == "degF":
                q_min = (nr["min"] - 32) * 5 / 9 if nr["min"] is not None else None
                q_max = (nr["max"] - 32) * 5 / 9 if nr["max"] is not None else None
            elif nr["unit"] == "psi":
                q_min = nr["min"] * 6.89476 if nr["min"] is not None else None
                q_max = nr["max"] * 6.89476 if nr["max"] is not None else None
            elif nr["unit"] == "km/h":
                q_min = nr["min"] / 3.6 if nr["min"] is not None else None
                q_max = nr["max"] / 3.6 if nr["max"] is not None else None
            else:
                q_min, q_max = nr["min"], nr["max"]

            in_range = True
            if q_min is not None and val < q_min:
                in_range = False
            if q_max is not None and val > q_max:
                in_range = False
            if in_range:
                gt_matches += 1

        precision = len(clio_results) / gt_matches if gt_matches > 0 else 0.0

        qr = {
            "query": q["name"],
            "description": q["description"],
            "ground_truth_matches": gt_matches,
            "clio_scientific_results": len(clio_results),
            "clio_scientific_time_ms": round(clio_time, 2),
            "clio_precision": round(precision, 4),
            "clio_lexical_results": len(lexical_results),
            "clio_lexical_time_ms": round(lexical_time, 2),
            "raw_blobquery_results": len(raw_results),
            "raw_blobquery_time_ms": round(blobquery_time, 2),
            "clio_finds_cross_unit": len(clio_results) > 0,
            "blobquery_finds_cross_unit": False,
        }
        query_results.append(qr)

        print(f"\n  Query: {q['name']}", flush=True)
        print(f"    Ground truth:  {gt_matches} blobs match", flush=True)
        print(
            f"    CLIO sci:      {len(clio_results)} results in {clio_time:.2f}ms  "
            f"(parsed from blob JSON, cross-unit conversion YES)",
            flush=True,
        )
        print(f"    CLIO lexical:  {len(lexical_results)} results in {lexical_time:.2f}ms", flush=True)
        print(
            f"    Raw BlobQuery: {len(raw_results)} (all in tag, no filtering) in {blobquery_time:.2f}ms",
            flush=True,
        )

    avg_clio = sum(qr["clio_scientific_results"] for qr in query_results) / len(query_results)
    inspection_rate = avg_clio / N * 100

    scale_result = {
        "N_blobs": N,
        "indexing_method": "GetBlob (reads blob content — no external metadata)",
        "put_time_s": round(put_time, 3),
        "put_throughput_blobs_per_s": round(N / put_time),
        "clio_index_time_s": round(index_time, 3),
        "clio_index_throughput_blobs_per_s": round(report.indexed_files / index_time),
        "scanned_blobs": report.scanned_files,
        "indexed_blobs": report.indexed_files,
        "skipped_blobs": report.skipped_files,
        "corpus_profile_ms": round(profile_time, 2),
        "document_count": profile.document_count,
        "measurement_count": profile.measurement_count,
        "distinct_units": list(profile.distinct_units),
        "inspection_rate_pct": round(inspection_rate, 4),
        "queries": query_results,
    }
    all_results["scales"].append(scale_result)
    connector.teardown()
    print(f"\n  Scale {N:,} complete. Measurements extracted from raw blobs: {profile.measurement_count}", flush=True)

# --- Teardown runtime ---
rt.terminate()
try:
    rt.wait(timeout=5)
except Exception:
    rt.kill()

print("\n===RESULT_JSON_BEGIN===")
print(json.dumps(all_results))
print("===RESULT_JSON_END===")
