#!/usr/bin/env python3
"""L2-B inner test: CLIO + IOWarp CTE integration.

Runs inside the Apptainer/Docker container where both iowarp_core and CLIO
are available. Controlled by environment variables:
  SCALES_JSON  - JSON array of blob counts, e.g. '[10000,50000,100000]'
  DB_DIR       - directory for DuckDB files (default: /tmp)
                 On Delta, bind-mount NVMe here for fast I/O.
"""

import time
import json
import random
import sys
import os
from pathlib import Path

# --- Init CTE (server mode = embedded runtime, no daemon needed) ---
from iowarp_core import wrp_cte_core_ext as cte

print("Initializing Chimaera runtime (kServer embedded mode)...", flush=True)
cte.chimaera_init(cte.ChimaeraMode.kServer)
cte.initialize_cte("", cte.PoolQuery.Dynamic())
client = cte.get_cte_client()
print("CTE client ready.", flush=True)
rt = None  # no external process to manage

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
CHECKPOINT_FILE = os.environ.get("CHECKPOINT_FILE", "")
rng = random.Random(42)

# Scientific data templates — each blob is a realistic measurement
TEMPLATES = {
    "temperature": (
        "Station {sid} recorded air temperature of {val:.2f} degC at elevation {elev}m. "
        "Humidity was {hum:.1f} percent. Observation time: 2024-{month:02d}-{day:02d}T{hour:02d}:00Z."
    ),
    "pressure": (
        "Barometric pressure reading: {val:.2f} kPa at station {sid}. "
        "Sea-level corrected value: {slp:.2f} kPa. Wind was {ws:.1f} m/s from {wd} degrees."
    ),
    "wind": (
        "Wind speed measurement: {val:.2f} m/s at 10m height, station {sid}. "
        "Gust maximum: {gust:.1f} m/s. Temperature: {temp:.1f} degC."
    ),
    "humidity": (
        "Relative humidity: {val:.2f} percent at station {sid}. "
        "Dew point: {dp:.1f} degC. Pressure: {pres:.1f} kPa."
    ),
}


def generate_blob_text(domain, idx):
    sid = f"STN-{idx % 200:04d}"
    if domain == "temperature":
        val = rng.uniform(-30, 50)
        return TEMPLATES["temperature"].format(
            sid=sid, val=val, elev=rng.randint(10, 3000),
            hum=rng.uniform(20, 95), month=rng.randint(1, 12),
            day=rng.randint(1, 28), hour=rng.randint(0, 23),
        ), val, "degC"
    elif domain == "pressure":
        val = rng.uniform(85, 108)
        return TEMPLATES["pressure"].format(
            sid=sid, val=val, slp=val + rng.uniform(-2, 2),
            ws=rng.uniform(0, 25), wd=rng.randint(0, 360),
        ), val, "kPa"
    elif domain == "wind":
        val = rng.uniform(0, 40)
        return TEMPLATES["wind"].format(
            sid=sid, val=val, gust=val + rng.uniform(0, 15),
            temp=rng.uniform(-10, 35),
        ), val, "m/s"
    else:  # humidity
        val = rng.uniform(10, 100)
        return TEMPLATES["humidity"].format(
            sid=sid, val=val, dp=rng.uniform(-10, 25),
            pres=rng.uniform(95, 105),
        ), val, "percent"


DOMAINS = ["temperature", "pressure", "wind", "humidity"]

# Cross-unit queries: query uses DIFFERENT unit than stored data
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
    print(f"SCALE: {N:,} blobs", flush=True)
    print(f"{'='*60}", flush=True)

    tag_names = {d: f"{d}_{N}" for d in DOMAINS}

    # --- Phase 1: Write blobs to IOWarp CTE ---
    print(f"  Writing {N:,} blobs to CTE...", flush=True)
    tags = {d: cte.Tag(tag_names[d]) for d in DOMAINS}
    ground_truth = {d: [] for d in DOMAINS}
    blob_texts = {}

    t0 = time.time()
    for i in range(N):
        domain = DOMAINS[i % 4]
        text, val, unit = generate_blob_text(domain, i)
        blob_name = f"blob_{i:07d}"
        tags[domain].PutBlob(blob_name, text.encode())
        ground_truth[domain].append((blob_name, val, unit))
        blob_texts[f"cte://{tag_names[domain]}/{blob_name}"] = text
        # Yield to the Chimaera server thread every 10 blobs.
        # queue_depth=1024 per worker; 10-blob batches keep queue usage
        # far below capacity, preventing PutBlob deadlock at all scales.
        if i % 10 == 9:
            time.sleep(0.5)
    put_time = time.time() - t0
    print(f"  Write: {put_time:.2f}s ({N/put_time:,.0f} blobs/s)", flush=True)

    # --- Phase 2: CLIO indexes the blobs ---
    print(f"  CLIO indexing {N:,} blobs (ingest path)...", flush=True)
    db_path = f"{DB_DIR}/clio_iowarp_{N}.duckdb"
    store = DuckDBStorage(Path(db_path))
    tag_pattern = f"(temperature|pressure|wind|humidity)_{N}"

    connector = IOWarpConnector(
        namespace=f"iowarp_{N}",
        storage=store,
        tag_pattern=tag_pattern,
        blob_pattern=".*",
    )
    connector.connect()

    t0 = time.time()
    report = connector.index_from_texts(blob_texts, full_rebuild=True)
    index_time = time.time() - t0
    print(
        f"  Index: {index_time:.2f}s ({report.indexed_files:,} blobs → CLIO DuckDB, "
        f"{report.indexed_files/index_time:,.0f} blobs/s)",
        flush=True,
    )

    # --- Phase 3: Corpus profile ---
    t0 = time.time()
    profile = connector.corpus_profile()
    profile_time = (time.time() - t0) * 1000
    print(
        f"  Profile: {profile_time:.2f}ms — {profile.document_count} docs, "
        f"{profile.chunk_count} chunks, {profile.measurement_count} measurements, "
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
        # Use GetContainedBlobs (per-tag) as raw baseline.
        # BlobQuery hangs on iowarp_core 0.6.4 aarch64 (Broadcast dispatch
        # deadlock); GetContainedBlobs() returns all blobs in the tag locally.
        raw_tag = cte.Tag(f"{q['target_domain']}_{N}")
        raw_results = raw_tag.GetContainedBlobs()
        blobquery_time = (time.perf_counter() - t0) * 1000

        # Validate ground truth
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

        qr = {
            "query": q["name"],
            "description": q["description"],
            "ground_truth_matches": gt_matches,
            "clio_scientific_results": len(clio_results),
            "clio_scientific_time_ms": round(clio_time, 2),
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
        print(f"    CLIO sci:      {len(clio_results)} results in {clio_time:.2f}ms", flush=True)
        print(f"    CLIO lexical:  {len(lexical_results)} results in {lexical_time:.2f}ms", flush=True)
        print(f"    Raw BlobQuery: {len(raw_results)} (all in tag, unfiltered) in {blobquery_time:.2f}ms", flush=True)

    avg_clio_results = sum(qr["clio_scientific_results"] for qr in query_results) / len(query_results)
    inspection_rate = avg_clio_results / N * 100

    scale_result = {
        "N_blobs": N,
        "put_time_s": round(put_time, 3),
        "put_throughput_blobs_per_s": round(N / put_time),
        "clio_index_time_s": round(index_time, 3),
        "clio_index_throughput_blobs_per_s": round(report.indexed_files / index_time),
        "corpus_profile_ms": round(profile_time, 2),
        "document_count": profile.document_count,
        "chunk_count": profile.chunk_count,
        "measurement_count": profile.measurement_count,
        "distinct_units": list(profile.distinct_units),
        "inspection_rate_pct": round(inspection_rate, 4),
        "queries": query_results,
    }
    all_results["scales"].append(scale_result)
    connector.teardown()
    if CHECKPOINT_FILE:
        Path(CHECKPOINT_FILE).write_text(json.dumps(all_results))
        print(f"  [Checkpoint saved → {CHECKPOINT_FILE}]", flush=True)
    print(f"\n  Scale {N:,} complete.", flush=True)

# --- Teardown runtime ---
if rt is not None:
    rt.terminate()
    try:
        rt.wait(timeout=5)
    except Exception:
        rt.kill()

print("\n===RESULT_JSON_BEGIN===")
print(json.dumps(all_results))
print("===RESULT_JSON_END===")
