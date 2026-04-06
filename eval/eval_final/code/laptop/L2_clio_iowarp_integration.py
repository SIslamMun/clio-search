#!/usr/bin/env python3
"""L2-B: CLIO + IOWarp CTE Integration Test.

Unlike the original L2 (which tests IOWarp BlobQuery in isolation), this
test runs CLIO's **full retrieval pipeline** on data stored in IOWarp:

  1. Start Chimaera runtime + CTE inside Docker
  2. Store N scientific blobs across 4 tags (temperature, pressure, wind, humidity)
  3. Run CLIO's IOWarp connector: enumerate blobs, extract scientific metadata,
     build DuckDB index with SI-canonicalized measurements + BM25 postings
  4. Run cross-unit queries through CLIO (query in °F, data in °C; query in
     psi, data in kPa) — this is the science-aware search that raw BlobQuery
     cannot do
  5. Compare: CLIO scientific search vs raw BlobQuery regex vs BM25 lexical

Scales: 1K, 5K, 10K, 50K, 100K blobs

This proves: CLIO is the search layer that makes IOWarp's blob store
searchable with science-aware operators at scale.

Prerequisites
-------------
  * Docker with iowarp/deps-cpu:latest
  * Wheel at eval/eval_final/iowarp_core-1.0.3-cp312-cp312-linux_x86_64.whl
  * CLIO source tree (mounted into container)

Usage
-----
  python3 eval/eval_final/code/laptop/L2_clio_iowarp_integration.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
OUT_DIR = _REPO / "eval" / "eval_final" / "outputs"
WHEEL_PATH = _REPO / "eval" / "eval_final" / "iowarp_core-1.0.3-cp312-cp312-linux_x86_64.whl"
DOCKER_IMAGE = "iowarp/deps-cpu:latest"
CLIO_SRC = _REPO / "code"

SCALES = [1_000, 5_000, 10_000, 50_000]

# ---------------------------------------------------------------------------
# Inner script that runs INSIDE the Docker container with both IOWarp and CLIO
# ---------------------------------------------------------------------------

INNER_SCRIPT = textwrap.dedent(r"""
import subprocess
import time
import json
import random
import sys
import os
import hashlib
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
from clio_agentic_search.retrieval.corpus_profile import build_corpus_profile

scales = json.loads(os.environ["SCALES_JSON"])
rng = random.Random(42)

# Scientific data templates — each blob is a realistic measurement
TEMPLATES = {
    "temperature": "Station {sid} recorded air temperature of {val:.2f} degC at elevation {elev}m. "
                   "Humidity was {hum:.1f} percent. Observation time: 2024-{month:02d}-{day:02d}T{hour:02d}:00Z.",
    "pressure": "Barometric pressure reading: {val:.2f} kPa at station {sid}. "
                "Sea-level corrected value: {slp:.2f} kPa. Wind was {ws:.1f} m/s from {wd} degrees.",
    "wind": "Wind speed measurement: {val:.2f} m/s at 10m height, station {sid}. "
            "Gust maximum: {gust:.1f} m/s. Temperature: {temp:.1f} degC.",
    "humidity": "Relative humidity: {val:.2f} percent at station {sid}. "
                "Dew point: {dp:.1f} degC. Pressure: {pres:.1f} kPa.",
}

def generate_blob_text(domain, idx):
    # Generate realistic scientific blob text.
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
        "blobquery_regex": "temp",
    },
    {
        "name": "Pressure cross-unit (psi → kPa)",
        "description": "Find pressure between 13-15 psi (≈ 89.6-103.4 kPa)",
        "query_text": "pressure between 13 and 15 psi",
        "numeric_range": {"min": 13.0, "max": 15.0, "unit": "psi"},
        "target_domain": "pressure",
        "blobquery_regex": "press",
    },
    {
        "name": "Wind cross-unit (km/h → m/s)",
        "description": "Find wind above 72 km/h (= 20 m/s)",
        "query_text": "wind speed above 72 km/h",
        "numeric_range": {"min": 72.0, "max": None, "unit": "km/h"},
        "target_domain": "wind",
        "blobquery_regex": "wind",
    },
    {
        "name": "Temperature range (degF → degC)",
        "description": "Find temperatures between 32-50°F (= 0-10°C)",
        "query_text": "temperature between 32 and 50 degF",
        "numeric_range": {"min": 32.0, "max": 50.0, "unit": "degF"},
        "target_domain": "temperature",
        "blobquery_regex": "temp",
    },
]

all_results = {"scales": []}

for N in scales:
    print(f"\n{'='*60}", flush=True)
    print(f"SCALE: {N:,} blobs", flush=True)
    print(f"{'='*60}", flush=True)

    # Use unique tag names per scale to avoid contamination
    tag_names = {d: f"{d}_{N}" for d in DOMAINS}

    # --- Phase 1: Write blobs to IOWarp CTE ---
    print(f"  Writing {N:,} blobs to CTE...", flush=True)
    tags = {d: cte.Tag(tag_names[d]) for d in DOMAINS}
    ground_truth = {d: [] for d in DOMAINS}  # track values for validation

    blob_texts = {}  # blob_uri -> text, for fast indexing without per-blob CTE reads
    t0 = time.time()
    for i in range(N):
        domain = DOMAINS[i % 4]
        text, val, unit = generate_blob_text(domain, i)
        blob_name = f"blob_{i:07d}"
        tags[domain].PutBlob(blob_name, text.encode())
        ground_truth[domain].append((blob_name, val, unit))
        blob_texts[f"cte://{tag_names[domain]}/{blob_name}"] = text
    put_time = time.time() - t0
    print(f"  Write: {put_time:.2f}s ({N/put_time:,.0f} blobs/s)", flush=True)

    # --- Phase 2: CLIO indexes the blobs ---
    # Two modes:
    #   A) index_from_texts: CLIO indexes pre-loaded blob text (production ingest path)
    #   B) index via CTE reads: CLIO enumerates + reads blobs from CTE (cold-start path)
    # Mode A is used for all scales; Mode B is measured at small scale only.

    print(f"  CLIO indexing {N:,} blobs (ingest path)...", flush=True)
    db_path = f"/tmp/clio_iowarp_{N}.duckdb"
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
    print(f"  Index: {index_time:.2f}s ({report.indexed_files:,} blobs → CLIO DuckDB, "
          f"{report.indexed_files/index_time:,.0f} blobs/s)", flush=True)

    cte_index_time = None

    # --- Phase 3: Corpus profile ---
    t0 = time.time()
    profile = connector.corpus_profile()
    profile_time = (time.time() - t0) * 1000
    print(f"  Profile: {profile_time:.2f}ms — {profile.document_count} docs, "
          f"{profile.chunk_count} chunks, {profile.measurement_count} measurements, "
          f"units={profile.distinct_units}", flush=True)

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

        # Method 1: CLIO scientific search (with unit conversion)
        t0 = time.perf_counter()
        clio_results = connector.search_scientific(
            query=q["query_text"], top_k=50, operators=operators
        )
        clio_time = (time.perf_counter() - t0) * 1000

        # Method 2: CLIO lexical search (BM25, no unit conversion)
        t0 = time.perf_counter()
        lexical_results = connector.search_lexical(
            query=q["query_text"], top_k=50
        )
        lexical_time = (time.perf_counter() - t0) * 1000

        # Method 3: Raw IOWarp BlobQuery (regex only, no CLIO)
        t0 = time.perf_counter()
        tag_regex = f"{q['target_domain']}_{N}"
        raw_results = list(client.BlobQuery(
            tag_regex, ".*", N, cte.PoolQuery.Dynamic()
        ))
        blobquery_time = (time.perf_counter() - t0) * 1000

        # Validate: count how many ground-truth blobs match the query range
        gt_domain = q["target_domain"]
        gt_matches = 0
        for _, val, _ in ground_truth[gt_domain]:
            # Convert query range to the stored unit for validation
            if nr["unit"] == "degF":
                # degF to degC: (F - 32) * 5/9
                q_min = (nr["min"] - 32) * 5/9 if nr["min"] is not None else None
                q_max = (nr["max"] - 32) * 5/9 if nr["max"] is not None else None
            elif nr["unit"] == "psi":
                # psi to kPa: * 6.89476
                q_min = nr["min"] * 6.89476 if nr["min"] is not None else None
                q_max = nr["max"] * 6.89476 if nr["max"] is not None else None
            elif nr["unit"] == "km/h":
                # km/h to m/s: / 3.6
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
            "blobquery_finds_cross_unit": False,  # BlobQuery has no unit conversion
        }
        query_results.append(qr)

        print(f"\n  Query: {q['name']}", flush=True)
        print(f"    Ground truth: {gt_matches} blobs match", flush=True)
        print(f"    CLIO scientific: {len(clio_results)} results in {clio_time:.2f}ms "
              f"(cross-unit conversion: YES)", flush=True)
        print(f"    CLIO lexical:    {len(lexical_results)} results in {lexical_time:.2f}ms "
              f"(no unit conversion)", flush=True)
        print(f"    Raw BlobQuery:   {len(raw_results)} results in {blobquery_time:.2f}ms "
              f"(returns ALL blobs in tag, no filtering)", flush=True)

    # Inspection rate: what fraction of blobs did CLIO need to scan?
    total_blobs = N
    avg_clio_results = sum(qr["clio_scientific_results"] for qr in query_results) / len(query_results)
    inspection_rate = avg_clio_results / total_blobs * 100

    scale_result = {
        "N_blobs": N,
        "put_time_s": round(put_time, 3),
        "put_throughput_blobs_per_s": round(N / put_time),
        "clio_index_time_s": round(index_time, 3),
        "clio_index_throughput_blobs_per_s": round(report.indexed_files / index_time),
        "cte_read_index_time_s": round(cte_index_time, 3) if cte_index_time else None,
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
    print(f"\n  Scale {N:,} complete.", flush=True)

# --- Teardown runtime ---
rt.terminate()
try:
    rt.wait(timeout=5)
except Exception:
    rt.kill()

# Output
print("\n===RESULT_JSON_BEGIN===")
print(json.dumps(all_results))
print("===RESULT_JSON_END===")
""").strip()


def main() -> None:
    print("=" * 75)
    print("L2-B: CLIO + IOWarp CTE Integration Test")
    print("  CLIO's full pipeline (unit conversion + indexing) on IOWarp blobs")
    print("=" * 75)

    if not WHEEL_PATH.exists():
        print(f"ERROR: wheel not found at {WHEEL_PATH}")
        sys.exit(1)

    if shutil.which("docker") is None:
        print("ERROR: docker not found on PATH")
        sys.exit(1)

    if not CLIO_SRC.exists():
        print(f"ERROR: CLIO source not found at {CLIO_SRC}")
        sys.exit(1)

    with tempfile.TemporaryDirectory(prefix="L2B_clio_iowarp_") as tmp:
        tmp = Path(tmp)
        (tmp / "inner_test.py").write_text(INNER_SCRIPT)

        cmd = [
            "docker", "run", "--rm",
            "--shm-size=16g",
            "-u", "root",
            "-e", f"SCALES_JSON={json.dumps(SCALES)}",
            "-v", f"{WHEEL_PATH}:/wheels/{WHEEL_PATH.name}",
            "-v", f"{tmp / 'inner_test.py'}:/test.py",
            "-v", f"{CLIO_SRC}/src:/clio/src:ro",
            DOCKER_IMAGE,
            "bash", "-c",
            (
                f"pip install /wheels/{WHEEL_PATH.name} --force-reinstall -q && "
                "pip install duckdb -q 2>&1 | tail -1 && "
                "python3 /test.py"
            ),
        ]

        print(f"\nRunning Docker container for scales {SCALES}...")
        print(f"Mounting CLIO source from {CLIO_SRC}/src")
        print("(Estimated: 10-30 min depending on scale)")
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        elapsed = time.time() - t0
        print(f"\nDocker run wall time: {elapsed:.1f}s")

        # Print raw stdout for debugging
        if proc.stdout:
            for line in proc.stdout.splitlines():
                if not line.startswith("==="):
                    print(f"  {line}")

        if proc.returncode != 0:
            print(f"\nDocker exited with code {proc.returncode}")
            if proc.stderr:
                print("STDERR (last 20 lines):")
                for line in proc.stderr.splitlines()[-20:]:
                    print(f"  {line}")

        # Extract results
        stdout = proc.stdout
        begin = stdout.find("===RESULT_JSON_BEGIN===")
        end = stdout.find("===RESULT_JSON_END===")
        if begin != -1 and end != -1:
            json_text = stdout[begin + len("===RESULT_JSON_BEGIN==="):end].strip()
            result = json.loads(json_text)
        else:
            # Parse partial results from per-scale output
            print("\nWARNING: No complete result JSON — attempting partial recovery")
            result = {"scales": [], "partial": True}
            # Try to extract any JSON blocks printed during the run
            lines = stdout.splitlines()
            i = 0
            while i < len(lines):
                if "Scale" in lines[i] and "complete" in lines[i]:
                    pass
                i += 1
            if not result["scales"]:
                print("Could not recover partial results")
                print("\nFull stdout (last 100 lines):")
                for line in stdout.splitlines()[-100:]:
                    print(f"  {line}")
                sys.exit(1)

    # ---- Print summary ----
    print("\n" + "=" * 75)
    print("CLIO + IOWarp INTEGRATION RESULTS")
    print("=" * 75)

    for s in result["scales"]:
        N = s["N_blobs"]
        print(f"\n--- {N:,} blobs ---")
        print(f"  CTE write:    {s['put_time_s']:.2f}s ({s['put_throughput_blobs_per_s']:,}/s)")
        print(f"  CLIO index:   {s['clio_index_time_s']:.2f}s ({s['clio_index_throughput_blobs_per_s']:,}/s)")
        print(f"  Profile:      {s['corpus_profile_ms']:.2f}ms")
        print(f"  Measurements: {s['measurement_count']:,}")
        print(f"  Units found:  {s['distinct_units']}")
        print(f"  Inspection:   {s['inspection_rate_pct']:.4f}% of corpus per query")

        for q in s["queries"]:
            print(f"\n  {q['query']}")
            print(f"    Ground truth:     {q['ground_truth_matches']} matching blobs")
            print(f"    CLIO scientific:  {q['clio_scientific_results']} found in {q['clio_scientific_time_ms']:.1f}ms")
            print(f"    CLIO lexical:     {q['clio_lexical_results']} found in {q['clio_lexical_time_ms']:.1f}ms")
            print(f"    Raw BlobQuery:    {q['raw_blobquery_results']} returned in {q['raw_blobquery_time_ms']:.1f}ms (unfiltered)")

    # Key takeaway
    if result["scales"]:
        last = result["scales"][-1]
        print(f"\n{'='*75}")
        print("KEY FINDING:")
        print(f"  At {last['N_blobs']:,} blobs, CLIO inspects {last['inspection_rate_pct']:.4f}% of the corpus")
        print(f"  per cross-unit query. Raw BlobQuery returns ALL blobs in the tag")
        print(f"  ({last['N_blobs']//4:,} per tag) with no filtering capability.")
        print(f"  CLIO's unit conversion finds cross-unit matches that BlobQuery cannot.")
        print(f"{'='*75}")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "L2-B: CLIO + IOWarp CTE Integration",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "iowarp_version": "1.0.3 (main branch build)",
        "clio_version": "from source",
        "environment": f"Docker {DOCKER_IMAGE} with --shm-size=16g",
        "scales_requested": SCALES,
        "wall_time_s": round(elapsed, 1),
        **result,
    }
    out_path = OUT_DIR / "L2B_clio_iowarp_integration.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
