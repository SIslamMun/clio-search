#!/usr/bin/env python3
"""Scaling study: measure CLIO performance from 1K to 100K synthetic blobs.

Generates synthetic scientific documents with measurements at increasing
scales, indexes each into CLIO, measures profile time, index time, query
time. Plots scaling behavior.

Output: eval/eval_eve/outputs/scale_test.json
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_CODE_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "code"
sys.path.insert(0, str(_CODE_ROOT / "src"))

from clio_agentic_search.models.contracts import (
    ChunkRecord,
    DocumentRecord,
    EmbeddingRecord,
    MetadataRecord,
)
from clio_agentic_search.indexing.scientific import canonicalize_measurement
from clio_agentic_search.retrieval.corpus_profile import build_corpus_profile
from clio_agentic_search.retrieval.scientific import NumericRangeOperator
from clio_agentic_search.storage.contracts import DocumentBundle, FileIndexState
from clio_agentic_search.storage.duckdb_store import DuckDBStorage

OUT_DIR = _CODE_ROOT.parent / "eval" / "eval_eve" / "outputs"

SCALES = [1_000, 5_000, 10_000, 25_000, 50_000]

UNITS = [
    ("degC", 0, 100, "temperature"),
    ("kPa", 90, 500, "pressure"),
    ("km/h", 0, 200, "wind speed"),
    ("percent", 0, 100, "humidity"),
]
DOMAINS = [
    "atmospheric", "oceanographic", "geological", "biological",
    "chemical", "physical", "climatological", "hydrological",
]


def generate_synthetic_blob(i: int, rng: random.Random) -> dict[str, Any]:
    """Generate one synthetic scientific document with realistic measurements."""
    unit_name, vmin, vmax, var_name = rng.choice(UNITS)
    domain = rng.choice(DOMAINS)
    value = round(rng.uniform(vmin, vmax), 2)
    doc_id = f"doc_{i:08d}"
    title = f"{domain.capitalize()} {var_name} dataset {i}"
    text = (
        f"# {title}\n\n"
        f"This {domain} dataset contains {var_name} measurements. "
        f"Recorded value: {value} {unit_name}. "
        f"Collected at station {rng.randint(1, 1000)}, "
        f"timestamp {2020 + rng.randint(0, 5)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}."
    )
    return {
        "doc_id": doc_id,
        "text": text,
        "value": value,
        "unit": unit_name,
        "domain": domain,
        "var_name": var_name,
    }


def bulk_insert(storage: DuckDBStorage, namespace: str, blobs: list[dict[str, Any]]) -> None:
    """Bulk insert blobs into DuckDB using direct SQL for speed."""
    conn = storage._require_connection()

    # Prepare bulk data
    now_ns = time.time_ns()
    docs_data = []
    chunks_data = []
    meas_data = []
    meta_data = []

    for b in blobs:
        doc_id = b["doc_id"]
        text = b["text"]
        chunk_id = f"{doc_id}_c0"
        checksum = hashlib.sha256(text.encode()).hexdigest()[:16]

        docs_data.append((namespace, doc_id, f"synth://{doc_id}", checksum, now_ns))
        chunks_data.append((namespace, chunk_id, doc_id, 0, text, 0, len(text)))

        # Canonicalize the measurement
        try:
            canonical_val, canonical_unit = canonicalize_measurement(
                b["value"], b["unit"]
            )
            meas_data.append((namespace, chunk_id, canonical_unit, canonical_val,
                              b["unit"], b["value"]))
        except (ValueError, KeyError):
            pass

        # Metadata
        meta_data.append((namespace, doc_id, "document", "domain", b["domain"]))
        meta_data.append((namespace, chunk_id, "chunk", "scientific.measurements",
                          f"{b['unit']}|{b['value']}"))

    # Bulk insert
    conn.executemany(
        "INSERT OR REPLACE INTO documents(namespace, document_id, uri, checksum, modified_at_ns) VALUES (?, ?, ?, ?, ?)",
        docs_data,
    )
    conn.executemany(
        "INSERT OR REPLACE INTO chunks(namespace, chunk_id, document_id, chunk_index, text, start_offset, end_offset) VALUES (?, ?, ?, ?, ?, ?, ?)",
        chunks_data,
    )
    if meas_data:
        conn.executemany(
            "INSERT INTO scientific_measurements(namespace, chunk_id, canonical_unit, canonical_value, raw_unit, raw_value) VALUES (?, ?, ?, ?, ?, ?)",
            meas_data,
        )
    conn.executemany(
        "INSERT OR REPLACE INTO metadata(namespace, record_id, scope, key, value) VALUES (?, ?, ?, ?, ?)",
        meta_data,
    )


def test_scale(scale: int, tmpdir: Path) -> dict[str, Any]:
    print(f"\n--- Scale: {scale:,} blobs ---", flush=True)
    rng = random.Random(42 + scale)  # deterministic

    # --- Generate ---
    t0 = time.time()
    blobs = [generate_synthetic_blob(i, rng) for i in range(scale)]
    gen_time = time.time() - t0
    print(f"  Generated {scale:,} blobs in {gen_time:.2f}s", flush=True)

    # --- Index (bulk insert) ---
    db_path = tmpdir / f"scale_{scale}.duckdb"
    storage = DuckDBStorage(database_path=db_path)
    storage.connect()

    t0 = time.time()
    # Batch inserts to avoid SQL parameter limits
    batch_size = 2000
    for i in range(0, len(blobs), batch_size):
        bulk_insert(storage, "scale_ns", blobs[i : i + batch_size])
        if (i // batch_size) % 5 == 0 and i > 0:
            print(f"    ... {i:,}/{len(blobs):,} ({time.time()-t0:.1f}s)", flush=True)
    index_time = time.time() - t0
    print(f"  Indexed in {index_time:.2f}s ({scale/index_time:.0f} blobs/s)", flush=True)

    # --- Profile (measure 5 times, take median) ---
    profile_times = []
    for _ in range(5):
        t0 = time.time()
        profile = build_corpus_profile(storage, "scale_ns")
        profile_times.append(time.time() - t0)
    profile_times.sort()
    profile_median = profile_times[len(profile_times) // 2]
    print(f"  Profile: median {profile_median*1000:.1f}ms (5 runs)")
    print(f"    → {profile.document_count} docs, {profile.chunk_count} chunks, "
          f"{profile.measurement_count} measurements")

    # --- Query (measure 10 times) ---
    query_times = []
    conn = storage._require_connection()

    # Test: range query on canonical_unit for temperature
    for _ in range(10):
        t0 = time.time()
        result = conn.execute(
            """
            SELECT COUNT(*) FROM scientific_measurements
            WHERE namespace = 'scale_ns'
            AND canonical_unit = '0,0,0,0,1,0,0'
            AND canonical_value BETWEEN 303.15 AND 333.15
            """
        ).fetchone()
        query_times.append(time.time() - t0)

    query_times.sort()
    query_median = query_times[len(query_times) // 2]
    print(f"  Query: median {query_median*1000:.2f}ms (10 runs)")
    print(f"    → {result[0]} matching measurements")

    # --- DB size ---
    db_size_mb = db_path.stat().st_size / (1024 * 1024)
    print(f"  DB size: {db_size_mb:.1f} MB")

    storage.teardown()

    return {
        "scale": scale,
        "generate_time_s": round(gen_time, 3),
        "index_time_s": round(index_time, 3),
        "index_throughput_blobs_per_s": round(scale / index_time, 0),
        "profile_time_ms_median": round(profile_median * 1000, 2),
        "profile_time_ms_all": [round(t * 1000, 2) for t in profile_times],
        "query_time_ms_median": round(query_median * 1000, 2),
        "query_time_ms_all": [round(t * 1000, 2) for t in query_times],
        "query_results": result[0],
        "db_size_mb": round(db_size_mb, 2),
        "documents": profile.document_count,
        "chunks": profile.chunk_count,
        "measurements": profile.measurement_count,
    }


def main() -> None:
    print("=" * 70)
    print("CLIO Scaling Study — Synthetic Data")
    print("=" * 70)
    print(f"Scales: {SCALES}")

    results: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="clio_scale_") as tmpdir:
        tmpdir = Path(tmpdir)
        for scale in SCALES:
            r = test_scale(scale, tmpdir)
            results.append(r)

    # Summary
    print("\n" + "=" * 70)
    print(f"{'Scale':>9} | {'Index (s)':>10} {'Throughput':>12} | "
          f"{'Profile':>10} {'Query':>10} | {'DB (MB)':>9}")
    print("-" * 70)
    for r in results:
        print(f"{r['scale']:>9,} | {r['index_time_s']:>10.2f} "
              f"{r['index_throughput_blobs_per_s']:>10,.0f}/s | "
              f"{r['profile_time_ms_median']:>8.1f}ms {r['query_time_ms_median']:>8.2f}ms | "
              f"{r['db_size_mb']:>9.1f}")

    # Check scaling behavior
    print("\n--- Scaling Behavior ---")
    if len(results) >= 2:
        # Profile time should be roughly constant (SQL aggregation on index)
        first_profile = results[0]["profile_time_ms_median"]
        last_profile = results[-1]["profile_time_ms_median"]
        profile_ratio = last_profile / first_profile if first_profile > 0 else 0
        scale_ratio = results[-1]["scale"] / results[0]["scale"]
        print(f"Scale ratio (last/first): {scale_ratio:.0f}x")
        print(f"Profile time ratio: {profile_ratio:.2f}x  "
              f"({'sub-linear' if profile_ratio < scale_ratio else 'linear or worse'})")

        first_query = results[0]["query_time_ms_median"]
        last_query = results[-1]["query_time_ms_median"]
        query_ratio = last_query / first_query if first_query > 0 else 0
        print(f"Query time ratio: {query_ratio:.2f}x  "
              f"({'sub-linear' if query_ratio < scale_ratio else 'linear or worse'})")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "test": "CLIO scaling study on synthetic data",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scales_tested": SCALES,
        "results": results,
    }
    out_path = OUT_DIR / "scale_test.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
