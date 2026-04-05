#!/usr/bin/env python3
"""Sampling-aware search test: how much data does CLIO actually inspect?

Builds a large corpus where ONE specific dataset contains the answer,
runs a query, and measures what fraction of the corpus CLIO actually
looked at vs. brute-force inspection.

Output: eval/eval_eve/outputs/sampling_test.json
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

from clio_agentic_search.indexing.scientific import canonicalize_measurement
from clio_agentic_search.retrieval.corpus_profile import build_corpus_profile
from clio_agentic_search.storage.duckdb_store import DuckDBStorage

OUT_DIR = _CODE_ROOT.parent / "eval" / "eval_eve" / "outputs"

TOTAL_BLOBS = 50_000
NEEDLE_COUNT = 3  # Number of "answer" documents hidden in haystack
NEEDLE_UNIT = "degC"
NEEDLE_RANGE = (85, 95)  # Very hot temperature values


def generate_haystack(n: int, rng: random.Random) -> list[dict[str, Any]]:
    """Generate n documents, mostly irrelevant."""
    other_units = [
        ("kPa", 90, 120),
        ("km/h", 0, 50),
        ("percent", 0, 100),
        ("degC", 0, 30),  # Normal temperature (not the answer)
    ]

    blobs = []
    for i in range(n):
        unit, vmin, vmax = rng.choice(other_units)
        value = round(rng.uniform(vmin, vmax), 2)
        doc_id = f"haystack_{i:08d}"
        text = f"# Dataset {i}\nMeasurement: {value} {unit}. Routine observation."
        blobs.append({
            "doc_id": doc_id,
            "text": text,
            "value": value,
            "unit": unit,
            "is_needle": False,
        })
    return blobs


def generate_needles(rng: random.Random) -> list[dict[str, Any]]:
    """Generate NEEDLE_COUNT answer documents with extreme temperatures."""
    needles = []
    for i in range(NEEDLE_COUNT):
        value = round(rng.uniform(*NEEDLE_RANGE), 2)
        doc_id = f"NEEDLE_{i:03d}"
        text = (
            f"# Extreme Temperature Event {i}\n"
            f"Critical measurement: {value} {NEEDLE_UNIT}. "
            f"Station recorded heat wave conditions."
        )
        needles.append({
            "doc_id": doc_id,
            "text": text,
            "value": value,
            "unit": NEEDLE_UNIT,
            "is_needle": True,
        })
    return needles


def bulk_insert(storage: DuckDBStorage, namespace: str, blobs: list[dict[str, Any]]) -> None:
    conn = storage._require_connection()
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

        try:
            canonical_val, canonical_unit = canonicalize_measurement(
                b["value"], b["unit"]
            )
            meas_data.append((namespace, chunk_id, canonical_unit, canonical_val,
                              b["unit"], b["value"]))
        except (ValueError, KeyError):
            pass

        meta_data.append((namespace, chunk_id, "chunk", "is_needle",
                          "true" if b["is_needle"] else "false"))

    batch = 5000
    for i in range(0, len(docs_data), batch):
        conn.executemany(
            "INSERT OR REPLACE INTO documents(namespace, document_id, uri, checksum, modified_at_ns) VALUES (?, ?, ?, ?, ?)",
            docs_data[i : i + batch],
        )
        conn.executemany(
            "INSERT OR REPLACE INTO chunks(namespace, chunk_id, document_id, chunk_index, text, start_offset, end_offset) VALUES (?, ?, ?, ?, ?, ?, ?)",
            chunks_data[i : i + batch],
        )
    if meas_data:
        for i in range(0, len(meas_data), batch):
            conn.executemany(
                "INSERT INTO scientific_measurements(namespace, chunk_id, canonical_unit, canonical_value, raw_unit, raw_value) VALUES (?, ?, ?, ?, ?, ?)",
                meas_data[i : i + batch],
            )
    for i in range(0, len(meta_data), batch):
        conn.executemany(
            "INSERT OR REPLACE INTO metadata(namespace, record_id, scope, key, value) VALUES (?, ?, ?, ?, ?)",
            meta_data[i : i + batch],
        )


def brute_force_search(
    storage: DuckDBStorage, namespace: str, query_min: float, query_max: float
) -> dict[str, Any]:
    """Baseline: inspect every chunk, check text for the query.

    This simulates what an LLM agent would do without CLIO — read every
    document and decide if it matches.
    """
    conn = storage._require_connection()
    t0 = time.time()

    # Fetch ALL chunks (simulating full inspection)
    rows = conn.execute(
        "SELECT chunk_id, text FROM chunks WHERE namespace = ?",
        [namespace],
    ).fetchall()

    fetch_time = time.time() - t0
    chunks_fetched = len(rows)

    # Compute "tokens" (words) that would need to be sent to an LLM
    total_tokens = sum(len(text.split()) for _, text in rows)

    # Manual filtering — pretend the LLM checks each chunk
    matches = []
    for chunk_id, text in rows:
        # Naive text search for "extreme" or high-temp indicators
        if "Extreme" in text or any(
            str(v) in text for v in range(int(query_min), int(query_max) + 1)
        ):
            matches.append(chunk_id)

    total_time = time.time() - t0

    return {
        "method": "brute_force",
        "chunks_inspected": chunks_fetched,
        "tokens_processed": total_tokens,
        "time_s": round(total_time, 3),
        "fetch_time_s": round(fetch_time, 3),
        "matches_found": len(matches),
    }


def clio_search(
    storage: DuckDBStorage, namespace: str, query_min: float, query_max: float
) -> dict[str, Any]:
    """CLIO's science-aware search: profile first, then range query on index."""
    conn = storage._require_connection()

    # Step 1: Profile
    t0 = time.time()
    profile = build_corpus_profile(storage, namespace)
    profile_time = time.time() - t0

    # Step 2: Canonicalize query bounds
    canonical_min, canonical_unit = canonicalize_measurement(query_min, NEEDLE_UNIT)
    canonical_max, _ = canonicalize_measurement(query_max, NEEDLE_UNIT)

    # Step 3: Index-backed range query (no full scan)
    t1 = time.time()
    rows = conn.execute(
        """
        SELECT DISTINCT c.chunk_id, c.text
        FROM scientific_measurements sm
        JOIN chunks c ON c.namespace = sm.namespace AND c.chunk_id = sm.chunk_id
        WHERE sm.namespace = ?
        AND sm.canonical_unit = ?
        AND sm.canonical_value BETWEEN ? AND ?
        """,
        [namespace, canonical_unit, canonical_min, canonical_max],
    ).fetchall()
    query_time = time.time() - t1

    chunks_inspected = len(rows)  # Only matching chunks returned
    # Tokens that would be sent to LLM: only the matched chunks
    tokens_processed = sum(len(text.split()) for _, text in rows)

    return {
        "method": "clio_science_aware",
        "profile_time_ms": round(profile_time * 1000, 2),
        "chunks_inspected": chunks_inspected,
        "tokens_processed": tokens_processed,
        "time_s": round(profile_time + query_time, 3),
        "query_time_ms": round(query_time * 1000, 2),
        "matches_found": len(rows),
        "profile": {
            "total_chunks": profile.chunk_count,
            "total_measurements": profile.measurement_count,
        },
    }


def main() -> None:
    print("=" * 70)
    print("Sampling-Aware Search Test")
    print(f"Total blobs: {TOTAL_BLOBS:,} | Needles: {NEEDLE_COUNT}")
    print("=" * 70)

    rng = random.Random(7)  # Deterministic

    print("\n[1/3] Generating corpus...")
    t0 = time.time()
    haystack = generate_haystack(TOTAL_BLOBS - NEEDLE_COUNT, rng)
    needles = generate_needles(rng)
    all_blobs = haystack + needles
    rng.shuffle(all_blobs)  # Mix needles among haystack
    print(f"  {len(all_blobs):,} blobs generated in {time.time() - t0:.2f}s")

    with tempfile.TemporaryDirectory(prefix="clio_sample_") as tmpdir:
        tmpdir = Path(tmpdir)
        db_path = tmpdir / "sample.duckdb"
        storage = DuckDBStorage(database_path=db_path)
        storage.connect()

        print("\n[2/3] Indexing...")
        t0 = time.time()
        bulk_insert(storage, "sample", all_blobs)
        index_time = time.time() - t0
        print(f"  Indexed in {index_time:.2f}s")

        db_size_mb = db_path.stat().st_size / (1024 * 1024)
        print(f"  DB size: {db_size_mb:.1f} MB")

        # --- Query: "find temperature above 85 degC" ---
        print(f"\n[3/3] Query: temperature in {NEEDLE_RANGE[0]}-{NEEDLE_RANGE[1]} degC")

        brute = brute_force_search(storage, "sample", *NEEDLE_RANGE)
        print(f"\n  BRUTE FORCE (baseline):")
        print(f"    Chunks inspected: {brute['chunks_inspected']:,}")
        print(f"    Tokens processed: {brute['tokens_processed']:,}")
        print(f"    Time: {brute['time_s']}s")
        print(f"    Matches found: {brute['matches_found']}")

        clio = clio_search(storage, "sample", *NEEDLE_RANGE)
        print(f"\n  CLIO (science-aware):")
        print(f"    Profile time: {clio['profile_time_ms']}ms")
        print(f"    Chunks inspected: {clio['chunks_inspected']}")
        print(f"    Tokens processed: {clio['tokens_processed']}")
        print(f"    Query time: {clio['query_time_ms']}ms")
        print(f"    Matches found: {clio['matches_found']}")

        storage.teardown()

    # Compute savings
    chunk_ratio = clio["chunks_inspected"] / brute["chunks_inspected"] if brute["chunks_inspected"] > 0 else 0
    token_ratio = clio["tokens_processed"] / brute["tokens_processed"] if brute["tokens_processed"] > 0 else 0
    chunk_reduction = (1 - chunk_ratio) * 100
    token_reduction = (1 - token_ratio) * 100

    print(f"\n{'='*70}")
    print("SAMPLING RESULTS")
    print(f"{'='*70}")
    print(f"CLIO inspected {chunk_ratio*100:.4f}% of chunks ({clio['chunks_inspected']} / {brute['chunks_inspected']:,})")
    print(f"CLIO processed {token_ratio*100:.4f}% of tokens ({clio['tokens_processed']} / {brute['tokens_processed']:,})")
    print(f"Chunk reduction: {chunk_reduction:.2f}%")
    print(f"Token reduction: {token_reduction:.2f}%")
    print()
    print("This demonstrates sampling-aware behavior: CLIO uses its index to")
    print("read only the chunks that match the query, not the entire corpus.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "test": "Sampling-aware search: brute force vs CLIO",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "corpus": {
            "total_blobs": TOTAL_BLOBS,
            "needle_count": NEEDLE_COUNT,
            "index_time_s": round(index_time, 3),
            "db_size_mb": round(db_size_mb, 2),
        },
        "query": {
            "description": f"Find temperature {NEEDLE_RANGE[0]}-{NEEDLE_RANGE[1]} degC",
        },
        "brute_force": brute,
        "clio": clio,
        "savings": {
            "chunks_inspected_ratio": round(chunk_ratio, 6),
            "tokens_processed_ratio": round(token_ratio, 6),
            "chunk_reduction_pct": round(chunk_reduction, 2),
            "token_reduction_pct": round(token_reduction, 2),
        },
    }
    out_path = OUT_DIR / "sampling_test.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
