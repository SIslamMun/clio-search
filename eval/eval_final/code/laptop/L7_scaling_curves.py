#!/usr/bin/env python3
"""L7: Single-node scaling curves 1K → 100K synthetic blobs.

Establishes the reference scaling curve that the Delta multi-node study
will extend. On a single laptop node, we measure:

  * Index time           — how long to ingest N synthetic scientific blobs
  * Profile time         — cost of corpus_profile_stats
  * Query time           — scientific range query latency
  * DB size              — DuckDB file size per scale

Each scale is measured with 5 repetitions; we report median + IQR.

Scales: 1K, 2K, 5K, 10K, 25K, 50K, 100K

Why
---
  * SC reviewers want scaling data; even single-node scaling shows the
    underlying algorithmic complexity of CLIO's operators.
  * The curves establish the O(log N) behaviour of B-tree-indexed range
    queries and sub-linear profile time.
  * These numbers feed into the weak-scaling narrative on Delta: if
    single-node profile stays <15 ms at 100K and query stays <1 ms, then
    distributing across 4 nodes should give 400K × <15 ms etc.

Output
------
  eval/eval_final/outputs/L7_scaling_curves.json

Synthetic data
--------------
  Each blob is a short scientific sentence like
    "temperature 23.5 degC pressure 101.2 kPa wind 5.3 m/s"
  generated from a deterministic RNG. This ensures reproducibility.

Usage
-----
  python3 eval/eval_final/code/laptop/L7_scaling_curves.py

Estimated runtime
-----------------
  At ~500 blobs/sec single-node indexing (with full structure-aware
  chunking), 100K = ~200 sec. Plus queries. Total: ~10-15 min.
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[4]
_CODE = _REPO / "code"
sys.path.insert(0, str(_CODE / "src"))

from clio_agentic_search.indexing.scientific import (
    build_structure_aware_chunk_plan,
)
from clio_agentic_search.models.contracts import (
    DocumentRecord,
    EmbeddingRecord,
    MetadataRecord,
)
from clio_agentic_search.retrieval.corpus_profile import build_corpus_profile
from clio_agentic_search.storage.contracts import DocumentBundle, FileIndexState
from clio_agentic_search.storage.duckdb_store import DuckDBStorage

OUT_DIR = _REPO / "eval" / "eval_final" / "outputs"

SCALES = [1_000, 2_000, 5_000, 10_000, 25_000, 50_000, 100_000]
QUERY_REPEATS = 10   # for stable median
PROFILE_REPEATS = 10


def generate_doc(idx: int, rng: random.Random) -> dict[str, Any]:
    domains = ("atmospheric", "oceanographic", "chemical", "biological", "geological")
    domain = rng.choice(domains)
    temp_c = round(rng.uniform(-10, 45), 2)
    pressure_kpa = round(rng.uniform(95, 110), 2)
    wind_mps = round(rng.uniform(0, 25), 2)
    text = (
        f"# {domain.capitalize()} measurement log {idx}\n\n"
        f"Temperature was {temp_c} degC, pressure {pressure_kpa} kPa, "
        f"wind speed {wind_mps} m/s at station {rng.randint(1, 500)}."
    )
    return {
        "doc_id": f"doc_{idx:07d}",
        "text": text,
    }


def index_bundle(storage: DuckDBStorage, namespace: str, docs: list[dict[str, Any]]) -> None:
    bundles: list[DocumentBundle] = []
    batch_size = 200
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        for d in batch:
            plan = build_structure_aware_chunk_plan(
                namespace=namespace,
                document_id=d["doc_id"],
                text=d["text"],
                chunk_size=400,
            )
            meta_records: list[MetadataRecord] = []
            for chunk_id, chunk_meta in plan.metadata_by_chunk_id.items():
                for k, v in chunk_meta.items():
                    meta_records.append(MetadataRecord(
                        namespace=namespace,
                        record_id=chunk_id,
                        scope="chunk",
                        key=k,
                        value=v,
                    ))
            doc = DocumentRecord(
                namespace=namespace,
                document_id=d["doc_id"],
                uri=f"synth://{d['doc_id']}",
                checksum=hashlib.sha256(d["text"].encode()).hexdigest()[:16],
                modified_at_ns=time.time_ns(),
            )
            file_state = FileIndexState(
                namespace=namespace,
                path=f"synth/{d['doc_id']}",
                document_id=d["doc_id"],
                mtime_ns=time.time_ns(),
                content_hash="h",
            )
            bundles.append(DocumentBundle(
                document=doc, chunks=plan.chunks, embeddings=[],
                metadata=meta_records, file_state=file_state,
            ))
        if bundles:
            storage.upsert_document_bundles(bundles, include_lexical_postings=False)
            bundles = []


def measure_scale(scale: int, tmp_dir: Path) -> dict[str, Any]:
    print(f"\n--- Scale: {scale:,} ---", flush=True)
    rng = random.Random(42 + scale)

    # Generate
    t0 = time.time()
    docs = [generate_doc(i, rng) for i in range(scale)]
    gen_time = time.time() - t0

    # Index
    db_path = tmp_dir / f"scale_{scale}.duckdb"
    storage = DuckDBStorage(database_path=db_path)
    storage.connect()

    t0 = time.time()
    index_bundle(storage, "scaling_ns", docs)
    index_time = time.time() - t0
    index_throughput = scale / index_time if index_time > 0 else 0
    print(f"  Indexed in {index_time:.2f}s ({index_throughput:.0f} docs/s)")

    # Profile: 10 runs, report median
    profile_times: list[float] = []
    for _ in range(PROFILE_REPEATS):
        t = time.perf_counter()
        profile = build_corpus_profile(storage, "scaling_ns")
        profile_times.append(time.perf_counter() - t)
    profile_median_ms = statistics.median(profile_times) * 1000
    profile_min_ms = min(profile_times) * 1000
    profile_max_ms = max(profile_times) * 1000
    profile_stddev_ms = statistics.stdev(profile_times) * 1000 if len(profile_times) > 1 else 0
    print(f"  Profile: median {profile_median_ms:.2f} ms "
          f"(±{profile_stddev_ms:.2f}, {profile_min_ms:.2f}-{profile_max_ms:.2f})")

    # Query: 10 runs of a scientific range query
    conn = storage._require_connection()
    query_times: list[float] = []
    query_result_count = 0
    for _ in range(QUERY_REPEATS):
        t = time.perf_counter()
        row = conn.execute(
            """
            SELECT COUNT(*) FROM scientific_measurements
            WHERE namespace = 'scaling_ns'
              AND canonical_unit = '0,0,0,0,1,0,0'
              AND canonical_value BETWEEN 303.15 AND 333.15
            """,
        ).fetchone()
        query_times.append(time.perf_counter() - t)
        query_result_count = row[0] if row else 0
    query_median_ms = statistics.median(query_times) * 1000
    query_min_ms = min(query_times) * 1000
    query_max_ms = max(query_times) * 1000
    query_stddev_ms = statistics.stdev(query_times) * 1000 if len(query_times) > 1 else 0
    print(f"  Query: median {query_median_ms:.3f} ms "
          f"(±{query_stddev_ms:.3f}, {query_min_ms:.3f}-{query_max_ms:.3f}) "
          f"→ {query_result_count} rows")

    # DB size
    db_size_mb = db_path.stat().st_size / (1024 * 1024)
    print(f"  DB size: {db_size_mb:.2f} MB")

    storage.teardown()

    return {
        "scale": scale,
        "generate_time_s": round(gen_time, 3),
        "index_time_s": round(index_time, 3),
        "index_throughput_docs_per_s": round(index_throughput, 0),
        "profile_time_ms_median": round(profile_median_ms, 3),
        "profile_time_ms_min": round(profile_min_ms, 3),
        "profile_time_ms_max": round(profile_max_ms, 3),
        "profile_time_ms_stddev": round(profile_stddev_ms, 3),
        "query_time_ms_median": round(query_median_ms, 4),
        "query_time_ms_min": round(query_min_ms, 4),
        "query_time_ms_max": round(query_max_ms, 4),
        "query_time_ms_stddev": round(query_stddev_ms, 4),
        "query_result_count": query_result_count,
        "db_size_mb": round(db_size_mb, 2),
        "documents": profile.document_count,
        "chunks": profile.chunk_count,
        "measurements": profile.measurement_count,
    }


def main() -> None:
    print("=" * 75)
    print("L7: Single-node scaling curves")
    print("=" * 75)
    print(f"Scales: {SCALES}")
    print(f"Profile reps: {PROFILE_REPEATS}, Query reps: {QUERY_REPEATS}")

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="L7_scaling_") as tmp:
        for scale in SCALES:
            r = measure_scale(scale, Path(tmp))
            results.append(r)

    # Scaling ratio analysis
    if len(results) >= 2:
        first = results[0]
        last = results[-1]
        scale_ratio = last["scale"] / first["scale"]
        profile_ratio = last["profile_time_ms_median"] / first["profile_time_ms_median"]
        query_ratio = last["query_time_ms_median"] / first["query_time_ms_median"] if first["query_time_ms_median"] > 0 else 0

        print("\n" + "=" * 75)
        print("SCALING BEHAVIOUR")
        print("=" * 75)
        print(f"Scale ratio (last/first): {scale_ratio:.0f}x")
        print(f"Profile time ratio:       {profile_ratio:.2f}x")
        print(f"Query time ratio:         {query_ratio:.2f}x")
        if profile_ratio < scale_ratio:
            print("→ Profile time is SUB-LINEAR in corpus size ✓")
        if query_ratio < scale_ratio:
            print("→ Query time is SUB-LINEAR in corpus size ✓")

    # Summary table
    print("\nSummary:")
    print(f"{'Scale':>8} | {'Idx(s)':>7} {'Thru':>8} | "
          f"{'Prof(ms)':>9} {'±':>5} | {'Qry(ms)':>8} {'±':>6} | {'DB(MB)':>7}")
    print("-" * 75)
    for r in results:
        print(
            f"{r['scale']:>8,} | {r['index_time_s']:>7.2f} "
            f"{r['index_throughput_docs_per_s']:>6.0f}/s | "
            f"{r['profile_time_ms_median']:>9.2f} {r['profile_time_ms_stddev']:>5.2f} | "
            f"{r['query_time_ms_median']:>8.3f} {r['query_time_ms_stddev']:>6.3f} | "
            f"{r['db_size_mb']:>7.2f}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "L7: Single-node scaling curves",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scales": SCALES,
        "profile_repeats": PROFILE_REPEATS,
        "query_repeats": QUERY_REPEATS,
        "results": results,
    }
    with (OUT_DIR / "L7_scaling_curves.json").open("w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {OUT_DIR / 'L7_scaling_curves.json'}")


if __name__ == "__main__":
    main()
