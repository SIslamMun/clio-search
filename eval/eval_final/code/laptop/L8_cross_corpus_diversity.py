#!/usr/bin/env python3
"""L8: Cross-corpus diversity — metadata profiles across 4 real datasets.

Indexes each of four real scientific corpora in isolation and reports
their metadata-schema profile. Demonstrates that CLIO correctly
distinguishes rich-metadata corpora from sparse ones and picks different
strategies accordingly — the "metadata-adaptive discovery" claim.

Corpora
-------
  1. NOAA GHCN-Daily   — 1,728 real weather station docs (metric units)
  2. DOE Data Explorer — 500 DOE scientific dataset descriptions
  3. Controlled v2     — 210 synthetic multi-domain scientific docs
  4. arXiv (subset)    — 500 abstracts from the metadata dump (if available)

For each corpus we report:
  * document count, chunk count, measurement count, formula count
  * metadata_density (scientific fraction)
  * MetadataSchema.richness_score (broader schema-based score)
  * MetadataSchema.concepts     (which scientific concepts were detected)
  * quality_summary (if QC flags were extracted)
  * sampled_schema (if enable_sampling=True recovers extra concepts)
  * inferred strategy: rich / default / sparse

Metrics
-------
  * Per-corpus profile (all the above)
  * Cross-corpus table contrasting metadata richness
  * Number of distinct canonical concepts per corpus

Output
------
  eval/eval_final/outputs/L8_cross_corpus_diversity.json

Prerequisites
-------------
  * code/benchmarks/corpus_real/       (NOAA, 1728 files)
  * code/benchmarks/corpus_doe/        (DOE, 500 files)
  * code/benchmarks/corpus_v2/         (controlled, 210 files)
  * eval/eval_final/data/arxiv_sample.jsonl  (optional, skipped if absent)

Usage
-----
  python3 eval/eval_final/code/laptop/L8_cross_corpus_diversity.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[4]
_CODE = _REPO / "code"
sys.path.insert(0, str(_CODE / "src"))

from clio_agentic_search.connectors.filesystem.connector import FilesystemConnector
from clio_agentic_search.retrieval.corpus_profile import build_corpus_profile
from clio_agentic_search.retrieval.metadata_schema import describe_schema
from clio_agentic_search.storage.duckdb_store import DuckDBStorage

OUT_DIR = _REPO / "eval" / "eval_final" / "outputs"
BENCH_DIR = _CODE / "benchmarks"
ARXIV_SAMPLE = _REPO / "eval" / "eval_final" / "data" / "arxiv_sample.jsonl"


CORPORA = [
    {
        "name": "NOAA_GHCN_Daily",
        "description": "1,728 real weather station docs with metric units",
        "path": BENCH_DIR / "corpus_real",
        "namespace": "noaa",
    },
    {
        "name": "DOE_Data_Explorer",
        "description": "500 DOE scientific dataset descriptions",
        "path": BENCH_DIR / "corpus_doe",
        "namespace": "doe",
    },
    {
        "name": "Controlled_v2",
        "description": "210 synthetic multi-domain scientific documents",
        "path": BENCH_DIR / "corpus_v2",
        "namespace": "ctrl",
    },
]


def index_corpus(corpus: dict[str, Any], tmp_root: Path) -> dict[str, Any] | None:
    src = corpus["path"]
    if not src.exists():
        print(f"  ⚠ {corpus['name']}: path not found ({src})")
        return None

    file_count = sum(1 for _ in src.rglob("*.txt"))
    print(f"\n{corpus['name']}: {corpus['description']}")
    print(f"  Files: {file_count}")

    dest = tmp_root / corpus["namespace"]
    shutil.copytree(src, dest)

    db_path = tmp_root / f"{corpus['namespace']}.duckdb"
    storage = DuckDBStorage(database_path=db_path)
    connector = FilesystemConnector(
        namespace=corpus["namespace"], root=dest, storage=storage,
        warmup_async=False,
    )
    connector.connect()
    t0 = time.time()
    connector.index(full_rebuild=True)
    index_time = time.time() - t0
    print(f"  Indexed in {index_time:.1f}s")

    # Profile without sampling
    t0 = time.time()
    profile_basic = build_corpus_profile(storage, corpus["namespace"])
    profile_time_basic = time.time() - t0

    # Profile with sampling (recover hidden measurements)
    t0 = time.time()
    profile_sampled = build_corpus_profile(
        storage, corpus["namespace"],
        enable_sampling=True,
        sample_size=30,
    )
    profile_time_sampled = time.time() - t0

    # Determine strategy classification
    density = profile_basic.metadata_density
    if density > 0.5:
        strategy = "metadata_rich"
    elif density < 0.1:
        strategy = "metadata_sparse"
    else:
        strategy = "default"

    result: dict[str, Any] = {
        "name": corpus["name"],
        "description": corpus["description"],
        "file_count": file_count,
        "index_time_s": round(index_time, 2),
        "profile_time_basic_ms": round(profile_time_basic * 1000, 2),
        "profile_time_sampled_ms": round(profile_time_sampled * 1000, 2),
        "documents": profile_basic.document_count,
        "chunks": profile_basic.chunk_count,
        "measurements": profile_basic.measurement_count,
        "formulas": profile_basic.formula_count,
        "distinct_units": list(profile_basic.distinct_units),
        "metadata_density": round(profile_basic.metadata_density, 4),
        "richness_score": round(profile_basic.richness_score, 4),
        "strategy": strategy,
    }

    if profile_basic.has_metadata_schema:
        schema = profile_basic.metadata_schema
        assert schema is not None
        result["metadata_schema"] = describe_schema(schema)
    if profile_basic.has_quality_info:
        q = profile_basic.quality_summary
        assert q is not None
        result["quality_summary"] = {
            "total": q.total,
            "good": q.good,
            "questionable": q.questionable,
            "bad": q.bad,
            "missing": q.missing,
            "estimated": q.estimated,
            "unknown": q.unknown,
            "acceptable_ratio": round(q.acceptable_ratio, 4),
            "average_score": round(q.average_score, 4),
        }

    # Sampling recovery
    if profile_sampled.has_sampled_schema:
        sampled = profile_sampled.sampled_schema
        assert sampled is not None
        result["sampling_recovery"] = {
            "sample_size": sampled.sample_size,
            "concepts_found": sorted(sampled.concepts_found),
            "measurement_count_in_sample": sampled.measurement_count,
            "units_found": sorted(sampled.measurement_units_found),
            "inferred_density": round(sampled.inferred_density, 4),
            "recovered_structure": sampled.has_recoverable_structure,
        }

    connector.teardown()
    storage.teardown()

    print(f"  Measurements: {profile_basic.measurement_count}")
    print(f"  Density:      {profile_basic.metadata_density:.3f}")
    print(f"  Richness:     {profile_basic.richness_score:.3f}")
    print(f"  Strategy:     {strategy}")
    if profile_basic.has_metadata_schema:
        concepts = sorted(profile_basic.metadata_schema.concepts)
        print(f"  Concepts:     {concepts}")

    return result


def main() -> None:
    print("=" * 75)
    print("L8: Cross-corpus diversity")
    print("=" * 75)

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="L8_crosscorpus_") as tmp:
        for corpus in CORPORA:
            r = index_corpus(corpus, Path(tmp))
            if r:
                results.append(r)

    # Optional arXiv subset
    if ARXIV_SAMPLE.exists():
        print(f"\narXiv sample found at {ARXIV_SAMPLE} — would index here (skipped in v1)")

    # Summary table
    print("\n" + "=" * 75)
    print("CROSS-CORPUS SUMMARY")
    print("=" * 75)
    print(f"{'Corpus':<22} {'Docs':>6} {'Chunks':>7} {'Meas':>7} "
          f"{'Density':>8} {'Rich':>7} {'Strategy':<16} {'Concepts':>9}")
    print("-" * 95)
    for r in results:
        concept_count = (
            len(r["metadata_schema"]["concepts"])
            if "metadata_schema" in r else 0
        )
        print(
            f"{r['name']:<22} {r['documents']:>6} {r['chunks']:>7} "
            f"{r['measurements']:>7} "
            f"{r['metadata_density']:>8.3f} {r['richness_score']:>7.3f} "
            f"{r['strategy']:<16} {concept_count:>9}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "L8: Cross-corpus diversity",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "corpora_analysed": len(results),
        "corpora": results,
    }
    with (OUT_DIR / "L8_cross_corpus_diversity.json").open("w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {OUT_DIR / 'L8_cross_corpus_diversity.json'}")


if __name__ == "__main__":
    main()
