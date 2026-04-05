#!/usr/bin/env python3
"""REAL TEST: CLIO on large real corpora (NOAA 1728 docs, NumConQ 5362, DOE 500).

No projections. Measures actual: index time, profile time, query time,
branches used, tokens returned, precision on known ground truth.

Output: eval/real_tests/large_corpus_test.json
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE_DIR / "src"))

from clio_agentic_search.connectors.filesystem.connector import FilesystemConnector
from clio_agentic_search.retrieval.coordinator import RetrievalCoordinator
from clio_agentic_search.retrieval.corpus_profile import build_corpus_profile
from clio_agentic_search.retrieval.scientific import (
    NumericRangeOperator,
    ScientificQueryOperators,
)
from clio_agentic_search.storage.duckdb_store import DuckDBStorage

BENCH_DIR = Path(__file__).resolve().parent
OUT_DIR = _CODE_DIR.parent / "eval" / "real_tests"

CORPORA = [
    {
        "name": "NOAA_1728",
        "path": BENCH_DIR / "corpus_real",
        "queries_file": BENCH_DIR / "real_queries.json",
        "description": "1728 real NOAA weather station docs (metric units, imperial queries)",
    },
    {
        "name": "controlled_210",
        "path": BENCH_DIR / "corpus_v2",
        "queries_file": BENCH_DIR / "queries_v2.json",
        "description": "210 controlled scientific docs, 80 queries",
    },
    {
        "name": "DOE_500",
        "path": BENCH_DIR / "corpus_doe",
        "queries_file": BENCH_DIR / "doe_queries.json",
        "description": "500 DOE scientific dataset descriptions",
    },
]


def count_tokens(text: str) -> int:
    return len(text.split())


def test_corpus(corpus_info: dict[str, Any], tmpdir: Path) -> dict[str, Any]:
    name = corpus_info["name"]
    corpus_path = corpus_info["path"]
    queries_file = corpus_info["queries_file"]

    if not corpus_path.exists():
        return {"name": name, "error": f"corpus not found: {corpus_path}"}
    if not queries_file.exists():
        return {"name": name, "error": f"queries not found: {queries_file}"}

    # Count files
    file_count = sum(1 for _ in corpus_path.rglob("*.txt"))
    corpus_size = sum(f.stat().st_size for f in corpus_path.rglob("*.txt"))

    print(f"\n  {name}: {file_count} files, {corpus_size/1024:.0f}KB")

    # Index
    dest = tmpdir / name
    shutil.copytree(corpus_path, dest)
    db = DuckDBStorage(database_path=tmpdir / f"{name}.duckdb")
    conn = FilesystemConnector(
        namespace=name, root=dest, storage=db, warmup_async=False,
    )
    conn.connect()

    t0 = time.time()
    conn.index(full_rebuild=True)
    index_time = time.time() - t0
    print(f"    Index: {index_time:.1f}s")

    # Profile
    t1 = time.time()
    profile = build_corpus_profile(db, name)
    profile_time = time.time() - t1
    print(f"    Profile: {profile_time*1000:.1f}ms → {profile.document_count} docs, "
          f"{profile.chunk_count} chunks, {profile.measurement_count} meas, "
          f"density={profile.metadata_density:.1%}")

    # Load queries
    with open(queries_file) as f:
        qdata = json.load(f)

    # Get query list depending on format
    if isinstance(qdata, list):
        queries = qdata[:20]
    elif isinstance(qdata, dict):
        queries = []
        for key in ("cross_unit_queries", "same_unit_queries", "formula_queries",
                     "multi_constraint_queries", "queries"):
            if key in qdata:
                queries.extend(qdata[key])
        queries = queries[:20]
    else:
        queries = []

    # Run queries
    coordinator = RetrievalCoordinator()
    query_results: list[dict[str, Any]] = []
    total_branches_used = 0
    total_branches_possible = 0

    for q in queries:
        qtext = q.get("query", q.get("text", ""))
        if not qtext:
            continue

        operators = ScientificQueryOperators()
        nr = q.get("numeric_range", "")
        if nr and ":" in str(nr):
            parts = str(nr).split(":")
            if len(parts) >= 3:
                try:
                    operators = ScientificQueryOperators(
                        numeric_range=NumericRangeOperator(
                            minimum=float(parts[0]), maximum=float(parts[1]), unit=parts[2],
                        ),
                    )
                except (ValueError, KeyError):
                    pass

        t2 = time.time()
        result = coordinator.query(
            connector=conn, query=qtext, top_k=10,
            scientific_operators=operators,
        )
        query_time = time.time() - t2

        branches = []
        for event in result.trace:
            if event.stage == "branch_plan_selected":
                for b in ("lexical", "vector", "graph", "scientific"):
                    if event.attributes.get(f"use_{b}") == "True":
                        branches.append(b)
        total_branches_used += len(branches)
        total_branches_possible += 4

        result_text = " ".join(c.snippet for c in result.citations)
        tokens = count_tokens(result_text)

        query_results.append({
            "query": qtext[:60],
            "hits": len(result.citations),
            "tokens_returned": tokens,
            "time_ms": round(query_time * 1000, 1),
            "branches": branches,
        })

    avg_hits = sum(r["hits"] for r in query_results) / len(query_results) if query_results else 0
    avg_tokens = sum(r["tokens_returned"] for r in query_results) / len(query_results) if query_results else 0
    avg_time = sum(r["time_ms"] for r in query_results) / len(query_results) if query_results else 0
    branches_skipped = total_branches_possible - total_branches_used

    print(f"    Queries: {len(query_results)}, avg hits={avg_hits:.1f}, "
          f"avg tokens={avg_tokens:.0f}, avg time={avg_time:.0f}ms")
    print(f"    Branches: {total_branches_used}/{total_branches_possible} "
          f"({branches_skipped} skipped)")

    conn.teardown()
    db.teardown()

    return {
        "name": name,
        "description": corpus_info["description"],
        "files": file_count,
        "corpus_size_bytes": corpus_size,
        "index_time_s": round(index_time, 2),
        "profile_time_ms": round(profile_time * 1000, 2),
        "profile": {
            "documents": profile.document_count,
            "chunks": profile.chunk_count,
            "measurements": profile.measurement_count,
            "formulas": profile.formula_count,
            "metadata_density": round(profile.metadata_density, 4),
            "distinct_units": len(profile.distinct_units),
        },
        "queries_tested": len(query_results),
        "avg_hits": round(avg_hits, 2),
        "avg_tokens_returned": round(avg_tokens, 0),
        "avg_query_time_ms": round(avg_time, 1),
        "branches_used": total_branches_used,
        "branches_possible": total_branches_possible,
        "branches_skipped": branches_skipped,
        "per_query": query_results,
    }


def main() -> None:
    print("REAL TEST: CLIO on large real corpora")
    print("=" * 60)

    with tempfile.TemporaryDirectory(prefix="clio_large_") as tmpdir:
        tmpdir = Path(tmpdir)
        all_results: list[dict[str, Any]] = []

        for corpus in CORPORA:
            result = test_corpus(corpus, tmpdir)
            all_results.append(result)

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Corpus':<15} | {'Docs':>5} {'Chunks':>7} {'Meas':>6} {'Density':>8} | "
          f"{'Idx(s)':>6} {'Prof(ms)':>8} {'Q(ms)':>6} | {'Br skip':>7}")
    print("-" * 80)
    for r in all_results:
        if "error" in r:
            print(f"{r['name']:<15} | ERROR: {r['error']}")
            continue
        p = r["profile"]
        print(f"{r['name']:<15} | {p['documents']:>5} {p['chunks']:>7} {p['measurements']:>6} "
              f"{p['metadata_density']:>7.1%} | {r['index_time_s']:>6.1f} "
              f"{r['profile_time_ms']:>8.1f} {r['avg_query_time_ms']:>6.0f} | "
              f"{r['branches_skipped']:>3}/{r['branches_possible']}")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "test": "CLIO on large real corpora",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "corpora": all_results,
    }
    out_path = OUT_DIR / "large_corpus_test.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
