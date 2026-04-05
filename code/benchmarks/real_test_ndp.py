#!/usr/bin/env python3
"""REAL TEST: NDP-MCP without CLIO vs with CLIO.

No projections. No estimates. Actual measurements.
Measures: time, API calls, result counts, token counts (actual text length).

Output: eval/real_tests/ndp_test.json
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE_DIR / "src"))

from clio_agentic_search.connectors.ndp.connector import NDPConnector
from clio_agentic_search.retrieval.coordinator import RetrievalCoordinator
from clio_agentic_search.retrieval.corpus_profile import build_corpus_profile
from clio_agentic_search.retrieval.scientific import (
    NumericRangeOperator,
    ScientificQueryOperators,
)
from clio_agentic_search.storage.duckdb_store import DuckDBStorage

OUT_DIR = _CODE_DIR.parent / "eval" / "real_tests"
NDP_URL = "http://155.101.6.191:8003"

QUERIES = [
    {"id": "t1", "terms": ["fahrenheit"], "query": "temperature above 86 fahrenheit", "range": "86:120:degF"},
    {"id": "t2", "terms": ["celsius"], "query": "temperature above 30 celsius", "range": "30:60:degC"},
    {"id": "t3", "terms": ["kPa"], "query": "pressure around 101 kPa", "range": "90:110:kPa"},
    {"id": "t4", "terms": ["pressure"], "query": "atmospheric pressure sensor data"},
    {"id": "t5", "terms": ["wind", "speed"], "query": "wind speed above 50 km/h", "range": "50:200:km/h"},
    {"id": "t6", "terms": ["wind"], "query": "wind speed measurements"},
    {"id": "t7", "terms": ["humidity"], "query": "relative humidity measurements"},
    {"id": "t8", "terms": ["radiation"], "query": "solar radiation data"},
    {"id": "t9", "terms": ["glacier", "temperature"], "query": "glacier temperature pressure data"},
    {"id": "t10", "terms": ["ocean", "temperature"], "query": "ocean surface temperature"},
]


def count_tokens_rough(text: str) -> int:
    """Rough token count: split on whitespace. Close enough for comparison."""
    return len(text.split())


def ndp_raw_search(terms: list[str]) -> dict[str, Any]:
    """Search NDP directly. Measure everything."""
    t0 = time.time()
    api_calls = 0
    all_results: list[dict[str, Any]] = []
    total_text = ""

    with httpx.Client(timeout=30.0) as client:
        for term in terms:
            api_calls += 1
            try:
                resp = client.get(f"{NDP_URL}/search", params={"terms": term, "server": "global"})
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list):
                    for ds in data:
                        # Measure actual text a user/agent would need to read
                        title = ds.get("title", "") or ""
                        notes = ds.get("notes", "") or ""
                        resources_text = ""
                        for r in ds.get("resources", []):
                            resources_text += f" {r.get('name','')} {r.get('description','')} {r.get('format','')}"
                        full_text = f"{title} {notes} {resources_text}"
                        total_text += full_text
                        if ds.get("id") not in {r.get("id") for r in all_results}:
                            all_results.append(ds)
            except Exception:
                pass

    elapsed = time.time() - t0
    tokens = count_tokens_rough(total_text)

    return {
        "results": len(all_results),
        "api_calls": api_calls,
        "time_s": round(elapsed, 3),
        "total_tokens": tokens,
        "total_chars": len(total_text),
    }


def clio_search(
    connector: NDPConnector,
    coordinator: RetrievalCoordinator,
    query: str,
    numeric_range: str | None,
) -> dict[str, Any]:
    """Search via CLIO. Measure everything."""
    operators = ScientificQueryOperators()
    if numeric_range:
        parts = numeric_range.split(":")
        operators = ScientificQueryOperators(
            numeric_range=NumericRangeOperator(
                minimum=float(parts[0]), maximum=float(parts[1]), unit=parts[2],
            ),
        )

    t0 = time.time()
    result = coordinator.query(
        connector=connector, query=query, top_k=10,
        scientific_operators=operators,
    )
    elapsed = time.time() - t0

    # Measure actual text returned
    total_text = " ".join(c.snippet for c in result.citations)
    tokens = count_tokens_rough(total_text)

    branches = []
    for event in result.trace:
        if event.stage == "branch_plan_selected":
            for b in ("lexical", "vector", "graph", "scientific"):
                if event.attributes.get(f"use_{b}") == "True":
                    branches.append(b)

    return {
        "results": len(result.citations),
        "time_s": round(elapsed, 3),
        "total_tokens": tokens,
        "total_chars": len(total_text),
        "branches": branches,
    }


def main() -> None:
    print("REAL TEST: NDP without CLIO vs with CLIO")
    print("=" * 60)

    with tempfile.TemporaryDirectory(prefix="clio_real_") as tmpdir:
        tmpdir = Path(tmpdir)

        # Index NDP data into CLIO
        print("\nIndexing NDP data...")
        db = DuckDBStorage(database_path=tmpdir / "ndp.duckdb")
        conn = NDPConnector(namespace="ndp", storage=db)
        conn.connect()

        t0 = time.time()
        for term in ["temperature", "pressure", "wind", "humidity", "radiation", "glacier", "ocean"]:
            try:
                ds = conn.discover_datasets(search_terms=[term], limit=25)
                conn.index_datasets(ds)
            except Exception:
                pass
        index_time = time.time() - t0

        profile = build_corpus_profile(db, "ndp")
        print(f"  Indexed: {profile.document_count} docs, {profile.chunk_count} chunks, "
              f"{profile.measurement_count} meas, density={profile.metadata_density:.1%}")
        print(f"  Index time: {index_time:.1f}s")

        t_profile = time.time()
        _ = build_corpus_profile(db, "ndp")
        profile_time = time.time() - t_profile
        print(f"  Profile time: {profile_time*1000:.1f}ms")

        # Run queries
        coordinator = RetrievalCoordinator()
        results: list[dict[str, Any]] = []

        print(f"\n{'ID':<5} | {'NDP':>6} res {'':>6} tok {'':>6} ms | "
              f"{'CLIO':>6} res {'':>6} tok {'':>6} ms | branches")
        print("-" * 80)

        for q in QUERIES:
            ndp = ndp_raw_search(q["terms"])
            clio = clio_search(conn, coordinator, q["query"], q.get("range"))

            print(f"{q['id']:<5} | {ndp['results']:>6} {ndp['total_tokens']:>9} {ndp['time_s']*1000:>7.0f} | "
                  f"{clio['results']:>6} {clio['total_tokens']:>9} {clio['time_s']*1000:>7.0f} | "
                  f"{','.join(clio['branches'])}")

            results.append({
                "id": q["id"],
                "terms": q["terms"],
                "query": q["query"],
                "has_range": "range" in q,
                "ndp_without_clio": ndp,
                "clio_search": clio,
            })

        # Totals
        ndp_total_results = sum(r["ndp_without_clio"]["results"] for r in results)
        ndp_total_tokens = sum(r["ndp_without_clio"]["total_tokens"] for r in results)
        ndp_total_time = sum(r["ndp_without_clio"]["time_s"] for r in results)
        ndp_total_api = sum(r["ndp_without_clio"]["api_calls"] for r in results)

        clio_total_results = sum(r["clio_search"]["results"] for r in results)
        clio_total_tokens = sum(r["clio_search"]["total_tokens"] for r in results)
        clio_total_time = sum(r["clio_search"]["time_s"] for r in results)

        print("-" * 80)
        print(f"TOTAL | {ndp_total_results:>6} {ndp_total_tokens:>9} {ndp_total_time*1000:>7.0f} | "
              f"{clio_total_results:>6} {clio_total_tokens:>9} {clio_total_time*1000:>7.0f} |")

        token_reduction = (1 - clio_total_tokens / ndp_total_tokens) * 100 if ndp_total_tokens > 0 else 0

        print(f"\nToken reduction: {token_reduction:.1f}%")
        print(f"NDP API calls: {ndp_total_api}")
        print(f"CLIO index time (one-time): {index_time:.1f}s")
        print(f"CLIO profile time: {profile_time*1000:.1f}ms")

        # Cleanup
        conn.teardown()
        db.teardown()

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "test": "NDP-MCP without CLIO vs with CLIO",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ndp_index": {
            "documents": profile.document_count,
            "chunks": profile.chunk_count,
            "measurements": profile.measurement_count,
            "metadata_density": round(profile.metadata_density, 4),
            "index_time_s": round(index_time, 2),
            "profile_time_ms": round(profile_time * 1000, 2),
        },
        "queries": results,
        "totals": {
            "ndp_results": ndp_total_results,
            "ndp_tokens": ndp_total_tokens,
            "ndp_time_s": round(ndp_total_time, 3),
            "ndp_api_calls": ndp_total_api,
            "clio_results": clio_total_results,
            "clio_tokens": clio_total_tokens,
            "clio_time_s": round(clio_total_time, 3),
            "token_reduction_pct": round(token_reduction, 1),
        },
    }
    out_path = OUT_DIR / "ndp_test.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
