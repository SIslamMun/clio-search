#!/usr/bin/env python3
"""Real-world evaluation benchmark for clio-agentic-search.

Evaluates retrieval quality on the NOAA GHCN-Daily corpus (288 documents,
8 weather stations, 3 years 2022-2024).  Queries are expressed in imperial
units (°F, inches, mph) while all documents contain only metric units
(degC, mm, m/s).  This tests whether our SI conversion operators can bridge
the gap that lexical and semantic baselines cannot.

Baselines:
  BM25 only          — lexical, no unit conversion
  Dense only         — semantic, partial unit awareness via embedding
  Hybrid             — BM25 + vector, no scientific operators
  String-norm        — hybrid + unit alias expansion (degF → Fahrenheit etc.)
  Full pipeline      — hybrid + SI numeric-range conversion operators

Usage:
    cd code && python3 benchmarks/evaluate_real.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE_DIR / "src"))

from clio_agentic_search.connectors.filesystem.connector import FilesystemConnector
from clio_agentic_search.retrieval.coordinator import RetrievalCoordinator
from clio_agentic_search.retrieval.scientific import (
    NumericRangeOperator,
    ScientificQueryOperators,
)
from clio_agentic_search.storage.duckdb_store import DuckDBStorage

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BENCHMARK_DIR = Path(__file__).resolve().parent
CORPUS_DIR = BENCHMARK_DIR / "corpus_real"
QUERIES_PATH = BENCHMARK_DIR / "real_queries.json"
EVAL_DIR = BENCHMARK_DIR.parent.parent / "eval"
RESULTS_PATH = EVAL_DIR / "real_benchmark_results.json"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    return sum(1 for d in top if d in relevant) / len(top) if top else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    top = retrieved[:k]
    return sum(1 for d in top if d in relevant) / len(relevant)


def f1_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def compute_metrics(retrieved: list[str], relevant: set[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for k in (1, 5, 10, 20):
        result[f"P@{k}"] = round(precision_at_k(retrieved, relevant, k), 4)
        result[f"R@{k}"] = round(recall_at_k(retrieved, relevant, k), 4)
        result[f"F1@{k}"] = round(f1_at_k(retrieved, relevant, k), 4)
    result["MRR"] = round(mrr(retrieved, relevant), 4)
    return result


def average_metrics(all_metrics: list[dict[str, float]]) -> dict[str, float]:
    if not all_metrics:
        return {}
    return {
        key: round(sum(m[key] for m in all_metrics) / len(all_metrics), 4)
        for key in all_metrics[0]
    }


# ---------------------------------------------------------------------------
# Connector setup
# ---------------------------------------------------------------------------

def _setup_connector(
    tmpdir: Path,
    corpus_src: Path,
    namespace: str = "real_bench",
) -> tuple[FilesystemConnector, DuckDBStorage]:
    dest = tmpdir / "corpus"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(corpus_src, dest)

    db_path = tmpdir / "real_bench.duckdb"
    storage = DuckDBStorage(database_path=db_path)
    connector = FilesystemConnector(
        namespace=namespace,
        root=dest,
        storage=storage,
        warmup_async=False,
    )
    connector.connect()
    connector.index(full_rebuild=True)
    return connector, storage


def _retrieved_doc_paths(citations: list[Any]) -> list[str]:
    """Return ordered deduplicated filenames from citation URIs."""
    paths: list[str] = []
    for c in citations:
        # URIs are bare filenames like 'USW00014732_2024_01.txt'
        path = Path(c.uri.split("#")[0]).name
        if path not in paths:
            paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Scientific operator builders
# (These encode the ground-truth conversions: what the query MEANS in metric)
# ---------------------------------------------------------------------------

# Temperature conversions: F -> C
# 86°F = 30.0°C, 32°F = 0°C, 23°F = -5°C, 59°F = 15°C, 77°F = 25°C
# Precipitation: 1 inch = 25.4mm, 2 inches = 50.8mm, 5 inches = 127mm
# Wind: 15 mph = 6.705 m/s

# Ground-truth unit conversions for each query.
# Temperature queries specify degF thresholds; the system converts to Kelvin.
# Precipitation/snow queries specify mm thresholds (canonical: m).
# Wind queries specify m/s thresholds (canonical: m/s).
_QUERY_OPERATORS: dict[str, ScientificQueryOperators] = {
    # "max temp > 86°F" → degF threshold → internally converted to K
    "real_001": ScientificQueryOperators(
        numeric_range=NumericRangeOperator(minimum=86.0, maximum=200.0, unit="degF")
    ),
    # "min temp < 32°F (freezing)" → degF threshold
    "real_002": ScientificQueryOperators(
        numeric_range=NumericRangeOperator(minimum=-200.0, maximum=32.0, unit="degF")
    ),
    # "avg max between 59°F and 77°F" → degF range
    "real_003": ScientificQueryOperators(
        numeric_range=NumericRangeOperator(minimum=59.0, maximum=77.0, unit="degF")
    ),
    # "total rain > 2 inches" = 50.8 mm threshold
    "real_004": ScientificQueryOperators(
        numeric_range=NumericRangeOperator(minimum=50.8, maximum=100000.0, unit="mm")
    ),
    # "daily rain > 1 inch" = 25.4 mm
    "real_005": ScientificQueryOperators(
        numeric_range=NumericRangeOperator(minimum=25.4, maximum=100000.0, unit="mm")
    ),
    # "wind > 15 mph" = 6.705 m/s
    "real_006": ScientificQueryOperators(
        numeric_range=NumericRangeOperator(minimum=6.705, maximum=1000.0, unit="m/s")
    ),
    # "snowfall > 5 inches" = 127 mm
    "real_007": ScientificQueryOperators(
        numeric_range=NumericRangeOperator(minimum=127.0, maximum=100000.0, unit="mm")
    ),
    # combined: temp < 23°F and snow > 2 inches
    "real_008": ScientificQueryOperators(
        numeric_range=NumericRangeOperator(minimum=127.0, maximum=100000.0, unit="mm")
    ),
}


# ---------------------------------------------------------------------------
# Baseline runners
# ---------------------------------------------------------------------------

def _run_bm25(connector: FilesystemConnector, query: str, top_k: int) -> list[str]:
    results = connector.search_lexical(query, top_k=top_k)
    return _retrieved_doc_paths([connector.build_citation(r) for r in results])


def _run_dense(connector: FilesystemConnector, query: str, top_k: int) -> list[str]:
    results = connector.search_vector(query, top_k=top_k)
    return _retrieved_doc_paths([connector.build_citation(r) for r in results])


def _run_hybrid(
    connector: FilesystemConnector,
    coordinator: RetrievalCoordinator,
    query: str,
    top_k: int,
) -> list[str]:
    result = coordinator.query(
        connector=connector,
        query=query,
        top_k=top_k,
        scientific_operators=ScientificQueryOperators(),
    )
    return _retrieved_doc_paths(result.citations)


def _run_string_norm(
    connector: FilesystemConnector,
    coordinator: RetrievalCoordinator,
    query: str,
    top_k: int,
) -> list[str]:
    try:
        from clio_agentic_search.retrieval.query_rewriter import _expand_unit_variants
        extra = _expand_unit_variants(query)
        expanded = f"{query} {' '.join(extra)}" if extra else query
    except Exception:
        expanded = query
    result = coordinator.query(
        connector=connector,
        query=expanded,
        top_k=top_k,
        scientific_operators=ScientificQueryOperators(),
    )
    return _retrieved_doc_paths(result.citations)


def _run_full_pipeline(
    connector: FilesystemConnector,
    coordinator: RetrievalCoordinator,
    query: str,
    operators: ScientificQueryOperators,
    top_k: int,
) -> list[str]:
    result = coordinator.query(
        connector=connector,
        query=query,
        top_k=top_k,
        scientific_operators=operators,
    )
    return _retrieved_doc_paths(result.citations)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    connector: FilesystemConnector,
    queries: list[dict[str, Any]],
    top_k: int = 20,
) -> dict[str, Any]:
    coordinator = RetrievalCoordinator()

    baselines = {
        "bm25_only": [],
        "dense_only": [],
        "hybrid": [],
        "string_norm": [],
        "full_pipeline": [],
    }

    per_query_results: dict[str, list[dict[str, Any]]] = {b: [] for b in baselines}

    for q in queries:
        qid = q["id"]
        query_text = q["query"]
        relevant = set(q["relevant_docs"])
        # Normalize to filename-only for comparison.
        # The connector returns URIs as bare filenames (relative to corpus root),
        # while the query stores absolute paths — so we compare basenames only.
        relevant_abs = set(Path(rp).name for rp in relevant)

        operators = _QUERY_OPERATORS.get(qid, ScientificQueryOperators())

        t0 = time.perf_counter()
        bm25_r = _run_bm25(connector, query_text, top_k)
        t_bm25 = time.perf_counter() - t0

        t0 = time.perf_counter()
        dense_r = _run_dense(connector, query_text, top_k)
        t_dense = time.perf_counter() - t0

        t0 = time.perf_counter()
        hybrid_r = _run_hybrid(connector, coordinator, query_text, top_k)
        t_hybrid = time.perf_counter() - t0

        t0 = time.perf_counter()
        sn_r = _run_string_norm(connector, coordinator, query_text, top_k)
        t_sn = time.perf_counter() - t0

        t0 = time.perf_counter()
        fp_r = _run_full_pipeline(connector, coordinator, query_text, operators, top_k)
        t_fp = time.perf_counter() - t0

        run_data = {
            "bm25_only":     (bm25_r,  t_bm25),
            "dense_only":    (dense_r,  t_dense),
            "hybrid":        (hybrid_r, t_hybrid),
            "string_norm":   (sn_r,     t_sn),
            "full_pipeline": (fp_r,     t_fp),
        }

        for bname, (retrieved, latency) in run_data.items():
            m = compute_metrics(retrieved, relevant_abs)
            baselines[bname].append(m)
            per_query_results[bname].append({
                "query_id": qid,
                "query": query_text,
                "type": q.get("type", ""),
                "relevant_count": len(relevant_abs),
                "metrics": m,
                "top5": retrieved[:5],
                "latency_s": round(latency, 4),
            })

    # Average across queries
    averages: dict[str, dict[str, float]] = {}
    for bname, metric_list in baselines.items():
        averages[bname] = average_metrics(metric_list)

    return {
        "averages": averages,
        "per_query": per_query_results,
    }


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------

def print_results(results: dict[str, Any], query_count: int, doc_count: int) -> None:
    sep = "-" * 85
    print("\n" + "=" * 85)
    print("  REAL-WORLD BENCHMARK: NOAA GHCN-Daily Cross-Unit Retrieval")
    print(f"  {doc_count} documents (8 stations × 3 years, metric units only)")
    print(f"  {query_count} queries in imperial units (°F, inches, mph)")
    print("  Baselines vs. Clio Full Pipeline (SI operator conversion)")
    print("=" * 85)
    header = (
        f"{'Method':<20} "
        f"{'P@5':>6} {'R@5':>6} {'F1@5':>6} "
        f"{'P@10':>6} {'R@10':>6} "
        f"{'MRR':>6}"
    )
    print(header)
    print(sep)

    avgs = results["averages"]
    display_order = ["bm25_only", "dense_only", "hybrid", "string_norm", "full_pipeline"]
    display_names = {
        "bm25_only": "BM25 Only",
        "dense_only": "Dense (Vector)",
        "hybrid": "Hybrid (BM25+Vec)",
        "string_norm": "String Norm.",
        "full_pipeline": "Clio Full Pipeline",
    }

    for bname in display_order:
        m = avgs.get(bname, {})
        print(
            f"{display_names[bname]:<20} "
            f"{m.get('P@5', 0):>6.3f} "
            f"{m.get('R@5', 0):>6.3f} "
            f"{m.get('F1@5', 0):>6.3f} "
            f"{m.get('P@10', 0):>6.3f} "
            f"{m.get('R@10', 0):>6.3f} "
            f"{m.get('MRR', 0):>6.3f}"
        )

    print(sep)

    # Improvement of full pipeline over hybrid
    fp = avgs.get("full_pipeline", {})
    hyb = avgs.get("hybrid", {})
    for k in ("P@5", "R@5", "MRR"):
        delta = fp.get(k, 0) - hyb.get(k, 0)
        print(f"  Full Pipeline vs Hybrid {k}: {delta:+.3f}")

    print("=" * 85 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not CORPUS_DIR.exists() or not any(CORPUS_DIR.glob("*.txt")):
        print("Real corpus not found. Run: python3 benchmarks/build_real_corpus.py")
        sys.exit(1)

    if not QUERIES_PATH.exists():
        print("real_queries.json not found. Run: python3 benchmarks/build_real_corpus.py")
        sys.exit(1)

    with open(QUERIES_PATH) as fh:
        data = json.load(fh)
    queries = data["queries"]

    doc_count = len(list(CORPUS_DIR.glob("*.txt")))
    print(f"Real-world benchmark: {doc_count} documents, {len(queries)} queries")

    with tempfile.TemporaryDirectory(prefix="clio_real_bench_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        print("Indexing real corpus ...")
        t_idx = time.perf_counter()
        connector, _ = _setup_connector(tmpdir, CORPUS_DIR)
        idx_time = time.perf_counter() - t_idx
        print(f"  Indexed {doc_count} documents in {idx_time:.1f}s")

        print("Running baseline evaluation ...")
        results = run_evaluation(connector, queries, top_k=20)

        connector.teardown()

    results["meta"] = {
        "corpus": "NOAA GHCN-Daily",
        "stations": data.get("stations", []),
        "doc_count": doc_count,
        "query_count": len(queries),
        "indexing_time_s": round(idx_time, 2),
        "years": [2022, 2023, 2024],
        "doc_units": "degC, mm, m/s",
        "query_units": "°F, inches, mph",
        "note": (
            "Documents contain ONLY metric/SI units. "
            "Queries expressed in imperial units. "
            "Cross-unit matching requires SI conversion operators."
        ),
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as fh:
        json.dump(results, fh, indent=2)

    print_results(results, len(queries), doc_count)
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
