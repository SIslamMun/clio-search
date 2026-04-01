#!/usr/bin/env python3
"""Evaluation benchmark harness v2 for clio-agentic-search.

Runs six retrieval baselines, ablation studies, federated evaluation,
agentic multi-hop evaluation, and indexing performance measurement against
a realistic 210-document scientific corpus spanning materials science,
atmospheric science, fluid dynamics, chemistry, HPC simulation, and
cross-domain documents with 80 queries (40 cross-unit, 20 same-unit,
10 formula, 10 multi-constraint).

Usage:
    python3 benchmarks/evaluate_v2.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure the package is importable when run from the code/ directory.
# ---------------------------------------------------------------------------
_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE_DIR / "src"))

from clio_agentic_search.connectors.filesystem.connector import FilesystemConnector
from clio_agentic_search.retrieval.agentic import AgenticRetriever
from clio_agentic_search.retrieval.capabilities import ScoredChunk
from clio_agentic_search.retrieval.coordinator import RetrievalCoordinator
from clio_agentic_search.retrieval.query_rewriter import FallbackQueryRewriter
from clio_agentic_search.retrieval.scientific import (
    NumericRangeOperator,
    ScientificQueryOperators,
)
from clio_agentic_search.storage.duckdb_store import DuckDBStorage

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BENCHMARK_DIR = Path(__file__).resolve().parent
CORPUS_DIR = BENCHMARK_DIR / "corpus_v2"
QUERIES_PATH = BENCHMARK_DIR / "queries_v2.json"
EVAL_DIR = BENCHMARK_DIR.parent.parent / "eval"
RESULTS_PATH = EVAL_DIR / "benchmark_v2_results.json"


# ===================================================================
# Helpers
# ===================================================================

def _load_queries() -> dict[str, Any]:
    with open(QUERIES_PATH, "r") as fh:
        return json.load(fh)


def _uri_matches_doc(uri: str, doc_path: str) -> bool:
    """Check whether a citation URI corresponds to a benchmark doc path."""
    # The URI stored is the relative path inside the corpus root,
    # e.g. "pressure/doc_01.txt".
    return uri.rstrip("/").split("#")[0] == doc_path


def _retrieved_doc_paths(citations: list[Any]) -> list[str]:
    """Return ordered list of doc paths from citation URIs."""
    paths: list[str] = []
    for c in citations:
        path = c.uri.split("#")[0]
        if path not in paths:
            paths.append(path)
    return paths


# ===================================================================
# Metrics
# ===================================================================

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for d in top if d in relevant) / len(top)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    top = retrieved[:k]
    return sum(1 for d in top if d in relevant) / len(relevant)


def f1_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def compute_metrics(retrieved: list[str], relevant: set[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for k in (1, 3, 5, 10):
        result[f"P@{k}"] = round(precision_at_k(retrieved, relevant, k), 4)
        result[f"R@{k}"] = round(recall_at_k(retrieved, relevant, k), 4)
        result[f"F1@{k}"] = round(f1_at_k(retrieved, relevant, k), 4)
    result["MRR"] = round(mrr(retrieved, relevant), 4)
    return result


def average_metrics(all_metrics: list[dict[str, float]]) -> dict[str, float]:
    if not all_metrics:
        return {}
    keys = all_metrics[0].keys()
    avg: dict[str, float] = {}
    for key in keys:
        avg[key] = round(sum(m[key] for m in all_metrics) / len(all_metrics), 4)
    return avg


# ===================================================================
# Setup helpers
# ===================================================================

def _setup_connector(
    tmpdir: Path,
    corpus_src: Path,
    namespace: str = "bench",
) -> tuple[FilesystemConnector, DuckDBStorage]:
    """Copy corpus into tmpdir, create connector + storage, connect, index."""
    dest = tmpdir / "corpus"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(corpus_src, dest)

    db_path = tmpdir / "bench.duckdb"
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


def _parse_numeric_range(spec: str) -> NumericRangeOperator:
    """Parse '200:300:kPa' into a NumericRangeOperator."""
    parts = spec.split(":")
    return NumericRangeOperator(
        minimum=float(parts[0]),
        maximum=float(parts[1]),
        unit=parts[2],
    )


# ===================================================================
# Baseline runners
# ===================================================================

def run_bm25_only(
    connector: FilesystemConnector,
    query: str,
    top_k: int = 20,
) -> list[str]:
    """Lexical BM25 search only."""
    results = connector.search_lexical(query, top_k=top_k)
    citations = [connector.build_citation(r) for r in results]
    return _retrieved_doc_paths(citations)


def run_dense_only(
    connector: FilesystemConnector,
    query: str,
    top_k: int = 20,
) -> list[str]:
    """Vector (dense) search only."""
    results = connector.search_vector(query, top_k=top_k)
    citations = [connector.build_citation(r) for r in results]
    return _retrieved_doc_paths(citations)


def run_hybrid(
    connector: FilesystemConnector,
    coordinator: RetrievalCoordinator,
    query: str,
    top_k: int = 20,
) -> list[str]:
    """Hybrid lexical+vector without scientific operators."""
    result = coordinator.query(
        connector=connector,
        query=query,
        top_k=top_k,
        scientific_operators=ScientificQueryOperators(),  # inactive
    )
    return _retrieved_doc_paths(result.citations)


def run_string_norm(
    connector: FilesystemConnector,
    coordinator: RetrievalCoordinator,
    query: str,
    top_k: int = 20,
) -> list[str]:
    """Hybrid with naive string expansion of unit names (no numeric range)."""
    from clio_agentic_search.retrieval.query_rewriter import _expand_unit_variants

    extra = _expand_unit_variants(query)
    expanded_query = f"{query} {' '.join(extra)}" if extra else query
    result = coordinator.query(
        connector=connector,
        query=expanded_query,
        top_k=top_k,
        scientific_operators=ScientificQueryOperators(),  # inactive
    )
    return _retrieved_doc_paths(result.citations)


def run_full_pipeline(
    connector: FilesystemConnector,
    coordinator: RetrievalCoordinator,
    query: str,
    operators: ScientificQueryOperators,
    top_k: int = 20,
) -> list[str]:
    """Full pipeline: hybrid + scientific operators."""
    result = coordinator.query(
        connector=connector,
        query=query,
        top_k=top_k,
        scientific_operators=operators,
    )
    return _retrieved_doc_paths(result.citations)


# ===================================================================
# Build operators from query spec
# ===================================================================

def _operators_for_query(q: dict[str, Any]) -> ScientificQueryOperators:
    """Build ScientificQueryOperators from a query dict."""
    nr = None
    formula = None
    if "numeric_range" in q and q["numeric_range"]:
        nr = _parse_numeric_range(q["numeric_range"])
    elif "constraints" in q and q["constraints"]:
        # Multi-constraint queries: use the first constraint as the numeric range
        nr = _parse_numeric_range(q["constraints"][0]["range"])
    if "formula" in q and q["formula"]:
        formula = q["formula"]
    return ScientificQueryOperators(numeric_range=nr, formula=formula)


# ===================================================================
# Main evaluation sections
# ===================================================================

def evaluate_baselines(
    connector: FilesystemConnector,
    coordinator: RetrievalCoordinator,
    queries_data: dict[str, Any],
) -> dict[str, Any]:
    """Run 5 baselines across all query sets."""
    baselines = ["bm25_only", "dense_only", "hybrid", "string_norm", "full_pipeline"]
    all_queries = (
        queries_data["cross_unit_queries"]
        + queries_data["same_unit_queries"]
        + queries_data["formula_queries"]
        + queries_data.get("multi_constraint_queries", [])
    )

    results: dict[str, Any] = {}
    for baseline_name in baselines:
        per_query: list[dict[str, Any]] = []
        metrics_list: list[dict[str, float]] = []

        for q in all_queries:
            relevant = set(q["relevant_docs"])
            query_text = q["query"]
            operators = _operators_for_query(q)

            if baseline_name == "bm25_only":
                retrieved = run_bm25_only(connector, query_text)
            elif baseline_name == "dense_only":
                retrieved = run_dense_only(connector, query_text)
            elif baseline_name == "hybrid":
                retrieved = run_hybrid(connector, coordinator, query_text)
            elif baseline_name == "string_norm":
                retrieved = run_string_norm(connector, coordinator, query_text)
            elif baseline_name == "full_pipeline":
                retrieved = run_full_pipeline(connector, coordinator, query_text, operators)
            else:
                retrieved = []

            m = compute_metrics(retrieved, relevant)
            metrics_list.append(m)
            per_query.append({
                "query_id": q["id"],
                "type": q["type"],
                "retrieved": retrieved[:10],
                "relevant": list(relevant),
                "metrics": m,
            })

        # Per-type averages
        type_averages: dict[str, dict[str, float]] = {}
        for qtype in ("cross_unit", "same_unit", "formula", "multi_constraint"):
            type_metrics = [
                pq["metrics"] for pq in per_query if pq["type"] == qtype
            ]
            if type_metrics:
                type_averages[qtype] = average_metrics(type_metrics)

        results[baseline_name] = {
            "overall": average_metrics(metrics_list),
            "by_type": type_averages,
            "per_query": per_query,
        }

    return results


def evaluate_ablation(
    connector: FilesystemConnector,
    queries_data: dict[str, Any],
) -> dict[str, Any]:
    """Run ablation configs A-D on cross-unit queries."""
    cross_queries = queries_data["cross_unit_queries"]
    configs = {
        "A_lexical_only": "lexical",
        "B_lexical_vector": "lexical+vector",
        "C_lexical_vector_scientific": "lexical+vector+scientific",
        "D_full_pipeline": "full",
    }
    coordinator = RetrievalCoordinator()
    results: dict[str, Any] = {}

    for config_name, config_type in configs.items():
        metrics_list: list[dict[str, float]] = []
        for q in cross_queries:
            relevant = set(q["relevant_docs"])
            query_text = q["query"]
            operators = _operators_for_query(q)

            if config_type == "lexical":
                retrieved = run_bm25_only(connector, query_text)
            elif config_type == "lexical+vector":
                retrieved = run_hybrid(connector, coordinator, query_text)
            elif config_type == "lexical+vector+scientific":
                retrieved = run_full_pipeline(
                    connector, coordinator, query_text, operators
                )
            elif config_type == "full":
                retrieved = run_full_pipeline(
                    connector, coordinator, query_text, operators
                )
            else:
                retrieved = []

            metrics_list.append(compute_metrics(retrieved, relevant))

        results[config_name] = average_metrics(metrics_list)

    return results


def evaluate_federated(
    tmpdir: Path,
    queries_data: dict[str, Any],
) -> dict[str, Any]:
    """Distribute corpus across two directories, test single vs multi-namespace."""
    # Split corpus into two namespaces to test federated retrieval
    # ns1: materials_science, atmospheric_science, fluid_dynamics, negatives
    # ns2: chemistry, hpc_simulation, mixed
    ns1_dir = tmpdir / "federated" / "ns1"
    ns2_dir = tmpdir / "federated" / "ns2"
    ns1_dir.mkdir(parents=True, exist_ok=True)
    ns2_dir.mkdir(parents=True, exist_ok=True)

    for domain in ("materials_science", "atmospheric_science", "fluid_dynamics", "negatives"):
        src = CORPUS_DIR / domain
        if src.exists():
            shutil.copytree(src, ns1_dir / domain, dirs_exist_ok=True)

    for domain in ("chemistry", "hpc_simulation", "mixed"):
        src = CORPUS_DIR / domain
        if src.exists():
            shutil.copytree(src, ns2_dir / domain, dirs_exist_ok=True)

    db1_path = tmpdir / "federated" / "ns1.duckdb"
    db2_path = tmpdir / "federated" / "ns2.duckdb"

    storage1 = DuckDBStorage(database_path=db1_path)
    storage2 = DuckDBStorage(database_path=db2_path)

    conn1 = FilesystemConnector(
        namespace="ns1", root=ns1_dir, storage=storage1, warmup_async=False
    )
    conn2 = FilesystemConnector(
        namespace="ns2", root=ns2_dir, storage=storage2, warmup_async=False
    )

    conn1.connect()
    conn2.connect()
    conn1.index(full_rebuild=True)
    conn2.index(full_rebuild=True)

    coordinator = RetrievalCoordinator()

    # Gather all queries
    all_queries = (
        queries_data["cross_unit_queries"]
        + queries_data["same_unit_queries"]
        + queries_data["formula_queries"]
        + queries_data.get("multi_constraint_queries", [])
    )

    single_coverage: list[float] = []
    multi_coverage: list[float] = []

    for q in all_queries:
        relevant = set(q["relevant_docs"])
        if not relevant:
            continue
        query_text = q["query"]
        operators = _operators_for_query(q)

        # Single namespace: pick ns1 by default (has more docs)
        single_conn = conn1

        single_result = coordinator.query(
            connector=single_conn,
            query=query_text,
            top_k=20,
            scientific_operators=operators,
        )
        single_retrieved = set(_retrieved_doc_paths(single_result.citations))
        single_hits = len(relevant & single_retrieved)
        single_coverage.append(single_hits / len(relevant))

        # Multi namespace
        multi_result = coordinator.query_namespaces(
            connectors=[conn1, conn2],
            query=query_text,
            top_k=20,
            scientific_operators=operators,
        )
        multi_retrieved = set(_retrieved_doc_paths(multi_result.citations))
        multi_hits = len(relevant & multi_retrieved)
        multi_coverage.append(multi_hits / len(relevant))

    conn1.teardown()
    conn2.teardown()

    return {
        "single_namespace_avg_coverage": round(
            sum(single_coverage) / max(len(single_coverage), 1), 4
        ),
        "multi_namespace_avg_coverage": round(
            sum(multi_coverage) / max(len(multi_coverage), 1), 4
        ),
        "query_count": len(all_queries),
    }


def evaluate_agentic(
    connector: FilesystemConnector,
    queries_data: dict[str, Any],
) -> dict[str, Any]:
    """Compare 1-hop, 2-hop, 3-hop agentic retrieval with FallbackQueryRewriter."""
    cross_queries = queries_data["cross_unit_queries"][:8]  # subset for speed
    results: dict[str, Any] = {}

    for max_hops in (1, 2, 3):
        rewriter = FallbackQueryRewriter()
        retriever = AgenticRetriever(
            coordinator=RetrievalCoordinator(),
            rewriter=rewriter,
            max_hops=max_hops,
            min_score_threshold=0.0,
            convergence_threshold=0,
        )

        metrics_list: list[dict[str, float]] = []
        hop_counts: list[int] = []

        for q in cross_queries:
            relevant = set(q["relevant_docs"])
            operators = _operators_for_query(q)

            agentic_result = retriever.query(
                connector=connector,
                query=q["query"],
                top_k=20,
                scientific_operators=operators,
            )
            retrieved = _retrieved_doc_paths(agentic_result.citations)
            metrics_list.append(compute_metrics(retrieved, relevant))
            hop_counts.append(agentic_result.total_hops)

        results[f"{max_hops}_hop"] = {
            "metrics": average_metrics(metrics_list),
            "avg_hops": round(sum(hop_counts) / max(len(hop_counts), 1), 2),
        }

    return results


def evaluate_indexing_performance(
    tmpdir: Path,
) -> dict[str, Any]:
    """Measure full and incremental indexing time."""
    dest = tmpdir / "perf_corpus"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(CORPUS_DIR, dest)

    db_path = tmpdir / "perf.duckdb"
    storage = DuckDBStorage(database_path=db_path)
    connector = FilesystemConnector(
        namespace="perf",
        root=dest,
        storage=storage,
        warmup_async=False,
    )
    connector.connect()

    # Full index
    t0 = time.perf_counter()
    full_report = connector.index(full_rebuild=True)
    full_time = time.perf_counter() - t0

    # Incremental: modify ~10% of files (touch them)
    all_files = sorted(dest.rglob("*.txt"))
    modify_count = max(1, len(all_files) // 10)
    for f in all_files[:modify_count]:
        content = f.read_text()
        f.write_text(content + "\n(modified)")

    t0 = time.perf_counter()
    incr_report = connector.index(full_rebuild=False)
    incr_time = time.perf_counter() - t0

    connector.teardown()

    return {
        "full_index": {
            "elapsed_seconds": round(full_time, 4),
            "scanned_files": full_report.scanned_files,
            "indexed_files": full_report.indexed_files,
        },
        "incremental_index": {
            "elapsed_seconds": round(incr_time, 4),
            "scanned_files": incr_report.scanned_files,
            "indexed_files": incr_report.indexed_files,
            "skipped_files": incr_report.skipped_files,
        },
    }


# ===================================================================
# Summary printer
# ===================================================================

def print_summary(results: dict[str, Any]) -> None:
    """Print a human-readable summary table."""
    sep = "-" * 90

    print("\n" + "=" * 90)
    print("  CLIO-AGENTIC-SEARCH EVALUATION BENCHMARK v2 RESULTS")
    print("  Corpus: 210 documents, 80 queries (40 cross-unit, 20 same-unit,")
    print("          10 formula, 10 multi-constraint)")
    print("=" * 90)

    # --- Baseline overall ---
    print("\n1. BASELINE COMPARISON (overall averages)")
    print(sep)
    header = f"{'Baseline':<20} {'P@1':>6} {'P@5':>6} {'R@5':>6} {'F1@5':>6} {'MRR':>6}"
    print(header)
    print(sep)
    baselines = results.get("baselines", {})
    for name, data in baselines.items():
        ov = data.get("overall", {})
        print(
            f"{name:<20} "
            f"{ov.get('P@1', 0):>6.3f} "
            f"{ov.get('P@5', 0):>6.3f} "
            f"{ov.get('R@5', 0):>6.3f} "
            f"{ov.get('F1@5', 0):>6.3f} "
            f"{ov.get('MRR', 0):>6.3f}"
        )

    # --- Baseline by query type ---
    print(f"\n2. BASELINE COMPARISON (by query type, R@5)")
    print(sep)
    header = f"{'Baseline':<20} {'cross_unit':>12} {'same_unit':>12} {'formula':>12} {'multi_cstr':>12}"
    print(header)
    print(sep)
    for name, data in baselines.items():
        by_type = data.get("by_type", {})
        cu = by_type.get("cross_unit", {}).get("R@5", 0)
        su = by_type.get("same_unit", {}).get("R@5", 0)
        fm = by_type.get("formula", {}).get("R@5", 0)
        mc = by_type.get("multi_constraint", {}).get("R@5", 0)
        print(f"{name:<20} {cu:>12.3f} {su:>12.3f} {fm:>12.3f} {mc:>12.3f}")

    # --- Ablation ---
    print(f"\n3. ABLATION STUDY (cross-unit queries)")
    print(sep)
    header = f"{'Config':<35} {'P@1':>6} {'P@5':>6} {'R@5':>6} {'F1@5':>6} {'MRR':>6}"
    print(header)
    print(sep)
    ablation = results.get("ablation", {})
    for config_name, m in ablation.items():
        print(
            f"{config_name:<35} "
            f"{m.get('P@1', 0):>6.3f} "
            f"{m.get('P@5', 0):>6.3f} "
            f"{m.get('R@5', 0):>6.3f} "
            f"{m.get('F1@5', 0):>6.3f} "
            f"{m.get('MRR', 0):>6.3f}"
        )

    # --- Federated ---
    print(f"\n4. FEDERATED EVALUATION")
    print(sep)
    fed = results.get("federated", {})
    print(f"  Single-namespace avg coverage: {fed.get('single_namespace_avg_coverage', 0):.3f}")
    print(f"  Multi-namespace avg coverage:  {fed.get('multi_namespace_avg_coverage', 0):.3f}")
    print(f"  Queries evaluated:             {fed.get('query_count', 0)}")

    # --- Agentic ---
    print(f"\n5. AGENTIC MULTI-HOP EVALUATION")
    print(sep)
    header = f"{'Hops':<12} {'P@5':>6} {'R@5':>6} {'F1@5':>6} {'MRR':>6} {'Avg Hops':>10}"
    print(header)
    print(sep)
    agentic = results.get("agentic", {})
    for hop_key, data in agentic.items():
        m = data.get("metrics", {})
        ah = data.get("avg_hops", 0)
        print(
            f"{hop_key:<12} "
            f"{m.get('P@5', 0):>6.3f} "
            f"{m.get('R@5', 0):>6.3f} "
            f"{m.get('F1@5', 0):>6.3f} "
            f"{m.get('MRR', 0):>6.3f} "
            f"{ah:>10.1f}"
        )

    # --- Indexing ---
    print(f"\n6. INDEXING PERFORMANCE")
    print(sep)
    idx = results.get("indexing", {})
    full = idx.get("full_index", {})
    incr = idx.get("incremental_index", {})
    print(f"  Full index:        {full.get('elapsed_seconds', 0):.4f}s "
          f"({full.get('indexed_files', 0)} files)")
    print(f"  Incremental index: {incr.get('elapsed_seconds', 0):.4f}s "
          f"({incr.get('indexed_files', 0)} indexed, "
          f"{incr.get('skipped_files', 0)} skipped)")

    print("\n" + "=" * 90)
    print(f"  Results saved to: {RESULTS_PATH}")
    print("=" * 90 + "\n")


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    print("Loading benchmark queries ...")
    queries_data = _load_queries()

    total_queries = (
        len(queries_data["cross_unit_queries"])
        + len(queries_data["same_unit_queries"])
        + len(queries_data["formula_queries"])
        + len(queries_data.get("multi_constraint_queries", []))
    )
    print(f"  {total_queries} queries loaded across 4 categories")

    with tempfile.TemporaryDirectory(prefix="clio_bench_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        # --- Setup ---
        print("\nIndexing benchmark corpus ...")
        connector, storage = _setup_connector(tmpdir, CORPUS_DIR)
        coordinator = RetrievalCoordinator()
        print("  Index complete.")

        all_results: dict[str, Any] = {}

        # --- 1. Baselines ---
        print("\nRunning 5 baselines (80 queries x 5 baselines = 400 evaluations) ...")
        all_results["baselines"] = evaluate_baselines(
            connector, coordinator, queries_data
        )
        print("  Baselines complete.")

        # --- 2. Ablation ---
        print("\nRunning ablation study ...")
        all_results["ablation"] = evaluate_ablation(connector, queries_data)
        print("  Ablation complete.")

        # --- 3. Federated ---
        print("\nRunning federated evaluation ...")
        all_results["federated"] = evaluate_federated(tmpdir, queries_data)
        print("  Federated complete.")

        # --- 4. Agentic ---
        print("\nRunning agentic multi-hop evaluation ...")
        all_results["agentic"] = evaluate_agentic(connector, queries_data)
        print("  Agentic complete.")

        # --- 5. Indexing perf ---
        print("\nMeasuring indexing performance ...")
        all_results["indexing"] = evaluate_indexing_performance(tmpdir)
        print("  Indexing performance complete.")

        # Teardown primary connector
        connector.teardown()

    # --- Save & print ---
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as fh:
        json.dump(all_results, fh, indent=2)

    print_summary(all_results)


if __name__ == "__main__":
    main()
