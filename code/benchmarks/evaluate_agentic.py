#!/usr/bin/env python3
"""Agentic search evaluation benchmark for CLIO Search.

Evaluates the agentic capabilities added beyond basic retrieval:
  1. Branch selection efficiency — does the agent skip unnecessary branches?
  2. Corpus profiling impact — does profiling improve efficiency without hurting quality?
  3. Namespace routing accuracy — does the agent prioritize the right namespace?
  4. Ablation study (A→E) — marginal contribution of each agentic component.
  5. Token efficiency — LLM cost tracking.

Usage:
    cd code
    python3 benchmarks/evaluate_agentic.py
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
# Ensure the package is importable when run from the code/ directory.
# ---------------------------------------------------------------------------
_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE_DIR / "src"))

from clio_agentic_search.connectors.filesystem.connector import FilesystemConnector
from clio_agentic_search.retrieval.agentic import AgenticRetriever
from clio_agentic_search.retrieval.coordinator import RetrievalCoordinator
from clio_agentic_search.retrieval.corpus_profile import CorpusProfile, build_corpus_profile
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
RESULTS_PATH = EVAL_DIR / "agentic_benchmark_results.json"


# ===================================================================
# Metrics (shared with evaluate_v2)
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
    result["MRR"] = round(mrr(retrieved, relevant), 4)
    return result


def average_metrics(all_metrics: list[dict[str, float]]) -> dict[str, float]:
    if not all_metrics:
        return {}
    keys = all_metrics[0].keys()
    return {key: round(sum(m[key] for m in all_metrics) / len(all_metrics), 4) for key in keys}


# ===================================================================
# Helpers
# ===================================================================

def _load_queries() -> dict[str, Any]:
    with open(QUERIES_PATH) as fh:
        return json.load(fh)


def _retrieved_doc_paths(citations: list[Any]) -> list[str]:
    paths: list[str] = []
    for c in citations:
        path = c.uri.split("#")[0]
        if path not in paths:
            paths.append(path)
    return paths


def _parse_numeric_range(spec: str) -> NumericRangeOperator:
    parts = spec.split(":")
    return NumericRangeOperator(
        minimum=float(parts[0]),
        maximum=float(parts[1]),
        unit=parts[2],
    )


def _operators_for_query(q: dict[str, Any]) -> ScientificQueryOperators:
    nr = None
    formula = None
    if "numeric_range" in q and q["numeric_range"]:
        nr = _parse_numeric_range(q["numeric_range"])
    elif "constraints" in q and q["constraints"]:
        nr = _parse_numeric_range(q["constraints"][0]["range"])
    if "formula" in q and q["formula"]:
        formula = q["formula"]
    return ScientificQueryOperators(numeric_range=nr, formula=formula)


def _setup_connector(
    tmpdir: Path,
    corpus_src: Path,
    namespace: str = "bench",
) -> tuple[FilesystemConnector, DuckDBStorage]:
    dest = tmpdir / f"corpus_{namespace}"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(corpus_src, dest)

    db_path = tmpdir / f"{namespace}.duckdb"
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


# ===================================================================
# Section 1: Corpus Profiling
# ===================================================================

def evaluate_corpus_profiling(
    connector: FilesystemConnector,
    storage: DuckDBStorage,
) -> dict[str, Any]:
    """Profile the corpus and report statistics."""
    print("  [1/5] Corpus profiling...")
    t0 = time.time()
    profile = build_corpus_profile(storage, connector.namespace)
    elapsed = time.time() - t0

    return {
        "namespace": profile.namespace,
        "document_count": profile.document_count,
        "chunk_count": profile.chunk_count,
        "measurement_count": profile.measurement_count,
        "formula_count": profile.formula_count,
        "distinct_units": list(profile.distinct_units),
        "distinct_formulas": list(profile.distinct_formulas),
        "metadata_density": round(profile.metadata_density, 4),
        "embedding_count": profile.embedding_count,
        "lexical_posting_count": profile.lexical_posting_count,
        "has_measurements": profile.has_measurements,
        "has_formulas": profile.has_formulas,
        "has_embeddings": profile.has_embeddings,
        "has_lexical": profile.has_lexical,
        "profile_elapsed_seconds": round(elapsed, 4),
    }


# ===================================================================
# Section 2: Branch Selection Efficiency
# ===================================================================

def evaluate_branch_selection(
    connector: FilesystemConnector,
    coordinator: RetrievalCoordinator,
    queries_data: dict[str, Any],
) -> dict[str, Any]:
    """Measure branch activation patterns and efficiency."""
    print("  [2/5] Branch selection efficiency...")
    all_queries = (
        queries_data["cross_unit_queries"]
        + queries_data["same_unit_queries"]
        + queries_data["formula_queries"]
        + queries_data.get("multi_constraint_queries", [])
    )

    branch_stats: list[dict[str, Any]] = []
    metrics_with_profiling: list[dict[str, float]] = []
    metrics_without_profiling: list[dict[str, float]] = []

    for q in all_queries:
        relevant = set(q["relevant_docs"])
        query_text = q["query"]
        operators = _operators_for_query(q)

        # With profiling (new system)
        result_with = coordinator.query(
            connector=connector,
            query=query_text,
            top_k=20,
            scientific_operators=operators,
        )
        retrieved_with = _retrieved_doc_paths(result_with.citations)
        metrics_with_profiling.append(compute_metrics(retrieved_with, relevant))

        # Count activated branches from trace
        branches_activated = []
        branches_produced = []
        for event in result_with.trace:
            if event.stage == "branch_plan_selected":
                for branch in ("lexical", "vector", "graph", "scientific"):
                    if event.attributes.get(f"use_{branch}") == "True":
                        branches_activated.append(branch)

            if event.stage.endswith("_completed") and event.stage != "merge_completed":
                branch_name = event.stage.replace("_completed", "")
                if branch_name in ("lexical", "vector", "graph", "scientific"):
                    candidates = int(event.attributes.get("candidates", "0"))
                    if candidates > 0:
                        branches_produced.append(branch_name)

        branch_stats.append({
            "query_id": q["id"],
            "type": q["type"],
            "branches_activated": branches_activated,
            "branches_produced_results": branches_produced,
            "activated_count": len(branches_activated),
            "produced_count": len(branches_produced),
        })

    # Compute aggregate branch efficiency
    total_activated = sum(s["activated_count"] for s in branch_stats)
    total_produced = sum(s["produced_count"] for s in branch_stats)
    branch_efficiency = total_produced / total_activated if total_activated > 0 else 0.0

    # Count how often each branch was activated
    branch_activation_counts: dict[str, int] = {}
    for s in branch_stats:
        for b in s["branches_activated"]:
            branch_activation_counts[b] = branch_activation_counts.get(b, 0) + 1

    return {
        "overall_metrics": average_metrics(metrics_with_profiling),
        "branch_efficiency": round(branch_efficiency, 4),
        "total_branches_activated": total_activated,
        "total_branches_produced_results": total_produced,
        "branch_activation_counts": branch_activation_counts,
        "per_type_stats": _branch_stats_by_type(branch_stats),
    }


def _branch_stats_by_type(
    branch_stats: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = {}
    for s in branch_stats:
        by_type.setdefault(s["type"], []).append(s)

    result: dict[str, dict[str, Any]] = {}
    for qtype, stats in sorted(by_type.items()):
        total_act = sum(s["activated_count"] for s in stats)
        total_prod = sum(s["produced_count"] for s in stats)
        result[qtype] = {
            "query_count": len(stats),
            "avg_branches_activated": round(total_act / len(stats), 2),
            "avg_branches_produced": round(total_prod / len(stats), 2),
            "branch_efficiency": round(total_prod / total_act, 4) if total_act > 0 else 0.0,
        }
    return result


# ===================================================================
# Section 3: Agentic Multi-Hop with Strategy
# ===================================================================

def evaluate_agentic_strategy(
    connector: FilesystemConnector,
    queries_data: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate metadata-adaptive strategy selection and hop efficiency."""
    print("  [3/5] Agentic strategy evaluation...")
    cross_queries = queries_data["cross_unit_queries"][:10]  # subset for speed

    retriever = AgenticRetriever(
        coordinator=RetrievalCoordinator(),
        rewriter=FallbackQueryRewriter(),
        max_hops=3,
    )

    results: list[dict[str, Any]] = []
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
        metrics = compute_metrics(retrieved, relevant)

        results.append({
            "query_id": q["id"],
            "strategy_used": agentic_result.strategy_used,
            "total_hops": agentic_result.total_hops,
            "token_usage": {
                "input_tokens": agentic_result.token_usage.total_input_tokens,
                "output_tokens": agentic_result.token_usage.total_output_tokens,
                "llm_calls": agentic_result.token_usage.llm_calls,
            },
            "metrics": metrics,
            "hop_details": [
                {
                    "hop": h.hop_number,
                    "strategy": h.strategy,
                    "citations_found": h.citations_found,
                    "new_citations": h.new_citations,
                }
                for h in agentic_result.hops
            ],
        })

    avg_hops = sum(r["total_hops"] for r in results) / len(results) if results else 0
    strategy_dist: dict[str, int] = {}
    for r in results:
        s = r["strategy_used"]
        strategy_dist[s] = strategy_dist.get(s, 0) + 1

    return {
        "query_count": len(results),
        "avg_hops": round(avg_hops, 2),
        "strategy_distribution": strategy_dist,
        "overall_metrics": average_metrics([r["metrics"] for r in results]),
        "per_query": results,
    }


# ===================================================================
# Section 4: Namespace Routing
# ===================================================================

def evaluate_namespace_routing(
    tmpdir: Path,
    queries_data: dict[str, Any],
) -> dict[str, Any]:
    """Test 4-namespace routing: does the agent pick the right namespace first?"""
    print("  [4/5] Namespace routing evaluation...")
    # Split corpus into 4 domain-specific namespaces
    ns_config = {
        "ns_pressure": ["fluid_dynamics"],
        "ns_temperature": ["atmospheric_science"],
        "ns_formulas": ["chemistry", "hpc_simulation"],
        "ns_mixed": ["materials_science", "mixed", "negatives"],
    }

    connectors: list[FilesystemConnector] = []
    storages: list[DuckDBStorage] = []

    for ns_name, domains in ns_config.items():
        ns_dir = tmpdir / "routing" / ns_name
        ns_dir.mkdir(parents=True, exist_ok=True)
        for domain in domains:
            src = CORPUS_DIR / domain
            if src.exists():
                shutil.copytree(src, ns_dir / domain, dirs_exist_ok=True)

        db_path = tmpdir / "routing" / f"{ns_name}.duckdb"
        storage = DuckDBStorage(database_path=db_path)
        conn = FilesystemConnector(
            namespace=ns_name,
            root=ns_dir,
            storage=storage,
            warmup_async=False,
        )
        conn.connect()
        conn.index(full_rebuild=True)
        connectors.append(conn)
        storages.append(storage)

    # Profile each namespace
    profiles: dict[str, dict[str, Any]] = {}
    for conn, storage in zip(connectors, storages):
        p = build_corpus_profile(storage, conn.namespace)
        profiles[conn.namespace] = {
            "document_count": p.document_count,
            "measurement_count": p.measurement_count,
            "formula_count": p.formula_count,
            "metadata_density": round(p.metadata_density, 4),
        }

    # Test routing with agentic retriever
    retriever = AgenticRetriever(
        coordinator=RetrievalCoordinator(),
        rewriter=FallbackQueryRewriter(),
        max_hops=2,
    )

    all_queries = (
        queries_data["cross_unit_queries"][:10]
        + queries_data["formula_queries"][:5]
    )

    routing_results: list[dict[str, Any]] = []
    for q in all_queries:
        relevant = set(q["relevant_docs"])
        operators = _operators_for_query(q)

        result = retriever.query_namespaces(
            connectors=connectors,
            query=q["query"],
            top_k=20,
            scientific_operators=operators,
        )

        retrieved = _retrieved_doc_paths(result.citations)
        metrics = compute_metrics(retrieved, relevant)

        # Extract routing order from trace
        routing_order = ""
        for event in result.trace:
            if event.stage == "agentic_multi_started":
                routing_order = event.attributes.get("routing_order", "")
                break

        # Determine which namespace has the most relevant docs
        ns_relevance: dict[str, int] = {}
        for doc in relevant:
            for ns_name, domains in ns_config.items():
                if any(doc.startswith(d) for d in domains):
                    ns_relevance[ns_name] = ns_relevance.get(ns_name, 0) + 1

        best_ns = max(ns_relevance, key=ns_relevance.get) if ns_relevance else ""
        first_routed = routing_order.split(",")[0] if routing_order else ""
        routing_correct = first_routed == best_ns

        routing_results.append({
            "query_id": q["id"],
            "type": q["type"],
            "routing_order": routing_order,
            "best_namespace": best_ns,
            "first_routed": first_routed,
            "routing_correct": routing_correct,
            "metrics": metrics,
        })

    correct_count = sum(1 for r in routing_results if r["routing_correct"])
    routing_accuracy = correct_count / len(routing_results) if routing_results else 0.0

    # Cleanup
    for conn in connectors:
        conn.teardown()
    for storage in storages:
        storage.teardown()

    return {
        "namespace_profiles": profiles,
        "routing_accuracy": round(routing_accuracy, 4),
        "correct_count": correct_count,
        "total_queries": len(routing_results),
        "overall_metrics": average_metrics([r["metrics"] for r in routing_results]),
        "per_query": routing_results,
    }


# ===================================================================
# Section 5: Ablation Study (Agentic Components)
# ===================================================================

def evaluate_agentic_ablation(
    connector: FilesystemConnector,
    queries_data: dict[str, Any],
) -> dict[str, Any]:
    """Ablation: measure marginal contribution of each agentic component.

    A: BM25 only (no intelligence)
    B: Hybrid (lexical + vector, no scientific)
    C: Full pipeline (hybrid + scientific operators)
    D: Agentic 1-hop (full pipeline + corpus profiling + branch selection)
    E: Agentic multi-hop (full agentic loop with strategy)
    """
    print("  [5/5] Agentic ablation study...")
    all_queries = (
        queries_data["cross_unit_queries"]
        + queries_data["same_unit_queries"]
        + queries_data["formula_queries"]
        + queries_data.get("multi_constraint_queries", [])
    )

    coordinator = RetrievalCoordinator()
    retriever = AgenticRetriever(
        coordinator=coordinator,
        rewriter=FallbackQueryRewriter(),
        max_hops=3,
    )

    configs = ["A_bm25_only", "B_hybrid", "C_full_pipeline", "D_agentic_1hop", "E_agentic_multi"]
    results: dict[str, dict[str, Any]] = {}

    for config in configs:
        metrics_list: list[dict[str, float]] = []
        total_branches = 0
        total_hops = 0

        for q in all_queries:
            relevant = set(q["relevant_docs"])
            query_text = q["query"]
            operators = _operators_for_query(q)

            if config == "A_bm25_only":
                lexical_results = connector.search_lexical(query_text, top_k=20)
                citations = [connector.build_citation(r) for r in lexical_results]
                retrieved = _retrieved_doc_paths(citations)
                total_branches += 1
                total_hops += 1

            elif config == "B_hybrid":
                result = coordinator.query(
                    connector=connector,
                    query=query_text,
                    top_k=20,
                    scientific_operators=ScientificQueryOperators(),
                )
                retrieved = _retrieved_doc_paths(result.citations)
                total_branches += 2  # lexical + vector
                total_hops += 1

            elif config == "C_full_pipeline":
                result = coordinator.query(
                    connector=connector,
                    query=query_text,
                    top_k=20,
                    scientific_operators=operators,
                )
                retrieved = _retrieved_doc_paths(result.citations)
                # Count branches from trace
                for event in result.trace:
                    if event.stage == "branch_plan_selected":
                        for b in ("lexical", "vector", "graph", "scientific"):
                            if event.attributes.get(f"use_{b}") == "True":
                                total_branches += 1
                total_hops += 1

            elif config == "D_agentic_1hop":
                one_hop = AgenticRetriever(
                    coordinator=coordinator,
                    rewriter=FallbackQueryRewriter(),
                    max_hops=1,
                )
                agentic_result = one_hop.query(
                    connector=connector,
                    query=query_text,
                    top_k=20,
                    scientific_operators=operators,
                )
                retrieved = _retrieved_doc_paths(agentic_result.citations)
                total_hops += agentic_result.total_hops

            elif config == "E_agentic_multi":
                agentic_result = retriever.query(
                    connector=connector,
                    query=query_text,
                    top_k=20,
                    scientific_operators=operators,
                )
                retrieved = _retrieved_doc_paths(agentic_result.citations)
                total_hops += agentic_result.total_hops
            else:
                retrieved = []

            metrics_list.append(compute_metrics(retrieved, relevant))

        n = len(all_queries)
        results[config] = {
            "overall": average_metrics(metrics_list),
            "avg_branches_per_query": round(total_branches / n, 2) if n > 0 else 0,
            "avg_hops_per_query": round(total_hops / n, 2) if n > 0 else 0,
        }

    return results


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    print("=" * 70)
    print("CLIO Search — Agentic Evaluation Benchmark")
    print("=" * 70)

    queries_data = _load_queries()
    total_queries = (
        len(queries_data["cross_unit_queries"])
        + len(queries_data["same_unit_queries"])
        + len(queries_data["formula_queries"])
        + len(queries_data.get("multi_constraint_queries", []))
    )
    print(f"Loaded {total_queries} queries from {QUERIES_PATH.name}")

    with tempfile.TemporaryDirectory(prefix="clio_agentic_eval_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        # Setup main connector
        print("\nIndexing corpus...")
        t0 = time.time()
        connector, storage = _setup_connector(tmpdir, CORPUS_DIR)
        index_time = time.time() - t0
        print(f"  Indexed in {index_time:.1f}s")

        all_results: dict[str, Any] = {
            "metadata": {
                "corpus": str(CORPUS_DIR),
                "total_queries": total_queries,
                "index_time_seconds": round(index_time, 2),
            },
        }

        # Section 1: Corpus profiling
        all_results["corpus_profile"] = evaluate_corpus_profiling(connector, storage)

        # Section 2: Branch selection
        coordinator = RetrievalCoordinator()
        all_results["branch_selection"] = evaluate_branch_selection(
            connector, coordinator, queries_data
        )

        # Section 3: Agentic strategy
        all_results["agentic_strategy"] = evaluate_agentic_strategy(
            connector, queries_data
        )

        # Section 4: Namespace routing
        all_results["namespace_routing"] = evaluate_namespace_routing(
            tmpdir, queries_data
        )

        # Section 5: Ablation
        all_results["agentic_ablation"] = evaluate_agentic_ablation(
            connector, queries_data
        )

        # Cleanup
        connector.teardown()
        storage.teardown()

    # Write results
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nResults written to {RESULTS_PATH}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    profile = all_results["corpus_profile"]
    print(f"\nCorpus: {profile['document_count']} docs, "
          f"{profile['chunk_count']} chunks, "
          f"{profile['measurement_count']} measurements, "
          f"{profile['formula_count']} formulas")
    print(f"Metadata density: {profile['metadata_density']:.1%}")
    print(f"Profile time: {profile['profile_elapsed_seconds']:.4f}s")

    branch = all_results["branch_selection"]
    print(f"\nBranch efficiency: {branch['branch_efficiency']:.1%} "
          f"({branch['total_branches_produced_results']}/{branch['total_branches_activated']})")
    print(f"Branch selection P@5: {branch['overall_metrics'].get('P@5', 'n/a')}")

    strategy = all_results["agentic_strategy"]
    print(f"\nAgentic avg hops: {strategy['avg_hops']}")
    print(f"Strategy distribution: {strategy['strategy_distribution']}")
    print(f"Agentic P@5: {strategy['overall_metrics'].get('P@5', 'n/a')}")

    routing = all_results["namespace_routing"]
    print(f"\nNamespace routing accuracy: {routing['routing_accuracy']:.1%} "
          f"({routing['correct_count']}/{routing['total_queries']})")
    print(f"Routed P@5: {routing['overall_metrics'].get('P@5', 'n/a')}")

    ablation = all_results["agentic_ablation"]
    print("\nAblation Study:")
    for config, data in ablation.items():
        p5 = data["overall"].get("P@5", "n/a")
        hops = data["avg_hops_per_query"]
        print(f"  {config}: P@5={p5}, avg_hops={hops}")

    print(f"\nFull results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
