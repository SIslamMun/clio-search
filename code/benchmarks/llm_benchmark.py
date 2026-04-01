#!/usr/bin/env python3
"""Multi-provider LLM benchmark for clio-agentic-search.

Evaluates query rewriting quality and retrieval performance across every
available LLM provider (Ollama, Gemini, Claude Agent SDK, llama-cpp,
and the no-LLM fallback baseline).

Usage:
    cd code && python3 benchmarks/llm_benchmark.py
    cd code && python3 benchmarks/llm_benchmark.py --gguf /path/to/model.gguf
"""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(_CODE_DIR / "benchmarks"))

from clio_agentic_search.connectors.filesystem.connector import FilesystemConnector
from clio_agentic_search.retrieval.agentic import AgenticRetriever
from clio_agentic_search.retrieval.coordinator import RetrievalCoordinator
from clio_agentic_search.retrieval.query_rewriter import FallbackQueryRewriter
from clio_agentic_search.retrieval.scientific import (
    NumericRangeOperator,
    ScientificQueryOperators,
)
from clio_agentic_search.storage.duckdb_store import DuckDBStorage

from llm_providers import (
    LLMProvider,
    RewriteResponse,
    discover_providers,
)

# ---------------------------------------------------------------------------
# Re-use metrics helpers from evaluate.py
# ---------------------------------------------------------------------------
from evaluate import (
    BENCHMARK_DIR,
    CORPUS_DIR,
    _operators_for_query,
    _retrieved_doc_paths,
    average_metrics,
    compute_metrics,
)

LLM_RESULTS_PATH = BENCHMARK_DIR / "llm_results.json"


# ===================================================================
# Benchmark query selection
# ===================================================================

def _select_benchmark_queries(queries_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Pick 10 representative queries: mix of cross-unit, formula, same-unit."""
    selected: list[dict[str, Any]] = []
    # 5 cross-unit (every other to get variety across domains)
    cross = queries_data["cross_unit_queries"]
    for i in range(0, len(cross), max(1, len(cross) // 5)):
        selected.append(cross[i])
        if len(selected) >= 5:
            break
    # 3 formula
    formulas = queries_data["formula_queries"]
    selected.extend(formulas[:3])
    # 2 same-unit
    same = queries_data["same_unit_queries"]
    selected.extend(same[:2])
    return selected[:10]


# ===================================================================
# Setup helpers
# ===================================================================

def _setup_connector(
    tmpdir: Path,
    corpus_src: Path,
    namespace: str = "llm_bench",
) -> tuple[FilesystemConnector, DuckDBStorage]:
    """Copy corpus into tmpdir, create connector + storage, connect, index."""
    dest = tmpdir / "corpus"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(corpus_src, dest)

    db_path = tmpdir / "llm_bench.duckdb"
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
# 1. Query-rewriting benchmark
# ===================================================================

def benchmark_rewriting(
    providers: list[LLMProvider],
    queries: list[dict[str, Any]],
    connector: FilesystemConnector,
    coordinator: RetrievalCoordinator,
) -> dict[str, Any]:
    """For each provider, rewrite every query and collect LLM metrics."""
    results: dict[str, Any] = {}

    for provider in providers:
        pname = provider.name()
        print(f"\n  Rewriting with {pname} ...")
        query_results: list[dict[str, Any]] = []
        total_latency = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost = 0.0
        error_count = 0

        for q in queries:
            query_text = q["query"]
            operators = _operators_for_query(q)

            # Get first-hop context via the existing pipeline
            try:
                first_hop = coordinator.query(
                    connector=connector,
                    query=query_text,
                    top_k=5,
                    scientific_operators=operators,
                )
                context_snippets = [
                    c.snippet for c in first_hop.citations if c.snippet
                ]
                context = "\n---\n".join(context_snippets) if context_snippets else "(no results)"
            except Exception:
                context = "(retrieval error)"

            resp = provider.rewrite_query(query_text, context)
            m = resp.metrics

            query_results.append({
                "query_id": q["id"],
                "query": query_text,
                "rewritten_query": resp.rewritten_query,
                "strategy": resp.strategy,
                "reasoning": resp.reasoning,
                "latency_s": m.latency_seconds,
                "prompt_tokens": m.prompt_tokens,
                "completion_tokens": m.completion_tokens,
                "cost_usd": m.cost_usd,
                "error": m.error,
            })

            total_latency += m.latency_seconds
            total_prompt_tokens += m.prompt_tokens
            total_completion_tokens += m.completion_tokens
            total_cost += m.cost_usd
            if m.error:
                error_count += 1

        n = len(queries)
        results[pname] = {
            "avg_latency_s": round(total_latency / max(n, 1), 4),
            "avg_prompt_tokens": round(total_prompt_tokens / max(n, 1), 1),
            "avg_completion_tokens": round(total_completion_tokens / max(n, 1), 1),
            "total_cost_usd": round(total_cost, 8),
            "error_count": error_count,
            "queries": query_results,
        }
        print(f"    avg latency={results[pname]['avg_latency_s']:.3f}s, "
              f"errors={error_count}/{n}")

    return results


# ===================================================================
# 2. Retrieval quality benchmark
# ===================================================================

def benchmark_retrieval(
    providers: list[LLMProvider],
    queries: list[dict[str, Any]],
    connector: FilesystemConnector,
) -> dict[str, Any]:
    """Run each provider's rewritten queries through retrieval and measure quality."""
    results: dict[str, Any] = {}
    coordinator = RetrievalCoordinator()

    for provider in providers:
        pname = provider.name()
        print(f"\n  Retrieval with {pname} ...")

        all_metrics: list[dict[str, float]] = []
        type_metrics: dict[str, list[dict[str, float]]] = {
            "cross_unit": [],
            "same_unit": [],
            "formula": [],
        }

        for q in queries:
            query_text = q["query"]
            qtype = q.get("type", "cross_unit")
            relevant = set(q["relevant_docs"])
            operators = _operators_for_query(q)

            # Get first-hop context
            try:
                first_hop = coordinator.query(
                    connector=connector,
                    query=query_text,
                    top_k=5,
                    scientific_operators=operators,
                )
                context_snippets = [
                    c.snippet for c in first_hop.citations if c.snippet
                ]
                context = "\n---\n".join(context_snippets) if context_snippets else "(no results)"
            except Exception:
                context = "(retrieval error)"

            # Rewrite
            resp = provider.rewrite_query(query_text, context)
            rewritten = resp.rewritten_query

            # Retrieve with rewritten query
            try:
                result = coordinator.query(
                    connector=connector,
                    query=rewritten,
                    top_k=20,
                    scientific_operators=operators,
                )
                retrieved = _retrieved_doc_paths(result.citations)
            except Exception:
                retrieved = []

            m = compute_metrics(retrieved, relevant)
            all_metrics.append(m)
            if qtype in type_metrics:
                type_metrics[qtype].append(m)

        per_type_avg: dict[str, dict[str, float]] = {}
        for tname, mlist in type_metrics.items():
            if mlist:
                per_type_avg[tname] = average_metrics(mlist)

        results[pname] = {
            "overall": average_metrics(all_metrics),
            **per_type_avg,
        }
        ov = results[pname]["overall"]
        print(f"    P@5={ov.get('P@5', 0):.3f}  R@5={ov.get('R@5', 0):.3f}  "
              f"MRR={ov.get('MRR', 0):.3f}")

    return results


# ===================================================================
# 3. Multi-hop benchmark
# ===================================================================

def benchmark_multi_hop(
    providers: list[LLMProvider],
    queries: list[dict[str, Any]],
    connector: FilesystemConnector,
) -> dict[str, Any]:
    """Compare 1-hop, 2-hop, 3-hop agentic retrieval per provider."""
    results: dict[str, Any] = {}
    # Use a small subset for speed (multi-hop is expensive with LLMs)
    subset = queries[:5]

    for provider in providers:
        pname = provider.name()
        print(f"\n  Multi-hop with {pname} ...")
        hop_results: dict[str, Any] = {}

        for max_hops in (1, 2, 3):
            rewriter = provider.as_rewriter()
            retriever = AgenticRetriever(
                coordinator=RetrievalCoordinator(),
                rewriter=rewriter,
                max_hops=max_hops,
                min_score_threshold=0.0,
                convergence_threshold=0,
            )

            metrics_list: list[dict[str, float]] = []
            hop_counts: list[int] = []
            latencies: list[float] = []

            for q in subset:
                relevant = set(q["relevant_docs"])
                operators = _operators_for_query(q)

                t0 = time.perf_counter()
                try:
                    agentic_result = retriever.query(
                        connector=connector,
                        query=q["query"],
                        top_k=20,
                        scientific_operators=operators,
                    )
                    elapsed = time.perf_counter() - t0
                    retrieved = _retrieved_doc_paths(agentic_result.citations)
                    metrics_list.append(compute_metrics(retrieved, relevant))
                    hop_counts.append(agentic_result.total_hops)
                except Exception as exc:
                    elapsed = time.perf_counter() - t0
                    metrics_list.append(compute_metrics([], relevant))
                    hop_counts.append(0)

                latencies.append(elapsed)

            avg_m = average_metrics(metrics_list)
            hop_results[f"{max_hops}_hop"] = {
                **avg_m,
                "avg_latency_s": round(
                    sum(latencies) / max(len(latencies), 1), 4
                ),
                "avg_actual_hops": round(
                    sum(hop_counts) / max(len(hop_counts), 1), 2
                ),
            }
            print(f"    {max_hops}-hop: R@5={avg_m.get('R@5', 0):.3f}, "
                  f"avg_latency={hop_results[f'{max_hops}_hop']['avg_latency_s']:.3f}s")

        results[pname] = hop_results

    return results


# ===================================================================
# Comparison table builder
# ===================================================================

def _build_comparison(
    rewriting: dict[str, Any],
    retrieval: dict[str, Any],
    multi_hop: dict[str, Any],
) -> dict[str, Any]:
    """Build a flat comparison table for easy consumption."""
    table: dict[str, Any] = {}
    for pname in rewriting:
        rw = rewriting[pname]
        ret = retrieval.get(pname, {})
        mh = multi_hop.get(pname, {})
        ov = ret.get("overall", {})
        table[pname] = {
            "rewriting_avg_latency_s": rw.get("avg_latency_s", 0),
            "rewriting_total_cost_usd": rw.get("total_cost_usd", 0),
            "rewriting_error_count": rw.get("error_count", 0),
            "retrieval_P@5": ov.get("P@5", 0),
            "retrieval_R@5": ov.get("R@5", 0),
            "retrieval_F1@5": ov.get("F1@5", 0),
            "retrieval_MRR": ov.get("MRR", 0),
            "multi_hop_1_R@5": mh.get("1_hop", {}).get("R@5", 0),
            "multi_hop_2_R@5": mh.get("2_hop", {}).get("R@5", 0),
            "multi_hop_3_R@5": mh.get("3_hop", {}).get("R@5", 0),
        }
    return table


# ===================================================================
# Summary printer
# ===================================================================

def print_summary(all_results: dict[str, Any]) -> None:
    """Print a human-readable comparison table to stdout."""
    sep = "-" * 105
    providers = all_results.get("providers_tested", [])
    comparison = all_results.get("comparison_table", {})

    print("\n" + "=" * 105)
    print("  LLM PROVIDER BENCHMARK RESULTS")
    print("=" * 105)

    # Rewriting summary
    print("\n1. QUERY REWRITING PERFORMANCE")
    print(sep)
    header = (
        f"{'Provider':<30} {'Avg Lat(s)':>10} {'Avg P-Tok':>10} "
        f"{'Avg C-Tok':>10} {'Cost($)':>10} {'Errors':>8}"
    )
    print(header)
    print(sep)
    rewriting = all_results.get("per_provider", {})
    for pname in providers:
        pdata = rewriting.get(pname, {})
        rw = pdata.get("rewriting", {})
        print(
            f"{pname:<30} "
            f"{rw.get('avg_latency_s', 0):>10.3f} "
            f"{rw.get('avg_prompt_tokens', 0):>10.1f} "
            f"{rw.get('avg_completion_tokens', 0):>10.1f} "
            f"{rw.get('total_cost_usd', 0):>10.6f} "
            f"{rw.get('error_count', 0):>8d}"
        )

    # Retrieval quality
    print(f"\n2. RETRIEVAL QUALITY (rewritten queries)")
    print(sep)
    header = (
        f"{'Provider':<30} {'P@5':>8} {'R@5':>8} {'F1@5':>8} {'MRR':>8}"
    )
    print(header)
    print(sep)
    for pname in providers:
        pdata = rewriting.get(pname, {})
        ret = pdata.get("retrieval", {})
        ov = ret.get("overall", {})
        print(
            f"{pname:<30} "
            f"{ov.get('P@5', 0):>8.4f} "
            f"{ov.get('R@5', 0):>8.4f} "
            f"{ov.get('F1@5', 0):>8.4f} "
            f"{ov.get('MRR', 0):>8.4f}"
        )

    # Multi-hop
    print(f"\n3. MULTI-HOP AGENTIC RETRIEVAL (R@5)")
    print(sep)
    header = (
        f"{'Provider':<30} {'1-hop R@5':>10} {'2-hop R@5':>10} {'3-hop R@5':>10} "
        f"{'1-hop Lat':>10} {'3-hop Lat':>10}"
    )
    print(header)
    print(sep)
    for pname in providers:
        pdata = rewriting.get(pname, {})
        mh = pdata.get("multi_hop", {})
        h1 = mh.get("1_hop", {})
        h2 = mh.get("2_hop", {})
        h3 = mh.get("3_hop", {})
        print(
            f"{pname:<30} "
            f"{h1.get('R@5', 0):>10.4f} "
            f"{h2.get('R@5', 0):>10.4f} "
            f"{h3.get('R@5', 0):>10.4f} "
            f"{h1.get('avg_latency_s', 0):>10.3f} "
            f"{h3.get('avg_latency_s', 0):>10.3f}"
        )

    print("\n" + "=" * 105)
    print(f"  Results saved to: {LLM_RESULTS_PATH}")
    print("=" * 105 + "\n")


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-provider LLM benchmark for clio-agentic-search"
    )
    parser.add_argument(
        "--gguf",
        type=str,
        default=None,
        help="Path to a GGUF model file for llama-cpp-python",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  clio-agentic-search: Multi-Provider LLM Benchmark")
    print("=" * 70)

    # ---- Load queries ----
    print("\nLoading benchmark queries ...")
    queries_path = BENCHMARK_DIR / "queries.json"
    with open(queries_path, "r") as fh:
        queries_data = json.load(fh)
    all_bench_queries = _select_benchmark_queries(queries_data)
    print(f"  Selected {len(all_bench_queries)} representative queries")

    # ---- Discover providers ----
    print("\nDiscovering LLM providers ...")
    providers = discover_providers(gguf_model_path=args.gguf)
    provider_names = [p.name() for p in providers]
    print(f"\n  {len(providers)} provider(s) available: {', '.join(provider_names)}")

    if not providers:
        print("\nNo providers available. Exiting.")
        sys.exit(1)

    # ---- Setup corpus ----
    print("\nIndexing benchmark corpus ...")
    tmpdir_obj = tempfile.TemporaryDirectory(prefix="clio_llm_bench_")
    tmpdir = Path(tmpdir_obj.name)
    connector, storage = _setup_connector(tmpdir, CORPUS_DIR)
    coordinator = RetrievalCoordinator()
    print("  Index complete.")

    try:
        # ---- 1. Rewriting benchmark ----
        print("\n" + "-" * 70)
        print("  Phase 1: Query Rewriting Benchmark")
        print("-" * 70)
        rewriting_results = benchmark_rewriting(
            providers, all_bench_queries, connector, coordinator
        )

        # ---- 2. Retrieval quality benchmark ----
        print("\n" + "-" * 70)
        print("  Phase 2: Retrieval Quality Benchmark")
        print("-" * 70)
        retrieval_results = benchmark_retrieval(
            providers, all_bench_queries, connector
        )

        # ---- 3. Multi-hop benchmark ----
        print("\n" + "-" * 70)
        print("  Phase 3: Multi-Hop Agentic Benchmark")
        print("-" * 70)
        multi_hop_results = benchmark_multi_hop(
            providers, all_bench_queries, connector
        )

        # ---- Assemble results ----
        comparison = _build_comparison(
            rewriting_results, retrieval_results, multi_hop_results
        )

        per_provider: dict[str, Any] = {}
        for pname in provider_names:
            per_provider[pname] = {
                "available": True,
                "rewriting": rewriting_results.get(pname, {}),
                "retrieval": retrieval_results.get(pname, {}),
                "multi_hop": multi_hop_results.get(pname, {}),
            }

        all_results: dict[str, Any] = {
            "providers_tested": provider_names,
            "benchmark_queries_count": len(all_bench_queries),
            "per_provider": per_provider,
            "comparison_table": comparison,
        }

        # ---- Save ----
        with open(LLM_RESULTS_PATH, "w") as fh:
            json.dump(all_results, fh, indent=2)

        # ---- Print summary ----
        print_summary(all_results)

    finally:
        connector.teardown()
        tmpdir_obj.cleanup()


if __name__ == "__main__":
    main()
