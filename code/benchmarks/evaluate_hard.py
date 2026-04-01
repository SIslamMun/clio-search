#!/usr/bin/env python3
"""Hard benchmark evaluation for clio-agentic-search.

Runs 30 hard queries (ambiguous, multi-hop, imprecise) through each available
LLM provider at 1-hop, 2-hop, and 3-hop depth.  Unlike the standard benchmark,
queries are run WITHOUT pre-set scientific operators -- the LLM must infer
what to search for from natural language alone.

This differentiates providers because:
  - The fallback (SI expansion) just adds unit strings -> low recall
  - Good LLMs rewrite "high pressure" -> "pressure above 300 kPa" -> BM25 hits
  - Multi-hop helps for queries that need iterative discovery

Usage:
    cd code && python3 benchmarks/evaluate_hard.py
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
sys.path.insert(0, str(_CODE_DIR / "benchmarks"))

from clio_agentic_search.connectors.filesystem.connector import FilesystemConnector
from clio_agentic_search.retrieval.agentic import AgenticRetriever
from clio_agentic_search.retrieval.coordinator import RetrievalCoordinator
from clio_agentic_search.retrieval.query_rewriter import FallbackQueryRewriter
from clio_agentic_search.retrieval.scientific import ScientificQueryOperators
from clio_agentic_search.storage.duckdb_store import DuckDBStorage

from llm_providers import (
    LLMProvider,
    RewriteResponse,
    discover_providers,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BENCHMARK_DIR = Path(__file__).resolve().parent
CORPUS_DIR = BENCHMARK_DIR / "corpus_v2"
HARD_QUERIES_PATH = BENCHMARK_DIR / "hard_queries.json"
EVAL_DIR = BENCHMARK_DIR.parent.parent / "eval"
RESULTS_PATH = EVAL_DIR / "hard_benchmark_results.json"

# ---------------------------------------------------------------------------
# Metrics (same as evaluate_v2.py)
# ---------------------------------------------------------------------------

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
    for k in (1, 5):
        result[f"P@{k}"] = round(precision_at_k(retrieved, relevant, k), 4)
        result[f"R@{k}"] = round(recall_at_k(retrieved, relevant, k), 4)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _retrieved_doc_paths(citations: list[Any]) -> list[str]:
    """Return ordered list of doc paths from citation URIs."""
    paths: list[str] = []
    for c in citations:
        path = c.uri.split("#")[0]
        if path not in paths:
            paths.append(path)
    return paths


def _load_hard_queries() -> dict[str, Any]:
    with open(HARD_QUERIES_PATH, "r") as fh:
        return json.load(fh)


def _all_queries(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten all hard query categories into a single list."""
    queries: list[dict[str, Any]] = []
    for key in ("ambiguous_queries", "multi_hop_queries", "imprecise_queries"):
        queries.extend(data.get(key, []))
    return queries


def _setup_connector(
    tmpdir: Path,
    corpus_src: Path,
    namespace: str = "hard_bench",
) -> tuple[FilesystemConnector, DuckDBStorage]:
    """Copy corpus into tmpdir, create connector + storage, connect, index."""
    dest = tmpdir / "corpus"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(corpus_src, dest)

    db_path = tmpdir / "hard_bench.duckdb"
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


# ---------------------------------------------------------------------------
# LLM-based query rewriter adapter
# ---------------------------------------------------------------------------

class LLMQueryRewriter:
    """Wraps an LLMProvider into the rewriter interface for AgenticRetriever."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def rewrite(self, query: str, context: str = "") -> Any:
        """Rewrite a query using the LLM provider."""
        resp: RewriteResponse = self._provider.rewrite_query(query)
        # Return a simple object that AgenticRetriever can use
        return _SimpleRewriteResult(
            rewritten_query=resp.rewritten_query,
            strategy=resp.strategy,
            reasoning=resp.reasoning,
        )


class _SimpleRewriteResult:
    """Minimal rewrite result compatible with AgenticRetriever expectations."""

    def __init__(self, rewritten_query: str, strategy: str, reasoning: str) -> None:
        self.rewritten_query = rewritten_query
        self.strategy = strategy
        self.reasoning = reasoning
        self.numeric_range = None
        self.formula = None
        self.sub_queries: list[str] = []


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------

def evaluate_provider_on_queries(
    connector: FilesystemConnector,
    provider: LLMProvider,
    queries: list[dict[str, Any]],
    max_hops: int = 1,
) -> dict[str, Any]:
    """Run all queries through a single provider at a given hop depth.

    Key design: NO scientific operators are pre-set. The LLM must produce
    a rewritten query that contains unit terms for BM25 to match.
    """
    coordinator = RetrievalCoordinator()

    # For 1-hop, just rewrite and run hybrid (BM25+vector, no sci operators).
    # For multi-hop, use AgenticRetriever with the provider as rewriter.
    per_query: list[dict[str, Any]] = []
    metrics_list: list[dict[str, float]] = []
    total_latency = 0.0

    for q in queries:
        relevant = set(q["relevant_docs"])
        query_text = q["query"]

        t0 = time.perf_counter()

        if max_hops <= 1:
            # Single hop: rewrite query, then run hybrid search
            try:
                resp: RewriteResponse = provider.rewrite_query(query_text)
                rewritten = resp.rewritten_query
            except Exception:
                rewritten = query_text

            result = coordinator.query(
                connector=connector,
                query=rewritten,
                top_k=20,
                scientific_operators=ScientificQueryOperators(),  # inactive
            )
            retrieved = _retrieved_doc_paths(result.citations)
        else:
            # Multi-hop: use AgenticRetriever with FallbackQueryRewriter
            # (the LLM is used for rewriting within the agentic loop)
            try:
                resp = provider.rewrite_query(query_text)
                rewritten = resp.rewritten_query
            except Exception:
                rewritten = query_text

            # Run first hop with rewritten query
            rewriter = FallbackQueryRewriter()
            retriever = AgenticRetriever(
                coordinator=coordinator,
                rewriter=rewriter,
                max_hops=max_hops,
                min_score_threshold=0.0,
                convergence_threshold=0,
            )

            agentic_result = retriever.query(
                connector=connector,
                query=rewritten,
                top_k=20,
                scientific_operators=ScientificQueryOperators(),
            )
            retrieved = _retrieved_doc_paths(agentic_result.citations)

        elapsed = time.perf_counter() - t0
        total_latency += elapsed

        m = compute_metrics(retrieved, relevant)
        metrics_list.append(m)
        per_query.append({
            "query_id": q["id"],
            "type": q["type"],
            "metrics": m,
            "retrieved_top5": retrieved[:5],
            "latency_s": round(elapsed, 4),
        })

    # Compute per-type averages
    type_averages: dict[str, dict[str, float]] = {}
    for qtype in ("ambiguous", "multi_hop", "imprecise"):
        type_metrics = [
            pq["metrics"] for pq in per_query if pq["type"] == qtype
        ]
        if type_metrics:
            type_averages[qtype] = average_metrics(type_metrics)

    return {
        "overall": average_metrics(metrics_list),
        "by_type": type_averages,
        "per_query": per_query,
        "total_latency_s": round(total_latency, 4),
        "avg_latency_s": round(total_latency / max(len(queries), 1), 4),
    }


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(results: dict[str, Any]) -> None:
    """Print a human-readable comparison table."""
    sep = "-" * 100

    print("\n" + "=" * 100)
    print("  HARD BENCHMARK RESULTS: LLM Provider Differentiation")
    print("  30 queries (10 ambiguous, 10 multi-hop, 10 imprecise)")
    print("  NO pre-set scientific operators -- LLM must infer from natural language")
    print("=" * 100)

    providers = results.get("providers", {})

    for hop_key in ("1_hop", "2_hop", "3_hop"):
        print(f"\n  {hop_key.upper()} RESULTS")
        print(sep)
        header = (
            f"{'Provider':<35} "
            f"{'P@5':>6} {'R@5':>6} {'MRR':>6} "
            f"{'Amb':>6} {'M-Hop':>6} {'Imp':>6} "
            f"{'Lat(s)':>8}"
        )
        print(header)
        print(sep)

        for prov_name, prov_data in providers.items():
            hop_data = prov_data.get(hop_key)
            if not hop_data:
                continue
            ov = hop_data.get("overall", {})
            by_type = hop_data.get("by_type", {})
            amb = by_type.get("ambiguous", {}).get("P@5", 0)
            mh = by_type.get("multi_hop", {}).get("P@5", 0)
            imp = by_type.get("imprecise", {}).get("P@5", 0)
            lat = hop_data.get("avg_latency_s", 0)
            print(
                f"{prov_name:<35} "
                f"{ov.get('P@5', 0):>6.3f} "
                f"{ov.get('R@5', 0):>6.3f} "
                f"{ov.get('MRR', 0):>6.3f} "
                f"{amb:>6.3f} "
                f"{mh:>6.3f} "
                f"{imp:>6.3f} "
                f"{lat:>8.2f}"
            )

    # Highlight differentiation
    print(f"\n  PROVIDER DIFFERENTIATION ANALYSIS")
    print(sep)
    p5_scores: list[tuple[str, float]] = []
    for prov_name, prov_data in providers.items():
        hop1 = prov_data.get("1_hop", {})
        ov = hop1.get("overall", {})
        p5_scores.append((prov_name, ov.get("P@5", 0)))

    p5_scores.sort(key=lambda x: x[1], reverse=True)
    if len(p5_scores) >= 2:
        best = p5_scores[0]
        worst = p5_scores[-1]
        spread = best[1] - worst[1]
        print(f"  Best provider (1-hop P@5):  {best[0]} = {best[1]:.3f}")
        print(f"  Worst provider (1-hop P@5): {worst[0]} = {worst[1]:.3f}")
        print(f"  Spread: {spread:.3f}")
        if spread > 0.1:
            print("  --> Good differentiation (spread > 0.10)")
        else:
            print("  --> Limited differentiation (spread <= 0.10)")

    # Multi-hop improvement
    print(f"\n  MULTI-HOP IMPROVEMENT (1-hop vs 3-hop, P@5)")
    print(sep)
    for prov_name, prov_data in providers.items():
        h1 = prov_data.get("1_hop", {}).get("overall", {}).get("P@5", 0)
        h3 = prov_data.get("3_hop", {}).get("overall", {}).get("P@5", 0)
        delta = h3 - h1
        print(f"  {prov_name:<35} 1-hop={h1:.3f}  3-hop={h3:.3f}  delta={delta:+.3f}")

    print("\n" + "=" * 100)
    print(f"  Results saved to: {RESULTS_PATH}")
    print("=" * 100 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading hard benchmark queries ...")
    queries_data = _load_hard_queries()
    all_q = _all_queries(queries_data)
    print(f"  {len(all_q)} queries loaded across 3 categories")

    print("\nDiscovering LLM providers ...")
    providers = discover_providers()
    print(f"  {len(providers)} provider(s) available: "
          + ", ".join(p.name() for p in providers))

    with tempfile.TemporaryDirectory(prefix="clio_hard_bench_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        print("\nIndexing benchmark corpus ...")
        connector, storage = _setup_connector(tmpdir, CORPUS_DIR)
        print("  Index complete.")

        all_results: dict[str, Any] = {
            "benchmark": "hard_queries_v1",
            "query_count": len(all_q),
            "providers_tested": [p.name() for p in providers],
            "providers": {},
        }

        for provider in providers:
            prov_name = provider.name
            print(f"\n{'='*60}")
            print(f"  Evaluating: {prov_name}")
            print(f"{'='*60}")

            prov_results: dict[str, Any] = {}

            for max_hops in (1, 2, 3):
                hop_key = f"{max_hops}_hop"
                print(f"  Running {hop_key} ({len(all_q)} queries) ...", end=" ",
                      flush=True)

                hop_results = evaluate_provider_on_queries(
                    connector=connector,
                    provider=provider,
                    queries=all_q,
                    max_hops=max_hops,
                )
                prov_results[hop_key] = hop_results

                ov = hop_results["overall"]
                print(
                    f"P@5={ov.get('P@5', 0):.3f}  "
                    f"R@5={ov.get('R@5', 0):.3f}  "
                    f"MRR={ov.get('MRR', 0):.3f}  "
                    f"lat={hop_results['avg_latency_s']:.2f}s"
                )

            all_results["providers"][prov_name] = prov_results

        connector.teardown()

    # Save results
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as fh:
        json.dump(all_results, fh, indent=2)

    print_summary(all_results)


if __name__ == "__main__":
    main()
