#!/usr/bin/env python3
"""Unified evaluation runner for all clio-agentic-search benchmarks.

Runs 5 baselines across all available corpora and saves results.

Usage:
    cd code && python3 benchmarks/evaluate_all.py [--dataset NAMES] [--top-k K]

Options:
    --dataset   Comma-separated list: controlled,noaa,numconq,pangaea,doe,all
                Default: all
    --top-k     Number of results to retrieve. Default: 20

Examples:
    python3 benchmarks/evaluate_all.py                    # Run all
    python3 benchmarks/evaluate_all.py --dataset noaa,doe # Run subset
"""

from __future__ import annotations

import argparse
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
from clio_agentic_search.retrieval.scientific import (
    NumericRangeOperator,
    ScientificQueryOperators,
)
from clio_agentic_search.storage.duckdb_store import DuckDBStorage

BENCHMARK_DIR = Path(__file__).resolve().parent
EVAL_DIR = BENCHMARK_DIR.parent.parent / "eval"

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict[str, Any]] = {
    "controlled": {
        "corpus": BENCHMARK_DIR / "corpus_v2",
        "queries": BENCHMARK_DIR / "queries_v2.json",
        "query_key": None,  # needs special handling (multiple query types)
        "output": EVAL_DIR / "benchmark_v2_results.json",
        "description": "Controlled benchmark (210 docs, 80 queries, 7 domains)",
        "script": "evaluate_v2.py",
    },
    "noaa": {
        "corpus": BENCHMARK_DIR / "corpus_real",
        "queries": BENCHMARK_DIR / "real_queries.json",
        "query_key": "queries",
        "output": EVAL_DIR / "real_benchmark_results.json",
        "description": "NOAA GHCN-Daily (288 docs, 8 stations, cross-unit °F↔°C)",
        "script": "evaluate_real.py",  # needs query-specific SI operators
    },
    "numconq": {
        "corpus": BENCHMARK_DIR / "corpus_numconq",
        "queries": BENCHMARK_DIR / "numconq_queries.json",
        "query_key": "queries",
        "output": EVAL_DIR / "numconq_benchmark_results.json",
        "description": "NumConQ (5362 docs, 6577 queries, 5 domains)",
    },
    "pangaea": {
        "corpus": BENCHMARK_DIR / "corpus_pangaea",
        "queries": BENCHMARK_DIR / "pangaea_queries.json",
        "query_key": "queries",
        "output": EVAL_DIR / "pangaea_benchmark_results.json",
        "description": "PANGAEA geoscience (500 docs, real metadata)",
    },
    "doe": {
        "corpus": BENCHMARK_DIR / "corpus_doe",
        "queries": BENCHMARK_DIR / "doe_queries.json",
        "query_key": "queries",
        "output": EVAL_DIR / "doe_benchmark_results.json",
        "description": "DOE Data Explorer (500 docs, scientific datasets)",
    },
}


# ---------------------------------------------------------------------------
# Metrics (same as evaluate_real.py)
# ---------------------------------------------------------------------------

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    return sum(1 for d in top if d in relevant) / len(top) if top else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    return sum(1 for d in retrieved[:k] if d in relevant) / len(relevant)


def f1_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    p, r = precision_at_k(retrieved, relevant, k), recall_at_k(retrieved, relevant, k)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for i, doc in enumerate(retrieved, 1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    import math
    dcg = sum(
        (1.0 if retrieved[i] in relevant else 0.0) / math.log2(i + 2)
        for i in range(min(k, len(retrieved)))
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal > 0 else 0.0


def compute_metrics(retrieved: list[str], relevant: set[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for k in (1, 5, 10, 20):
        result[f"P@{k}"] = round(precision_at_k(retrieved, relevant, k), 4)
        result[f"R@{k}"] = round(recall_at_k(retrieved, relevant, k), 4)
        result[f"F1@{k}"] = round(f1_at_k(retrieved, relevant, k), 4)
        result[f"nDCG@{k}"] = round(ndcg_at_k(retrieved, relevant, k), 4)
    result["MRR"] = round(mrr(retrieved, relevant), 4)
    return result


def average_metrics(all_metrics: list[dict[str, float]]) -> dict[str, float]:
    if not all_metrics:
        return {}
    return {k: round(sum(m[k] for m in all_metrics) / len(all_metrics), 4) for k in all_metrics[0]}


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def _setup_connector(tmpdir: Path, corpus_src: Path, ns: str) -> FilesystemConnector:
    dest = tmpdir / "corpus"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(corpus_src, dest)
    db = DuckDBStorage(database_path=tmpdir / f"{ns}.duckdb")
    conn = FilesystemConnector(namespace=ns, root=dest, storage=db, warmup_async=False)
    conn.connect()
    conn.index(full_rebuild=True)
    return conn


def _retrieved_doc_paths(citations: list[Any]) -> list[str]:
    paths: list[str] = []
    for c in citations:
        p = Path(c.uri.split("#")[0]).name
        if p not in paths:
            paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Baseline runners
# ---------------------------------------------------------------------------

def _run_bm25(conn: FilesystemConnector, q: str, k: int) -> list[str]:
    return _retrieved_doc_paths([conn.build_citation(r) for r in conn.search_lexical(q, top_k=k)])


def _run_dense(conn: FilesystemConnector, q: str, k: int) -> list[str]:
    return _retrieved_doc_paths([conn.build_citation(r) for r in conn.search_vector(q, top_k=k)])


def _run_hybrid(conn: FilesystemConnector, coord: RetrievalCoordinator, q: str, k: int) -> list[str]:
    return _retrieved_doc_paths(coord.query(connector=conn, query=q, top_k=k,
                                            scientific_operators=ScientificQueryOperators()).citations)


def _run_string_norm(conn: FilesystemConnector, coord: RetrievalCoordinator, q: str, k: int) -> list[str]:
    try:
        from clio_agentic_search.retrieval.query_rewriter import _expand_unit_variants
        extra = _expand_unit_variants(q)
        expanded = f"{q} {' '.join(extra)}" if extra else q
    except Exception:
        expanded = q
    return _retrieved_doc_paths(coord.query(connector=conn, query=expanded, top_k=k,
                                            scientific_operators=ScientificQueryOperators()).citations)


def _run_full(conn: FilesystemConnector, coord: RetrievalCoordinator, q: str, k: int,
              ops: ScientificQueryOperators | None = None) -> list[str]:
    return _retrieved_doc_paths(coord.query(connector=conn, query=q, top_k=k,
                                            scientific_operators=ops or ScientificQueryOperators()).citations)


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------

def evaluate_dataset(
    name: str,
    config: dict[str, Any],
    top_k: int = 20,
    max_queries: int = 500,
) -> dict[str, Any] | None:
    """Run 5 baselines on a dataset."""
    corpus_dir = config["corpus"]
    queries_path = config["queries"]

    if not corpus_dir.exists():
        print(f"  [skip] Corpus not found: {corpus_dir}")
        return None
    if not queries_path.exists():
        print(f"  [skip] Queries not found: {queries_path}")
        return None

    with open(queries_path) as f:
        qdata = json.load(f)

    query_key = config.get("query_key")
    if query_key:
        queries = qdata[query_key]
    else:
        # Controlled benchmark: flatten all query types
        queries = []
        for key in ("cross_unit_queries", "same_unit_queries", "formula_queries", "multi_constraint_queries"):
            queries.extend(qdata.get(key, []))

    doc_count = len(list(corpus_dir.glob("*.txt")))
    print(f"  {config['description']}")
    print(f"  {doc_count} docs, {len(queries)} queries (using up to {min(len(queries), max_queries)})")

    # Limit queries for large datasets
    if len(queries) > max_queries:
        queries = queries[:max_queries]

    with tempfile.TemporaryDirectory(prefix=f"clio_{name}_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        t_idx = time.perf_counter()
        conn = _setup_connector(tmpdir, corpus_dir, name)
        idx_time = time.perf_counter() - t_idx
        print(f"  Indexed in {idx_time:.1f}s")

        coord = RetrievalCoordinator()

        baselines = {b: [] for b in ["bm25_only", "dense_only", "hybrid", "string_norm", "full_pipeline"]}
        per_query: dict[str, list] = {b: [] for b in baselines}

        total = len(queries)
        for qi, q in enumerate(queries):
            if qi % 100 == 0 and qi > 0:
                print(f"    Progress: {qi}/{total}")

            query_text = q.get("query", "")
            relevant = set(Path(rp).name for rp in q.get("relevant_docs", []))
            if not relevant:
                continue

            results = {
                "bm25_only": _run_bm25(conn, query_text, top_k),
                "dense_only": _run_dense(conn, query_text, top_k),
                "hybrid": _run_hybrid(conn, coord, query_text, top_k),
                "string_norm": _run_string_norm(conn, coord, query_text, top_k),
                "full_pipeline": _run_full(conn, coord, query_text, top_k),
            }

            for bname, retrieved in results.items():
                m = compute_metrics(retrieved, relevant)
                baselines[bname].append(m)
                per_query[bname].append({
                    "query_id": q.get("id", f"q_{qi}"),
                    "metrics": m,
                })

        conn.teardown()

    averages = {b: average_metrics(mlist) for b, mlist in baselines.items()}

    return {
        "dataset": name,
        "description": config["description"],
        "doc_count": doc_count,
        "query_count": len(queries),
        "indexing_time_s": round(idx_time, 2),
        "top_k": top_k,
        "averages": averages,
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(all_results: dict[str, Any]) -> None:
    sep = "-" * 90
    print("\n" + "=" * 90)
    print("  CLIO-AGENTIC-SEARCH: MULTI-DATASET EVALUATION SUMMARY")
    print("=" * 90)

    for dname, result in all_results.items():
        if result is None:
            continue
        print(f"\n  {result['description']}")
        print(f"  {result['doc_count']} docs, {result['query_count']} queries, indexed in {result['indexing_time_s']}s")
        print(sep)

        header = f"{'Method':<20} {'P@5':>6} {'R@5':>6} {'nDCG@10':>7} {'MRR':>6}"
        print(header)
        print(sep)

        for bname in ["bm25_only", "dense_only", "hybrid", "string_norm", "full_pipeline"]:
            m = result["averages"].get(bname, {})
            label = {
                "bm25_only": "BM25 Only",
                "dense_only": "Dense",
                "hybrid": "Hybrid",
                "string_norm": "String Norm",
                "full_pipeline": "Clio (ours)",
            }[bname]
            print(f"{label:<20} {m.get('P@5',0):>6.3f} {m.get('R@5',0):>6.3f} "
                  f"{m.get('nDCG@10',0):>7.3f} {m.get('MRR',0):>6.3f}")

    print("\n" + "=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run all clio benchmarks")
    parser.add_argument("--dataset", default="all", help="Comma-separated: controlled,noaa,numconq,pangaea,doe,all")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-queries", type=int, default=500, help="Max queries per dataset")
    args = parser.parse_args()

    if args.dataset == "all":
        selected = list(DATASETS.keys())
    else:
        selected = [s.strip() for s in args.dataset.split(",")]

    all_results: dict[str, Any] = {}

    for dname in selected:
        if dname not in DATASETS:
            print(f"Unknown dataset: {dname}")
            continue
        config = DATASETS[dname]

        # For controlled benchmark, use its own script
        if dname == "controlled" and config.get("script"):
            print(f"\n{'='*60}")
            print(f"  [{dname}] Use dedicated script: python3 benchmarks/{config['script']}")
            print(f"{'='*60}")
            continue

        print(f"\n{'='*60}")
        print(f"  Evaluating: {dname}")
        print(f"{'='*60}")

        result = evaluate_dataset(dname, config, top_k=args.top_k, max_queries=args.max_queries)
        all_results[dname] = result

        # Save individual results
        if result:
            output_path = config["output"]
            EVAL_DIR.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved: {output_path}")

    # Save combined results
    combined_path = EVAL_DIR / "all_benchmark_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print_summary(all_results)
    print(f"\nCombined results: {combined_path}")


if __name__ == "__main__":
    main()
