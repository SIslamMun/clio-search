#!/usr/bin/env python3
"""Statistical significance tests for clio evaluation results.

Implements paired bootstrap resampling (10,000 iterations, p < 0.05)
for comparing retrieval methods. Also runs scalability benchmarks.

Usage:
    cd code && python3 benchmarks/statistical_tests.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE_DIR / "src"))

from clio_agentic_search.connectors.filesystem.connector import FilesystemConnector
from clio_agentic_search.storage.duckdb_store import DuckDBStorage

EVAL_DIR = _CODE_DIR.parent / "eval"
CORPUS_REAL = Path(__file__).resolve().parent / "corpus_real"


# ---------------------------------------------------------------------------
# Bootstrap significance test
# ---------------------------------------------------------------------------

def paired_bootstrap(
    scores_a: list[float],
    scores_b: list[float],
    n_iterations: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap resampling test.

    Returns p-value for H0: method A and B perform equally.
    If p < 0.05, the difference is statistically significant.
    """
    rng = np.random.RandomState(seed)
    n = len(scores_a)
    assert n == len(scores_b), "Score lists must have equal length"

    observed_diff = np.mean(scores_a) - np.mean(scores_b)
    count_ge = 0

    for _ in range(n_iterations):
        indices = rng.randint(0, n, size=n)
        sample_a = [scores_a[i] for i in indices]
        sample_b = [scores_b[i] for i in indices]
        delta = np.mean(sample_a) - np.mean(sample_b)
        if delta >= observed_diff:
            count_ge += 1

    p_value = count_ge / n_iterations

    return {
        "mean_a": round(float(np.mean(scores_a)), 4),
        "mean_b": round(float(np.mean(scores_b)), 4),
        "observed_diff": round(float(observed_diff), 4),
        "p_value": round(float(p_value), 4),
        "significant": p_value < 0.05,
        "n_iterations": n_iterations,
        "n_samples": n,
    }


def run_significance_tests(results_path: Path) -> dict[str, Any]:
    """Run bootstrap tests on per-query results from a benchmark."""
    with open(results_path) as f:
        data = json.load(f)

    per_query = data.get("per_query", {})
    if not per_query:
        print(f"  No per-query data in {results_path}")
        return {}

    # Extract P@5 scores per query for each baseline
    baseline_scores: dict[str, list[float]] = {}
    for bname, queries in per_query.items():
        baseline_scores[bname] = [q["metrics"]["P@5"] for q in queries]

    # Compare full_pipeline against each baseline
    tests: dict[str, Any] = {}
    fp_scores = baseline_scores.get("full_pipeline", [])
    if not fp_scores:
        return {}

    for bname, scores in baseline_scores.items():
        if bname == "full_pipeline":
            continue
        if len(scores) != len(fp_scores):
            continue
        result = paired_bootstrap(fp_scores, scores)
        tests[f"full_pipeline_vs_{bname}"] = result
        sig = "***" if result["significant"] else "n.s."
        print(f"  Full Pipeline vs {bname:15s}: "
              f"Δ={result['observed_diff']:+.4f}, p={result['p_value']:.4f} {sig}")

    return tests


# ---------------------------------------------------------------------------
# Scalability benchmark
# ---------------------------------------------------------------------------

def run_scalability_benchmark() -> dict[str, Any]:
    """Measure indexing time at different corpus sizes."""
    print("\n=== Scalability Benchmark ===")

    if not CORPUS_REAL.exists():
        print("  NOAA corpus not found. Run build_real_corpus.py first.")
        return {}

    all_docs = sorted(CORPUS_REAL.glob("*.txt"))
    total = len(all_docs)
    print(f"  Total NOAA docs available: {total}")

    sizes = [100, 250, 500, 1000, min(1728, total)]
    if total < 1000:
        sizes = [s for s in sizes if s <= total]

    results: list[dict[str, Any]] = []

    for size in sizes:
        docs = all_docs[:size]

        with tempfile.TemporaryDirectory(prefix="clio_scale_") as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            dest = tmpdir / "corpus"
            dest.mkdir()
            for doc in docs:
                shutil.copy2(doc, dest / doc.name)

            db = DuckDBStorage(database_path=tmpdir / "scale.duckdb")
            conn = FilesystemConnector(
                namespace="scale", root=dest, storage=db, warmup_async=False,
            )
            conn.connect()

            t0 = time.perf_counter()
            conn.index(full_rebuild=True)
            elapsed = time.perf_counter() - t0

            throughput = size / elapsed

            conn.teardown()

        results.append({
            "doc_count": size,
            "index_time_s": round(elapsed, 2),
            "throughput_docs_per_s": round(throughput, 2),
        })
        print(f"  {size:5d} docs: {elapsed:.1f}s ({throughput:.1f} docs/s)")

    return {
        "scalability": results,
        "max_docs": total,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Statistical Significance Tests ===\n")

    all_tests: dict[str, Any] = {}

    # Test each available result file
    for name, path in [
        ("controlled_v2", EVAL_DIR / "benchmark_v2_results.json"),
        ("noaa_real", EVAL_DIR / "real_benchmark_results.json"),
        ("doe", EVAL_DIR / "doe_benchmark_results.json"),
        ("numconq", EVAL_DIR / "numconq_benchmark_results.json"),
    ]:
        if path.exists():
            print(f"\n[{name}]")
            tests = run_significance_tests(path)
            if tests:
                all_tests[name] = tests

    # Scalability
    scale_results = run_scalability_benchmark()
    all_tests["scalability"] = scale_results

    # Save
    output = EVAL_DIR / "statistical_tests.json"
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(all_tests, f, indent=2)

    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
