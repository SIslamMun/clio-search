#!/usr/bin/env python3
"""D1-D6: Delta distributed experiments driver.

Runs the suite of experiments against a running distributed CLIO cluster
on DeltaAI. Assumes distributed_clio.py is already running:
  * 1 coordinator on the login or first compute node
  * N workers on compute nodes (1, 2, or 4)

Experiments
-----------

  D1: Distributed strong scaling
      Fixed corpus (2.5M arXiv abstracts), vary worker count: 1, 2, 4
      Measures query latency, profile time, aggregate throughput.

  D2: Distributed weak scaling
      Scale data and workers together: 1 worker/625K, 2/1.25M, 4/2.5M
      Measures per-worker work, efficiency ratio.

  D3: Large-scale distributed indexing
      Time to index 2.5M arXiv across 4 workers in parallel.
      Measures per-worker throughput, aggregate throughput, imbalance.

  D4: Cross-unit precision at 2.5M scale
      Same queries as L3 but on the full arXiv corpus distributed.

  D5: NumConQ at scale
      6,500 queries over a distributed index.

  D6: 100-namespace federation distributed
      Each of 100 namespaces holds 25K docs (= 2.5M total) across 4 workers.

Usage
-----
  # Must already have distributed_clio.py running (coordinator + workers)
  export CLIO_COORDINATOR=http://dtai-login:9200
  python3 D1_D6_experiments.py --experiment strong --corpus-size 2500000

  # Or run all experiments:
  python3 D1_D6_experiments.py --experiment all

Output
------
  eval/eval_final/outputs/D1_strong_scaling.json
  eval/eval_final/outputs/D2_weak_scaling.json
  eval/eval_final/outputs/D3_indexing.json
  eval/eval_final/outputs/D4_cross_unit_at_scale.json
  eval/eval_final/outputs/D5_numconq_distributed.json
  eval/eval_final/outputs/D6_federation_distributed.json

Prerequisites
-------------
  * distributed_clio.py running with 1, 2, or 4 workers
  * arXiv corpus already sharded and loaded into worker DBs
  * (optional) NumConQ data downloaded and sharded
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    import aiohttp
except ImportError:
    print("aiohttp required: pip install aiohttp", file=sys.stderr)
    sys.exit(1)

_REPO = Path(__file__).resolve().parents[4]
OUT_DIR = _REPO / "eval" / "eval_final" / "outputs"

COORDINATOR_URL = os.environ.get("CLIO_COORDINATOR", "http://localhost:9200")


async def _post(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    async with aiohttp.ClientSession() as s:
        async with s.post(
            f"{COORDINATOR_URL.rstrip('/')}{endpoint}",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            return await resp.json()


async def _get(endpoint: str) -> dict[str, Any]:
    async with aiohttp.ClientSession() as s:
        async with s.get(
            f"{COORDINATOR_URL.rstrip('/')}{endpoint}",
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            return await resp.json()


# ----------------------------------------------------------------------------
# D1: Strong scaling
# ----------------------------------------------------------------------------


async def d1_strong_scaling(repeats: int = 10) -> dict[str, Any]:
    """Run a fixed query mix against the cluster; report latency percentiles.

    Expect the caller to orchestrate multiple runs: first with 1 worker,
    then 2, then 4. Each invocation writes a separate JSON. The slurm
    script manages this.
    """
    queries = [
        {"query": "temperature above 30 celsius", "min_value": 30, "max_value": 100, "unit": "degC"},
        {"query": "pressure around 101 kPa", "min_value": 95, "max_value": 110, "unit": "kPa"},
        {"query": "wind speed above 50", "min_value": 50, "max_value": 200, "unit": "km/h"},
        {"query": "humidity sensor", "top_k": 10},
        {"query": "solar radiation", "top_k": 10},
    ]
    latencies: list[float] = []
    per_query_results = []
    for q in queries:
        for _ in range(repeats):
            t0 = time.perf_counter()
            r = await _post("/query", {"top_k": 10, **q})
            latencies.append(time.perf_counter() - t0)
            per_query_results.append({
                "query": q,
                "coordinator_elapsed_s": r.get("coordinator_elapsed_s"),
                "fanout_elapsed_s": r.get("fanout_elapsed_s"),
                "workers": r.get("workers_contacted"),
                "hits": len(r.get("results", [])),
            })

    latencies.sort()
    def pct(p: float) -> float:
        idx = int(len(latencies) * p)
        return latencies[min(idx, len(latencies) - 1)]

    return {
        "experiment": "D1: strong scaling single run",
        "coordinator_url": COORDINATOR_URL,
        "query_count": len(queries),
        "repeats": repeats,
        "total_measurements": len(latencies),
        "latency_p50_s": pct(0.5),
        "latency_p95_s": pct(0.95),
        "latency_p99_s": pct(0.99),
        "latency_max_s": latencies[-1],
        "latency_min_s": latencies[0],
        "per_query": per_query_results,
    }


# ----------------------------------------------------------------------------
# D2: Weak scaling
# ----------------------------------------------------------------------------


async def d2_weak_scaling() -> dict[str, Any]:
    """Identical to D1 but the caller is expected to have scaled data and
    workers together. The output here is just the measurements for one
    configuration; multiple runs (1/625K, 2/1.25M, 4/2.5M) are stitched
    together by the slurm batch."""
    result = await d1_strong_scaling(repeats=10)
    result["experiment"] = "D2: weak scaling single run"
    return result


# ----------------------------------------------------------------------------
# D3: Distributed indexing throughput
# ----------------------------------------------------------------------------


async def d3_indexing() -> dict[str, Any]:
    """The indexing itself is done by a separate loading script on each
    worker node, in parallel. This function just queries the profile
    endpoint before and after to record the result."""
    t0 = time.perf_counter()
    profile = await _get("/profile")
    elapsed = time.perf_counter() - t0
    return {
        "experiment": "D3: distributed indexing (profile query)",
        "elapsed_s": elapsed,
        "profile": profile,
    }


# ----------------------------------------------------------------------------
# D4: Cross-unit at scale
# ----------------------------------------------------------------------------


async def d4_cross_unit() -> dict[str, Any]:
    """Issue the same 6 cross-unit queries as L3 but against the distributed
    corpus. Measures hits per unit variant."""
    probes = [
        ("pressure", 100000, "Pa"),
        ("pressure", 100, "kPa"),
        ("pressure", 1, "bar"),
        ("pressure", 0.987, "atm"),
        ("temperature", 30, "degC"),
        ("temperature", 86, "degF"),
        ("temperature", 303.15, "kelvin"),
    ]
    results = []
    for quantity, value, unit in probes:
        delta = abs(value) * 0.02
        t0 = time.perf_counter()
        r = await _post("/query", {
            "query": f"{quantity} around {value} {unit}",
            "min_value": value - delta,
            "max_value": value + delta,
            "unit": unit,
            "top_k": 20,
        })
        results.append({
            "quantity": quantity,
            "value": value,
            "unit": unit,
            "hits": len(r.get("results", [])),
            "total_shards_with_hits": sum(
                1 for res in r.get("results", [])
                if res.get("shard") is not None
            ),
            "latency_s": time.perf_counter() - t0,
        })
    return {
        "experiment": "D4: cross-unit at scale",
        "probes": results,
    }


# ----------------------------------------------------------------------------
# D5 / D6 — stubs (populate queries as needed)
# ----------------------------------------------------------------------------


async def d5_numconq() -> dict[str, Any]:
    """Placeholder. When NumConQ data is sharded and loaded, this iterates
    through all 6,500 queries and reports aggregated recall@10."""
    return {
        "experiment": "D5: NumConQ distributed",
        "status": "placeholder — populate with NumConQ query set on Delta",
    }


async def d6_federation() -> dict[str, Any]:
    """Placeholder. When 100 namespaces × 25K docs are distributed across
    4 workers, this runs the federation routing test."""
    return {
        "experiment": "D6: 100-namespace federation distributed",
        "status": "placeholder — populate with sharded 100-ns setup",
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


async def run_experiment(name: str) -> dict[str, Any]:
    table = {
        "strong": d1_strong_scaling,
        "weak": d2_weak_scaling,
        "indexing": d3_indexing,
        "cross_unit": d4_cross_unit,
        "numconq": d5_numconq,
        "federation": d6_federation,
    }
    fn = table[name]
    return await fn()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=["strong", "weak", "indexing", "cross_unit", "numconq", "federation", "all"],
        required=True,
    )
    parser.add_argument("--output-suffix", default="", help="Append to output filename")
    args = parser.parse_args()

    experiments = (
        ["strong", "weak", "indexing", "cross_unit", "numconq", "federation"]
        if args.experiment == "all"
        else [args.experiment]
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in experiments:
        print(f"\n{'=' * 75}")
        print(f"Running D-{name}")
        print('=' * 75)
        result = asyncio.run(run_experiment(name))
        result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        fn = f"D_{name}{args.output_suffix}.json"
        out_path = OUT_DIR / fn
        with out_path.open("w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Saved: {out_path}")
        print(json.dumps(result, indent=2, default=str)[:1000])


if __name__ == "__main__":
    main()
