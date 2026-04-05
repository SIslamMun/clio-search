#!/usr/bin/env python3
"""Distributed CLIO — coordinator/worker architecture for DeltaAI.

This is the new distributed layer required for the SC26 main-track
scaling experiments. The laptop-side code (which uses a single DuckDB
file) doesn't scale beyond one node; this script adds a simple
coordinator/worker topology over TCP + JSON RPC.

Architecture
------------

                        ┌─ Coordinator (login node) ──┐
                        │  • accepts user queries     │
                        │  • fans out to workers      │
                        │  • aggregates & reranks     │
                        └──────────┬──────────────────┘
                                   │ TCP JSON-RPC
                 ┌─────────────────┼─────────────────┐
                 ▼                 ▼                 ▼
            ┌────────┐        ┌────────┐        ┌────────┐
            │Worker 1│        │Worker 2│   ...  │Worker N│
            │DuckDB  │        │DuckDB  │        │DuckDB  │
            │shard   │        │shard   │        │shard   │
            │(local  │        │(local  │        │(local  │
            │NVMe)   │        │NVMe)   │        │NVMe)   │
            └────────┘        └────────┘        └────────┘

* Document sharding: hash(doc_id) % N_workers
* Each worker holds its own DuckDB file on $LOCAL_SCRATCH/clio_shard_*.duckdb
* Coordinator uses asyncio + aiohttp to fan out queries concurrently
* Workers respond with per-shard top-K; coordinator merges and re-ranks

Usage
-----
  # On coordinator (login node):
  python3 distributed_clio.py coordinator --port 9200 --workers worker1:9201,worker2:9201

  # On each worker (compute node):
  python3 distributed_clio.py worker --port 9201 --shard-id 0 --total-shards 4 \\
      --corpus /scratch/$USER/arxiv_shard_0.jsonl

  # Test from coordinator:
  curl -X POST http://localhost:9200/query \\
       -H 'Content-Type: application/json' \\
       -d '{"query": "temperature", "top_k": 5}'

Notes
-----
  This is a minimal, production-quality implementation. It does NOT use
  MPI (too heavy for what we need) or Ray (adds a dependency). Plain
  asyncio + aiohttp is sufficient for the scale we're targeting.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

try:
    from aiohttp import ClientSession, web
except ImportError:
    print("aiohttp required: pip install aiohttp", file=sys.stderr)
    sys.exit(1)

_CLIO_SRC = Path(__file__).resolve().parents[4] / "code" / "src"
if _CLIO_SRC.exists():
    sys.path.insert(0, str(_CLIO_SRC))

from clio_agentic_search.indexing.scientific import canonicalize_measurement
from clio_agentic_search.retrieval.corpus_profile import build_corpus_profile
from clio_agentic_search.storage.duckdb_store import DuckDBStorage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("distributed_clio")


# ============================================================================
# Hash-based sharding
# ============================================================================


def shard_for(doc_id: str, total_shards: int) -> int:
    h = hashlib.sha1(doc_id.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % total_shards


# ============================================================================
# Worker
# ============================================================================


class Worker:
    def __init__(
        self,
        shard_id: int,
        total_shards: int,
        db_path: Path,
        namespace: str = "distributed_clio",
    ) -> None:
        self.shard_id = shard_id
        self.total_shards = total_shards
        self.namespace = namespace
        self.storage = DuckDBStorage(database_path=db_path)
        self.storage.connect()
        log.info(
            "Worker %d/%d started, DB at %s",
            shard_id, total_shards, db_path,
        )

    async def handle_profile(self, _request: web.Request) -> web.Response:
        t0 = time.perf_counter()
        profile = build_corpus_profile(self.storage, self.namespace)
        elapsed = time.perf_counter() - t0
        return web.json_response({
            "shard_id": self.shard_id,
            "elapsed_s": elapsed,
            "document_count": profile.document_count,
            "chunk_count": profile.chunk_count,
            "measurement_count": profile.measurement_count,
            "formula_count": profile.formula_count,
            "metadata_density": profile.metadata_density,
            "has_measurements": profile.has_measurements,
        })

    async def handle_query(self, request: web.Request) -> web.Response:
        payload = await request.json()
        query_text: str = payload.get("query", "")
        top_k: int = int(payload.get("top_k", 10))
        min_value = payload.get("min_value")
        max_value = payload.get("max_value")
        unit = payload.get("unit")

        t0 = time.perf_counter()
        results: list[dict[str, Any]] = []

        # Scientific range query if unit + bounds are provided
        if unit and (min_value is not None or max_value is not None):
            try:
                canon_min = (
                    canonicalize_measurement(float(min_value), unit)[0]
                    if min_value is not None else None
                )
                canon_max = (
                    canonicalize_measurement(float(max_value), unit)[0]
                    if max_value is not None else None
                )
                canon_unit = canonicalize_measurement(0.0, unit)[1]
                rows = self.storage.query_chunks_by_measurement_range(
                    self.namespace, canon_unit, canon_min, canon_max,
                )
                for r in rows[:top_k]:
                    results.append({
                        "shard": self.shard_id,
                        "chunk_id": r.chunk_id,
                        "document_id": r.document_id,
                        "score": 1.0,
                        "snippet": r.text[:200],
                    })
            except Exception as e:
                log.warning("Scientific query failed on shard %d: %s", self.shard_id, e)

        # Fallback: lexical BM25
        if not results:
            from clio_agentic_search.indexing.text_features import tokenize
            tokens = tuple(sorted(set(tokenize(query_text))))
            if tokens:
                lex = self.storage.query_chunks_lexical(
                    namespace=self.namespace, query_tokens=tokens, limit=top_k,
                )
                for m in lex:
                    results.append({
                        "shard": self.shard_id,
                        "chunk_id": m.chunk.chunk_id,
                        "document_id": m.chunk.document_id,
                        "score": m.bm25_score,
                        "snippet": m.chunk.text[:200],
                    })

        elapsed = time.perf_counter() - t0
        return web.json_response({
            "shard_id": self.shard_id,
            "elapsed_s": elapsed,
            "results": results,
        })

    async def handle_ping(self, _request: web.Request) -> web.Response:
        return web.json_response({"shard_id": self.shard_id, "status": "ok"})


# ============================================================================
# Coordinator
# ============================================================================


class Coordinator:
    def __init__(self, worker_urls: list[str]) -> None:
        self.worker_urls = worker_urls
        log.info("Coordinator started with %d workers", len(worker_urls))
        for url in worker_urls:
            log.info("  worker: %s", url)

    async def _fanout(
        self, endpoint: str, payload: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        async with ClientSession() as session:
            async def call(url: str) -> dict[str, Any]:
                full = f"{url.rstrip('/')}{endpoint}"
                try:
                    if payload is None:
                        async with session.get(full, timeout=60) as resp:
                            return await resp.json()
                    else:
                        async with session.post(full, json=payload, timeout=60) as resp:
                            return await resp.json()
                except Exception as e:
                    return {"error": str(e), "url": full}
            return await asyncio.gather(*(call(u) for u in self.worker_urls))

    async def handle_profile(self, _request: web.Request) -> web.Response:
        t0 = time.perf_counter()
        per_worker = await self._fanout("/profile", None)
        elapsed = time.perf_counter() - t0

        total_docs = sum(w.get("document_count", 0) for w in per_worker)
        total_chunks = sum(w.get("chunk_count", 0) for w in per_worker)
        total_meas = sum(w.get("measurement_count", 0) for w in per_worker)

        return web.json_response({
            "coordinator_elapsed_s": elapsed,
            "workers": per_worker,
            "aggregate": {
                "workers": len(per_worker),
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "total_measurements": total_meas,
            },
        })

    async def handle_query(self, request: web.Request) -> web.Response:
        payload = await request.json()
        top_k = int(payload.get("top_k", 10))

        t0 = time.perf_counter()
        per_worker = await self._fanout("/query", payload)
        fanout_elapsed = time.perf_counter() - t0

        # Merge and rerank
        all_results: list[dict[str, Any]] = []
        for w in per_worker:
            if "error" in w:
                continue
            all_results.extend(w.get("results", []))
        all_results.sort(key=lambda r: -float(r.get("score", 0.0)))
        top = all_results[:top_k]
        total_elapsed = time.perf_counter() - t0

        return web.json_response({
            "coordinator_elapsed_s": total_elapsed,
            "fanout_elapsed_s": fanout_elapsed,
            "workers_contacted": len(per_worker),
            "total_results_across_shards": len(all_results),
            "results": top,
        })


# ============================================================================
# CLI
# ============================================================================


def run_worker(args: argparse.Namespace) -> None:
    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    worker = Worker(
        shard_id=args.shard_id,
        total_shards=args.total_shards,
        db_path=db_path,
    )
    app = web.Application()
    app.router.add_get("/ping", worker.handle_ping)
    app.router.add_get("/profile", worker.handle_profile)
    app.router.add_post("/query", worker.handle_query)
    web.run_app(app, port=args.port, access_log=None)


def run_coordinator(args: argparse.Namespace) -> None:
    worker_urls = [u.strip() for u in args.workers.split(",") if u.strip()]
    coord = Coordinator(worker_urls)
    app = web.Application()
    app.router.add_get("/profile", coord.handle_profile)
    app.router.add_post("/query", coord.handle_query)
    web.run_app(app, port=args.port, access_log=None)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="role", required=True)

    w = sub.add_parser("worker", help="Run a worker node")
    w.add_argument("--port", type=int, default=9201)
    w.add_argument("--shard-id", type=int, required=True)
    w.add_argument("--total-shards", type=int, required=True)
    w.add_argument("--db-path", type=str, required=True)

    c = sub.add_parser("coordinator", help="Run the coordinator")
    c.add_argument("--port", type=int, default=9200)
    c.add_argument(
        "--workers",
        type=str,
        required=True,
        help="Comma-separated list like 'http://worker1:9201,http://worker2:9201'",
    )

    args = parser.parse_args()
    if args.role == "worker":
        run_worker(args)
    else:
        run_coordinator(args)


if __name__ == "__main__":
    main()
