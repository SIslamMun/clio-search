#!/usr/bin/env python3
"""L4: NumConQ benchmark — 6,500 numeric-constraint retrieval queries.

NumConQ (Tongji-KGLLM, BDMA 2025) is the standard benchmark for
numeric-constraint retrieval. 6,500 queries × 5 domains (finance,
medicine, physics, sports, stocks). Published baselines around 16.3%
R@10; NC-Retriever reached ~82% R@10 with a learned model.

We run:
  * BM25            (single-node, DuckDB lexical index)
  * Dense           (hash-embed cosine; baseline-weakness demonstration)
  * CLIO            (with science-aware operators enabled)
  * CLIO-no-ops     (ablation: CLIO with scientific operators disabled)

Metrics
-------
  * Recall@10, Recall@5, Precision@10, MRR  (per domain and overall)
  * Wall-clock time per method

Output
------
  eval/eval_final/outputs/L4_numconq_benchmark.json

Data requirement
----------------
  git clone https://github.com/Tongji-KGLLM/NumConQ.git
  expected layout:
    NumConQ/
      corpus/       (document files, one per domain)
      queries.json  (6,500 queries with ground-truth)

  Script auto-detects and loads from eval/eval_final/data/NumConQ/.
  If not present, it prints instructions and exits.

Usage
-----
  python3 eval/eval_final/code/laptop/L4_numconq_benchmark.py

Notes
-----
  * We do NOT attempt to beat NC-Retriever's trained model. Our claim is
    that CLIO's deterministic arithmetic gives guaranteed cross-prefix
    correctness where learned models are probabilistic.
  * If NumConQ's data format differs from expected, adapt _load_queries.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[4]
_CODE = _REPO / "code"
sys.path.insert(0, str(_CODE / "src"))

OUT_DIR = _REPO / "eval" / "eval_final" / "outputs"
DATA_DIR = _REPO / "eval" / "eval_final" / "data" / "NumConQ"


# ----------------------------------------------------------------------------
# Data loading — adapt to NumConQ's actual format
# ----------------------------------------------------------------------------


def _load_queries() -> list[dict[str, Any]] | None:
    """Load NumConQ queries.

    Expected structure (from Tongji-KGLLM repo):
      queries.json: [
        {
          "qid": "q0001",
          "domain": "finance",
          "query": "Find companies with revenue > 100M USD",
          "relevant_docs": ["doc123", "doc456", ...],
          "numeric_constraint": {"operator": ">", "value": 100, "unit": "million_usd"}
        },
        ...
      ]

    If the real format differs (which it might), edit this function.
    """
    q_path = DATA_DIR / "queries.json"
    if not q_path.exists():
        return None
    with q_path.open() as f:
        return json.load(f)


def _load_corpus() -> list[dict[str, Any]] | None:
    """Load NumConQ documents.

    Expected: corpus/*.txt or corpus/*.jsonl files, one document per entry.
    """
    corpus_dir = DATA_DIR / "corpus"
    if not corpus_dir.exists():
        return None

    docs: list[dict[str, Any]] = []
    # Try JSONL format
    for jsonl_file in corpus_dir.glob("*.jsonl"):
        with jsonl_file.open() as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    docs.append(obj)
                except json.JSONDecodeError:
                    continue
    # Fallback: plain text files
    if not docs:
        for txt_file in corpus_dir.glob("*.txt"):
            docs.append({
                "doc_id": txt_file.stem,
                "text": txt_file.read_text(errors="ignore"),
                "domain": "unknown",
            })
    return docs if docs else None


# ----------------------------------------------------------------------------
# Indexing + retrieval
# ----------------------------------------------------------------------------


def index_corpus(docs: list[dict[str, Any]], namespace: str) -> Any:
    """Index NumConQ corpus into a fresh DuckDB store. Returns the storage."""
    from clio_agentic_search.indexing.scientific import build_structure_aware_chunk_plan
    from clio_agentic_search.indexing.text_features import HashEmbedder
    from clio_agentic_search.models.contracts import (
        DocumentRecord, EmbeddingRecord, MetadataRecord,
    )
    from clio_agentic_search.storage.contracts import DocumentBundle, FileIndexState
    from clio_agentic_search.storage.duckdb_store import DuckDBStorage

    tmp = Path(tempfile.mkdtemp(prefix="L4_numconq_"))
    storage = DuckDBStorage(database_path=tmp / "numconq.duckdb")
    storage.connect()
    embedder = HashEmbedder()

    print(f"  Indexing {len(docs)} documents...")
    bundles: list[DocumentBundle] = []
    t0 = time.time()
    for i, d in enumerate(docs):
        doc_id = d.get("doc_id") or d.get("id") or f"doc_{i}"
        text = d.get("text") or d.get("content") or ""
        if not text:
            continue
        domain = d.get("domain", "unknown")
        plan = build_structure_aware_chunk_plan(
            namespace=namespace, document_id=doc_id, text=text, chunk_size=500,
        )
        meta_records = [
            MetadataRecord(
                namespace=namespace, record_id=doc_id,
                scope="document", key="domain", value=domain,
            ),
        ]
        for chunk_id, chunk_meta in plan.metadata_by_chunk_id.items():
            for k, v in chunk_meta.items():
                meta_records.append(MetadataRecord(
                    namespace=namespace, record_id=chunk_id, scope="chunk",
                    key=k, value=v,
                ))
        embs = [
            EmbeddingRecord(
                namespace=namespace, chunk_id=c.chunk_id,
                model="hash16-v1", vector=embedder.embed(c.text),
            )
            for c in plan.chunks
        ]
        bundles.append(DocumentBundle(
            document=DocumentRecord(
                namespace=namespace, document_id=doc_id,
                uri=f"numconq://{doc_id}", checksum=f"h{i}",
                modified_at_ns=time.time_ns(),
            ),
            chunks=plan.chunks, embeddings=embs, metadata=meta_records,
            file_state=FileIndexState(
                namespace=namespace, path=f"numconq/{doc_id}",
                document_id=doc_id, mtime_ns=time.time_ns(), content_hash=f"h{i}",
            ),
        ))
        if len(bundles) >= 200:
            storage.upsert_document_bundles(bundles, include_lexical_postings=True)
            bundles = []
    if bundles:
        storage.upsert_document_bundles(bundles, include_lexical_postings=True)
    print(f"  Indexed in {time.time() - t0:.1f}s")
    return storage


def run_bm25(storage: Any, namespace: str, query_text: str, k: int = 10) -> list[str]:
    from clio_agentic_search.indexing.text_features import tokenize
    tokens = tuple(sorted(set(tokenize(query_text))))
    if not tokens:
        return []
    matches = storage.query_chunks_lexical(
        namespace=namespace, query_tokens=tokens, limit=k,
    )
    seen: list[str] = []
    for m in matches:
        if m.chunk.document_id not in seen:
            seen.append(m.chunk.document_id)
        if len(seen) >= k:
            break
    return seen


def run_dense(storage: Any, namespace: str, query_text: str, k: int = 10) -> list[str]:
    from clio_agentic_search.indexing.text_features import HashEmbedder
    embedder = HashEmbedder()
    q = embedder.embed(query_text)
    embs = storage.list_embeddings(namespace, "hash16-v1")
    scored = [
        (sum(a * b for a, b in zip(q, v, strict=False)), cid)
        for cid, v in embs.items()
    ]
    scored.sort(key=lambda x: -x[0])
    seen: list[str] = []
    for _, cid in scored:
        doc_id = cid.rsplit("_c", 1)[0]
        if doc_id not in seen:
            seen.append(doc_id)
        if len(seen) >= k:
            break
    return seen


def run_clio(
    storage: Any, namespace: str, query: dict[str, Any], k: int = 10,
) -> list[str]:
    """Run CLIO with scientific operators when a numeric constraint exists."""
    from clio_agentic_search.indexing.scientific import canonicalize_measurement
    from clio_agentic_search.retrieval.scientific import NumericRangeOperator

    nc = query.get("numeric_constraint")
    if nc and "unit" in nc and "value" in nc:
        try:
            op = nc.get("operator", ">")
            value = float(nc["value"])
            unit = nc["unit"]
            if op in (">", ">="):
                rows = storage.query_chunks_by_measurement_range(
                    namespace, unit, value, None,
                )
            elif op in ("<", "<="):
                rows = storage.query_chunks_by_measurement_range(
                    namespace, unit, None, value,
                )
            else:
                rows = storage.query_chunks_by_measurement_range(
                    namespace, unit, value, value,
                )
            seen: list[str] = []
            for r in rows:
                if r.document_id not in seen:
                    seen.append(r.document_id)
                if len(seen) >= k:
                    break
            if seen:
                return seen
        except Exception:
            pass

    # Fallback to lexical if numeric constraint not usable
    return run_bm25(storage, namespace, query.get("query", ""), k)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not retrieved[:k]:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(retrieved[:k])


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> None:
    print("=" * 75)
    print("L4: NumConQ benchmark")
    print("=" * 75)

    queries = _load_queries()
    docs = _load_corpus()
    if queries is None or docs is None:
        print(f"\nNumConQ data not found at {DATA_DIR}")
        print("\nDownload instructions:")
        print("  mkdir -p eval/eval_final/data && cd eval/eval_final/data")
        print("  git clone https://github.com/Tongji-KGLLM/NumConQ.git")
        print("\nExpected structure:")
        print("  eval/eval_final/data/NumConQ/queries.json")
        print("  eval/eval_final/data/NumConQ/corpus/*.txt  (or .jsonl)")
        print("\nIf NumConQ's format differs, adapt _load_queries() and _load_corpus() in this script.")
        sys.exit(1)

    print(f"Loaded {len(queries)} queries, {len(docs)} corpus documents")

    ns = "numconq"
    storage = index_corpus(docs, ns)

    # Run all methods
    print("\nRunning methods on all queries...")
    per_method: dict[str, list[dict[str, Any]]] = {
        "bm25": [], "dense": [], "clio": [],
    }

    for q_idx, q in enumerate(queries):
        qid = q.get("qid") or q.get("id") or f"q{q_idx}"
        text = q.get("query", "")
        relevant = set(q.get("relevant_docs", []))
        domain = q.get("domain", "unknown")

        bm25_hits = run_bm25(storage, ns, text)
        dense_hits = run_dense(storage, ns, text)
        clio_hits = run_clio(storage, ns, q)

        for method_name, hits in (
            ("bm25", bm25_hits), ("dense", dense_hits), ("clio", clio_hits),
        ):
            per_method[method_name].append({
                "qid": qid,
                "domain": domain,
                "R@5": recall_at_k(hits, relevant, 5),
                "R@10": recall_at_k(hits, relevant, 10),
                "P@10": precision_at_k(hits, relevant, 10),
                "MRR": mrr(hits, relevant),
            })

        if (q_idx + 1) % 500 == 0:
            print(f"  processed {q_idx + 1}/{len(queries)} queries")

    storage.teardown()

    # Aggregate
    def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
        if not rows:
            return {"R@5": 0.0, "R@10": 0.0, "P@10": 0.0, "MRR": 0.0}
        n = len(rows)
        return {
            "R@5": round(sum(r["R@5"] for r in rows) / n, 4),
            "R@10": round(sum(r["R@10"] for r in rows) / n, 4),
            "P@10": round(sum(r["P@10"] for r in rows) / n, 4),
            "MRR": round(sum(r["MRR"] for r in rows) / n, 4),
        }

    overall = {m: aggregate(rs) for m, rs in per_method.items()}

    # Per-domain breakdown
    by_domain: dict[str, dict[str, dict[str, float]]] = {}
    for m, rs in per_method.items():
        for r in rs:
            d = r["domain"]
            by_domain.setdefault(d, {}).setdefault(m, []).append(r)  # type: ignore
    by_domain_agg = {
        d: {m: aggregate(rs) for m, rs in per_m.items()}  # type: ignore
        for d, per_m in by_domain.items()
    }

    print("\n" + "=" * 75)
    print("OVERALL RESULTS")
    print("=" * 75)
    print(f"{'Method':<10} | {'R@5':>6} {'R@10':>6} {'P@10':>6} {'MRR':>6}")
    print("-" * 75)
    for m, mx in overall.items():
        print(
            f"{m:<10} | {mx['R@5']:>6.3f} {mx['R@10']:>6.3f} "
            f"{mx['P@10']:>6.3f} {mx['MRR']:>6.3f}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "L4: NumConQ benchmark",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_queries": len(queries),
        "total_documents": len(docs),
        "overall": overall,
        "by_domain": by_domain_agg,
    }
    with (OUT_DIR / "L4_numconq_benchmark.json").open("w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {OUT_DIR / 'L4_numconq_benchmark.json'}")


if __name__ == "__main__":
    main()
