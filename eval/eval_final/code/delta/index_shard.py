#!/usr/bin/env python3
"""Build a per-worker DuckDB index from an arXiv shard JSONL.

Called by the slurm scripts before launching distributed_clio workers.
Each compute node runs this once against its assigned shard file to
produce a DuckDB file on local scratch. The distributed_clio worker
then mounts that file.

Usage
-----
  python3 index_shard.py \\
      --shard-jsonl /scratch/$USER/arxiv/arxiv_shard_0.jsonl \\
      --db-path /scratch/$USER/clio_shard_0.duckdb \\
      --namespace distributed_clio

The script uses CLIO's full structure-aware chunking, so the indexed
database contains scientific_measurements, metadata, and lexical_postings
tables ready for the distributed query layer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

_CLIO_SRC = Path(__file__).resolve().parents[4] / "code" / "src"
if _CLIO_SRC.exists():
    sys.path.insert(0, str(_CLIO_SRC))

from clio_agentic_search.indexing.scientific import build_structure_aware_chunk_plan
from clio_agentic_search.models.contracts import (
    DocumentRecord,
    EmbeddingRecord,
    MetadataRecord,
)
from clio_agentic_search.storage.contracts import DocumentBundle, FileIndexState
from clio_agentic_search.storage.duckdb_store import DuckDBStorage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-jsonl", required=True)
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--namespace", default="distributed_clio")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0, help="Cap docs for quick tests")
    args = parser.parse_args()

    shard_path = Path(args.shard_jsonl)
    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Indexing shard: {shard_path}")
    print(f"Target DB:      {db_path}")
    print(f"Namespace:      {args.namespace}")

    storage = DuckDBStorage(database_path=db_path)
    storage.connect()

    t0 = time.time()
    batch: list[DocumentBundle] = []
    total = 0

    with shard_path.open() as f:
        for line in f:
            if args.limit and total >= args.limit:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            doc_id = rec.get("doc_id")
            title = rec.get("title", "")
            abstract = rec.get("abstract", "")
            if not doc_id or not abstract:
                continue

            text = f"# {title}\n\n{abstract}"
            plan = build_structure_aware_chunk_plan(
                namespace=args.namespace,
                document_id=doc_id,
                text=text,
                chunk_size=400,
            )
            meta_records: list[MetadataRecord] = [
                MetadataRecord(
                    namespace=args.namespace, record_id=doc_id,
                    scope="document", key="title", value=title[:200],
                ),
                MetadataRecord(
                    namespace=args.namespace, record_id=doc_id,
                    scope="document", key="source", value="arxiv",
                ),
            ]
            if rec.get("categories"):
                meta_records.append(MetadataRecord(
                    namespace=args.namespace, record_id=doc_id,
                    scope="document", key="categories",
                    value=",".join(rec["categories"][:5]),
                ))
            for chunk_id, chunk_meta in plan.metadata_by_chunk_id.items():
                for k, v in chunk_meta.items():
                    meta_records.append(MetadataRecord(
                        namespace=args.namespace, record_id=chunk_id, scope="chunk",
                        key=k, value=v,
                    ))

            doc = DocumentRecord(
                namespace=args.namespace,
                document_id=doc_id,
                uri=f"arxiv://{doc_id}",
                checksum=hashlib.sha256(text.encode()).hexdigest()[:16],
                modified_at_ns=time.time_ns(),
            )
            file_state = FileIndexState(
                namespace=args.namespace,
                path=f"arxiv/{doc_id}",
                document_id=doc_id,
                mtime_ns=time.time_ns(),
                content_hash="h",
            )
            batch.append(DocumentBundle(
                document=doc, chunks=plan.chunks, embeddings=[],
                metadata=meta_records, file_state=file_state,
            ))
            total += 1

            if len(batch) >= args.batch_size:
                storage.upsert_document_bundles(batch, include_lexical_postings=True)
                batch = []
                if total % 5000 == 0:
                    elapsed = time.time() - t0
                    print(f"  {total:,} docs ({total / elapsed:,.0f}/s)")

    if batch:
        storage.upsert_document_bundles(batch, include_lexical_postings=True)

    elapsed = time.time() - t0
    print(f"\nIndexed {total:,} docs in {elapsed:.1f}s ({total / elapsed:,.0f} docs/s)")
    print(f"DB size: {db_path.stat().st_size / (1024**2):.1f} MB")
    storage.teardown()


if __name__ == "__main__":
    main()
