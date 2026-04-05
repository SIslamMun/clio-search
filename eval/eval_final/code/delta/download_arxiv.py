#!/usr/bin/env python3
"""Download and prepare arXiv metadata for Delta experiments.

Sources
-------
  1. Kaggle arXiv dataset (preferred): https://www.kaggle.com/datasets/Cornell-University/arxiv
     Format: JSONL, ~5 GB, ~2.5M papers
     Requires: `kaggle` CLI and a Kaggle API token

  2. Direct arXiv OAI-PMH (fallback, slow): harvests ~100K/day
     Format: XML

This script:
  1. Downloads the Kaggle dump (if `kaggle` is installed) OR falls back
     to fetching a curated subset.
  2. Converts to a normalized JSONL where each line has:
       {"doc_id": "...", "title": "...", "abstract": "...", "categories": [...]}
  3. Shards the output into N files (default 4) for distributed indexing.
     Sharding is hash-based on doc_id so the distribution is deterministic.

Output
------
  $OUTPUT_DIR/arxiv_full.jsonl           — full normalized dump
  $OUTPUT_DIR/arxiv_shard_0.jsonl  ..    — N shards

Usage
-----
  # Full run (requires Kaggle credentials)
  python3 download_arxiv.py --output-dir /scratch/$USER/arxiv --shards 4

  # Skip download, just re-shard an existing file
  python3 download_arxiv.py --output-dir /scratch/$USER/arxiv --reshard-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def shard_for(doc_id: str, total_shards: int) -> int:
    h = hashlib.sha1(doc_id.encode()).digest()
    return int.from_bytes(h[:8], "big") % total_shards


def download_kaggle(output_dir: Path) -> Path | None:
    """Try to download the arXiv dump via Kaggle CLI."""
    if shutil.which("kaggle") is None:
        print("kaggle CLI not found; install with: pip install kaggle")
        print("Then set KAGGLE_USERNAME and KAGGLE_KEY env vars")
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading arXiv metadata via Kaggle CLI...")
    t0 = time.time()
    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", "Cornell-University/arxiv",
                "-p", str(output_dir),
                "--unzip",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Kaggle download failed: {e}")
        return None
    print(f"Downloaded in {time.time() - t0:.0f}s")
    raw = output_dir / "arxiv-metadata-oai-snapshot.json"
    return raw if raw.exists() else None


def normalize_and_shard(
    raw_path: Path, output_dir: Path, shards: int,
) -> dict[str, int]:
    """Read the Kaggle JSONL, normalize each record, write sharded output."""
    shard_files = [
        (output_dir / f"arxiv_shard_{i}.jsonl").open("w")
        for i in range(shards)
    ]
    full_out = (output_dir / "arxiv_full.jsonl").open("w")

    counts = [0] * shards
    total = 0
    t0 = time.time()

    print(f"Normalizing + sharding into {shards} files...")
    with raw_path.open() as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = obj.get("id")
            if not doc_id:
                continue
            abstract = (obj.get("abstract") or "").strip()
            if not abstract:
                continue
            norm = {
                "doc_id": f"arxiv_{doc_id}",
                "title": obj.get("title", "").strip(),
                "abstract": abstract,
                "categories": obj.get("categories", "").split(),
                "authors": obj.get("authors_parsed", [])[:5],
            }
            text = json.dumps(norm, separators=(",", ":"))
            full_out.write(text + "\n")
            s = shard_for(norm["doc_id"], shards)
            shard_files[s].write(text + "\n")
            counts[s] += 1
            total += 1
            if total % 100_000 == 0:
                elapsed = time.time() - t0
                print(f"  {total:,} records ({total / elapsed:,.0f}/s)")

    for fh in shard_files:
        fh.close()
    full_out.close()

    print(f"\nDone. {total:,} total records in {time.time() - t0:.0f}s")
    print("Per-shard counts:")
    for i, c in enumerate(counts):
        print(f"  shard {i}: {c:,}")
    return {"total": total, "per_shard": counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument(
        "--reshard-only",
        action="store_true",
        help="Skip download, re-shard an existing arxiv-metadata-oai-snapshot.json",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    raw: Path | None
    if args.reshard_only:
        raw = output_dir / "arxiv-metadata-oai-snapshot.json"
        if not raw.exists():
            print(f"No existing file at {raw}")
            sys.exit(1)
    else:
        raw = download_kaggle(output_dir)
        if raw is None:
            print("Download failed. Manually download from Kaggle:")
            print("  https://www.kaggle.com/datasets/Cornell-University/arxiv")
            print(f"Place the JSONL at: {output_dir}/arxiv-metadata-oai-snapshot.json")
            print("Then rerun with --reshard-only")
            sys.exit(1)

    stats = normalize_and_shard(raw, output_dir, args.shards)
    with (output_dir / "shard_stats.json").open("w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
