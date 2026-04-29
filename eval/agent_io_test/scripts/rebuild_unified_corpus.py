#!/usr/bin/env python3
"""Rebuild a unified 200-file Argo test corpus so all arms (A naive, B/C CLIO,
D federated) operate on EXACTLY THE SAME data.

Steps:
  1. Pick 200 valid NetCDFs from /home/sislam6/jarvis-work/argo_data/raw
  2. Replace data/ symlinks in arm_a workspace to point to those (real files)
  3. Convert 200 NetCDFs -> blobs.json (text-form measurement blobs)
  4. Build test_index.duckdb (single-shard CLIO index over those 200 blobs)
  5. Compute groundtruth.json (count of profiles with surface temp > 30°C)

Idempotent: rerun safely after any data churn.
"""
import json
import random
import shutil
import sys
from pathlib import Path

random.seed(42)

ARGO_RAW = Path("/home/sislam6/jarvis-work/argo_data/raw")
TEST_BASE = Path("/home/sislam6/clio-search/eval/agent_io_test")
DATA_DIR = TEST_BASE / "workspaces/arm_a/data"
BLOBS_OUT = TEST_BASE / "blobs_unified.json"
INDEX_OUT = TEST_BASE / "test_index_unified.duckdb"
GT_OUT = TEST_BASE / "groundtruth_unified.json"

N_FILES = 200


def step1_pick_files():
    print(f"[1] Scanning {ARGO_RAW} for NetCDFs...", flush=True)
    all_nc = list(ARGO_RAW.rglob("*.nc"))
    print(f"    found {len(all_nc)} total")
    random.shuffle(all_nc)
    picked = all_nc[:N_FILES]
    return picked


def step2_relink_data_dir(picked):
    print(f"[2] Replacing symlinks in {DATA_DIR} ({N_FILES} real NetCDFs)...", flush=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Remove existing symlinks/files
    for f in DATA_DIR.iterdir():
        f.unlink()
    for src in picked:
        (DATA_DIR / src.name).symlink_to(src)
    real = sum(1 for f in DATA_DIR.iterdir() if f.exists())
    print(f"    {real} symlinks point to real files")


def step3_extract_blobs(picked):
    print(f"[3] Extracting measurement-text blobs from {N_FILES} NetCDFs...", flush=True)
    sys.path.insert(0, "/home/sislam6/jarvis-work")
    from argo_to_iowarp_blobs import extract_profile

    blobs = {}
    for nc in picked:
        try:
            prof = extract_profile(nc)
            if prof is None:
                continue
            wmo = prof["wmo"]
            cyc = prof["cycle"]
            uri = f"cte://argo_{wmo}/{wmo}_{cyc:03d}"
            blobs[uri] = prof["text"]
        except Exception as e:
            print(f"    skip {nc.name}: {type(e).__name__}: {e}")
    # Workaround: agent_io_test/ has a Lustre dirent state that rejects fresh writes;
    # write to /jarvis-work first then mv (which works).
    import shutil
    tmp_path = Path("/home/sislam6/jarvis-work") / BLOBS_OUT.name
    tmp_path.write_text(json.dumps(blobs, indent=2))
    if BLOBS_OUT.exists():
        BLOBS_OUT.unlink()
    shutil.move(tmp_path, BLOBS_OUT)
    print(f"    wrote {len(blobs)} blobs to {BLOBS_OUT}")
    return blobs


def step4_build_index(blobs):
    print(f"[4] Building CLIO DuckDB index over {len(blobs)} blobs...", flush=True)
    sys.path.insert(0, "/home/sislam6/clio-search/code/src")
    from clio_agentic_search.connectors.iowarp.connector import IOWarpConnector
    from clio_agentic_search.storage.duckdb_store import DuckDBStorage

    if INDEX_OUT.exists():
        INDEX_OUT.unlink()
    store = DuckDBStorage(INDEX_OUT)
    store.connect()

    # Use index_from_texts (text-only path, no iowarp at index time —
    # purely DuckDB-side indexing)
    conn = IOWarpConnector(
        namespace="agent_test", storage=store,
        tag_pattern="argo_.*", blob_pattern=".*",
    )
    conn.connect()
    # Load blobs into the index via iowarp_core (kClient connecting to running daemon)
    # — but we already showed that hangs. Use the text-only ingest path instead.
    # Fallback: ingest texts directly into DuckDBStorage as documents+chunks+metadata.
    try:
        from clio_agentic_search.indexing.scientific import canonicalize_measurement, build_structure_aware_chunk_plan
        from clio_agentic_search.indexing.text_features import HashEmbedder, tokenize
        from clio_agentic_search.models.contracts import (
            ChunkRecord, DocumentRecord, EmbeddingRecord, MetadataRecord,
        )
        from clio_agentic_search.storage.contracts import DocumentBundle
        import hashlib, time

        embedder = HashEmbedder()
        for uri, text in blobs.items():
            doc_id = hashlib.sha1(uri.encode()).hexdigest()
            doc = DocumentRecord(
                namespace="agent_test", document_id=doc_id, uri=uri,
                checksum=hashlib.sha1(text.encode()).hexdigest(),
                modified_at_ns=int(time.time() * 1e9),
            )
            chunk_id = hashlib.sha1((uri + "_0").encode()).hexdigest()
            chunk = ChunkRecord(
                namespace="agent_test", chunk_id=chunk_id, document_id=doc_id,
                chunk_index=0, text=text, start_offset=0, end_offset=len(text),
            )
            embedding = EmbeddingRecord(
                namespace="agent_test", chunk_id=chunk_id,
                model="hash16-v1",
                vector_json=json.dumps(embedder.embed(tokenize(text))),
            )
            # Scientific metadata: parse measurements from text
            metadata = []
            for m in canonicalize_measurement(text):
                metadata.append(MetadataRecord(
                    namespace="agent_test", record_id=chunk_id,
                    scope="chunk", key=m.field,
                    value=json.dumps({
                        "amount": m.amount,
                        "unit": m.unit,
                        "amount_si": m.amount_si,
                        "unit_si": m.unit_si,
                    }),
                ))
            bundle = DocumentBundle(
                document=doc, chunks=[chunk], embeddings=[embedding], metadata=metadata,
            )
            store.upsert(bundle)
        print(f"    index built")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise

    conn.teardown()


def step5_compute_gt(picked):
    print(f"[5] Computing ground truth (surface temp > 30°C)...", flush=True)
    sys.path.insert(0, "/home/sislam6/jarvis-work")
    from argo_to_iowarp_blobs import extract_profile

    matches = []
    surf_temps = []
    for nc in picked:
        try:
            prof = extract_profile(nc)
            if prof is None: continue
            # The profile dict has 'measurements' or surface temp embedded in 'text'
            # Easier: parse the text for "surface: ... temperature N degC"
            import re
            m = re.search(r"surface:.*?temperature\s+(-?[\d.]+)\s*degC", prof["text"])
            if m:
                t = float(m.group(1))
                surf_temps.append((nc.name, t))
                if t > 30.0:
                    matches.append({"file": nc.name, "uri": prof["uri"], "surface_temp_degC": t})
        except Exception:
            pass
    gt = {
        "question": "How many Argo profile NetCDFs in data/ have a surface temperature > 30 degC?",
        "n_files_scanned": len(picked),
        "n_files_with_surface_data": len(surf_temps),
        "n_match": len(matches),
        "matches": matches,
    }
    import shutil
    tmp_gt = Path("/home/sislam6/jarvis-work") / GT_OUT.name
    tmp_gt.write_text(json.dumps(gt, indent=2))
    if GT_OUT.exists():
        GT_OUT.unlink()
    shutil.move(tmp_gt, GT_OUT)
    print(f"    GT: {len(matches)}/{len(surf_temps)} have surface temp > 30°C")
    print(f"    saved: {GT_OUT}")
    return gt


def main():
    picked = step1_pick_files()
    step2_relink_data_dir(picked)
    blobs = step3_extract_blobs(picked)
    step4_build_index(blobs)
    gt = step5_compute_gt(picked)

    print("\n=== SUMMARY ===")
    print(f"  data/                      : {N_FILES} symlinks -> real Argo NetCDFs")
    print(f"  blobs_unified.json         : {len(blobs)} text-form blobs")
    print(f"  test_index_unified.duckdb  : built")
    print(f"  groundtruth_unified.json   : {gt['n_match']} of {gt['n_files_with_surface_data']} match")


if __name__ == "__main__":
    main()
