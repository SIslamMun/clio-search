#!/usr/bin/env python3
"""Build evaluation corpora from external real-world datasets.

Converts three external sources into clio-compatible text documents
with cross-unit queries and ground-truth relevance judgments:

1. NumConQ (NC-Retriever, 2025) — 6,577 numeric constraint queries
2. PANGAEA — 500 geoscience dataset metadata records
3. DOE Data Explorer — 500 DOE scientific dataset descriptions

Usage:
    cd code && python3 benchmarks/build_external_corpora.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = ROOT / "data" / "raw"
BENCH_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 1. NumConQ corpus builder
# ---------------------------------------------------------------------------

NUMCONQ_BASE = Path("/tmp/NumConQ/evaluation/dataset/numcial_constraint_v9")
NUMCONQ_DOMAINS = ["qs_data", "diabetes_data", "imdb_data", "olympic_data", "league_data"]


def build_numconq() -> None:
    """Convert NumConQ benchmark into clio-compatible format."""
    out_dir = BENCH_DIR / "corpus_numconq"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_queries: list[dict[str, Any]] = []
    doc_id = 0
    qid = 0

    for domain in NUMCONQ_DOMAINS:
        domain_dir = NUMCONQ_BASE / domain / "test"
        doc_path = domain_dir / "doc.json"
        query_path = domain_dir / "query.json"

        if not doc_path.exists():
            print(f"  [skip] {domain}: no data")
            continue

        with open(doc_path) as f:
            docs = json.load(f)
        with open(query_path) as f:
            queries = json.load(f)

        print(f"  {domain}: {len(docs)} docs, {len(queries)} queries")

        # Write docs as text files
        doc_filenames: dict[int, str] = {}
        for i, doc_text in enumerate(docs):
            if isinstance(doc_text, dict):
                doc_text = doc_text.get("text", str(doc_text))
            fname = f"numconq_{domain}_{i:04d}.txt"
            (out_dir / fname).write_text(str(doc_text), encoding="utf-8")
            doc_filenames[i] = fname
            doc_id += 1

        # Build doc text -> filename index for matching
        doc_text_to_file: dict[str, str] = {}
        for i, doc_text in enumerate(docs):
            text = str(doc_text).strip()[:200]  # first 200 chars as key
            doc_text_to_file[text] = doc_filenames[i]

        # Convert queries with positive_text matching
        for q in queries:
            query_text = q.get("query", "")
            positive_texts = q.get("positive_text", [])

            # Match positive texts to doc filenames
            relevant_docs = []
            for pt in positive_texts:
                pt_key = str(pt).strip()[:200]
                if pt_key in doc_text_to_file:
                    relevant_docs.append(doc_text_to_file[pt_key])

            if not relevant_docs:
                # Try substring matching
                for pt in positive_texts:
                    pt_str = str(pt).strip()[:100]
                    for txt_key, fname in doc_text_to_file.items():
                        if pt_str in txt_key or txt_key in pt_str:
                            relevant_docs.append(fname)
                            break

            if relevant_docs:
                qid += 1
                all_queries.append({
                    "id": f"numconq_{qid:04d}",
                    "query": query_text,
                    "domain": domain.replace("_data", ""),
                    "type": "numeric_constraint",
                    "relevant_docs": relevant_docs,
                })

    # Save queries
    queries_path = BENCH_DIR / "numconq_queries.json"
    with open(queries_path, "w") as f:
        json.dump({
            "description": "NumConQ numeric constraint benchmark (NC-Retriever, 2025)",
            "source": "https://github.com/Tongji-KGLLM/NumConQ",
            "domains": NUMCONQ_DOMAINS,
            "doc_count": doc_id,
            "query_count": len(all_queries),
            "queries": all_queries,
        }, f, indent=2)

    print(f"  NumConQ total: {doc_id} docs, {len(all_queries)} queries with ground truth")
    print(f"  Saved to {out_dir}/ and {queries_path}")


# ---------------------------------------------------------------------------
# 2. PANGAEA corpus builder
# ---------------------------------------------------------------------------

def build_pangaea() -> None:
    """Convert PANGAEA geoscience metadata into clio text documents."""
    raw_path = DATA_RAW / "pangaea_500.json"
    if not raw_path.exists():
        print("  [skip] PANGAEA: pangaea_500.json not found")
        return

    out_dir = BENCH_DIR / "corpus_pangaea"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(raw_path) as f:
        data = json.load(f)

    hits = data.get("hits", {}).get("hits", [])
    print(f"  PANGAEA: {len(hits)} raw records")

    docs_written = 0
    doc_metadata: list[dict[str, Any]] = []

    for i, hit in enumerate(hits):
        src = hit.get("_source", {})
        xml_thumb = src.get("xml-thumb", "")

        # Extract title
        title_m = re.search(r"<md:title>([^<]+)</md:title>", xml_thumb)
        title = title_m.group(1) if title_m else ""
        if not title:
            continue

        # Extract citation year
        year_m = re.search(r"<md:year>(\d{4})</md:year>", xml_thumb)
        year = year_m.group(1) if year_m else ""

        # Extract parameters with units
        param_matches = re.findall(
            r"<md:matrixColumn[^>]*>.*?<md:name>([^<]+)</md:name>"
            r"(?:.*?<md:unit>([^<]*)</md:unit>)?.*?</md:matrixColumn>",
            xml_thumb, re.DOTALL,
        )
        # Also try simpler pattern
        if not param_matches:
            param_names = re.findall(r"<md:name>([^<]+)</md:name>", xml_thumb)
            param_matches = [(n, "") for n in param_names if n not in ("", title)]

        # Extract geo coverage
        geo = src.get("geoCoverage", "")
        topics = src.get("agg-topic", [])
        lat = src.get("meanPosition", "")
        npoints = src.get("nDataPoints", "")
        uri = src.get("URI", "")

        # Build document text
        lines = [
            f"PANGAEA Dataset Record",
            f"Title: {title}",
        ]
        if year:
            lines.append(f"Year: {year}")
        if uri:
            lines.append(f"DOI: {uri}")
        if geo:
            lines.append(f"Geographic Coverage: {geo}")
        if lat:
            lines.append(f"Position: {lat}")
        if topics:
            lines.append(f"Topics: {', '.join(topics[:5])}")
        if npoints:
            lines.append(f"Data Points: {npoints}")
        if param_matches:
            lines.append("")
            lines.append("Parameters measured:")
            for pname, punit in param_matches[:20]:
                if punit:
                    lines.append(f"  - {pname} ({punit})")
                else:
                    lines.append(f"  - {pname}")

        lines.append("")
        lines.append(
            "Source: PANGAEA Data Publisher for Earth & Environmental Science. "
            "All measurements from peer-reviewed scientific studies."
        )

        fname = f"pangaea_{i:04d}.txt"
        (out_dir / fname).write_text("\n".join(lines), encoding="utf-8")
        docs_written += 1

        # Track metadata for query generation
        units_found = [u for _, u in param_matches if u]
        doc_metadata.append({
            "filename": fname,
            "title": title,
            "units": units_found,
            "topics": topics,
            "params": [p for p, _ in param_matches],
        })

    # Generate cross-unit queries
    queries = _build_pangaea_queries(doc_metadata)

    queries_path = BENCH_DIR / "pangaea_queries.json"
    with open(queries_path, "w") as f:
        json.dump({
            "description": "PANGAEA geoscience cross-unit benchmark",
            "source": "PANGAEA Data Publisher (https://www.pangaea.de)",
            "doc_count": docs_written,
            "query_count": len(queries),
            "queries": queries,
        }, f, indent=2)

    print(f"  PANGAEA: {docs_written} docs, {len(queries)} queries")
    print(f"  Saved to {out_dir}/ and {queries_path}")


def _build_pangaea_queries(metadata: list[dict]) -> list[dict]:
    """Generate cross-unit queries from PANGAEA metadata."""
    queries = []
    qid = 0

    # Find docs with specific unit types
    temp_docs = [m["filename"] for m in metadata
                 if any("°C" in u or "deg C" in u or "°c" in u for u in m["units"])]
    pressure_docs = [m["filename"] for m in metadata
                     if any("hPa" in u or "mbar" in u or "kPa" in u or "Pa" in u for u in m["units"])]
    depth_docs = [m["filename"] for m in metadata
                  if any("m" == u.strip() or "km" in u for u in m["units"])]
    salinity_docs = [m["filename"] for m in metadata
                     if any("PSU" in u or "psu" in u or "‰" in u for u in m["units"])]

    if temp_docs:
        qid += 1
        queries.append({
            "id": f"pangaea_{qid:03d}",
            "query": "geoscience datasets with temperature measurements in degrees Celsius",
            "type": "unit_match_temperature",
            "relevant_docs": temp_docs,
        })

    if pressure_docs:
        qid += 1
        queries.append({
            "id": f"pangaea_{qid:03d}",
            "query": "ocean or atmospheric datasets measuring pressure in hectopascals or millibars",
            "type": "unit_match_pressure",
            "relevant_docs": pressure_docs,
        })

    if depth_docs:
        qid += 1
        queries.append({
            "id": f"pangaea_{qid:03d}",
            "query": "datasets with depth or elevation measurements in meters",
            "type": "unit_match_depth",
            "relevant_docs": depth_docs,
        })

    if salinity_docs:
        qid += 1
        queries.append({
            "id": f"pangaea_{qid:03d}",
            "query": "oceanographic datasets measuring salinity in practical salinity units",
            "type": "unit_match_salinity",
            "relevant_docs": salinity_docs,
        })

    # Cross-unit: temperature in Fahrenheit vs docs in Celsius
    if temp_docs:
        qid += 1
        queries.append({
            "id": f"pangaea_{qid:03d}",
            "query": "datasets where water temperature exceeded 68 degrees Fahrenheit",
            "type": "cross_unit_temperature",
            "unit_in_query": "°F",
            "unit_in_docs": "°C",
            "relevant_docs": temp_docs,
        })

    # Topic-based queries
    climate_docs = [m["filename"] for m in metadata
                    if any("Climate" in t or "Atmosphere" in t for t in m["topics"])]
    ocean_docs = [m["filename"] for m in metadata
                  if any("Ocean" in t or "Marine" in t for t in m["topics"])]

    if climate_docs:
        qid += 1
        queries.append({
            "id": f"pangaea_{qid:03d}",
            "query": "climate and atmospheric science datasets with quantitative measurements",
            "type": "topic_climate",
            "relevant_docs": climate_docs,
        })

    if ocean_docs:
        qid += 1
        queries.append({
            "id": f"pangaea_{qid:03d}",
            "query": "marine and oceanographic measurement datasets",
            "type": "topic_ocean",
            "relevant_docs": ocean_docs,
        })

    return queries


# ---------------------------------------------------------------------------
# 3. DOE Data Explorer corpus builder
# ---------------------------------------------------------------------------

def build_doe() -> None:
    """Convert DOE Data Explorer records into clio text documents."""
    raw_path = DATA_RAW / "doe_500.json"
    if not raw_path.exists():
        print("  [skip] DOE: doe_500.json not found")
        return

    out_dir = BENCH_DIR / "corpus_doe"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(raw_path) as f:
        data = json.load(f)

    records = data if isinstance(data, list) else data.get("records", [])
    print(f"  DOE: {len(records)} raw records")

    docs_written = 0
    doc_metadata: list[dict[str, Any]] = []

    for i, rec in enumerate(records):
        title = rec.get("title", "")
        desc = rec.get("description", "")
        if not title or not desc:
            continue

        authors = rec.get("authors", "")
        subjects = rec.get("subjects", "")
        doi = rec.get("doi", "")
        pub_date = rec.get("publication_date", "")
        orgs = rec.get("research_orgs", "") or rec.get("contributing_org", "")

        lines = [
            "DOE Scientific Dataset Record",
            f"Title: {title}",
        ]
        if authors:
            lines.append(f"Authors: {authors[:200]}")
        if pub_date:
            lines.append(f"Date: {pub_date}")
        if doi:
            lines.append(f"DOI: {doi}")
        if orgs:
            lines.append(f"Organization: {orgs[:200]}")
        if subjects:
            lines.append(f"Subjects: {subjects[:300]}")
        lines.append("")
        lines.append(desc)
        lines.append("")
        lines.append(
            "Source: DOE Data Explorer (OSTI). "
            "U.S. Department of Energy scientific dataset."
        )

        fname = f"doe_{i:04d}.txt"
        (out_dir / fname).write_text("\n".join(lines), encoding="utf-8")
        docs_written += 1

        # Extract measurement indicators from description
        has_pressure = bool(re.search(r"pressure|psi|kpa|mpa|bar|hpa", desc, re.I))
        has_temperature = bool(re.search(r"temperature|°[CF]|deg[CF]|celsius|fahrenheit|kelvin", desc, re.I))
        has_depth = bool(re.search(r"depth|feet|meter|elevation", desc, re.I))
        has_flow = bool(re.search(r"flow|rate|gpm|m3/s|liter", desc, re.I))

        doc_metadata.append({
            "filename": fname,
            "title": title,
            "has_pressure": has_pressure,
            "has_temperature": has_temperature,
            "has_depth": has_depth,
            "has_flow": has_flow,
        })

    # Generate queries
    queries = _build_doe_queries(doc_metadata)

    queries_path = BENCH_DIR / "doe_queries.json"
    with open(queries_path, "w") as f:
        json.dump({
            "description": "DOE Data Explorer scientific dataset benchmark",
            "source": "https://www.osti.gov/dataexplorer",
            "doc_count": docs_written,
            "query_count": len(queries),
            "queries": queries,
        }, f, indent=2)

    print(f"  DOE: {docs_written} docs, {len(queries)} queries")
    print(f"  Saved to {out_dir}/ and {queries_path}")


def _build_doe_queries(metadata: list[dict]) -> list[dict]:
    """Generate queries from DOE dataset metadata."""
    queries = []
    qid = 0

    pressure_docs = [m["filename"] for m in metadata if m["has_pressure"]]
    temp_docs = [m["filename"] for m in metadata if m["has_temperature"]]
    depth_docs = [m["filename"] for m in metadata if m["has_depth"]]
    flow_docs = [m["filename"] for m in metadata if m["has_flow"]]

    if pressure_docs:
        qid += 1
        queries.append({
            "id": f"doe_{qid:03d}",
            "query": "datasets containing pressure measurements or pressure-temperature logs",
            "type": "measurement_pressure",
            "relevant_docs": pressure_docs,
        })

    if temp_docs:
        qid += 1
        queries.append({
            "id": f"doe_{qid:03d}",
            "query": "DOE datasets with temperature data from geothermal or energy experiments",
            "type": "measurement_temperature",
            "relevant_docs": temp_docs,
        })

    if depth_docs:
        qid += 1
        queries.append({
            "id": f"doe_{qid:03d}",
            "query": "datasets recording depth or elevation measurements in wells or boreholes",
            "type": "measurement_depth",
            "relevant_docs": depth_docs,
        })

    if flow_docs:
        qid += 1
        queries.append({
            "id": f"doe_{qid:03d}",
            "query": "flow rate measurement data from energy or geothermal systems",
            "type": "measurement_flow",
            "relevant_docs": flow_docs,
        })

    # Combined queries
    pt_docs = [m["filename"] for m in metadata if m["has_pressure"] and m["has_temperature"]]
    if pt_docs:
        qid += 1
        queries.append({
            "id": f"doe_{qid:03d}",
            "query": "combined pressure and temperature monitoring data from subsurface experiments",
            "type": "combined_pressure_temperature",
            "relevant_docs": pt_docs,
        })

    return queries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Building external evaluation corpora ...\n")

    print("[1/3] NumConQ (NC-Retriever benchmark) ...")
    if NUMCONQ_BASE.exists():
        build_numconq()
    else:
        print("  [skip] NumConQ not found at /tmp/NumConQ")
        print("  Run: git clone https://github.com/Tongji-KGLLM/NumConQ.git /tmp/NumConQ")

    print("\n[2/3] PANGAEA (geoscience metadata) ...")
    build_pangaea()

    print("\n[3/3] DOE Data Explorer (scientific datasets) ...")
    build_doe()

    print("\nDone. Corpora ready for evaluation.")


if __name__ == "__main__":
    main()
