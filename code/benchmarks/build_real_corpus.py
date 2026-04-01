#!/usr/bin/env python3
"""Build real-world corpus from NOAA GHCN-Daily station data.

Downloads and parses .dly fixed-width files from NOAA's GHCN-Daily dataset.
Each station-month pair becomes one text document. Documents contain raw
measurements in metric units (tenths-of-C, tenths-of-mm) plus natural
language descriptions. Cross-unit queries then test whether SI conversion
operators can bridge the gap between query units and document units.

Station metadata (name, location) is embedded in the document so BM25 can
match on geographic terms as well as measurements.

Usage:
    cd code && python3 benchmarks/build_real_corpus.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Station metadata (ID -> human-readable name and location)
# ---------------------------------------------------------------------------

STATION_META: dict[str, dict[str, str]] = {
    "USW00014732": {
        "name": "Boston Logan International Airport",
        "city": "Boston",
        "state": "Massachusetts",
        "country": "USA",
        "lat": "42.37°N",
        "lon": "70.99°W",
        "elev_m": "6",
    },
    "USW00094728": {
        "name": "Central Park Weather Station",
        "city": "New York City",
        "state": "New York",
        "country": "USA",
        "lat": "40.78°N",
        "lon": "73.97°W",
        "elev_m": "42",
    },
    "USW00023174": {
        "name": "Los Angeles International Airport",
        "city": "Los Angeles",
        "state": "California",
        "country": "USA",
        "lat": "33.94°N",
        "lon": "118.40°W",
        "elev_m": "30",
    },
    "USW00012960": {
        "name": "Miami International Airport",
        "city": "Miami",
        "state": "Florida",
        "country": "USA",
        "lat": "25.82°N",
        "lon": "80.28°W",
        "elev_m": "4",
    },
    "USW00013723": {
        "name": "Hartsfield-Jackson Atlanta Airport",
        "city": "Atlanta",
        "state": "Georgia",
        "country": "USA",
        "lat": "33.63°N",
        "lon": "84.44°W",
        "elev_m": "313",
    },
    "USW00025501": {
        "name": "Anchorage International Airport",
        "city": "Anchorage",
        "state": "Alaska",
        "country": "USA",
        "lat": "61.17°N",
        "lon": "150.02°W",
        "elev_m": "40",
    },
    "USC00050848": {
        "name": "Boulder Weather Observing Station",
        "city": "Boulder",
        "state": "Colorado",
        "country": "USA",
        "lat": "40.01°N",
        "lon": "105.25°W",
        "elev_m": "1671",
    },
    "USW00024157": {
        "name": "Portland International Airport",
        "city": "Portland",
        "state": "Oregon",
        "country": "USA",
        "lat": "45.60°N",
        "lon": "122.60°W",
        "elev_m": "12",
    },
}

# ---------------------------------------------------------------------------
# DLY parser
# ---------------------------------------------------------------------------

_ELEMENT_SCALE: dict[str, float] = {
    "TMAX": 0.1,   # tenths of degC -> degC
    "TMIN": 0.1,
    "TAVG": 0.1,
    "PRCP": 0.1,   # tenths of mm -> mm
    "SNOW": 1.0,   # mm
    "SNWD": 1.0,   # mm
    "AWND": 0.1,   # tenths of m/s -> m/s
    "WSF2": 0.1,
    "WSF5": 0.1,
}

MISSING = -9999


def parse_dly(path: Path, year: int) -> dict[str, dict[str, list[float | None]]]:
    """Parse a GHCN .dly file for a specific year.

    Returns: {month_str: {element: [day1, day2, ..., day31]}}
    """
    months: dict[str, dict[str, list[float | None]]] = {}
    with open(path, "r") as fh:
        for line in fh:
            if len(line) < 21:
                continue
            line_year = int(line[11:15])
            if line_year != year:
                continue
            month = int(line[15:17])
            element = line[17:21].strip()
            if element not in _ELEMENT_SCALE:
                continue
            scale = _ELEMENT_SCALE[element]
            values: list[float | None] = []
            for day in range(31):
                offset = 21 + day * 8
                raw_val = int(line[offset: offset + 5])
                qflag = line[offset + 6] if offset + 6 < len(line) else " "
                if raw_val == MISSING or qflag != " ":
                    values.append(None)
                else:
                    values.append(round(raw_val * scale, 2))
            month_key = f"{year}-{month:02d}"
            if month_key not in months:
                months[month_key] = {}
            months[month_key][element] = values
    return months


# ---------------------------------------------------------------------------
# Monthly summary stats
# ---------------------------------------------------------------------------

def _stats(values: list[float | None]) -> dict[str, float] | None:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return {
        "mean": round(sum(valid) / len(valid), 2),
        "max": round(max(valid), 2),
        "min": round(min(valid), 2),
        "sum": round(sum(valid), 2),
        "count": len(valid),
    }


def monthly_summary(
    month_data: dict[str, list[float | None]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for elem, vals in month_data.items():
        s = _stats(vals)
        if s is not None:
            result[elem] = s
    return result


# ---------------------------------------------------------------------------
# Conversions for cross-unit variety in documents
# ---------------------------------------------------------------------------

def c_to_f(c: float) -> float:
    return round(c * 9 / 5 + 32, 1)


def mm_to_inches(mm: float) -> float:
    return round(mm / 25.4, 2)


def mps_to_mph(mps: float) -> float:
    return round(mps * 2.237, 1)


_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


# ---------------------------------------------------------------------------
# Document generator
# ---------------------------------------------------------------------------

def make_document(
    station_id: str,
    meta: dict[str, str],
    month_key: str,
    summary: dict[str, Any],
) -> str:
    """Format a monthly summary into a natural-language text document.

    IMPORTANT: Documents use ONLY metric/SI units (°C, mm, m/s) — NO imperial
    conversions are embedded.  This makes cross-unit queries non-trivially hard:
    BM25 cannot match "86°F" against "30.0 °C" without unit conversion.
    """
    year, mon = month_key.split("-")
    month_name = _MONTH_NAMES[int(mon) - 1]

    lines: list[str] = []
    lines.append("NOAA GHCN-Daily Weather Station Report")
    lines.append(f"Station: {station_id} | {meta['name']}")
    lines.append(
        f"Location: {meta['city']}, {meta['state']}, {meta['country']} "
        f"({meta['lat']}, {meta['lon']}, elevation {meta['elev_m']} m)"
    )
    lines.append(f"Period: {month_name} {year} (monthly aggregated observations)")
    lines.append("")

    if "TMAX" in summary:
        s = summary["TMAX"]
        lines.append(
            f"Maximum Temperature: avg {s['mean']} degC, "
            f"peak {s['max']} degC, "
            f"lowest-max {s['min']} degC"
        )
    if "TMIN" in summary:
        s = summary["TMIN"]
        lines.append(
            f"Minimum Temperature: avg {s['mean']} degC, "
            f"warmest-min {s['max']} degC, "
            f"coldest-min {s['min']} degC"
        )
    if "TAVG" in summary:
        s = summary["TAVG"]
        lines.append(f"Average Temperature: {s['mean']} degC")
    if "PRCP" in summary:
        s = summary["PRCP"]
        total_mm = s["sum"]
        lines.append(
            f"Precipitation: total {total_mm:.1f} mm, "
            f"daily avg {s['mean']:.2f} mm, "
            f"daily max {s['max']:.1f} mm"
        )
    if "SNOW" in summary:
        s = summary["SNOW"]
        total_mm = s["sum"]
        lines.append(
            f"Snowfall: total {total_mm:.0f} mm, "
            f"daily max {s['max']:.0f} mm"
        )
    if "SNWD" in summary:
        s = summary["SNWD"]
        lines.append(f"Snow Depth: max observed {s['max']:.0f} mm")
    if "AWND" in summary:
        s = summary["AWND"]
        lines.append(
            f"Average Wind Speed: {s['mean']:.2f} m/s, "
            f"maximum {s['max']:.2f} m/s"
        )
    if "WSF2" in summary:
        s = summary["WSF2"]
        lines.append(f"2-min Fastest Wind: max {s['max']:.2f} m/s")
    if "WSF5" in summary:
        s = summary["WSF5"]
        lines.append(f"5-sec Fastest Wind Gust: max {s['max']:.2f} m/s")

    lines.append("")
    lines.append(
        "Data source: NOAA Global Historical Climatology Network - Daily (GHCN-D). "
        "All measurements from certified instruments. "
        "Temperature in degrees Celsius (degC). "
        "Precipitation and snow in millimeters (mm). "
        "Wind speed in meters per second (m/s)."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Query generator (cross-unit benchmark)
# ---------------------------------------------------------------------------

def build_queries(
    documents: list[dict[str, Any]],
    corpus_dir: Path,
) -> list[dict[str, Any]]:
    """Generate cross-unit queries with ground-truth relevance judgments.

    Query categories:
      A) Temperature in °F — documents report in °C
      B) Precipitation in inches — documents report in mm
      C) Wind speed in mph — documents report in m/s
      D) Temperature threshold questions
    """
    queries: list[dict[str, Any]] = []
    qid = 1

    # --- Category A: High summer temperatures (°F) ---
    # Find months where peak TMAX >= 30°C (86°F)
    hot_docs = [
        d for d in documents
        if d.get("TMAX_max") is not None and d["TMAX_max"] >= 30.0
    ]
    if hot_docs:
        queries.append({
            "id": f"real_{qid:03d}",
            "query": "weather stations with maximum temperature exceeding 86 degrees Fahrenheit",
            "type": "cross_unit_temperature_F",
            "unit_in_query": "°F",
            "unit_in_docs": "°C",
            "threshold_query": 86.0,
            "threshold_doc": 30.0,
            "relevant_docs": [d["doc_path"] for d in hot_docs],
        })
        qid += 1

    # --- Category A2: Cold winter temperatures (°F) ---
    cold_docs = [
        d for d in documents
        if d.get("TMIN_min") is not None and d["TMIN_min"] <= 0.0
    ]
    if cold_docs:
        queries.append({
            "id": f"real_{qid:03d}",
            "query": "stations where minimum temperature dropped below freezing (32°F) in the month",
            "type": "cross_unit_temperature_F",
            "unit_in_query": "°F",
            "unit_in_docs": "°C",
            "threshold_query": 32.0,
            "threshold_doc": 0.0,
            "relevant_docs": [d["doc_path"] for d in cold_docs],
        })
        qid += 1

    # --- Category A3: Mild temperature range ---
    mild_docs = [
        d for d in documents
        if d.get("TMAX_mean") is not None
        and 15.0 <= d["TMAX_mean"] <= 25.0
    ]
    if mild_docs:
        queries.append({
            "id": f"real_{qid:03d}",
            "query": "monthly observations with average high temperatures between 59°F and 77°F (mild climate)",
            "type": "cross_unit_temperature_range",
            "unit_in_query": "°F",
            "unit_in_docs": "°C",
            "relevant_docs": [d["doc_path"] for d in mild_docs],
        })
        qid += 1

    # --- Category B: Heavy precipitation in inches ---
    wet_docs = [
        d for d in documents
        if d.get("PRCP_total") is not None and d["PRCP_total"] >= 50.0  # 50mm = ~2 inches
    ]
    if wet_docs:
        queries.append({
            "id": f"real_{qid:03d}",
            "query": "stations that received more than 2 inches of rainfall in a month",
            "type": "cross_unit_precipitation",
            "unit_in_query": "inches",
            "unit_in_docs": "mm",
            "threshold_query": 2.0,
            "threshold_doc": 50.8,
            "relevant_docs": [d["doc_path"] for d in wet_docs],
        })
        qid += 1

    # Heavy rain event: daily max >= 25mm (1 inch)
    heavy_rain_docs = [
        d for d in documents
        if d.get("PRCP_max") is not None and d["PRCP_max"] >= 25.4
    ]
    if heavy_rain_docs:
        queries.append({
            "id": f"real_{qid:03d}",
            "query": "weather records showing single-day precipitation over 1 inch",
            "type": "cross_unit_precipitation_daily",
            "unit_in_query": "inches",
            "unit_in_docs": "mm",
            "threshold_query": 1.0,
            "threshold_doc": 25.4,
            "relevant_docs": [d["doc_path"] for d in heavy_rain_docs],
        })
        qid += 1

    # --- Category C: Wind speed in mph ---
    windy_docs = [
        d for d in documents
        if d.get("AWND_max") is not None and d["AWND_max"] >= 6.7  # 6.7 m/s ~ 15 mph
    ]
    if windy_docs:
        queries.append({
            "id": f"real_{qid:03d}",
            "query": "monthly records with maximum average wind speed above 15 miles per hour",
            "type": "cross_unit_wind",
            "unit_in_query": "mph",
            "unit_in_docs": "m/s",
            "threshold_query": 15.0,
            "threshold_doc": 6.7,
            "relevant_docs": [d["doc_path"] for d in windy_docs],
        })
        qid += 1

    # --- Category D: Snowfall in inches ---
    snowy_docs = [
        d for d in documents
        if d.get("SNOW_total") is not None and d["SNOW_total"] >= 127.0  # 127mm = 5 inches
    ]
    if snowy_docs:
        queries.append({
            "id": f"real_{qid:03d}",
            "query": "stations with more than 5 inches of snowfall accumulation",
            "type": "cross_unit_snow",
            "unit_in_query": "inches",
            "unit_in_docs": "mm",
            "threshold_query": 5.0,
            "threshold_doc": 127.0,
            "relevant_docs": [d["doc_path"] for d in snowy_docs],
        })
        qid += 1

    # --- Category E: Combined condition ---
    cold_and_snowy = [
        d for d in documents
        if d.get("TMIN_min") is not None
        and d.get("SNOW_total") is not None
        and d["TMIN_min"] <= -5.0
        and d["SNOW_total"] >= 50.0
    ]
    if cold_and_snowy:
        queries.append({
            "id": f"real_{qid:03d}",
            "query": "winter months with temperatures below 23°F and significant snowfall over 2 inches",
            "type": "cross_unit_combined",
            "unit_in_query": "°F and inches",
            "unit_in_docs": "°C and mm",
            "relevant_docs": [d["doc_path"] for d in cold_and_snowy],
        })
        qid += 1

    return queries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    raw_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
    out_dir = Path(__file__).resolve().parent / "corpus_real"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect metadata from each station
    dly_files = sorted(raw_dir.glob("*.dly"))
    print(f"Found {len(dly_files)} GHCN station files")

    all_documents: list[dict[str, Any]] = []

    for dly_path in dly_files:
        station_id = dly_path.stem
        meta = STATION_META.get(station_id)
        if meta is None:
            # Auto-generate metadata for unknown stations
            meta = {
                "name": f"GHCN Station {station_id}",
                "city": "Unknown",
                "state": "US",
                "country": "USA",
                "lat": "0°N",
                "lon": "0°W",
                "elev_m": "0",
            }

        print(f"  Parsing {station_id} ({meta['name']}) ...")

        # Parse recent years (2020-2024)
        for year in (2022, 2023, 2024):
            monthly = parse_dly(dly_path, year)
            if not monthly:
                continue

            for month_key, month_data in sorted(monthly.items()):
                summary = monthly_summary(month_data)
                if not summary:
                    continue

                doc_text = make_document(station_id, meta, month_key, summary)

                # Store doc to file
                safe_key = month_key.replace("-", "_")
                filename = f"{station_id}_{safe_key}.txt"
                doc_path = out_dir / filename
                doc_path.write_text(doc_text, encoding="utf-8")

                # Build index record with numeric fields for query generation
                record: dict[str, Any] = {
                    "station_id": station_id,
                    "month_key": month_key,
                    "doc_path": str(doc_path),
                }
                if "TMAX" in summary:
                    record["TMAX_max"] = summary["TMAX"]["max"]
                    record["TMAX_mean"] = summary["TMAX"]["mean"]
                if "TMIN" in summary:
                    record["TMIN_min"] = summary["TMIN"]["min"]
                if "PRCP" in summary:
                    record["PRCP_total"] = summary["PRCP"]["sum"]
                    record["PRCP_max"] = summary["PRCP"]["max"]
                if "AWND" in summary:
                    record["AWND_max"] = summary["AWND"]["max"]
                if "SNOW" in summary:
                    record["SNOW_total"] = summary["SNOW"]["sum"]
                if "SNWD" in summary:
                    record["SNWD_max"] = summary["SNWD"]["max"]

                all_documents.append(record)

    print(f"\nGenerated {len(all_documents)} monthly station documents in {out_dir}/")

    # Build cross-unit queries
    queries = build_queries(all_documents, out_dir)
    print(f"Generated {len(queries)} cross-unit benchmark queries")

    # Write queries file
    queries_path = Path(__file__).resolve().parent / "real_queries.json"
    with open(queries_path, "w") as fh:
        json.dump({
            "description": "NOAA GHCN-Daily cross-unit benchmark queries",
            "source": "NOAA Global Historical Climatology Network - Daily (GHCN-D)",
            "stations": list(STATION_META.keys()),
            "doc_count": len(all_documents),
            "query_count": len(queries),
            "queries": queries,
        }, fh, indent=2)

    print(f"Saved queries to {queries_path}")

    # Print query summary
    print("\nQuery types generated:")
    for q in queries:
        rel_count = len(q["relevant_docs"])
        print(f"  [{q['id']}] {q['type']}: {rel_count} relevant docs")
        print(f"       Query: {q['query'][:80]}")

    return queries_path, out_dir


if __name__ == "__main__":
    main()
