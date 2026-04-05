#!/usr/bin/env python3
"""L6: CIMIS quality filter end-to-end on real NOAA-standard QC data.

Downloads 5 CIMIS weather station CSVs (California Irrigation Management
Information System, publicly available via NDP) and runs CLIO's
science-aware CSV parser + quality filter on them. The CSVs contain:

  * Air Temp (C), Rel Hum (%), Wind Speed (m/s), Sol Rad (W/sq.m),
    Vap Pres (kPa), Dew Point (C), Precip (mm), Soil Temp (C)
  * A "qc" column after each measurement with CIMIS flags:
      blank → good
      Y     → questionable
      R     → rejected (bad)
      M     → missing
  * ~130K rows per station (15 years × 24 hours × 365 days)

We measure:
  * How many rows each station has
  * Quality flag distribution per column
  * How many rows survive the default quality filter (GOOD + ESTIMATED)
  * For a cross-unit query ("temperature above 30°C"), how many rows
    match with vs without the quality filter
  * Unit canonicalisation trace (C → degC → Kelvin)

This demonstrates capability #3 (data quality) on REAL scientific data
with REAL QC flags from a production weather station network.

Output
------
  eval/eval_final/outputs/L6_cimis_quality_filter.json

Stations
--------
  105 Westlands, 124 Panoche, 125 Arvin-Edison,
  131 Fair Oaks, 146 Belridge

(All publicly accessible via
 https://f3i-supercomputing.s3.us-east-2.amazonaws.com/data/stations/*.csv)
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

_REPO = Path(__file__).resolve().parents[4]
_CODE = _REPO / "code"
sys.path.insert(0, str(_CODE / "src"))

from clio_agentic_search.indexing.csv_parser import (
    filter_rows_by_concept,
    parse_scientific_csv,
)
from clio_agentic_search.indexing.quality import QualityFlag
from clio_agentic_search.indexing.scientific import canonicalize_measurement

OUT_DIR = _REPO / "eval" / "eval_final" / "outputs"

CIMIS_STATIONS = [
    ("105-westlands", "CIMIS 105 Westlands"),
    ("124-panoche", "CIMIS 124 Panoche"),
    ("125-arvinedison", "CIMIS 125 Arvin-Edison"),
    ("131-fairoaks", "CIMIS 131 Fair Oaks"),
    ("146-belridge", "CIMIS 146 Belridge"),
]
BASE_URL = "https://f3i-supercomputing.s3.us-east-2.amazonaws.com/data/stations/"


def fetch_station(station_id: str, timeout: float = 120.0) -> str | None:
    url = f"{BASE_URL}{station_id}.csv"
    try:
        with httpx.Client(timeout=timeout) as c:
            resp = c.get(url)
            resp.raise_for_status()
            return resp.text
    except Exception as e:
        print(f"  ERROR fetching {station_id}: {e}")
        return None


def analyse_station(station_id: str, station_name: str, text: str) -> dict[str, Any]:
    """Run parse_scientific_csv on one station and report quality breakdown."""
    t0 = time.time()
    parsed = parse_scientific_csv(text, max_rows=200_000)
    parse_time = time.time() - t0

    # Per-column quality distribution
    col_quality: dict[str, Counter[str]] = {}
    for row in parsed.rows:
        for col, m in zip(
            parsed.schema.measurement_columns, row.measurements, strict=False,
        ):
            col_name = col.display_name
            col_quality.setdefault(col_name, Counter())[m.quality] += 1

    # Convert Counter → plain dict for JSON serialisation
    col_quality_json = {
        col: dict(counter) for col, counter in col_quality.items()
    }

    # Apply the cross-unit query: temperature ≥ 30°C = 303.15 K
    try:
        threshold_K, _ = canonicalize_measurement(30.0, "degC")
    except Exception:
        threshold_K = 303.15

    # With quality filter (default: GOOD + ESTIMATED)
    with_qc = filter_rows_by_concept(
        parsed, "temperature",
        min_value_canonical=threshold_K,
    )
    # Without quality filter (accept all flags)
    without_qc = filter_rows_by_concept(
        parsed, "temperature",
        min_value_canonical=threshold_K,
        acceptable_quality=frozenset(QualityFlag),
    )

    return {
        "station_id": station_id,
        "station_name": station_name,
        "total_rows": parsed.total_rows,
        "rows_parsed": len(parsed.rows),
        "parse_errors": parsed.parse_errors,
        "parse_time_s": round(parse_time, 2),
        "measurement_columns": [
            {
                "name": c.display_name,
                "raw_unit": c.raw_unit,
                "canonical_unit_for_registry": c.canonical_unit_for_registry,
                "canonical_concept": c.canonical_concept,
                "qc_column_detected": c.qc_column_index is not None,
            }
            for c in parsed.schema.measurement_columns
        ],
        "quality_by_column": col_quality_json,
        "query_temperature_above_30C": {
            "threshold_canonical_K": round(threshold_K, 2),
            "rows_matching_with_quality_filter": len(with_qc),
            "rows_matching_without_quality_filter": len(without_qc),
            "rows_dropped_by_quality_filter": len(without_qc) - len(with_qc),
            "sample_matches": with_qc[:3] if with_qc else [],
        },
    }


def main() -> None:
    print("=" * 75)
    print("L6: CIMIS quality filter end-to-end on real data")
    print("=" * 75)

    station_results: list[dict[str, Any]] = []
    for station_id, station_name in CIMIS_STATIONS:
        print(f"\n{station_name}")
        print(f"  Fetching {BASE_URL}{station_id}.csv ...")
        t0 = time.time()
        text = fetch_station(station_id)
        if text is None:
            continue
        download_mb = len(text) / (1024 * 1024)
        print(f"  Downloaded {download_mb:.1f} MB in {time.time() - t0:.1f}s")

        result = analyse_station(station_id, station_name, text)
        station_results.append(result)

        qm = result["query_temperature_above_30C"]
        print(
            f"  Rows: {result['rows_parsed']} parsed; "
            f"temp>30°C: {qm['rows_matching_with_quality_filter']} "
            f"(dropped {qm['rows_dropped_by_quality_filter']} by QC)"
        )
        # Print per-column quality summary
        print("  Quality distribution by measurement column:")
        for col, counts in result["quality_by_column"].items():
            total = sum(counts.values())
            good = counts.get("good", 0)
            bad = counts.get("bad", 0)
            missing = counts.get("missing", 0)
            quest = counts.get("questionable", 0)
            print(
                f"    {col:<20} | good={good:>6} bad={bad:>4} "
                f"miss={missing:>4} ques={quest:>4} (total={total})"
            )

    # Aggregate
    total_rows = sum(r["rows_parsed"] for r in station_results)
    total_matches_with_qc = sum(
        r["query_temperature_above_30C"]["rows_matching_with_quality_filter"]
        for r in station_results
    )
    total_matches_without_qc = sum(
        r["query_temperature_above_30C"]["rows_matching_without_quality_filter"]
        for r in station_results
    )
    total_dropped = total_matches_without_qc - total_matches_with_qc

    print("\n" + "=" * 75)
    print("AGGREGATE ACROSS ALL STATIONS")
    print("=" * 75)
    print(f"  Stations analysed: {len(station_results)}")
    print(f"  Total rows parsed: {total_rows:,}")
    print(f"  Temperature ≥ 30°C matches:")
    print(f"    With quality filter:    {total_matches_with_qc:,}")
    print(f"    Without quality filter: {total_matches_without_qc:,}")
    print(f"    Dropped by QC filter:   {total_dropped:,}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "L6: CIMIS quality filter end-to-end",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stations_analysed": len(station_results),
        "total_rows_parsed": total_rows,
        "aggregate": {
            "total_rows": total_rows,
            "matches_with_quality_filter": total_matches_with_qc,
            "matches_without_quality_filter": total_matches_without_qc,
            "rows_dropped_by_quality_filter": total_dropped,
        },
        "per_station": station_results,
    }
    with (OUT_DIR / "L6_cimis_quality_filter.json").open("w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {OUT_DIR / 'L6_cimis_quality_filter.json'}")


if __name__ == "__main__":
    main()
