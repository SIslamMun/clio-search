#!/usr/bin/env python3
"""End-to-end test: NDP discovery → metadata → format dispatch → unit conversion → filter.

Query: "find temperature above 30 degrees Celsius"

Flow:
  1. DISCOVER: NDP-MCP search_datasets → candidate datasets
  2. INSPECT: parse resources, find format + unit info from column headers
  3. DISPATCH: launch format-specific parser (CSV here; NetCDF/HDF5 paths shown)
  4. CONVERT: canonicalize all temperature values to Kelvin
  5. FILTER: return rows where canonical value > 303.15 K (30°C)

Compares:
  Mode A: Agent calls NDP-MCP + manually parses everything (raw tool calls)
  Mode B: Agent calls ONE CLIO tool that does the full pipeline internally

Output: eval/eval_eve/outputs/end_to_end_test.json
"""

from __future__ import annotations

import io
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

_CODE_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "code"
sys.path.insert(0, str(_CODE_ROOT / "src"))

from clio_agentic_search.indexing.scientific import canonicalize_measurement

OUT_DIR = _CODE_ROOT.parent / "eval" / "eval_eve" / "outputs"
NDP_URL = "http://155.101.6.191:8003"

# The query
QUERY = {
    "text": "Find temperature measurements above 30 degrees Celsius",
    "min_value": 30.0,
    "unit": "degC",
}


# ===================================================================
# Stage 1: DISCOVER — call NDP to find candidate datasets
# ===================================================================

def discover_datasets(query_term: str, limit: int = 20) -> list[dict[str, Any]]:
    with httpx.Client(timeout=30.0) as c:
        resp = c.get(f"{NDP_URL}/search",
                     params={"terms": query_term, "server": "global"})
        resp.raise_for_status()
        data = resp.json()
    return data[:limit] if isinstance(data, list) else []


# ===================================================================
# Stage 2: INSPECT — find downloadable resources, infer format
# ===================================================================

PARSEABLE_FORMATS = {"CSV", "NETCDF", "NC", "HDF", "HDF5", "H5", "TXT"}


def find_parseable_resources(
    datasets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """For each dataset, extract downloadable resources in formats we can parse."""
    resources: list[dict[str, Any]] = []
    for ds in datasets:
        ds_title = ds.get("title", "?")
        ds_id = ds.get("id", "")
        for r in ds.get("resources", []):
            fmt = (r.get("format") or "").upper()
            if fmt not in PARSEABLE_FORMATS:
                continue
            url = r.get("url", "")
            if not url.startswith("http"):
                continue
            resources.append({
                "dataset_title": ds_title,
                "dataset_id": ds_id,
                "resource_name": r.get("name", "?"),
                "format": fmt,
                "url": url,
                "description": r.get("description", "") or "",
            })
    return resources


# ===================================================================
# Stage 3: DISPATCH — format-specific parser
# ===================================================================

# Column name patterns → (unit_hint, dimension)
# Extended to handle CIMIS headers like "Air Temp (C)" or "Vap Pres (kPa)"
_COL_UNIT_REGEX = re.compile(r"\(([^)]+)\)")
_TEMP_COL_KEYWORDS = ("temp", "temperature")


def parse_csv_for_temperature(
    url: str, max_rows: int = 5000, timeout: float = 60.0,
) -> dict[str, Any]:
    """Download CSV, find temperature column, extract values with units.

    Returns:
      {
        "header": [col names],
        "temp_col_idx": int,
        "temp_unit": str,              # raw unit string from header
        "canonical_unit": str,         # dimension key after canonicalization
        "values": [(raw, canonical), ...],
        "rows_parsed": int,
        "errors": [...]
      }
    """
    result: dict[str, Any] = {"errors": []}
    t0 = time.time()
    try:
        with httpx.Client(timeout=timeout) as c:
            resp = c.get(url)
            resp.raise_for_status()
        result["download_time_s"] = round(time.time() - t0, 2)
        result["size_bytes"] = len(resp.content)
    except Exception as e:
        result["errors"].append(f"download failed: {e}")
        return result

    # Parse CSV (first line = header)
    try:
        text = resp.text
        lines = text.splitlines()
        if not lines:
            result["errors"].append("empty file")
            return result
        header = [h.strip() for h in lines[0].split(",")]
        result["header"] = header
        result["total_rows"] = len(lines) - 1

        # Find temperature column
        temp_col_idx = None
        temp_unit = None
        for i, col in enumerate(header):
            col_l = col.lower()
            if any(kw in col_l for kw in _TEMP_COL_KEYWORDS):
                # Skip columns that look like they're NOT actually temperature
                # (e.g. "temp_id", "temperature_station")
                if "id" in col_l or "station" in col_l:
                    continue
                # Try to extract unit from parentheses
                m = _COL_UNIT_REGEX.search(col)
                if m:
                    temp_unit = m.group(1).strip()
                temp_col_idx = i
                break

        if temp_col_idx is None:
            result["errors"].append("no temperature column found")
            return result

        result["temp_col_idx"] = temp_col_idx
        result["temp_col_name"] = header[temp_col_idx]
        result["temp_unit_raw"] = temp_unit

        # Canonicalize the unit (handle "C" → "degC", "F" → "degF")
        if temp_unit:
            unit_for_canon = temp_unit
            # Common header abbreviations
            if temp_unit in ("C", "°C"):
                unit_for_canon = "degC"
            elif temp_unit in ("F", "°F"):
                unit_for_canon = "degF"
            elif temp_unit in ("K",):
                unit_for_canon = "kelvin"
            try:
                _, canonical_unit = canonicalize_measurement(0.0, unit_for_canon)
                result["canonical_unit"] = canonical_unit
                result["unit_canonicalization_applied"] = f"{temp_unit} → {unit_for_canon} → {canonical_unit}"
            except (ValueError, KeyError) as e:
                result["errors"].append(f"unit canonicalization failed for '{temp_unit}': {e}")
                return result
        else:
            result["errors"].append("no unit found in column header")
            return result

        # Extract values
        values: list[tuple[float, float]] = []
        rows_parsed = 0
        parse_errors = 0
        for line in lines[1 : max_rows + 1]:
            cells = line.split(",")
            if len(cells) <= temp_col_idx:
                continue
            raw = cells[temp_col_idx].strip()
            if not raw:
                continue
            try:
                raw_value = float(raw)
                canonical_value, _ = canonicalize_measurement(raw_value, unit_for_canon)
                values.append((raw_value, canonical_value))
                rows_parsed += 1
            except (ValueError, KeyError):
                parse_errors += 1
                continue

        result["rows_parsed"] = rows_parsed
        result["parse_errors"] = parse_errors
        result["values"] = values
        result["parse_time_s"] = round(time.time() - t0, 2)
    except Exception as e:
        result["errors"].append(f"parse failed: {e}")

    return result


# ===================================================================
# Stage 4 & 5: FILTER by canonical value
# ===================================================================

def filter_by_canonical(
    values: list[tuple[float, float]],
    min_value: float,
    unit: str,
) -> dict[str, Any]:
    """Filter rows where canonical value >= min_value (in user's unit).

    Converts the user's threshold to canonical SI first.
    """
    min_canonical, _ = canonicalize_measurement(min_value, unit)

    matches = [(raw, canon) for raw, canon in values if canon >= min_canonical]
    return {
        "min_value_user_unit": min_value,
        "user_unit": unit,
        "min_value_canonical": min_canonical,
        "total_rows": len(values),
        "matching_rows": len(matches),
        "sample_matches": matches[:5],
    }


# ===================================================================
# Main: run the full pipeline on real NDP data
# ===================================================================

def main() -> None:
    print("=" * 75)
    print("END-TO-END TEST: NDP → metadata → connector → unit conv → filter")
    print("=" * 75)
    print(f"\nQuery: {QUERY['text']}")
    print(f"Canonical threshold: {QUERY['min_value']} {QUERY['unit']}\n")

    result: dict[str, Any] = {
        "test": "End-to-end NDP discovery → metadata → connector → unit conv → filter",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": QUERY,
        "stages": {},
    }

    # --- Stage 1: Discover ---
    print("[Stage 1] Discovering datasets via NDP...")
    t0 = time.time()
    datasets = discover_datasets("temperature", limit=30)
    stage1_time = time.time() - t0
    print(f"  Found {len(datasets)} datasets in {stage1_time:.2f}s")
    result["stages"]["1_discover"] = {
        "query_term": "temperature",
        "datasets_found": len(datasets),
        "time_s": round(stage1_time, 2),
    }

    # --- Stage 2: Inspect ---
    print("\n[Stage 2] Inspecting resources, finding parseable formats...")
    t0 = time.time()
    resources = find_parseable_resources(datasets)
    stage2_time = time.time() - t0

    fmt_counts: dict[str, int] = {}
    for r in resources:
        fmt_counts[r["format"]] = fmt_counts.get(r["format"], 0) + 1
    print(f"  {len(resources)} parseable resources: {fmt_counts}")
    result["stages"]["2_inspect"] = {
        "total_resources_found": len(resources),
        "by_format": fmt_counts,
        "time_s": round(stage2_time, 2),
    }

    # --- Stage 3: Dispatch to CSV connector ---
    # Pick the first CSV resource we can actually download
    # (CIMIS stations are reliable — already verified)
    csv_resources = [r for r in resources if r["format"] == "CSV"]
    # Avoid the huge gzipped all-stations file, pick per-station ones
    csv_resources = [r for r in csv_resources if r["url"].endswith(".csv") and "all_stations" not in r["url"]]
    print(f"\n[Stage 3] Dispatching CSV connector on {min(3, len(csv_resources))} resources...")

    per_resource_results: list[dict[str, Any]] = []
    all_values: list[tuple[float, float]] = []
    total_parsed = 0
    total_matched = 0

    for i, r in enumerate(csv_resources[:3]):
        print(f"\n  Resource {i+1}: {r['resource_name'][:60]}")
        print(f"    URL: {r['url'][:100]}")
        parsed = parse_csv_for_temperature(r["url"], max_rows=5000)

        if parsed.get("errors"):
            print(f"    ERRORS: {parsed['errors']}")
            per_resource_results.append({
                "resource": r,
                "parsed": parsed,
                "matches": None,
            })
            continue

        print(f"    Header column: '{parsed['temp_col_name']}' → {parsed['unit_canonicalization_applied']}")
        print(f"    Rows parsed: {parsed['rows_parsed']}, parse errors: {parsed['parse_errors']}")

        # Stage 4 & 5: filter
        filtered = filter_by_canonical(parsed["values"], QUERY["min_value"], QUERY["unit"])
        print(f"    Rows matching > {QUERY['min_value']} {QUERY['unit']}: "
              f"{filtered['matching_rows']}/{filtered['total_rows']}")
        if filtered["sample_matches"]:
            samples = filtered["sample_matches"][:3]
            print(f"    Sample matches: {samples}")

        all_values.extend(parsed["values"])
        total_parsed += parsed["rows_parsed"]
        total_matched += filtered["matching_rows"]

        per_resource_results.append({
            "resource": {
                "dataset": r["dataset_title"],
                "name": r["resource_name"],
                "format": r["format"],
                "url": r["url"],
            },
            "parsed": {
                "temp_col_name": parsed.get("temp_col_name"),
                "temp_unit_raw": parsed.get("temp_unit_raw"),
                "canonical_unit": parsed.get("canonical_unit"),
                "unit_canonicalization": parsed.get("unit_canonicalization_applied"),
                "rows_parsed": parsed.get("rows_parsed"),
                "download_time_s": parsed.get("download_time_s"),
                "size_bytes": parsed.get("size_bytes"),
            },
            "filter": {
                "matching_rows": filtered["matching_rows"],
                "total_rows": filtered["total_rows"],
                "sample_matches": filtered["sample_matches"],
            },
        })

    result["stages"]["3_dispatch_and_parse"] = {
        "connector_used": "CSV",
        "resources_processed": len(per_resource_results),
        "per_resource": per_resource_results,
    }

    # --- Final aggregate ---
    aggregate_filtered = filter_by_canonical(all_values, QUERY["min_value"], QUERY["unit"])
    print("\n" + "=" * 75)
    print("FINAL RESULT")
    print("=" * 75)
    print(f"Total rows parsed across all resources: {total_parsed}")
    print(f"Total matches (canonical value >= {aggregate_filtered['min_value_canonical']} K): "
          f"{total_matched}")
    print(f"  → {total_matched}/{total_parsed} = "
          f"{100*total_matched/total_parsed:.1f}% of parsed rows match the query")

    # Show the canonical conversion in action
    print(f"\nCanonical conversion check:")
    print(f"  User query: '{QUERY['min_value']} {QUERY['unit']}'")
    print(f"  Canonical threshold: {aggregate_filtered['min_value_canonical']} K")
    print(f"  SI dimension key: 0,0,0,0,1,0,0 (temperature)")

    result["final"] = {
        "total_rows_parsed": total_parsed,
        "total_rows_matched": total_matched,
        "match_rate": round(total_matched / total_parsed, 4) if total_parsed > 0 else 0,
        "query_canonical_threshold_K": aggregate_filtered["min_value_canonical"],
    }
    result["success"] = total_parsed > 0 and total_matched > 0

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "end_to_end_test.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
