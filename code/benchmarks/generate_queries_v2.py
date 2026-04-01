#!/usr/bin/env python3
"""Generate queries_v2.json by scanning corpus_v2/ for actual measurements.

This ensures ground truth is accurate: we parse each document for numeric values
with units, then build queries whose ranges match specific documents.

Run from the code/ directory:
    python3 benchmarks/generate_queries_v2.py
"""

import json
import re
import os
from pathlib import Path
from collections import defaultdict

CORPUS_DIR = Path(__file__).resolve().parent / "corpus_v2"
OUTPUT_PATH = Path(__file__).resolve().parent / "queries_v2.json"

# ---------------------------------------------------------------------------
# Unit conversion tables (must match what clio-agentic-search uses)
# ---------------------------------------------------------------------------
# We define conversions TO a canonical (SI) unit.
UNIT_TO_SI = {
    # Pressure
    "Pa":  ("Pa", 1.0),
    "kPa": ("Pa", 1000.0),
    "MPa": ("Pa", 1e6),
    "GPa": ("Pa", 1e9),
    # Distance
    "mm":  ("m", 0.001),
    "cm":  ("m", 0.01),
    "m":   ("m", 1.0),
    "km":  ("m", 1000.0),
    # Mass
    "mg":  ("g", 0.001),
    "g":   ("g", 1.0),
    "kg":  ("g", 1000.0),
    # Time
    "s":   ("s", 1.0),
    "min": ("s", 60.0),
    "h":   ("s", 3600.0),
    # Velocity
    "m/s":  ("m/s", 1.0),
    "km/h": ("m/s", 1.0/3.6),
}

# Units NOT in our conversion table (for graceful degradation testing)
UNSUPPORTED_UNITS = {"hPa", "degC", "degF", "mmHg", "dBZ", "Hz", "kHz",
                     "MHz", "GHz", "nm", "um", "mN", "kN", "W", "kW", "MW",
                     "mV", "V", "Ohm", "cP", "rpm", "sccm", "J", "kJ",
                     "mL", "L", "uL", "nL"}

# Physical dimension grouping
UNIT_DIMENSION = {}
for u in ("Pa", "kPa", "MPa", "GPa"):
    UNIT_DIMENSION[u] = "pressure"
for u in ("mm", "cm", "m", "km"):
    UNIT_DIMENSION[u] = "distance"
for u in ("mg", "g", "kg"):
    UNIT_DIMENSION[u] = "mass"
for u in ("s", "min", "h"):
    UNIT_DIMENSION[u] = "time"
for u in ("m/s", "km/h"):
    UNIT_DIMENSION[u] = "velocity"


def extract_measurements(text: str) -> list[tuple[float, str]]:
    """Extract (value, unit) pairs from text."""
    # Pattern: number followed by unit (with optional space)
    # We need to be careful: "m/s" must be matched before bare "m"
    unit_pattern = "|".join(sorted(
        set(UNIT_TO_SI.keys()) | UNSUPPORTED_UNITS,
        key=lambda x: -len(x)  # longest first
    ))
    pattern = rf'(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*({unit_pattern})(?:\b|[^a-zA-Z/])'
    results = []
    for match in re.finditer(pattern, text):
        val = float(match.group(1))
        unit = match.group(2)
        results.append((val, unit))
    return results


def convert_to_si(value: float, unit: str) -> tuple[float, str] | None:
    """Convert value+unit to SI. Returns None if unsupported."""
    if unit not in UNIT_TO_SI:
        return None
    si_unit, factor = UNIT_TO_SI[unit]
    return (value * factor, si_unit)


def value_in_range(value: float, unit: str, range_min: float, range_max: float, range_unit: str) -> bool:
    """Check if a measurement falls in a query range (after unit conversion)."""
    si_val = convert_to_si(value, unit)
    si_range_min_t = convert_to_si(range_min, range_unit)
    si_range_max_t = convert_to_si(range_max, range_unit)
    if si_val is None or si_range_min_t is None or si_range_max_t is None:
        return False
    # Must be same dimension
    if si_val[1] != si_range_min_t[1]:
        return False
    return si_range_min_t[0] <= si_val[0] <= si_range_max_t[0]


def scan_corpus():
    """Scan all documents and extract measurements."""
    doc_measurements = {}  # relative_path -> list of (value, unit)
    for domain_dir in sorted(CORPUS_DIR.iterdir()):
        if not domain_dir.is_dir():
            continue
        for doc_path in sorted(domain_dir.glob("*.txt")):
            rel_path = f"{domain_dir.name}/{doc_path.name}"
            text = doc_path.read_text()
            measurements = extract_measurements(text)
            doc_measurements[rel_path] = measurements
    return doc_measurements


def find_relevant_docs(doc_measurements, range_min, range_max, range_unit, target_dimension=None):
    """Find all docs with at least one measurement in the given range."""
    relevant = []
    for doc_path, measurements in doc_measurements.items():
        for val, unit in measurements:
            if unit not in UNIT_TO_SI:
                continue
            dim = UNIT_DIMENSION.get(unit)
            if target_dimension and dim != target_dimension:
                continue
            if value_in_range(val, unit, range_min, range_max, range_unit):
                relevant.append(doc_path)
                break  # one match per doc is enough
    return sorted(relevant)


def find_docs_with_unit(doc_measurements, target_unit):
    """Find docs containing measurements in a specific unit."""
    relevant = []
    for doc_path, measurements in doc_measurements.items():
        for val, unit in measurements:
            if unit == target_unit:
                relevant.append(doc_path)
                break
    return sorted(relevant)


def find_docs_with_formula(doc_measurements, formula_pattern, corpus_dir=CORPUS_DIR):
    """Find docs containing a formula pattern."""
    relevant = []
    for domain_dir in sorted(corpus_dir.iterdir()):
        if not domain_dir.is_dir():
            continue
        for doc_path in sorted(domain_dir.glob("*.txt")):
            rel_path = f"{domain_dir.name}/{doc_path.name}"
            text = doc_path.read_text()
            if re.search(formula_pattern, text, re.IGNORECASE):
                relevant.append(rel_path)
    return sorted(relevant)


def find_docs_multi_constraint(doc_measurements, constraints):
    """Find docs matching ALL constraints (list of (min, max, unit, dimension) tuples)."""
    candidate_sets = []
    for range_min, range_max, range_unit, dimension in constraints:
        docs = find_relevant_docs(doc_measurements, range_min, range_max, range_unit, dimension)
        candidate_sets.append(set(docs))
    if not candidate_sets:
        return []
    result = candidate_sets[0]
    for s in candidate_sets[1:]:
        result &= s
    return sorted(result)


def main():
    print("Scanning corpus_v2/ for measurements...")
    doc_measurements = scan_corpus()
    print(f"  Scanned {len(doc_measurements)} documents")

    # Count measurements per dimension
    dim_counts = defaultdict(int)
    for doc_path, measurements in doc_measurements.items():
        dims_seen = set()
        for val, unit in measurements:
            dim = UNIT_DIMENSION.get(unit)
            if dim and dim not in dims_seen:
                dim_counts[dim] += 1
                dims_seen.add(dim)
    print(f"  Docs with pressure: {dim_counts['pressure']}")
    print(f"  Docs with distance: {dim_counts['distance']}")
    print(f"  Docs with mass: {dim_counts['mass']}")
    print(f"  Docs with time: {dim_counts['time']}")
    print(f"  Docs with velocity: {dim_counts['velocity']}")

    queries = {
        "cross_unit_queries": [],
        "same_unit_queries": [],
        "formula_queries": [],
        "multi_constraint_queries": [],
    }

    # ==========================================================================
    # CROSS-UNIT QUERIES (40 total, 8 per domain)
    # ==========================================================================

    # --- Pressure (8) ---
    pressure_specs = [
        ("xp_01", "experiments with chamber pressure between 200 and 400 kPa",
         200, 400, "kPa", "pressure"),
        ("xp_02", "simulations with pressure above 300 kPa",
         300, 1000, "kPa", "pressure"),
        ("xp_03", "low pressure measurements between 5 and 50 kPa",
         5, 50, "kPa", "pressure"),
        ("xp_04", "tests at near-atmospheric pressure around 100 kPa",
         90, 110, "kPa", "pressure"),
        ("xp_05", "high-pressure experiments above 1 MPa",
         1, 100, "MPa", "pressure"),
        ("xp_06", "vacuum or low-pressure systems below 10 kPa",
         0, 10, "kPa", "pressure"),
        ("xp_07", "pressure in the range of 50000 to 100000 Pa",
         50000, 100000, "Pa", "pressure"),
        ("xp_08", "moderate pressure between 500 kPa and 2 MPa",
         500, 2000, "kPa", "pressure"),
    ]
    for qid, query, lo, hi, unit, dim in pressure_specs:
        relevant = find_relevant_docs(doc_measurements, lo, hi, unit, dim)
        queries["cross_unit_queries"].append({
            "id": qid,
            "query": query,
            "domain": dim,
            "numeric_range": f"{lo}:{hi}:{unit}",
            "relevant_docs": relevant,
            "type": "cross_unit",
        })

    # --- Distance (8) ---
    distance_specs = [
        ("xd_01", "sample thickness around 1 to 5 mm",
         1, 5, "mm", "distance"),
        ("xd_02", "pipe or nozzle diameter between 20 and 60 mm",
         20, 60, "mm", "distance"),
        ("xd_03", "large-scale dimensions between 2 and 15 m",
         2, 15, "m", "distance"),
        ("xd_04", "mesh or grid spacing below 1 mm",
         0, 1, "mm", "distance"),
        ("xd_05", "gauge length around 25 to 50 mm",
         25, 50, "mm", "distance"),
        ("xd_06", "small specimen width between 5 and 15 mm",
         5, 15, "mm", "distance"),
        ("xd_07", "component dimensions around 10 to 30 cm",
         10, 30, "cm", "distance"),
        ("xd_08", "displacements or deflections between 0.05 and 5 mm",
         0.05, 5, "mm", "distance"),
    ]
    for qid, query, lo, hi, unit, dim in distance_specs:
        relevant = find_relevant_docs(doc_measurements, lo, hi, unit, dim)
        queries["cross_unit_queries"].append({
            "id": qid,
            "query": query,
            "domain": dim,
            "numeric_range": f"{lo}:{hi}:{unit}",
            "relevant_docs": relevant,
            "type": "cross_unit",
        })

    # --- Mass (8) ---
    mass_specs = [
        ("xm_01", "reagent or sample mass between 50 and 500 mg",
         50, 500, "mg", "mass"),
        ("xm_02", "product mass around 1 to 5 g",
         1, 5, "g", "mass"),
        ("xm_03", "heavy specimens above 1 kg",
         1, 100, "kg", "mass"),
        ("xm_04", "small sample mass below 20 mg",
         0, 20, "mg", "mass"),
        ("xm_05", "electrode or thin film mass below 50 mg",
         0, 50, "mg", "mass"),
        ("xm_06", "batch or component mass between 0.1 and 1 g",
         0.1, 1, "g", "mass"),
        ("xm_07", "kilogram-scale samples between 2 and 50 kg",
         2, 50, "kg", "mass"),
        ("xm_08", "sub-gram reagent quantities between 100 and 1000 mg",
         100, 1000, "mg", "mass"),
    ]
    for qid, query, lo, hi, unit, dim in mass_specs:
        relevant = find_relevant_docs(doc_measurements, lo, hi, unit, dim)
        queries["cross_unit_queries"].append({
            "id": qid,
            "query": query,
            "domain": dim,
            "numeric_range": f"{lo}:{hi}:{unit}",
            "relevant_docs": relevant,
            "type": "cross_unit",
        })

    # --- Time (8) ---
    time_specs = [
        ("xt_01", "short experiments under 60 seconds",
         0, 60, "s", "time"),
        ("xt_02", "reaction or test duration between 1 and 30 minutes",
         1, 30, "min", "time"),
        ("xt_03", "long simulations or experiments above 10 hours",
         10, 10000, "h", "time"),
        ("xt_04", "process lasting between 100 and 600 seconds",
         100, 600, "s", "time"),
        ("xt_05", "multi-hour experiments between 1 and 10 hours",
         1, 10, "h", "time"),
        ("xt_06", "quick measurements under 5 minutes",
         0, 5, "min", "time"),
        ("xt_07", "simulation or observation around 30 to 60 min",
         30, 60, "min", "time"),
        ("xt_08", "duration between 1000 and 10000 seconds",
         1000, 10000, "s", "time"),
    ]
    for qid, query, lo, hi, unit, dim in time_specs:
        relevant = find_relevant_docs(doc_measurements, lo, hi, unit, dim)
        queries["cross_unit_queries"].append({
            "id": qid,
            "query": query,
            "domain": dim,
            "numeric_range": f"{lo}:{hi}:{unit}",
            "relevant_docs": relevant,
            "type": "cross_unit",
        })

    # --- Velocity (8) ---
    velocity_specs = [
        ("xv_01", "flow velocity between 1 and 10 m/s",
         1, 10, "m/s", "velocity"),
        ("xv_02", "high-speed flow between 50 and 150 m/s",
         50, 150, "m/s", "velocity"),
        ("xv_03", "wind speed between 5 and 20 m/s",
         5, 20, "m/s", "velocity"),
        ("xv_04", "speed around 50 to 150 km/h",
         50, 150, "km/h", "velocity"),
        ("xv_05", "low flow velocity below 2 m/s",
         0, 2, "m/s", "velocity"),
        ("xv_06", "high wind or flow speed between 30 and 80 m/s",
         30, 80, "m/s", "velocity"),
        ("xv_07", "moderate velocity between 10 and 30 m/s",
         10, 30, "m/s", "velocity"),
        ("xv_08", "vehicle or wind speed around 100 to 250 km/h",
         100, 250, "km/h", "velocity"),
    ]
    for qid, query, lo, hi, unit, dim in velocity_specs:
        relevant = find_relevant_docs(doc_measurements, lo, hi, unit, dim)
        queries["cross_unit_queries"].append({
            "id": qid,
            "query": query,
            "domain": dim,
            "numeric_range": f"{lo}:{hi}:{unit}",
            "relevant_docs": relevant,
            "type": "cross_unit",
        })

    # ==========================================================================
    # SAME-UNIT CONTROL QUERIES (20)
    # ==========================================================================

    same_unit_specs = [
        ("sp_01", "pressure around 100 to 200 kPa", 100, 200, "kPa", "pressure"),
        ("sp_02", "pressure between 80000 and 120000 Pa", 80000, 120000, "Pa", "pressure"),
        ("sp_03", "stress between 200 and 500 MPa", 200, 500, "MPa", "pressure"),
        ("sp_04", "modulus in the 50 to 150 GPa range", 50, 150, "GPa", "pressure"),
        ("sd_01", "lengths between 100 and 300 mm", 100, 300, "mm", "distance"),
        ("sd_02", "dimensions between 1 and 10 m", 1, 10, "m", "distance"),
        ("sd_03", "distances in the 1 to 50 km range", 1, 50, "km", "distance"),
        ("sm_01", "mass between 100 and 500 mg", 100, 500, "mg", "mass"),
        ("sm_02", "mass between 1 and 10 g", 1, 10, "g", "mass"),
        ("sm_03", "mass between 1 and 50 kg", 1, 50, "kg", "mass"),
        ("st_01", "time between 100 and 1000 s", 100, 1000, "s", "time"),
        ("st_02", "duration between 10 and 60 min", 10, 60, "min", "time"),
        ("st_03", "duration between 1 and 50 h", 1, 50, "h", "time"),
        ("sv_01", "velocity between 5 and 50 m/s", 5, 50, "m/s", "velocity"),
        ("sv_02", "speed between 50 and 200 km/h", 50, 200, "km/h", "velocity"),
        ("sp_05", "low pressure below 50 kPa", 0, 50, "kPa", "pressure"),
        ("sd_04", "sub-millimeter dimensions", 0, 1, "mm", "distance"),
        ("sm_04", "mass below 10 mg", 0, 10, "mg", "mass"),
        ("st_04", "time below 1 second", 0, 1, "s", "time"),
        ("sv_03", "velocity below 1 m/s", 0, 1, "m/s", "velocity"),
    ]
    for qid, query, lo, hi, unit, dim in same_unit_specs:
        relevant = find_relevant_docs(doc_measurements, lo, hi, unit, dim)
        queries["same_unit_queries"].append({
            "id": qid,
            "query": query,
            "domain": dim,
            "numeric_range": f"{lo}:{hi}:{unit}",
            "relevant_docs": relevant,
            "type": "same_unit",
        })

    # ==========================================================================
    # FORMULA QUERIES (10)
    # ==========================================================================

    formula_specs = [
        ("f_01", "Newton's second law force equals mass times acceleration",
         "F=ma", r'F\s*=\s*m\s*[\*x]?\s*a\b'),
        ("f_02", "Einstein mass-energy equivalence E equals mc squared",
         "E=mc^2", r'E\s*=\s*m\s*c\s*[\^]\s*2'),
        ("f_03", "kinetic energy one half mv squared",
         "KE=0.5*m*v^{2}", r'0\.5\s*\*?\s*(?:rho|m)\s*\*?\s*[vVA]\s*[\^]?\s*[23]|q\s*=\s*0\.5'),
        ("f_04", "ideal gas law PV equals nRT",
         "PV=nRT", r'PV\s*=\s*nRT|PV/RT|n\s*=\s*PV\s*/\s*RT'),
        ("f_05", "Arrhenius equation rate constant exponential",
         "k=Ae^{-Ea/RT}", r'k\s*=\s*A\s*[\*]?\s*e\s*\^\s*\{?\s*-\s*Ea\s*/\s*R\s*T|Arrhenius'),
        ("f_06", "Reynolds number ratio of inertial to viscous forces",
         "Re=rhoVD/mu", r'Re\s*=|Reynolds\s+number'),
        ("f_07", "Bernoulli equation pressure velocity relationship",
         "P+0.5*rho*v^2=const", r'[Bb]ernoulli|0\.5\s*\*?\s*rho\s*\*?\s*[vV]\s*\^?\s*2'),
        ("f_08", "Stoney equation for thin film stress",
         "sigma=Es*ts^2/(6*R*tf)", r'[Ss]toney'),
        ("f_09", "Paris law for fatigue crack growth",
         "da/dN=C*(DeltaK)^m", r'[Pp]aris[\s-]*law|da/dN\s*=\s*C'),
        ("f_10", "Darcy friction factor for pipe flow",
         "f=DarcyFriction", r'[Dd]arcy\s+friction|[Mm]oody\s+diagram'),
    ]
    for qid, query, formula, pattern in formula_specs:
        relevant = find_docs_with_formula(doc_measurements, pattern)
        queries["formula_queries"].append({
            "id": qid,
            "query": query,
            "formula": formula,
            "relevant_docs": relevant,
            "type": "formula",
        })

    # ==========================================================================
    # MULTI-CONSTRAINT QUERIES (10)
    # ==========================================================================

    multi_specs = [
        ("mc_01",
         "simulations with pressure above 200 kPa and velocity above 10 m/s",
         [(200, 100000, "kPa", "pressure"), (10, 5000, "m/s", "velocity")]),
        ("mc_02",
         "experiments with sample mass below 1 g and test duration under 10 minutes",
         [(0, 1, "g", "mass"), (0, 10, "min", "time")]),
        ("mc_03",
         "tests with stress above 300 MPa and specimen thickness under 5 mm",
         [(300, 10000, "MPa", "pressure"), (0, 5, "mm", "distance")]),
        ("mc_04",
         "CFD cases with pressure over 100 kPa and mesh spacing below 2 mm",
         [(100, 100000, "kPa", "pressure"), (0, 2, "mm", "distance")]),
        ("mc_05",
         "atmospheric observations with pressure near 100 kPa and wind speed above 5 m/s",
         [(90, 110, "kPa", "pressure"), (5, 500, "m/s", "velocity")]),
        ("mc_06",
         "high-pressure high-velocity flow experiments",
         [(500, 100000, "kPa", "pressure"), (50, 5000, "m/s", "velocity")]),
        ("mc_07",
         "material tests with mass above 10 g and duration above 1 hour",
         [(10, 1e6, "g", "mass"), (1, 1e6, "h", "time")]),
        ("mc_08",
         "pipe or channel flow with diameter 20-100 mm and velocity 0.5-5 m/s",
         [(20, 100, "mm", "distance"), (0.5, 5, "m/s", "velocity")]),
        ("mc_09",
         "simulations with time step below 0.01 s and pressure above 50 kPa",
         [(0, 0.01, "s", "time"), (50, 100000, "kPa", "pressure")]),
        ("mc_10",
         "experiments with specimen length 100-500 mm and mass 1-100 g",
         [(100, 500, "mm", "distance"), (1, 100, "g", "mass")]),
    ]
    for qid, query, constraints in multi_specs:
        relevant = find_docs_multi_constraint(doc_measurements, constraints)
        queries["multi_constraint_queries"].append({
            "id": qid,
            "query": query,
            "constraints": [
                {"range": f"{lo}:{hi}:{unit}", "dimension": dim}
                for lo, hi, unit, dim in constraints
            ],
            "relevant_docs": relevant,
            "type": "multi_constraint",
        })

    # ==========================================================================
    # SUMMARY & WRITE
    # ==========================================================================

    total_queries = sum(len(v) for v in queries.values())
    print(f"\nGenerated {total_queries} queries:")
    for category, qlist in queries.items():
        print(f"  {category}: {len(qlist)}")
        # Report how many have at least 1 relevant doc
        with_docs = sum(1 for q in qlist if q["relevant_docs"])
        empty = sum(1 for q in qlist if not q["relevant_docs"])
        print(f"    with relevant docs: {with_docs}, empty: {empty}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(queries, f, indent=2)
    print(f"\nWritten to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
