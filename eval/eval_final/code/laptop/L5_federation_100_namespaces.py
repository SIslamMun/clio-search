#!/usr/bin/env python3
"""100-namespace federation test.

Generates 100 synthetic namespaces with labeled characteristics and runs
queries against them to demonstrate CLIO's per-dataset strategy adaptation.

Why: The paper's motivation says "100+ datasets, no single query strategy
works for all of them". This test actually creates 100 datasets with
heterogeneous characteristics and measures:

  1. How fast CLIO profiles all 100 (should be sub-second total).
  2. How many branches CLIO skips per query because the profile shows
     the branch wouldn't help.
  3. Whether CLIO correctly routes queries to the subset of namespaces
     that actually contain the answer.
  4. How much work is avoided vs a naive "query every namespace, every
     branch" baseline.

Namespace types generated:
  - rich_sci     : structured measurements, schema_density ≥ 0.5
  - sparse_sci   : some measurements hidden in prose (recoverable via sampling)
  - pure_text    : no measurements, only prose
  - formula_heavy: contains equations
  - empty        : no chunks

Each namespace is tagged with a domain (temperature, pressure, humidity,
wind, radiation) so the test can check routing correctness.

Output: eval/eval_final/outputs/L5_federation_100_namespaces.json
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_CODE_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent / "code"
sys.path.insert(0, str(_CODE_ROOT / "src"))

from clio_agentic_search.indexing.scientific import (
    build_structure_aware_chunk_plan,
)
from clio_agentic_search.models.contracts import (
    ChunkRecord,
    DocumentRecord,
    EmbeddingRecord,
    MetadataRecord,
)
from clio_agentic_search.retrieval.corpus_profile import build_corpus_profile
from clio_agentic_search.retrieval.scientific import (
    NumericRangeOperator,
    ScientificQueryOperators,
)
from clio_agentic_search.retrieval.strategy import select_branches
from clio_agentic_search.storage.contracts import DocumentBundle, FileIndexState
from clio_agentic_search.storage.duckdb_store import DuckDBStorage

OUT_DIR = _CODE_ROOT.parent / "eval" / "eval_final" / "outputs"


# ---------------------------------------------------------------------------
# Synthetic corpus generation
# ---------------------------------------------------------------------------


DOMAINS = ("temperature", "pressure", "humidity", "wind", "radiation")
DOMAIN_UNITS = {
    "temperature": ("degC", 15.0, 45.0),      # ambient to hot
    "pressure": ("kPa", 95.0, 110.0),         # atmospheric
    "humidity": ("percent", 20.0, 95.0),      # %
    "wind": ("m/s", 0.0, 25.0),
    "radiation": ("MJ", 0.1, 30.0),
}
DOMAIN_SI_DIM = {
    "temperature": "0,0,0,0,1,0,0",
    "pressure": "1,-1,-2,0,0,0,0",
    "humidity": None,  # dimensionless, not in registry
    "wind": "0,1,-1,0,0,0,0",
    "radiation": "1,2,-2,0,0,0,0",  # energy dimension
}


@dataclass(frozen=True, slots=True)
class NamespaceSpec:
    name: str
    ns_type: str  # rich_sci | sparse_sci | pure_text | formula_heavy | empty
    domain: str
    doc_count: int
    should_match_temperature_query: bool
    should_match_pressure_query: bool


def generate_namespace_specs(seed: int = 17) -> list[NamespaceSpec]:
    """Generate a labeled set of 100 synthetic namespace specifications."""
    rng = random.Random(seed)
    specs: list[NamespaceSpec] = []

    # 30 rich_sci: one per domain combination, 6 × 5 domains
    for i in range(30):
        domain = DOMAINS[i % len(DOMAINS)]
        specs.append(NamespaceSpec(
            name=f"ns_rich_{i:03d}",
            ns_type="rich_sci",
            domain=domain,
            doc_count=rng.randint(3, 8),
            should_match_temperature_query=(domain == "temperature"),
            should_match_pressure_query=(domain == "pressure"),
        ))

    # 30 sparse_sci: measurements present but hidden in plain text (no
    # structured extraction at ingest time — sampling can recover them)
    for i in range(30):
        domain = DOMAINS[i % len(DOMAINS)]
        specs.append(NamespaceSpec(
            name=f"ns_sparse_{i:03d}",
            ns_type="sparse_sci",
            domain=domain,
            doc_count=rng.randint(2, 6),
            should_match_temperature_query=(domain == "temperature"),
            should_match_pressure_query=(domain == "pressure"),
        ))

    # 30 pure_text: prose only, no numbers matching the domain units
    for i in range(30):
        domain = DOMAINS[i % len(DOMAINS)]
        specs.append(NamespaceSpec(
            name=f"ns_text_{i:03d}",
            ns_type="pure_text",
            domain=domain,
            doc_count=rng.randint(2, 6),
            should_match_temperature_query=False,
            should_match_pressure_query=False,
        ))

    # 5 formula_heavy: contains equations
    for i in range(5):
        specs.append(NamespaceSpec(
            name=f"ns_formula_{i:03d}",
            ns_type="formula_heavy",
            domain="physics",
            doc_count=rng.randint(2, 5),
            should_match_temperature_query=False,
            should_match_pressure_query=False,
        ))

    # 5 empty: no chunks at all
    for i in range(5):
        specs.append(NamespaceSpec(
            name=f"ns_empty_{i:03d}",
            ns_type="empty",
            domain="none",
            doc_count=0,
            should_match_temperature_query=False,
            should_match_pressure_query=False,
        ))

    return specs


def _generate_rich_sci_text(domain: str, rng: random.Random) -> str:
    """Generate prose containing a clear measurement.

    The regex extractor in indexing.scientific will pick this up when the
    structured chunk pipeline runs, producing real scientific_measurements
    rows in DuckDB.
    """
    unit, lo, hi = DOMAIN_UNITS[domain]
    value = round(rng.uniform(lo, hi), 2)
    return (
        f"# {domain.capitalize()} measurement report\n\n"
        f"The recorded {domain} value was {value} {unit} at 14:00 local time.\n"
        f"This reading was validated against the reference station.\n"
    )


def _generate_sparse_sci_text(domain: str, rng: random.Random) -> str:
    """Generate text with a measurement that will NOT be captured by the
    structured pipeline, but IS recoverable via sampling.

    We achieve this by using an unusual phrasing: the value is present but
    without the tight number-unit proximity the default regex requires.
    The structured pipeline's primary extraction runs at index time on
    build_structure_aware_chunk_plan — which uses _MEASUREMENT_PATTERN.
    By putting the number + unit further apart, structured extraction
    misses it but sampling (which runs the same regex on sampled chunk
    text post-hoc) catches it if we inline a compact form.

    For simplicity we just write the measurement in a less-obvious format
    that is still parseable but is inside a text body without a section
    header — so it survives structured extraction as a single chunk of
    prose with ONE measurement, producing a moderate (not rich) density.
    """
    unit, lo, hi = DOMAIN_UNITS[domain]
    value = round(rng.uniform(lo, hi), 2)
    return (
        f"This is an unstructured narrative discussing various factors. "
        f"The {domain} observation on that particular date was approximately "
        f"{value} {unit}, which was notable given the surrounding conditions. "
        f"Further context about the experimental setup follows in subsequent "
        f"paragraphs but does not contain additional numeric measurements."
    )


def _generate_pure_text(domain: str, rng: random.Random) -> str:
    """Prose with no extractable measurements."""
    return (
        f"This dataset contains narrative descriptions related to {domain} "
        f"research but does not include any structured numeric measurements. "
        f"All content is qualitative discussion, methodology notes, and "
        f"bibliographic references for context. The reader should consult "
        f"the related publications for quantitative data."
    )


def _generate_formula_text(rng: random.Random) -> str:
    """Content heavy in equations."""
    formulas = [
        "E = mc^2",
        "F = ma",
        "PV = nRT",
        "a^2 + b^2 = c^2",
        "dx/dt = -kx",
    ]
    f = rng.choice(formulas)
    return f"# Theoretical derivation\n\nThe governing equation is ${f}$."


def ingest_namespace(
    storage: DuckDBStorage,
    spec: NamespaceSpec,
    rng: random.Random,
) -> None:
    """Ingest the specified number of docs into the namespace using the
    structured chunk pipeline (so rich_sci gets measurements, pure_text
    gets nothing in scientific_measurements, etc.)."""
    if spec.doc_count == 0:
        return

    bundles: list[DocumentBundle] = []
    for doc_idx in range(spec.doc_count):
        if spec.ns_type == "rich_sci":
            text = _generate_rich_sci_text(spec.domain, rng)
        elif spec.ns_type == "sparse_sci":
            text = _generate_sparse_sci_text(spec.domain, rng)
        elif spec.ns_type == "pure_text":
            text = _generate_pure_text(spec.domain, rng)
        elif spec.ns_type == "formula_heavy":
            text = _generate_formula_text(rng)
        else:
            text = "placeholder"

        doc_id = f"{spec.name}_doc_{doc_idx:03d}"
        plan = build_structure_aware_chunk_plan(
            namespace=spec.name,
            document_id=doc_id,
            text=text,
            chunk_size=400,
        )

        # Build metadata records from the chunk plan (captures any measurements
        # the structured extractor found). Also add document-level metadata.
        meta_records: list[MetadataRecord] = [
            MetadataRecord(
                namespace=spec.name, record_id=doc_id,
                scope="document", key="ns_type", value=spec.ns_type,
            ),
            MetadataRecord(
                namespace=spec.name, record_id=doc_id,
                scope="document", key="domain", value=spec.domain,
            ),
        ]
        for chunk_id, chunk_meta in plan.metadata_by_chunk_id.items():
            for k, v in chunk_meta.items():
                meta_records.append(MetadataRecord(
                    namespace=spec.name, record_id=chunk_id,
                    scope="chunk", key=k, value=v,
                ))

        doc = DocumentRecord(
            namespace=spec.name, document_id=doc_id,
            uri=f"synth://{spec.name}/{doc_id}",
            checksum=f"h{doc_idx}", modified_at_ns=1_000_000 + doc_idx,
        )
        file_state = FileIndexState(
            namespace=spec.name,
            path=f"{spec.name}/{doc_id}",
            document_id=doc_id,
            mtime_ns=1_000_000 + doc_idx,
            content_hash=f"h{doc_idx}",
        )

        bundles.append(DocumentBundle(
            document=doc,
            chunks=plan.chunks,
            embeddings=[],
            metadata=meta_records,
            file_state=file_state,
        ))

    storage.upsert_document_bundles(bundles, include_lexical_postings=True)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    query_id: str
    query_text: str
    domain: str
    numeric_range: tuple[float, float, str]
    total_namespaces: int
    # Per-namespace breakdown
    profiled: int
    skipped_scientific: int
    skipped_vector: int
    skipped_lexical: int
    activated_scientific: int
    activated_any: int
    # Routing correctness
    namespaces_that_should_match: int
    namespaces_we_visited_correctly: int
    namespaces_we_skipped_correctly: int
    false_positives: int  # scientific branch activated where it shouldn't
    false_negatives: int  # scientific branch skipped where it should match
    # Performance
    total_profile_time_ms: float
    total_selection_time_ms: float
    # For sampling test
    recovered_via_sampling: int = 0


def run_query(
    storage: DuckDBStorage,
    specs: list[NamespaceSpec],
    query_id: str,
    query_text: str,
    domain: str,
    numeric_range: tuple[float, float, str],
    enable_sampling: bool = False,
) -> QueryResult:
    """Profile every namespace, run select_branches for each, and tally."""
    operators = ScientificQueryOperators(
        numeric_range=NumericRangeOperator(
            minimum=numeric_range[0],
            maximum=numeric_range[1],
            unit=numeric_range[2],
        ),
    )

    total_profile_time = 0.0
    total_selection_time = 0.0
    profiled = 0
    skipped_scientific = 0
    skipped_vector = 0
    skipped_lexical = 0
    activated_scientific = 0
    activated_any = 0
    namespaces_we_visited_correctly = 0
    namespaces_we_skipped_correctly = 0
    false_positives = 0
    false_negatives = 0
    recovered_via_sampling = 0

    target_field = (
        "should_match_temperature_query"
        if domain == "temperature"
        else "should_match_pressure_query"
    )
    namespaces_that_should_match = sum(
        1 for s in specs if getattr(s, target_field)
    )

    for spec in specs:
        t0 = time.perf_counter()
        profile = build_corpus_profile(
            storage, spec.name,
            enable_sampling=enable_sampling,
        )
        total_profile_time += (time.perf_counter() - t0)
        profiled += 1

        if enable_sampling and profile.has_sampled_schema:
            recovered_via_sampling += 1

        t0 = time.perf_counter()
        plan = select_branches(
            query=query_text,
            operators=operators,
            profile=profile,
            connector_has_lexical=True,
            connector_has_vector=True,
            connector_has_graph=False,
            connector_has_scientific=True,
        )
        total_selection_time += (time.perf_counter() - t0)

        if not plan.use_scientific:
            skipped_scientific += 1
        else:
            activated_scientific += 1
        if not plan.use_vector:
            skipped_vector += 1
        if not plan.use_lexical:
            skipped_lexical += 1
        if plan.use_scientific or plan.use_vector or plan.use_lexical:
            activated_any += 1

        # Routing correctness vs ground truth
        should_match = getattr(spec, target_field)
        if should_match and plan.use_scientific:
            namespaces_we_visited_correctly += 1
        elif (not should_match) and (not plan.use_scientific):
            namespaces_we_skipped_correctly += 1
        elif should_match and not plan.use_scientific:
            false_negatives += 1
        else:  # not should_match and plan.use_scientific
            false_positives += 1

    return QueryResult(
        query_id=query_id,
        query_text=query_text,
        domain=domain,
        numeric_range=numeric_range,
        total_namespaces=len(specs),
        profiled=profiled,
        skipped_scientific=skipped_scientific,
        skipped_vector=skipped_vector,
        skipped_lexical=skipped_lexical,
        activated_scientific=activated_scientific,
        activated_any=activated_any,
        namespaces_that_should_match=namespaces_that_should_match,
        namespaces_we_visited_correctly=namespaces_we_visited_correctly,
        namespaces_we_skipped_correctly=namespaces_we_skipped_correctly,
        false_positives=false_positives,
        false_negatives=false_negatives,
        total_profile_time_ms=round(total_profile_time * 1000, 2),
        total_selection_time_ms=round(total_selection_time * 1000, 2),
        recovered_via_sampling=recovered_via_sampling,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _result_to_dict(r: QueryResult) -> dict[str, Any]:
    return {
        "query_id": r.query_id,
        "query_text": r.query_text,
        "domain": r.domain,
        "numeric_range": list(r.numeric_range),
        "total_namespaces": r.total_namespaces,
        "profiled": r.profiled,
        "skipped_scientific": r.skipped_scientific,
        "activated_scientific": r.activated_scientific,
        "skipped_vector": r.skipped_vector,
        "skipped_lexical": r.skipped_lexical,
        "activated_any": r.activated_any,
        "namespaces_that_should_match": r.namespaces_that_should_match,
        "namespaces_we_visited_correctly": r.namespaces_we_visited_correctly,
        "namespaces_we_skipped_correctly": r.namespaces_we_skipped_correctly,
        "false_positives": r.false_positives,
        "false_negatives": r.false_negatives,
        "routing_precision": round(
            r.namespaces_we_visited_correctly
            / max(r.namespaces_we_visited_correctly + r.false_positives, 1),
            3,
        ),
        "routing_recall": round(
            r.namespaces_we_visited_correctly
            / max(r.namespaces_that_should_match, 1),
            3,
        ),
        "total_profile_time_ms": r.total_profile_time_ms,
        "avg_profile_time_per_ns_ms": round(r.total_profile_time_ms / r.total_namespaces, 3),
        "total_selection_time_ms": r.total_selection_time_ms,
        "recovered_via_sampling": r.recovered_via_sampling,
    }


def main() -> None:
    print("=" * 75)
    print("100-NAMESPACE FEDERATION TEST")
    print("=" * 75)

    specs = generate_namespace_specs()
    print(f"\nGenerated {len(specs)} namespace specs:")
    type_counts: dict[str, int] = {}
    for s in specs:
        type_counts[s.ns_type] = type_counts.get(s.ns_type, 0) + 1
    for t, c in sorted(type_counts.items()):
        print(f"  {t:15} → {c}")

    with tempfile.TemporaryDirectory(prefix="clio_fed100_") as tmpdir:
        db_path = Path(tmpdir) / "fed100.duckdb"
        storage = DuckDBStorage(database_path=db_path)
        storage.connect()

        # --- Ingest all 100 namespaces into one DuckDB file ---
        print("\n[1/3] Ingesting 100 namespaces...")
        t0 = time.time()
        rng = random.Random(7)
        for spec in specs:
            ingest_namespace(storage, spec, rng)
        ingest_time = time.time() - t0
        print(f"  Done in {ingest_time:.2f}s")
        print(f"  DB size: {db_path.stat().st_size / 1024:.1f} KB")

        # --- Run queries against all 100 ---
        print("\n[2/3] Running queries across all 100 namespaces...")
        queries = [
            {
                "id": "q_temp",
                "text": "Find temperature measurements above 25 degrees Celsius",
                "domain": "temperature",
                "numeric_range": (25.0, 50.0, "degC"),
            },
            {
                "id": "q_press",
                "text": "Find pressure measurements around 101 kPa",
                "domain": "pressure",
                "numeric_range": (95.0, 110.0, "kPa"),
            },
        ]

        # First pass: sampling DISABLED
        print("\n  --- Without sampling (primary profile only) ---")
        results_no_sampling: list[QueryResult] = []
        for q in queries:
            r = run_query(
                storage, specs,
                query_id=q["id"],
                query_text=q["text"],
                domain=q["domain"],
                numeric_range=q["numeric_range"],
                enable_sampling=False,
            )
            results_no_sampling.append(r)
            print(f"  {r.query_id}: scientific activated on "
                  f"{r.activated_scientific}/{r.total_namespaces} namespaces; "
                  f"profile={r.total_profile_time_ms:.1f}ms total; "
                  f"correct visits={r.namespaces_we_visited_correctly}/"
                  f"{r.namespaces_that_should_match}")

        # Second pass: sampling ENABLED
        print("\n  --- With sampling (recovers sparse_sci) ---")
        results_sampling: list[QueryResult] = []
        for q in queries:
            r = run_query(
                storage, specs,
                query_id=q["id"],
                query_text=q["text"],
                domain=q["domain"],
                numeric_range=q["numeric_range"],
                enable_sampling=True,
            )
            results_sampling.append(r)
            print(f"  {r.query_id}: scientific activated on "
                  f"{r.activated_scientific}/{r.total_namespaces}; "
                  f"recovered={r.recovered_via_sampling}; "
                  f"profile={r.total_profile_time_ms:.1f}ms total; "
                  f"correct visits={r.namespaces_we_visited_correctly}/"
                  f"{r.namespaces_that_should_match}")

        storage.teardown()

    # --- Summary + save ---
    print("\n[3/3] Summary:")
    for label, results in (("WITHOUT sampling", results_no_sampling),
                           ("WITH sampling",    results_sampling)):
        for r in results:
            total_branches_possible = r.total_namespaces * 3  # lex/vec/sci
            total_branches_activated = (
                (r.total_namespaces - r.skipped_lexical)
                + (r.total_namespaces - r.skipped_vector)
                + r.activated_scientific
            )
            branches_saved = total_branches_possible - total_branches_activated
            print(f"\n  [{label}] {r.query_id} ({r.domain}):")
            print(f"    Namespaces: {r.total_namespaces}")
            print(f"    Scientific activated on: {r.activated_scientific}")
            print(f"    Scientific correctly skipped: {r.namespaces_we_skipped_correctly}")
            print(f"    True positives: {r.namespaces_we_visited_correctly}")
            print(f"    False positives: {r.false_positives}")
            print(f"    False negatives: {r.false_negatives}")
            print(f"    Branches saved: {branches_saved}/{total_branches_possible} "
                  f"({100 * branches_saved / total_branches_possible:.1f}%)")
            print(f"    Total profile time: {r.total_profile_time_ms} ms "
                  f"({r.total_profile_time_ms / r.total_namespaces:.2f} ms/ns)")
            if r.recovered_via_sampling > 0:
                print(f"    Recovered via sampling: {r.recovered_via_sampling} namespaces")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "test": "100-namespace federation with per-dataset strategy selection",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "namespace_counts": type_counts,
        "ingest_time_s": round(ingest_time, 2),
        "results_without_sampling": [_result_to_dict(r) for r in results_no_sampling],
        "results_with_sampling":    [_result_to_dict(r) for r in results_sampling],
    }
    out_path = OUT_DIR / "L5_federation_100_namespaces.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
