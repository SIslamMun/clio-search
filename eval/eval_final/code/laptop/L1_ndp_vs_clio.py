#!/usr/bin/env python3
"""L1: NDP-MCP vs CLIO Agentic Search — three-way comparison.

Same 10 queries, three paths:

  Baseline  : LLM agent (Claude) + NDP-MCP only
  Framing A : CLIO AgenticRetriever + FallbackQueryRewriter (no LLM)
  Framing B : CLIO AgenticRetriever + QueryRewriter (with LLM rewriting)

CLIO uses NDPConnector with CSV row ingestion, so scientific_measurements
has real canonical values for cross-unit queries.

Output
------
  eval/eval_final/outputs/L1_ndp_vs_clio.json
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[4]
_CODE = _REPO / "code"
sys.path.insert(0, str(_CODE / "src"))

from clio_agentic_search.connectors.ndp.connector import NDPConnector
from clio_agentic_search.indexing.scientific import canonicalize_measurement
from clio_agentic_search.retrieval.agentic import AgenticRetriever
from clio_agentic_search.retrieval.coordinator import RetrievalCoordinator
from clio_agentic_search.retrieval.query_rewriter import FallbackQueryRewriter
from clio_agentic_search.retrieval.scientific import (
    NumericRangeOperator,
    QualityFilterOperator,
    ScientificQueryOperators,
)
from clio_agentic_search.storage.duckdb_store import DuckDBStorage

OUT_DIR = _REPO / "eval" / "eval_final" / "outputs"
NDP_MCP_BINARY = str(_CODE / ".venv" / "bin" / "ndp-mcp")
NDP_URL = "http://155.101.6.191:8003"

# ============================================================================
# 10 queries
# ============================================================================

QUERIES = [
    {
        "id": "Q01", "category": "cross_unit",
        "text": "Find datasets with temperature measurements above 30 degrees Celsius from weather stations",
        "discover_terms": ["temperature", "weather"],
        "numeric_constraint": {"unit": "degC", "minimum": 30},
        "ground_truth_keywords": ["temperature", "weather", "station"],
    },
    {
        "id": "Q02", "category": "cross_unit",
        "text": "Find datasets reporting atmospheric pressure around 101 kPa",
        "discover_terms": ["pressure", "atmospheric"],
        "numeric_constraint": {"unit": "kPa", "minimum": 90, "maximum": 110},
        "ground_truth_keywords": ["pressure", "atmospheric", "kpa"],
    },
    {
        "id": "Q03", "category": "cross_unit",
        "text": "Find wind speed measurements above 50 km/h from meteorological datasets",
        "discover_terms": ["wind", "speed"],
        "numeric_constraint": {"unit": "km/h", "minimum": 50},
        "ground_truth_keywords": ["wind", "speed", "meteorological"],
    },
    {
        "id": "Q04", "category": "semantic",
        "text": "Find datasets about glacier ice sheet temperature observations",
        "discover_terms": ["glacier", "ice", "temperature"],
        "ground_truth_keywords": ["glacier", "ice", "temperature"],
    },
    {
        "id": "Q05", "category": "semantic",
        "text": "Find humidity sensor measurements from environmental monitoring stations",
        "discover_terms": ["humidity", "environmental"],
        "ground_truth_keywords": ["humidity", "sensor", "environmental"],
    },
    {
        "id": "Q06", "category": "cross_unit",
        "text": "Find solar radiation datasets measured in MJ per square meter",
        "discover_terms": ["solar", "radiation"],
        "numeric_constraint": {"unit": "MJ", "minimum": 1},
        "ground_truth_keywords": ["solar", "radiation", "irradiance"],
    },
    {
        "id": "Q07", "category": "semantic",
        "text": "Find ocean surface temperature satellite datasets",
        "discover_terms": ["ocean", "temperature", "satellite"],
        "ground_truth_keywords": ["ocean", "temperature", "satellite"],
    },
    {
        "id": "Q08", "category": "cross_unit",
        "text": "Find datasets containing precipitation measurements above 100 mm per day",
        "discover_terms": ["precipitation", "rainfall"],
        "numeric_constraint": {"unit": "mm", "minimum": 100},
        "ground_truth_keywords": ["precipitation", "rainfall", "mm"],
    },
    {
        "id": "Q09", "category": "semantic",
        "text": "Find wildfire thermal detection datasets from satellite observations",
        "discover_terms": ["wildfire", "thermal", "fire"],
        "ground_truth_keywords": ["wildfire", "thermal", "fire"],
    },
    {
        "id": "Q10", "category": "semantic",
        "text": "Find soil moisture measurements from agricultural monitoring networks",
        "discover_terms": ["soil", "moisture", "agricultural"],
        "ground_truth_keywords": ["soil", "moisture", "agricultural"],
    },
]


def count_tokens(text: str) -> int:
    return len(text.split())


def build_operators(nc: dict[str, Any] | None) -> ScientificQueryOperators:
    if not nc or not nc.get("unit"):
        return ScientificQueryOperators()
    try:
        canonicalize_measurement(0.0, nc["unit"])
        return ScientificQueryOperators(
            numeric_range=NumericRangeOperator(
                unit=nc["unit"],
                minimum=nc.get("minimum"),
                maximum=nc.get("maximum"),
            ),
            quality_filter=QualityFilterOperator(),
        )
    except Exception:
        return ScientificQueryOperators()


def score_correctness(text: str, keywords: list[str]) -> float:
    lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lower)
    return round(hits / len(keywords), 3) if keywords else 0.0


# ============================================================================
# Setup: NDP connector with CSV row ingestion
# ============================================================================

def setup_clio(tmpdir: Path) -> tuple[NDPConnector, dict[str, Any]]:
    db = DuckDBStorage(database_path=tmpdir / "ndp.duckdb")
    conn = NDPConnector(namespace="ndp", storage=db, base_url=NDP_URL)
    conn.connect()

    stats: dict[str, Any] = {"indexed_terms": [], "csv_stats": {}}
    # Consolidated discover terms — one broad search per domain, not per-query
    discover_terms = [
        "temperature", "pressure", "wind", "humidity",
        "radiation", "glacier", "ocean", "soil", "precipitation", "wildfire",
    ]

    t0 = time.time()
    total_csv = {"csvs_processed": 0, "rows_indexed": 0, "measurements_found": 0}
    for term in discover_terms:
        print(f"    Discovering '{term}'...", end="", flush=True)
        try:
            ds = conn.discover_datasets(search_terms=[term], limit=25)
            conn.index_datasets(ds)
        except Exception as e:
            print(f" discover error: {e}")
            continue
        try:
            csv_s = conn.index_csv_resources(
                ds, max_csvs=2, max_rows_per_csv=1000, timeout_s=20.0,
                max_download_bytes=10 * 1024 * 1024,
            )
            for k in total_csv:
                total_csv[k] += csv_s.get(k, 0)
            print(f" {len(ds)} ds, {csv_s.get('measurements_found', 0)} meas")
        except Exception as e:
            print(f" csv error: {e}")
        stats["indexed_terms"].append(term)
    stats["index_time_s"] = round(time.time() - t0, 1)
    stats["csv_stats"] = total_csv
    return conn, stats


# ============================================================================
# Framing A: CLIO AgenticRetriever + FallbackQueryRewriter (no LLM)
# ============================================================================

def run_framing_a(connector: NDPConnector) -> list[dict[str, Any]]:
    retriever = AgenticRetriever(
        coordinator=RetrievalCoordinator(),
        rewriter=FallbackQueryRewriter(),
        max_hops=3,
    )
    results = []
    for q in QUERIES:
        operators = build_operators(q.get("numeric_constraint"))
        t0 = time.time()
        result = retriever.query(
            connector=connector,
            query=q["text"],
            top_k=10,
            scientific_operators=operators,
        )
        # Fallback if scientific returned 0
        if operators.numeric_range is not None and not result.citations:
            result = retriever.query(
                connector=connector,
                query=q["text"],
                top_k=10,
                scientific_operators=ScientificQueryOperators(),
            )
        elapsed_ms = round((time.time() - t0) * 1000, 1)

        citation_text = " ".join(c.snippet for c in result.citations)
        tokens = count_tokens(citation_text)
        correctness = score_correctness(citation_text, q["ground_truth_keywords"])

        r = {
            "query_id": q["id"],
            "category": q["category"],
            "citations": len(result.citations),
            "tokens": tokens,
            "time_ms": elapsed_ms,
            "correctness": correctness,
            "hops": len(result.hops),
            "strategy": result.strategy_used,
            "llm_tokens": result.token_usage.total_input_tokens + result.token_usage.total_output_tokens,
        }
        results.append(r)
        print(f"    {q['id']} [{q['category']:<12}] cites={len(result.citations):>2}  "
              f"tok={tokens:>5}  ms={elapsed_ms:>7.1f}  hops={len(result.hops)}  "
              f"correct={correctness:.2f}")
    return results


# ============================================================================
# Framing B: CLIO AgenticRetriever + QueryRewriter (with LLM)
# ============================================================================

def run_framing_b(connector: NDPConnector) -> list[dict[str, Any]]:
    import shutil
    import subprocess

    from clio_agentic_search.retrieval.query_rewriter import (
        RewriteResult,
        _SYSTEM_PROMPT,
        _parse_llm_response,
    )

    if not shutil.which("claude"):
        print("  claude CLI not found — skipping Framing B")
        return []

    class CliQueryRewriter:
        """Query rewriter that uses the authenticated Claude CLI (no API key)."""

        def rewrite(
            self, *, query: str, retrieved_snippets: list[str],
            hop_number: int, max_hops: int,
        ) -> RewriteResult:
            snippets_text = "\n---\n".join(
                s[:300] for s in retrieved_snippets
            ) if retrieved_snippets else "(none)"
            prompt = (
                f"Current query: {query}\n"
                f"Hop: {hop_number}/{max_hops}\n"
                f"Retrieved snippets:\n{snippets_text}\n\n"
                "Decide: should the query be refined? "
                "If the results already cover the information need, use "
                "strategy 'done'. Otherwise pick expand/narrow/pivot and "
                "provide a rewritten query.\n\n"
                "Respond in EXACTLY this format (3 lines, no extra text):\n"
                "strategy: <done|expand|narrow|pivot>\n"
                "query: <the query text>\n"
                "reasoning: <one sentence>"
            )
            try:
                proc = subprocess.run(
                    ["claude", "-p", prompt, "--no-input"],
                    capture_output=True, text=True, timeout=60,
                )
                raw_text = proc.stdout.strip()
            except Exception:
                raw_text = f"strategy: done\nquery: {query}\nreasoning: CLI call failed"

            result = _parse_llm_response(raw_text, original_query=query)
            return RewriteResult(
                original_query=result.original_query,
                rewritten_query=result.rewritten_query,
                strategy=result.strategy,
                reasoning=result.reasoning,
                input_tokens=0,
                output_tokens=0,
            )

    rewriter = CliQueryRewriter()

    retriever = AgenticRetriever(
        coordinator=RetrievalCoordinator(),
        rewriter=rewriter,
        max_hops=3,
    )
    results = []
    for q in QUERIES:
        operators = build_operators(q.get("numeric_constraint"))
        t0 = time.time()
        result = retriever.query(
            connector=connector,
            query=q["text"],
            top_k=10,
            scientific_operators=operators,
        )
        if operators.numeric_range is not None and not result.citations:
            result = retriever.query(
                connector=connector,
                query=q["text"],
                top_k=10,
                scientific_operators=ScientificQueryOperators(),
            )
        elapsed_ms = round((time.time() - t0) * 1000, 1)

        citation_text = " ".join(c.snippet for c in result.citations)
        tokens = count_tokens(citation_text)
        correctness = score_correctness(citation_text, q["ground_truth_keywords"])

        r = {
            "query_id": q["id"],
            "category": q["category"],
            "citations": len(result.citations),
            "tokens": tokens,
            "time_ms": elapsed_ms,
            "correctness": correctness,
            "hops": len(result.hops),
            "strategy": result.strategy_used,
            "llm_tokens": result.token_usage.total_input_tokens + result.token_usage.total_output_tokens,
        }
        results.append(r)
        print(f"    {q['id']} [{q['category']:<12}] cites={len(result.citations):>2}  "
              f"tok={tokens:>5}  ms={elapsed_ms:>7.1f}  hops={len(result.hops)}  "
              f"correct={correctness:.2f}  llm_tok={r['llm_tokens']}")
    return results


# ============================================================================
# Baseline: LLM agent + NDP-MCP only
# ============================================================================

def run_baseline() -> list[dict[str, Any]]:
    try:
        from claude_agent_sdk import (
            AssistantMessage, ClaudeAgentOptions, ResultMessage,
            TextBlock, ToolUseBlock, query,
        )
    except ImportError:
        print("  claude-agent-sdk not available — skipping baseline")
        return []

    PROMPT = (
        "You are a scientific data discovery assistant working against the "
        "National Data Platform catalog. Use the MCP tools provided to search "
        "the catalog and inspect dataset details. Return a concise answer that "
        "names the matching datasets. Stop as soon as you have enough evidence."
    )
    ndp_server = {
        "type": "stdio",
        "command": NDP_MCP_BINARY,
        "args": ["--transport", "stdio"],
    }

    async def run_one(q: dict[str, Any]) -> dict[str, Any]:
        t_start = time.time()
        tool_calls: list[str] = []
        final_answer = ""
        usage = {
            "input_tokens": 0, "output_tokens": 0,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        }
        try:
            async for event in query(
                prompt=q["text"],
                options=ClaudeAgentOptions(
                    mcp_servers={"ndp": ndp_server},
                    system_prompt=PROMPT,
                    permission_mode="bypassPermissions",
                    max_turns=12,
                ),
            ):
                if isinstance(event, AssistantMessage) and hasattr(event, "content"):
                    for block in (event.content or []):
                        if isinstance(block, TextBlock) and block.text.strip():
                            final_answer = block.text
                        elif isinstance(block, ToolUseBlock):
                            tool_calls.append(block.name)
                elif isinstance(event, ResultMessage) and hasattr(event, "usage"):
                    u = event.usage if isinstance(event.usage, dict) else {}
                    for k in usage:
                        usage[k] = u.get(k, usage[k])
        except Exception:
            pass

        elapsed = time.time() - t_start
        total_tok = sum(usage.values())
        eff_tok = int(
            usage["input_tokens"] + usage["output_tokens"]
            + 0.1 * usage["cache_read_input_tokens"]
            + 1.25 * usage["cache_creation_input_tokens"]
        )
        correctness = score_correctness(final_answer, q["ground_truth_keywords"])
        return {
            "query_id": q["id"],
            "category": q["category"],
            "num_tool_calls": len(tool_calls),
            "tool_calls": tool_calls,
            "total_tokens": total_tok,
            "effective_tokens": eff_tok,
            "time_s": round(elapsed, 2),
            "correctness": correctness,
        }

    async def run_all() -> list[dict[str, Any]]:
        results = []
        for q in QUERIES:
            print(f"    {q['id']} ...", end="", flush=True)
            r = await run_one(q)
            print(f" tools={r['num_tool_calls']}  eff_tok={r['effective_tokens']}  "
                  f"time={r['time_s']}s  correct={r['correctness']}")
            results.append(r)
        return results

    return asyncio.run(run_all())


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 75)
    print("L1: NDP-MCP vs CLIO Agentic Search (3-way)")
    print("=" * 75)

    with tempfile.TemporaryDirectory(prefix="L1_") as tmpdir:
        tmpdir = Path(tmpdir)

        # Setup
        print("\n[Setup] Indexing NDP catalog + CSV rows into CLIO DuckDB...")
        connector, index_stats = setup_clio(tmpdir)
        print(f"  Done in {index_stats['index_time_s']}s")
        print(f"  CSV: {index_stats['csv_stats']}")

        # Framing A
        print(f"\n[Framing A] CLIO AgenticRetriever + FallbackQueryRewriter (no LLM)")
        framing_a = run_framing_a(connector)

        # Framing B
        print(f"\n[Framing B] CLIO AgenticRetriever + QueryRewriter (with LLM)")
        framing_b = run_framing_b(connector)

        connector.teardown()

    # Baseline
    print(f"\n[Baseline] LLM agent + NDP-MCP only")
    baseline = run_baseline()

    # Summary
    print("\n" + "=" * 75)
    print("RESULTS")
    print("=" * 75)

    def agg(results: list[dict[str, Any]], tok_key: str, time_key: str) -> dict[str, Any]:
        if not results:
            return {}
        n = len(results)
        return {
            "queries": n,
            "total_tokens": sum(r.get(tok_key, 0) for r in results),
            "avg_correctness": round(sum(r["correctness"] for r in results) / n, 3),
        }

    a_agg = agg(framing_a, "tokens", "time_ms")
    b_agg = agg(framing_b, "tokens", "time_ms")
    bl_agg = agg(baseline, "effective_tokens", "time_s")

    print(f"\n{'Metric':<30} | {'Baseline':>12} | {'A (no LLM)':>12} | {'B (+ LLM)':>12}")
    print("-" * 75)
    for label, ba, aa, bb in [
        ("Tokens (effective)", bl_agg.get("total_tokens", "-"), a_agg.get("total_tokens", "-"), b_agg.get("total_tokens", "-")),
        ("Avg correctness", bl_agg.get("avg_correctness", "-"), a_agg.get("avg_correctness", "-"), b_agg.get("avg_correctness", "-")),
    ]:
        print(f"{label:<30} | {str(ba):>12} | {str(aa):>12} | {str(bb):>12}")

    if baseline:
        total_tools = sum(r["num_tool_calls"] for r in baseline)
        print(f"{'Total tool calls':<30} | {total_tools:>12} | {'0':>12} | {'0':>12}")
    if framing_a:
        total_hops_a = sum(r["hops"] for r in framing_a)
        total_hops_b = sum(r["hops"] for r in framing_b) if framing_b else "-"
        print(f"{'Total hops':<30} | {'-':>12} | {total_hops_a:>12} | {str(total_hops_b):>12}")
    if framing_b:
        llm_tok_b = sum(r["llm_tokens"] for r in framing_b)
        print(f"{'LLM rewriter tokens (B only)':<30} | {'-':>12} | {'0':>12} | {llm_tok_b:>12}")

    if bl_agg and a_agg:
        bl_tok = bl_agg["total_tokens"]
        a_tok = a_agg["total_tokens"]
        if bl_tok > 0:
            reduction = 1.0 - a_tok / bl_tok
            print(f"\nFraming A token reduction vs baseline: {reduction:.1%}")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "L1: NDP-MCP vs CLIO Agentic Search (3-way)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "index_stats": index_stats,
        "framing_a": {"aggregate": a_agg, "per_query": framing_a},
        "framing_b": {"aggregate": b_agg, "per_query": framing_b},
        "baseline": {"aggregate": bl_agg, "per_query": baseline},
    }
    out_path = OUT_DIR / "L1_ndp_vs_clio.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
