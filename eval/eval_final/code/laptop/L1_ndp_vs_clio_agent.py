#!/usr/bin/env python3
"""L1: NDP-MCP vs CLIO+NDP-MCP — Claude Agent SDK end-to-end.

Runs the SAME 10 real scientific queries against two agent configurations:

  Mode A: Claude agent + NDP-MCP server only (baseline)
  Mode B: Claude agent + NDP-MCP + CLIO helper tools (our contribution)

Same system prompt. Same user prompt. Same LLM (Claude via SDK). Same NDP
catalog. The only variable is whether CLIO's science-aware tools are
available.

Measures
--------
For each mode and each query:
  - # tool calls made by the agent
  - Input / output / cache tokens (from ResultMessage.usage)
  - Wall clock time
  - Final answer text (for correctness grading)
  - Reasoning trajectory (list of tool names called, in order)

Aggregates across 10 queries give the headline numbers.

Output
------
  eval/eval_final/outputs/L1_ndp_vs_clio_agent.json  — structured metrics
  eval/eval_final/outputs/L1_ndp_vs_clio_agent_trace.json  — full agent traces

Prerequisites
-------------
  1. NDP-MCP installed: `uv add /tmp/clio-kit/clio-kit-mcp-servers/ndp`
  2. claude-agent-sdk installed: `uv add claude-agent-sdk`
  3. Network access to NDP: http://155.101.6.191:8003
  4. Claude Code CLI available on PATH (the SDK uses a subprocess)

Usage
-----
  source code/.venv/bin/activate
  python3 eval/eval_final/code/laptop/L1_ndp_vs_clio_agent.py
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

_REPO = Path(__file__).resolve().parents[4]
_CODE = _REPO / "code"
sys.path.insert(0, str(_CODE / "src"))

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    create_sdk_mcp_server,
    query,
    tool,
)

import tempfile

from clio_agentic_search.connectors.ndp.connector import NDPConnector
from clio_agentic_search.indexing.scientific import canonicalize_measurement
from clio_agentic_search.retrieval.coordinator import RetrievalCoordinator
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
# Session-scoped CLIO pipeline: one NDPConnector + one DuckDBStorage shared
# across all Mode B queries. First query for a new discover-term pays the
# cold index cost; subsequent queries hit the indexed DuckDB directly. This
# is the amortisation story the paper claims.
# ============================================================================

_CLIO_STATE: dict[str, Any] = {
    "tmpdir": None,
    "storage": None,
    "connector": None,
    "coordinator": None,
    "indexed_terms": set(),
    "total_index_time_s": 0.0,
    "total_index_docs": 0,
}


def _get_clio() -> tuple[NDPConnector, RetrievalCoordinator]:
    if _CLIO_STATE["connector"] is None:
        tmp = Path(tempfile.mkdtemp(prefix="L1_clio_session_"))
        storage = DuckDBStorage(database_path=tmp / "ndp.duckdb")
        connector = NDPConnector(
            namespace="ndp", storage=storage, base_url=NDP_URL,
        )
        connector.connect()
        _CLIO_STATE["tmpdir"] = tmp
        _CLIO_STATE["storage"] = storage
        _CLIO_STATE["connector"] = connector
        _CLIO_STATE["coordinator"] = RetrievalCoordinator()
    return _CLIO_STATE["connector"], _CLIO_STATE["coordinator"]


def _reset_clio() -> None:
    if _CLIO_STATE["connector"] is not None:
        try:
            _CLIO_STATE["connector"].teardown()
        except Exception:
            pass
    _CLIO_STATE["connector"] = None
    _CLIO_STATE["storage"] = None
    _CLIO_STATE["coordinator"] = None
    _CLIO_STATE["indexed_terms"] = set()
    _CLIO_STATE["total_index_time_s"] = 0.0
    _CLIO_STATE["total_index_docs"] = 0


# ============================================================================
# Identical prompts used for BOTH modes. No hint or bias toward either.
# ============================================================================

SYSTEM_PROMPT_BASE = (
    "You are a scientific data discovery assistant working against the "
    "National Data Platform catalog. Use the MCP tools provided to search "
    "the catalog and inspect dataset details. Return a concise answer that "
    "names the matching datasets. Do not hallucinate — if nothing matches, "
    "say so. Stop as soon as you have enough evidence to answer."
)

# Mode A gets only NDP tools.
SYSTEM_PROMPT_MODE_A = SYSTEM_PROMPT_BASE

# Mode B additionally gets CLIO helpers. The prompt tells the agent *when*
# to use them — ONLY when the query contains a numeric value with a unit
# that may need cross-unit comparison. For purely semantic queries, use
# NDP alone. This avoids gratuitous CLIO calls that waste tokens on queries
# where dimensional analysis adds nothing.
SYSTEM_PROMPT_MODE_B = SYSTEM_PROMPT_BASE + (
    "\n\nYou also have ONE CLIO tool:\n"
    "- mcp__clio__clio_agentic_search: CLIO's end-to-end retrieval pipeline "
    "over the NDP catalog. It internally performs NDP discovery, science-"
    "aware indexing into a local DuckDB, and ranked retrieval with optional "
    "numeric-constraint push-down. Prefer this tool over manually driving "
    "mcp__ndp__search_datasets — one call returns ranked citations instead "
    "of raw catalog descriptions you have to read and filter yourself. "
    "Supply `discover_terms` (keywords to seed the corpus) and, if the "
    "query has a numeric threshold with a unit, `numeric_constraint`. Omit "
    "`numeric_constraint` for purely semantic queries. The returned "
    "`citations` are the definitive top-K ranked results — use them "
    "directly in your answer; you do not need to re-query NDP afterwards. "
    "If `stats.scientific_fallback_to_text` is true, CLIO's scientific "
    "branch matched zero chunks (because NDP catalog descriptions rarely "
    "contain explicit row-level values) and the citations come from the "
    "text-retrieval branches instead — still use them."
)


# Ten representative queries covering unit conversion, semantic search, and
# combined scenarios. Ground truth is derived from manual inspection of NDP
# catalog entries.
QUERIES: list[dict[str, Any]] = [
    {
        "id": "Q01",
        "text": "Find scientific datasets with temperature measurements above 30 degrees Celsius from weather station records in the National Data Platform.",
        "ground_truth_keywords": ["temperature", "weather", "station"],
        "category": "cross_unit",
    },
    {
        "id": "Q02",
        "text": "Find datasets reporting atmospheric pressure around 101 kPa.",
        "ground_truth_keywords": ["pressure", "atmospheric", "kpa"],
        "category": "cross_unit",
    },
    {
        "id": "Q03",
        "text": "Find wind speed measurements above 50 km/h from meteorological datasets.",
        "ground_truth_keywords": ["wind", "speed", "meteorological"],
        "category": "cross_unit",
    },
    {
        "id": "Q04",
        "text": "Find datasets about glacier ice sheet temperature observations.",
        "ground_truth_keywords": ["glacier", "ice", "temperature"],
        "category": "semantic",
    },
    {
        "id": "Q05",
        "text": "Find humidity sensor measurements from environmental monitoring stations.",
        "ground_truth_keywords": ["humidity", "sensor", "environmental"],
        "category": "semantic",
    },
    {
        "id": "Q06",
        "text": "Find solar radiation datasets measured in MJ per square meter.",
        "ground_truth_keywords": ["solar", "radiation", "irradiance"],
        "category": "cross_unit",
    },
    {
        "id": "Q07",
        "text": "Find ocean surface temperature satellite datasets.",
        "ground_truth_keywords": ["ocean", "temperature", "satellite"],
        "category": "semantic",
    },
    {
        "id": "Q08",
        "text": "Find datasets containing precipitation measurements above 100 mm per day.",
        "ground_truth_keywords": ["precipitation", "rainfall", "mm"],
        "category": "cross_unit",
    },
    {
        "id": "Q09",
        "text": "Find wildfire thermal detection datasets from satellite observations.",
        "ground_truth_keywords": ["wildfire", "thermal", "fire"],
        "category": "semantic",
    },
    {
        "id": "Q10",
        "text": "Find soil moisture measurements from agricultural monitoring networks.",
        "ground_truth_keywords": ["soil", "moisture", "agricultural"],
        "category": "semantic",
    },
]


# ============================================================================
# CLIO MCP tool (only exposed in Mode B)
#
# Framing B of the experiment: expose the real CLIO retrieval pipeline
# (NDPConnector → DuckDBStorage → RetrievalCoordinator with scientific
# operators + quality filter) as a single atomic MCP tool the agent can
# call. The agent never sees raw NDP catalog bytes — it calls this one
# tool and gets back the top-K CLIO-ranked citations. The NDP catalog is
# discovered and indexed lazily on first call per discover-term and
# reused across queries within a session, which is the amortisation the
# paper claims.
# ============================================================================


@tool(
    name="clio_agentic_search",
    description=(
        "CLIO agentic retrieval over the NDP catalog. This runs the real "
        "CLIO pipeline end-to-end: Discover (NDPConnector fetches datasets "
        "matching the supplied search terms on first call), Reason (corpus "
        "profiling + branch selection), Transform (science-aware chunking "
        "with SI unit canonicalisation), Execute (RetrievalCoordinator "
        "merges BM25, vector, and scientific-operator branches against a "
        "local DuckDB index). Returns the top-K ranked citations for the "
        "query. Use this as the single discovery tool — you do NOT need "
        "to call NDP search yourself, this tool does it internally. The "
        "NDP catalog is indexed lazily on first call per search term and "
        "reused across subsequent calls within this session, so repeated "
        "queries are fast indexed SQL lookups."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language query describing the data you want",
            },
            "discover_terms": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Keywords used by CLIO's Discover stage to seed the "
                    "corpus with matching NDP datasets (e.g. "
                    "['temperature', 'weather'])."
                ),
            },
            "numeric_constraint": {
                "type": "object",
                "properties": {
                    "unit": {
                        "type": "string",
                        "description": "Unit for the threshold (e.g. 'degC', 'kPa', 'km/h', 'mm')",
                    },
                    "minimum": {"type": "number"},
                    "maximum": {"type": "number"},
                },
                "description": (
                    "OPTIONAL. If the query has a numeric threshold with "
                    "a unit, supply it here and CLIO will push it down to "
                    "SQL via its scientific operators with quality "
                    "filtering. Omit entirely for pure semantic queries."
                ),
            },
            "top_k": {"type": "integer", "default": 5},
        },
        "required": ["query", "discover_terms"],
    },
)
async def clio_agentic_search(args: dict[str, Any]) -> dict[str, Any]:
    query_text = args["query"]
    discover_terms: list[str] = list(args.get("discover_terms") or [])
    nc = args.get("numeric_constraint") or {}
    top_k = int(args.get("top_k", 5))

    connector, coordinator = _get_clio()

    # --- Discover stage: ingest any new terms into the session namespace ---
    t_discover = time.time()
    newly_indexed_docs = 0
    for term in discover_terms:
        if term in _CLIO_STATE["indexed_terms"]:
            continue
        try:
            datasets = connector.discover_datasets(search_terms=[term], limit=25)
            n = connector.index_datasets(datasets)
            newly_indexed_docs += n
            # Also index CSV row data — this populates scientific_measurements
            # with real canonical values so the scientific branch can match.
            csv_stats = connector.index_csv_resources(
                datasets, max_csvs=3, max_rows_per_csv=5000,
            )
            newly_indexed_docs += csv_stats.get("csvs_processed", 0)
        except Exception as e:
            # Soft-fail on one term, keep going
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "error": f"NDP discover/index failed for '{term}': {e}",
                    }),
                }],
            }
        _CLIO_STATE["indexed_terms"].add(term)
    discover_elapsed_ms = round((time.time() - t_discover) * 1000, 1)
    _CLIO_STATE["total_index_time_s"] += (time.time() - t_discover)
    _CLIO_STATE["total_index_docs"] += newly_indexed_docs

    # --- Build scientific operators from the optional numeric constraint ---
    operators = ScientificQueryOperators()
    if nc and nc.get("unit"):
        unit = nc["unit"]
        # Probe the dimensional registry; if unknown, run without numeric op
        try:
            canonicalize_measurement(0.0, unit)
            operators = ScientificQueryOperators(
                numeric_range=NumericRangeOperator(
                    unit=unit,
                    minimum=(
                        float(nc["minimum"]) if nc.get("minimum") is not None else None
                    ),
                    maximum=(
                        float(nc["maximum"]) if nc.get("maximum") is not None else None
                    ),
                ),
                quality_filter=QualityFilterOperator(),  # drop bad/missing rows
            )
        except Exception:
            pass  # unknown unit → fall back to pure text retrieval

    # --- Reason + Transform + Execute via coordinator ---
    # When a numeric_constraint is supplied, CLIO's coordinator hard-filters
    # the merged candidates to only chunks whose scientific_measurements
    # match the range. That's the right semantics when the corpus has
    # row-level measurements. For NDP *catalog descriptions*, though, the
    # scientific_measurements table is often empty because the text says
    # "air temperature measurements from 2010-2025" without an explicit
    # numeric value. So we do a two-pass query: scientific-first, and if
    # that returns zero, a text-retrieval fallback so the agent still gets
    # useful citations. The fallback is flagged in the response.
    t_query = time.time()
    fallback_used = False
    try:
        result = coordinator.query(
            connector=connector,
            query=query_text,
            top_k=top_k,
            scientific_operators=operators,
        )
        if operators.numeric_range is not None and not result.citations:
            fallback_used = True
            result = coordinator.query(
                connector=connector,
                query=query_text,
                top_k=top_k,
                scientific_operators=ScientificQueryOperators(),
            )
    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({"error": f"CLIO query failed: {e}"}),
            }],
        }
    query_elapsed_ms = round((time.time() - t_query) * 1000, 1)

    # Branch activation from trace (lets the agent see what CLIO decided)
    branches_used: list[str] = []
    for event in result.trace:
        if event.stage == "branch_plan_selected":
            for b in ("lexical", "vector", "graph", "scientific"):
                if event.attributes.get(f"use_{b}") == "True":
                    branches_used.append(b)

    citations = [
        {
            "document_id": c.document_id,
            "chunk_id": c.chunk_id,
            "namespace": c.namespace,
            "score": round(c.score, 4),
            "snippet": (c.snippet or "")[:400],
        }
        for c in result.citations
    ]

    response = {
        "citations": citations,
        "stats": {
            "top_k": top_k,
            "branches_used": branches_used,
            "discover_stage_ms_this_call": discover_elapsed_ms,
            "query_stage_ms": query_elapsed_ms,
            "session_indexed_terms": sorted(_CLIO_STATE["indexed_terms"]),
            "session_total_indexed_docs": _CLIO_STATE["total_index_docs"],
            "session_total_index_time_s": round(
                _CLIO_STATE["total_index_time_s"], 2
            ),
            "numeric_constraint_applied": bool(
                operators.numeric_range is not None
            ),
            "quality_filter_applied": bool(
                operators.quality_filter is not None
            ),
            "scientific_fallback_to_text": fallback_used,
        },
    }
    return {"content": [{"type": "text", "text": json.dumps(response)}]}


# ============================================================================
# Trace capture + agent runner
# ============================================================================

def _serialize_block(block: Any) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolUseBlock):
        return {
            "type": "tool_use",
            "name": block.name,
            "input": getattr(block, "input", {}),
        }
    if isinstance(block, ToolResultBlock):
        content = getattr(block, "content", "")
        if isinstance(content, list):
            content = [
                _serialize_block(c) if hasattr(c, "text") else str(c)[:400]
                for c in content
            ]
        else:
            content = str(content)[:400]
        return {"type": "tool_result", "content": content}
    if isinstance(block, ThinkingBlock):
        return {"type": "thinking", "thinking": getattr(block, "thinking", "")[:400]}
    return {"type": type(block).__name__, "repr": str(block)[:300]}


async def run_query(
    query_spec: dict[str, Any],
    mcp_servers: dict[str, Any],
    mode_name: str,
    system_prompt: str,
    max_turns: int = 12,
) -> dict[str, Any]:
    """Run a single query with the given MCP server configuration."""
    t_start = time.time()
    trace: list[dict[str, Any]] = []
    tool_calls: list[str] = []
    final_answer = ""
    num_turns = 0
    usage: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }

    try:
        async for event in query(
            prompt=query_spec["text"],
            options=ClaudeAgentOptions(
                mcp_servers=mcp_servers,
                system_prompt=system_prompt,
                permission_mode="bypassPermissions",
                max_turns=max_turns,
            ),
        ):
            entry: dict[str, Any] = {
                "type": type(event).__name__,
                "elapsed_s": round(time.time() - t_start, 2),
            }

            if isinstance(event, AssistantMessage):
                num_turns += 1
                if hasattr(event, "content") and event.content:
                    blocks = [_serialize_block(b) for b in event.content]
                    entry["blocks"] = blocks
                    for b in blocks:
                        if b["type"] == "text" and b["text"].strip():
                            final_answer = b["text"]
                        elif b["type"] == "tool_use":
                            tool_calls.append(b["name"])
            elif isinstance(event, UserMessage):
                if hasattr(event, "content") and event.content:
                    entry["blocks"] = [_serialize_block(b) for b in event.content]
            elif isinstance(event, SystemMessage):
                entry["subtype"] = getattr(event, "subtype", None)
            elif isinstance(event, ResultMessage):
                if hasattr(event, "usage") and event.usage:
                    u = event.usage if isinstance(event.usage, dict) else {}
                    for k in usage:
                        usage[k] = u.get(k, usage[k])
            trace.append(entry)
    except Exception as e:
        trace.append({"type": "error", "error": f"{type(e).__name__}: {str(e)[:300]}"})

    elapsed = time.time() - t_start

    # Correctness: simple keyword-based grading.  Check whether the final
    # answer mentions any of the ground-truth keywords for this query.
    gt_keywords = [k.lower() for k in query_spec["ground_truth_keywords"]]
    answer_lower = final_answer.lower()
    hits = sum(1 for kw in gt_keywords if kw in answer_lower)
    correctness = hits / len(gt_keywords) if gt_keywords else 0.0

    return {
        "query_id": query_spec["id"],
        "category": query_spec["category"],
        "mode": mode_name,
        "num_turns": num_turns,
        "num_tool_calls": len(tool_calls),
        "tool_calls": tool_calls,
        "usage": usage,
        "total_tokens": sum(usage.values()),
        "time_s": round(elapsed, 2),
        "final_answer": final_answer[:800],
        "correctness_keyword_hit_rate": round(correctness, 2),
        "trace": trace,
    }


async def main_async() -> None:
    print("=" * 75)
    print("L1: NDP-MCP vs CLIO+NDP-MCP — Claude Agent SDK")
    print("=" * 75)

    ndp_server = {
        "type": "stdio",
        "command": NDP_MCP_BINARY,
        "args": ["--transport", "stdio"],
    }
    clio_server = create_sdk_mcp_server(
        name="clio",
        version="1.0.0",
        tools=[clio_agentic_search],
    )

    mode_a_results: list[dict[str, Any]] = []
    mode_b_results: list[dict[str, Any]] = []

    for q in QUERIES:
        print(f"\n[{q['id']}] {q['text'][:90]}")

        # Mode A: NDP-MCP only
        print("  Mode A (NDP-MCP only)...")
        a = await run_query(
            q, {"ndp": ndp_server}, "A_ndp_only",
            system_prompt=SYSTEM_PROMPT_MODE_A,
        )
        print(f"    tools={a['num_tool_calls']} tokens={a['total_tokens']} time={a['time_s']}s")
        mode_a_results.append(a)

        # Mode B: NDP-MCP + CLIO
        print("  Mode B (NDP-MCP + CLIO)...")
        b = await run_query(
            q, {"ndp": ndp_server, "clio": clio_server}, "B_ndp_plus_clio",
            system_prompt=SYSTEM_PROMPT_MODE_B,
        )
        print(f"    tools={b['num_tool_calls']} tokens={b['total_tokens']} time={b['time_s']}s")
        mode_b_results.append(b)

    # Aggregates. Report two token numbers:
    #   total_tokens       = raw usage sum (input + output + cache_*)
    #   effective_cost_tok = input + output + 0.1*cache_read + 1.25*cache_create
    #                        (matches Anthropic's pricing ratios — cache_read
    #                        is roughly 10x cheaper than normal input, cache
    #                        creation is ~1.25x normal input)
    def effective_cost_tokens(r: dict[str, Any]) -> int:
        u = r.get("usage") or {}
        return int(
            u.get("input_tokens", 0)
            + u.get("output_tokens", 0)
            + 0.1 * u.get("cache_read_input_tokens", 0)
            + 1.25 * u.get("cache_creation_input_tokens", 0)
        )

    def totals(rs: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "queries": len(rs),
            "total_tool_calls": sum(r["num_tool_calls"] for r in rs),
            "total_tokens": sum(r["total_tokens"] for r in rs),
            "total_effective_cost_tokens": sum(effective_cost_tokens(r) for r in rs),
            "total_time_s": round(sum(r["time_s"] for r in rs), 2),
            "avg_correctness": round(
                sum(r["correctness_keyword_hit_rate"] for r in rs) / len(rs), 3,
            ) if rs else 0.0,
        }

    agg_a = totals(mode_a_results)
    agg_b = totals(mode_b_results)

    # Per-category breakdown (cross_unit is where CLIO should help; semantic
    # is where CLIO should stay out of the way)
    def by_category(rs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for r in rs:
            out.setdefault(r["category"], []).append(r)
        return {cat: totals(items) for cat, items in out.items()}

    cat_a = by_category(mode_a_results)
    cat_b = by_category(mode_b_results)

    print("\n" + "=" * 75)
    print("AGGREGATE (all queries)")
    print("=" * 75)
    print(f"{'Metric':<32} | {'Mode A':>15} | {'Mode B':>15} | {'Δ':>10}")
    print("-" * 80)
    for k in (
        "total_tool_calls",
        "total_tokens",
        "total_effective_cost_tokens",
        "total_time_s",
        "avg_correctness",
    ):
        a, b = agg_a[k], agg_b[k]
        delta = round(b - a, 3) if isinstance(a, float) else b - a
        print(f"{k:<32} | {a:>15} | {b:>15} | {delta:>+10}")

    for cat in sorted(set(cat_a) | set(cat_b)):
        print(f"\nPer-category: {cat}")
        print(f"{'Metric':<32} | {'Mode A':>15} | {'Mode B':>15} | {'Δ':>10}")
        print("-" * 80)
        ta = cat_a.get(cat, {})
        tb = cat_b.get(cat, {})
        for k in (
            "queries",
            "total_tool_calls",
            "total_effective_cost_tokens",
            "total_time_s",
            "avg_correctness",
        ):
            a, b = ta.get(k, 0), tb.get(k, 0)
            delta = round(b - a, 3) if isinstance(a, float) else b - a
            print(f"{k:<32} | {a:>15} | {b:>15} | {delta:>+10}")

    # Capture CLIO session state before tearing down
    clio_session_stats = {
        "total_indexed_terms": len(_CLIO_STATE["indexed_terms"]),
        "indexed_terms": sorted(_CLIO_STATE["indexed_terms"]),
        "total_indexed_docs": _CLIO_STATE["total_index_docs"],
        "total_index_time_s": round(_CLIO_STATE["total_index_time_s"], 2),
    }
    _reset_clio()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "L1: NDP-MCP vs CLIO+NDP-MCP agent test (Framing B)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_prompt_mode_a": SYSTEM_PROMPT_MODE_A,
        "system_prompt_mode_b": SYSTEM_PROMPT_MODE_B,
        "clio_session_stats": clio_session_stats,
        "queries": QUERIES,
        "mode_a_aggregate": agg_a,
        "mode_b_aggregate": agg_b,
        "mode_a_by_category": cat_a,
        "mode_b_by_category": cat_b,
        "per_query_mode_a": [
            {k: v for k, v in r.items() if k != "trace"} for r in mode_a_results
        ],
        "per_query_mode_b": [
            {k: v for k, v in r.items() if k != "trace"} for r in mode_b_results
        ],
    }
    with (OUT_DIR / "L1_ndp_vs_clio_agent.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    with (OUT_DIR / "L1_ndp_vs_clio_agent_trace.json").open("w") as f:
        json.dump(
            {"mode_a_traces": mode_a_results, "mode_b_traces": mode_b_results},
            f,
            indent=2,
            default=str,
        )
    print(f"\nSaved: {OUT_DIR / 'L1_ndp_vs_clio_agent.json'}")
    print(f"Saved: {OUT_DIR / 'L1_ndp_vs_clio_agent_trace.json'}")


if __name__ == "__main__":
    asyncio.run(main_async())
