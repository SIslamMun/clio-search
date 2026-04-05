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

from clio_agentic_search.indexing.scientific import canonicalize_measurement

OUT_DIR = _REPO / "eval" / "eval_final" / "outputs"
NDP_MCP_BINARY = str(_CODE / ".venv" / "bin" / "ndp-mcp")
NDP_URL = "http://155.101.6.191:8003"


# ============================================================================
# Identical prompts used for BOTH modes. No hint or bias toward either.
# ============================================================================

SYSTEM_PROMPT = (
    "You are a scientific data discovery assistant. Answer the user's "
    "question using the tools available to you. Be concise and accurate. "
    "If a numeric or unit-aware filter is needed, apply it correctly."
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
# CLIO helper tools (only exposed in Mode B)
# ============================================================================


@tool(
    name="clio_canonicalize_unit",
    description=(
        "Convert a measurement to canonical SI base units using CLIO's "
        "dimensional-analysis registry. Supports 58 units across 11 SI "
        "dimensions (temperature, pressure, velocity, length, mass, time, "
        "energy, power, etc.). Use this when you need to compare or filter "
        "measurements that may be expressed in different units or prefixes."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "value": {"type": "number"},
            "unit": {
                "type": "string",
                "description": "Unit string: 'degC', 'kPa', 'km/h', 'kelvin', etc.",
            },
        },
        "required": ["value", "unit"],
    },
)
async def clio_canonicalize_unit(args: dict[str, Any]) -> dict[str, Any]:
    try:
        val, dim_key = canonicalize_measurement(float(args["value"]), args["unit"])
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "raw_value": args["value"],
                    "raw_unit": args["unit"],
                    "canonical_value": val,
                    "si_dimension_key": dim_key,
                }),
            }],
        }
    except Exception as e:
        return {"content": [{"type": "text", "text": json.dumps({"error": str(e)})}]}


@tool(
    name="clio_scientific_search",
    description=(
        "Full CLIO pipeline for scientific data discovery: discovers datasets "
        "from NDP, downloads CSV/text resources, extracts measurements with "
        "unit inference from column headers, canonicalises to SI base units, "
        "and filters by threshold. Use this for 'find datasets containing X "
        "with measurement above/below Y' queries. Returns structured results."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "search_term": {"type": "string", "description": "NDP search term"},
            "column_hint": {
                "type": "string",
                "description": "Keyword for target column (e.g. 'temp', 'press', 'wind')",
            },
            "min_value": {"type": "number"},
            "unit": {"type": "string", "description": "Unit for the threshold"},
            "max_resources": {
                "type": "integer",
                "description": "Max CSV resources to process (default 3)",
            },
        },
        "required": ["search_term", "column_hint", "min_value", "unit"],
    },
)
async def clio_scientific_search(args: dict[str, Any]) -> dict[str, Any]:
    search_term = args["search_term"]
    column_hint = args["column_hint"].lower()
    min_value = float(args["min_value"])
    user_unit = args["unit"]
    max_resources = int(args.get("max_resources", 3))

    try:
        with httpx.Client(timeout=30.0) as c:
            resp = c.get(
                f"{NDP_URL}/search",
                params={"terms": search_term, "server": "global"},
            )
            datasets = resp.json() if resp.status_code == 200 else []
    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({"error": f"NDP discovery failed: {e}"}),
            }],
        }

    # Find CSV resources
    csv_urls: list[dict[str, Any]] = []
    for ds in datasets:
        for r in ds.get("resources", []):
            if (r.get("format") or "").upper() == "CSV":
                url = r.get("url", "")
                if url.startswith("http") and "all_stations" not in url and url.endswith(".csv"):
                    csv_urls.append({
                        "url": url,
                        "dataset": ds.get("title", "?"),
                        "resource": r.get("name", "?"),
                    })
                    if len(csv_urls) >= max_resources:
                        break
        if len(csv_urls) >= max_resources:
            break

    try:
        min_canonical, _ = canonicalize_measurement(min_value, user_unit)
    except Exception:
        min_canonical = min_value

    per_resource = []
    total_parsed = 0
    total_matched = 0
    samples: list[dict[str, Any]] = []

    for r in csv_urls:
        try:
            with httpx.Client(timeout=60.0) as c:
                text = c.get(r["url"]).text
            lines = text.splitlines()
            if not lines:
                continue
            header = [h.strip() for h in lines[0].split(",")]
            col_idx = None
            col_unit = None
            for i, col in enumerate(header):
                if column_hint in col.lower() and "id" not in col.lower():
                    m = re.search(r"\(([^)]+)\)", col)
                    if m:
                        col_unit = m.group(1).strip()
                    col_idx = i
                    break
            if col_idx is None or col_unit is None:
                continue
            unit_for_canon = {"C": "degC", "°C": "degC", "F": "degF"}.get(col_unit, col_unit)
            stn_idx = next(
                (i for i, h in enumerate(header) if "stn name" in h.lower()), None,
            )
            parsed_count = 0
            matched_count = 0
            for line in lines[1:5001]:
                cells = line.split(",")
                if len(cells) <= col_idx:
                    continue
                try:
                    raw = float(cells[col_idx].strip())
                    canon, _ = canonicalize_measurement(raw, unit_for_canon)
                except (ValueError, KeyError):
                    continue
                parsed_count += 1
                if canon >= min_canonical:
                    matched_count += 1
                    if len(samples) < 3:
                        stn = (
                            cells[stn_idx].strip()
                            if stn_idx is not None and len(cells) > stn_idx
                            else ""
                        )
                        samples.append({
                            "raw_value": raw,
                            "raw_unit": col_unit,
                            "canonical_K": round(canon, 2),
                            "station": stn,
                            "dataset": r["dataset"][:60],
                        })
            per_resource.append({
                "dataset": r["dataset"][:60],
                "resource": r["resource"][:50],
                "rows_parsed": parsed_count,
                "rows_matched": matched_count,
            })
            total_parsed += parsed_count
            total_matched += matched_count
        except Exception:
            continue

    response = {
        "datasets_discovered": len(datasets),
        "csv_resources_processed": len(per_resource),
        "total_rows_parsed": total_parsed,
        "total_rows_matched": total_matched,
        "canonical_threshold_K": round(min_canonical, 2),
        "sample_matches": samples,
        "per_resource": per_resource,
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
                system_prompt=SYSTEM_PROMPT,
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
        tools=[clio_canonicalize_unit, clio_scientific_search],
    )

    mode_a_results: list[dict[str, Any]] = []
    mode_b_results: list[dict[str, Any]] = []

    for q in QUERIES:
        print(f"\n[{q['id']}] {q['text'][:90]}")

        # Mode A: NDP-MCP only
        print("  Mode A (NDP-MCP only)...")
        a = await run_query(q, {"ndp": ndp_server}, "A_ndp_only")
        print(f"    tools={a['num_tool_calls']} tokens={a['total_tokens']} time={a['time_s']}s")
        mode_a_results.append(a)

        # Mode B: NDP-MCP + CLIO
        print("  Mode B (NDP-MCP + CLIO)...")
        b = await run_query(
            q, {"ndp": ndp_server, "clio": clio_server}, "B_ndp_plus_clio",
        )
        print(f"    tools={b['num_tool_calls']} tokens={b['total_tokens']} time={b['time_s']}s")
        mode_b_results.append(b)

    # Aggregates
    def totals(rs: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "queries": len(rs),
            "total_tool_calls": sum(r["num_tool_calls"] for r in rs),
            "total_tokens": sum(r["total_tokens"] for r in rs),
            "total_time_s": round(sum(r["time_s"] for r in rs), 2),
            "avg_correctness": round(
                sum(r["correctness_keyword_hit_rate"] for r in rs) / len(rs), 3,
            ) if rs else 0.0,
        }

    agg_a = totals(mode_a_results)
    agg_b = totals(mode_b_results)

    print("\n" + "=" * 75)
    print("AGGREGATE")
    print("=" * 75)
    print(f"{'Metric':<28} | {'Mode A':>15} | {'Mode B':>15} | {'Δ':>10}")
    print("-" * 75)
    for k in ("total_tool_calls", "total_tokens", "total_time_s", "avg_correctness"):
        a, b = agg_a[k], agg_b[k]
        delta = b - a
        print(f"{k:<28} | {a:>15} | {b:>15} | {delta:>+10}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "L1: NDP-MCP vs CLIO+NDP-MCP agent test",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_prompt": SYSTEM_PROMPT,
        "queries": QUERIES,
        "mode_a_aggregate": agg_a,
        "mode_b_aggregate": agg_b,
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
