#!/usr/bin/env python3
"""FAIR end-to-end comparison: CLIO Search vs naked MCP.

Same Claude Agent SDK. Same system prompt. Same user prompt. Same NDP-MCP server.
Same Bash tool. The only difference: Mode B additionally has CLIO helper tools.

Mode A tools: NDP-MCP (search_datasets, list_organizations, get_dataset_details) + Bash
Mode B tools: Mode A + CLIO tools (canonicalize_unit, parse_csv, search_ndp)

Captures: full trace, tool calls, tokens, time, workflow, reasoning, answer.
Output: eval/eval_final/outputs/
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

_CODE_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "code"
sys.path.insert(0, str(_CODE_ROOT / "src"))

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

OUT_DIR = _CODE_ROOT.parent / "eval" / "eval_final" / "outputs"
NDP_MCP_BINARY = str(_CODE_ROOT / ".venv" / "bin" / "ndp-mcp")

# ===================================================================
# The identical prompts for BOTH modes
# ===================================================================

SYSTEM_PROMPT = (
    "You are a scientific data discovery agent. Use the available tools to "
    "answer the user's question. Think step by step about what you need."
)

USER_PROMPT = (
    "Find temperature measurements above 30 degrees Celsius from weather "
    "station datasets in the National Data Platform. Tell me how many rows "
    "match this criterion and give me 3 example values with the station name "
    "or dataset name if possible."
)


# ===================================================================
# CLIO helper tools (only exposed in Mode B)
# ===================================================================

@tool(
    name="clio_canonicalize_unit",
    description=(
        "Convert a measurement to canonical SI base units using CLIO's "
        "dimensional analysis registry. Supports 46+ units across 12 physical "
        "domains (temperature: degC/degF/K/celsius/fahrenheit; pressure: "
        "Pa/kPa/MPa/hPa/bar/atm/psi; velocity: m/s/km/h/kn; length, mass, "
        "time, energy, power, etc). Returns canonical value and SI dimension key. "
        "Use this when you need to compare or filter measurements that may be "
        "in different units."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "value": {"type": "number", "description": "Numeric value"},
            "unit": {"type": "string", "description": "Unit string (e.g. 'degC', 'kPa', 'km/h')"},
        },
        "required": ["value", "unit"],
    },
)
async def clio_canonicalize_unit(args: dict[str, Any]) -> dict[str, Any]:
    try:
        canonical_value, dim_key = canonicalize_measurement(
            float(args["value"]), args["unit"],
        )
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "raw_value": args["value"],
                    "raw_unit": args["unit"],
                    "canonical_value": canonical_value,
                    "si_dimension_key": dim_key,
                }),
            }],
        }
    except Exception as e:
        return {"content": [{"type": "text", "text": json.dumps({"error": str(e)})}]}


@tool(
    name="clio_parse_csv",
    description=(
        "Download a CSV file, find a column matching the hint, parse values, "
        "infer the unit from the column header (e.g. 'Air Temp (C)' → Celsius), "
        "canonicalize all values to SI base, filter by threshold, return count "
        "and samples. This is CLIO's science-aware CSV connector. Use it when "
        "you have a CSV URL and want to filter rows by a numeric threshold "
        "with unit conversion handled automatically."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "column_hint": {"type": "string", "description": "Keyword to find the column (e.g. 'temp')"},
            "min_value": {"type": "number", "description": "Threshold value in user's unit"},
            "unit": {"type": "string", "description": "User's unit (e.g. 'degC')"},
            "max_rows": {"type": "integer", "description": "Max rows to parse (default 5000)"},
        },
        "required": ["url", "column_hint", "min_value", "unit"],
    },
)
async def clio_parse_csv(args: dict[str, Any]) -> dict[str, Any]:
    url = args["url"]
    column_hint = args["column_hint"].lower()
    min_value = float(args["min_value"])
    user_unit = args["unit"]
    max_rows = int(args.get("max_rows", 5000))

    try:
        with httpx.Client(timeout=60.0) as c:
            resp = c.get(url)
            resp.raise_for_status()
        lines = resp.text.splitlines()
        if not lines:
            return {"content": [{"type": "text", "text": '{"error":"empty file"}'}]}

        header = [h.strip() for h in lines[0].split(",")]

        col_idx = None
        col_unit = None
        for i, col in enumerate(header):
            col_l = col.lower()
            if column_hint in col_l and "id" not in col_l and "station" not in col_l:
                m = re.search(r"\(([^)]+)\)", col)
                if m:
                    col_unit = m.group(1).strip()
                col_idx = i
                break

        if col_idx is None:
            return {"content": [{"type": "text", "text": json.dumps({
                "error": f"no column matching '{column_hint}' found",
                "header": header,
            })}]}

        unit_for_canon = {"C": "degC", "°C": "degC", "F": "degF", "°F": "degF", "K": "kelvin"}.get(
            col_unit or "", col_unit or user_unit,
        )

        try:
            min_canonical, _ = canonicalize_measurement(min_value, user_unit)
        except Exception as e:
            return {"content": [{"type": "text", "text": json.dumps({"error": f"bad user unit: {e}"})}]}

        station_idx = next(
            (i for i, c in enumerate(header) if "stn name" in c.lower() or "station name" in c.lower()),
            None,
        )

        parsed = 0
        matched = 0
        samples: list[dict[str, Any]] = []
        for line in lines[1 : max_rows + 1]:
            cells = line.split(",")
            if len(cells) <= col_idx:
                continue
            try:
                raw = float(cells[col_idx].strip())
                canon, _ = canonicalize_measurement(raw, unit_for_canon)
            except (ValueError, KeyError):
                continue
            parsed += 1
            if canon >= min_canonical:
                matched += 1
                if len(samples) < 3:
                    station = cells[station_idx].strip() if station_idx is not None and len(cells) > station_idx else ""
                    samples.append({
                        "raw_value": raw,
                        "raw_unit": col_unit,
                        "canonical_K": round(canon, 2),
                        "station": station,
                    })

        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "url": url,
                    "column_found": header[col_idx],
                    "column_unit_raw": col_unit,
                    "canonical_unit_used": unit_for_canon,
                    "min_value_canonical": round(min_canonical, 2),
                    "rows_parsed": parsed,
                    "rows_matched": matched,
                    "sample_matches": samples,
                }),
            }],
        }
    except Exception as e:
        return {"content": [{"type": "text", "text": json.dumps({"error": str(e)})}]}


@tool(
    name="clio_search_ndp",
    description=(
        "Run the full CLIO pipeline on NDP: discover datasets, find parseable "
        "CSV/text resources, download them, parse columns with unit inference, "
        "canonicalize to SI, filter by threshold. This is the one-shot tool "
        "that combines all CLIO stages: Discover → Reason → Transform → Execute. "
        "Use when you want 'find X in NDP' answered in a single call."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "search_term": {"type": "string", "description": "NDP search term (e.g. 'temperature')"},
            "column_hint": {"type": "string", "description": "Column keyword (e.g. 'temp')"},
            "min_value": {"type": "number"},
            "unit": {"type": "string"},
            "max_resources": {"type": "integer", "description": "Max CSVs to process (default 3)"},
        },
        "required": ["search_term", "column_hint", "min_value", "unit"],
    },
)
async def clio_search_ndp(args: dict[str, Any]) -> dict[str, Any]:
    search_term = args["search_term"]
    column_hint = args["column_hint"].lower()
    min_value = float(args["min_value"])
    user_unit = args["unit"]
    max_resources = int(args.get("max_resources", 3))

    NDP_URL = "http://155.101.6.191:8003"

    # Stage 1: discover
    try:
        with httpx.Client(timeout=30.0) as c:
            resp = c.get(f"{NDP_URL}/search",
                         params={"terms": search_term, "server": "global"})
            datasets = resp.json() if resp.status_code == 200 else []
    except Exception as e:
        return {"content": [{"type": "text", "text": json.dumps({"error": f"discover failed: {e}"})}]}

    # Stage 2: find parseable resources
    csv_urls = []
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

    # Stage 3+4: parse + canonicalize + filter each
    min_canonical, _ = canonicalize_measurement(min_value, user_unit)
    per_resource: list[dict[str, Any]] = []
    total_parsed = 0
    total_matched = 0
    all_samples: list[dict[str, Any]] = []

    for r in csv_urls:
        try:
            with httpx.Client(timeout=60.0) as c:
                resp = c.get(r["url"])
                resp.raise_for_status()
            lines = resp.text.splitlines()
        except Exception:
            continue

        if not lines:
            continue
        header = [h.strip() for h in lines[0].split(",")]

        col_idx = None
        col_unit = None
        for i, col in enumerate(header):
            col_l = col.lower()
            if column_hint in col_l and "id" not in col_l and "station" not in col_l:
                m = re.search(r"\(([^)]+)\)", col)
                if m:
                    col_unit = m.group(1).strip()
                col_idx = i
                break

        if col_idx is None or not col_unit:
            continue

        unit_for_canon = {"C": "degC", "°C": "degC", "F": "degF", "°F": "degF", "K": "kelvin"}.get(
            col_unit, col_unit,
        )

        station_idx = next(
            (i for i, c in enumerate(header) if "stn name" in c.lower()),
            None,
        )

        parsed = 0
        matched = 0
        for line in lines[1:5001]:
            cells = line.split(",")
            if len(cells) <= col_idx:
                continue
            try:
                raw = float(cells[col_idx].strip())
                canon, _ = canonicalize_measurement(raw, unit_for_canon)
            except (ValueError, KeyError):
                continue
            parsed += 1
            if canon >= min_canonical:
                matched += 1
                if len(all_samples) < 3:
                    station = cells[station_idx].strip() if station_idx is not None and len(cells) > station_idx else ""
                    all_samples.append({
                        "raw_value": raw,
                        "raw_unit": col_unit,
                        "canonical_K": round(canon, 2),
                        "station": station,
                        "dataset": r["dataset"][:60],
                    })

        per_resource.append({
            "dataset": r["dataset"][:60],
            "resource": r["resource"][:50],
            "column_unit": col_unit,
            "rows_parsed": parsed,
            "rows_matched": matched,
        })
        total_parsed += parsed
        total_matched += matched

    return {
        "content": [{
            "type": "text",
            "text": json.dumps({
                "datasets_discovered": len(datasets),
                "csv_resources_processed": len(per_resource),
                "total_rows_parsed": total_parsed,
                "total_rows_matched": total_matched,
                "canonical_threshold_K": round(min_canonical, 2),
                "sample_matches": all_samples,
                "per_resource": per_resource,
            }),
        }],
    }


# ===================================================================
# Trace capture & runner
# ===================================================================

def serialize_block(block: Any) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    elif isinstance(block, ToolUseBlock):
        return {
            "type": "tool_use",
            "name": block.name,
            "input": getattr(block, "input", {}),
        }
    elif isinstance(block, ToolResultBlock):
        content = getattr(block, "content", "")
        if isinstance(content, list):
            content = [serialize_block(c) if hasattr(c, "text") else str(c)[:500] for c in content]
        else:
            content = str(content)[:500]
        return {"type": "tool_result", "content": content, "is_error": getattr(block, "is_error", False)}
    elif isinstance(block, ThinkingBlock):
        return {"type": "thinking", "thinking": getattr(block, "thinking", "")[:500]}
    else:
        return {"type": type(block).__name__, "repr": str(block)[:300]}


async def run_agent(
    mode_name: str,
    mcp_servers: dict[str, Any],
    max_turns: int = 15,
) -> dict[str, Any]:
    print(f"\n{'='*75}")
    print(f"RUNNING: {mode_name}")
    print('='*75)
    print(f"\n[USER PROMPT]\n{USER_PROMPT}")
    print(f"\n[SYSTEM PROMPT]\n{SYSTEM_PROMPT}\n")

    t0 = time.time()
    trace: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    tool_call_names: list[str] = []
    input_tokens = 0
    output_tokens = 0
    cache_creation = 0
    cache_read = 0
    final_answer = ""
    num_turns = 0
    step = 0

    try:
        async for event in query(
            prompt=USER_PROMPT,
            options=ClaudeAgentOptions(
                mcp_servers=mcp_servers,
                system_prompt=SYSTEM_PROMPT,
                permission_mode="bypassPermissions",
                max_turns=max_turns,
            ),
        ):
            step += 1
            etype = type(event).__name__
            entry: dict[str, Any] = {
                "step": step,
                "event_type": etype,
                "elapsed_s": round(time.time() - t0, 2),
            }

            if isinstance(event, SystemMessage):
                entry["subtype"] = getattr(event, "subtype", None)

            elif isinstance(event, AssistantMessage):
                num_turns += 1
                blocks = []
                if hasattr(event, "content") and event.content:
                    for block in event.content:
                        b = serialize_block(block)
                        blocks.append(b)
                        if b["type"] == "text" and b["text"].strip():
                            final_answer = b["text"]
                            print(f"[STEP {step}] TEXT: {b['text'][:250]}")
                        elif b["type"] == "tool_use":
                            tool_calls.append({"name": b["name"], "input": b["input"]})
                            tool_call_names.append(b["name"])
                            print(f"[STEP {step}] TOOL: {b['name']}({str(b['input'])[:150]})")
                        elif b["type"] == "thinking":
                            print(f"[STEP {step}] THINKING: {b['thinking'][:180]}")
                entry["blocks"] = blocks

            elif isinstance(event, UserMessage):
                blocks = []
                if hasattr(event, "content") and event.content:
                    for block in event.content:
                        b = serialize_block(block)
                        blocks.append(b)
                        if b["type"] == "tool_result":
                            preview = str(b["content"])[:200]
                            print(f"[STEP {step}] RESULT: {preview}")
                entry["blocks"] = blocks

            elif isinstance(event, ResultMessage):
                if hasattr(event, "usage") and event.usage:
                    u = event.usage if isinstance(event.usage, dict) else {}
                    input_tokens = u.get("input_tokens", 0)
                    output_tokens = u.get("output_tokens", 0)
                    cache_creation = u.get("cache_creation_input_tokens", 0)
                    cache_read = u.get("cache_read_input_tokens", 0)
                    entry["usage"] = u

            trace.append(entry)

    except Exception as e:
        print(f"[WARN] SDK exception at step {step}: {type(e).__name__}: {str(e)[:200]}")

    elapsed = round(time.time() - t0, 2)
    distinct_tools = sorted(set(tool_call_names))
    tool_count_by_name: dict[str, int] = {}
    for n in tool_call_names:
        tool_count_by_name[n] = tool_count_by_name.get(n, 0) + 1

    return {
        "mode": mode_name,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": USER_PROMPT,
        "trace": trace,
        "tool_calls": tool_calls,
        "tool_call_count": len(tool_calls),
        "tool_call_names": tool_call_names,
        "tool_count_by_name": tool_count_by_name,
        "distinct_tools_used": distinct_tools,
        "num_tool_kinds": len(distinct_tools),
        "num_turns": num_turns,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_tokens": cache_creation,
        "cache_read_tokens": cache_read,
        "total_tokens": input_tokens + output_tokens + cache_creation,
        "time_s": elapsed,
        "final_answer": final_answer,
    }


async def main_async() -> None:
    print("=" * 75)
    print("FAIR COMPARISON: Same agent, same prompt, CLIO as extra toolkit")
    print("=" * 75)

    ndp_mcp_server_config = {
        "type": "stdio",
        "command": NDP_MCP_BINARY,
        "args": ["--transport", "stdio"],
    }

    clio_sdk_server = create_sdk_mcp_server(
        name="clio",
        version="1.0.0",
        tools=[clio_canonicalize_unit, clio_parse_csv, clio_search_ndp],
    )

    # Mode A: naked MCP (NDP-MCP + built-in Bash)
    result_a = await run_agent(
        mode_name="MODE A — NDP-MCP only (+ built-in Bash)",
        mcp_servers={"ndp": ndp_mcp_server_config},
        max_turns=15,
    )

    # Mode B: same + CLIO helpers
    result_b = await run_agent(
        mode_name="MODE B — NDP-MCP + CLIO helpers (+ built-in Bash)",
        mcp_servers={"ndp": ndp_mcp_server_config, "clio": clio_sdk_server},
        max_turns=15,
    )

    # ========== Comparison ==========
    print("\n" + "=" * 75)
    print("METRICS")
    print("=" * 75)
    rows = [
        ("Tool calls (total)", result_a["tool_call_count"], result_b["tool_call_count"]),
        ("Distinct tool kinds", result_a["num_tool_kinds"], result_b["num_tool_kinds"]),
        ("Turns", result_a["num_turns"], result_b["num_turns"]),
        ("Input tokens", result_a["input_tokens"], result_b["input_tokens"]),
        ("Output tokens", result_a["output_tokens"], result_b["output_tokens"]),
        ("Cache creation tokens", result_a["cache_creation_tokens"], result_b["cache_creation_tokens"]),
        ("Cache read tokens", result_a["cache_read_tokens"], result_b["cache_read_tokens"]),
        ("Total tokens", result_a["total_tokens"], result_b["total_tokens"]),
        ("Wall time (s)", result_a["time_s"], result_b["time_s"]),
    ]
    print(f"{'Metric':<28} | {'Mode A (MCP only)':>20} | {'Mode B (MCP+CLIO)':>20}")
    print("-" * 75)
    for label, a, b in rows:
        print(f"{label:<28} | {str(a):>20} | {str(b):>20}")

    print(f"\nDistinct tools used in A: {result_a['distinct_tools_used']}")
    print(f"Distinct tools used in B: {result_b['distinct_tools_used']}")
    print(f"\nTool call distribution A: {result_a['tool_count_by_name']}")
    print(f"Tool call distribution B: {result_b['tool_count_by_name']}")

    if result_a["total_tokens"] > 0:
        delta_pct = (result_a["total_tokens"] - result_b["total_tokens"]) / result_a["total_tokens"] * 100
        print(f"\nToken delta: {result_a['total_tokens'] - result_b['total_tokens']:+,} ({delta_pct:+.1f}%)")
    if result_a["time_s"] > 0:
        time_delta = result_a["time_s"] - result_b["time_s"]
        print(f"Time delta: {time_delta:+.1f}s")

    # Save full results
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "test": "Fair end-to-end comparison: same agent, same prompt, MCP vs MCP+CLIO",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": USER_PROMPT,
        "mode_a": {k: v for k, v in result_a.items() if k != "trace"},
        "mode_b": {k: v for k, v in result_b.items() if k != "trace"},
        "comparison": {
            "tool_call_delta": result_a["tool_call_count"] - result_b["tool_call_count"],
            "token_delta": result_a["total_tokens"] - result_b["total_tokens"],
            "token_delta_pct": round(
                (result_a["total_tokens"] - result_b["total_tokens"]) / result_a["total_tokens"] * 100, 1,
            ) if result_a["total_tokens"] > 0 else 0,
            "time_delta_s": round(result_a["time_s"] - result_b["time_s"], 2),
        },
    }
    with open(OUT_DIR / "fair_comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save full traces separately (they're large)
    with open(OUT_DIR / "fair_comparison_mode_a_trace.json", "w") as f:
        json.dump(result_a, f, indent=2, default=str)
    with open(OUT_DIR / "fair_comparison_mode_b_trace.json", "w") as f:
        json.dump(result_b, f, indent=2, default=str)

    print(f"\nSaved:")
    print(f"  {OUT_DIR / 'fair_comparison_summary.json'}")
    print(f"  {OUT_DIR / 'fair_comparison_mode_a_trace.json'}")
    print(f"  {OUT_DIR / 'fair_comparison_mode_b_trace.json'}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
