#!/usr/bin/env python3
"""FAIR end-to-end comparison: agent WITHOUT CLIO vs agent WITH CLIO.

Same question: "Find temperature measurements above 30°C from NDP weather
station data. Report how many rows match and give 3 example values."

Mode A (without CLIO):
  Agent tools: NDP-MCP (search_datasets, get_dataset_details) + Bash
  Agent must: discover → parse resources → download CSV → identify units →
              convert units → filter rows itself

Mode B (with CLIO):
  Agent tools: ONE tool `clio_temperature_pipeline(query, min_value, unit)`
  CLIO handles: discover via NDP, parse resources, download CSV, canonicalize
                units, filter. Returns structured matches.

Measured:
  - Input/output/cache tokens
  - Tool calls
  - Wall clock time
  - Whether the answer is correct

Output: eval/eval_eve/outputs/end_to_end_agent_comparison.json
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
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    query,
    tool,
)

from clio_agentic_search.indexing.scientific import canonicalize_measurement

OUT_DIR = _CODE_ROOT.parent / "eval" / "eval_eve" / "outputs"
NDP_URL = "http://155.101.6.191:8003"

QUESTION = (
    "Find temperature measurements above 30 degrees Celsius from weather "
    "station datasets in the National Data Platform. Tell me how many rows "
    "match and give 3 example values (raw value and the station name if you "
    "can find it). Use the NDP-MCP tools to discover datasets, then fetch "
    "and parse the actual data."
)


# ===================================================================
# CLIO single tool (Mode B): runs the full pipeline internally
# ===================================================================

@tool(
    name="clio_temperature_pipeline",
    description=(
        "Run the FULL temperature search pipeline: discover datasets from NDP, "
        "parse downloadable resources, identify temperature columns and their "
        "units, canonicalize to Kelvin, and filter rows matching the threshold. "
        "Returns structured results with counts and sample matches. Use this "
        "for any 'find temperature above X degrees' query."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "min_value": {"type": "number", "description": "Threshold value"},
            "unit": {"type": "string", "description": "Unit (e.g. 'degC', 'degF', 'kelvin')"},
            "max_resources": {"type": "integer", "description": "Max CSV resources to process", "default": 3},
        },
        "required": ["min_value", "unit"],
    },
)
async def clio_temperature_pipeline(args: dict[str, Any]) -> dict[str, Any]:
    min_value = float(args.get("min_value", 30.0))
    unit = args.get("unit", "degC")
    max_resources = int(args.get("max_resources", 3))

    # Stage 1: discover via NDP
    with httpx.Client(timeout=30.0) as c:
        resp = c.get(f"{NDP_URL}/search",
                     params={"terms": "temperature", "server": "global"})
        datasets = resp.json() if resp.status_code == 200 else []

    # Stage 2: find parseable CSV resources
    csv_resources = []
    for ds in datasets:
        for r in ds.get("resources", []):
            fmt = (r.get("format") or "").upper()
            url = r.get("url", "")
            if fmt == "CSV" and url.startswith("http") and "all_stations" not in url:
                csv_resources.append({
                    "dataset": ds.get("title", "?"),
                    "name": r.get("name", "?"),
                    "url": url,
                })
                if len(csv_resources) >= max_resources:
                    break
        if len(csv_resources) >= max_resources:
            break

    # Stage 3 + 4: dispatch CSV connector, canonicalize, filter
    min_canonical, _ = canonicalize_measurement(min_value, unit)
    total_parsed = 0
    total_matched = 0
    per_resource = []
    sample_matches: list[dict[str, Any]] = []

    for r in csv_resources:
        try:
            with httpx.Client(timeout=60.0) as c:
                resp = c.get(r["url"])
                resp.raise_for_status()
            text = resp.text
        except Exception:
            continue

        lines = text.splitlines()
        if not lines:
            continue
        header = [h.strip() for h in lines[0].split(",")]

        temp_idx = None
        temp_unit = None
        for i, col in enumerate(header):
            if "temp" in col.lower() and "id" not in col.lower() and "station" not in col.lower():
                m = re.search(r"\(([^)]+)\)", col)
                if m:
                    temp_unit = m.group(1).strip()
                temp_idx = i
                break
        if temp_idx is None or not temp_unit:
            continue

        unit_for_canon = {"C": "degC", "°C": "degC", "F": "degF", "°F": "degF", "K": "kelvin"}.get(
            temp_unit, temp_unit,
        )

        station_idx = None
        for i, col in enumerate(header):
            if "stn name" in col.lower() or "station" in col.lower():
                station_idx = i
                break

        parsed = 0
        matched = 0
        for line in lines[1:5001]:
            cells = line.split(",")
            if len(cells) <= temp_idx:
                continue
            try:
                raw = float(cells[temp_idx].strip())
            except ValueError:
                continue
            try:
                canon, _ = canonicalize_measurement(raw, unit_for_canon)
            except (ValueError, KeyError):
                continue
            parsed += 1
            if canon >= min_canonical:
                matched += 1
                if len(sample_matches) < 3:
                    station = cells[station_idx].strip() if station_idx is not None and len(cells) > station_idx else "?"
                    sample_matches.append({
                        "raw_value": raw,
                        "raw_unit": temp_unit,
                        "canonical_K": round(canon, 2),
                        "station": station,
                    })

        per_resource.append({
            "dataset": r["dataset"][:70],
            "resource": r["name"][:60],
            "temp_column_unit": temp_unit,
            "rows_parsed": parsed,
            "rows_matched": matched,
        })
        total_parsed += parsed
        total_matched += matched

    response = {
        "query": f"temperature >= {min_value} {unit}",
        "canonical_threshold_K": round(min_canonical, 2),
        "datasets_discovered": len(datasets),
        "csv_resources_processed": len(per_resource),
        "total_rows_parsed": total_parsed,
        "total_rows_matched": total_matched,
        "match_rate": round(total_matched / total_parsed, 4) if total_parsed > 0 else 0,
        "sample_matches": sample_matches,
        "per_resource": per_resource,
    }
    return {"content": [{"type": "text", "text": json.dumps(response)}]}


# ===================================================================
# Agent runner with full trace capture
# ===================================================================

async def run_agent(
    question: str,
    mcp_servers: dict[str, Any],
    system_prompt: str,
    max_turns: int = 12,
) -> dict[str, Any]:
    t0 = time.time()
    tool_calls: list[dict[str, Any]] = []
    input_tokens = 0
    output_tokens = 0
    cache_creation = 0
    cache_read = 0
    final_answer = ""
    num_turns = 0

    try:
        async for event in query(
            prompt=question,
            options=ClaudeAgentOptions(
                mcp_servers=mcp_servers,
                system_prompt=system_prompt,
                permission_mode="bypassPermissions",
                max_turns=max_turns,
            ),
        ):
            if isinstance(event, AssistantMessage):
                num_turns += 1
                if hasattr(event, "content") and event.content:
                    for block in event.content:
                        if isinstance(block, TextBlock):
                            final_answer = block.text
                        elif isinstance(block, ToolUseBlock):
                            tool_calls.append({
                                "name": block.name,
                                "input": str(getattr(block, "input", {}))[:200],
                            })
            elif isinstance(event, ResultMessage):
                if hasattr(event, "usage") and event.usage:
                    u = event.usage if isinstance(event.usage, dict) else {}
                    input_tokens = u.get("input_tokens", 0)
                    output_tokens = u.get("output_tokens", 0)
                    cache_creation = u.get("cache_creation_input_tokens", 0)
                    cache_read = u.get("cache_read_input_tokens", 0)
    except Exception as e:
        print(f"[WARN] SDK exception: {type(e).__name__}: {str(e)[:150]}", flush=True)

    return {
        "tool_calls": tool_calls,
        "num_tool_calls": len(tool_calls),
        "num_turns": num_turns,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_tokens": cache_creation,
        "cache_read_tokens": cache_read,
        "total_tokens": input_tokens + output_tokens + cache_creation,
        "time_s": round(time.time() - t0, 2),
        "answer": final_answer[:1200],
    }


async def main_async() -> None:
    print("=" * 75)
    print("END-TO-END AGENT COMPARISON: without CLIO vs with CLIO")
    print("=" * 75)
    print(f"\nQuestion: {QUESTION}\n")

    ndp_mcp_server = {
        "type": "stdio",
        "command": str(_CODE_ROOT / ".venv" / "bin" / "ndp-mcp"),
        "args": ["--transport", "stdio"],
    }

    # --- Mode A: agent has NDP-MCP + Bash (Claude Code built-in) ---
    print("=" * 75)
    print("MODE A: Claude Agent + NDP-MCP + Bash (no CLIO)")
    print("Agent must discover → download → parse → filter on its own")
    print("=" * 75)
    result_a = await run_agent(
        question=QUESTION,
        mcp_servers={"ndp": ndp_mcp_server},
        system_prompt=(
            "You are a scientific data discovery agent. Use NDP MCP tools "
            "(search_datasets) to find temperature datasets. When you find "
            "downloadable CSV resources, use the Bash tool to download them "
            "(e.g., 'curl -s URL | head -n 100') and parse them to find "
            "temperature values. You must infer units from column headers "
            "(e.g., 'Air Temp (C)' means Celsius) and convert to check the "
            "threshold. Report matching count and 3 examples."
        ),
        max_turns=15,
    )
    print(f"Tool calls: {result_a['num_tool_calls']}")
    for tc in result_a["tool_calls"][:15]:
        print(f"  - {tc['name']}: {tc['input'][:100]}")
    print(f"Turns: {result_a['num_turns']}")
    print(f"Tokens: in={result_a['input_tokens']} out={result_a['output_tokens']} "
          f"cache_create={result_a['cache_creation_tokens']} "
          f"cache_read={result_a['cache_read_tokens']}")
    print(f"Total: {result_a['total_tokens']} | Time: {result_a['time_s']}s")
    print(f"\nAnswer:\n{result_a['answer'][:600]}")

    # --- Mode B: agent has ONE CLIO tool that does the whole pipeline ---
    print("\n" + "=" * 75)
    print("MODE B: Claude Agent + CLIO pipeline tool")
    print("CLIO handles discovery, download, parsing, unit conversion, filtering")
    print("=" * 75)
    clio_server = create_sdk_mcp_server(
        name="clio", version="1.0.0",
        tools=[clio_temperature_pipeline],
    )
    result_b = await run_agent(
        question=QUESTION,
        mcp_servers={"clio": clio_server},
        system_prompt=(
            "You have ONE tool: clio_temperature_pipeline. It runs the "
            "complete pipeline (discover from NDP, download CSVs, parse "
            "columns, convert units, filter rows). Call it once with "
            "appropriate parameters (min_value and unit), then summarize "
            "the results for the user. Report matching count and 3 examples."
        ),
        max_turns=5,
    )
    print(f"Tool calls: {result_b['num_tool_calls']}")
    for tc in result_b["tool_calls"]:
        print(f"  - {tc['name']}: {tc['input'][:100]}")
    print(f"Turns: {result_b['num_turns']}")
    print(f"Tokens: in={result_b['input_tokens']} out={result_b['output_tokens']} "
          f"cache_create={result_b['cache_creation_tokens']} "
          f"cache_read={result_b['cache_read_tokens']}")
    print(f"Total: {result_b['total_tokens']} | Time: {result_b['time_s']}s")
    print(f"\nAnswer:\n{result_b['answer'][:600]}")

    # --- Comparison ---
    print("\n" + "=" * 75)
    print("COMPARISON")
    print("=" * 75)
    t_a, t_b = result_a["total_tokens"], result_b["total_tokens"]
    tc_a, tc_b = result_a["num_tool_calls"], result_b["num_tool_calls"]
    tm_a, tm_b = result_a["time_s"], result_b["time_s"]

    print(f"{'Metric':<30} | {'Mode A (no CLIO)':>18} | {'Mode B (CLIO)':>18}")
    print("-" * 75)
    print(f"{'Tool calls':<30} | {tc_a:>18} | {tc_b:>18}")
    print(f"{'Turns':<30} | {result_a['num_turns']:>18} | {result_b['num_turns']:>18}")
    print(f"{'Input tokens':<30} | {result_a['input_tokens']:>18} | {result_b['input_tokens']:>18}")
    print(f"{'Output tokens':<30} | {result_a['output_tokens']:>18} | {result_b['output_tokens']:>18}")
    print(f"{'Cache creation tokens':<30} | {result_a['cache_creation_tokens']:>18} | {result_b['cache_creation_tokens']:>18}")
    print(f"{'Cache read tokens':<30} | {result_a['cache_read_tokens']:>18} | {result_b['cache_read_tokens']:>18}")
    print(f"{'Total tokens':<30} | {t_a:>18} | {t_b:>18}")
    print(f"{'Wall time (s)':<30} | {tm_a:>18} | {tm_b:>18}")

    if t_a > 0:
        tok_reduction = (1 - t_b / t_a) * 100
        print(f"\nToken savings:   {t_a - t_b:,} tokens  ({tok_reduction:+.1f}%)")
    if tc_a > 0:
        call_reduction = (1 - tc_b / tc_a) * 100
        print(f"Tool call reduction: {tc_a - tc_b} calls  ({call_reduction:+.1f}%)")
    if tm_a > 0:
        time_reduction = (1 - tm_b / tm_a) * 100
        print(f"Time savings: {tm_a - tm_b:.1f}s  ({time_reduction:+.1f}%)")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "test": "End-to-end agent comparison: NDP-MCP+Bash vs CLIO pipeline",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": QUESTION,
        "mode_a_no_clio": result_a,
        "mode_b_with_clio": result_b,
        "comparison": {
            "token_delta": t_a - t_b,
            "token_reduction_pct": round((1 - t_b / t_a) * 100, 1) if t_a > 0 else 0,
            "tool_call_delta": tc_a - tc_b,
            "time_delta_s": round(tm_a - tm_b, 2),
        },
    }
    out_path = OUT_DIR / "end_to_end_agent_comparison.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
