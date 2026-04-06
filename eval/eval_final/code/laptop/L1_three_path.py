#!/usr/bin/env python3
"""L1: Three-path comparison — Raw Agent vs LLM+NDP-MCP vs LLM+CLIO.

Run 0: Raw Claude agent (no MCP) — uses web search, sub-agents, etc.
Run 1: LLM agent + NDP-MCP directly (per query)
Run 2: LLM agent + CLIO tool (CLIO uses NDP-MCP internally)

Captures: main agent tokens/time, sub-agent tokens/time, tool calls, answers.
"""
from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[4]
_CODE = _REPO / "code"
sys.path.insert(0, str(_CODE / "src"))

OUT_DIR = _REPO / "eval" / "eval_final" / "outputs"
NDP_MCP_BINARY = str(_CODE / ".venv" / "bin" / "ndp-mcp")

QUERIES = [
    {"id": "Q01", "text": "Find datasets with temperature measurements above 30 degrees Celsius from weather stations", "terms": ["temperature", "weather"]},
    {"id": "Q02", "text": "Find datasets reporting atmospheric pressure around 101 kPa", "terms": ["pressure", "atmospheric"]},
    {"id": "Q03", "text": "Find wind speed measurements above 50 km/h from meteorological datasets", "terms": ["wind", "speed"]},
    {"id": "Q04", "text": "Find datasets about glacier ice sheet temperature observations", "terms": ["glacier", "ice", "temperature"]},
    {"id": "Q05", "text": "Find humidity sensor measurements from environmental monitoring stations", "terms": ["humidity", "environmental"]},
    {"id": "Q06", "text": "Find solar radiation datasets measured in MJ per square meter", "terms": ["solar", "radiation"]},
    {"id": "Q07", "text": "Find ocean surface temperature satellite datasets", "terms": ["ocean", "temperature", "satellite"]},
    {"id": "Q08", "text": "Find datasets containing precipitation measurements above 100 mm per day", "terms": ["precipitation", "rainfall"]},
    {"id": "Q09", "text": "Find wildfire thermal detection datasets from satellite observations", "terms": ["wildfire", "thermal"]},
    {"id": "Q10", "text": "Find soil moisture measurements from agricultural monitoring networks", "terms": ["soil", "moisture", "agricultural"]},
]


# ============================================================================
# Run 0: Raw Claude Agent via CLI (spawns real sub-agents)
# ============================================================================
def run_raw_agent_query(query_text: str) -> dict[str, Any]:
    """Run query through Claude CLI with stream-json to capture sub-agents."""
    t0 = time.time()

    prompt = (
        f"{query_text}\n\n"
        "Search thoroughly: check multiple data repositories (NOAA, NASA, NDP, etc.), "
        "verify the data actually exists, check units, and report specific dataset names and URLs."
    )

    proc = subprocess.run(
        [
            "claude", "-p",
            "--output-format", "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
            "--max-budget-usd", "2.0",
            "--model", "sonnet",
            prompt,
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(_REPO),
    )
    elapsed = time.time() - t0

    # Parse stream-json events
    events = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass

    # Extract main agent usage from result event
    main_usage = {"input_tokens": 0, "output_tokens": 0,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    num_turns = 0
    for evt in events:
        if evt.get("type") == "result":
            u = evt.get("usage", {})
            for k in main_usage:
                main_usage[k] = u.get(k, 0)
            num_turns = evt.get("num_turns", 0)

    # Extract tool calls from assistant events
    tool_calls = []
    for evt in events:
        if evt.get("type") == "assistant":
            content = evt.get("message", {}).get("content", [])
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    tool_calls.append(b.get("name", "?"))

    # Extract sub-agent info from Agent tool results
    sub_agents = []
    usage_pattern = re.compile(
        r"<usage>total_tokens:\s*(\d+)\s*\n"
        r"tool_uses:\s*(\d+)\s*\n"
        r"duration_ms:\s*(\d+)\s*</usage>"
    )
    for evt in events:
        if evt.get("type") == "user":
            content = evt.get("message", {}).get("content", [])
            for b in content:
                if not isinstance(b, dict):
                    continue
                text = ""
                if b.get("type") == "tool_result":
                    if isinstance(b.get("content"), str):
                        text = b["content"]
                    elif isinstance(b.get("content"), list):
                        for c in b["content"]:
                            if isinstance(c, dict) and c.get("type") == "text":
                                text += c.get("text", "")
                m = usage_pattern.search(text)
                if m:
                    sub_agents.append({
                        "total_tokens": int(m.group(1)),
                        "tool_uses": int(m.group(2)),
                        "duration_ms": int(m.group(3)),
                    })

    # Extract final answer text
    final_answer = ""
    for evt in reversed(events):
        if evt.get("type") == "assistant":
            content = evt.get("message", {}).get("content", [])
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text" and len(b.get("text", "")) > 100:
                    final_answer = b["text"]
                    break
            if final_answer:
                break

    main_tokens = sum(main_usage.values())
    sub_tokens = sum(s["total_tokens"] for s in sub_agents)
    sub_tool_uses = sum(s["tool_uses"] for s in sub_agents)

    return {
        "elapsed_s": round(elapsed, 2),
        "num_turns": num_turns,
        "num_tool_calls": len(tool_calls),
        "tool_calls": tool_calls,
        "main_usage": main_usage,
        "main_tokens": main_tokens,
        "sub_agents": sub_agents,
        "sub_agent_count": len(sub_agents),
        "sub_agent_tokens": sub_tokens,
        "sub_agent_tool_uses": sub_tool_uses,
        "total_tokens": main_tokens + sub_tokens,
        "total_tool_calls": len(tool_calls) + sub_tool_uses,
        "answer_length": len(final_answer),
        "answer_snippet": final_answer[:500],
    }


# ============================================================================
# CLIO setup (same as original L1)
# ============================================================================
_connector = None
_db = None


async def setup_clio():
    global _connector, _db
    from clio_agentic_search.connectors.ndp.mcp_client import NDPMCPClient
    from clio_agentic_search.connectors.ndp.connector import NDPConnector
    from clio_agentic_search.retrieval.corpus_profile import build_corpus_profile
    from clio_agentic_search.storage.duckdb_store import DuckDBStorage

    tmp = Path(tempfile.mkdtemp(prefix="L1_"))
    _db = DuckDBStorage(database_path=tmp / "ndp.duckdb")
    _connector = NDPConnector(namespace="ndp", storage=_db)
    _connector.connect()

    all_terms = sorted({t for q in QUERIES for t in q["terms"]})
    print(f"  Discover terms: {all_terms}")

    client = NDPMCPClient(ndp_mcp_binary=NDP_MCP_BINARY)
    t0 = time.time()
    async with client.connect() as c:
        print(f"  NDP-MCP tools: {await c.list_tools()}")
        datasets = await c.search_datasets(all_terms, limit=25)
    print(f"  Discovered {len(datasets)} datasets via MCP")

    _connector.index_datasets(datasets)
    csv_stats = _connector.index_csv_resources(
        datasets, max_csvs=10, max_rows_per_csv=2000, timeout_s=45.0
    )
    index_time = time.time() - t0

    profile = build_corpus_profile(_db, "ndp")
    stats = {
        "index_time_s": round(index_time, 2),
        "documents": profile.document_count,
        "chunks": profile.chunk_count,
        "measurements": profile.measurement_count,
        "density": round(profile.metadata_density, 4),
        "csv": csv_stats,
    }
    print(
        f"  Index: {profile.document_count} docs, {profile.chunk_count} chunks, "
        f"{profile.measurement_count} meas, density={profile.metadata_density:.1%}, "
        f"CSV: {csv_stats.get('csvs_processed',0)} files/"
        f"{csv_stats.get('measurements_found',0)} meas, time={index_time:.1f}s"
    )
    return stats


# ============================================================================
# Run 1 & 2: agent query via SDK (with sub-agent tracking)
# ============================================================================
async def run_agent_query(
    query_text: str, mcp_servers: dict, system_prompt: str
) -> dict[str, Any]:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TaskNotificationMessage,
        TaskStartedMessage,
        TextBlock,
        ToolUseBlock,
        query,
    )

    t0 = time.time()
    tool_calls: list[str] = []
    final_answer = ""
    main_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    sub_agents: list[dict] = []
    current_subs: dict[str, float] = {}

    try:
        async for event in query(
            prompt=query_text,
            options=ClaudeAgentOptions(
                mcp_servers=mcp_servers,
                system_prompt=system_prompt,
                permission_mode="bypassPermissions",
                max_turns=12,
            ),
        ):
            if isinstance(event, AssistantMessage) and hasattr(event, "content"):
                for b in event.content or []:
                    if isinstance(b, TextBlock) and b.text.strip():
                        final_answer = b.text
                    elif isinstance(b, ToolUseBlock):
                        tool_calls.append(b.name)
            elif isinstance(event, TaskStartedMessage):
                current_subs[event.task_id] = time.time()
            elif isinstance(event, TaskNotificationMessage):
                start_t = current_subs.pop(event.task_id, t0)
                u = event.usage
                sub_agents.append({
                    "total_tokens": getattr(u, "total_tokens", 0) if u else 0,
                    "tool_uses": getattr(u, "tool_uses", 0) if u else 0,
                    "duration_ms": getattr(u, "duration_ms", 0) if u else 0,
                })
            elif isinstance(event, ResultMessage) and hasattr(event, "usage"):
                u = event.usage if isinstance(event.usage, dict) else {}
                for k in main_usage:
                    main_usage[k] = u.get(k, main_usage[k])
    except Exception as e:
        final_answer = f"ERROR: {e}"

    main_tokens = sum(main_usage.values())
    sub_tokens = sum(s["total_tokens"] for s in sub_agents)
    sub_tool_uses = sum(s["tool_uses"] for s in sub_agents)

    return {
        "elapsed_s": round(time.time() - t0, 2),
        "num_tool_calls": len(tool_calls),
        "tool_calls": tool_calls,
        "main_usage": main_usage,
        "main_tokens": main_tokens,
        "sub_agents": sub_agents,
        "sub_agent_count": len(sub_agents),
        "sub_agent_tokens": sub_tokens,
        "sub_agent_tool_uses": sub_tool_uses,
        "total_tokens": main_tokens + sub_tokens,
        "total_tool_calls": len(tool_calls) + sub_tool_uses,
        "answer_length": len(final_answer),
        "answer_snippet": final_answer[:500],
    }


# ============================================================================
# Main
# ============================================================================
async def main():
    from claude_agent_sdk import create_sdk_mcp_server, tool
    from clio_agentic_search.indexing.scientific import canonicalize_measurement
    from clio_agentic_search.retrieval.agentic import AgenticRetriever
    from clio_agentic_search.retrieval.coordinator import RetrievalCoordinator
    from clio_agentic_search.retrieval.query_rewriter import FallbackQueryRewriter
    from clio_agentic_search.retrieval.scientific import (
        NumericRangeOperator,
        QualityFilterOperator,
        ScientificQueryOperators,
    )

    print("=" * 70)
    print("L1: Raw Agent vs LLM+NDP-MCP vs LLM+CLIO — 10 queries")
    print("=" * 70)

    # Setup CLIO
    print("\n[Setup] CLIO indexing NDP via MCP...")
    index_stats = await setup_clio()

    # CLIO tool
    @tool(
        name="clio_search",
        description=(
            "CLIO agentic retrieval over NDP catalog. Returns ranked citations. "
            "Supports numeric constraints with SI unit canonicalization."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "numeric_constraint": {
                    "type": "object",
                    "properties": {
                        "unit": {"type": "string"},
                        "minimum": {"type": "number"},
                        "maximum": {"type": "number"},
                    },
                },
                "top_k": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    )
    async def clio_tool(args: dict[str, Any]) -> dict[str, Any]:
        nc = args.get("numeric_constraint") or {}
        ops = ScientificQueryOperators()
        if nc.get("unit"):
            try:
                canonicalize_measurement(0.0, nc["unit"])
                ops = ScientificQueryOperators(
                    numeric_range=NumericRangeOperator(
                        unit=nc["unit"],
                        minimum=nc.get("minimum"),
                        maximum=nc.get("maximum"),
                    ),
                    quality_filter=QualityFilterOperator(),
                )
            except Exception:
                pass
        retriever = AgenticRetriever(
            coordinator=RetrievalCoordinator(),
            rewriter=FallbackQueryRewriter(),
            max_hops=3,
        )
        result = retriever.query(
            connector=_connector,
            query=args["query"],
            top_k=int(args.get("top_k", 10)),
            scientific_operators=ops,
        )
        if ops.numeric_range and not result.citations:
            result = retriever.query(
                connector=_connector,
                query=args["query"],
                top_k=int(args.get("top_k", 10)),
                scientific_operators=ScientificQueryOperators(),
            )
        cites = [
            {
                "doc": c.document_id,
                "score": round(c.score, 3),
                "title": (c.snippet or "").split("\n")[0][:120],
                "snippet": (c.snippet or "")[:400],
            }
            for c in result.citations
        ]
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({"citations": cites, "hops": len(result.hops)}),
                }
            ]
        }

    clio_server = create_sdk_mcp_server(
        name="clio", version="1.0.0", tools=[clio_tool]
    )

    ndp_mcp = {
        "type": "stdio",
        "command": NDP_MCP_BINARY,
        "args": ["--transport", "stdio"],
    }
    prompt_ndp = (
        "You are a scientific data discovery assistant. Use NDP-MCP to search "
        "the National Data Platform. Return matching datasets concisely."
    )
    prompt_clio = (
        "You are a scientific data discovery assistant. Use the clio_search tool "
        "to find datasets. It handles NDP access internally. One call returns "
        "ranked results. Summarize them."
    )

    results_r0: list[dict] = []
    results_r1: list[dict] = []
    results_r2: list[dict] = []

    for q in QUERIES:
        print(f"\n{'='*70}")
        print(f"  {q['id']}: {q['text'][:65]}")
        print(f"{'='*70}")

        # Run 0: Raw Claude Agent (via CLI)
        print(f"  Run 0 (Raw Agent)...", end="", flush=True)
        r0 = run_raw_agent_query(q["text"])
        r0["query_id"] = q["id"]
        results_r0.append(r0)
        print(
            f" total_tok={r0['total_tokens']:,} "
            f"(main={r0['main_tokens']:,} + subs={r0['sub_agent_tokens']:,}) "
            f"tools={r0['total_tool_calls']} "
            f"sub_agents={r0['sub_agent_count']} "
            f"time={r0['elapsed_s']}s "
            f"ans={r0['answer_length']}ch"
        )

        # Run 2: CLIO (cheaper, run first)
        print(f"  Run 2 (LLM+CLIO)...", end="", flush=True)
        r2 = await run_agent_query(q["text"], {"clio": clio_server}, prompt_clio)
        r2["query_id"] = q["id"]
        results_r2.append(r2)
        print(
            f" total_tok={r2['total_tokens']:,} "
            f"tools={r2['total_tool_calls']} "
            f"time={r2['elapsed_s']}s "
            f"ans={r2['answer_length']}ch"
        )

        # Run 1: NDP-MCP
        print(f"  Run 1 (LLM+NDP)...", end="", flush=True)
        r1 = await run_agent_query(q["text"], {"ndp": ndp_mcp}, prompt_ndp)
        r1["query_id"] = q["id"]
        results_r1.append(r1)
        print(
            f" total_tok={r1['total_tokens']:,} "
            f"tools={r1['total_tool_calls']} "
            f"time={r1['elapsed_s']}s "
            f"ans={r1['answer_length']}ch"
        )

    # Teardown
    try:
        _connector.teardown()
    except Exception:
        pass

    # ---- SUMMARY ----
    def _sum(results: list[dict], key: str) -> int | float:
        return sum(r.get(key, 0) for r in results)

    r0_tok = _sum(results_r0, "total_tokens")
    r1_tok = _sum(results_r1, "total_tokens")
    r2_tok = _sum(results_r2, "total_tokens")
    r0_main = _sum(results_r0, "main_tokens")
    r1_main = _sum(results_r1, "main_tokens")
    r2_main = _sum(results_r2, "main_tokens")
    r0_sub = _sum(results_r0, "sub_agent_tokens")
    r1_sub = _sum(results_r1, "sub_agent_tokens")
    r2_sub = _sum(results_r2, "sub_agent_tokens")
    r0_tools = _sum(results_r0, "total_tool_calls")
    r1_tools = _sum(results_r1, "total_tool_calls")
    r2_tools = _sum(results_r2, "total_tool_calls")
    r0_subs = _sum(results_r0, "sub_agent_count")
    r1_subs = _sum(results_r1, "sub_agent_count")
    r2_subs = _sum(results_r2, "sub_agent_count")
    r0_time = _sum(results_r0, "elapsed_s")
    r1_time = _sum(results_r1, "elapsed_s")
    r2_time = _sum(results_r2, "elapsed_s")
    r0_ans = sum(1 for r in results_r0 if r["answer_length"] > 200)
    r1_ans = sum(1 for r in results_r1 if r["answer_length"] > 200)
    r2_ans = sum(1 for r in results_r2 if r["answer_length"] > 200)

    print(f"\n{'='*70}")
    print("SUMMARY (10 queries)")
    print(f"{'='*70}")
    print(
        f"{'Metric':<30} | {'Raw Agent':>12} | {'LLM+NDP':>12} | {'LLM+CLIO':>12}"
    )
    print("-" * 74)
    print(f"{'Main tokens':<30} | {r0_main:>12,} | {r1_main:>12,} | {r2_main:>12,}")
    print(f"{'Sub-agent tokens':<30} | {r0_sub:>12,} | {r1_sub:>12,} | {r2_sub:>12,}")
    print(f"{'Total tokens':<30} | {r0_tok:>12,} | {r1_tok:>12,} | {r2_tok:>12,}")
    print(f"{'Tool calls (total)':<30} | {r0_tools:>12} | {r1_tools:>12} | {r2_tools:>12}")
    print(f"{'Sub-agents spawned':<30} | {r0_subs:>12} | {r1_subs:>12} | {r2_subs:>12}")
    print(f"{'Wall time':<30} | {r0_time:>11.1f}s | {r1_time:>11.1f}s | {r2_time:>11.1f}s")
    print(f"{'Queries with results':<30} | {r0_ans:>10}/10 | {r1_ans:>10}/10 | {r2_ans:>10}/10")

    print(f"\n>>> CLIO vs Raw Agent: {(1-r2_tok/max(r0_tok,1))*100:.1f}% token reduction")
    print(f">>> CLIO vs NDP-MCP:  {(1-r2_tok/max(r1_tok,1))*100:.1f}% token reduction")

    # Per-query table
    print(f"\n{'QID':<4} | {'R0 tokens':>10} {'R0 subs':>7} {'R0 time':>7} {'R0 ans':>6} | "
          f"{'R1 tokens':>10} {'R1 time':>7} {'R1 ans':>6} | "
          f"{'R2 tokens':>10} {'R2 time':>7} {'R2 ans':>6}")
    print("-" * 105)
    for r0, r1, r2 in zip(results_r0, results_r1, results_r2):
        print(
            f"{r0['query_id']:<4} | "
            f"{r0['total_tokens']:>10,} {r0['sub_agent_count']:>7} {r0['elapsed_s']:>6.1f}s {r0['answer_length']:>6} | "
            f"{r1['total_tokens']:>10,} {r1['elapsed_s']:>6.1f}s {r1['answer_length']:>6} | "
            f"{r2['total_tokens']:>10,} {r2['elapsed_s']:>6.1f}s {r2['answer_length']:>6}"
        )

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "L1: Raw Agent vs LLM+NDP-MCP vs LLM+CLIO — 10 queries",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "queries": QUERIES,
        "index_stats": index_stats,
        "summary": {
            "run0_total_tokens": r0_tok,
            "run0_main_tokens": r0_main,
            "run0_sub_tokens": r0_sub,
            "run0_sub_agents": int(r0_subs),
            "run0_total_tools": int(r0_tools),
            "run0_total_time_s": round(r0_time, 1),
            "run0_queries_with_results": r0_ans,
            "run1_total_tokens": r1_tok,
            "run1_main_tokens": r1_main,
            "run1_sub_tokens": r1_sub,
            "run1_total_tools": int(r1_tools),
            "run1_total_time_s": round(r1_time, 1),
            "run1_queries_with_results": r1_ans,
            "run2_total_tokens": r2_tok,
            "run2_main_tokens": r2_main,
            "run2_sub_tokens": r2_sub,
            "run2_total_tools": int(r2_tools),
            "run2_total_time_s": round(r2_time, 1),
            "run2_queries_with_results": r2_ans,
            "clio_vs_raw_token_reduction_pct": round((1 - r2_tok / max(r0_tok, 1)) * 100, 1),
            "clio_vs_ndp_token_reduction_pct": round((1 - r2_tok / max(r1_tok, 1)) * 100, 1),
        },
        "per_query_run0_raw_agent": results_r0,
        "per_query_run1_ndp": results_r1,
        "per_query_run2_clio": results_r2,
    }
    out_path = OUT_DIR / "L1_three_path.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
