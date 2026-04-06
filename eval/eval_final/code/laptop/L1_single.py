#!/usr/bin/env python3
"""L1: LLM+NDP-MCP vs LLM+CLIO(NDP-MCP) — 10 queries.

Run 1: LLM agent + NDP-MCP directly (per query)
Run 2: LLM agent + CLIO tool (CLIO uses NDP-MCP internally, per query)

Same MCP server. Same LLM. Same queries. Does CLIO reduce token consumption?
"""
from __future__ import annotations
import asyncio, json, sys, tempfile, time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[4]
_CODE = _REPO / "code"
sys.path.insert(0, str(_CODE / "src"))

OUT_DIR = _REPO / "eval" / "eval_final" / "outputs"
NDP_MCP_BINARY = str(_CODE / ".venv" / "bin" / "ndp-mcp")

QUERIES = [
    {"id":"temp_above_30C","text":"Find datasets with temperature measurements above 30 degrees Celsius from weather stations","terms":["temperature","weather"]},
    {"id":"pressure_101kPa","text":"Find datasets reporting atmospheric pressure around 101 kPa","terms":["pressure","atmospheric"]},
    {"id":"wind_above_50kmh","text":"Find wind speed measurements above 50 km/h from meteorological datasets","terms":["wind","speed"]},
    {"id":"glacier_ice_temp","text":"Find datasets about glacier ice sheet temperature observations","terms":["glacier","ice","temperature"]},
    {"id":"humidity_sensor","text":"Find humidity sensor measurements from environmental monitoring stations","terms":["humidity","environmental"]},
    {"id":"solar_radiation_MJ","text":"Find solar radiation datasets measured in MJ per square meter","terms":["solar","radiation"]},
    {"id":"ocean_surface_temp","text":"Find ocean surface temperature satellite datasets","terms":["ocean","temperature","satellite"]},
    {"id":"precip_above_100mm","text":"Find datasets containing precipitation measurements above 100 mm per day","terms":["precipitation","rainfall"]},
    {"id":"wildfire_thermal","text":"Find wildfire thermal detection datasets from satellite observations","terms":["wildfire","thermal"]},
    {"id":"soil_moisture","text":"Find soil moisture measurements from agricultural monitoring networks","terms":["soil","moisture","agricultural"]},
]

# ============================================================================
# CLIO setup
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
    csv_stats = _connector.index_csv_resources(datasets, max_csvs=10, max_rows_per_csv=2000, timeout_s=45.0)
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
    print(f"  Index: {profile.document_count} docs, {profile.chunk_count} chunks, "
          f"{profile.measurement_count} meas, density={profile.metadata_density:.1%}, "
          f"CSV: {csv_stats.get('csvs_processed',0)} files/{csv_stats.get('measurements_found',0)} meas, "
          f"time={index_time:.1f}s")
    return stats


# ============================================================================
# Run single query through agent
# ============================================================================
async def run_agent_query(query_text: str, mcp_servers: dict, system_prompt: str) -> dict[str, Any]:
    from claude_agent_sdk import (
        AssistantMessage, ClaudeAgentOptions, ResultMessage,
        TextBlock, ToolUseBlock, query,
    )
    t0 = time.time()
    tool_calls, final_answer = [], ""
    usage = {"input_tokens":0,"output_tokens":0,
             "cache_creation_input_tokens":0,"cache_read_input_tokens":0}

    try:
        async for event in query(
            prompt=query_text,
            options=ClaudeAgentOptions(
                mcp_servers=mcp_servers,
                system_prompt=system_prompt,
                permission_mode="bypassPermissions", max_turns=12,
            ),
        ):
            if isinstance(event, AssistantMessage) and hasattr(event, "content"):
                for b in (event.content or []):
                    if isinstance(b, TextBlock) and b.text.strip(): final_answer = b.text
                    elif isinstance(b, ToolUseBlock): tool_calls.append(b.name)
            elif isinstance(event, ResultMessage) and hasattr(event, "usage"):
                u = event.usage if isinstance(event.usage, dict) else {}
                for k in usage: usage[k] = u.get(k, usage[k])
    except Exception as e:
        final_answer = f"ERROR: {e}"

    return {
        "elapsed_s": round(time.time() - t0, 2),
        "num_tool_calls": len(tool_calls),
        "tool_calls": tool_calls,
        "usage": usage,
        "total_tokens": sum(usage.values()),
        "answer_length": len(final_answer),
    }


# ============================================================================
# Main
# ============================================================================
async def main():
    from claude_agent_sdk import create_sdk_mcp_server, tool
    from clio_agentic_search.retrieval.agentic import AgenticRetriever
    from clio_agentic_search.retrieval.coordinator import RetrievalCoordinator
    from clio_agentic_search.retrieval.query_rewriter import FallbackQueryRewriter
    from clio_agentic_search.retrieval.scientific import (
        NumericRangeOperator, QualityFilterOperator, ScientificQueryOperators,
    )
    from clio_agentic_search.indexing.scientific import canonicalize_measurement

    print("="*70)
    print("L1: LLM+NDP-MCP vs LLM+CLIO(NDP-MCP) — 10 queries")
    print("="*70)

    # Setup
    print("\n[Setup] CLIO indexing NDP via MCP...")
    index_stats = await setup_clio()

    # CLIO tool
    @tool(
        name="clio_search",
        description="CLIO agentic retrieval over NDP catalog. Returns ranked citations. Supports numeric constraints with SI unit canonicalization.",
        input_schema={"type":"object","properties":{
            "query":{"type":"string"},
            "numeric_constraint":{"type":"object","properties":{"unit":{"type":"string"},"minimum":{"type":"number"},"maximum":{"type":"number"}}},
            "top_k":{"type":"integer","default":10},
        },"required":["query"]},
    )
    async def clio_tool(args: dict[str,Any]) -> dict[str,Any]:
        nc = args.get("numeric_constraint") or {}
        ops = ScientificQueryOperators()
        if nc.get("unit"):
            try:
                canonicalize_measurement(0.0, nc["unit"])
                ops = ScientificQueryOperators(
                    numeric_range=NumericRangeOperator(unit=nc["unit"],minimum=nc.get("minimum"),maximum=nc.get("maximum")),
                    quality_filter=QualityFilterOperator(),
                )
            except: pass
        retriever = AgenticRetriever(coordinator=RetrievalCoordinator(), rewriter=FallbackQueryRewriter(), max_hops=3)
        result = retriever.query(connector=_connector, query=args["query"], top_k=int(args.get("top_k",10)), scientific_operators=ops)
        if ops.numeric_range and not result.citations:
            result = retriever.query(connector=_connector, query=args["query"], top_k=int(args.get("top_k",10)), scientific_operators=ScientificQueryOperators())
        cites = [{"doc":c.document_id,"score":round(c.score,3),"title":(c.snippet or "").split("\n")[0][:120],"snippet":(c.snippet or "")[:400]} for c in result.citations]
        return {"content":[{"type":"text","text":json.dumps({"citations":cites,"hops":len(result.hops)})}]}

    clio_server = create_sdk_mcp_server(name="clio", version="1.0.0", tools=[clio_tool])

    ndp_mcp = {"type":"stdio","command":NDP_MCP_BINARY,"args":["--transport","stdio"]}
    prompt_ndp = "You are a scientific data discovery assistant. Use NDP-MCP to search the National Data Platform. Return matching datasets concisely."
    prompt_clio = "You are a scientific data discovery assistant. Use the clio_search tool to find datasets. It handles NDP access internally. One call returns ranked results. Summarize them."

    # Run all queries
    results_r1, results_r2 = [], []

    for q in QUERIES:
        print(f"\n--- {q['id']}: {q['text'][:70]} ---")

        # Run 2 first (CLIO — cheaper, CLIO index already warm)
        print(f"  Run 2 (LLM+CLIO)...", end="", flush=True)
        r2 = await run_agent_query(q["text"], {"clio": clio_server}, prompt_clio)
        r2["query_id"] = q["id"]
        results_r2.append(r2)
        print(f" tok={r2['total_tokens']:,} tools={r2['num_tool_calls']} time={r2['elapsed_s']}s")

        # Run 1 (NDP-MCP only)
        print(f"  Run 1 (LLM+NDP)...", end="", flush=True)
        r1 = await run_agent_query(q["text"], {"ndp": ndp_mcp}, prompt_ndp)
        r1["query_id"] = q["id"]
        results_r1.append(r1)
        print(f" tok={r1['total_tokens']:,} tools={r1['num_tool_calls']} time={r1['elapsed_s']}s")

    # Teardown
    try: _connector.teardown()
    except: pass

    # ---- SUMMARY ----
    r1_total_tok = sum(r["total_tokens"] for r in results_r1)
    r2_total_tok = sum(r["total_tokens"] for r in results_r2)
    r1_total_tools = sum(r["num_tool_calls"] for r in results_r1)
    r2_total_tools = sum(r["num_tool_calls"] for r in results_r2)
    r1_total_time = sum(r["elapsed_s"] for r in results_r1)
    r2_total_time = sum(r["elapsed_s"] for r in results_r2)

    print(f"\n{'='*70}")
    print("SUMMARY (10 queries)")
    print(f"{'='*70}")
    print(f"{'Metric':<30} | {'Run1: LLM+NDP':>15} | {'Run2: LLM+CLIO':>15} | {'Change':>10}")
    print("-"*76)
    print(f"{'Total tokens':<30} | {r1_total_tok:>15,} | {r2_total_tok:>15,} | {(r2_total_tok-r1_total_tok)/r1_total_tok*100:>+9.1f}%")
    print(f"{'Total tool calls':<30} | {r1_total_tools:>15} | {r2_total_tools:>15} | {(r2_total_tools-r1_total_tools)/max(r1_total_tools,1)*100:>+9.1f}%")
    print(f"{'Total wall time':<30} | {r1_total_time:>14.1f}s | {r2_total_time:>14.1f}s | {(r2_total_time-r1_total_time)/r1_total_time*100:>+9.1f}%")
    print(f"{'Avg tokens/query':<30} | {r1_total_tok//10:>15,} | {r2_total_tok//10:>15,} |")
    print(f"{'Avg tool calls/query':<30} | {r1_total_tools/10:>15.1f} | {r2_total_tools/10:>15.1f} |")

    print(f"\n>>> Token reduction: {(1-r2_total_tok/r1_total_tok)*100:.1f}%")
    print(f">>> Tokens saved: {r1_total_tok-r2_total_tok:,}")

    # Per-query table
    print(f"\n{'QID':<5} | {'R1 tokens':>10} {'R1 tools':>9} {'R1 time':>8} | {'R2 tokens':>10} {'R2 tools':>9} {'R2 time':>8} | {'Δ tok%':>7}")
    print("-"*80)
    for r1, r2 in zip(results_r1, results_r2):
        dtok = (r2["total_tokens"]-r1["total_tokens"])/max(r1["total_tokens"],1)*100
        print(f"{r1['query_id']:<5} | {r1['total_tokens']:>10,} {r1['num_tool_calls']:>9} {r1['elapsed_s']:>7.1f}s | {r2['total_tokens']:>10,} {r2['num_tool_calls']:>9} {r2['elapsed_s']:>7.1f}s | {dtok:>+6.1f}%")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "L1: LLM+NDP-MCP vs LLM+CLIO(NDP-MCP) — 10 queries",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "queries": QUERIES,
        "index_stats": index_stats,
        "summary": {
            "run1_total_tokens": r1_total_tok,
            "run2_total_tokens": r2_total_tok,
            "token_reduction_pct": round((1-r2_total_tok/r1_total_tok)*100, 1),
            "run1_total_tools": r1_total_tools,
            "run2_total_tools": r2_total_tools,
            "run1_total_time_s": round(r1_total_time, 1),
            "run2_total_time_s": round(r2_total_time, 1),
        },
        "per_query_run1": results_r1,
        "per_query_run2": results_r2,
    }
    with (OUT_DIR / "L1_10queries.json").open("w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {OUT_DIR / 'L1_10queries.json'}")


if __name__ == "__main__":
    asyncio.run(main())
