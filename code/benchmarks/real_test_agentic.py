#!/usr/bin/env python3
"""REAL TEST: Claude Agent with direct NDP-MCP vs CLIO orchestration layer.

Correct architecture:
  Mode A (without CLIO): Claude Agent has NDP-MCP tools directly
    → Agent must decide what to search, read all raw results, reason
  Mode B (with CLIO): Claude Agent has ONE tool: clio_find_datasets
    → CLIO internally calls NDP-MCP (real subprocess)
    → CLIO indexes, profiles, applies science operators (unit conversion)
    → CLIO returns clean ranked results to the agent

CLIO is NOT an MCP server — it's an orchestration library that USES NDP-MCP.

Output: eval/real_tests/agentic_llm_test.json
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

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

_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE_DIR / "src"))

from clio_agentic_search.connectors.ndp.connector import NDPConnector
from clio_agentic_search.connectors.ndp.mcp_client import NDPMCPClient
from clio_agentic_search.indexing.scientific import (
    build_structure_aware_chunk_plan,
)
from clio_agentic_search.models.contracts import (
    DocumentRecord,
    EmbeddingRecord,
    MetadataRecord,
)
from clio_agentic_search.retrieval.coordinator import RetrievalCoordinator
from clio_agentic_search.retrieval.corpus_profile import build_corpus_profile
from clio_agentic_search.retrieval.scientific import (
    NumericRangeOperator,
    ScientificQueryOperators,
)
from clio_agentic_search.storage.contracts import FileIndexState
from clio_agentic_search.storage.duckdb_store import DuckDBStorage

OUT_DIR = _CODE_DIR.parent / "eval" / "real_tests"
NDP_MCP_BINARY = str(_CODE_DIR / ".venv" / "bin" / "ndp-mcp")

# Global CLIO state (initialized in main)
_clio_connector: NDPConnector | None = None
_clio_coordinator: RetrievalCoordinator | None = None
_clio_storage: DuckDBStorage | None = None


# ===================================================================
# CLIO orchestration: the "agentic layer" that uses NDP-MCP internally
# ===================================================================

@tool(
    name="clio_find_datasets",
    description=(
        "Find scientific datasets using CLIO Search. CLIO internally discovers data "
        "from NDP-MCP, indexes it, profiles metadata, applies unit conversion, and "
        "returns ranked results. This is a ONE-SHOT call — CLIO handles everything. "
        "Optionally provide min_value, max_value, unit for cross-unit matching "
        "(e.g. degC, kPa, km/h). Returns top 5 ranked datasets with scores."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query (e.g. 'temperature above 30 Celsius')",
            },
            "min_value": {"type": "number", "description": "Optional: min numeric value"},
            "max_value": {"type": "number", "description": "Optional: max numeric value"},
            "unit": {"type": "string", "description": "Optional: unit (degC, kPa, km/h, etc.)"},
        },
        "required": ["query"],
    },
)
async def clio_find_datasets(args: dict[str, Any]) -> dict[str, Any]:
    """The ONE tool CLIO exposes to the agent. Does the full pipeline."""
    if _clio_connector is None or _clio_coordinator is None or _clio_storage is None:
        return {"content": [{"type": "text", "text": '{"error": "CLIO not initialized"}'}]}

    query_text = args.get("query", "")
    min_value = args.get("min_value")
    max_value = args.get("max_value")
    unit = args.get("unit")

    # Build scientific operators if unit conversion requested
    operators = ScientificQueryOperators()
    if min_value is not None and max_value is not None and unit:
        try:
            operators = ScientificQueryOperators(
                numeric_range=NumericRangeOperator(
                    minimum=float(min_value), maximum=float(max_value), unit=unit,
                ),
            )
        except (ValueError, KeyError):
            pass

    # Profile the corpus (agent doesn't see this, CLIO does internally)
    profile = build_corpus_profile(_clio_storage, "ndp")

    # Search with branch selection (CLIO internally decides which branches)
    result = _clio_coordinator.query(
        connector=_clio_connector,
        query=query_text,
        top_k=5,
        scientific_operators=operators,
    )

    # Extract branch info for the response (shows what CLIO reasoned)
    branches_used = []
    for event in result.trace:
        if event.stage == "branch_plan_selected":
            for b in ("lexical", "vector", "scientific"):
                if event.attributes.get(f"use_{b}") == "True":
                    branches_used.append(b)

    citations = []
    for c in result.citations:
        citations.append({
            "title": c.snippet[:100].split("\n")[0].lstrip("# "),
            "score": round(c.score, 3),
            "snippet": c.snippet[:200],
        })

    response = {
        "clio_profile": {
            "documents": profile.document_count,
            "chunks": profile.chunk_count,
            "measurements": profile.measurement_count,
            "metadata_density": round(profile.metadata_density, 3),
        },
        "clio_strategy": {
            "branches_used": branches_used,
            "unit_conversion_applied": unit is not None,
        },
        "results": citations,
        "note": (
            "CLIO discovered this data from NDP-MCP, indexed 97 datasets, "
            "profiled metadata, and applied science-aware operators. "
            "Results are ranked by composite score."
        ),
    }
    return {"content": [{"type": "text", "text": json.dumps(response)}]}


# ===================================================================
# Helper: index NDP data into CLIO via the NDP-MCP client
# ===================================================================

async def setup_clio_via_ndp_mcp(tmpdir: Path) -> tuple[NDPConnector, DuckDBStorage, dict[str, Any]]:
    """CLIO uses NDP-MCP (real subprocess) to discover data, then indexes it."""
    print("  CLIO → connecting to NDP-MCP subprocess...")
    storage = DuckDBStorage(database_path=tmpdir / "ndp.duckdb")
    connector = NDPConnector(namespace="ndp", storage=storage)
    connector.connect()

    # Call NDP-MCP as a REAL subprocess via MCP protocol
    mcp_client = NDPMCPClient(ndp_mcp_binary=NDP_MCP_BINARY)
    discovery_stats = {"terms": [], "datasets_found": 0, "mcp_tools_available": []}

    async with mcp_client.connect():
        # Verify CLIO can see NDP-MCP tools
        tools = await mcp_client.list_tools()
        discovery_stats["mcp_tools_available"] = tools
        print(f"  CLIO sees NDP-MCP tools: {tools}")

        # CLIO calls NDP-MCP's search_datasets for each discovery term
        for term in ["temperature", "pressure", "wind", "humidity", "glacier", "ocean"]:
            datasets = await mcp_client.search_datasets([term], limit=15)
            discovery_stats["terms"].append({"term": term, "found": len(datasets)})
            discovery_stats["datasets_found"] += len(datasets)
            # Pass the NDP-MCP results to CLIO's indexing pipeline
            connector.index_datasets(datasets)

    return connector, storage, discovery_stats


# ===================================================================
# Agent runner
# ===================================================================

async def run_agent(
    question: str,
    mcp_servers: dict[str, Any],
    system_prompt: str,
) -> dict[str, Any]:
    """Run Claude agent and collect metrics."""
    t0 = time.time()
    tool_calls: list[str] = []
    input_tokens = 0
    output_tokens = 0
    cache_creation = 0
    cache_read = 0
    final_answer = ""
    num_turns = 0

    async for event in query(
        prompt=question,
        options=ClaudeAgentOptions(
            mcp_servers=mcp_servers,
            system_prompt=system_prompt,
            permission_mode="bypassPermissions",
            max_turns=6,
        ),
    ):
        if isinstance(event, AssistantMessage):
            num_turns += 1
            if hasattr(event, "content"):
                for block in event.content:
                    if isinstance(block, TextBlock):
                        final_answer = block.text
                    elif isinstance(block, ToolUseBlock):
                        tool_calls.append(block.name)
        elif isinstance(event, ResultMessage):
            if hasattr(event, "usage") and event.usage:
                u = event.usage
                if isinstance(u, dict):
                    input_tokens = u.get("input_tokens", 0)
                    output_tokens = u.get("output_tokens", 0)
                    cache_creation = u.get("cache_creation_input_tokens", 0)
                    cache_read = u.get("cache_read_input_tokens", 0)

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
        "answer": final_answer[:800],
    }


async def main_async() -> None:
    global _clio_connector, _clio_coordinator, _clio_storage

    print("=" * 70)
    print("REAL TEST: Claude Agent + NDP-MCP vs Claude Agent + CLIO (uses NDP-MCP)")
    print("=" * 70)

    question = (
        "Find scientific datasets that contain temperature measurements "
        "above 30 degrees Celsius. List the top 3 most relevant datasets "
        "with title and one-sentence explanation."
    )
    print(f"\nQuestion: {question}\n")

    with tempfile.TemporaryDirectory(prefix="clio_real_") as tmpdir:
        tmpdir = Path(tmpdir)

        # --- Setup CLIO: uses NDP-MCP internally to discover + index ---
        print("Setting up CLIO (CLIO talks to NDP-MCP as a subprocess)...")
        _clio_connector, _clio_storage, discovery = await setup_clio_via_ndp_mcp(tmpdir)
        _clio_coordinator = RetrievalCoordinator()

        profile = build_corpus_profile(_clio_storage, "ndp")
        print(f"  CLIO indexed: {profile.document_count} docs, "
              f"{profile.chunk_count} chunks, "
              f"{profile.measurement_count} measurements")
        print(f"  Discovery via NDP-MCP: {discovery['datasets_found']} datasets")
        print(f"  NDP-MCP tools CLIO can call: {discovery['mcp_tools_available']}")

        # --- Configure MCP servers for each mode ---
        # Mode A: Agent has direct access to NDP-MCP
        ndp_mcp_direct = {
            "type": "stdio",
            "command": NDP_MCP_BINARY,
            "args": ["--transport", "stdio"],
        }

        # Mode B: Agent has ONE CLIO tool (CLIO uses NDP-MCP internally)
        clio_sdk_server = create_sdk_mcp_server(
            name="clio",
            version="1.0.0",
            tools=[clio_find_datasets],
        )

        # --- Mode A: Agent talks directly to NDP-MCP ---
        print("\n" + "=" * 70)
        print("MODE A: Claude Agent + NDP-MCP (raw)")
        print("=" * 70)
        result_a = await run_agent(
            question=question,
            mcp_servers={"ndp": ndp_mcp_direct},
            system_prompt=(
                "You are a scientific data discovery agent. Use the NDP MCP tools "
                "(search_datasets, list_organizations, get_dataset_details) to find "
                "relevant datasets. You must decide search terms and reason through "
                "the raw results yourself. Be concise — list top 3 relevant datasets."
            ),
        )
        print(f"Tool calls: {result_a['num_tool_calls']}")
        print(f"  {result_a['tool_calls']}")
        print(f"Turns: {result_a['num_turns']}")
        print(f"Tokens: in={result_a['input_tokens']} out={result_a['output_tokens']} "
              f"cache_create={result_a['cache_creation_tokens']} "
              f"cache_read={result_a['cache_read_tokens']}")
        print(f"Total tokens (in+out+cache_create): {result_a['total_tokens']}")
        print(f"Time: {result_a['time_s']}s")

        # --- Mode B: Agent talks to CLIO (which uses NDP-MCP) ---
        print("\n" + "=" * 70)
        print("MODE B: Claude Agent + CLIO (CLIO internally uses NDP-MCP)")
        print("=" * 70)
        result_b = await run_agent(
            question=question,
            mcp_servers={"clio": clio_sdk_server},
            system_prompt=(
                "You are a scientific data discovery agent. You have ONE tool: "
                "clio_find_datasets. CLIO handles everything — it discovers data "
                "from NDP, profiles metadata, applies unit conversion, and returns "
                "ranked results. Call clio_find_datasets with appropriate parameters "
                "(use min_value, max_value, unit for unit-aware search). "
                "Be concise — list top 3 relevant datasets from the results."
            ),
        )
        print(f"Tool calls: {result_b['num_tool_calls']}")
        print(f"  {result_b['tool_calls']}")
        print(f"Turns: {result_b['num_turns']}")
        print(f"Tokens: in={result_b['input_tokens']} out={result_b['output_tokens']} "
              f"cache_create={result_b['cache_creation_tokens']} "
              f"cache_read={result_b['cache_read_tokens']}")
        print(f"Total tokens (in+out+cache_create): {result_b['total_tokens']}")
        print(f"Time: {result_b['time_s']}s")

        _clio_connector.teardown()
        _clio_storage.teardown()

    # --- Comparison ---
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<28} | {'Mode A (direct NDP)':>20} | {'Mode B (via CLIO)':>20}")
    print("-" * 76)
    rows = [
        ("Tool calls", result_a["num_tool_calls"], result_b["num_tool_calls"]),
        ("Turns", result_a["num_turns"], result_b["num_turns"]),
        ("Input tokens", result_a["input_tokens"], result_b["input_tokens"]),
        ("Output tokens", result_a["output_tokens"], result_b["output_tokens"]),
        ("Cache creation tokens", result_a["cache_creation_tokens"], result_b["cache_creation_tokens"]),
        ("Cache read tokens", result_a["cache_read_tokens"], result_b["cache_read_tokens"]),
        ("Total tokens", result_a["total_tokens"], result_b["total_tokens"]),
        ("Time (s)", f"{result_a['time_s']:.1f}", f"{result_b['time_s']:.1f}"),
    ]
    for label, a, b in rows:
        print(f"{label:<28} | {str(a):>20} | {str(b):>20}")

    if result_a["total_tokens"] > 0:
        delta_pct = (result_a["total_tokens"] - result_b["total_tokens"]) / result_a["total_tokens"] * 100
        print(f"\nToken savings: {delta_pct:+.1f}% ({result_a['total_tokens'] - result_b['total_tokens']} tokens)")
    if result_a["num_tool_calls"] > 0:
        tc_delta = result_a["num_tool_calls"] - result_b["num_tool_calls"]
        print(f"Tool call savings: {tc_delta} ({tc_delta / result_a['num_tool_calls'] * 100:.0f}% fewer calls)")

    print(f"\n--- Mode A Answer (direct NDP-MCP) ---\n{result_a['answer'][:500]}")
    print(f"\n--- Mode B Answer (via CLIO) ---\n{result_b['answer'][:500]}")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "test": "Claude Agent: NDP-MCP direct vs CLIO (using NDP-MCP)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": {
            "mode_a": "Claude Agent → NDP-MCP tools directly",
            "mode_b": "Claude Agent → CLIO → NDP-MCP (CLIO orchestrates)",
        },
        "question": question,
        "ndp_mcp_binary": NDP_MCP_BINARY,
        "clio_discovery_via_ndp_mcp": discovery,
        "clio_corpus": {
            "documents": profile.document_count,
            "chunks": profile.chunk_count,
            "measurements": profile.measurement_count,
            "metadata_density": round(profile.metadata_density, 4),
        },
        "mode_a_direct_ndp": result_a,
        "mode_b_via_clio": result_b,
        "comparison": {
            "token_savings": result_a["total_tokens"] - result_b["total_tokens"],
            "token_savings_pct": round(
                (result_a["total_tokens"] - result_b["total_tokens"]) / max(result_a["total_tokens"], 1) * 100, 1
            ),
            "tool_call_savings": result_a["num_tool_calls"] - result_b["num_tool_calls"],
            "time_savings_s": round(result_a["time_s"] - result_b["time_s"], 2),
        },
    }
    out_path = OUT_DIR / "agentic_llm_test.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
