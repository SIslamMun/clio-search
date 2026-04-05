#!/usr/bin/env python3
"""VERBOSE version of the real test — logs EVERY step for verification.

Captures the full conversation trace:
  - User prompt
  - System prompt
  - Each assistant message (text, thinking, tool calls)
  - Each tool call with arguments
  - Each tool result
  - Final answer
  - Token usage

Output: eval/real_tests/agentic_full_trace.json
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

_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE_DIR / "src"))

from clio_agentic_search.connectors.ndp.connector import NDPConnector
from clio_agentic_search.connectors.ndp.mcp_client import NDPMCPClient
from clio_agentic_search.retrieval.coordinator import RetrievalCoordinator
from clio_agentic_search.retrieval.corpus_profile import build_corpus_profile
from clio_agentic_search.retrieval.scientific import (
    NumericRangeOperator,
    ScientificQueryOperators,
)
from clio_agentic_search.storage.duckdb_store import DuckDBStorage

OUT_DIR = _CODE_DIR.parent / "eval" / "real_tests"
NDP_MCP_BINARY = str(_CODE_DIR / ".venv" / "bin" / "ndp-mcp")

_clio_connector: NDPConnector | None = None
_clio_coordinator: RetrievalCoordinator | None = None
_clio_storage: DuckDBStorage | None = None


@tool(
    name="clio_find_datasets",
    description=(
        "Find scientific datasets using CLIO Search. CLIO internally discovers data "
        "from NDP-MCP, indexes it, profiles metadata, applies unit conversion, and "
        "returns ranked results. This is a ONE-SHOT call. Optionally provide "
        "min_value, max_value, unit for cross-unit matching."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "min_value": {"type": "number"},
            "max_value": {"type": "number"},
            "unit": {"type": "string"},
        },
        "required": ["query"],
    },
)
async def clio_find_datasets(args: dict[str, Any]) -> dict[str, Any]:
    if _clio_connector is None or _clio_coordinator is None or _clio_storage is None:
        return {"content": [{"type": "text", "text": '{"error":"not init"}'}]}

    query_text = args.get("query", "")
    min_value = args.get("min_value")
    max_value = args.get("max_value")
    unit = args.get("unit")

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

    profile = build_corpus_profile(_clio_storage, "ndp")
    # Ask for more chunks so we can deduplicate by document
    result = _clio_coordinator.query(
        connector=_clio_connector, query=query_text, top_k=20,
        scientific_operators=operators,
    )

    branches_used = []
    for event in result.trace:
        if event.stage == "branch_plan_selected":
            for b in ("lexical", "vector", "scientific"):
                if event.attributes.get(f"use_{b}") == "True":
                    branches_used.append(b)

    # DEDUPLICATE by document_id — keep highest-scored chunk per document.
    # This prevents multiple "fake datasets" from one real dataset's resources.
    seen_docs: dict[str, Any] = {}
    for c in result.citations:
        if c.document_id not in seen_docs or c.score > seen_docs[c.document_id].score:
            seen_docs[c.document_id] = c

    # Sort by score and take top 5 UNIQUE documents
    top_docs = sorted(seen_docs.values(), key=lambda x: -x.score)[:5]

    citations = []
    for c in top_docs:
        # Fetch full document metadata (title, URLs, formats, DOI)
        meta = _clio_storage.get_document_metadata("ndp", c.document_id)
        citations.append({
            "title": meta.get("title", c.snippet[:100].split("\n")[0].lstrip("# ")),
            "organization": meta.get("organization", ""),
            "score": round(c.score, 3),
            "ndp_id": meta.get("ndp_id", ""),
            "formats": meta.get("resource_formats", ""),
            "urls": meta.get("resource_urls", "")[:300],
            "doi": meta.get("doi", ""),
            "snippet": c.snippet[:250],
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
            "deduplication": f"deduplicated {len(result.citations)} chunks → {len(citations)} unique datasets",
        },
        "results": citations,
    }
    return {"content": [{"type": "text", "text": json.dumps(response)}]}


def serialize_block(block: Any) -> dict[str, Any]:
    """Convert a content block to a JSON-serializable dict."""
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    elif isinstance(block, ToolUseBlock):
        return {
            "type": "tool_use",
            "id": getattr(block, "id", ""),
            "name": block.name,
            "input": getattr(block, "input", {}),
        }
    elif isinstance(block, ToolResultBlock):
        content = getattr(block, "content", "")
        if isinstance(content, list):
            content = [serialize_block(c) if hasattr(c, "text") else str(c) for c in content]
        return {
            "type": "tool_result",
            "tool_use_id": getattr(block, "tool_use_id", ""),
            "content": content,
            "is_error": getattr(block, "is_error", False),
        }
    elif isinstance(block, ThinkingBlock):
        return {"type": "thinking", "thinking": getattr(block, "thinking", "")}
    else:
        return {"type": type(block).__name__, "repr": str(block)[:500]}


async def run_agent_with_trace(
    mode_name: str,
    question: str,
    mcp_servers: dict[str, Any],
    system_prompt: str,
) -> dict[str, Any]:
    """Run agent and capture EVERY event in the trace."""
    print(f"\n{'='*70}")
    print(f"RUNNING: {mode_name}")
    print('='*70)
    print(f"\n[USER PROMPT]\n{question}")
    print(f"\n[SYSTEM PROMPT]\n{system_prompt}")

    t0 = time.time()
    trace: list[dict[str, Any]] = []
    step = 0

    try:
     async for event in query(
        prompt=question,
        options=ClaudeAgentOptions(
            mcp_servers=mcp_servers,
            system_prompt=system_prompt,
            permission_mode="bypassPermissions",
            max_turns=6,
        ),
     ):
        step += 1
        event_type = type(event).__name__
        event_data: dict[str, Any] = {
            "step": step,
            "event_type": event_type,
            "elapsed_s": round(time.time() - t0, 2),
        }

        if isinstance(event, SystemMessage):
            event_data["subtype"] = getattr(event, "subtype", None)
            print(f"\n[STEP {step}] SystemMessage (subtype={event_data['subtype']})")

        elif isinstance(event, AssistantMessage):
            blocks = []
            if hasattr(event, "content") and event.content:
                for block in event.content:
                    b = serialize_block(block)
                    blocks.append(b)
                    if b["type"] == "text":
                        print(f"\n[STEP {step}] ASSISTANT TEXT:\n{b['text'][:400]}")
                    elif b["type"] == "tool_use":
                        print(f"\n[STEP {step}] ASSISTANT TOOL CALL: {b['name']}")
                        print(f"  args: {json.dumps(b['input'])[:300]}")
                    elif b["type"] == "thinking":
                        print(f"\n[STEP {step}] THINKING: {b['thinking'][:200]}")
            event_data["blocks"] = blocks
            if hasattr(event, "model"):
                event_data["model"] = event.model

        elif isinstance(event, UserMessage):
            blocks = []
            if hasattr(event, "content") and event.content:
                for block in event.content:
                    b = serialize_block(block)
                    blocks.append(b)
                    if b["type"] == "tool_result":
                        content_preview = str(b["content"])[:400]
                        print(f"\n[STEP {step}] TOOL RESULT:\n{content_preview}")
            event_data["blocks"] = blocks

        elif isinstance(event, ResultMessage):
            print(f"\n[STEP {step}] ResultMessage")
            if hasattr(event, "usage") and event.usage:
                u = event.usage if isinstance(event.usage, dict) else {}
                event_data["usage"] = u
                print(f"  Usage: input={u.get('input_tokens',0)} "
                      f"output={u.get('output_tokens',0)} "
                      f"cache_create={u.get('cache_creation_input_tokens',0)} "
                      f"cache_read={u.get('cache_read_input_tokens',0)}")
            if hasattr(event, "result"):
                event_data["final_result"] = str(event.result)[:500]
            if hasattr(event, "total_cost_usd"):
                event_data["cost_usd"] = event.total_cost_usd

        else:
            event_data["repr"] = str(event)[:300]

        trace.append(event_data)
    except Exception as e:
        print(f"\n[WARN] SDK exception after step {step}: {type(e).__name__}: {str(e)[:200]}")

    elapsed = time.time() - t0
    print(f"\n[DONE] Total steps: {step}, Time: {elapsed:.1f}s")

    return {
        "mode": mode_name,
        "steps": step,
        "time_s": round(elapsed, 2),
        "trace": trace,
    }


async def main_async() -> None:
    global _clio_connector, _clio_coordinator, _clio_storage

    print("=" * 70)
    print("VERBOSE REAL TEST: Full trace from request to response")
    print("=" * 70)

    question = (
        "Find scientific datasets that contain temperature measurements "
        "above 30 degrees Celsius. List the top 3 most relevant datasets."
    )

    with tempfile.TemporaryDirectory(prefix="clio_verbose_") as tmpdir:
        tmpdir = Path(tmpdir)

        # Setup CLIO (uses NDP-MCP internally)
        print("\n[SETUP] CLIO connecting to NDP-MCP...")
        _clio_storage = DuckDBStorage(database_path=tmpdir / "ndp.duckdb")
        _clio_connector = NDPConnector(namespace="ndp", storage=_clio_storage)
        _clio_connector.connect()

        mcp_client = NDPMCPClient(ndp_mcp_binary=NDP_MCP_BINARY)
        async with mcp_client.connect():
            tools = await mcp_client.list_tools()
            print(f"  NDP-MCP tools discovered: {tools}")
            for term in ["temperature", "pressure", "wind", "humidity", "glacier", "ocean"]:
                datasets = await mcp_client.search_datasets([term], limit=15)
                _clio_connector.index_datasets(datasets)
                print(f"  '{term}': {len(datasets)} datasets from NDP-MCP")

        profile = build_corpus_profile(_clio_storage, "ndp")
        print(f"  CLIO indexed: {profile.document_count} docs, {profile.chunk_count} chunks")
        _clio_coordinator = RetrievalCoordinator()

        # --- Mode A: direct NDP-MCP ---
        ndp_mcp_server = {
            "type": "stdio",
            "command": NDP_MCP_BINARY,
            "args": ["--transport", "stdio"],
        }

        result_a = await run_agent_with_trace(
            mode_name="MODE A: Claude Agent + NDP-MCP (direct)",
            question=question,
            mcp_servers={"ndp": ndp_mcp_server},
            system_prompt=(
                "You are a scientific data discovery agent. Use the NDP MCP tools "
                "(search_datasets, list_organizations, get_dataset_details) to find "
                "relevant datasets. List top 3 most relevant."
            ),
        )

        # --- Mode B: via CLIO ---
        clio_sdk_server = create_sdk_mcp_server(
            name="clio", version="1.0.0", tools=[clio_find_datasets],
        )

        result_b = await run_agent_with_trace(
            mode_name="MODE B: Claude Agent + CLIO (CLIO uses NDP-MCP)",
            question=question,
            mcp_servers={"clio": clio_sdk_server},
            system_prompt=(
                "You have ONE tool: clio_find_datasets. CLIO handles everything — "
                "it discovers data from NDP, profiles metadata, applies unit "
                "conversion, and returns ranked results. Use min_value, max_value, "
                "unit for unit-aware search. List top 3 relevant datasets."
            ),
        )

        _clio_connector.teardown()
        _clio_storage.teardown()

    # Save full trace
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "test": "Full trace comparison: NDP-MCP direct vs CLIO (uses NDP-MCP)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "clio_corpus": {
            "documents": profile.document_count,
            "chunks": profile.chunk_count,
            "measurements": profile.measurement_count,
        },
        "mode_a": result_a,
        "mode_b": result_b,
    }
    out_path = OUT_DIR / "agentic_full_trace.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n{'='*70}")
    print(f"Full trace saved: {out_path}")
    print(f"Mode A: {result_a['steps']} steps in {result_a['time_s']}s")
    print(f"Mode B: {result_b['steps']} steps in {result_b['time_s']}s")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
