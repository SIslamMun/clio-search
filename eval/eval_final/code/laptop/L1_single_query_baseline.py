#!/usr/bin/env python3
"""L1 Step 1: Run ONE query with Claude agent + NDP-MCP only.

Save the full result (answer, tool calls, tokens, time) so we can
compare against CLIO in the next step.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[4]
_CODE = _REPO / "code"
sys.path.insert(0, str(_CODE / "src"))

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    query,
)

OUT_DIR = _REPO / "eval" / "eval_final" / "outputs"
NDP_MCP_BINARY = str(_CODE / ".venv" / "bin" / "ndp-mcp")

QUERY = "Find scientific datasets with temperature measurements above 30 degrees Celsius from weather station records in the National Data Platform."

SYSTEM_PROMPT = (
    "You are a scientific data discovery assistant working against the "
    "National Data Platform catalog. Use the MCP tools provided to search "
    "the catalog and inspect dataset details. Return a concise answer that "
    "names the matching datasets with their key details."
)


async def main() -> None:
    print("=" * 70)
    print("L1 Step 1: Claude agent + NDP-MCP (single query)")
    print("=" * 70)
    print(f"\nQuery: {QUERY}\n")

    ndp_server = {
        "type": "stdio",
        "command": NDP_MCP_BINARY,
        "args": ["--transport", "stdio"],
    }

    t_start = time.time()
    tool_calls: list[dict[str, Any]] = []
    final_answer = ""
    usage: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    all_events: list[dict[str, Any]] = []

    async for event in query(
        prompt=QUERY,
        options=ClaudeAgentOptions(
            mcp_servers={"ndp": ndp_server},
            system_prompt=SYSTEM_PROMPT,
            permission_mode="bypassPermissions",
            max_turns=12,
        ),
    ):
        event_record: dict[str, Any] = {"type": type(event).__name__}

        if isinstance(event, AssistantMessage):
            if hasattr(event, "content") and event.content:
                for block in event.content:
                    if isinstance(block, TextBlock) and block.text.strip():
                        final_answer = block.text
                        event_record["text"] = block.text[:500]
                    elif isinstance(block, ToolUseBlock):
                        tc = {
                            "name": block.name,
                            "input": getattr(block, "input", {}),
                        }
                        tool_calls.append(tc)
                        event_record["tool_use"] = tc

        elif isinstance(event, ResultMessage):
            if hasattr(event, "usage") and event.usage:
                u = event.usage if isinstance(event.usage, dict) else {}
                for k in usage:
                    usage[k] = u.get(k, usage[k])
                event_record["usage"] = dict(usage)

        all_events.append(event_record)

    elapsed = time.time() - t_start
    total_tokens = sum(usage.values())

    # Print results
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Tool calls: {len(tool_calls)}")
    for i, tc in enumerate(tool_calls):
        inp = json.dumps(tc["input"])[:100] if tc["input"] else ""
        print(f"  [{i+1}] {tc['name']}({inp})")
    print(f"\nToken usage:")
    for k, v in usage.items():
        print(f"  {k}: {v:,}")
    print(f"  TOTAL: {total_tokens:,}")
    print(f"\nFinal answer ({len(final_answer)} chars):")
    print(final_answer[:2000])

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": "L1 Step 1: single query baseline (Claude + NDP-MCP)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": QUERY,
        "system_prompt": SYSTEM_PROMPT,
        "elapsed_s": round(elapsed, 2),
        "num_tool_calls": len(tool_calls),
        "tool_calls": tool_calls,
        "usage": usage,
        "total_tokens": total_tokens,
        "final_answer": final_answer,
        "events": all_events,
    }
    out_path = OUT_DIR / "L1_step1_baseline.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
