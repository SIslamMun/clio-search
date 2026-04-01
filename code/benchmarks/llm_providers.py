"""Multi-provider LLM query rewriting for benchmark evaluation.

Abstracts over Ollama, Google Gemini, Claude Agent SDK, llama-cpp-python,
and a no-LLM fallback so that each can be benchmarked with the same
scientific query-rewriting prompt.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure the package is importable when run from the code/ directory.
# ---------------------------------------------------------------------------
_CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE_DIR / "src"))

from clio_agentic_search.retrieval.query_rewriter import (
    FallbackQueryRewriter,
    RewriteResult,
    _SYSTEM_PROMPT,
    _parse_llm_response,
)

# ===================================================================
# Data classes
# ===================================================================

@dataclass(frozen=True)
class LLMMetrics:
    provider: str
    model: str
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float  # estimated
    error: str | None = None


@dataclass(frozen=True)
class RewriteResponse:
    original_query: str
    rewritten_query: str
    strategy: str
    reasoning: str
    metrics: LLMMetrics


# ===================================================================
# Abstract provider
# ===================================================================

class LLMProvider(ABC):
    """Base class for all LLM providers used in the benchmark."""

    @abstractmethod
    def rewrite_query(self, query: str, context: str) -> RewriteResponse:
        """Rewrite *query* given retrieval *context*. Returns response + metrics."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable provider/model identifier (e.g. 'ollama/llama3')."""
        ...

    # ------------------------------------------------------------------
    # Adapter so providers plug into AgenticRetriever as a rewriter
    # ------------------------------------------------------------------
    def as_rewriter(self) -> _LLMProviderRewriterAdapter:
        """Return an object with the same .rewrite() interface expected by
        AgenticRetriever (matching FallbackQueryRewriter / QueryRewriter)."""
        return _LLMProviderRewriterAdapter(self)


class _LLMProviderRewriterAdapter:
    """Adapts an LLMProvider into the rewriter interface expected by
    AgenticRetriever (same signature as FallbackQueryRewriter.rewrite)."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def rewrite(
        self,
        *,
        query: str,
        retrieved_snippets: list[str],
        hop_number: int,
        max_hops: int,
    ) -> RewriteResult:
        snippets_text = (
            "\n---\n".join(retrieved_snippets) if retrieved_snippets else "(none)"
        )
        context = (
            f"Hop: {hop_number}/{max_hops}\n"
            f"Retrieved snippets:\n{snippets_text}\n\n"
            "Decide: should the query be refined? "
            "If the results already cover the information need, use strategy 'done' "
            "and return the original query unchanged. Otherwise pick expand/narrow/pivot "
            "and provide a rewritten query."
        )
        try:
            resp = self._provider.rewrite_query(query, context)
            return RewriteResult(
                original_query=resp.original_query,
                rewritten_query=resp.rewritten_query,
                strategy=resp.strategy,
                reasoning=resp.reasoning,
            )
        except Exception as exc:  # noqa: BLE001
            return RewriteResult(
                original_query=query,
                rewritten_query=query,
                strategy="done",
                reasoning=f"Provider error: {exc}",
            )


# ===================================================================
# Shared prompt builder
# ===================================================================

def _build_user_message(query: str, context: str) -> str:
    return (
        f"Current query: {query}\n"
        f"{context}\n\n"
        "Respond in JSON: "
        '{"strategy": "...", "rewritten_query": "...", "reasoning": "..."}'
    )


def _parse_raw_response(
    raw_text: str, *, original_query: str
) -> tuple[str, str, str]:
    """Parse strategy / rewritten_query / reasoning from raw LLM text.

    Returns (rewritten_query, strategy, reasoning).
    """
    result = _parse_llm_response(raw_text, original_query=original_query)
    return result.rewritten_query, result.strategy, result.reasoning


# ===================================================================
# 1. Ollama
# ===================================================================

class OllamaProvider(LLMProvider):
    """Query rewriter backed by a local Ollama model."""

    def __init__(self, model: str | None = None) -> None:
        import ollama as _ollama  # type: ignore[import-untyped]

        self._client = _ollama
        if model is None:
            # Auto-detect first available model
            models = _ollama.list()
            available = []
            if hasattr(models, "models"):
                available = [m.model for m in models.models]
            elif isinstance(models, dict):
                available = [m.get("model", m.get("name", "")) for m in models.get("models", [])]
            if not available:
                raise RuntimeError("Ollama is running but no models are pulled.")
            self._model = available[0]
        else:
            self._model = model

    def name(self) -> str:
        return f"ollama/{self._model}"

    def rewrite_query(self, query: str, context: str) -> RewriteResponse:
        user_msg = _build_user_message(query, context)
        t0 = time.perf_counter()
        error: str | None = None
        rewritten = query
        strategy = "done"
        reasoning = ""
        prompt_tokens = 0
        completion_tokens = 0

        try:
            response = self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            latency = time.perf_counter() - t0

            raw_text = response.get("message", {}).get("content", "")
            if hasattr(response, "message"):
                raw_text = getattr(response.message, "content", raw_text)

            rewritten, strategy, reasoning = _parse_raw_response(
                raw_text, original_query=query
            )

            # Token counts
            if hasattr(response, "prompt_eval_count"):
                prompt_tokens = getattr(response, "prompt_eval_count", 0) or 0
                completion_tokens = getattr(response, "eval_count", 0) or 0
            elif isinstance(response, dict):
                prompt_tokens = response.get("prompt_eval_count", 0) or 0
                completion_tokens = response.get("eval_count", 0) or 0
        except Exception as exc:  # noqa: BLE001
            latency = time.perf_counter() - t0
            error = str(exc)
            strategy = "done"
            reasoning = f"Ollama error: {exc}"

        total_tokens = prompt_tokens + completion_tokens
        return RewriteResponse(
            original_query=query,
            rewritten_query=rewritten,
            strategy=strategy,
            reasoning=reasoning,
            metrics=LLMMetrics(
                provider="ollama",
                model=self._model,
                latency_seconds=round(latency, 4),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=0.0,  # local
                error=error,
            ),
        )


# ===================================================================
# 2. Google Gemini
# ===================================================================

class GeminiProvider(LLMProvider):
    """Query rewriter backed by Google Gemini."""

    # Pricing per 1M tokens (USD) -- Gemini 1.5 Flash
    _INPUT_COST_PER_M = 0.075
    _OUTPUT_COST_PER_M = 0.30

    def __init__(self, model: str = "gemini-1.5-flash") -> None:
        import google.generativeai as genai  # type: ignore[import-untyped]

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable is not set.")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model,
            system_instruction=_SYSTEM_PROMPT,
        )
        self._model_name = model

    def name(self) -> str:
        return f"gemini/{self._model_name}"

    def rewrite_query(self, query: str, context: str) -> RewriteResponse:
        user_msg = _build_user_message(query, context)
        t0 = time.perf_counter()
        error: str | None = None
        rewritten = query
        strategy = "done"
        reasoning = ""
        prompt_tokens = 0
        completion_tokens = 0

        try:
            response = self._model.generate_content(user_msg)
            latency = time.perf_counter() - t0

            raw_text = response.text or ""
            rewritten, strategy, reasoning = _parse_raw_response(
                raw_text, original_query=query
            )

            # Token counts from usage_metadata
            meta = getattr(response, "usage_metadata", None)
            if meta is not None:
                prompt_tokens = getattr(meta, "prompt_token_count", 0) or 0
                completion_tokens = getattr(meta, "candidates_token_count", 0) or 0
        except Exception as exc:  # noqa: BLE001
            latency = time.perf_counter() - t0
            error = str(exc)
            strategy = "done"
            reasoning = f"Gemini error: {exc}"

        total_tokens = prompt_tokens + completion_tokens
        cost = (
            prompt_tokens * self._INPUT_COST_PER_M / 1_000_000
            + completion_tokens * self._OUTPUT_COST_PER_M / 1_000_000
        )

        return RewriteResponse(
            original_query=query,
            rewritten_query=rewritten,
            strategy=strategy,
            reasoning=reasoning,
            metrics=LLMMetrics(
                provider="gemini",
                model=self._model_name,
                latency_seconds=round(latency, 4),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=round(cost, 8),
                error=error,
            ),
        )


# ===================================================================
# 3. Claude Agent SDK
# ===================================================================

class ClaudeAgentProvider(LLMProvider):
    """Query rewriter backed by Claude via the claude-agent-sdk.

    Uses the async SDK and wraps calls with asyncio.run().
    No API key needed -- runs through the local Claude Code session.
    """

    # Pricing per 1M tokens (USD) -- Claude Sonnet (free via Max plan)
    _INPUT_COST_PER_M = 3.0
    _OUTPUT_COST_PER_M = 15.0

    def __init__(self, model: str = "sonnet") -> None:
        # Import check -- will raise if not installed
        import claude_agent_sdk  # type: ignore[import-untyped]  # noqa: F401
        self._model = model

    def name(self) -> str:
        return f"claude-agent-sdk/{self._model}"

    def rewrite_query(self, query: str, context: str) -> RewriteResponse:
        import asyncio

        user_msg = _build_user_message(query, context)
        full_prompt = f"System instruction: {_SYSTEM_PROMPT}\n\n{user_msg}"

        t0 = time.perf_counter()
        error: str | None = None
        rewritten = query
        strategy = "done"
        reasoning = ""

        try:
            raw_text = asyncio.run(self._ask(full_prompt))
            latency = time.perf_counter() - t0
            rewritten, strategy, reasoning = _parse_raw_response(
                raw_text, original_query=query
            )
        except Exception as exc:  # noqa: BLE001
            latency = time.perf_counter() - t0
            error = str(exc)
            strategy = "done"
            reasoning = f"Claude Agent SDK error: {exc}"

        # The SDK does not expose token counts; estimate from character length.
        est_prompt = len(full_prompt) // 4
        est_completion = len(rewritten) // 4
        cost = (
            est_prompt * self._INPUT_COST_PER_M / 1_000_000
            + est_completion * self._OUTPUT_COST_PER_M / 1_000_000
        )

        return RewriteResponse(
            original_query=query,
            rewritten_query=rewritten,
            strategy=strategy,
            reasoning=reasoning,
            metrics=LLMMetrics(
                provider="claude-agent-sdk",
                model=self._model,
                latency_seconds=round(latency, 4),
                prompt_tokens=est_prompt,
                completion_tokens=est_completion,
                total_tokens=est_prompt + est_completion,
                cost_usd=round(cost, 8),
                error=error,
            ),
        )

    @staticmethod
    async def _ask(prompt: str) -> str:
        from claude_agent_sdk import (  # type: ignore[import-untyped]
            AssistantMessage,
            ClaudeAgentOptions,
            query as sdk_query,
        )

        text_parts: list[str] = []
        async for msg in sdk_query(
            prompt=prompt,
            options=ClaudeAgentOptions(max_turns=1),
        ):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
        return "".join(text_parts)


# ===================================================================
# 4. llama-cpp-python
# ===================================================================

class LlamaCppProvider(LLMProvider):
    """Query rewriter backed by a local GGUF model via llama-cpp-python."""

    def __init__(self, model_path: str) -> None:
        from llama_cpp import Llama  # type: ignore[import-untyped]

        if not Path(model_path).is_file():
            raise FileNotFoundError(f"GGUF model not found: {model_path}")
        self._llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)
        self._model_path = model_path
        self._model_name = Path(model_path).stem

    def name(self) -> str:
        return f"llama-cpp/{self._model_name}"

    def rewrite_query(self, query: str, context: str) -> RewriteResponse:
        user_msg = _build_user_message(query, context)
        full_prompt = (
            f"<|system|>\n{_SYSTEM_PROMPT}\n<|end|>\n"
            f"<|user|>\n{user_msg}\n<|end|>\n"
            f"<|assistant|>\n"
        )

        t0 = time.perf_counter()
        error: str | None = None
        rewritten = query
        strategy = "done"
        reasoning = ""
        prompt_tokens = 0
        completion_tokens = 0

        try:
            output = self._llm(full_prompt, max_tokens=512, stop=["<|end|>"])
            latency = time.perf_counter() - t0

            raw_text = output["choices"][0]["text"].strip()
            rewritten, strategy, reasoning = _parse_raw_response(
                raw_text, original_query=query
            )

            usage = output.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
        except Exception as exc:  # noqa: BLE001
            latency = time.perf_counter() - t0
            error = str(exc)
            strategy = "done"
            reasoning = f"llama-cpp error: {exc}"

        total_tokens = prompt_tokens + completion_tokens
        return RewriteResponse(
            original_query=query,
            rewritten_query=rewritten,
            strategy=strategy,
            reasoning=reasoning,
            metrics=LLMMetrics(
                provider="llama-cpp",
                model=self._model_name,
                latency_seconds=round(latency, 4),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=0.0,  # local
                error=error,
            ),
        )


# ===================================================================
# 5. Fallback (no LLM -- SI expansion only)
# ===================================================================

class FallbackProvider(LLMProvider):
    """Wraps the existing FallbackQueryRewriter (no LLM) as a provider
    so it can be compared against LLM-based providers."""

    def __init__(self) -> None:
        self._rewriter = FallbackQueryRewriter()

    def name(self) -> str:
        return "fallback/si-expansion"

    def rewrite_query(self, query: str, context: str) -> RewriteResponse:
        t0 = time.perf_counter()
        result = self._rewriter.rewrite(
            query=query,
            retrieved_snippets=[],
            hop_number=1,
            max_hops=1,
        )
        latency = time.perf_counter() - t0

        # Estimate tokens from string length (for fair comparison columns)
        est_prompt = len(query) // 4
        est_completion = len(result.rewritten_query) // 4

        return RewriteResponse(
            original_query=result.original_query,
            rewritten_query=result.rewritten_query,
            strategy=result.strategy,
            reasoning=result.reasoning,
            metrics=LLMMetrics(
                provider="fallback",
                model="si-expansion",
                latency_seconds=round(latency, 6),
                prompt_tokens=est_prompt,
                completion_tokens=est_completion,
                total_tokens=est_prompt + est_completion,
                cost_usd=0.0,
                error=None,
            ),
        )


# ===================================================================
# Provider discovery
# ===================================================================

def discover_providers(
    *,
    gguf_model_path: str | None = None,
) -> list[LLMProvider]:
    """Auto-detect which LLM providers are available and return instances.

    Always returns at least the FallbackProvider.
    """
    providers: list[LLMProvider] = []

    # 1. Ollama
    try:
        import ollama as _ollama  # type: ignore[import-untyped]
        _ollama.list()  # will raise if server is not running
        providers.append(OllamaProvider())
        print(f"  [OK] Ollama detected: {providers[-1].name()}")
    except Exception as exc:
        print(f"  [--] Ollama not available: {exc}")

    # 2. Gemini
    try:
        if os.environ.get("GOOGLE_API_KEY"):
            providers.append(GeminiProvider())
            print(f"  [OK] Gemini detected: {providers[-1].name()}")
        else:
            print("  [--] Gemini not available: GOOGLE_API_KEY not set")
    except Exception as exc:
        print(f"  [--] Gemini not available: {exc}")

    # 3. Claude Agent SDK
    try:
        providers.append(ClaudeAgentProvider())
        print(f"  [OK] Claude Agent SDK detected: {providers[-1].name()}")
    except Exception as exc:
        print(f"  [--] Claude Agent SDK not available: {exc}")

    # 4. llama-cpp-python
    if gguf_model_path:
        try:
            providers.append(LlamaCppProvider(gguf_model_path))
            print(f"  [OK] llama-cpp detected: {providers[-1].name()}")
        except Exception as exc:
            print(f"  [--] llama-cpp not available: {exc}")
    else:
        # Try to find a GGUF file in common locations
        search_dirs = [
            Path.home() / ".cache" / "llama-cpp",
            Path.home() / "models",
            Path.cwd() / "models",
        ]
        gguf_found = None
        for d in search_dirs:
            if d.is_dir():
                for f in d.glob("*.gguf"):
                    gguf_found = str(f)
                    break
            if gguf_found:
                break

        if gguf_found:
            try:
                providers.append(LlamaCppProvider(gguf_found))
                print(f"  [OK] llama-cpp detected: {providers[-1].name()}")
            except Exception as exc:
                print(f"  [--] llama-cpp not available: {exc}")
        else:
            try:
                import llama_cpp  # type: ignore[import-untyped]  # noqa: F401
                print("  [--] llama-cpp installed but no GGUF model found")
            except ImportError:
                print("  [--] llama-cpp not installed")

    # 5. Fallback -- always available
    providers.append(FallbackProvider())
    print(f"  [OK] Fallback provider: {providers[-1].name()}")

    return providers
