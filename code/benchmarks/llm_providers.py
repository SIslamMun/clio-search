"""Multi-provider LLM query rewriting for benchmark evaluation.

Abstracts over Ollama, Google Gemini, Claude Agent SDK, llama-cpp-python,
OpenAI-compatible servers (LM Studio, vLLM, OpenAI, Together AI, Groq),
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
# 5. OpenAI-compatible providers (LM Studio, vLLM, OpenAI, Together, Groq)
# ===================================================================

class OpenAICompatibleProvider(LLMProvider):
    """Base for any OpenAI-compatible API (LM Studio, vLLM, OpenAI, Together, Groq)."""

    def __init__(
        self,
        *,
        provider_name: str,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        cost_input_per_m: float = 0.0,
        cost_output_per_m: float = 0.0,
    ) -> None:
        from openai import OpenAI

        kwargs: dict[str, Any] = {}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        self._client = OpenAI(**kwargs)
        self._provider_name = provider_name
        self._model = model
        self._cost_input = cost_input_per_m
        self._cost_output = cost_output_per_m

    def name(self) -> str:
        return f"{self._provider_name}/{self._model}"

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
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
            )
            latency = time.perf_counter() - t0

            raw_text = response.choices[0].message.content or ""
            rewritten, strategy, reasoning = _parse_raw_response(
                raw_text, original_query=query
            )

            usage = response.usage
            if usage is not None:
                prompt_tokens = usage.prompt_tokens or 0
                completion_tokens = usage.completion_tokens or 0
        except Exception as exc:  # noqa: BLE001
            latency = time.perf_counter() - t0
            error = str(exc)
            strategy = "done"
            reasoning = f"{self._provider_name} error: {exc}"

        total_tokens = prompt_tokens + completion_tokens
        cost = (
            prompt_tokens * self._cost_input / 1_000_000
            + completion_tokens * self._cost_output / 1_000_000
        )

        return RewriteResponse(
            original_query=query,
            rewritten_query=rewritten,
            strategy=strategy,
            reasoning=reasoning,
            metrics=LLMMetrics(
                provider=self._provider_name,
                model=self._model,
                latency_seconds=round(latency, 4),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=round(cost, 8),
                error=error,
            ),
        )


class LMStudioProvider(OpenAICompatibleProvider):
    """LM Studio local server (OpenAI-compatible API at localhost:1234)."""

    def __init__(
        self, model: str = "local-model", base_url: str = "http://localhost:1234/v1"
    ) -> None:
        super().__init__(
            provider_name="lmstudio",
            model=model,
            base_url=base_url,
            api_key="lm-studio",
            cost_input_per_m=0.0,
            cost_output_per_m=0.0,
        )


class VLLMProvider(OpenAICompatibleProvider):
    """vLLM server (OpenAI-compatible API, typically at localhost:8000)."""

    def __init__(
        self, model: str = "default", base_url: str = "http://localhost:8000/v1"
    ) -> None:
        super().__init__(
            provider_name="vllm",
            model=model,
            base_url=base_url,
            api_key="vllm",
            cost_input_per_m=0.0,
            cost_output_per_m=0.0,
        )


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI API (requires OPENAI_API_KEY environment variable)."""

    # Pricing per 1M tokens (USD) -- GPT-4o-mini
    _DEFAULT_INPUT_COST = 0.15
    _DEFAULT_OUTPUT_COST = 0.60

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        super().__init__(
            provider_name="openai",
            model=model,
            api_key=api_key,
            cost_input_per_m=self._DEFAULT_INPUT_COST,
            cost_output_per_m=self._DEFAULT_OUTPUT_COST,
        )


class TogetherProvider(OpenAICompatibleProvider):
    """Together AI (requires TOGETHER_API_KEY). OpenAI-compatible."""

    # Pricing per 1M tokens (USD) -- Llama-3-8B
    _DEFAULT_INPUT_COST = 0.20
    _DEFAULT_OUTPUT_COST = 0.20

    def __init__(self, model: str = "meta-llama/Llama-3-8b-chat-hf") -> None:
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError("TOGETHER_API_KEY environment variable is not set.")
        super().__init__(
            provider_name="together",
            model=model,
            base_url="https://api.together.xyz/v1",
            api_key=api_key,
            cost_input_per_m=self._DEFAULT_INPUT_COST,
            cost_output_per_m=self._DEFAULT_OUTPUT_COST,
        )


class GroqProvider(OpenAICompatibleProvider):
    """Groq (requires GROQ_API_KEY). OpenAI-compatible."""

    # Pricing per 1M tokens (USD) -- Llama3-8B
    _DEFAULT_INPUT_COST = 0.05
    _DEFAULT_OUTPUT_COST = 0.08

    def __init__(self, model: str = "llama3-8b-8192") -> None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set.")
        super().__init__(
            provider_name="groq",
            model=model,
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            cost_input_per_m=self._DEFAULT_INPUT_COST,
            cost_output_per_m=self._DEFAULT_OUTPUT_COST,
        )


# ===================================================================
# 10. Fallback (no LLM -- SI expansion only)
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

    # 1. Ollama — multiple models (prefer llama, qwen, mistral, deepseek)
    _OLLAMA_PREFERRED = [
        "llama3.1:8b", "llama3.2:latest", "qwen2.5:14b",
        "deepseek-r1:14b", "mistral:latest", "mistral-small3.2:24b",
    ]
    try:
        import ollama as _ollama  # type: ignore[import-untyped]
        _list_result = _ollama.list()
        # Handle both dict and pydantic response formats
        if hasattr(_list_result, "models"):
            available = {m.model for m in _list_result.models}
        else:
            available = {m["name"] for m in _list_result.get("models", [])}
        added_ollama = False
        for model_name in _OLLAMA_PREFERRED:
            if model_name in available:
                try:
                    providers.append(OllamaProvider(model=model_name))
                    print(f"  [OK] Ollama detected: {providers[-1].name()}")
                    added_ollama = True
                except Exception:
                    pass
        if not added_ollama:
            # Fall back to first available model
            providers.append(OllamaProvider())
            print(f"  [OK] Ollama detected: {providers[-1].name()}")
    except Exception as exc:
        print(f"  [--] Ollama not available: {exc}")

    # 2. Gemini — multiple models
    _GEMINI_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
    try:
        if os.environ.get("GOOGLE_API_KEY"):
            for gmodel in _GEMINI_MODELS:
                try:
                    providers.append(GeminiProvider(model=gmodel))
                    print(f"  [OK] Gemini detected: {providers[-1].name()}")
                except Exception:
                    pass
        else:
            print("  [--] Gemini not available: GOOGLE_API_KEY not set")
    except Exception as exc:
        print(f"  [--] Gemini not available: {exc}")

    # 3. Claude Agent SDK — multiple models
    _CLAUDE_MODELS = ["sonnet", "haiku", "opus"]
    try:
        for cmodel in _CLAUDE_MODELS:
            try:
                providers.append(ClaudeAgentProvider(model=cmodel))
                print(f"  [OK] Claude Agent SDK detected: {providers[-1].name()}")
            except Exception:
                pass
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

    # 5. LM Studio (localhost:1234)
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:1234/v1/models", method="GET")
        urllib.request.urlopen(req, timeout=2)
        providers.append(LMStudioProvider())
        print(f"  [OK] LM Studio detected: {providers[-1].name()}")
    except Exception as exc:
        print(f"  [--] LM Studio not available: {exc}")

    # 6. vLLM (localhost:8000)
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:8000/v1/models", method="GET")
        urllib.request.urlopen(req, timeout=2)
        providers.append(VLLMProvider())
        print(f"  [OK] vLLM detected: {providers[-1].name()}")
    except Exception as exc:
        print(f"  [--] vLLM not available: {exc}")

    # 7. OpenAI
    try:
        if os.environ.get("OPENAI_API_KEY"):
            providers.append(OpenAIProvider())
            print(f"  [OK] OpenAI detected: {providers[-1].name()}")
        else:
            print("  [--] OpenAI not available: OPENAI_API_KEY not set")
    except Exception as exc:
        print(f"  [--] OpenAI not available: {exc}")

    # 8. Together AI
    try:
        if os.environ.get("TOGETHER_API_KEY"):
            providers.append(TogetherProvider())
            print(f"  [OK] Together AI detected: {providers[-1].name()}")
        else:
            print("  [--] Together AI not available: TOGETHER_API_KEY not set")
    except Exception as exc:
        print(f"  [--] Together AI not available: {exc}")

    # 9. Groq
    try:
        if os.environ.get("GROQ_API_KEY"):
            providers.append(GroqProvider())
            print(f"  [OK] Groq detected: {providers[-1].name()}")
        else:
            print("  [--] Groq not available: GROQ_API_KEY not set")
    except Exception as exc:
        print(f"  [--] Groq not available: {exc}")

    # 10. Fallback -- always available
    providers.append(FallbackProvider())
    print(f"  [OK] Fallback provider: {providers[-1].name()}")

    return providers
