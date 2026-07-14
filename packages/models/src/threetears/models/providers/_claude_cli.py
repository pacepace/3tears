"""Claude **subscription** backend for the anthropic provider (CLI / OAuth, not the HTTP API).

A Claude Pro/Max subscription used as a LangChain chat model, via ``langchain-claude-code``'s
``ClaudeCodeChatModel`` (which drives the Claude Code CLI / Claude Agent SDK). The anthropic provider
selects it when the credential is an OAuth token (``sk-ant-oat…`` from ``claude setup-token``) rather
than an API key — so the SAME Anthropic model ids resolve to a subscription-backed model with no new
provider, and it behaves like any other 3tears model (the factory attaches the usual cost/breaker
callbacks).

Runtime: needs Node + the Claude Code CLI on PATH (the SDK shells out to it). ``langchain-claude-code``
is an optional extra (``3tears-models[claude-cli]``), imported lazily here so the base install stays
free of it.

The token is passed per **instance** (``oauth_token=…``), which the package threads into the SDK's
per-subprocess ``ClaudeAgentOptions.env`` — NOT a global ``os.environ`` write. That matters for
multi-user concurrency: two users' subscription models build with distinct tokens and each drives
its own subprocess with its own credential, with no shared-process-env race.

Upstream bug worked around: the package's bound-tool wrapper invokes a LangChain tool via the private
``tool._run(**args)`` path, which current langchain-core rejects ("missing ``config``"). We subclass
and override that one method to call the public ``tool.ainvoke(args)`` instead, so a caller's OWN
tools work (each tool call is one Agent-SDK turn). Filed upstream; the override is the local fix.

Tool-status events (claude-max-convergence Chunk 7): ``ClaudeCodeChatModel._astream``/``_agenerate``
only ever fire ``on_llm_new_token`` — the SDK subprocess's internal tool-calling loop is invisible to
LangChain's own instrumentation (verified by reading the package source: ``astream_events`` never
emits ``on_tool_start``/``on_tool_end`` for tools invoked this way). ``_wrap_langchain_tool``'s
``wrapped`` closure is the one place with real-time visibility into each call, so it dispatches the
SAME typed events node-path tools already emit (:mod:`threetears.langgraph.events`) around
``tool.ainvoke`` -- a subscription-backed turn's tool-status chips render through the identical
consumer-side code path a normal turn's do, no new event vocabulary needed downstream.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from threetears.langgraph.events import (
    ToolCompletedEvent,
    ToolDispatchedEvent,
    ToolStartedEvent,
    dispatch_event,
)

__all__ = ["OAUTH_TOKEN_PREFIX", "is_subscription_token", "create_subscription_chat"]

#: A Claude subscription OAuth token (``claude setup-token``) starts with this; an API key does not.
OAUTH_TOKEN_PREFIX = "sk-ant-oat"

#: The ``ClaudeCodeChatModel`` constructor params we forward. The anthropic factory's other kwargs
#: (``timeout`` / ``max_retries`` / ``max_tokens`` / ``base_url`` …) are HTTP-API concepts the CLI
#: backend does not accept — dropped rather than passed through to a constructor error.
_FORWARDED_KWARGS = frozenset(
    {
        "system_prompt",
        "max_turns",
        "permission_mode",
        "allowed_tools",
        "disallowed_tools",
        "cwd",
        "fallback_model",
        "max_budget_usd",
    }
)


def is_subscription_token(credential: str) -> bool:
    """True when ``credential`` is a Claude subscription OAuth token (vs an API key)."""
    return credential.startswith(OAUTH_TOKEN_PREFIX)


def _subscription_model_cls() -> type:
    """The ``ClaudeCodeChatModel`` subclass with the bound-tool wrapper fixed (lazy import)."""
    from claude_agent_sdk import tool as sdk_tool
    from langchain_claude_code import ClaudeCodeChatModel

    class _SubscriptionChatModel(ClaudeCodeChatModel):
        """``ClaudeCodeChatModel`` whose bound-tool wrapper invokes via the public ``ainvoke`` API."""

        def _wrap_langchain_tool(self, tool: BaseTool, schema: dict[str, Any]) -> Callable[..., Any]:
            props = schema.get("properties", {})
            tmap = {"string": str, "integer": int, "number": float, "boolean": bool, "array": list, "object": dict}
            param_types = {n: tmap.get(p.get("type", "string"), str) for n, p in props.items()}

            async def _emit(event: Any) -> None:
                # Best-effort: a broken event bus must never break tool
                # execution or the turn. No `config` is threaded through --
                # `dispatch_event` resolves the ambient RunnableConfig via
                # langchain_core's own context propagation, same as any
                # other custom-event dispatch not holding a config handle.
                try:
                    await dispatch_event(event, config=None)
                except Exception:  # prawduct:allow prawduct/broad-except -- tool-status is observability, never load-bearing for the turn
                    pass

            @sdk_tool(tool.name, tool.description or "", param_types)
            async def wrapped(args: dict[str, Any]) -> dict[str, Any]:
                await _emit(ToolDispatchedEvent(tool_name=tool.name))
                await _emit(ToolStartedEvent(tool_name=tool.name, tool_args=args))
                start = time.monotonic()
                try:
                    result = await tool.ainvoke(args)  # public API (handles config/run_manager); was tool._run
                    captured = self._tool_results_var.get(None) if self._tool_results_var else None
                    if captured is not None:
                        captured.append({"name": tool.name, "args": args, "result": result})
                    await _emit(
                        ToolCompletedEvent(
                            tool_name=tool.name,
                            tool_status="completed",
                            tool_duration_ms=int((time.monotonic() - start) * 1000),
                        )
                    )
                    return {"content": [{"type": "text", "text": str(result)}]}
                except Exception as exc:  # surfaced to the model as a tool error so its loop continues
                    await _emit(
                        ToolCompletedEvent(
                            tool_name=tool.name,
                            tool_status="failed",
                            tool_duration_ms=int((time.monotonic() - start) * 1000),
                        )
                    )
                    return {"content": [{"type": "text", "text": f"Error: {exc}"}], "is_error": True}

            return wrapped  # type: ignore[return-value]  # @sdk_tool wraps `wrapped` into an SdkMcpTool,
            # which the base class's own `_wrap_langchain_tool -> Callable[..., Any]` signature doesn't
            # account for -- a stub gap in langchain-claude-code itself (pre-existing: this exact
            # decorator-then-return shape is unchanged by this chunk's edit, only newly surfaced because
            # 3tears-models isn't in CI's mypy invocation, so nothing here has been type-checked before).

    return _SubscriptionChatModel


def create_subscription_chat(model_name: str, token: str, **extra_kwargs: Any) -> BaseChatModel:
    """Build a Claude **subscription**-backed chat model for ``model_name``.

    ``token`` is the OAuth token (``sk-ant-oat…``), passed per **instance** via ``oauth_token`` — the
    package threads it into the SDK's per-subprocess ``ClaudeAgentOptions.env``, so concurrent models
    with different tokens never share process env (no global ``os.environ`` write, no cross-user race).
    HTTP-API kwargs the CLI backend cannot take are dropped (see :data:`_FORWARDED_KWARGS`).
    """
    opts = {k: v for k, v in extra_kwargs.items() if k in _FORWARDED_KWARGS}
    model: BaseChatModel = _subscription_model_cls()(model=model_name, oauth_token=token, **opts)
    return model
