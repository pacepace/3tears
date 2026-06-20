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

Upstream bug worked around: the package's bound-tool wrapper invokes a LangChain tool via the private
``tool._run(**args)`` path, which current langchain-core rejects ("missing ``config``"). We subclass
and override that one method to call the public ``tool.ainvoke(args)`` instead, so a caller's OWN
tools work (each tool call is one Agent-SDK turn). Filed upstream; the override is the local fix.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

#: A Claude subscription OAuth token (``claude setup-token``) starts with this; an API key does not.
OAUTH_TOKEN_PREFIX = "sk-ant-oat"

#: The ``ClaudeCodeChatModel`` constructor params we forward. The anthropic factory's other kwargs
#: (``timeout`` / ``max_retries`` / ``max_tokens`` / ``base_url`` …) are HTTP-API concepts the CLI
#: backend does not accept — dropped rather than passed through to a constructor error.
_FORWARDED_KWARGS = frozenset(
    {"system_prompt", "max_turns", "permission_mode", "allowed_tools", "disallowed_tools", "cwd", "fallback_model", "max_budget_usd"}
)


def is_subscription_token(credential: str) -> bool:
    """True when ``credential`` is a Claude subscription OAuth token (vs an API key)."""
    return credential.startswith(OAUTH_TOKEN_PREFIX)


def _subscription_model_cls() -> type:
    """The ``ClaudeCodeChatModel`` subclass with the bound-tool wrapper fixed (lazy import)."""
    from claude_agent_sdk import tool as sdk_tool
    from langchain_claude_code import ClaudeCodeChatModel

    class _SubscriptionChatModel(ClaudeCodeChatModel):  # type: ignore[misc]
        """``ClaudeCodeChatModel`` whose bound-tool wrapper invokes via the public ``ainvoke`` API."""

        def _wrap_langchain_tool(self, tool: BaseTool, schema: dict[str, Any]) -> Callable[..., Any]:
            props = schema.get("properties", {})
            tmap = {"string": str, "integer": int, "number": float, "boolean": bool, "array": list, "object": dict}
            param_types = {n: tmap.get(p.get("type", "string"), str) for n, p in props.items()}

            @sdk_tool(tool.name, tool.description or "", param_types)
            async def wrapped(args: dict[str, Any]) -> dict[str, Any]:
                try:
                    result = await tool.ainvoke(args)  # public API (handles config/run_manager); was tool._run
                    captured = self._tool_results_var.get(None) if self._tool_results_var else None
                    if captured is not None:
                        captured.append({"name": tool.name, "args": args, "result": result})
                    return {"content": [{"type": "text", "text": str(result)}]}
                except Exception as exc:  # surfaced to the model as a tool error so its loop continues
                    return {"content": [{"type": "text", "text": f"Error: {exc}"}], "is_error": True}

            return wrapped

    return _SubscriptionChatModel


def create_subscription_chat(model_name: str, token: str, **extra_kwargs: Any) -> BaseChatModel:
    """Build a Claude **subscription**-backed chat model for ``model_name``.

    ``token`` is the OAuth token (``sk-ant-oat…``); the Agent SDK / CLI reads the credential from the
    ``CLAUDE_CODE_OAUTH_TOKEN`` env var, so it is exported here. HTTP-API kwargs that the CLI backend
    cannot take are dropped (see :data:`_FORWARDED_KWARGS`).
    """
    os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = token
    opts = {k: v for k, v in extra_kwargs.items() if k in _FORWARDED_KWARGS}
    model: BaseChatModel = _subscription_model_cls()(model=model_name, **opts)
    return model
