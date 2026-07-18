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

Token-level streaming (post-Chunk-9 follow-up, see
``.prawduct/artifacts/3tears-change-claude-max-token-streaming.md`` in metallm for the full
sign-off): ``ClaudeCodeChatModel._astream`` sets ``include_partial_messages=True`` -- which makes
the Agent SDK subprocess actually emit granular ``StreamEvent`` deltas (the raw Anthropic
``content_block_delta``/``text_delta`` shape) -- but the method never handles ``StreamEvent`` at
all, only the terminal, whole-block ``AssistantMessage``. Every delta is silently dropped, so a
turn arrives as one or two large lumps instead of a real token stream. ``_astream`` is overridden
here to consume ``StreamEvent`` text deltas as they arrive and yield each one immediately, tracked
per content-block index so the terminal ``AssistantMessage`` never re-yields (and thereby doubles)
text a delta already streamed. A turn that somehow gets no ``StreamEvent`` at all (older CLI build,
future SDK regression) falls back to the base class's whole-message behavior for that block, so
this is strictly additive -- never worse than today.

Tool-name mangling (found live: every real call to a bound 3tears builtin tool -- e.g.
``threetears.web_search`` -- was silently denied under a subscription turn: "Claude requested
permissions to use ... but you haven't granted it yet", with nothing logged anywhere, while the
SAME tool worked fine on every other backend). Canonical 3tears tool names are dotted
(``threetears.web_search``, per ``BaseAgentTool.mcp_name()``); every other provider wrapper
(``anthropic.py``, ``openrouter.py``) mixes in ``NameTranslatingChatMixin`` because Anthropic's own
tool-name validator rejects the dot (``^[a-zA-Z0-9_-]{1,128}$``) -- but ``create_anthropic_chat``
routes an OAuth token straight to :func:`create_subscription_chat` BEFORE that mixin is applied, so
this backend never got the same treatment. The SDK/CLI's own ``bind_tools`` already tries to
auto-approve bound tools by deriving ``allowed_tools`` from each tool's raw ``.name``, but that
entry never matched the underscored identity the CLI normalizes tool calls to -- so the
auto-approval it was already attempting silently failed to match, and every call needed (and never
got) interactive approval. ``bind_tools`` here substitutes each dotted tool for a
``NameMangledToolProxy`` (:func:`~threetears.models.tool_name_translation.build_name_translation`,
the same translation the other providers apply) BEFORE the base class ever sees it, so the
``allowed_tools`` entry it derives matches exactly. Dispatch is unaffected (the proxy delegates
``_arun``/``_run`` straight through); event emission resolves the proxy's delegate to keep
tool-status events on the canonical dotted name metallm's own tracking expects.

Optional-parameter schema mistyping AND false-required advertising (found live: a bound tool's
``list``-typed parameter -- e.g. ``memory_search``'s ``ids: list[str] | None`` -- arrived at the
tool handler as the literal string ``"[]"`` instead of an empty list, failing pydantic validation
("Input should be a valid list") every time the model tried to pass it; separately, EVERY optional
filter on that same tool -- ``date_after``, ``date_before``, ``alias``, ... -- arrived populated
with an empty string on every call instead of being omitted, degrading search quality). Two
compounding root causes, both from how ``_wrap_langchain_tool`` used to hand ``@sdk_tool`` a bare
``{param_name: python_type}`` mapping instead of a full JSON Schema:

1. For an ``X | None`` field, pydantic's ``model_json_schema()`` renders
   ``anyOf: [{type: X}, {type: null}]`` with NO top-level ``type`` key on that property -- the
   base package's own schema-to-``param_types`` conversion (and our prior copy of it) did a bare
   ``prop.get("type", "string")``, which found nothing and silently defaulted EVERY optional
   parameter to ``string``. The SDK then advertised that parameter to the model as a string, so
   the model dutifully stringified whatever it meant to send (a list became ``"[]"``).
2. ``claude_agent_sdk.create_sdk_mcp_server``'s own schema builder, when handed that bare
   ``{name: type}`` mapping (rather than an already-``{"type": "object", "properties": ...}``-shaped
   dict), marks **every** key ``required`` (``"required": list(properties.keys())``) with no regard
   for which fields the original tool schema actually required. The model was then forced to invent
   a value for every optional filter on every call -- hence the empty-string placeholders.

``_wrap_langchain_tool`` now builds the full ``{"type": "object", "properties": ..., "required":
...}`` schema itself: each property keeps (or gains, resolved from ``anyOf``/``oneOf``) a definite
top-level ``type``, and ``required`` is copied verbatim from the original tool schema's own
``required`` list -- not synthesized from "every key present". Handing the SDK an
already-full-shaped schema also makes it skip its own required-everything path entirely (verified
by reading ``create_sdk_mcp_server``'s ``_build_schema``: it returns a dict verbatim, unmodified,
whenever ``"type"`` and ``"properties"`` are already top-level keys).
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence
from typing import Any, Callable

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import ensure_config
from langchain_core.tools import BaseTool
from langgraph.errors import GraphBubbleUp
from pydantic import Field

from threetears.models.tool_name_translation import NameMangledToolProxy, build_name_translation

from threetears.langgraph.events import (
    ToolCompletedEvent,
    ToolDispatchedEvent,
    ToolStartedEvent,
    dispatch_event,
)
from threetears.observe import get_logger

__all__ = ["OAUTH_TOKEN_PREFIX", "is_subscription_token", "create_subscription_chat"]

_logger = get_logger(__name__)

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
        "tools",  # see _SubscriptionChatModel.tools -- NOT a ClaudeCodeChatModel field,
        # declared on our subclass below so it actually binds (a bare kwarg the base
        # class doesn't declare is silently dropped by its `extra="ignore"` config).
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
    from claude_agent_sdk import AssistantMessage, ClaudeSDKClient, ResultMessage, StreamEvent
    from claude_agent_sdk import tool as sdk_tool
    from langchain_claude_code import ClaudeCodeChatModel

    class _SubscriptionChatModel(ClaudeCodeChatModel):
        """``ClaudeCodeChatModel`` whose bound-tool wrapper invokes via the public ``ainvoke`` API.

        Also default-denies Claude Code's own built-in tool belt (Bash, Read, Write, Edit,
        WebFetch, WebSearch, …) — see :attr:`tools`.
        """

        # ``ClaudeAgentOptions.tools`` gates whether ANY built-in tool is even available, fully
        # independent of ``allowed_tools``/``disallowed_tools`` (which only gate auto-approval /
        # removal of tools ``tools`` already made available). ``ClaudeCodeChatModel`` itself
        # declares no ``tools`` field and silently drops an unknown constructor kwarg (its
        # ``model_config`` is ``extra="ignore"``), so this MUST be declared here to bind at all.
        # Defaults to ``[]`` (every built-in disabled, only LangChain-bound tools available) --
        # default-deny rather than requiring every caller to remember to pass it, since a
        # subscription-backed turn otherwise gets the model server-side Bash / filesystem /
        # network access with zero caller-side gate.
        tools: list[str] | None = Field(default_factory=list)

        def _build_options(self, **overrides: Any) -> Any:
            """Force ``ClaudeAgentOptions.tools`` from :attr:`tools` unless a call overrides it."""
            overrides.setdefault("tools", self.tools)
            return super()._build_options(**overrides)

        def bind_tools(  # type: ignore[override] # narrows the supertype's Sequence[dict | type |
            # Callable | BaseTool] to Sequence[BaseTool] -- every real caller (metallm's own tool
            # loop) only ever binds BaseTool instances; the base class's own bind_tools makes the
            # same assumption in its body (`lc_tool.name` on each entry), so the wider supertype
            # signature is already unused in practice, not a contract this override narrows away.
            self,
            tools: Sequence[BaseTool],
            *,
            tool_choice: str | dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> Runnable[Any, Any]:
            """Mangle dotted tool names BEFORE the base class derives ``allowed_tools`` from them.

            Found live: every 3tears builtin's canonical name is dotted (``threetears.web_search``,
            per ``BaseAgentTool.mcp_name()``). The base class's own ``bind_tools`` already tries to
            auto-approve bound tools -- it builds ``allowed_tools = [f"mcp__langchain-tools__{name}"
            for name in tool_names]`` from each tool's raw ``.name`` -- but the SDK/CLI normalizes
            dots out of tool identities on the wire, so an entry built from the dotted name never
            matches the underscored identity the CLI actually checks against. Every real call to a
            dotted tool was silently denied: "Claude requested permissions to use ... but you
            haven't granted it yet" -- with no exception and nothing logged, because the denial
            happens inside the SDK/CLI boundary before any of our instrumented code ever runs.

            Substituting each dotted tool for a :class:`NameMangledToolProxy` here (the SAME
            translation :mod:`threetears.models.providers.anthropic` /
            :mod:`threetears.models.providers.openrouter` already apply for the identical
            ``^[a-zA-Z0-9_-]{1,128}$`` constraint on the direct-API path) means the base class's
            ``tool_names.append(lc_tool.name)`` sees the already-mangled name, so the
            ``allowed_tools`` entry it derives matches exactly what the CLI normalizes tool calls
            to. Dotless tools pass through unchanged (see :func:`build_name_translation`).
            """
            wire_tools, _reverse_map = build_name_translation(list(tools))
            return super().bind_tools(wire_tools, tool_choice=tool_choice, **kwargs)

        def _wrap_langchain_tool(self, tool: BaseTool, schema: dict[str, Any]) -> Callable[..., Any]:
            props = schema.get("properties", {})

            def _resolve_property(prop: dict[str, Any]) -> dict[str, Any]:
                """Resolve ``prop`` to a JSON Schema property with a definite top-level ``type``,
                unwrapping the ``anyOf``/``oneOf`` shape pydantic emits for an ``X | None`` field
                (no top-level ``type`` key there -- see the "Optional-parameter schema mistyping"
                note in this module's docstring)."""
                if "type" in prop:
                    return prop
                for branch in prop.get("anyOf") or prop.get("oneOf") or ():
                    branch_type = branch.get("type")
                    if branch_type and branch_type != "null":
                        resolved = dict(branch)
                        if "description" in prop:
                            resolved.setdefault("description", prop["description"])
                        return resolved
                return {"type": "string"}

            # A full JSON Schema (not a bare {name: python_type} map) so the SDK's own schema
            # builder uses it verbatim, INCLUDING our `required` list -- rather than its fallback
            # path, which marks every key required regardless of the source tool's actual schema.
            input_schema: dict[str, Any] = {
                "type": "object",
                "properties": {n: _resolve_property(p) for n, p in props.items()},
                "required": schema.get("required", []),
            }

            async def _emit(event: Any) -> None:
                # Best-effort: a broken event bus must never break tool
                # execution or the turn. No `config` is threaded through --
                # `dispatch_event` resolves the ambient RunnableConfig via
                # langchain_core's own context propagation, same as any
                # other custom-event dispatch not holding a config handle.
                # Logged (not silently swallowed): ambient-config propagation
                # across the SDK's subprocess-callback boundary is the one
                # thing this chunk couldn't verify by reading source alone --
                # a failure here is exactly the signal that verification needs.
                try:
                    await dispatch_event(event, config=None)
                except Exception as exc:  # prawduct:allow prawduct/broad-except -- tool-status is observability, never load-bearing for the turn
                    _logger.warning(
                        "subscription tool-status event dispatch failed",
                        extra={"extra_data": {"event_type": type(event).__name__, "error": str(exc)}},
                    )

            # `tool` may already be a `NameMangledToolProxy` by the time it reaches here (see
            # `bind_tools` below) -- its `.name` is the wire-mangled form, which is exactly what
            # `@sdk_tool` must register under so it matches the `allowed_tools` entry the base
            # class's own `bind_tools` derives from the SAME `.name`. Event emission, however,
            # must report the CANONICAL dotted name metallm's own tool-status tracking expects --
            # resolved through the proxy's `canonical_name` accessor when `tool` is a proxy.
            canonical_name = tool.canonical_name if isinstance(tool, NameMangledToolProxy) else tool.name

            @sdk_tool(tool.name, tool.description or "", input_schema)
            async def wrapped(args: dict[str, Any]) -> dict[str, Any]:
                await _emit(ToolDispatchedEvent(tool_name=canonical_name))
                await _emit(ToolStartedEvent(tool_name=canonical_name, tool_args=args))
                start = time.monotonic()
                try:
                    result = await tool.ainvoke(args)  # public API (handles config/run_manager); was tool._run
                    captured = self._tool_results_var.get(None) if self._tool_results_var else None
                    if captured is not None:
                        captured.append({"name": canonical_name, "args": args, "result": result})
                    await _emit(
                        ToolCompletedEvent(
                            tool_name=canonical_name,
                            tool_status="completed",
                            tool_duration_ms=int((time.monotonic() - start) * 1000),
                        )
                    )
                    return {"content": [{"type": "text", "text": str(result)}]}
                except GraphBubbleUp:
                    # LangGraph HITL control flow (interrupt()/Command et al.), NOT a tool failure --
                    # a caller's tool pausing the graph for human confirmation must propagate exactly
                    # like every other tool-calling path (see the base ToolNode / scriob's own
                    # ToolErrorMiddleware), never get reported to the model as a fake tool error. Found
                    # live: a bound tool calling `langgraph.types.interrupt(...)` was silently turned
                    # into a "the tool failed" reply the model then paraphrased as an apology, while the
                    # graph never actually paused -- no interrupt state was ever recorded, so a caller
                    # relying on a HITL confirm step never got one.
                    raise
                except Exception as exc:  # surfaced to the model as a tool error so its loop continues
                    await _emit(
                        ToolCompletedEvent(
                            tool_name=canonical_name,
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

        async def _astream(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> AsyncIterator[ChatGenerationChunk]:
            """Real token-level streaming: consumes the ``StreamEvent`` deltas the base
            class already requests (``include_partial_messages=True``) but never reads.

            Reimplements the base class's query/receive loop rather than wrapping it --
            ``ClaudeCodeChatModel._astream`` has no seam to inject a new branch into an
            already-running generator. Text deltas are yielded the moment they arrive,
            tracked per content-block index (``StreamEvent.event["index"]``) so the
            terminal ``AssistantMessage`` for a block whose text was already streamed is
            not re-yielded (which would double it). A block that produces no
            ``StreamEvent`` at all (older CLI build, future SDK regression) still gets
            its text emitted whole from the ``AssistantMessage`` -- the fallback the base
            class always used, so this is strictly additive.
            """
            config = kwargs.pop("_config", None)
            prompt, system_prompt = self._convert_messages(messages)
            if system_prompt and not self.system_prompt:
                kwargs["system_prompt"] = system_prompt
            kwargs["include_partial_messages"] = True
            cfg = ensure_config(config) if config is not None else None
            session_id = kwargs.pop("session_id", None) or kwargs.pop("resume", None)
            if cfg:
                session_id = session_id or cfg.get("configurable", {}).get("session_id")
            options = self._build_options(**kwargs)
            if session_id:
                options.resume = session_id
                options.continue_conversation = True

            tool_calls_buffer: list[dict[str, Any]] = []
            tool_results_buffer: list[dict[str, Any]] = []
            # Content-block indices whose text has already been streamed via a
            # StreamEvent delta THIS assistant message -- reset each time a new
            # AssistantMessage boundary is crossed, matching the SDK's own framing
            # (deltas for a message's blocks, then one AssistantMessage closing it).
            streamed_block_indices: set[int] = set()

            async with ClaudeSDKClient(options=options) as client:
                await client.query(prompt)

                async for msg in client.receive_response():
                    if isinstance(msg, StreamEvent):
                        event = msg.event or {}
                        if event.get("type") != "content_block_delta":
                            continue
                        delta = event.get("delta") or {}
                        if delta.get("type") != "text_delta":
                            continue
                        text = delta.get("text", "")
                        if not text:
                            continue
                        block_index = event.get("index")
                        if isinstance(block_index, int):
                            streamed_block_indices.add(block_index)
                        chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
                        if run_manager:
                            await run_manager.on_llm_new_token(text, chunk=chunk)
                        yield chunk

                    elif isinstance(msg, AssistantMessage):
                        text, tool_calls, tool_results = self._parse_assistant_message(msg)
                        # Fallback path only: if StreamEvent deltas already covered
                        # this message's text (the common case), re-yielding it here
                        # would double every character the client already received.
                        if text and not streamed_block_indices:
                            chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
                            if run_manager:
                                await run_manager.on_llm_new_token(text, chunk=chunk)
                            yield chunk
                        streamed_block_indices = set()
                        tool_calls_buffer.extend(tool_calls)
                        tool_results_buffer.extend(tool_results)

                    elif isinstance(msg, ResultMessage):
                        self._last_result = msg
                        generation_info: dict[str, Any] = {
                            "total_cost_usd": msg.total_cost_usd,
                            "duration_ms": msg.duration_ms,
                            "duration_api_ms": msg.duration_api_ms,
                            "session_id": msg.session_id,
                            "finish_reason": "stop" if not msg.is_error else "error",
                        }
                        if msg.usage:
                            generation_info["usage"] = msg.usage
                        if tool_calls_buffer:
                            generation_info["internal_tool_calls"] = tool_calls_buffer
                        if tool_results_buffer:
                            generation_info["internal_tool_results"] = tool_results_buffer

                        yield ChatGenerationChunk(
                            message=AIMessageChunk(content="", chunk_position="last"),
                            generation_info=generation_info,
                        )

    return _SubscriptionChatModel


def create_subscription_chat(model_name: str, token: str, **extra_kwargs: Any) -> BaseChatModel:
    """Build a Claude **subscription**-backed chat model for ``model_name``.

    ``token`` is the OAuth token (``sk-ant-oat…``), passed per **instance** via ``oauth_token`` — the
    package threads it into the SDK's per-subprocess ``ClaudeAgentOptions.env``, so concurrent models
    with different tokens never share process env (no global ``os.environ`` write, no cross-user race).
    HTTP-API kwargs the CLI backend cannot take are dropped (see :data:`_FORWARDED_KWARGS`).

    Claude Code's own built-in tool belt (Bash, Read, Write, Edit, WebFetch, WebSearch, …) is
    active by default and independent of any LangChain tools bound via ``bind_tools()``. A caller
    that wants the model to use ONLY its own bound tools should pass ``tools=[]`` — omitting it
    keeps the full built-in preset available.
    """
    opts = {k: v for k, v in extra_kwargs.items() if k in _FORWARDED_KWARGS}
    model: BaseChatModel = _subscription_model_cls()(model=model_name, oauth_token=token, **opts)
    return model
