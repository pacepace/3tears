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

LangGraph HITL interrupts survive this backend's MCP boundary (found live: a bound tool calling
``langgraph.types.interrupt(...)`` -- the standard confirm-mode "pause the graph and wait for a
human" pattern -- never actually paused anything under this backend; the model just reported the
tool as having failed). Two layers were swallowing the resulting ``GraphInterrupt``:

1. This module's own ``wrapped()`` closure had a bare ``except Exception``, and
   ``GraphInterrupt`` is a plain ``Exception`` subclass.
2. Even with that fixed, the ``mcp`` package's OWN ``Server.call_tool`` request handler
   (``mcp/server/lowlevel/server.py``, third-party, not ours) *also* catches every exception
   unconditionally and converts it to a normal ``CallToolResult(isError=True, ...)`` -- verified
   directly against the real ``mcp`` dispatch path, not just by reading its source. No exception
   of any kind can survive that boundary; re-raising harder inside ``wrapped()`` alone cannot fix
   this.

Because of (2), the interrupt cannot be *propagated* through the tool-call boundary at all -- it
has to be **captured** at the point it occurs (inside ``wrapped()``, the only code with real-time
visibility into the call, same rationale as the tool-status events above) and **replayed** from a
point that genuinely sits inside LangGraph's own call stack. ``_captured_interrupts_var`` (a
``ContextVar``, matching this class's existing ``_tool_results_var`` pattern -- a plain instance
attribute would race across concurrent turns sharing a model instance) holds whatever
``GraphInterrupt``s a turn's tool calls raised; both ``_astream`` and ``_agenerate`` set it fresh
per invocation and, once the underlying CLI turn completes, re-raise a combined ``GraphInterrupt``
if anything was captured -- from directly inside the model's own method, which DOES sit inside the
LangGraph agent node's call stack, so the graph genuinely pauses and checkpoints this time.

Resuming is the other half: this backend has no separate LangGraph "tools" node to replay in
isolation (unlike the native, non-CLI tool-calling path) -- the whole decide-and-call round-trip
lives inside ONE model call, so a resume means calling the model again from scratch, and a plain
``Command(resume=...)`` never touches ``messages`` -- the replayed conversation looks identical to
the interrupted attempt, with nothing telling the model a decision was made.
``_messages_with_resume_hint`` reads LangGraph's own ``__pregel_resuming`` configurable flag (via
``_is_resume_replay`` -- a pure signal, never ``interrupt()``'s own scratchpad ``.resume`` list,
which starts empty even on a genuine resume and would give the wrong answer) and, when this call
is a resume replay, appends one explicit synthetic human turn asking the model to retry the exact
same tool call -- it doesn't need the actual decision value, only the tool does. The underlying
CLI session is continued (``options.resume``/``continue_conversation``, already wired) so the
model has full context of what it just attempted. On that retry, the tool's OWN
``interrupt()`` call resolves via LangGraph's normal resume-value matching (positional, not tied to
which physical call site raised it) and the tool completes for real -- no capture, no replay,
this time.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
from typing import Any, Callable

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import ensure_config
from langchain_core.tools import BaseTool
from langgraph.errors import GraphBubbleUp, GraphInterrupt
from pydantic import Field

from threetears.models.claude_cli_isolation import claude_cli_isolation
from threetears.models.claude_cli_pool import (
    TOOL_SERVER_NAME,
    ClaudeCliPoolExhausted,
    ClaudeCliSessionError,
    claude_cli_pool,
)
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


#: Interrupts a bound tool raised THIS turn, captured because the ``mcp`` package's own
#: ``call_tool`` dispatch swallows any exception unconditionally (see the module docstring's
#: "LangGraph HITL interrupts" section) -- a ``ContextVar`` (not a plain instance attribute)
#: because a model instance can serve concurrent turns; each turn gets its own isolated list via
#: ``.set([])``/``.reset(token)`` around the call, mirroring this class's existing
#: ``_tool_results_var`` pattern.
_captured_interrupts_var: ContextVar[list[Any] | None] = ContextVar("_claude_cli_captured_interrupts", default=None)


def _is_resume_replay() -> bool:
    """Whether THIS call is a resume replay -- a real ``Command(resume=...)`` was submitted for
    this node's task -- read via LangGraph's own ``__pregel_resuming`` configurable flag.

    This is deliberately NOT the scratchpad's ``.resume`` list ``langgraph.types.interrupt()``
    itself consults: that list starts EMPTY even on a genuine resume and is populated lazily, one
    value at a time, exactly as each ``interrupt()`` call in the task consumes its own entry --
    peeking at it here (an earlier version of this function did) tells you nothing about whether a
    resume is in progress. ``__pregel_resuming`` is a pure signal, not a value -- reading it never
    disturbs ``interrupt()``'s own resolution, and this function doesn't need the actual decision
    value anyway (see :func:`_messages_with_resume_hint`).

    Returns ``False`` when there is no ambient graph config at all (the model invoked directly,
    outside any LangGraph run), or LangGraph's internal shape doesn't match what this reads --
    fully defensive, since this touches an underscore-prefixed LangGraph internal a future release
    could change; degrading to "not a resume" (send the prompt unchanged) is always the safe
    fallback here, never a crash.
    """
    try:
        from langgraph._internal._constants import CONFIG_KEY_RESUMING
        from langgraph.config import get_config

        resuming = bool(get_config()["configurable"].get(CONFIG_KEY_RESUMING))
    except Exception:  # the internals above may not exist/match in a future LangGraph release
        resuming = False
    return resuming


def _messages_with_resume_hint(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Append a synthetic human turn describing a pending HITL decision, when this call is a
    resume replay for an interrupt this backend captured (see the module docstring). This backend
    has no separate LangGraph "tools" node to replay in isolation -- the whole decide-and-call
    round-trip lives inside ONE model call -- so resuming means calling the model again from
    scratch, and a plain ``Command(resume=...)`` never edits ``messages``: the conversation history
    looks identical to the interrupted attempt, with nothing signalling a decision was made.

    The hint deliberately does not carry the actual decision value -- the model doesn't need it,
    only the tool does (via its own ``interrupt()`` call resolving normally on retry); the model's
    only job is to make the SAME call again.

    A no-op (returns ``messages`` unchanged) when this is not a resume replay.
    """
    if not _is_resume_replay():
        return messages
    hint = (
        "A pending action awaiting human confirmation has just been decided. Immediately retry "
        "the exact same tool call you were making, with the exact same arguments -- the tool "
        "itself will act correctly on the recorded decision."
    )
    return [*messages, HumanMessage(content=hint)]


def _content_text(content: Any) -> str:
    """The text of a message's content, whether it is a string or a list of content blocks.

    The base class did ``str(msg.content)``. On a caller that caches prompts the content is a list
    of blocks, so the CLI received the Python repr -- ``[{'type': 'text', 'text': '## SYSTEM\\n…',
    'cache_control': {...}}]`` with literal ``\\n`` sequences -- as the agent's persona.

    :param content: a LangChain message's ``content``
    :ptype content: Any
    :return: the text, blocks joined by blank lines; a non-text block is named, not dumped
    :rtype: str
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            if block.get("type") == "text" or "text" in block:
                parts.append(str(block.get("text", "")))
            else:
                parts.append(f"[{block.get('type', 'non-text')} content omitted]")
    return "\n\n".join(part for part in parts if part)


def _split_system(content: Any) -> tuple[str, str]:
    """A system message's content split at its last cache marker: ``(stable, variable)``.

    A caller that caches prompts marks where the stable part ends with ``cache_control`` on the last
    stable block; everything after it changes turn to turn. A string, or a list with no marker, is
    all stable.

    :param content: a system message's ``content``
    :ptype content: Any
    :return: the stable text and the variable text (either may be empty)
    :rtype: tuple[str, str]
    """
    if not isinstance(content, list):
        return _content_text(content), ""
    last_marked = -1
    for index, block in enumerate(content):
        if isinstance(block, dict) and block.get("cache_control"):
            last_marked = index
    if last_marked == -1:
        return _content_text(content), ""
    return _content_text(content[: last_marked + 1]), _content_text(content[last_marked + 1 :])


def _pooled_launch_options(options: Any) -> Any:
    """The options a pooled session launches with, derived from one call's options.

    Three differences, each because a live session must serve more than one call:

    - tool auto-approval is granted for the whole tool server rather than per tool, so a tool set
      swapped in later needs no relaunch;
    - the session launches with an empty placeholder tool server, replaced on every checkout;
    - partial messages are always on, so a streaming and a non-streaming call share a session
      (a non-streaming reader ignores the partial events).

    :param options: the call's ``ClaudeAgentOptions``
    :ptype options: Any
    :return: a new options object; the call's own is not mutated
    :rtype: Any
    """
    from claude_agent_sdk import create_sdk_mcp_server  # noqa: PLC0415

    per_tool_prefix = f"mcp__{TOOL_SERVER_NAME}__"
    allowed = [t for t in (options.allowed_tools or []) if not str(t).startswith(per_tool_prefix)]
    server_rule = f"mcp__{TOOL_SERVER_NAME}"
    if server_rule not in allowed:
        allowed.append(server_rule)
    return dataclasses.replace(
        options,
        allowed_tools=allowed,
        mcp_servers={TOOL_SERVER_NAME: create_sdk_mcp_server(name=TOOL_SERVER_NAME, version="1.0.0", tools=[])},
        include_partial_messages=True,
        env=dict(options.env or {}),
        extra_args=dict(options.extra_args or {}),
    )


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
            """Force ``tools`` from :attr:`tools`, and cut the CLI off from the host's Claude config.

            Isolation is applied to the built options rather than passed as overrides: ``env`` is a
            single dict the base class assembles from the token, so an override would replace the
            credential rather than add to it. See :mod:`threetears.models.claude_cli_isolation`
            for what an un-isolated CLI reads.
            """
            overrides.setdefault("tools", self.tools)
            options = super()._build_options(**overrides)
            isolation = claude_cli_isolation(self.oauth_token)
            # A caller that deliberately points the CLI at a configuration or directory of its
            # own has made that choice; only an unset one falls back to the isolated default.
            options.env = {**isolation.env, **(options.env or {})}
            if not options.cwd:
                options.cwd = isolation.cwd
            options.extra_args = {**(options.extra_args or {}), **isolation.extra_args}
            return options

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
                except GraphInterrupt as exc:
                    # LangGraph HITL control flow (interrupt()), NOT a tool failure -- but re-raising
                    # here is pointless: the `mcp` package's OWN call_tool dispatch (third-party, not
                    # ours) ALSO catches every exception unconditionally and turns it into a normal
                    # CallToolResult, no matter what we do at this layer (verified directly against
                    # the real `mcp` dispatch, not just by reading its source). So we capture it
                    # instead -- stash its payload where `_astream`/`_agenerate` (which DO sit inside
                    # LangGraph's own call stack) will find and re-raise it once this call returns --
                    # and report a benign, non-error result so the CLI subprocess's own turn winds
                    # down cleanly rather than crashing. See the module docstring for the full design.
                    bucket = _captured_interrupts_var.get()
                    if bucket is not None:
                        # GraphInterrupt carries its `interrupts` sequence as its sole positional
                        # arg (`GraphInterrupt(interrupts: Sequence[Interrupt] = ())`) -- it has no
                        # named attribute for it, just the plain exception `.args` tuple.
                        bucket.extend(exc.args[0] if exc.args else ())
                    await _emit(
                        ToolCompletedEvent(
                            tool_name=canonical_name,
                            tool_status="interrupted",
                            tool_duration_ms=int((time.monotonic() - start) * 1000),
                        )
                    )
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": "This action is awaiting a human confirmation decision.",
                            }
                        ]
                    }
                except GraphBubbleUp:
                    # Some OTHER LangGraph control-flow signal (not a value-carrying interrupt) --
                    # there is nothing to capture-and-replay for these (no `.interrupts` payload), so
                    # this can only attempt to propagate as before (still swallowed by the `mcp`
                    # layer either way) rather than pretend to handle a shape `interrupt()` never
                    # produces.
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

        def _convert_messages(self, messages: list[BaseMessage]) -> tuple[str, str | None]:
            """The CLI's query text and its system prompt, from LangChain messages.

            Returns the STABLE part of the system prompt as the system prompt and folds the
            variable part into the query, ahead of the conversation. A running CLI's system prompt
            cannot change, so this is what lets one CLI serve turn after turn while retrieved
            memory, tool results and notices change underneath it. Every content list is read as
            text rather than ``str()``-ed into a repr.

            :param messages: the conversation
            :ptype messages: list[BaseMessage]
            :return: ``(query_text, system_prompt)``
            :rtype: tuple[str, str | None]
            """
            stable_parts: list[str] = []
            variable_parts: list[str] = []
            conversation: list[str] = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    stable, variable = _split_system(msg.content)
                    if stable:
                        stable_parts.append(stable)
                    if variable:
                        variable_parts.append(variable)
                elif isinstance(msg, HumanMessage):
                    conversation.append(f"Human: {_content_text(msg.content)}")
                elif isinstance(msg, AIMessage):
                    content = _content_text(msg.content)
                    if getattr(msg, "tool_calls", None):
                        calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in msg.tool_calls)
                        content = f"{content}\n[Tool calls: {calls}]" if content else f"[Tool calls: {calls}]"
                    conversation.append(f"Assistant: {content}")
                elif isinstance(msg, ToolMessage):
                    conversation.append(f"Tool ({msg.name}): {_content_text(msg.content)}")
            query = "\n\n".join([*variable_parts, *conversation])
            system_prompt = "\n\n".join(stable_parts) if stable_parts else None
            return query, system_prompt

        @asynccontextmanager
        async def _cli_client(self, options: Any, *, pooled: bool) -> AsyncIterator[Any]:
            """A connected CLI client for one call: a pooled session when one can serve it.

            Falls back to a CLI of the call's own -- exactly the behaviour before pooling -- when
            the host turned pooling off, when the call resumes a stored CLI session (which isolation
            disables, so it cannot share), when every session stays busy past the checkout timeout,
            or when a pooled session cannot be started or prepared. A call is never refused for
            want of a pooled session.

            :param options: the call's ``ClaudeAgentOptions``
            :ptype options: Any
            :param pooled: ``False`` forces a CLI of the call's own
            :ptype pooled: bool
            :return: the connected ``ClaudeSDKClient``
            :rtype: AsyncIterator[Any]
            """
            pool = claude_cli_pool() if pooled else None
            async with AsyncExitStack() as stack:
                client: Any = None
                if pool is not None:
                    server = (
                        (options.mcp_servers or {}).get(TOOL_SERVER_NAME)
                        if isinstance(options.mcp_servers, dict)
                        else None
                    )
                    instance = server.get("instance") if isinstance(server, dict) else None
                    try:
                        client = await stack.enter_async_context(
                            pool.checkout(_pooled_launch_options(options), token=self.oauth_token, tool_server=instance)
                        )
                    except (ClaudeCliPoolExhausted, ClaudeCliSessionError) as exc:
                        _logger.info(
                            "No pooled Claude CLI for this call; running it on its own CLI",
                            extra={"extra_data": {"reason": str(exc)}},
                        )
                if client is None:
                    client = await stack.enter_async_context(ClaudeSDKClient(options=options))
                yield client

        async def _aquery(
            self,
            prompt: str,
            config: Any = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
            """The base class's non-streaming query, on a pooled CLI instead of a fresh one.

            Identical to ``ClaudeCodeChatModel._aquery`` apart from where the client comes from:
            the base opens ``ClaudeSDKClient(options=options)`` inline, which is the per-call
            subprocess this backend exists to avoid.

            :param prompt: the query text
            :ptype prompt: str
            :param config: the runnable config
            :ptype config: Any
            :param run_manager: the callback manager
            :ptype run_manager: AsyncCallbackManagerForLLMRun | None
            :return: ``(content, tool_calls, generation_info)``
            :rtype: tuple[str, list[dict[str, Any]], dict[str, Any]]
            """
            cfg = ensure_config(config) if config is not None else None
            session_id = kwargs.pop("session_id", None) or kwargs.pop("resume", None)
            if cfg:
                session_id = session_id or cfg.get("configurable", {}).get("session_id")
            options = self._build_options(**kwargs)
            if session_id:
                options.resume = session_id
                options.continue_conversation = True

            all_text: list[str] = []
            all_tool_calls: list[dict[str, Any]] = []
            all_tool_results: list[dict[str, Any]] = []
            generation_info: dict[str, Any] = {}
            tool_results_token = self._tool_results_var.set([])
            try:
                async with self._cli_client(options, pooled=not session_id) as client:
                    await client.query(prompt)
                    async for msg in client.receive_response():
                        if isinstance(msg, AssistantMessage):
                            text, tool_calls, tool_results = self._parse_assistant_message(msg)
                            if text:
                                all_text.append(text)
                                if run_manager:
                                    await run_manager.on_llm_new_token(text)
                            all_tool_calls.extend(tool_calls)
                            all_tool_results.extend(tool_results)
                        elif isinstance(msg, ResultMessage):
                            self._last_result = msg
                            generation_info = {
                                "total_cost_usd": msg.total_cost_usd,
                                "duration_ms": msg.duration_ms,
                                "duration_api_ms": msg.duration_api_ms,
                                "num_turns": msg.num_turns,
                                "session_id": msg.session_id,
                                "is_error": msg.is_error,
                            }
                            if msg.usage:
                                generation_info["usage"] = msg.usage
                captured = self._tool_results_var.get()
                if captured:
                    all_tool_results.extend(captured)
            finally:
                self._tool_results_var.reset(tool_results_token)
            if all_tool_results:
                generation_info["tool_results"] = all_tool_results
            return "\n".join(all_text), all_tool_calls, generation_info

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

            Also owns the interrupt capture/replay half of the module docstring's "LangGraph HITL
            interrupts" design: a resume-hint message is appended when this call is a replay
            (:func:`_messages_with_resume_hint`), and once the underlying CLI turn completes, any
            interrupt a bound tool raised (captured via :data:`_captured_interrupts_var` -- the
            ``mcp`` boundary swallows a raised one before it ever reaches here) is re-raised from
            THIS point, which genuinely sits inside LangGraph's own call stack.
            """
            messages = _messages_with_resume_hint(messages)
            interrupt_token = _captured_interrupts_var.set([])
            try:
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

                async with self._cli_client(options, pooled=not session_id) as client:
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
            finally:
                captured_interrupts = _captured_interrupts_var.get() or []
                _captured_interrupts_var.reset(interrupt_token)
            if captured_interrupts:
                raise GraphInterrupt(tuple(captured_interrupts))

        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            """Non-streaming twin of :meth:`_astream`'s interrupt capture/replay (see its docstring
            and the module docstring). The base class's ``_agenerate``/``_aquery`` drive the SAME
            MCP-bridged tool dispatch (:meth:`_wrap_langchain_tool`), so a captured interrupt has to
            be raised here too -- otherwise the non-streaming (HTTP route) turn path never pauses
            while the streaming (WS) path does.
            """
            messages = _messages_with_resume_hint(messages)
            interrupt_token = _captured_interrupts_var.set([])
            try:
                result = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            finally:
                captured_interrupts = _captured_interrupts_var.get() or []
                _captured_interrupts_var.reset(interrupt_token)
            if captured_interrupts:
                raise GraphInterrupt(tuple(captured_interrupts))
            return result

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
