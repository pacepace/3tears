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

Tool calls belong to the caller. The base package runs a bound tool INSIDE the CLI: the model
asks, the CLI calls the tool over an in-process MCP server, feeds the result back and carries on,
and the caller only ever sees the finished text -- ``AIMessage.tool_calls`` stays empty and the
calls travel in ``response_metadata["internal_tool_calls"]``. Every other chat model hands tool
calls back, and a caller's graph decides what runs: approval gates, output shaping, a ledger of
what the turn did, loading more tools mid-turn, a round cap. Under a subscription all of that was
skipped (found live: ``tool_search`` ran, the turn recorded no call and no ledger row, and the
tools it found could never be bound).

So this backend keeps the standard contract. Each call is ONE model turn (``--max-turns 1``,
forced). Verified against the bundled CLI: the model's tool-use blocks -- parallel ones included --
are all emitted, the CLI stops with ``error_max_turns`` before a second model turn, and the same
session answers the next query normally. The bound-tool handler the CLI calls in between runs
nothing; it answers with a placeholder no model turn ever reads. The tool-use blocks become
``AIMessage.tool_calls`` under the caller's own tool names, and the caller's next round arrives
with the results as ordinary history.

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
tool calls handed back under the canonical dotted name the caller bound.

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

import contextvars
import dataclasses
import json
from collections.abc import AsyncIterator, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, Callable

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import ensure_config
from langchain_core.tools import BaseTool
from pydantic import Field

from threetears.models.claude_cli_isolation import claude_cli_isolation
from threetears.models.claude_cli_pool import (
    TOOL_SERVER_NAME,
    ClaudeCliPoolExhausted,
    ClaudeCliSessionError,
    claude_cli_pool,
)
from threetears.models.tool_name_translation import NameMangledToolProxy, build_name_translation

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


#: Model turns per call. One: the call ends where the model asks for tools, and the caller runs them.
_MODEL_TURNS_PER_CALL = 1

#: The CLI's name for a tool on the bound tool server is this prefix plus the tool's wire name.
_BOUND_TOOL_PREFIX = f"mcp__{TOOL_SERVER_NAME}__"

#: What the CLI's tool handler answers. No model turn reads it: the call ends first.
_HANDED_BACK = "This tool call was handed to the caller."

#: The CLI's own tool for a structured answer. Asked for a schema (``--json-schema``), the model
#: answers by calling it, and the CLI puts the answer on ``ResultMessage.structured_output``. It is
#: the CLI's, never the caller's: a caller's tool arrives under :data:`_BOUND_TOOL_PREFIX`.
_STRUCTURED_OUTPUT_TOOL = "StructuredOutput"


def _output_format(output_config: Any) -> dict[str, Any]:
    """The Agent SDK's ``output_format`` for the Messages API's ``output_config``.

    Structured output is asked for the way the anthropic provider spells it everywhere else --
    ``output_config={"format": {"type": "json_schema", "schema": ...}}``, from
    ``anthropic_structured_output_kwargs`` -- because a subscription token resolves to this backend
    under the SAME provider, and a caller cannot tell which one it holds. The SDK has no
    ``output_config``: its options drop an unknown key without a word, so the schema never reached
    the CLI and the model answered in prose (found live: every split on a subscription decision
    model refused, "the provider was asked for JSON and did not return it"). The SDK spells the same
    directive ``output_format``, which the CLI takes as ``--json-schema``.

    :param output_config: the ``output_config`` a caller bound
    :ptype output_config: Any
    :return: the equivalent ``output_format``
    :rtype: dict[str, Any]
    :raises ValueError: when ``output_config`` asks for anything but a JSON schema -- a directive
        this backend cannot honour is refused, never dropped
    """
    fmt = output_config.get("format") if isinstance(output_config, dict) else None
    if not isinstance(fmt, dict) or fmt.get("type") != "json_schema" or not isinstance(fmt.get("schema"), dict):
        raise ValueError(f"a subscription model can only honour a json_schema output_config, not {output_config!r}")
    return {"type": "json_schema", "schema": fmt["schema"]}


def is_subscription_token(credential: str) -> bool:
    """True when ``credential`` is a Claude subscription OAuth token (vs an API key)."""
    return credential.startswith(OAUTH_TOKEN_PREFIX)


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
    from claude_agent_sdk import AssistantMessage, ClaudeSDKClient, ResultMessage, StreamEvent, TextBlock
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
            if "output_config" in overrides:
                overrides["output_format"] = _output_format(overrides.pop("output_config"))
            from claude_agent_sdk import ClaudeAgentOptions  # noqa: PLC0415

            # The base class keeps only keys ClaudeAgentOptions declares and drops the rest without
            # a word. A dropped key is a directive the caller believes was sent -- which is how a
            # schema went missing -- so each one is named.
            dropped = sorted(
                k for k in overrides if not k.startswith("_") and k not in ClaudeAgentOptions.__dataclass_fields__
            )
            if dropped:
                _logger.warning(
                    "A subscription model has no option for these; they were not sent",
                    extra={"extra_data": {"dropped": dropped}},
                )
            options = super()._build_options(**overrides)
            isolation = claude_cli_isolation(self.oauth_token)
            # A caller that deliberately points the CLI at a configuration or directory of its
            # own has made that choice; only an unset one falls back to the isolated default.
            options.env = {**isolation.env, **(options.env or {})}
            if not options.cwd:
                options.cwd = isolation.cwd
            options.extra_args = {**(options.extra_args or {}), **isolation.extra_args}
            # One model turn per call, whatever was asked for: the call ends where the model asks
            # for tools, and the caller runs them (see the module docstring).
            options.max_turns = _MODEL_TURNS_PER_CALL
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

            @sdk_tool(tool.name, tool.description or "", input_schema)
            async def wrapped(args: dict[str, Any]) -> dict[str, Any]:
                # Runs nothing: the call is handed back to the caller (see the module docstring).
                # The CLI still calls this between the model's tool use and its turn limit, and
                # only a model turn would read the answer -- there is none.
                return {"content": [{"type": "text", "text": _HANDED_BACK}]}

            return wrapped  # type: ignore[return-value]  # @sdk_tool wraps `wrapped` into an SdkMcpTool,
            # which the base class's own `_wrap_langchain_tool -> Callable[..., Any]` signature doesn't
            # account for -- a stub gap in langchain-claude-code itself.

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
            # A ToolMessage need not carry its tool's name; the call it answers does.
            called = {
                call["id"]: call["name"]
                for msg in messages
                if isinstance(msg, AIMessage)
                for call in (msg.tool_calls or [])
                if call.get("id")
            }
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
                    name = msg.name or called.get(msg.tool_call_id) or "tool"
                    conversation.append(f"Tool ({name}): {_content_text(msg.content)}")
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
            # The borrower's context, captured now -- inside its own run, with its interrupt list,
            # tool-result list, callbacks and runnable config set. Tool calls on a reused CLI run in
            # a copy of it; without that they ran in whichever caller first started the CLI.
            call_context = contextvars.copy_context()
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
                            pool.checkout(
                                _pooled_launch_options(options),
                                token=self.oauth_token,
                                tool_server=instance,
                                call_context=call_context,
                            )
                        )
                    except (ClaudeCliPoolExhausted, ClaudeCliSessionError) as exc:
                        _logger.info(
                            "No pooled Claude CLI for this call; running it on its own CLI",
                            extra={"extra_data": {"reason": str(exc)}},
                        )
                if client is None:
                    client = await stack.enter_async_context(ClaudeSDKClient(options=options))
                yield client

        def _caller_tool_calls(self, blocks: list[Any]) -> list[dict[str, Any]]:
            """The model's tool-use blocks as ``tool_calls`` under the names the caller bound.

            The CLI names a bound tool ``mcp__langchain-tools__<wire name>``, and a dotted name went
            onto the wire underscored (see :meth:`bind_tools`). Both are undone, so a caller looks the
            call up under the name it bound. A tool-use block for anything else -- the CLI's own
            built-ins, when a caller turned some on -- keeps its name.

            :param blocks: an assistant message's content blocks
            :ptype blocks: list[Any]
            :return: LangChain tool calls
            :rtype: list[dict[str, Any]]
            """
            from claude_agent_sdk import ToolUseBlock  # noqa: PLC0415

            canonical = {
                bound.name: (bound.canonical_name if isinstance(bound, NameMangledToolProxy) else bound.name)
                for bound in (self._bound_tools or [])
            }
            calls: list[dict[str, Any]] = []
            for block in blocks:
                if not isinstance(block, ToolUseBlock) or block.name == _STRUCTURED_OUTPUT_TOOL:
                    continue
                name = block.name
                if name.startswith(_BOUND_TOOL_PREFIX):
                    wire = name[len(_BOUND_TOOL_PREFIX) :]
                    name = canonical.get(wire, wire)
                calls.append({"id": block.id, "name": name, "args": dict(block.input or {}), "type": "tool_call"})
            return calls

        async def _aquery(
            self,
            prompt: str,
            config: Any = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
            """The base class's non-streaming query, on a pooled CLI, with tool calls handed back.

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
            tool_calls: list[dict[str, Any]] = []
            generation_info: dict[str, Any] = {}
            async with self._cli_client(options, pooled=not session_id) as client:
                await client.query(prompt)
                async for msg in client.receive_response():
                    if isinstance(msg, AssistantMessage):
                        text = "\n".join(b.text for b in msg.content if isinstance(b, TextBlock))
                        if text:
                            all_text.append(text)
                            if run_manager:
                                await run_manager.on_llm_new_token(text)
                        tool_calls.extend(self._caller_tool_calls(msg.content))
                    elif isinstance(msg, ResultMessage):
                        self._last_result = msg
                        generation_info = _generation_info(msg, tool_calls)
                        if options.output_format is not None and msg.structured_output is not None:
                            # The answer is the structured one. The model's prose before it is
                            # not: asked for a shape, Haiku still wrote a paragraph first.
                            return json.dumps(msg.structured_output), tool_calls, generation_info
            return "\n".join(all_text), tool_calls, generation_info

        def _create_ai_message(
            self,
            content: str,
            tool_calls: list[dict[str, Any]] | None = None,
            generation_info: dict[str, Any] | None = None,
        ) -> AIMessage:
            """An ``AIMessage`` carrying its tool calls and usage the way every chat model does.

            The base class moved tool calls into ``response_metadata`` so a caller would not run
            calls the CLI had already run. None has run here.
            """
            info = dict(generation_info or {})
            usage = _usage_metadata(info.get("usage"))
            return AIMessage(
                content=content,
                tool_calls=list(tool_calls or []),
                response_metadata=info,
                usage_metadata=usage,
            )

        async def _astream(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> AsyncIterator[ChatGenerationChunk]:
            """Real token-level streaming, with the model's tool calls handed back on the last chunk.

            Reimplements the base class's query/receive loop rather than wrapping it --
            ``ClaudeCodeChatModel._astream`` has no seam to inject a new branch into an
            already-running generator. Text deltas (``StreamEvent``, which the base class requests
            but never reads) are yielded the moment they arrive, tracked per content-block index so
            the terminal ``AssistantMessage`` for a block whose text was already streamed is not
            re-yielded (which would double it). A block that produces no ``StreamEvent`` at all
            still gets its text emitted whole from the ``AssistantMessage``.
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

            tool_calls: list[dict[str, Any]] = []
            # Content-block indices whose text has already been streamed via a StreamEvent delta
            # THIS assistant message -- reset each time a new AssistantMessage boundary is crossed,
            # matching the SDK's own framing (deltas for a message's blocks, then one
            # AssistantMessage closing it).
            streamed_block_indices: set[int] = set()
            # Asked for a shape, the answer is the structured output on the result, and the prose the
            # model writes on the way is held rather than streamed: a caller parsing the stream as
            # JSON must not be handed a paragraph first. Held, not thrown away -- when no structured
            # answer comes, the prose is what the caller gets, so its failure names what was said.
            structured = options.output_format is not None
            held: list[str] = []

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
                        if structured:
                            continue
                        chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
                        if run_manager:
                            await run_manager.on_llm_new_token(text, chunk=chunk)
                        yield chunk

                    elif isinstance(msg, AssistantMessage):
                        text = "\n".join(b.text for b in msg.content if isinstance(b, TextBlock))
                        if structured:
                            if text:
                                held.append(text)
                            tool_calls.extend(self._caller_tool_calls(msg.content))
                            continue
                        # Fallback path only: if StreamEvent deltas already covered this message's
                        # text (the common case), re-yielding it here would double every character.
                        if text and not streamed_block_indices:
                            chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
                            if run_manager:
                                await run_manager.on_llm_new_token(text, chunk=chunk)
                            yield chunk
                        streamed_block_indices = set()
                        tool_calls.extend(self._caller_tool_calls(msg.content))

                    elif isinstance(msg, ResultMessage):
                        self._last_result = msg
                        generation_info = _generation_info(msg, tool_calls)
                        usage = _usage_metadata(msg.usage)
                        content = ""
                        if structured:
                            content = (
                                json.dumps(msg.structured_output)
                                if msg.structured_output is not None
                                else "\n".join(held)
                            )
                            if content and run_manager:
                                await run_manager.on_llm_new_token(content)
                        yield ChatGenerationChunk(
                            message=AIMessageChunk(
                                content=content,
                                chunk_position="last",
                                tool_call_chunks=[
                                    {
                                        "id": call["id"],
                                        "name": call["name"],
                                        "args": json.dumps(call["args"]),
                                        "index": index,
                                        "type": "tool_call_chunk",
                                    }
                                    for index, call in enumerate(tool_calls)
                                ],
                                response_metadata=generation_info,
                                usage_metadata=usage,
                            ),
                            generation_info=generation_info,
                        )

    return _SubscriptionChatModel


def _generation_info(result: Any, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """What a call's ``ResultMessage`` says, as generation info.

    A call that asked for tools ends on the CLI's turn limit (``error_max_turns``). That is the
    designed end of such a call, not a failure.

    :param result: the CLI's ``ResultMessage``
    :ptype result: Any
    :param tool_calls: the tool calls the call handed back
    :ptype tool_calls: list[dict[str, Any]]
    :return: generation info
    :rtype: dict[str, Any]
    """
    ended_for_tools = bool(tool_calls) and result.subtype == "error_max_turns"
    failed = bool(result.is_error) and not ended_for_tools
    info: dict[str, Any] = {
        "total_cost_usd": result.total_cost_usd,
        "duration_ms": result.duration_ms,
        "duration_api_ms": result.duration_api_ms,
        "num_turns": result.num_turns,
        "session_id": result.session_id,
        "is_error": failed,
        "finish_reason": "error" if failed else ("tool_calls" if tool_calls else "stop"),
    }
    if result.usage:
        info["usage"] = result.usage
    return info


def _usage_metadata(usage: Any) -> UsageMetadata | None:
    """The CLI's usage as LangChain ``usage_metadata``.

    ``input_tokens`` counts every prompt token, cached or not -- the convention
    ``langchain_anthropic`` uses, so a caller reads a subscription call's usage the way it reads
    an API call's.

    :param usage: ``ResultMessage.usage``
    :ptype usage: Any
    :return: usage metadata, or ``None`` when the CLI reported none
    :rtype: UsageMetadata | None
    """
    if not isinstance(usage, dict):
        return None
    uncached = int(usage.get("input_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
    output = int(usage.get("output_tokens") or 0)
    prompt = uncached + cache_read + cache_creation
    return UsageMetadata(
        input_tokens=prompt,
        output_tokens=output,
        total_tokens=prompt + output,
        input_token_details={"cache_read": cache_read, "cache_creation": cache_creation},
    )


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
