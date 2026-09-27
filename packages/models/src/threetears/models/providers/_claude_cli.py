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

One kind of call gets more turns: a call that asks for a schema and gives the model no tool but the
CLI's own ``StructuredOutput`` (found live, in 0.55.0: about a third of one consumer's structured
calls failed with ``error_max_turns``). The CLI does not constrain the model's output to the schema,
as the Messages API's ``output_config`` does; it checks the model's ``StructuredOutput`` call, answers
a mismatch with what did not match, and lets the model try again in a turn of its own. A model
that fills the call with a placeholder -- ``{"$PARAMETER_VALUE": "<the answer, as a string>"}`` -- or
wraps the answer in one key too many is corrected on the next turn, and one turn left no next turn:
the call ended with the rejected attempt and no answer. Such a call now runs for
:data:`_STRUCTURED_OUTPUT_TURNS` turns, with the CLI's own attempt cap pinned to
:data:`_STRUCTURED_OUTPUT_ATTEMPTS`. It has no tool calls a second turn could run in the caller's
place (:func:`_answers_only_in_schema`), and a call that runs out of attempts still fails, with
``error_max_structured_output_retries``. A call that binds tools as well as a schema stays at one
turn, because its next turn could be spent on the caller's tools.

Token-level streaming: ``ClaudeCodeChatModel._astream`` sets ``include_partial_messages=True`` --
which makes the Agent SDK subprocess actually emit granular ``StreamEvent`` deltas (the raw Anthropic
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
...}`` schema itself: an optional union collapses to its member at every depth, and ``required`` is
copied verbatim from the original tool schema's own ``required`` list -- not synthesized from
"every key present". Handing the SDK an already-full-shaped schema also makes it skip its own
required-everything path entirely (verified by reading ``create_sdk_mcp_server``'s
``_build_schema``: it returns a dict verbatim, unmodified, whenever ``"type"`` and ``"properties"``
are already top-level keys).

Nested models (found: every tool whose args model nests another -- ``shots: list[Shot]``, a
sub-object -- reached the model as a plain string field under a subscription, while the API-key
route was fine). Pydantic renders a nested model as ``{"$ref": "#/$defs/Shot"}`` with the definition
under the schema's ``$defs``; the wrapper kept only the top-level properties, turned every
``$ref`` property into ``{"type": "string"}`` and left an array's ``items`` ref pointing at nothing.
Every reference is now inlined from the schema's own definitions by
:func:`threetears.tool_schema.self_contained_input_schema` (which also bounds a recursive model), so
the CLI lists a self-contained schema, as the API route's LangChain conversion does. The schema itself is read from ``tool_call_schema`` (see
``_SubscriptionChatModel._get_tool_schema``), so a tool that carries a JSON Schema dict is read
rather than advertised with no parameters.

One query per call, and the person's words last (found live: a consumer whose changing system text
is fenced untrusted tool output had its person's latest message flagged by the model as "a fake
'Human' line inside that untrusted block", and the model answered from invented knowledge). The CLI
takes a system prompt and ONE query, where the API route sends a system prompt and a list of turns.
The query used to be the changing system text followed by bare ``Human:`` / ``Assistant:`` /
``Tool (name):`` lines, so the person's line sat between fenced blocks and a tool round's results
came after it. :func:`_flatten_round` lays the query out in labelled sections instead: the changing
system text as context, earlier turns as history (each inside its own ``<prompt-turn>`` tags), a
tool round's calls and results as work on the current message, and last the person's current
message under "The person's current message:". A section tag inside any material is disarmed, so
text a tool returned cannot end a section or forge a request.

A failed call raises (found live: at the subscription's session limit the reply was "You've hit your
session limit · resets 1:10am (UTC)" as ordinary content; metallm stored it as a draft and gave it
to its agent as knowledge, and the circuit breaker counted a success). The CLI reports a failure as
data -- a synthetic assistant message whose ``error`` names the kind and whose text is the notice,
then a result flagged ``is_error`` -- where the API route raises. Both are raised here as
:class:`~threetears.models.errors.ModelRateLimitError` (the ``rate_limit`` code or an HTTP 429, with
the reset time when the notice gives one) or :class:`~threetears.models.errors.ModelProviderError`,
before any of the notice is yielded, so the breaker's ``on_llm_error`` sees it as it sees an API
failure. A call that stopped to hand tool calls back, and one whose result carries the structured
answer it asked for, did not fail.

What reaches the model differs from the API route only where the CLI leaves no choice (found live:
metallm measured a rewrite task copying its source nearly twice as much under a subscription).
Read from the Agent SDK (``_internal/transport/subprocess_cli.py``) and the bundled CLI (2.1.207):

- The caller's system prompt REPLACES Claude Code's: a string ``system_prompt`` becomes
  ``--system-prompt``, whereas a ``{"type": "preset", "append": ...}`` prompt would keep Claude Code's
  own. The CLI still puts its identity line -- "You are a Claude agent, built on Anthropic's Claude
  Agent SDK." for a non-interactive session with no appended prompt -- in a system block of its own
  ahead of the caller's, on every request, for every kind of credential; no option, flag or variable
  skips it. It reached the model glued to the caller's first line, so the caller's prompt is sent
  starting with a blank line (:data:`_AFTER_CLI_IDENTITY`).
- The CLI sends a ``<system-reminder>`` user message ahead of the conversation carrying
  ``# currentDate`` ("Today's date is ..."). The user context that holds it adds the date
  unconditionally, and the reminder is sent whenever that context is not empty; nothing turns it
  off. This, the identity line, and an ``x-anthropic-billing-header`` block (the CLI's version and
  entrypoint) that the CLI adds to the system prompt are the differences that remain. That block
  alone could be switched off, with ``CLAUDE_CODE_ATTRIBUTION_HEADER``; it is left on, because what
  a subscription request without it does has not been measured.
- Left unset, the CLI's query engine turns adaptive thinking on, and sends each model's own launch
  effort (``xhigh`` on one current model). The API route sends neither, so the model does not think
  and the API applies its default effort, ``high``. Both are pinned to that
  (:data:`_NO_THINKING`, :data:`_API_DEFAULT_EFFORT`), and a caller's ``thinking`` / ``effort`` --
  the API route's own parameters, in the same shapes -- are passed through instead. The effort is
  also set in ``CLAUDE_CODE_EFFORT_LEVEL``, which outranks every other source in the CLI's effort
  resolution, launch pins included.

"""

from __future__ import annotations

import contextvars
import dataclasses
import hashlib
import html
import json
import re
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
from pydantic import BaseModel, Field

from threetears.models.claude_cli_isolation import claude_cli_isolation
from threetears.models.claude_cli_pool import (
    TOOL_SERVER_NAME,
    ClaudeCliPoolExhausted,
    ClaudeCliSessionError,
    claude_cli_pool,
)
from threetears.models.errors import ModelProviderError, ModelRateLimitError
from threetears.models.tool_name_translation import NameMangledToolProxy, build_name_translation
from threetears.tool_schema import self_contained_input_schema

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
        # The API route's ChatAnthropic takes these two as well, so a caller sets them once for
        # either credential (see _SubscriptionChatModel.thinking / .effort).
        "thinking",
        "effort",
    }
)


#: Model turns per call. One: the call ends where the model asks for tools, and the caller runs them.
_MODEL_TURNS_PER_CALL = 1

#: Attempts a structured answer gets. The CLI checks each ``StructuredOutput`` call against the schema
#: and answers a mismatch with what did not match, and the model tries again in a turn of its own.
#: Five is the CLI's own default (2.1.207), pinned so a CLI release cannot move it.
_STRUCTURED_OUTPUT_ATTEMPTS = 5

#: The CLI's variable for :data:`_STRUCTURED_OUTPUT_ATTEMPTS`. When they run out, the CLI ends the call
#: with ``error_max_structured_output_retries``.
_STRUCTURED_OUTPUT_ATTEMPTS_ENV = "MAX_STRUCTURED_OUTPUT_RETRIES"

#: Model turns for a call whose only callable tool is ``StructuredOutput``: one per attempt, and one
#: more for a turn that does not call the tool at all, so the attempt cap -- not the turn cap -- is
#: what ends a call that never produces a valid answer.
_STRUCTURED_OUTPUT_TURNS = _STRUCTURED_OUTPUT_ATTEMPTS + 1

#: The CLI's name for a tool on the bound tool server is this prefix plus the tool's wire name.
_BOUND_TOOL_PREFIX = f"mcp__{TOOL_SERVER_NAME}__"

#: What the CLI's tool handler answers. No model turn reads it: the call ends first.
_HANDED_BACK = "This tool call was handed to the caller."

#: The CLI's own tool for a structured answer. Asked for a schema (``--json-schema``), the model
#: answers by calling it, and the CLI puts the answer on ``ResultMessage.structured_output``. It is
#: the CLI's, never the caller's: a caller's tool arrives under :data:`_BOUND_TOOL_PREFIX`.
_STRUCTURED_OUTPUT_TOOL = "StructuredOutput"

#: The provider a failed subscription call names, in errors and in the messages worded from them.
_CLI_PROVIDER = "Claude subscription"

#: When a limit notice says the limit resets: "resets 1:10am (UTC)", "will reset at 5pm".
_RESETS = re.compile(r"\bresets?\s+(?:at\s+)?(?P<when>[^\n]+)", re.IGNORECASE)

#: Thinking when a caller asks for none. The API route sends no ``thinking`` then, and the model does
#: not think; the CLI left unset turns adaptive thinking on. On a model that rejects a disabled
#: thinking parameter the CLI omits it, which is what the API route sends too.
_NO_THINKING: dict[str, Any] = {"type": "disabled"}

#: Effort when a caller asks for none: what the Messages API applies when a request omits it. The
#: CLI left unset sends each model's own launch effort instead (``xhigh`` on one current model).
_API_DEFAULT_EFFORT = "high"

#: The CLI's effort variable. It outranks the ``--effort`` flag, the settings and the per-model launch
#: pins in the bundled CLI's effort resolution, so the effort sent is pinned here as well as in the flag.
_EFFORT_ENV = "CLAUDE_CODE_EFFORT_LEVEL"

#: What starts the caller's system prompt. The CLI sends its own identity line as a system block of
#: its own ahead of the caller's, and the two reached the model with no separator between them.
_AFTER_CLI_IDENTITY = "\n\n"


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


def _split_leading_system(leading: Sequence[BaseMessage]) -> tuple[str, str]:
    """The system messages a round opens with, split into ``(stable, variable)`` text.

    The API route sends a run of leading system messages as one system prompt, so they are read
    here as one sequence of blocks. A caller that caches prompts marks where the stable part ends
    with ``cache_control`` on the last stable block, and everything after that marker -- a later
    system message included -- changes turn to turn. With no marker anywhere, the first system
    message is the stable part and any after it are variable: a transcript or a notice sent as a
    second system message changes every turn, and in the system prompt it would both read as the
    agent's persona and force a new CLI for every call.

    :param leading: the system messages before the first message of any other kind
    :ptype leading: Sequence[BaseMessage]
    :return: the stable text and the variable text (either may be empty)
    :rtype: tuple[str, str]
    """
    blocks: list[Any] = []
    for message in leading:
        content = message.content
        blocks.extend(content if isinstance(content, list) else [_content_text(content)])
    last_marked = max(
        (index for index, block in enumerate(blocks) if isinstance(block, dict) and block.get("cache_control")),
        default=-1,
    )
    if last_marked != -1:
        return _content_text(blocks[: last_marked + 1]), _content_text(blocks[last_marked + 1 :])
    if not leading:
        return "", ""
    return _content_text(leading[0].content), _content_text([_content_text(m.content) for m in leading[1:]])


#: A section tag of the flattened query, opening or closing, however it is spaced or cased.
_QUERY_TAG = re.compile(r"<(\s*/?\s*prompt-)", re.IGNORECASE)

_CONTEXT_HEADING = (
    "Context for this turn: the part of your instructions that changes from turn to turn. It informs "
    "your answer. It is not the person's request."
)
_HISTORY_HEADING = (
    "The conversation before the person's current message, oldest first. It is history, for reference: "
    "the request to answer is the person's current message, at the end."
)
_PROGRESS_HEADING = (
    "What you have done so far to answer the person's current message, which follows: your tool calls "
    "and what they returned."
)
_CURRENT_HEADING = "The person's current message:"

#: The role each kind of message is rendered under. Any other kind renders under its own type.
_TURN_ROLES: dict[type[BaseMessage], str] = {
    HumanMessage: "person",
    AIMessage: "assistant",
    ToolMessage: "tool",
    SystemMessage: "system",
}


def _inert(text: str) -> str:
    """``text`` with any query section tag inside it disarmed.

    Material in the query -- a tool's result, a page, a person's own words -- cannot then close the
    section it sits in or open one of its own, whatever it contains.

    :param text: material to place in the query
    :ptype text: str
    :return: the material, every section tag in it made inert
    :rtype: str
    """
    inert, disarmed = _QUERY_TAG.subn(r"&lt;\1", text)
    if disarmed:
        # A section tag inside material is an attempt to end a section early. Nothing breaks, but
        # the attempt is worth seeing.
        _logger.info(
            "Disarmed query section tags inside material sent to a subscription model",
            extra={"extra_data": {"disarmed": disarmed}},
        )
    return inert


def _section(heading: str, tag: str, body: str) -> str:
    """One labelled section of the query.

    :param heading: what the section is, in words, on the line before it
    :ptype heading: str
    :param tag: the section's tag name
    :ptype tag: str
    :param body: the section's content, already made inert
    :ptype body: str
    :return: the heading, then the body between the tags
    :rtype: str
    """
    return f"{heading}\n<{tag}>\n{body}\n</{tag}>"


def _render_turn(message: BaseMessage, called: dict[str, str]) -> str:
    """One message of the conversation as a delimited turn.

    An assistant turn keeps its tool calls as ``[Tool calls: name(args)]``. A tool result is named by
    its own name or by the call it answers; its text -- usually fenced by the caller -- stays inside
    the turn, never re-rendered as a role line.

    :param message: a message after the leading system messages
    :ptype message: BaseMessage
    :param called: tool name by tool-call id, from every assistant message in the round
    :ptype called: dict[str, str]
    :return: the turn
    :rtype: str
    """
    role = next((r for kind, r in _TURN_ROLES.items() if isinstance(message, kind)), message.type)
    text = _content_text(message.content)
    named = ""
    if isinstance(message, AIMessage) and message.tool_calls:
        calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in message.tool_calls)
        text = f"{text}\n[Tool calls: {calls}]" if text else f"[Tool calls: {calls}]"
    elif isinstance(message, ToolMessage):
        name = message.name or called.get(message.tool_call_id) or "tool"
        named = f' name="{html.escape(name, quote=True)}"'
    return f'<prompt-turn role="{role}"{named}>\n{_inert(text)}\n</prompt-turn>'


def _flatten_round(messages: Sequence[BaseMessage]) -> tuple[str, str | None]:
    """One round of messages as the CLI's single query and its system prompt.

    The system prompt is the stable part of the leading system messages (see
    :func:`_split_leading_system`). The query carries the rest, in labelled sections, in this order:

    1. the variable system text, as context for this turn;
    2. the conversation before the person's current message, as history;
    3. in a round that has already called tools, those calls and their results, as work done on
       the current message;
    4. the person's current message -- the trailing run of their messages, which the API route
       sends as one user turn -- under the heading "The person's current message:".

    So the request is always last and delimited, after every piece of material that could carry an
    instruction, and each earlier turn sits inside its own tags. A round with no message from the
    person has no current message, and everything after the system prompt is history.

    :param messages: the round
    :ptype messages: Sequence[BaseMessage]
    :return: ``(query_text, system_prompt)``
    :rtype: tuple[str, str | None]
    """
    lead = 0
    while lead < len(messages) and isinstance(messages[lead], SystemMessage):
        lead += 1
    stable, variable = _split_leading_system(messages[:lead])
    conversation = list(messages[lead:])
    # A ToolMessage need not carry its tool's name; the call it answers does.
    called = {
        call_id: call["name"]
        for msg in conversation
        if isinstance(msg, AIMessage)
        for call in (msg.tool_calls or [])
        if (call_id := call.get("id"))
    }
    last_person = max((i for i, m in enumerate(conversation) if isinstance(m, HumanMessage)), default=-1)
    first_person = last_person
    while first_person > 0 and isinstance(conversation[first_person - 1], HumanMessage):
        first_person -= 1
    if last_person == -1:
        history, current, progress = conversation, [], []
    else:
        history = conversation[:first_person]
        current = conversation[first_person : last_person + 1]
        progress = conversation[last_person + 1 :]

    sections: list[str] = []
    if variable:
        sections.append(_section(_CONTEXT_HEADING, "prompt-context", _inert(variable)))
    if history:
        turns = "\n\n".join(_render_turn(m, called) for m in history)
        sections.append(_section(_HISTORY_HEADING, "prompt-history", turns))
    if progress:
        turns = "\n\n".join(_render_turn(m, called) for m in progress)
        sections.append(_section(_PROGRESS_HEADING, "prompt-progress", turns))
    if current:
        request = "\n\n".join(_content_text(m.content) for m in current)
        sections.append(_section(_CURRENT_HEADING, "prompt-current-message", _inert(request)))
    return "\n\n".join(sections), (stable or None)


def _answers_only_in_schema(options: Any, bound_tools: Sequence[BaseTool]) -> bool:
    """Whether a call asks for a schema and gives the model no tool but the CLI's ``StructuredOutput``.

    Such a call may run for more than one model turn. Its extra turns can only ever be the CLI's
    schema retries: with no caller tool bound, no built-in tool and no other MCP server, there is no
    tool call a second turn could run in the caller's place. Any other call stays at one turn.

    :param options: the call's ``ClaudeAgentOptions``
    :ptype options: Any
    :param bound_tools: the caller's tools bound to the model
    :ptype bound_tools: Sequence[BaseTool]
    :return: ``True`` when the only tool the model can call is ``StructuredOutput``
    :rtype: bool
    """
    servers = options.mcp_servers
    only_the_bound_tool_server = not servers or (isinstance(servers, dict) and set(servers) <= {TOOL_SERVER_NAME})
    return (
        options.output_format is not None
        and isinstance(options.tools, list)
        and not options.tools
        and not bound_tools
        and only_the_bound_tool_server
    )


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
    from claude_agent_sdk import AssistantMessage, ClaudeSDKClient, ResultMessage, StreamEvent, TextBlock, UserMessage
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

        # The API route's ``thinking`` and ``effort``, in the same shapes (the Agent SDK's
        # ``ThinkingConfig`` is the Messages API's). Declared here for the same reason as
        # :attr:`tools`: the base class drops a field it does not declare. ``None`` means what it
        # means on the API route -- no extended thinking, the API's default effort -- not Claude
        # Code's defaults (see :data:`_NO_THINKING` and :data:`_API_DEFAULT_EFFORT`).
        thinking: dict[str, Any] | None = None
        effort: str | None = None

        def _build_options(self, **overrides: Any) -> Any:
            """Force ``tools`` from :attr:`tools`, and cut the CLI off from the host's Claude config.

            Isolation is applied to the built options rather than passed as overrides: ``env`` is a
            single dict the base class assembles from the token, so an override would replace the
            credential rather than add to it. See :mod:`threetears.models.claude_cli_isolation`
            for what an un-isolated CLI reads.
            """
            overrides.setdefault("tools", self.tools)
            overrides.setdefault("thinking", self.thinking or _NO_THINKING)
            overrides.setdefault("effort", self.effort or _API_DEFAULT_EFFORT)
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
            options.env = {**isolation.env, **(options.env or {}), _EFFORT_ENV: str(options.effort)}
            if isinstance(options.system_prompt, str) and options.system_prompt.strip():
                options.system_prompt = _AFTER_CLI_IDENTITY + options.system_prompt.lstrip("\n")
            if not options.cwd:
                options.cwd = isolation.cwd
            options.extra_args = {**(options.extra_args or {}), **isolation.extra_args}
            # One model turn per call, whatever was asked for: the call ends where the model asks
            # for tools, and the caller runs them (see the module docstring). A call that can only
            # answer in its schema has no tool calls to hand back, and gets the turns the CLI's own
            # schema retries need.
            options.max_turns = _MODEL_TURNS_PER_CALL
            if _answers_only_in_schema(options, self._bound_tools):
                options.max_turns = _STRUCTURED_OUTPUT_TURNS
                options.env = {**options.env, _STRUCTURED_OUTPUT_ATTEMPTS_ENV: str(_STRUCTURED_OUTPUT_ATTEMPTS)}
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

        def _get_tool_schema(self, tool: BaseTool) -> dict[str, Any]:
            """The JSON Schema of the arguments the model fills in for ``tool``.

            Read from ``tool_call_schema``, which is what the API route shows the model: it leaves
            out arguments the caller injects (a tool-call id, a runnable config) and it is the
            schema itself when a tool carries one as a JSON Schema dict. The base class called
            ``args_schema.model_json_schema()``, which a dict does not have, and answered every
            failure with an empty schema -- a tool advertised with no parameters, and nothing said.
            A schema that cannot be rendered now fails the bind, as it does on the API route.

            :param tool: a bound tool
            :ptype tool: BaseTool
            :return: the tool's argument schema
            :rtype: dict[str, Any]
            """
            call_schema = tool.tool_call_schema
            rendered: dict[str, Any]
            if isinstance(call_schema, dict):
                rendered = dict(call_schema)
            elif issubclass(call_schema, BaseModel):
                rendered = call_schema.model_json_schema()
            else:
                rendered = call_schema.schema()  # a pydantic v1 model
            return rendered

        def _wrap_langchain_tool(self, tool: BaseTool, schema: dict[str, Any]) -> Callable[..., Any]:
            """The SDK tool the CLI lists for ``tool``: its schema made self-contained, its handler inert.

            :param tool: a bound tool
            :ptype tool: BaseTool
            :param schema: the tool's argument schema, from :meth:`_get_tool_schema`
            :ptype schema: dict[str, Any]
            :return: the SDK tool
            :rtype: Callable[..., Any]
            :raises ValueError: when the schema holds a reference that cannot be inlined
            """
            input_schema = self_contained_input_schema(schema, tool_name=tool.name)

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

            Returns the STABLE part of the system prompt as the system prompt and carries
            everything else in the query. A running CLI's system prompt cannot change, so this is
            what lets one CLI serve turn after turn while retrieved memory, tool results and notices
            change underneath it. The query's layout -- context, history, work on the current
            message, then the person's current message last -- is :func:`_flatten_round`'s.

            :param messages: the conversation
            :ptype messages: list[BaseMessage]
            :return: ``(query_text, system_prompt)``
            :rtype: tuple[str, str | None]
            """
            return _flatten_round(messages)

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
            # tool-result list, callbacks and runnable config set. Tool calls on a pooled CLI run in
            # a copy of it; without that they ran in whatever context started the CLI.
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
            attempts = _StructuredAttempts()
            async with self._cli_client(options, pooled=not session_id) as client:
                await client.query(prompt)
                async for msg in client.receive_response():
                    if isinstance(msg, AssistantMessage):
                        # A failure the CLI met is sent as a message whose text is the notice.
                        # It is raised, never answered with, and before any token callback sees it.
                        if (failure := _assistant_failure(msg)) is not None:
                            raise failure
                        text = "\n".join(b.text for b in msg.content if isinstance(b, TextBlock))
                        if text:
                            all_text.append(text)
                            if run_manager:
                                await run_manager.on_llm_new_token(text)
                        tool_calls.extend(self._caller_tool_calls(msg.content))
                        attempts.saw(msg.content)
                    elif isinstance(msg, UserMessage):
                        attempts.answered(msg)
                    elif isinstance(msg, ResultMessage):
                        self._last_result = msg
                        unwrapped = _unwrapped_answer(msg, options.output_format, attempts)
                        if (
                            unwrapped is None
                            and (failure := _result_failure(msg, tool_calls, "\n".join(all_text), attempts)) is not None
                        ):
                            raise failure
                        generation_info = _generation_info(msg, tool_calls, unwrapped=unwrapped is not None)
                        answer = msg.structured_output if msg.structured_output is not None else unwrapped
                        if options.output_format is not None and answer is not None:
                            # The answer is the structured one. The model's prose before it is
                            # not: asked for a shape, Haiku still wrote a paragraph first.
                            return json.dumps(answer), tool_calls, generation_info
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
            # Every message's text, streamed or held, for a failed result that says nothing itself.
            produced: list[str] = []
            attempts = _StructuredAttempts()

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
                        # A failure the CLI met is sent as a message whose text is the notice; it
                        # is raised before a character of it is streamed to the person.
                        if (failure := _assistant_failure(msg)) is not None:
                            raise failure
                        text = "\n".join(b.text for b in msg.content if isinstance(b, TextBlock))
                        produced.append(text)
                        attempts.saw(msg.content)
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

                    elif isinstance(msg, UserMessage):
                        attempts.answered(msg)

                    elif isinstance(msg, ResultMessage):
                        self._last_result = msg
                        unwrapped = _unwrapped_answer(msg, options.output_format, attempts)
                        if (
                            unwrapped is None
                            and (failure := _result_failure(msg, tool_calls, "\n".join(produced), attempts)) is not None
                        ):
                            raise failure
                        generation_info = _generation_info(msg, tool_calls, unwrapped=unwrapped is not None)
                        usage = _usage_metadata(msg.usage)
                        answer = msg.structured_output if msg.structured_output is not None else unwrapped
                        content = ""
                        if structured:
                            content = json.dumps(answer) if answer is not None else "\n".join(held)
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


def _ended_for_tools(result: Any, tool_calls: list[dict[str, Any]]) -> bool:
    """Whether a call ended on the CLI's turn limit because the model asked for tools.

    That is the designed end of such a call (see the module docstring), not a failure, although the
    CLI flags it as an error.

    :param result: the CLI's ``ResultMessage``
    :ptype result: Any
    :param tool_calls: the tool calls the call handed back
    :ptype tool_calls: list[dict[str, Any]]
    :return: ``True`` when the call stopped to hand its tool calls back
    :rtype: bool
    """
    return bool(tool_calls) and result.subtype == "error_max_turns"


class _StructuredAttempts:
    """The ``StructuredOutput`` attempts one call made, and the last one the CLI rejected.

    The CLI answers each attempt with a tool result: an error naming what did not match the schema,
    or an acceptance. A call that ends without an answer raises, and the error carries the last
    rejected attempt and its reason, so whoever reads it can see which field the model kept missing
    without replaying the call.
    """

    def __init__(self) -> None:
        self._inputs: dict[str, dict[str, Any]] = {}
        self.rejected_output: dict[str, Any] | None = None
        self.rejection: str | None = None
        #: Every attempt the CLI rejected, oldest first.
        self.rejected: list[dict[str, Any]] = []

    def saw(self, blocks: list[Any]) -> None:
        """Note every ``StructuredOutput`` call among an assistant message's blocks.

        :param blocks: the assistant message's content blocks
        :ptype blocks: list[Any]
        """
        from claude_agent_sdk import ToolUseBlock  # noqa: PLC0415

        for block in blocks:
            if isinstance(block, ToolUseBlock) and block.name == _STRUCTURED_OUTPUT_TOOL:
                self._inputs[block.id] = dict(block.input or {})

    def answered(self, message: Any) -> None:
        """Note the CLI's rejection of a ``StructuredOutput`` call, when a user message carries one.

        :param message: a ``UserMessage`` the CLI sent
        :ptype message: Any
        """
        from claude_agent_sdk import ToolResultBlock  # noqa: PLC0415

        if not isinstance(message.content, list):
            return
        for block in message.content:
            if isinstance(block, ToolResultBlock) and block.is_error and block.tool_use_id in self._inputs:
                self.rejected_output = self._inputs[block.tool_use_id]
                self.rejection = _content_text(block.content) or None
                self.rejected.append(self.rejected_output)


#: The tool-call template placeholders the model leaks as a parameter name, wrapping a whole
#: structured answer as a JSON string under one of them. Each was seen live (0.56.0,
#: ``claude-sonnet-5``) repeated on every attempt of a call until the CLI's attempt cap ended it.
_TEMPLATE_PLACEHOLDER_KEYS = frozenset({"$PARAMETER_VALUE", "$PARAMETER_NAME", "$FUNCTION_NAME"})


def _placeholder_answer(attempt: dict[str, Any], schema: dict[str, Any]) -> Any | None:
    """The answer inside a placeholder-wrapped attempt, when it is one and it matches the schema.

    Only exactly one key from :data:`_TEMPLATE_PLACEHOLDER_KEYS` holding a string qualifies, and only
    when that string parses as JSON and the parsed value validates against the schema the call asked
    for. Anything else is ``None``.

    :param attempt: a ``StructuredOutput`` input the CLI rejected
    :ptype attempt: dict[str, Any]
    :param schema: the JSON schema the call asked for
    :ptype schema: dict[str, Any]
    :return: the parsed answer, or ``None``
    :rtype: Any | None
    """
    from jsonschema import Draft202012Validator  # noqa: PLC0415

    if len(attempt) != 1:
        return None
    [(key, wrapped)] = attempt.items()
    if key not in _TEMPLATE_PLACEHOLDER_KEYS or not isinstance(wrapped, str):
        return None
    try:
        answer = json.loads(wrapped)
    except json.JSONDecodeError:
        # NOSILENT: a placeholder holding text that is not JSON is not an answer; the caller keeps
        # the rejection it already has, and the CLI's own rejection message is logged with it.
        return None
    return answer if Draft202012Validator(schema).is_valid(answer) else None


def _unwrapped_answer(result: Any, output_format: Any, attempts: _StructuredAttempts) -> Any | None:
    """A failed structured call's answer, recovered from an attempt the model wrapped in a placeholder.

    Found live (0.56.0, ``claude-sonnet-5``): about one structured call in forty spent all five
    attempts sending the whole answer, correct, as a JSON string under a template placeholder --
    ``{"$PARAMETER_VALUE": "<the answer>"}``, or ``$PARAMETER_NAME`` / ``$FUNCTION_NAME`` -- which the
    CLI rejects every time. When the call failed with no structured answer, the most recent rejected
    attempt of exactly that shape whose
    JSON validates against the call's own schema is the answer. It is logged, once, at WARNING and
    marked on the result's metadata (``structured_output_unwrapped``).

    :param result: the CLI's ``ResultMessage``
    :ptype result: Any
    :param output_format: the call's ``output_format``, ``None`` for a call with no schema
    :ptype output_format: Any
    :param attempts: the call's ``StructuredOutput`` attempts
    :ptype attempts: _StructuredAttempts
    :return: the recovered answer, or ``None``
    :rtype: Any | None
    """
    schema = output_format.get("schema") if isinstance(output_format, dict) else None
    if not isinstance(schema, dict) or result.structured_output is not None or not result.is_error:
        return None
    answer = next(
        (
            found
            for attempt in reversed(attempts.rejected)
            if (found := _placeholder_answer(attempt, schema)) is not None
        ),
        None,
    )
    if answer is not None:
        _logger.warning(
            "A structured answer the model wrapped in a placeholder parameter was unwrapped",
            extra={
                "extra_data": {
                    "schema": schema.get("title")
                    or hashlib.sha256(json.dumps(schema, sort_keys=True).encode("utf-8")).hexdigest()[:12],
                    "reason": result.subtype,
                    "rejected_attempts": len(attempts.rejected),
                }
            },
        )
    return answer


def _cli_failure(
    detail: str, *, reason: str | None, status: int | None, attempts: _StructuredAttempts | None = None
) -> ModelProviderError:
    """The typed error for a failure the CLI reported.

    A limit -- the CLI's own ``rate_limit`` code, which it sends with a subscription's session-limit
    notice, or an HTTP 429 on the result -- is a :class:`ModelRateLimitError` carrying when it resets,
    when the notice says. Anything else is a :class:`ModelProviderError`. Either carries the last
    structured answer the CLI rejected, and why, when there was one. The log line names the rejection
    and the rejected answer's top-level keys, not its values: they are the model's words about the
    caller's material.

    :param detail: what the CLI said
    :ptype detail: str
    :param reason: the CLI's code for the failure, when it gave one
    :ptype reason: str | None
    :param status: HTTP status of the failing API call, when the CLI reported one
    :ptype status: int | None
    :param attempts: the call's ``StructuredOutput`` attempts, when it asked for a schema
    :ptype attempts: _StructuredAttempts | None
    :return: the error to raise
    :rtype: ModelProviderError
    """
    said = detail.strip() or "the Claude CLI reported an error and gave no reason"
    rejected_output = attempts.rejected_output if attempts is not None else None
    rejection = attempts.rejection if attempts is not None else None
    failure: ModelProviderError
    if reason == "rate_limit" or status == 429:
        resets = _RESETS.search(said)
        failure = ModelRateLimitError(
            said,
            provider=_CLI_PROVIDER,
            reason=reason,
            status=status,
            resets=resets.group("when").strip().rstrip(".") if resets else None,
            rejected_output=rejected_output,
            rejection=rejection,
        )
    else:
        failure = ModelProviderError(
            said,
            provider=_CLI_PROVIDER,
            reason=reason,
            status=status,
            rejected_output=rejected_output,
            rejection=rejection,
        )
    _logger.warning(
        "A subscription model call failed",
        extra={
            "extra_data": {
                "error_type": type(failure).__name__,
                "reason": reason,
                "status": status,
                "detail": said,
                "rejection": rejection,
                "rejected_keys": sorted(rejected_output) if rejected_output is not None else None,
            }
        },
    )
    return failure


def _assistant_failure(message: Any) -> ModelProviderError | None:
    """The error an assistant message reports, when the CLI flagged it as one.

    The CLI sends a failure it met calling the API -- a subscription's session limit among them --
    as a synthetic assistant message whose text is the notice and whose ``error`` names the kind.

    :param message: the CLI's ``AssistantMessage``
    :ptype message: Any
    :return: the error to raise, or ``None`` for an ordinary message
    :rtype: ModelProviderError | None
    """
    if message.error is None:
        return None
    text = "\n".join(block.text for block in message.content if hasattr(block, "text"))
    return _cli_failure(text, reason=str(message.error), status=None)


def _result_failure(
    result: Any, tool_calls: list[dict[str, Any]], text: str, attempts: _StructuredAttempts
) -> ModelProviderError | None:
    """The error a call's result reports, when the call failed.

    Not a failure: a call that stopped to hand tool calls back (:func:`_ended_for_tools`), and a call
    whose result carries the structured answer it was asked for -- raising would throw that answer
    away.

    :param result: the CLI's ``ResultMessage``
    :ptype result: Any
    :param tool_calls: the tool calls the call handed back
    :ptype tool_calls: list[dict[str, Any]]
    :param text: the text the call produced, for a result that says nothing itself
    :ptype text: str
    :param attempts: the call's ``StructuredOutput`` attempts
    :ptype attempts: _StructuredAttempts
    :return: the error to raise, or ``None`` when the call did not fail
    :rtype: ModelProviderError | None
    """
    if not result.is_error or _ended_for_tools(result, tool_calls) or result.structured_output is not None:
        return None
    said = result.result or "; ".join(result.errors or []) or text
    reason = None if result.subtype == "success" else result.subtype
    return _cli_failure(said, reason=reason, status=result.api_error_status, attempts=attempts)


def _generation_info(result: Any, tool_calls: list[dict[str, Any]], *, unwrapped: bool = False) -> dict[str, Any]:
    """What a call's ``ResultMessage`` says, as generation info.

    A call that asked for tools ends on the CLI's turn limit (``error_max_turns``). That is the
    designed end of such a call, not a failure.

    :param result: the CLI's ``ResultMessage``
    :ptype result: Any
    :param tool_calls: the tool calls the call handed back
    :ptype tool_calls: list[dict[str, Any]]
    :param unwrapped: whether the answer was recovered from a placeholder (:func:`_unwrapped_answer`)
    :ptype unwrapped: bool
    :return: generation info
    :rtype: dict[str, Any]
    """
    failed = bool(result.is_error) and not _ended_for_tools(result, tool_calls) and not unwrapped
    info: dict[str, Any] = {
        "total_cost_usd": result.total_cost_usd,
        "duration_ms": result.duration_ms,
        "duration_api_ms": result.duration_api_ms,
        "num_turns": result.num_turns,
        "session_id": result.session_id,
        "is_error": failed,
        "finish_reason": "error" if failed else ("tool_calls" if tool_calls else "stop"),
    }
    if unwrapped:
        info["structured_output_unwrapped"] = 1
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

    Claude Code's own built-in tool belt (Bash, Read, Write, Edit, WebFetch, WebSearch, …) is OFF
    unless asked for: ``tools`` defaults to ``[]`` (see ``_SubscriptionChatModel.tools``), a list
    enables the built-ins it names, and ``tools=None`` enables the whole preset. Built-ins are
    independent of any LangChain tools bound via ``bind_tools()``. The choice also sets how many
    model turns a structured call gets: with no built-in and no bound tool it may take the CLI's
    schema retries (:data:`_STRUCTURED_OUTPUT_TURNS`); with any, one turn, so a rejected
    ``StructuredOutput`` attempt fails the call (see :func:`_answers_only_in_schema`).

    :param model_name: the Anthropic model id
    :ptype model_name: str
    :param token: the subscription OAuth token
    :ptype token: str
    :param extra_kwargs: provider kwargs; only :data:`_FORWARDED_KWARGS` are kept
    :ptype extra_kwargs: Any
    :return: the subscription-backed chat model
    :rtype: BaseChatModel
    """
    opts = {k: v for k, v in extra_kwargs.items() if k in _FORWARDED_KWARGS}
    model: BaseChatModel = _subscription_model_cls()(model=model_name, oauth_token=token, **opts)
    return model
