"""prompt-caching helpers for the shared :func:`agent_node`.

provides the pure functions the caching hook leans on:
:class:`ChatModelCapabilities`, :func:`detect_capabilities`,
:func:`annotate_system_prompt`, :func:`extract_cache_usage`, and
:func:`should_bind_tools_fresh`. the module is
langchain-adapter-agnostic -- capability detection inspects the
class name and the ``model`` / ``model_name`` attribute rather
than importing provider-specific classes, so a workspace that
ships only one provider's langchain integration does not pay an
import cost for the others.

the hook that wires these helpers into the node lives in
:mod:`threetears.langgraph.hooks` (class
:class:`threetears.langgraph.hooks.PromptCachingHook`).

anthropic prompt caching contract
=================================

anthropic reads ``cache_control={"type": "ephemeral"}`` only when it
appears on a structured content block, not on ``additional_kwargs``.
the canonical shape emitted by :func:`annotate_system_prompt` for a
caching-capable model is::

    SystemMessage(content=[
        {"type": "text", "text": <prompt>, "cache_control": {"type": "ephemeral"}},
    ])

for non-caching models the function returns the bare-string form
``SystemMessage(content=<prompt>)`` -- cross-provider degradation is
silent and zero-warning.

tool-binding stability
======================

tool-schema bytes participate in the cached prefix; any drift in
tool ordering or schema JSON invalidates the prefix. the
:func:`compute_tool_key` helper hashes the sorted
``(tool.name, args_schema.model_json_schema())`` pairs into a
stable 16-hex-char digest, and :func:`should_bind_tools_fresh`
compares the previous digest to the current one to decide whether
:meth:`bind_tools` must be called again or the cached bound-model
reference can be reused.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import SystemMessage

__all__ = [
    "ANTHROPIC_MIN_CACHEABLE_TOKENS",
    "ANTHROPIC_EPHEMERAL_TTL_SECONDS",
    "ChatModelCapabilities",
    "annotate_system_prompt",
    "compute_tool_key",
    "detect_capabilities",
    "extract_cache_usage",
    "should_bind_tools_fresh",
]


ANTHROPIC_MIN_CACHEABLE_TOKENS: int = 1024
"""minimum cacheable prefix length for current Sonnet / Opus / Haiku 4 families.

anthropic refuses to cache prefixes shorter than this; shorter
prompts still route through the model, just without cache_control
effect. exposed as a constant so callers (docs, tests) reference
the same number the helper uses.
"""

ANTHROPIC_EPHEMERAL_TTL_SECONDS: int = 5 * 60
"""ephemeral cache entry lifetime (5 minutes, extended per turn).

the anthropic platform refreshes the TTL each time the cached
prefix is read, so conversations with turn gaps under 5 minutes
see the prefix stay hot across the whole session.
"""


_ANTHROPIC_CACHE_MODEL_PREFIXES: tuple[str, ...] = (
    "claude-3",
    "claude-sonnet",
    "claude-opus",
    "claude-haiku",
    "claude-4",
)

_ANTHROPIC_CHAT_MODEL_CLASSES: frozenset[str] = frozenset({
    "ChatAnthropic",
    "AnthropicLLM",
})

_OPENAI_CHAT_MODEL_CLASSES: frozenset[str] = frozenset({
    "ChatOpenAI",
    "AzureChatOpenAI",
})


@dataclass(frozen=True)
class ChatModelCapabilities:
    """per-model capability record consulted by the caching path.

    production use treats instances as immutable; callers assemble
    fresh records rather than mutating. the frozen dataclass makes
    accidental mutation a :class:`TypeError` at assignment time.

    :param supports_anthropic_cache_control: model accepts
        ``cache_control={"type": "ephemeral"}`` on structured system
        content and honors it on the provider request
    :ptype supports_anthropic_cache_control: bool
    :param supports_openai_auto_cache: model participates in openai's
        automatic prompt-caching (no opt-in; response surfaces
        ``prompt_tokens_details.cached_tokens``)
    :ptype supports_openai_auto_cache: bool
    :param min_cacheable_tokens: shortest prefix length at which the
        provider actually caches; prefixes shorter than this pay the
        full price even when ``cache_control`` is attached
    :ptype min_cacheable_tokens: int
    :param cache_ttl_seconds: provider-side ephemeral cache lifetime
        in seconds; ``0`` means the provider does not expose a TTL
    :ptype cache_ttl_seconds: int
    """

    supports_anthropic_cache_control: bool
    supports_openai_auto_cache: bool
    min_cacheable_tokens: int
    cache_ttl_seconds: int


_NO_CACHE_CAPABILITIES: ChatModelCapabilities = ChatModelCapabilities(
    supports_anthropic_cache_control=False,
    supports_openai_auto_cache=False,
    min_cacheable_tokens=0,
    cache_ttl_seconds=0,
)


def _extract_model_name(chat_model: Any) -> str:
    """read the model identifier off a langchain chat-model instance.

    langchain adapters disagree on the attribute name:
    :class:`ChatAnthropic` uses ``model``, :class:`ChatOpenAI` uses
    ``model_name``. function checks both, returning the empty
    string when neither is set.

    :param chat_model: langchain chat-model instance
    :ptype chat_model: Any
    :return: model identifier (e.g. ``"claude-sonnet-4-5"``,
        ``"gpt-4o"``) or empty string
    :rtype: str
    """
    candidates = ("model", "model_name")
    name = ""
    for attr in candidates:
        value = getattr(chat_model, attr, None)
        if isinstance(value, str) and value:
            name = value
            break
    return name


def _model_matches_anthropic_cache_prefix(model_name: str) -> bool:
    """return True when ``model_name`` identifies a cache-capable anthropic model.

    :param model_name: provider model identifier
    :ptype model_name: str
    :return: ``True`` if the name starts with a known cache-capable
        anthropic family prefix
    :rtype: bool
    """
    lowered = model_name.lower()
    matched = False
    for prefix in _ANTHROPIC_CACHE_MODEL_PREFIXES:
        if lowered.startswith(prefix):
            matched = True
            break
    return matched


def detect_capabilities(chat_model: Any) -> ChatModelCapabilities:
    """inspect a langchain chat-model instance and return its caching capabilities.

    detection strategy:

    1. read the adapter class name (``type(chat_model).__name__``).
    2. read the model identifier via :func:`_extract_model_name`.
    3. map ``(class_name, model_name)`` onto a capability record.

    detection is conservative: when the class name is unknown, the
    function returns the all-False record so the node path emits a
    bare system message and no cache annotations. this is the
    deliberate cross-provider-degradation behavior the shard pins.

    :param chat_model: langchain chat-model instance to inspect
    :ptype chat_model: Any
    :return: capability record for this model
    :rtype: ChatModelCapabilities
    """
    class_name = type(chat_model).__name__
    model_name = _extract_model_name(chat_model)
    caps: ChatModelCapabilities = _NO_CACHE_CAPABILITIES
    if class_name in _ANTHROPIC_CHAT_MODEL_CLASSES:
        if _model_matches_anthropic_cache_prefix(model_name):
            caps = ChatModelCapabilities(
                supports_anthropic_cache_control=True,
                supports_openai_auto_cache=False,
                min_cacheable_tokens=ANTHROPIC_MIN_CACHEABLE_TOKENS,
                cache_ttl_seconds=ANTHROPIC_EPHEMERAL_TTL_SECONDS,
            )
    elif class_name in _OPENAI_CHAT_MODEL_CLASSES:
        caps = ChatModelCapabilities(
            supports_anthropic_cache_control=False,
            supports_openai_auto_cache=True,
            min_cacheable_tokens=0,
            cache_ttl_seconds=0,
        )
    return caps


def annotate_system_prompt(
    prompt: str,
    caps: ChatModelCapabilities,
) -> SystemMessage:
    """build a :class:`SystemMessage` carrying cache annotations when supported.

    when ``caps.supports_anthropic_cache_control`` is True the
    message's ``content`` is a one-element list of structured text
    blocks with ``cache_control={"type": "ephemeral"}`` attached to
    the final block. this is the shape
    :class:`langchain_anthropic.ChatAnthropic` recognizes and
    forwards to the provider request.

    when caching is not supported the function returns the bare
    ``SystemMessage(content=prompt)`` shape so non-anthropic
    adapters (openai, bedrock, local models, test doubles) see the
    exact message type they already expect.

    :param prompt: system prompt text
    :ptype prompt: str
    :param caps: capability record produced by
        :func:`detect_capabilities`
    :ptype caps: ChatModelCapabilities
    :return: :class:`SystemMessage` with provider-appropriate
        content shape
    :rtype: SystemMessage
    """
    if caps.supports_anthropic_cache_control:
        block: dict[Any, Any] = {
            "type": "text",
            "text": prompt,
            "cache_control": {"type": "ephemeral"},
        }
        content_blocks: list[str | dict[Any, Any]] = [block]
        result = SystemMessage(content=content_blocks)
    else:
        result = SystemMessage(content=prompt)
    return result


def extract_cache_usage(response: Any) -> dict[str, int]:
    """pull normalized cache-usage counters off a model response.

    reads the two families of telemetry shape:

    - anthropic on ``usage_metadata``:
      ``input_token_details.cache_read`` /
      ``input_token_details.cache_creation`` (langchain-anthropic
      also surfaces ``cache_read_input_tokens`` /
      ``cache_creation_input_tokens`` on ``response_metadata`` for
      some model versions; both are consulted).
    - openai on ``usage_metadata``:
      ``input_token_details.cache_read``.

    missing fields default to zero. function never raises -- a
    response with no ``usage_metadata`` yields the all-zero dict.

    :param response: langchain response object (typically an
        :class:`AIMessage`)
    :ptype response: Any
    :return: ``{"cache_read_input_tokens": int,
        "cache_creation_input_tokens": int, "cached_tokens": int}``
    :rtype: dict[str, int]
    """
    result: dict[str, int] = {
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cached_tokens": 0,
    }
    usage_metadata = getattr(response, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        details = usage_metadata.get("input_token_details")
        if isinstance(details, dict):
            cache_read = details.get("cache_read", 0)
            cache_creation = details.get("cache_creation", 0)
            if isinstance(cache_read, int):
                result["cache_read_input_tokens"] = cache_read
                result["cached_tokens"] = cache_read
            if isinstance(cache_creation, int):
                result["cache_creation_input_tokens"] = cache_creation
    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        rm_usage = response_metadata.get("usage", {})
        if isinstance(rm_usage, dict):
            rm_read = rm_usage.get("cache_read_input_tokens", 0)
            rm_create = rm_usage.get("cache_creation_input_tokens", 0)
            if isinstance(rm_read, int) and rm_read > result["cache_read_input_tokens"]:
                result["cache_read_input_tokens"] = rm_read
                result["cached_tokens"] = max(result["cached_tokens"], rm_read)
            if isinstance(rm_create, int) and rm_create > result["cache_creation_input_tokens"]:
                result["cache_creation_input_tokens"] = rm_create
    return result


def compute_tool_key(tools: list[Any]) -> str:
    """compute a stable digest over a tool list's names and schemas.

    sorts tools by ``tool.name`` then serializes
    ``(name, args_schema.model_json_schema())`` pairs into a sorted
    JSON payload, and hashes the bytes with sha256. returns the
    first 16 hex chars -- collision risk at 16 hex chars (64 bits)
    is negligible for tool-set fingerprints and the short digest
    keeps logs readable.

    tools that do not expose ``args_schema.model_json_schema()``
    (plain functions without a pydantic schema, fake tools in tests)
    contribute only their ``name`` to the digest; this is still
    sufficient to detect tool-set membership changes.

    returns the digest of the empty list when ``tools`` is empty so
    callers can compare to a cached key without guarding for the
    ``None`` case.

    :param tools: list of tool instances
    :ptype tools: list[Any]
    :return: 16-hex-char stable digest of the tool list
    :rtype: str
    """
    payload: list[list[Any]] = []
    for tool in sorted(tools, key=lambda t: getattr(t, "name", "")):
        name = getattr(tool, "name", "")
        schema: Any
        args_schema = getattr(tool, "args_schema", None)
        if args_schema is not None and hasattr(args_schema, "model_json_schema"):
            try:
                schema = args_schema.model_json_schema()
            except Exception:  # noqa: BLE001 - schema generation may fail on exotic tools
                schema = None
        else:
            schema = None
        payload.append([name, schema])
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return digest


def should_bind_tools_fresh(
    prev_tool_key: str | None,
    current_tool_key: str,
) -> bool:
    """decide whether :meth:`bind_tools` must be re-called.

    returns True when the tool set has changed (``prev_tool_key`` is
    None or differs from ``current_tool_key``). returns False when
    the keys match -- the cached bound-model reference from the
    previous invocation is still valid.

    :param prev_tool_key: digest from the prior invocation, or None
        when no binding has happened yet
    :ptype prev_tool_key: str | None
    :param current_tool_key: digest for the current tool set
    :ptype current_tool_key: str
    :return: True when rebinding is required
    :rtype: bool
    """
    result = prev_tool_key is None or prev_tool_key != current_tool_key
    return result
