"""unit tests for :mod:`threetears.langgraph.caching`.

exercises the four pure helpers (capability detection, system prompt
annotation, cache-usage extraction, tool-key + rebind decision) in
isolation against stub chat-model / response shapes. no graph is
compiled, no provider is contacted.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from threetears.langgraph.caching import (
    ANTHROPIC_EPHEMERAL_TTL_SECONDS,
    ANTHROPIC_MIN_CACHEABLE_TOKENS,
    ChatModelCapabilities,
    annotate_system_prompt,
    compute_tool_key,
    detect_capabilities,
    extract_cache_usage,
    should_bind_tools_fresh,
)


class _StubChatAnthropic:
    """lookalike for :class:`langchain_anthropic.ChatAnthropic`.

    class name is matched on by :func:`detect_capabilities` so tests
    avoid importing the real adapter.
    """

    def __init__(self, model: str) -> None:
        """capture the model identifier like the real adapter.

        :param model: anthropic model name (e.g. ``"claude-sonnet-4-5"``)
        :ptype model: str
        """
        self.model = model


# the real langchain_anthropic class must match by name for
# :func:`detect_capabilities` to recognize a stub; we rename the
# stub class at definition time so the lookup succeeds.
_StubChatAnthropic.__name__ = "ChatAnthropic"


class _StubChatOpenAI:
    """lookalike for :class:`langchain_openai.ChatOpenAI`."""

    def __init__(self, model_name: str) -> None:
        """capture the model identifier like the real adapter.

        :param model_name: openai model identifier (e.g. ``"gpt-4o"``)
        :ptype model_name: str
        """
        self.model_name = model_name


_StubChatOpenAI.__name__ = "ChatOpenAI"


class _StubUnknownChatModel:
    """lookalike for an unrecognized adapter (bedrock, local, fake)."""

    def __init__(self, model: str = "") -> None:
        """capture a model identifier.

        :param model: optional model identifier; default empty
        :ptype model: str
        """
        self.model = model


class _StubChatOpenRouter:
    """lookalike for :class:`langchain_openrouter.ChatOpenRouter`.

    ``ChatOpenRouter`` uses ``model`` for the OpenRouter slug
    (``upstream/model-id`` form). Renamed to match the class name
    :func:`detect_capabilities` recognizes.
    """

    def __init__(self, model: str) -> None:
        """capture the openrouter slug.

        :param model: openrouter slug (e.g. ``"anthropic/claude-sonnet-4.6"``)
        :ptype model: str
        """
        self.model = model


_StubChatOpenRouter.__name__ = "ChatOpenRouter"


class _StubNameTranslatingChatOpenRouter:
    """lookalike for the 3tears ``_NameTranslatingChatOpenRouter`` subclass.

    The factory at :func:`threetears.models.providers.openrouter.create_openrouter_chat`
    returns this subclass to add tool-name dot/underscore translation.
    Production traffic flows through the subclass, not the bare
    ``ChatOpenRouter``, so capability detection must recognize both
    class names.
    """

    def __init__(self, model: str) -> None:
        """capture the openrouter slug.

        :param model: openrouter slug
        :ptype model: str
        """
        self.model = model


_StubNameTranslatingChatOpenRouter.__name__ = "_NameTranslatingChatOpenRouter"


class _StubPydanticArgsSchema:
    """minimal ``args_schema`` exposing ``model_json_schema``.

    mimics a pydantic ``BaseModel`` class well enough for
    :func:`compute_tool_key` to sample the schema.
    """

    def __init__(self, schema: dict[str, Any]) -> None:
        """stash the target schema dict.

        :param schema: json-schema dict returned from
            :meth:`model_json_schema`
        :ptype schema: dict[str, Any]
        """
        self._schema = schema

    def model_json_schema(self) -> dict[str, Any]:
        """return the stashed json schema dict.

        :return: schema dict
        :rtype: dict[str, Any]
        """
        return self._schema


class _StubTool:
    """minimal tool double with ``name`` and optional ``args_schema``."""

    def __init__(
        self,
        name: str,
        args_schema: _StubPydanticArgsSchema | None = None,
    ) -> None:
        """capture the name and optional args_schema like a real tool.

        :param name: tool name
        :ptype name: str
        :param args_schema: optional schema stub
        :ptype args_schema: _StubPydanticArgsSchema | None
        """
        self.name = name
        self.args_schema = args_schema


class TestDetectCapabilities:
    """:func:`detect_capabilities` maps ``(class, model)`` to capability record."""

    def test_chat_anthropic_sonnet_supports_cache_control(self) -> None:
        """Sonnet 4.5 is flagged as cache-capable with anthropic ttl.

        :raises AssertionError: when the record is not the expected shape
        """
        caps = detect_capabilities(_StubChatAnthropic("claude-sonnet-4-5"))
        assert caps.supports_anthropic_cache_control is True
        assert caps.supports_openai_auto_cache is False
        assert caps.min_cacheable_tokens == ANTHROPIC_MIN_CACHEABLE_TOKENS
        assert caps.cache_ttl_seconds == ANTHROPIC_EPHEMERAL_TTL_SECONDS

    def test_chat_anthropic_opus_supports_cache_control(self) -> None:
        """Opus family also returns cache-capable record.

        :raises AssertionError: when opus is not recognized
        """
        caps = detect_capabilities(_StubChatAnthropic("claude-opus-4-1"))
        assert caps.supports_anthropic_cache_control is True

    def test_chat_anthropic_haiku_supports_cache_control(self) -> None:
        """Haiku family also returns cache-capable record.

        :raises AssertionError: when haiku is not recognized
        """
        caps = detect_capabilities(_StubChatAnthropic("claude-haiku-4-5"))
        assert caps.supports_anthropic_cache_control is True

    def test_chat_anthropic_legacy_claude_3_supports_cache_control(self) -> None:
        """claude-3 generation models continue to be recognized.

        :raises AssertionError: when legacy models regress
        """
        caps = detect_capabilities(_StubChatAnthropic("claude-3-5-sonnet-20241022"))
        assert caps.supports_anthropic_cache_control is True

    def test_chat_anthropic_unknown_model_degrades(self) -> None:
        """anthropic adapter with an unfamiliar model name degrades cleanly.

        :raises AssertionError: when degraded record is not all-False
        """
        caps = detect_capabilities(_StubChatAnthropic("mystery-model"))
        assert caps.supports_anthropic_cache_control is False

    def test_chat_openai_reports_auto_cache(self) -> None:
        """openai adapters report the auto-cache capability, not anthropic.

        :raises AssertionError: when openai record is missed
        """
        caps = detect_capabilities(_StubChatOpenAI("gpt-4o"))
        assert caps.supports_openai_auto_cache is True
        assert caps.supports_anthropic_cache_control is False

    def test_unknown_adapter_degrades(self) -> None:
        """unknown adapter class returns the all-False record.

        :raises AssertionError: when degradation is not silent
        """
        caps = detect_capabilities(_StubUnknownChatModel("anything"))
        assert caps.supports_anthropic_cache_control is False
        assert caps.supports_openai_auto_cache is False
        assert caps.min_cacheable_tokens == 0
        assert caps.cache_ttl_seconds == 0

    def test_openrouter_anthropic_slug_supports_cache_control(self) -> None:
        """openrouter routed to anthropic supports cache_control directives.

        the wire provider is openrouter; the upstream is anthropic
        (the slug carries ``anthropic/...``). OpenRouter forwards
        ``cache_control`` markers in the request body to the
        upstream Anthropic API; the response surfaces the same
        cache-usage telemetry as a direct Anthropic call.

        :raises AssertionError: when openrouter-anthropic is not
            mapped to the anthropic capability record
        """
        caps = detect_capabilities(
            _StubChatOpenRouter("anthropic/claude-sonnet-4.6"),
        )
        assert caps.supports_anthropic_cache_control is True
        assert caps.supports_openai_auto_cache is False
        assert caps.min_cacheable_tokens == 1024
        assert caps.cache_ttl_seconds > 0

    def test_openrouter_anthropic_subclass_supports_cache_control(self) -> None:
        """the metallm-side ``_NameTranslatingChatOpenRouter`` subclass is recognized.

        production traffic flows through the tool-name-translating
        subclass returned by
        :func:`threetears.models.providers.openrouter.create_openrouter_chat`;
        the bare ``ChatOpenRouter`` is not the runtime class.
        capability detection must check both names.

        :raises AssertionError: when the subclass falls through to
            the all-False record
        """
        caps = detect_capabilities(
            _StubNameTranslatingChatOpenRouter("anthropic/claude-opus-4.5"),
        )
        assert caps.supports_anthropic_cache_control is True

    def test_openrouter_openai_slug_supports_auto_cache(self) -> None:
        """openrouter routed to openai gets the openai auto-cache record.

        :raises AssertionError: when openrouter-openai is not
            mapped to the openai capability record
        """
        caps = detect_capabilities(_StubChatOpenRouter("openai/gpt-4o"))
        assert caps.supports_openai_auto_cache is True
        assert caps.supports_anthropic_cache_control is False

    def test_openrouter_deepseek_slug_supports_auto_cache(self) -> None:
        """openrouter routed to deepseek gets the openai-style auto-cache record.

        deepseek's direct API runs automatic context caching and
        surfaces ``cached_tokens`` on the response; from the request
        side it is the same shape as openai auto-cache (stable
        prefix, no ``cache_control`` markers). Cache pricing differs
        and is the caller's concern; capability detection is only
        about what cache hints the request body should carry.

        :raises AssertionError: when openrouter-deepseek is not
            mapped to the auto-cache capability record
        """
        caps = detect_capabilities(
            _StubChatOpenRouter("deepseek/deepseek-v4-pro:nitro"),
        )
        assert caps.supports_openai_auto_cache is True
        assert caps.supports_anthropic_cache_control is False

    def test_openrouter_unknown_upstream_degrades(self) -> None:
        """openrouter routing to an unrecognized slug returns the all-False record.

        Conservative degradation: when the upstream prefix is not in
        the known set (``anthropic/``, ``openai/``, ``deepseek/``)
        the function falls through to the no-cache record so the
        node path emits a bare system message instead of guessing
        cache support that the upstream may not honor.

        :raises AssertionError: when an unknown slug is not degraded
        """
        caps = detect_capabilities(
            _StubChatOpenRouter("meta-llama/llama-3-70b"),
        )
        assert caps.supports_anthropic_cache_control is False
        assert caps.supports_openai_auto_cache is False


class TestAnnotateSystemPrompt:
    """:func:`annotate_system_prompt` shapes content for the adapter."""

    def test_cache_capable_returns_structured_content(self) -> None:
        """caching-capable caps produce structured content with cache_control.

        :raises AssertionError: when the structured shape is not emitted
        """
        caps = ChatModelCapabilities(
            supports_anthropic_cache_control=True,
            supports_openai_auto_cache=False,
            min_cacheable_tokens=1024,
            cache_ttl_seconds=300,
        )
        msg = annotate_system_prompt("You are helpful.", caps)
        assert isinstance(msg, SystemMessage)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 1
        block = msg.content[0]
        assert block == {
            "type": "text",
            "text": "You are helpful.",
            "cache_control": {"type": "ephemeral"},
        }

    def test_non_caching_returns_bare_string(self) -> None:
        """non-caching caps leave the content as a plain string.

        :raises AssertionError: when the bare-string shape is altered
        """
        caps = ChatModelCapabilities(
            supports_anthropic_cache_control=False,
            supports_openai_auto_cache=True,
            min_cacheable_tokens=0,
            cache_ttl_seconds=0,
        )
        msg = annotate_system_prompt("You are helpful.", caps)
        assert isinstance(msg, SystemMessage)
        assert msg.content == "You are helpful."


class TestExtractCacheUsage:
    """:func:`extract_cache_usage` normalizes diverse telemetry shapes."""

    def test_anthropic_cache_read_surfaces_in_normalized_dict(self) -> None:
        """anthropic-style usage_metadata maps to ``cache_read_input_tokens``.

        :raises AssertionError: when the key is not populated
        """
        response = AIMessage(
            content="ok",
            usage_metadata={
                "input_tokens": 1200,
                "output_tokens": 40,
                "total_tokens": 1240,
                "input_token_details": {
                    "cache_read": 1100,
                    "cache_creation": 0,
                },
            },
        )
        usage = extract_cache_usage(response)
        assert usage["cache_read_input_tokens"] == 1100
        assert usage["cached_tokens"] == 1100
        assert usage["cache_creation_input_tokens"] == 0

    def test_anthropic_cache_creation_surfaces(self) -> None:
        """anthropic cache_creation maps to ``cache_creation_input_tokens``.

        :raises AssertionError: when creation counter is missed
        """
        response = AIMessage(
            content="ok",
            usage_metadata={
                "input_tokens": 1200,
                "output_tokens": 40,
                "total_tokens": 1240,
                "input_token_details": {
                    "cache_read": 0,
                    "cache_creation": 900,
                },
            },
        )
        usage = extract_cache_usage(response)
        assert usage["cache_creation_input_tokens"] == 900

    def test_openai_cache_read_surfaces_as_cached_tokens(self) -> None:
        """openai ``input_token_details.cache_read`` maps to ``cached_tokens``.

        :raises AssertionError: when openai telemetry is missed
        """
        response = AIMessage(
            content="ok",
            usage_metadata={
                "input_tokens": 1000,
                "output_tokens": 25,
                "total_tokens": 1025,
                "input_token_details": {"cache_read": 750},
            },
        )
        usage = extract_cache_usage(response)
        assert usage["cached_tokens"] == 750

    def test_missing_usage_metadata_returns_zeros(self) -> None:
        """response without ``usage_metadata`` yields an all-zero dict.

        :raises AssertionError: when the function raises or populates
            non-zero values
        """
        response = AIMessage(content="ok")
        usage = extract_cache_usage(response)
        assert usage == {
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cached_tokens": 0,
        }

    def test_response_metadata_fallback_picks_up_cache_fields(self) -> None:
        """cache counters on ``response_metadata.usage`` populate the dict.

        :raises AssertionError: when the fallback is not consulted
        """
        response = AIMessage(content="ok")
        response.response_metadata = {
            "usage": {
                "cache_read_input_tokens": 500,
                "cache_creation_input_tokens": 200,
            },
        }
        usage = extract_cache_usage(response)
        assert usage["cache_read_input_tokens"] == 500
        assert usage["cache_creation_input_tokens"] == 200


class TestComputeToolKey:
    """:func:`compute_tool_key` hashes the tool fingerprint stably."""

    def test_empty_list_returns_deterministic_digest(self) -> None:
        """empty list hashes to a stable 16-hex-char digest.

        :raises AssertionError: when the digest length differs
        """
        digest = compute_tool_key([])
        assert isinstance(digest, str)
        assert len(digest) == 16

    def test_same_tools_in_different_order_yield_same_key(self) -> None:
        """tool-list ordering does not influence the digest.

        :raises AssertionError: when ordering leaks into the hash
        """
        tool_a = _StubTool("a", _StubPydanticArgsSchema({"type": "object"}))
        tool_b = _StubTool("b", _StubPydanticArgsSchema({"type": "object"}))
        key1 = compute_tool_key([tool_a, tool_b])
        key2 = compute_tool_key([tool_b, tool_a])
        assert key1 == key2

    def test_different_tool_sets_yield_different_keys(self) -> None:
        """changing a tool name flips the digest.

        :raises AssertionError: when the digest is insensitive to names
        """
        tool_a = _StubTool("a", _StubPydanticArgsSchema({"type": "object"}))
        tool_c = _StubTool("c", _StubPydanticArgsSchema({"type": "object"}))
        assert compute_tool_key([tool_a]) != compute_tool_key([tool_c])

    def test_different_schemas_yield_different_keys(self) -> None:
        """changing args_schema output flips the digest.

        :raises AssertionError: when schema drift does not propagate
        """
        tool_a = _StubTool("a", _StubPydanticArgsSchema({"type": "object"}))
        tool_a_v2 = _StubTool(
            "a",
            _StubPydanticArgsSchema({"type": "object", "properties": {"x": {}}}),
        )
        assert compute_tool_key([tool_a]) != compute_tool_key([tool_a_v2])

    def test_tool_without_args_schema_still_hashes(self) -> None:
        """tools lacking ``args_schema`` still contribute the name.

        :raises AssertionError: when hashing crashes or collapses keys
        """
        tool_plain = _StubTool("plain", args_schema=None)
        tool_other = _StubTool("other", args_schema=None)
        assert compute_tool_key([tool_plain]) != compute_tool_key([tool_other])


class TestShouldBindToolsFresh:
    """:func:`should_bind_tools_fresh` triggers rebinds on tool-set drift."""

    def test_first_bind_is_always_fresh(self) -> None:
        """``None`` previous key means the model has never been bound.

        :raises AssertionError: when the first bind is skipped
        """
        assert should_bind_tools_fresh(None, "abc") is True

    def test_matching_keys_skip_rebind(self) -> None:
        """matching previous and current keys skip the rebind.

        :raises AssertionError: when matching keys still trigger rebind
        """
        assert should_bind_tools_fresh("abc", "abc") is False

    def test_drifting_keys_trigger_rebind(self) -> None:
        """differing previous and current keys trigger rebind.

        :raises AssertionError: when key drift is ignored
        """
        assert should_bind_tools_fresh("abc", "def") is True
