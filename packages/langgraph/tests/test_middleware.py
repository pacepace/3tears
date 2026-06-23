"""Behavioral tests for :class:`PromptCachingMiddleware`.

The framework-aligned successor to the old ``PromptCachingHook`` AST guards
(``test_prompt_caching_ast_guards.py``, removed with ``hooks.py``). These verify
the two PRESERVED invariants behaviorally rather than by source structure:

1. cache-path SystemMessages carry list (``cache_control``) content for
   cache-capable models, and are left untouched otherwise (idempotent / no-op);
2. the cache-usage path normalizes counters via ``extract_cache_usage``.

The third old guard (``bind_tools`` memoization) is intentionally NOT covered:
that behavior was retired in the move to ``create_agent`` (binds once at
construction, so per-round memoization is moot).
"""

from __future__ import annotations

from types import SimpleNamespace

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from threetears.models import DEFAULT_CHAT_MODEL

from threetears.langgraph.middleware import (
    PromptCachingMiddleware,
    _annotate_messages_for_cache,
    _stamp_cache_usage,
)


class _StubChatAnthropic:
    """lookalike whose class name ``detect_capabilities`` recognizes as Anthropic."""

    def __init__(self, model: str) -> None:
        self.model = model


_StubChatAnthropic.__name__ = "ChatAnthropic"


class _StubUnknownChatModel:
    """unrecognized adapter -> no cache support."""

    def __init__(self, model: str = "") -> None:
        self.model = model


class TestAnnotateForCache:
    def test_adds_cache_control_for_anthropic(self) -> None:
        msgs = [SystemMessage(content="sys"), HumanMessage(content="hi")]
        out = _annotate_messages_for_cache(msgs, _StubChatAnthropic(DEFAULT_CHAT_MODEL))
        assert out is not msgs  # rewritten
        assert isinstance(out[0], SystemMessage)
        # structured (list) content is the cache_control-carrying form
        assert isinstance(out[0].content, list)
        assert out[1] is msgs[1]  # non-system turns untouched

    def test_noop_for_non_caching_model(self) -> None:
        msgs = [SystemMessage(content="sys"), HumanMessage(content="hi")]
        out = _annotate_messages_for_cache(msgs, _StubUnknownChatModel("gpt-4"))
        assert out is msgs

    def test_idempotent_on_already_structured_content(self) -> None:
        structured = SystemMessage(
            content=[{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
        )
        msgs = [structured, HumanMessage(content="hi")]
        out = _annotate_messages_for_cache(msgs, _StubChatAnthropic(DEFAULT_CHAT_MODEL))
        assert out is msgs  # already structured -> left alone

    def test_noop_without_leading_system_message(self) -> None:
        msgs = [HumanMessage(content="hi")]
        out = _annotate_messages_for_cache(msgs, _StubChatAnthropic(DEFAULT_CHAT_MODEL))
        assert out is msgs


class TestStampCacheUsage:
    def test_normalizes_cache_usage_onto_ai_message(self) -> None:
        ai = AIMessage(
            content="x",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "input_token_details": {"cache_read": 4},
            },
        )
        _stamp_cache_usage(SimpleNamespace(result=[ai]))
        assert isinstance(ai.usage_metadata["cache_usage"], dict)

    def test_handles_empty_result_without_error(self) -> None:
        _stamp_cache_usage(SimpleNamespace(result=[]))
        _stamp_cache_usage(SimpleNamespace(result=None))


class TestMiddlewareShape:
    def test_is_agent_middleware(self) -> None:
        m = PromptCachingMiddleware()
        assert isinstance(m, AgentMiddleware)
        assert m.name == "PromptCachingMiddleware"
        assert hasattr(m, "awrap_model_call")
        assert hasattr(m, "wrap_model_call")
