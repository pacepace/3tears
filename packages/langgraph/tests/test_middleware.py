"""Behavioral tests for :class:`PromptCachingMiddleware`.

The framework-aligned successor to the old ``PromptCachingHook`` AST guards
(``test_prompt_caching_ast_guards.py``, removed with ``hooks.py``). These verify
the two PRESERVED invariants behaviorally rather than by source structure:

1. the request's bare-string ``system_message`` is rewritten to ``cache_control``
   structured content for cache-capable models, and left untouched otherwise
   (idempotent / no-op);
2. the cache-usage path normalizes counters via ``extract_cache_usage``.

Invariant (1) is verified BOTH on the annotation helper AND *through the seam*
(driving ``awrap_model_call`` with a real ``ModelRequest``): ``create_agent``
carries the system prompt on ``request.system_message`` and EXCLUDES it from
``request.messages``, so a test that only inspects ``messages[0]`` would pass
while the middleware silently annotated nothing.

The third old guard (``bind_tools`` memoization) is intentionally NOT covered:
that behavior was retired in the move to ``create_agent`` (binds once at
construction, so per-round memoization is moot).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from threetears.models import DEFAULT_CHAT_MODEL

from threetears.langgraph.middleware import PromptCachingMiddleware


class _StubChatAnthropic:
    """lookalike whose class name ``detect_capabilities`` recognizes as Anthropic."""

    def __init__(self, model: str) -> None:
        self.model = model


_StubChatAnthropic.__name__ = "ChatAnthropic"


class _StubUnknownChatModel:
    """unrecognized adapter -> no cache support."""

    def __init__(self, model: str = "") -> None:
        self.model = model


def _system_message_seen(system_message: SystemMessage | None, chat_model: Any) -> SystemMessage | None:
    """run one model call through the middleware and return the system message the model received.

    :param system_message: the request's system message
    :ptype system_message: SystemMessage | None
    :param chat_model: the model the request targets
    :ptype chat_model: Any
    :return: the system message on the request the handler saw
    :rtype: SystemMessage | None
    """
    seen: list[ModelRequest] = []

    def _handler(req: ModelRequest) -> Any:
        seen.append(req)
        return SimpleNamespace(result=[AIMessage(content="ok")])

    request = ModelRequest(
        model=cast("BaseChatModel", chat_model),
        messages=[HumanMessage(content="hi")],
        system_message=system_message,
    )
    PromptCachingMiddleware().wrap_model_call(request, _handler)
    return seen[0].system_message


def _stamp_through_the_middleware(result: list[Any] | None) -> None:
    """run one model call whose handler answers *result*, so the middleware stamps it.

    :param result: the response's message list
    :ptype result: list[Any] | None
    :return: nothing
    :rtype: None
    """
    request = ModelRequest(
        model=cast("BaseChatModel", _StubUnknownChatModel("gpt-4")),
        messages=[HumanMessage(content="hi")],
        system_message=None,
    )
    PromptCachingMiddleware().wrap_model_call(request, lambda _req: SimpleNamespace(result=result))


class TestAnnotateForCache:
    def test_adds_cache_control_for_anthropic(self) -> None:
        out = _system_message_seen(SystemMessage(content="sys"), _StubChatAnthropic(DEFAULT_CHAT_MODEL))
        assert isinstance(out, SystemMessage)
        # structured (list) content is the cache_control-carrying form
        assert isinstance(out.content, list)
        assert out.content[-1]["cache_control"] == {"type": "ephemeral"}

    def test_noop_for_non_caching_model(self) -> None:
        original = SystemMessage(content="sys")
        out = _system_message_seen(original, _StubUnknownChatModel("gpt-4"))
        assert out is original

    def test_idempotent_on_already_structured_content(self) -> None:
        structured = SystemMessage(
            content=[{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
        )
        out = _system_message_seen(structured, _StubChatAnthropic(DEFAULT_CHAT_MODEL))
        assert out is structured  # already structured -> left alone

    def test_noop_without_system_message(self) -> None:
        out = _system_message_seen(None, _StubChatAnthropic(DEFAULT_CHAT_MODEL))
        assert out is None


class TestThroughTheSeam:
    """Drive the real ``awrap_model_call`` so annotation is verified on the
    ``request.system_message`` the model actually receives -- the regression a
    ``messages[0]``-only test would miss.
    """

    def test_awrap_annotates_system_message(self) -> None:
        captured: dict[str, Any] = {}

        async def _handler(req: ModelRequest) -> Any:
            captured["req"] = req
            return SimpleNamespace(result=[AIMessage(content="ok")])

        async def _run() -> None:
            request = ModelRequest(
                model=cast("BaseChatModel", _StubChatAnthropic(DEFAULT_CHAT_MODEL)),
                messages=[HumanMessage(content="hi")],
                system_message=SystemMessage(content="sys"),
            )
            await PromptCachingMiddleware().awrap_model_call(request, _handler)

        asyncio.run(_run())
        seen = captured["req"]
        # annotated cache_control structured form, applied THROUGH the seam
        assert isinstance(seen.system_message.content, list)
        # the reduced messages list still excludes the system message
        assert all(not isinstance(m, SystemMessage) for m in seen.messages)

    def test_awrap_noop_for_non_caching_model_leaves_system_bare(self) -> None:
        captured: dict[str, Any] = {}

        async def _handler(req: ModelRequest) -> Any:
            captured["req"] = req
            return SimpleNamespace(result=[AIMessage(content="ok")])

        async def _run() -> None:
            request = ModelRequest(
                model=cast("BaseChatModel", _StubUnknownChatModel("gpt-4")),
                messages=[HumanMessage(content="hi")],
                system_message=SystemMessage(content="sys"),
            )
            await PromptCachingMiddleware().awrap_model_call(request, _handler)

        asyncio.run(_run())
        assert captured["req"].system_message.content == "sys"  # unchanged


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
        _stamp_through_the_middleware([ai])
        assert isinstance(ai.usage_metadata["cache_usage"], dict)

    def test_handles_empty_result_without_error(self) -> None:
        _stamp_through_the_middleware([])
        _stamp_through_the_middleware(None)


class TestMiddlewareShape:
    def test_is_agent_middleware(self) -> None:
        m = PromptCachingMiddleware()
        assert isinstance(m, AgentMiddleware)
        assert m.name == "PromptCachingMiddleware"
        assert hasattr(m, "awrap_model_call")
        assert hasattr(m, "wrap_model_call")
