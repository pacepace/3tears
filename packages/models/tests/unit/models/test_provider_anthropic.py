"""tests for :func:`create_anthropic_chat` factory and capability registration."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from threetears.models.capabilities import get_capabilities
from threetears.models.enums import ModelTier, ModelType
from threetears.models.providers.anthropic import (
    ANTHROPIC_PROVIDER_NAME,
    create_anthropic_chat,
    strip_v1_suffix,
)


class TestCreateAnthropicChat:
    """tests for the ``create_anthropic_chat`` factory function."""

    def test_returns_base_chat_model(self) -> None:
        """factory returns a ``BaseChatModel`` subclass instance."""
        model = create_anthropic_chat("claude-sonnet-4-20250514", "sk-test")
        assert isinstance(model, BaseChatModel)
        assert isinstance(model, ChatAnthropic)

    def test_model_name_propagated(self) -> None:
        """factory forwards model_name to ``ChatAnthropic``."""
        model = create_anthropic_chat("claude-3-5-haiku-20241022", "sk-test")
        assert model.model == "claude-3-5-haiku-20241022"

    def test_strips_v1_suffix_from_base_url(self) -> None:
        """trailing ``/v1`` is stripped from base_url before instantiation."""
        model = create_anthropic_chat(
            "claude-sonnet-4-20250514",
            "sk-test",
            base_url="https://api.anthropic.com/v1",
        )
        assert model.anthropic_api_url == "https://api.anthropic.com"

    def test_strips_v1_slash_suffix(self) -> None:
        """trailing ``/v1/`` is also stripped."""
        model = create_anthropic_chat(
            "claude-sonnet-4-20250514",
            "sk-test",
            base_url="https://api.anthropic.com/v1/",
        )
        assert model.anthropic_api_url == "https://api.anthropic.com"

    def test_no_base_url_means_default(self) -> None:
        """omitting base_url leaves ``ChatAnthropic`` to apply its default."""
        model = create_anthropic_chat("claude-sonnet-4-20250514", "sk-test")
        # ChatAnthropic defaults the URL itself; just verify it is not None.
        assert model.anthropic_api_url is not None


class TestStripV1Suffix:
    """tests for the ``strip_v1_suffix`` helper."""

    def test_strips_v1(self) -> None:
        """``/v1`` suffix is removed."""
        assert strip_v1_suffix("https://api.anthropic.com/v1") == "https://api.anthropic.com"

    def test_strips_v1_slash(self) -> None:
        """``/v1/`` suffix is removed."""
        assert strip_v1_suffix("https://api.anthropic.com/v1/") == "https://api.anthropic.com"

    def test_no_v1_unchanged(self) -> None:
        """URL without ``/v1`` suffix is returned unchanged."""
        assert strip_v1_suffix("https://custom.api.com") == "https://custom.api.com"

    def test_v1_in_middle_unchanged(self) -> None:
        """URL with ``/v1`` in middle path is returned unchanged."""
        assert strip_v1_suffix("https://api.com/v1/extra") == "https://api.com/v1/extra"


class TestAnthropicCapabilityRegistration:
    """tests that anthropic-canonical models register at import time."""

    def test_sonnet_registered(self) -> None:
        """``claude-sonnet-4-20250514`` resolves to anthropic chat capabilities."""
        caps = get_capabilities("claude-sonnet-4-20250514")
        assert caps is not None
        assert caps.provider_name == ANTHROPIC_PROVIDER_NAME
        assert caps.model_type == ModelType.CHAT
        assert caps.model_tier == ModelTier.LARGE
        assert caps.supports_tools is True

    def test_haiku_registered(self) -> None:
        """``claude-3-5-haiku-20241022`` resolves to anthropic small-tier chat."""
        caps = get_capabilities("claude-3-5-haiku-20241022")
        assert caps is not None
        assert caps.provider_name == ANTHROPIC_PROVIDER_NAME
        assert caps.model_tier == ModelTier.SMALL


class TestAnthropicWrapperStreaming:
    """Regression coverage for the wrapper-_astream callback-chain bug.

    The wrapper used to override ``_astream`` to translate
    ``tool_call_chunks`` names back to canonical form. That override --
    even as a pass-through -- broke LangGraph's
    ``astream_events(version="v2")`` event tap, leaving event-driven
    UIs (metallm's WS handler, debug-inject endpoint) with the saved DB
    content but a blank live stream. Same bug class as the OpenRouter
    wrapper (see ``test_provider_openrouter.py`` for the full story).

    The fix moves translation off ``_astream`` and onto ``astream`` (the
    public Runnable method), so ``BaseChatModel.astream``'s callback
    wiring runs unchanged against ``ChatAnthropic._astream``'s untouched
    output. These tests pin that contract.
    """

    @pytest.mark.asyncio
    async def test_astream_events_emits_on_chat_model_stream(self) -> None:
        """``astream_events(version="v2")`` must emit
        ``on_chat_model_stream`` events for every chunk the wrapper
        passes through.
        """

        async def _fake_super_astream(
            self: Any,
            messages: Any,
            stop: Any = None,
            run_manager: Any = None,
            **kwargs: Any,
        ):
            del self, messages, stop, kwargs
            for text in ("anthro", "pic ", "wrapper ", "ok"):
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
                if run_manager is not None:
                    await run_manager.on_llm_new_token(token=text, chunk=chunk)
                yield chunk

        model = create_anthropic_chat("claude-sonnet-4-5-20250929", "sk-test")

        original_astream = ChatAnthropic._astream
        try:
            ChatAnthropic._astream = _fake_super_astream  # type: ignore[method-assign]
            stream_event_count = 0
            collected_text = ""
            async for event in model.astream_events("hi", version="v2"):
                if event["event"] == "on_chat_model_stream":
                    stream_event_count += 1
                    collected_text += event["data"]["chunk"].content
        finally:
            ChatAnthropic._astream = original_astream  # type: ignore[method-assign]

        # 4 fake chunks plus the framework's final empty chunk.
        assert stream_event_count >= 4, (
            f"Expected >=4 on_chat_model_stream events; got"
            f" {stream_event_count}. The Anthropic wrapper is breaking"
            f" the callback chain that drives astream_events(v2)."
        )
        assert collected_text == "anthropic wrapper ok"
