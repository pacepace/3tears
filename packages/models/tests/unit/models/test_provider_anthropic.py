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

    def test_sonnet_cache_fields(self) -> None:
        """``claude-sonnet-4-20250514`` carries anthropic-shape cache fields.

        Every Anthropic chat-model registration declares the anthropic
        cache shape (cache_control supported, 1024-token minimum,
        300-second ephemeral TTL) so consumers that resolve through
        ``get_capabilities`` get the right caching record without
        having to maintain a parallel per-provider table.
        """
        caps = get_capabilities("claude-sonnet-4-20250514")
        assert caps is not None
        assert caps.supports_anthropic_cache_control is True
        assert caps.supports_openai_auto_cache is False
        assert caps.min_cacheable_tokens == 1024
        assert caps.cache_ttl_seconds == 300

    def test_opus_cache_fields(self) -> None:
        """``claude-opus-4-5-20251101`` carries anthropic-shape cache fields."""
        caps = get_capabilities("claude-opus-4-5-20251101")
        assert caps is not None
        assert caps.supports_anthropic_cache_control is True
        assert caps.supports_openai_auto_cache is False
        assert caps.min_cacheable_tokens == 1024
        assert caps.cache_ttl_seconds == 300

    def test_haiku_cache_fields(self) -> None:
        """``claude-3-5-haiku-20241022`` carries anthropic-shape cache fields."""
        caps = get_capabilities("claude-3-5-haiku-20241022")
        assert caps is not None
        assert caps.supports_anthropic_cache_control is True
        assert caps.supports_openai_auto_cache is False
        assert caps.min_cacheable_tokens == 1024
        assert caps.cache_ttl_seconds == 300


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

    @pytest.mark.asyncio
    async def test_astream_events_survives_with_config_callbacks(self) -> None:
        """``with_config(callbacks=[...])`` must not strip the event_streamer.

        Same production failure mode as the OpenRouter wrapper -- when
        wrapped by ``model.with_config(callbacks=[UsageTracker,
        CircuitBreaker])`` (as ``threetears.models.factory.create_chat_model``
        does), the wrapper used to forward its incoming ``config``
        verbatim to ``super().astream(...)``, causing
        ``BaseChatModel.astream``'s ``ensure_config(config)`` to replace
        the contextvar's ``AsyncCallbackManager`` (carrying
        ``astream_events``' event_streamer) with the bound list of
        handlers. The fix pre-merges via
        :func:`merge_configs(ensure_config(None), config)` so both the
        event_streamer AND the bound list propagate.

        Anthropic shares the failure mode with OpenRouter / OpenAI
        because all three wrappers use the same override shape. The
        OpenRouter case is the one metallm reproduced in production
        (2026-05-13 conv ``019e2243-de0c``); this test pins the same
        contract on the Anthropic wrapper so any future divergence in
        wrapper behavior fails CI on this provider too.
        """
        from langchain_core.callbacks import AsyncCallbackHandler

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

        class _RecordingCallback(AsyncCallbackHandler):
            def __init__(self) -> None:
                self.start_seen = 0
                self.token_seen = 0

            async def on_chat_model_start(
                self,
                serialized: Any,
                messages: Any,
                **_: Any,
            ) -> None:
                del serialized, messages
                self.start_seen += 1

            async def on_llm_new_token(self, token: str, **_: Any) -> None:
                del token
                self.token_seen += 1

        bound_cb = _RecordingCallback()
        model = create_anthropic_chat("claude-sonnet-4-5-20250929", "sk-test")
        bound_model = model.with_config(callbacks=[bound_cb])

        original_astream = ChatAnthropic._astream
        try:
            ChatAnthropic._astream = _fake_super_astream  # type: ignore[method-assign]
            stream_event_count = 0
            async for event in bound_model.astream_events("hi", version="v2"):
                if event["event"] == "on_chat_model_stream":
                    stream_event_count += 1
        finally:
            ChatAnthropic._astream = original_astream  # type: ignore[method-assign]

        assert stream_event_count >= 4, (
            "with_config-bound list callbacks REPLACED the contextvar's"
            " event_streamer manager — `on_chat_model_stream` events"
            f" never reached astream_events. Got {stream_event_count}."
        )
        assert bound_cb.start_seen >= 1, (
            "Bound callback's on_chat_model_start never fired — the"
            " fix dropped the with_config list when preserving the"
            " contextvar manager. Both must propagate."
        )
        assert bound_cb.token_seen >= 4, (
            f"Bound callback's on_llm_new_token fired {bound_cb.token_seen}"
            " times — expected >=4 (one per fake chunk). The fix"
            " silently dropped the list of bound handlers."
        )
