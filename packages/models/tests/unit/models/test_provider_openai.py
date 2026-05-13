"""tests for OpenAI chat and embedding factories and capability registration."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from threetears.models.capabilities import get_capabilities
from threetears.models.enums import ModelTier, ModelType
from threetears.models.providers.openai import (
    OPENAI_PROVIDER_NAME,
    create_openai_chat,
    create_openai_embedding,
)


class TestCreateOpenAIChat:
    """tests for ``create_openai_chat`` factory."""

    def test_returns_base_chat_model(self) -> None:
        """factory returns a ``BaseChatModel`` subclass instance."""
        model = create_openai_chat("gpt-4o", "sk-test")
        assert isinstance(model, BaseChatModel)
        assert isinstance(model, ChatOpenAI)

    def test_model_name_propagated(self) -> None:
        """factory forwards model name to ``ChatOpenAI``."""
        model = create_openai_chat("gpt-4o-mini", "sk-test")
        assert model.model_name == "gpt-4o-mini"

    def test_base_url_passed_through(self) -> None:
        """custom base_url is forwarded to ``ChatOpenAI``."""
        model = create_openai_chat("gpt-4o", "sk-test", base_url="https://api.example.com")
        assert str(model.openai_api_base) == "https://api.example.com"


class TestCreateOpenAIEmbedding:
    """tests for ``create_openai_embedding`` factory."""

    def test_returns_embeddings(self) -> None:
        """factory returns an ``Embeddings`` subclass instance."""
        model = create_openai_embedding("text-embedding-3-small", "sk-test")
        assert isinstance(model, Embeddings)
        assert isinstance(model, OpenAIEmbeddings)

    def test_dimensions_forwarded(self) -> None:
        """``embedding_dimensions`` is mapped to ``OpenAIEmbeddings.dimensions``."""
        model = create_openai_embedding("text-embedding-3-small", "sk-test", embedding_dimensions=512)
        assert model.dimensions == 512


class TestOpenAICapabilityRegistration:
    """tests that openai canonical models register at import time."""

    def test_gpt4o_registered(self) -> None:
        """``gpt-4o`` resolves to openai LARGE chat capabilities."""
        caps = get_capabilities("gpt-4o")
        assert caps is not None
        assert caps.provider_name == OPENAI_PROVIDER_NAME
        assert caps.model_type == ModelType.CHAT
        assert caps.model_tier == ModelTier.LARGE

    def test_text_embedding_3_small_registered(self) -> None:
        """``text-embedding-3-small`` resolves to openai embedding capabilities."""
        caps = get_capabilities("text-embedding-3-small")
        assert caps is not None
        assert caps.provider_name == OPENAI_PROVIDER_NAME
        assert caps.model_type == ModelType.EMBEDDING
        assert caps.embedding_dimensions == 1536


class TestOpenAIWrapperStreaming:
    """Regression coverage for the wrapper-_astream callback-chain bug.

    Same bug class as the OpenRouter and Anthropic wrappers (see
    ``test_provider_openrouter.py`` for the full story). The OpenAI
    wrapper had the same broken ``_astream`` override pattern; this
    test pins the contract that the new ``astream`` override emits
    ``on_chat_model_stream`` events through ``astream_events(v2)``.

    Today's gateway path drives ``astream`` not ``astream_events`` so
    the OpenAI wrapper wasn't actually triggering the bug in
    production -- but the parity fix landed in v0.5.1 along with the
    OpenRouter and Anthropic fixes so a future consumer driving the v2
    event tap through ``create_openai_chat`` doesn't repeat the saga.
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
            for text in ("open", "ai ", "wrapper ", "ok"):
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
                if run_manager is not None:
                    await run_manager.on_llm_new_token(token=text, chunk=chunk)
                yield chunk

        model = create_openai_chat("gpt-4o", "sk-test")

        original_astream = ChatOpenAI._astream
        try:
            ChatOpenAI._astream = _fake_super_astream  # type: ignore[method-assign]
            stream_event_count = 0
            collected_text = ""
            async for event in model.astream_events("hi", version="v2"):
                if event["event"] == "on_chat_model_stream":
                    stream_event_count += 1
                    collected_text += event["data"]["chunk"].content
        finally:
            ChatOpenAI._astream = original_astream  # type: ignore[method-assign]

        # 4 fake chunks plus the framework's final empty chunk.
        assert stream_event_count >= 4, (
            f"Expected >=4 on_chat_model_stream events; got"
            f" {stream_event_count}. The OpenAI wrapper is breaking the"
            f" callback chain that drives astream_events(v2)."
        )
        assert collected_text == "openai wrapper ok"

    @pytest.mark.asyncio
    async def test_astream_events_survives_with_config_callbacks(self) -> None:
        """``with_config(callbacks=[...])`` must not strip the event_streamer.

        Same production failure mode as the OpenRouter wrapper. See
        :class:`TestNameTranslatingChatOpenRouter` (or the OpenRouter
        wrapper docstring) for the full incident write-up. This pins
        the contract on the OpenAI wrapper so the bug can't reappear
        here either.
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
            for text in ("open", "ai ", "wrapper ", "ok"):
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
        model = create_openai_chat("gpt-4o", "sk-test")
        bound_model = model.with_config(callbacks=[bound_cb])

        original_astream = ChatOpenAI._astream
        try:
            ChatOpenAI._astream = _fake_super_astream  # type: ignore[method-assign]
            stream_event_count = 0
            async for event in bound_model.astream_events("hi", version="v2"):
                if event["event"] == "on_chat_model_stream":
                    stream_event_count += 1
        finally:
            ChatOpenAI._astream = original_astream  # type: ignore[method-assign]

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
