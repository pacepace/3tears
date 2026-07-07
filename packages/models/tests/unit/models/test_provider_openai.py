"""tests for OpenAI chat and embedding factories and capability registration."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from threetears.models.capabilities import get_capabilities
from threetears.models.enums import ModelTier, ModelType
from threetears.models.providers.openai import (
    OPENAI_PROVIDER_NAME,
    create_openai_chat,
    create_openai_embedding,
)

from ._translation_helpers import DottedTool


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

    def test_gpt4o_cache_fields(self) -> None:
        """``gpt-4o`` carries openai auto-cache fields.

        Every OpenAI chat-model registration declares the auto-cache
        shape (cache_control not supported, openai-auto-cache supported,
        no token minimum, no TTL) so consumers that resolve through
        ``get_capabilities`` get the right caching record.
        """
        caps = get_capabilities("gpt-4o")
        assert caps is not None
        assert caps.supports_anthropic_cache_control is False
        assert caps.supports_openai_auto_cache is True
        assert caps.min_cacheable_tokens == 0
        assert caps.cache_ttl_seconds == 0

    def test_gpt4o_mini_cache_fields(self) -> None:
        """``gpt-4o-mini`` carries openai auto-cache fields."""
        caps = get_capabilities("gpt-4o-mini")
        assert caps is not None
        assert caps.supports_anthropic_cache_control is False
        assert caps.supports_openai_auto_cache is True
        assert caps.min_cacheable_tokens == 0
        assert caps.cache_ttl_seconds == 0

    def test_embedding_models_have_no_cache_fields(self) -> None:
        """embedding-model entries leave cache fields at None (tri-state).

        Cache fields are chat-specific; embedding models don't
        participate in prompt caching, so their entries explicitly
        leave the fields unset. This protects callers that filter
        by ``supports_anthropic_cache_control is None`` to skip
        non-chat models.
        """
        small = get_capabilities("text-embedding-3-small")
        assert small is not None
        assert small.supports_anthropic_cache_control is None
        assert small.supports_openai_auto_cache is None
        assert small.min_cacheable_tokens is None
        assert small.cache_ttl_seconds is None

        large = get_capabilities("text-embedding-3-large")
        assert large is not None
        assert large.supports_anthropic_cache_control is None
        assert large.supports_openai_auto_cache is None
        assert large.min_cacheable_tokens is None
        assert large.cache_ttl_seconds is None


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


class TestOpenAIForwardTranslation:
    """The OpenAI wrapper forward-translates dotted tool-call names on the
    OUTBOUND ``messages`` before the provider call.

    OpenAI's tools API validates names against ``^[a-zA-Z0-9_-]{1,64}$`` and
    rejects the dot, so a prior round's ``AIMessage`` carrying a canonical
    dotted ``tool_calls`` name would 400 the turn when re-sent. Parity with
    the OpenRouter / Anthropic forward translation.
    """

    @pytest.mark.asyncio
    async def test_astream_forward_translates_outbound_dotted_names(self) -> None:
        """``astream`` sends the wire (underscored) name; the caller's message
        keeps the canonical dotted name."""

        captured: dict[str, Any] = {}

        async def _fake_super_astream(
            self: Any,
            messages: Any,
            stop: Any = None,
            run_manager: Any = None,
            **kwargs: Any,
        ):
            del self, stop, run_manager, kwargs
            captured["messages"] = messages
            yield ChatGenerationChunk(message=AIMessageChunk(content="ok"))

        model = create_openai_chat("gpt-4o", "sk-test")
        outbound = [
            SystemMessage(content="sys"),
            AIMessage(
                content="",
                tool_calls=[{"name": "threetears.web_search", "args": {"q": "x"}, "id": "c1"}],
            ),
            ToolMessage(content="hit", tool_call_id="c1"),
        ]

        original_astream = ChatOpenAI._astream
        try:
            ChatOpenAI._astream = _fake_super_astream  # type: ignore[method-assign]
            async for _ in model.astream(outbound):
                pass
        finally:
            ChatOpenAI._astream = original_astream  # type: ignore[method-assign]

        sent = captured["messages"]
        ai = [m for m in sent if isinstance(m, AIMessage) and m.tool_calls][0]
        assert ai.tool_calls[0]["name"] == "threetears_web_search"
        assert outbound[1].tool_calls[0]["name"] == "threetears.web_search"

    @pytest.mark.asyncio
    async def test_agenerate_forward_translates_outbound_dotted_names(self) -> None:
        """The non-streaming path mangles the outbound names too."""
        from langchain_core.outputs import ChatGeneration, ChatResult

        captured: dict[str, Any] = {}

        async def _fake_super_agenerate(
            self: Any,
            messages: Any,
            stop: Any = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> ChatResult:
            del self, stop, run_manager, kwargs
            captured["messages"] = messages
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

        model = create_openai_chat("gpt-4o", "sk-test")
        outbound = [
            AIMessage(
                content="",
                tool_calls=[{"name": "threetears.web_search", "args": {"q": "x"}, "id": "c1"}],
            ),
        ]

        original = ChatOpenAI._agenerate
        try:
            ChatOpenAI._agenerate = _fake_super_agenerate  # type: ignore[method-assign]
            await model._agenerate(outbound)
        finally:
            ChatOpenAI._agenerate = original  # type: ignore[method-assign]

        sent = captured["messages"]
        assert sent[0].tool_calls[0]["name"] == "threetears_web_search"
        assert outbound[0].tool_calls[0]["name"] == "threetears.web_search"


class TestOpenAIInvokeParity:
    """The OpenAI wrapper overrides the PUBLIC ``ainvoke`` / ``invoke`` too
    (via the shared ``NameTranslatingChatMixin``).

    Overriding ``astream`` + ``_agenerate`` is not sufficient: with a streaming
    callback attached (the converged ``agent_node`` path) ``ainvoke`` routes
    through the protected ``_astream`` aggregate, bypassing both. Regression for
    the 2026-06-22 converged-loop tool-dispatch failure, now covered uniformly
    across all three provider wrappers.
    """

    @pytest.mark.asyncio
    async def test_ainvoke_untranslates_when_aggregating_from_astream(self) -> None:
        """``ainvoke`` returns the canonical dotted name even when it aggregates
        internally from the protected ``_astream`` (the bypass path)."""

        async def _fake_super_astream(
            self: Any,
            messages: Any,
            stop: Any = None,
            run_manager: Any = None,
            **kwargs: Any,
        ):
            del self, messages, stop, run_manager, kwargs
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {"name": "threetears_calculator", "args": "{}", "id": "call_1", "index": 0},
                    ],
                ),
            )

        model = create_openai_chat("gpt-4o", "sk-test")
        model.bind_tools([DottedTool()])

        original_astream = ChatOpenAI._astream
        try:
            ChatOpenAI._astream = _fake_super_astream  # type: ignore[method-assign]
            result = await model.ainvoke("hi", stream=True)
        finally:
            ChatOpenAI._astream = original_astream  # type: ignore[method-assign]

        assert result.tool_calls, "expected an aggregated tool call"
        assert result.tool_calls[0]["name"] == "threetears.calculator"

    @pytest.mark.asyncio
    async def test_ainvoke_forward_translates_outbound_dotted_names(self) -> None:
        """``ainvoke`` mangles a dotted outbound tool-call name to wire form via
        its own forward-translation (the bypassed ``astream`` override does not
        run); the caller's message keeps the dotted name."""

        captured: dict[str, Any] = {}

        async def _fake_super_astream(
            self: Any,
            messages: Any,
            stop: Any = None,
            run_manager: Any = None,
            **kwargs: Any,
        ):
            del self, stop, run_manager, kwargs
            captured["messages"] = messages
            yield ChatGenerationChunk(message=AIMessageChunk(content="ok"))

        model = create_openai_chat("gpt-4o", "sk-test")
        outbound = [
            AIMessage(
                content="",
                tool_calls=[{"name": "threetears.web_search", "args": {"q": "x"}, "id": "c1"}],
            ),
        ]

        original_astream = ChatOpenAI._astream
        try:
            ChatOpenAI._astream = _fake_super_astream  # type: ignore[method-assign]
            await model.ainvoke(outbound, stream=True)
        finally:
            ChatOpenAI._astream = original_astream  # type: ignore[method-assign]

        sent = captured["messages"]
        ai = [m for m in sent if isinstance(m, AIMessage) and m.tool_calls][0]
        assert ai.tool_calls[0]["name"] == "threetears_web_search"
        assert outbound[0].tool_calls[0]["name"] == "threetears.web_search"


class TestOpenAIWrapperToolNameValidation:
    """The OpenAI wrapper drops ``invalid_tool_calls`` entries whose names
    fail the canonical 3tears tool-name regex.

    Parity with the OpenRouter / Anthropic wrappers (chunk 03): the OpenAI
    wrapper previously lacked the junk-name filter. Both the streaming and
    non-streaming paths now drop junk names before they reach downstream
    dispatch / persistence.
    """

    @pytest.mark.asyncio
    async def test_astream_drops_invalid_tool_calls_with_junk_names(self) -> None:
        """``astream`` drops ``invalid_tool_calls`` entries whose names fail
        the canonical regex."""
        junk_name = 'memory_recall" name="memory_recall'

        async def _fake_super_astream(
            self: Any,
            messages: Any,
            stop: Any = None,
            run_manager: Any = None,
            **kwargs: Any,
        ):
            del self, messages, stop, run_manager, kwargs
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    invalid_tool_calls=[
                        {
                            "name": junk_name,
                            "args": "{}",
                            "id": "call_junk",
                            "error": "JSONDecodeError",
                        },
                        {
                            "name": "threetears_calculator",
                            "args": "{partial",
                            "id": "call_ok",
                            "error": "JSONDecodeError",
                        },
                    ],
                ),
            )

        model = create_openai_chat("gpt-4o", "sk-test")

        original_astream = ChatOpenAI._astream
        try:
            ChatOpenAI._astream = _fake_super_astream  # type: ignore[method-assign]
            chunks: list[AIMessageChunk] = []
            async for chunk in model.astream("hi"):
                chunks.append(chunk)
        finally:
            ChatOpenAI._astream = original_astream  # type: ignore[method-assign]

        carrier_chunks = [c for c in chunks if c.invalid_tool_calls]
        assert len(carrier_chunks) == 1
        kept = carrier_chunks[0].invalid_tool_calls
        assert len(kept) == 1
        assert kept[0]["name"] == "threetears_calculator"
        assert all(call["name"] != junk_name for call in kept)

    @pytest.mark.asyncio
    async def test_agenerate_drops_invalid_tool_calls_with_junk_names(self) -> None:
        """``_agenerate`` mirrors the streaming-path filter for non-streaming calls."""
        from langchain_core.outputs import ChatGeneration, ChatResult

        junk_name = 'memory_recall" name="memory_recall'

        async def _fake_super_agenerate(
            self: Any,
            messages: Any,
            stop: Any = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> ChatResult:
            del self, messages, stop, run_manager, kwargs
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            invalid_tool_calls=[
                                {
                                    "name": junk_name,
                                    "args": "{}",
                                    "id": "call_junk",
                                    "error": "JSONDecodeError",
                                },
                                {
                                    "name": "threetears_calculator",
                                    "args": "{partial",
                                    "id": "call_ok",
                                    "error": "JSONDecodeError",
                                },
                            ],
                        ),
                    ),
                ],
            )

        model = create_openai_chat("gpt-4o", "sk-test")

        original_agenerate = ChatOpenAI._agenerate
        try:
            ChatOpenAI._agenerate = _fake_super_agenerate  # type: ignore[method-assign]
            result = await model._agenerate([AIMessage(content="hi")])
        finally:
            ChatOpenAI._agenerate = original_agenerate  # type: ignore[method-assign]

        kept = result.generations[0].message.invalid_tool_calls
        assert len(kept) == 1
        assert kept[0]["name"] == "threetears_calculator"
        assert all(call["name"] != junk_name for call in kept)
