"""tests for :func:`create_anthropic_chat` factory and capability registration."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGenerationChunk

from threetears.models import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_FAST_MODEL,
    DEFAULT_LARGE_MODEL,
)
from threetears.models.capabilities import get_capabilities
from threetears.models.enums import ModelTier, ModelType
from threetears.models.providers.anthropic import (
    ANTHROPIC_PROVIDER_NAME,
    create_anthropic_chat,
    strip_v1_suffix,
)
from threetears.models.providers._claude_cli import is_subscription_token


class TestSubscriptionRouting:
    """The anthropic provider routes an OAuth subscription token to the CLI backend; an API key
    keeps the HTTP-API ``ChatAnthropic`` path — same model ids, no separate provider. The token is
    just another provider credential (stored like any API key); routing is by its value, not config."""

    def test_is_subscription_token_distinguishes_oauth_from_api_key(self) -> None:
        assert is_subscription_token("sk-ant-oat01-abc123")  # claude setup-token
        assert not is_subscription_token("sk-ant-api03-abc123")  # an API key
        assert not is_subscription_token("sk-test")

    def test_api_key_builds_chatanthropic(self) -> None:
        model = create_anthropic_chat(DEFAULT_CHAT_MODEL, "sk-ant-api03-xyz")
        assert isinstance(model, ChatAnthropic)

    def test_oauth_token_routes_to_the_subscription_backend(self) -> None:
        # Needs the optional extra (langchain-claude-code → claude-agent-sdk); skip if absent.
        pytest.importorskip("langchain_claude_code")
        pytest.importorskip("claude_agent_sdk")
        token = "sk-ant-oat01-faketokenfortest"
        model = create_anthropic_chat(DEFAULT_CHAT_MODEL, token)
        # The subscription backend is NOT a ChatAnthropic — it's the CLI/Agent-SDK-backed model.
        assert not isinstance(model, ChatAnthropic)
        assert isinstance(model, BaseChatModel)
        # The token is carried per-instance (→ per-subprocess env), never the process-global env.
        assert model.oauth_token == token  # type: ignore[attr-defined]
        import os

        assert os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") != token  # no global mutation


class TestCreateAnthropicChat:
    """tests for the ``create_anthropic_chat`` factory function."""

    def test_returns_base_chat_model(self) -> None:
        """factory returns a ``BaseChatModel`` subclass instance."""
        model = create_anthropic_chat(DEFAULT_CHAT_MODEL, "sk-test")
        assert isinstance(model, BaseChatModel)
        assert isinstance(model, ChatAnthropic)

    def test_model_name_propagated(self) -> None:
        """factory forwards model_name to ``ChatAnthropic``."""
        model = create_anthropic_chat(DEFAULT_FAST_MODEL, "sk-test")
        assert model.model == DEFAULT_FAST_MODEL

    def test_strips_v1_suffix_from_base_url(self) -> None:
        """trailing ``/v1`` is stripped from base_url before instantiation."""
        model = create_anthropic_chat(
            DEFAULT_CHAT_MODEL,
            "sk-test",
            base_url="https://api.anthropic.com/v1",
        )
        assert model.anthropic_api_url == "https://api.anthropic.com"

    def test_strips_v1_slash_suffix(self) -> None:
        """trailing ``/v1/`` is also stripped."""
        model = create_anthropic_chat(
            DEFAULT_CHAT_MODEL,
            "sk-test",
            base_url="https://api.anthropic.com/v1/",
        )
        assert model.anthropic_api_url == "https://api.anthropic.com"

    def test_no_base_url_means_default(self) -> None:
        """omitting base_url leaves ``ChatAnthropic`` to apply its default."""
        model = create_anthropic_chat(DEFAULT_CHAT_MODEL, "sk-test")
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
        """``claude-sonnet-4-6`` resolves to anthropic chat capabilities."""
        caps = get_capabilities(DEFAULT_CHAT_MODEL)
        assert caps is not None
        assert caps.provider_name == ANTHROPIC_PROVIDER_NAME
        assert caps.model_type == ModelType.CHAT
        assert caps.model_tier == ModelTier.LARGE
        assert caps.supports_tools is True

    def test_haiku_registered(self) -> None:
        """``claude-haiku-4-5-20251001`` resolves to anthropic small-tier chat."""
        caps = get_capabilities(DEFAULT_FAST_MODEL)
        assert caps is not None
        assert caps.provider_name == ANTHROPIC_PROVIDER_NAME
        assert caps.model_tier == ModelTier.SMALL

    def test_sonnet_cache_fields(self) -> None:
        """``claude-sonnet-4-6`` carries anthropic-shape cache fields.

        Every Anthropic chat-model registration declares the anthropic
        cache shape (cache_control supported, 1024-token minimum,
        300-second ephemeral TTL) so consumers that resolve through
        ``get_capabilities`` get the right caching record without
        having to maintain a parallel per-provider table.
        """
        caps = get_capabilities(DEFAULT_CHAT_MODEL)
        assert caps is not None
        assert caps.supports_anthropic_cache_control is True
        assert caps.supports_openai_auto_cache is False
        assert caps.min_cacheable_tokens == 1024
        assert caps.cache_ttl_seconds == 300

    def test_opus_cache_fields(self) -> None:
        """``claude-opus-4-8`` carries anthropic-shape cache fields."""
        caps = get_capabilities(DEFAULT_LARGE_MODEL)
        assert caps is not None
        assert caps.supports_anthropic_cache_control is True
        assert caps.supports_openai_auto_cache is False
        assert caps.min_cacheable_tokens == 1024
        assert caps.cache_ttl_seconds == 300

    def test_haiku_cache_fields(self) -> None:
        """``claude-haiku-4-5-20251001`` carries anthropic-shape cache fields."""
        caps = get_capabilities(DEFAULT_FAST_MODEL)
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
    UIs (the consumer's WS handler, debug-inject endpoint) with the saved DB
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

        model = create_anthropic_chat(DEFAULT_CHAT_MODEL, "sk-test")

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
    async def test_ainvoke_preserves_streaming_callbacks(self) -> None:
        """The ``ainvoke`` override must NOT break token streaming.

        The converged tool loop runs ``model.ainvoke`` under an outer
        ``astream_events`` tap, so ``ainvoke`` aggregates from the protected
        ``_astream`` and fires ``on_llm_new_token``. This pins that the public
        ``ainvoke`` override (+ its ``merge_configs`` callback-preservation)
        still delivers streamed tokens to a callback handler — i.e. the
        un-translation post-processing did not swallow the streaming path.
        """
        from langchain_anthropic import ChatAnthropic
        from langchain_core.callbacks import AsyncCallbackHandler

        tokens: list[str] = []

        class _Recorder(AsyncCallbackHandler):
            async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
                tokens.append(token)

        async def _fake_super_astream(
            self: Any,
            messages: Any,
            stop: Any = None,
            run_manager: Any = None,
            **kwargs: Any,
        ):
            del self, messages, stop, kwargs
            for text in ("a", "b", "c"):
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
                if run_manager is not None:
                    await run_manager.on_llm_new_token(token=text, chunk=chunk)
                yield chunk

        model = create_anthropic_chat(DEFAULT_CHAT_MODEL, "sk-test")

        original_astream = ChatAnthropic._astream
        try:
            ChatAnthropic._astream = _fake_super_astream  # type: ignore[method-assign]
            # stream=True forces the _astream aggregation path; the callback
            # handler rides config["callbacks"], which the override's
            # merge_configs must preserve.
            result = await model.ainvoke("hi", stream=True, config={"callbacks": [_Recorder()]})
        finally:
            ChatAnthropic._astream = original_astream  # type: ignore[method-assign]

        # Tokens streamed to the handler (a trailing framework empty chunk is
        # normal); the override did not swallow the streaming path.
        assert "".join(tokens) == "abc"
        assert result.content == "abc"

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
        OpenRouter case is the one reproduced in production
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
        model = create_anthropic_chat(DEFAULT_CHAT_MODEL, "sk-test")
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


class TestAnthropicWrapperToolNameValidation:
    """Wrapper-level defense against junk ``invalid_tool_calls`` names.

    Mirrors the OpenRouter wrapper's validation hook. Both wrappers
    drop ``invalid_tool_calls`` entries whose names fail the canonical
    3tears tool-name regex before yielding the chunk / returning the
    result. This blocks the 2026-05-19 prod incident
    (conv ``019e3e26-9870-7a03-8f04-8cc6a4f5f418``) from recurring
    on either provider.
    """

    @pytest.mark.asyncio
    async def test_astream_drops_invalid_tool_calls_with_junk_names(self) -> None:
        """``astream`` drops ``invalid_tool_calls`` entries whose names
        fail the canonical regex.
        """
        from langchain_core.messages import AIMessage

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

        model = create_anthropic_chat(DEFAULT_CHAT_MODEL, "sk-test")

        original_astream = ChatAnthropic._astream
        try:
            ChatAnthropic._astream = _fake_super_astream  # type: ignore[method-assign]
            chunks: list[AIMessageChunk] = []
            async for chunk in model.astream("hi"):
                chunks.append(chunk)
        finally:
            ChatAnthropic._astream = original_astream  # type: ignore[method-assign]

        # Confirm the AIMessage shape so the typing import is used.
        _ = AIMessage(content="placeholder")
        carrier_chunks = [c for c in chunks if c.invalid_tool_calls]
        assert len(carrier_chunks) == 1
        kept = carrier_chunks[0].invalid_tool_calls
        assert len(kept) == 1
        assert kept[0]["name"] == "threetears_calculator"
        assert all(call["name"] != junk_name for call in kept)

    @pytest.mark.asyncio
    async def test_agenerate_drops_invalid_tool_calls_with_junk_names(self) -> None:
        """``_agenerate`` mirrors the streaming-path filter for
        non-streaming calls.
        """
        from langchain_core.messages import AIMessage
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

        model = create_anthropic_chat(DEFAULT_CHAT_MODEL, "sk-test")

        original_agenerate = ChatAnthropic._agenerate
        try:
            ChatAnthropic._agenerate = _fake_super_agenerate  # type: ignore[method-assign]
            result = await model.ainvoke("hi")
        finally:
            ChatAnthropic._agenerate = original_agenerate  # type: ignore[method-assign]

        kept = result.invalid_tool_calls
        assert len(kept) == 1
        assert kept[0]["name"] == "threetears_calculator"
        assert all(call["name"] != junk_name for call in kept)

    @pytest.mark.asyncio
    async def test_ainvoke_untranslates_when_aggregating_from_astream(self) -> None:
        """``ainvoke`` un-translates tool-call names even when it aggregates
        internally from the protected ``_astream``.

        Regression for the converged-loop tool-dispatch failure
        (2026-06-22): with a streaming callback active (the converged
        ``agent_node`` runs ``model.ainvoke`` under an ``astream_events``
        tap), ``BaseChatModel.ainvoke`` routes through
        ``_agenerate_with_cache`` -> ``self._astream`` instead of calling
        ``_agenerate``, bypassing BOTH the public ``astream`` override AND
        ``_agenerate``. The underscored wire name ``threetears_calculator``
        leaked to the caller and missed the dotted dispatch map. The public
        ``ainvoke`` override post-processes the aggregated result.

        ``stream=True`` makes ``_should_stream`` true, forcing the
        ``_astream`` aggregation path. Without the override the returned name
        stays ``threetears_calculator`` and this fails.
        """
        from langchain_core.tools import BaseTool

        class _DottedTool(BaseTool):
            name: str = "threetears.calculator"
            description: str = "test calculator"

            def _run(self, **kwargs: Any) -> str:
                return "ok"

            async def _arun(self, **kwargs: Any) -> str:
                return "ok"

        async def _fake_super_astream(
            self: Any,
            messages: Any,
            stop: Any = None,
            run_manager: Any = None,
            **kwargs: Any,
        ):
            del self, messages, stop, run_manager, kwargs
            # The wire form: the tool was called by its mangled (underscored)
            # name; un-translation has not happened yet.
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": "threetears_calculator",
                            "args": "{}",
                            "id": "call_1",
                            "index": 0,
                        },
                    ],
                ),
            )

        model = create_anthropic_chat(DEFAULT_CHAT_MODEL, "sk-test")
        model.bind_tools([_DottedTool()])

        original_astream = ChatAnthropic._astream
        try:
            ChatAnthropic._astream = _fake_super_astream  # type: ignore[method-assign]
            result = await model.ainvoke("hi", stream=True)
        finally:
            ChatAnthropic._astream = original_astream  # type: ignore[method-assign]

        assert result.tool_calls, "expected an aggregated tool call"
        assert result.tool_calls[0]["name"] == "threetears.calculator"


class TestAnthropicForwardTranslation:
    """The Anthropic wrapper forward-translates dotted tool-call names on the
    OUTBOUND ``messages`` before the provider call.

    The Anthropic Messages API validates tool names against
    ``^[a-zA-Z0-9_-]{1,128}$`` and rejects the dot, so a prior round's
    ``AIMessage`` carrying a canonical dotted ``tool_calls`` name would 400
    the turn when re-sent. Parity with the OpenRouter / OpenAI forward
    translation.
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

        model = create_anthropic_chat(DEFAULT_CHAT_MODEL, "sk-test")
        outbound = [
            SystemMessage(content="sys"),
            AIMessage(
                content="",
                tool_calls=[{"name": "threetears.web_search", "args": {"q": "x"}, "id": "c1"}],
            ),
            ToolMessage(content="hit", tool_call_id="c1"),
        ]

        original_astream = ChatAnthropic._astream
        try:
            ChatAnthropic._astream = _fake_super_astream  # type: ignore[method-assign]
            async for _ in model.astream(outbound):
                pass
        finally:
            ChatAnthropic._astream = original_astream  # type: ignore[method-assign]

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

        model = create_anthropic_chat(DEFAULT_CHAT_MODEL, "sk-test")
        outbound = [
            AIMessage(
                content="",
                tool_calls=[{"name": "threetears.web_search", "args": {"q": "x"}, "id": "c1"}],
            ),
        ]

        original = ChatAnthropic._agenerate
        try:
            ChatAnthropic._agenerate = _fake_super_agenerate  # type: ignore[method-assign]
            await model._agenerate(outbound)
        finally:
            ChatAnthropic._agenerate = original  # type: ignore[method-assign]

        sent = captured["messages"]
        assert sent[0].tool_calls[0]["name"] == "threetears_web_search"
        assert outbound[0].tool_calls[0]["name"] == "threetears.web_search"
