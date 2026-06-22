"""tests for ``create_openrouter_chat`` factory and capability registration."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.tools import BaseTool

from threetears.models import DEFAULT_CHAT_MODEL
from threetears.models.capabilities import get_capabilities
from threetears.models.enums import ModelTier, ModelType
from threetears.models.providers.openrouter import (
    OPENROUTER_PROVIDER_NAME,
    create_openrouter_chat,
)
from threetears.models.tool_name_translation import (
    NameMangledToolProxy,
    build_name_translation,
    mangle_tool_name,
    reverse_translate_message,
)


class TestCreateOpenRouterChat:
    """tests for ``create_openrouter_chat`` factory."""

    def test_returns_base_chat_model(self) -> None:
        """factory returns a ``BaseChatModel`` subclass instance."""
        model = create_openrouter_chat("deepseek/deepseek-chat-v3-0324", "sk-test")
        assert isinstance(model, BaseChatModel)


class TestOpenRouterCapabilityRegistration:
    """tests that openrouter canonical models register at import time."""

    def test_deepseek_chat_registered(self) -> None:
        """``deepseek/deepseek-chat-v3-0324`` resolves to openrouter chat capabilities."""
        caps = get_capabilities("deepseek/deepseek-chat-v3-0324")
        assert caps is not None
        assert caps.provider_name == OPENROUTER_PROVIDER_NAME
        assert caps.model_type == ModelType.CHAT
        assert caps.model_tier == ModelTier.LARGE
        assert caps.requires_alternating_roles is True

    def test_deepseek_chat_cache_fields(self) -> None:
        """``deepseek/deepseek-chat-v3-0324`` carries auto-cache fields.

        DeepSeek's direct API runs automatic context caching and
        surfaces ``cached_tokens`` on the response. The ``deepseek/``
        slug routed through OpenRouter inherits the same behavior,
        so the capability record matches OpenAI's auto-cache shape
        (cache_control not supported, openai-auto-cache supported,
        no minimum, no TTL).
        """
        caps = get_capabilities("deepseek/deepseek-chat-v3-0324")
        assert caps is not None
        assert caps.supports_anthropic_cache_control is False
        assert caps.supports_openai_auto_cache is True
        assert caps.min_cacheable_tokens == 0
        assert caps.cache_ttl_seconds == 0

    def test_deepseek_r1_cache_fields(self) -> None:
        """``deepseek/deepseek-r1`` carries auto-cache fields."""
        caps = get_capabilities("deepseek/deepseek-r1")
        assert caps is not None
        assert caps.supports_anthropic_cache_control is False
        assert caps.supports_openai_auto_cache is True
        assert caps.min_cacheable_tokens == 0
        assert caps.cache_ttl_seconds == 0


# -- name translation --------------------------------------------------------


class _DottedTool(BaseTool):
    """Minimal :class:`BaseTool` whose ``.name`` carries a dot, for tests."""

    name: str = "threetears.calculator"
    description: str = "test calculator"
    invoked_with: list[dict[str, Any]] = []

    def _run(self, **kwargs: Any) -> str:
        self.invoked_with.append(dict(kwargs))
        return "ok"

    async def _arun(self, **kwargs: Any) -> str:
        self.invoked_with.append(dict(kwargs))
        return "ok"


class TestMangleToolName:
    """``mangle_tool_name`` produces wire-safe names."""

    def test_dot_replaced_with_underscore(self) -> None:
        assert mangle_tool_name("threetears.calculator") == "threetears_calculator"

    def test_nested_dots_all_replaced(self) -> None:
        assert mangle_tool_name("threetears.workspace.fs_read") == "threetears_workspace_fs_read"

    def test_no_dots_passes_through(self) -> None:
        assert mangle_tool_name("plain_name") == "plain_name"

    def test_existing_underscores_preserved(self) -> None:
        """Underscores in the source are preserved -- the round-trip
        relies on the reverse map (built per-bind) rather than on a
        symmetrical underscore<->dot inverse so ``threetears.web_search``
        and any future tool ``threetears_web_search`` (no dot) coexist
        unambiguously when one's wire form happens to collide with the
        other's canonical name.
        """
        assert mangle_tool_name("threetears.web_search") == "threetears_web_search"


class TestBuildNameTranslation:
    """``build_name_translation`` returns proxies + reverse map."""

    def test_dotted_tool_gets_proxy(self) -> None:
        tool = _DottedTool()
        wire_tools, reverse_map = build_name_translation([tool])
        assert len(wire_tools) == 1
        assert wire_tools[0] is not tool
        assert isinstance(wire_tools[0], NameMangledToolProxy)
        assert wire_tools[0].name == "threetears_calculator"
        assert reverse_map == {"threetears_calculator": "threetears.calculator"}

    def test_dotless_tool_passes_through(self) -> None:
        """Tools with no dot in their name pass through unchanged --
        the proxy is unnecessary and the reverse map stays empty for
        them, so the response un-translation no-ops on their tool calls.
        """

        class _Plain(BaseTool):
            name: str = "plain_tool"
            description: str = "no dots here"

            def _run(self, **kwargs: Any) -> str:
                return "ok"

            async def _arun(self, **kwargs: Any) -> str:
                return "ok"

        tool = _Plain()
        wire_tools, reverse_map = build_name_translation([tool])
        assert wire_tools == [tool]
        assert reverse_map == {}

    def test_proxy_preserves_description_and_args_schema(self) -> None:
        tool = _DottedTool()
        wire_tools, _ = build_name_translation([tool])
        proxy = wire_tools[0]
        assert proxy.description == tool.description
        assert proxy.args_schema is tool.args_schema

    @pytest.mark.asyncio
    async def test_proxy_arun_delegates_to_original(self) -> None:
        """Calling the proxy's ``_arun`` runs the dotted-named original."""
        tool = _DottedTool()
        wire_tools, _ = build_name_translation([tool])
        proxy = wire_tools[0]
        result = await proxy._arun(expression="1+1")
        assert result == "ok"
        assert tool.invoked_with == [{"expression": "1+1"}]


class TestNameTranslatingChatOpenRouter:
    """End-to-end name-translation via the ``ChatOpenRouter`` subclass."""

    def test_factory_returns_translating_subclass(self) -> None:
        """The factory builds the translating subclass, not vanilla
        ``ChatOpenRouter``. The class name carries ``Translating`` so a
        debugger / log line shows the active behaviour.
        """
        model = create_openrouter_chat("deepseek/deepseek-chat-v3-0324", "sk-test")
        assert "Translating" in type(model).__name__

    def test_bind_tools_populates_reverse_map(self) -> None:
        """``bind_tools`` mutates the instance's reverse map so a
        subsequent ``_astream`` / ``_agenerate`` can rewrite tool-call
        names.
        """
        model = create_openrouter_chat("deepseek/deepseek-chat-v3-0324", "sk-test")
        tool = _DottedTool()
        model.bind_tools([tool])
        # PrivateAttr: access via the standard pydantic shape
        reverse = model._name_reverse_map
        assert reverse == {"threetears_calculator": "threetears.calculator"}

    def test_reverse_translate_message_rewrites_tool_calls(self) -> None:
        """A finished ``AIMessage`` with underscored tool-call names
        gets its names rewritten back to the canonical dotted form.
        """
        model = create_openrouter_chat("deepseek/deepseek-chat-v3-0324", "sk-test")
        model.bind_tools([_DottedTool()])
        msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "threetears_calculator",
                    "args": {"expression": "2+2"},
                    "id": "call_1",
                },
            ],
        )
        reverse_translate_message(msg, model._name_reverse_map)
        assert msg.tool_calls[0]["name"] == "threetears.calculator"

    def test_reverse_translate_message_rewrites_tool_call_chunks(self) -> None:
        """Streaming chunks carry partial ``tool_call_chunks``; the name
        field arrives once at the start of each call. The reverse
        translation rewrites that first chunk so consumers accumulating
        tool calls see canonical names from the start.
        """
        model = create_openrouter_chat("deepseek/deepseek-chat-v3-0324", "sk-test")
        model.bind_tools([_DottedTool()])
        chunk = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "name": "threetears_calculator",
                    "args": "",
                    "id": "call_1",
                    "index": 0,
                },
            ],
        )
        reverse_translate_message(chunk, model._name_reverse_map)
        assert chunk.tool_call_chunks[0]["name"] == "threetears.calculator"

    def test_reverse_translate_message_rewrites_invalid_tool_calls(self) -> None:
        """Malformed streamed tool calls land in ``invalid_tool_calls``
        and metallm / aibots-agents both inspect them when ``tool_calls``
        is empty. Translate those names too so the recovery code sees
        canonical form.
        """
        model = create_openrouter_chat("deepseek/deepseek-chat-v3-0324", "sk-test")
        model.bind_tools([_DottedTool()])
        msg = AIMessage(
            content="",
            invalid_tool_calls=[
                {
                    "name": "threetears_calculator",
                    "args": "{partial",
                    "id": "call_1",
                    "error": "JSONDecodeError",
                },
            ],
        )
        reverse_translate_message(msg, model._name_reverse_map)
        assert msg.invalid_tool_calls[0]["name"] == "threetears.calculator"

    @pytest.mark.asyncio
    async def test_ainvoke_untranslates_when_aggregating_from_astream(self) -> None:
        """``ainvoke`` un-translates tool-call names even when it aggregates
        internally from the protected ``_astream``.

        Regression for the metallm converged-loop tool-dispatch failure
        (2026-06-22): the 3tears ``agent_node`` calls ``model.ainvoke`` while
        an outer ``astream_events`` tap is active, so a v1 streaming handler
        is attached and ``BaseChatModel.ainvoke`` routes through
        ``_agenerate_with_cache`` -> ``self._astream`` (NOT ``_agenerate``).
        That path bypassed BOTH the public ``astream`` override AND
        ``_agenerate``, leaking the underscored wire name
        ``threetears_calculator`` to the caller, which then missed the dotted
        dispatch map and the model flailed on tool names. The public
        ``ainvoke`` override post-processes the aggregated result, so names
        are canonical regardless of the internal route.

        ``stream=True`` makes ``_should_stream`` true (``chat_models.py``:
        ``if kwargs.get("stream"): return True``), forcing the ``_astream``
        aggregation path this test must exercise. Without the override the
        returned name stays ``threetears_calculator`` and this fails.
        """
        from langchain_core.outputs import ChatGenerationChunk
        from langchain_openrouter import ChatOpenRouter

        async def _fake_super_astream(
            self: Any,
            messages: Any,
            stop: Any = None,
            run_manager: Any = None,
            **kwargs: Any,
        ):
            del self, messages, stop, run_manager, kwargs
            # The wire form: the model called the tool by its mangled
            # (underscored) name; un-translation has NOT happened yet.
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

        model = create_openrouter_chat("deepseek/deepseek-chat-v3-0324", "sk-test")
        model.bind_tools([_DottedTool()])

        original_astream = ChatOpenRouter._astream
        try:
            ChatOpenRouter._astream = _fake_super_astream  # type: ignore[method-assign]
            result = await model.ainvoke("hi", stream=True)
        finally:
            ChatOpenRouter._astream = original_astream  # type: ignore[method-assign]

        # The aggregated message's tool call carries the canonical dotted
        # name, not the underscored wire form.
        assert result.tool_calls, "expected an aggregated tool call"
        assert result.tool_calls[0]["name"] == "threetears.calculator"

    @pytest.mark.asyncio
    async def test_astream_drops_invalid_tool_calls_with_junk_names(self) -> None:
        """``astream`` drops ``invalid_tool_calls`` entries whose names
        fail the canonical regex.

        Regression test for metallm prod incident on 2026-05-19 (conv
        ``019e3e26-9870-7a03-8f04-8cc6a4f5f418``): the model emitted a
        tool call whose ``function.name`` carried an embedded
        XML-attribute fragment (``memory_recall" name="memory_recall``).
        That value landed in ``invalid_tool_calls``, passed through
        the wrapper unfiltered, and reached metallm's dispatch layer
        where it was persisted as an unrecoverable invocation. The
        wrapper now drops those entries before yielding the chunk.
        """
        from langchain_core.outputs import ChatGenerationChunk
        from langchain_openrouter import ChatOpenRouter

        junk_name = 'memory_recall" name="memory_recall'

        async def _fake_super_astream(
            self: Any,
            messages: Any,
            stop: Any = None,
            run_manager: Any = None,
            **kwargs: Any,
        ):
            del self, messages, stop, run_manager, kwargs
            # Two invalid_tool_calls: one with a junk name, one
            # plausibly recoverable.
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

        model = create_openrouter_chat(
            "deepseek/deepseek-chat-v3-0324",
            "sk-test",
        )

        original_astream = ChatOpenRouter._astream
        try:
            ChatOpenRouter._astream = _fake_super_astream  # type: ignore[method-assign]
            chunks: list[AIMessageChunk] = []
            async for chunk in model.astream("hi"):
                chunks.append(chunk)
        finally:
            ChatOpenRouter._astream = original_astream  # type: ignore[method-assign]

        # The chunk carrying invalid_tool_calls should keep only the
        # well-named entry; the junk name must be dropped.
        carrier_chunks = [c for c in chunks if c.invalid_tool_calls]
        assert len(carrier_chunks) == 1
        kept = carrier_chunks[0].invalid_tool_calls
        assert len(kept) == 1
        assert kept[0]["name"] == "threetears_calculator"
        assert kept[0]["id"] == "call_ok"
        assert all(call["name"] != junk_name for call in kept)

    @pytest.mark.asyncio
    async def test_astream_keeps_nameless_streaming_continuation(self) -> None:
        """``astream`` keeps ``name=None`` invalid_tool_calls — they are not junk.

        Regression for metallm conv ``019ecdfd-0b17-7b40-b27b-6c4508f4ec3b``
        (2026-06-16): every DeepSeek tool turn logged dozens of
        ``dropped invalid_tool_calls entry with junk name: None`` WARNINGs.
        Those entries are normal streaming continuation fragments (only the
        first delta carries the name; the rest accumulate by index in
        ``tool_call_chunks`` and merge into a valid tool call). The wrapper
        must keep them — dropping was a false positive (and harmless to args,
        since the merge re-derives from ``tool_call_chunks``), but the
        per-chunk log storm was the real cost.
        """
        from langchain_core.outputs import ChatGenerationChunk
        from langchain_openrouter import ChatOpenRouter

        async def _fake_super_astream(
            self: Any,
            messages: Any,
            stop: Any = None,
            run_manager: Any = None,
            **kwargs: Any,
        ):
            del self, messages, stop, run_manager, kwargs
            # A streaming continuation fragment: no name, partial args.
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    invalid_tool_calls=[
                        {
                            "name": None,
                            "args": ' "2+2"}',
                            "id": None,
                            "error": "JSONDecodeError",
                        },
                    ],
                ),
            )

        model = create_openrouter_chat(
            "deepseek/deepseek-chat-v3-0324",
            "sk-test",
        )

        original_astream = ChatOpenRouter._astream
        try:
            ChatOpenRouter._astream = _fake_super_astream  # type: ignore[method-assign]
            chunks: list[AIMessageChunk] = []
            async for chunk in model.astream("hi"):
                chunks.append(chunk)
        finally:
            ChatOpenRouter._astream = original_astream  # type: ignore[method-assign]

        carrier_chunks = [c for c in chunks if c.invalid_tool_calls]
        assert len(carrier_chunks) == 1
        kept = carrier_chunks[0].invalid_tool_calls
        assert len(kept) == 1
        assert kept[0]["name"] is None

    @pytest.mark.asyncio
    async def test_agenerate_drops_invalid_tool_calls_with_junk_names(self) -> None:
        """``_agenerate`` mirrors the streaming-path filter for non-streaming calls.

        Same prod incident as the streaming test above; the
        non-streaming path (``ainvoke`` and friends) needs the same
        defense so consumers that don't stream are equally protected.
        """
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_openrouter import ChatOpenRouter

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

        model = create_openrouter_chat(
            "deepseek/deepseek-chat-v3-0324",
            "sk-test",
        )

        original_agenerate = ChatOpenRouter._agenerate
        try:
            ChatOpenRouter._agenerate = _fake_super_agenerate  # type: ignore[method-assign]
            result = await model.ainvoke("hi")
        finally:
            ChatOpenRouter._agenerate = original_agenerate  # type: ignore[method-assign]

        kept = result.invalid_tool_calls
        assert len(kept) == 1
        assert kept[0]["name"] == "threetears_calculator"
        assert all(call["name"] != junk_name for call in kept)

    def test_reverse_translate_message_noop_when_no_tools_bound(self) -> None:
        """Without any prior ``bind_tools`` call the reverse map is
        empty; ``_reverse_translate_message`` short-circuits without
        mutating the message.
        """
        model = create_openrouter_chat("deepseek/deepseek-chat-v3-0324", "sk-test")
        msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "external_tool",
                    "args": {},
                    "id": "call_1",
                },
            ],
        )
        reverse_translate_message(msg, model._name_reverse_map)
        # No bind_tools means no translation map; the name stays as-is.
        assert msg.tool_calls[0]["name"] == "external_tool"

    def test_reverse_translate_message_noop_for_unmatched_name(self) -> None:
        """A tool-call name not in the reverse map (e.g. from a tool
        that was already underscored, or an LLM hallucination) passes
        through unchanged.
        """
        model = create_openrouter_chat("deepseek/deepseek-chat-v3-0324", "sk-test")
        model.bind_tools([_DottedTool()])
        msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "some_other_tool",
                    "args": {},
                    "id": "call_1",
                },
            ],
        )
        reverse_translate_message(msg, model._name_reverse_map)
        assert msg.tool_calls[0]["name"] == "some_other_tool"

    def test_rebind_replaces_reverse_map(self) -> None:
        """A second ``bind_tools`` call with a different tool set
        replaces the reverse map wholesale. Otherwise stale entries
        from a prior bind would translate names that don't belong to
        the current bind.
        """

        class _OtherTool(_DottedTool):
            name: str = "threetears.web_search"

        model = create_openrouter_chat("deepseek/deepseek-chat-v3-0324", "sk-test")
        model.bind_tools([_DottedTool()])
        model.bind_tools([_OtherTool()])
        assert model._name_reverse_map == {
            "threetears_web_search": "threetears.web_search",
        }

    @pytest.mark.asyncio
    async def test_astream_translates_aimessage_chunk_tool_calls(self) -> None:
        """``astream`` must translate ``tool_call_chunks[i]["name"]`` on
        every yielded ``AIMessageChunk``.

        The wrapper used to override ``_astream`` and translate the
        nested ``ChatGenerationChunk.message``. That broke LangGraph's
        ``astream_events(version="v2")`` event tap -- 190 chunks would
        reach the consumer's ``async for`` loop but zero
        ``on_chat_model_stream`` callbacks would fire -- observed in
        metallm conv ``019e1f3d`` on 2026-05-13. Fixed by moving the
        translation off ``_astream`` and onto ``astream`` (the public
        Runnable method), so ``BaseChatModel.astream``'s callback wiring
        runs unchanged against the parent's untouched ``_astream``
        output and we post-process the AIMessageChunks after they're
        yielded. This test pins the new contract.
        """
        from langchain_core.outputs import ChatGenerationChunk
        from langchain_openrouter import ChatOpenRouter

        async def _fake_super_astream(
            self: Any,
            messages: Any,
            stop: Any = None,
            run_manager: Any = None,
            **kwargs: Any,
        ):
            """Stand in for ``ChatOpenRouter._astream`` -- yields a
            single ChatGenerationChunk whose nested AIMessageChunk
            carries the wire-form tool_call name the LLM emitted.

            Takes ``self`` because we patch it onto the class -- the
            method-binding semantics need the receiver slot.
            """
            del self, messages, stop, run_manager, kwargs
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": "threetears_calculator",
                            "args": "",
                            "id": "call_1",
                            "index": 0,
                        },
                    ],
                ),
            )

        model = create_openrouter_chat(
            "deepseek/deepseek-chat-v3-0324",
            "sk-test",
        )
        model.bind_tools([_DottedTool()])

        # Patch the parent ``_astream`` so the test does not need a
        # real OpenRouter HTTP call. Our wrapper inherits ``_astream``
        # from ChatOpenRouter; ``model.astream(...)`` flows through
        # BaseChatModel.astream -> ChatOpenRouter._astream -> back up
        # through our ``astream`` override which post-processes each
        # AIMessageChunk.
        original_astream = ChatOpenRouter._astream
        try:
            ChatOpenRouter._astream = _fake_super_astream  # type: ignore[method-assign]
            chunks: list[AIMessageChunk] = []
            async for chunk in model.astream("hi"):
                chunks.append(chunk)
        finally:
            ChatOpenRouter._astream = original_astream  # type: ignore[method-assign]

        # ``BaseChatModel.astream`` auto-yields a final empty chunk
        # with ``chunk_position="last"`` after the source iterator
        # completes (line ~942 of chat_models.py). Filter for the
        # tool-call-carrying chunk so the assertion stays robust to
        # that framework behavior.
        translated = [c for c in chunks if c.tool_call_chunks]
        assert len(translated) == 1
        assert translated[0].tool_call_chunks[0]["name"] == "threetears.calculator", (
            "astream did not translate "
            '``chunk.tool_call_chunks[i]["name"]`` on the yielded'
            " AIMessageChunk; reverse translation regressed."
        )

    @pytest.mark.asyncio
    async def test_astream_events_emits_on_chat_model_stream(self) -> None:
        """``astream_events(version="v2")`` must emit
        ``on_chat_model_stream`` events for every chunk the wrapper
        passes through.

        Regression test for metallm conv ``019e1f3d``: the previous
        ``_astream`` override silently dropped callback events even
        when the chunk iteration itself worked, leaving event-driven
        UIs (WS streaming) with the saved DB content but a blank live
        stream. Pinning this contract here means a future refactor
        that re-introduces an ``_astream`` override (or any other
        change that breaks the callback chain) fails CI loudly
        instead of shipping silently and surfacing only as a prod
        incident.
        """
        from langchain_core.outputs import ChatGenerationChunk
        from langchain_openrouter import ChatOpenRouter

        async def _fake_super_astream(
            self: Any,
            messages: Any,
            stop: Any = None,
            run_manager: Any = None,
            **kwargs: Any,
        ):
            del self, messages, stop, kwargs
            for text in ("hello ", "world", "!"):
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
                if run_manager is not None:
                    await run_manager.on_llm_new_token(token=text, chunk=chunk)
                yield chunk

        model = create_openrouter_chat(
            "deepseek/deepseek-chat-v3-0324",
            "sk-test",
        )

        original_astream = ChatOpenRouter._astream
        try:
            ChatOpenRouter._astream = _fake_super_astream  # type: ignore[method-assign]
            stream_event_count = 0
            collected_text = ""
            async for event in model.astream_events("hi", version="v2"):
                if event["event"] == "on_chat_model_stream":
                    stream_event_count += 1
                    collected_text += event["data"]["chunk"].content
        finally:
            ChatOpenRouter._astream = original_astream  # type: ignore[method-assign]

        # ``BaseChatModel.astream`` adds a final empty chunk with
        # ``chunk_position="last"`` after the source iterator finishes,
        # producing one extra ``on_chat_model_stream`` event. Three real
        # fake chunks → at least 3 events, and ``collected_text`` is
        # robust to that empty tail.
        assert stream_event_count >= 3, (
            f"Expected >=3 on_chat_model_stream events (one per fake"
            f" chunk plus the framework's final empty chunk); got"
            f" {stream_event_count}. The wrapper is breaking the"
            f" callback chain that drives astream_events(v2) — exactly"
            f" the metallm 2026-05-13 fingerprint (chunks delivered to"
            f" consumer, zero stream events emitted)."
        )
        assert collected_text == "hello world!", (
            "Stream events fired but their chunks did not carry the"
            f" parent's content as-yielded; got {collected_text!r}."
        )

    @pytest.mark.asyncio
    async def test_astream_events_survives_with_config_callbacks(self) -> None:
        """``with_config(callbacks=[...])`` must not strip the event_streamer.

        Production failure mode from 2026-05-13 (metallm conv
        ``019e2243-de0c``): the previous ``astream`` override took its
        ``config`` argument and forwarded it verbatim to
        ``super().astream(...)``. When the wrapper instance was wrapped
        again by ``model.with_config(callbacks=[UsageTracker,
        CircuitBreaker])`` (which ``threetears.models.factory.create_chat_model``
        does for every chat model), ``RunnableBinding._merge_configs``
        produced a config whose ``callbacks`` was a plain list of those
        bound handlers. Inside ``BaseChatModel.astream``,
        ``ensure_config(config)`` then performed a key-by-key
        ``dict.update`` that REPLACED the contextvar's
        ``AsyncCallbackManager`` (which carries
        ``astream_events``' event_streamer as an inheritable handler)
        with the bound list. The event_streamer disappeared for the
        duration of the model run, no ``on_chat_model_*`` events fired,
        and the live UI stream stayed blank while the saved DB message
        was complete -- the exact ``saved_content_length > 0`` /
        ``tokens_dispatched_count == 0`` fingerprint metallm hit.

        The previous regression test (above) only exercises the bare
        wrapper instance, so it passed CI while production was broken.
        This test mirrors what ``create_chat_model`` actually does --
        wrap the model with ``with_config(callbacks=[...])`` -- so the
        contextvar-vs-bound-callbacks merge path is on the test surface.
        """
        from langchain_core.callbacks import AsyncCallbackHandler
        from langchain_core.outputs import ChatGenerationChunk
        from langchain_openrouter import ChatOpenRouter

        async def _fake_super_astream(
            self: Any,
            messages: Any,
            stop: Any = None,
            run_manager: Any = None,
            **kwargs: Any,
        ):
            del self, messages, stop, kwargs
            for text in ("hello ", "world", "!"):
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
                if run_manager is not None:
                    await run_manager.on_llm_new_token(token=text, chunk=chunk)
                yield chunk

        class _RecordingCallback(AsyncCallbackHandler):
            """Stand in for ``UsageTrackingCallback`` /
            ``CircuitBreakerCallback`` -- the bound, list-form callbacks
            ``create_chat_model`` attaches via ``with_config``."""

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
        model = create_openrouter_chat(
            "deepseek/deepseek-chat-v3-0324",
            "sk-test",
        )
        # Mirror ``threetears.models.factory.create_chat_model``'s
        # ``model.with_config(callbacks=[...])`` step.
        bound_model = model.with_config(callbacks=[bound_cb])

        original_astream = ChatOpenRouter._astream
        try:
            ChatOpenRouter._astream = _fake_super_astream  # type: ignore[method-assign]
            stream_event_count = 0
            async for event in bound_model.astream_events("hi", version="v2"):
                if event["event"] == "on_chat_model_stream":
                    stream_event_count += 1
        finally:
            ChatOpenRouter._astream = original_astream  # type: ignore[method-assign]

        # Event_streamer must still see chat-model-stream events even
        # though the bound list of callbacks is also present.
        assert stream_event_count >= 3, (
            "with_config-bound list callbacks REPLACED the contextvar's"
            " event_streamer manager — `on_chat_model_stream` events"
            f" never reached astream_events. Got {stream_event_count}"
            " events (need >=3 from three fake chunks plus framework's"
            " trailing empty chunk). This is the metallm 2026-05-13"
            " production fingerprint — fix the wrapper's `astream`"
            " override (do not forward `config` verbatim; pre-merge it"
            " with the contextvar via merge_configs)."
        )
        # And the bound callbacks must STILL fire — the fix can't"
        # silently drop UsageTracker / CircuitBreaker either.
        assert bound_cb.start_seen >= 1, (
            "Bound callback's on_chat_model_start never fired —"
            " the fix dropped the with_config list when preserving the"
            " contextvar manager. Both must propagate."
        )
        assert bound_cb.token_seen >= 3, (
            "Bound callback's on_llm_new_token fired"
            f" {bound_cb.token_seen} times — expected >=3 (one per"
            " fake chunk). The fix silently dropped the list of bound"
            " handlers somewhere in the merge."
        )


class TestVanillaChatAnthropicBaseline:
    """Baseline confirming vanilla ``ChatAnthropic`` -- with no wrapper
    in the inheritance chain -- emits ``on_chat_model_stream`` events
    correctly. Comparison case for the wrapper tests above. If THIS
    test fails the bug isn't in our wrapper; it's in the LangChain
    framework or our monkey-patch shape. If this passes and the
    wrapper test fails, the wrapper is the culprit.

    Anthropic-direct is the second provider Pace asked us to verify on
    2026-05-13 before cutting the 3tears bump that ships the wrapper
    fix.
    """

    @pytest.mark.asyncio
    async def test_anthropic_direct_emits_on_chat_model_stream(self) -> None:
        """Vanilla ``ChatAnthropic.astream_events(v2)`` emits one
        ``on_chat_model_stream`` per yielded chunk (plus the framework's
        final empty chunk). No wrapper subclassing involved.
        """
        from langchain_anthropic import ChatAnthropic
        from langchain_core.outputs import ChatGenerationChunk

        async def _fake_super_astream(
            self: Any,
            messages: Any,
            stop: Any = None,
            run_manager: Any = None,
            **kwargs: Any,
        ):
            del self, messages, stop, kwargs
            for text in ("anthropic ", "direct ", "streams"):
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
                if run_manager is not None:
                    await run_manager.on_llm_new_token(token=text, chunk=chunk)
                yield chunk

        # ChatAnthropic requires an api_key to construct; the fake
        # ``_astream`` short-circuits before any HTTP call.
        model = ChatAnthropic(
            model=DEFAULT_CHAT_MODEL,  # type: ignore[call-arg]
            anthropic_api_key="sk-test",  # type: ignore[arg-type]
        )

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

        assert stream_event_count >= 3, (
            f"Vanilla ChatAnthropic.astream_events(v2) did not emit a"
            f" stream event per chunk (got {stream_event_count}). This"
            f" baseline failing means the framework is broken or the"
            f" monkey-patch shape is wrong -- look there before"
            f" suspecting the OpenRouter wrapper."
        )
        assert collected_text == "anthropic direct streams"
