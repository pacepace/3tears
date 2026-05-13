"""tests for ``create_openrouter_chat`` factory and capability registration."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.tools import BaseTool

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
            model="claude-sonnet-4-5-20250929",  # type: ignore[call-arg]
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
