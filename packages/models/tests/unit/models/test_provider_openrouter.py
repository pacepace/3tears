"""tests for ``create_openrouter_chat`` factory and capability registration."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
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
    forward_translate_input,
    forward_translate_message,
    mangle_tool_name,
    reverse_translate_message,
)

from .provider_wire import (
    AnthropicMessagesWire,
    ChatCompletionsWire,
    TextBlock,
    openrouter_model,
    text_deltas,
    tool_call_delta,
)
from .translation_helpers import DottedTool as _DottedTool

#: the XML-attribute-leak tool name from the 2026-05-19 prod incident
_JUNK_NAME = 'memory_recall" name="memory_recall'


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

    def test_deepseek_v4_flash_registered(self) -> None:
        """``deepseek/deepseek-v4-flash`` resolves to openrouter chat capabilities.

        Unlike the two entries above it is ``MEDIUM``, not ``LARGE`` -- it is the
        cheap/fast sibling (roughly 3x cheaper input and 6x cheaper output per
        token than deepseek-chat-v3-0324) despite a much larger context window.
        """
        caps = get_capabilities("deepseek/deepseek-v4-flash")
        assert caps is not None
        assert caps.provider_name == OPENROUTER_PROVIDER_NAME
        assert caps.model_type == ModelType.CHAT
        assert caps.model_tier == ModelTier.MEDIUM
        assert caps.requires_alternating_roles is True

    def test_deepseek_v4_flash_cache_fields(self) -> None:
        """``deepseek/deepseek-v4-flash`` carries auto-cache fields.

        These four fields are the reason the entry exists: a model absent from
        the registry falls back to ``_NO_CACHE_CAPABILITIES`` in
        :mod:`threetears.langgraph.caching`, which reports both cache flags
        ``False`` -- so prompt caching silently does not happen. Registering the
        model is what makes the openai-auto-cache path apply to it.
        """
        caps = get_capabilities("deepseek/deepseek-v4-flash")
        assert caps is not None
        assert caps.supports_anthropic_cache_control is False
        assert caps.supports_openai_auto_cache is True
        assert caps.min_cacheable_tokens == 0
        assert caps.cache_ttl_seconds == 0

    def test_deepseek_v4_flash_context_and_modality(self) -> None:
        """``deepseek/deepseek-v4-flash`` declares its 1M context and text-only input.

        Both were confirmed against OpenRouter's live models endpoint when the
        entry was added; the sibling entries understate their own context windows,
        so this pins the accurate value against future copy-paste drift.
        """
        caps = get_capabilities("deepseek/deepseek-v4-flash")
        assert caps is not None
        assert caps.context_window == 1_048_576
        assert caps.supports_vision is False
        assert caps.supports_streaming is True
        assert caps.supports_tools is True


# -- name translation --------------------------------------------------------


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
        """Invoking the proxy runs the dotted-named original.

        Via ``ainvoke`` (the real calling convention), not ``_arun`` directly -- ``_arun`` is an
        internal protocol method LangChain's own ``arun``/``ainvoke`` machinery calls with a
        ``config`` it supplies; calling it directly bypasses that and is not how any real caller
        (production or test) invokes a tool. See ``tool_name_translation``'s "config propagation
        bug" docstring section and ``test_tool_name_translation.py``'s
        ``TestConfigPropagationToDelegate`` for the live bug this shape used to mask.
        """
        tool = _DottedTool()
        wire_tools, _ = build_name_translation([tool])
        proxy = wire_tools[0]
        result = await proxy.ainvoke({"expression": "1+1"})
        assert result == "ok"
        assert tool.invoked_with == [{"expression": "1+1"}]


def _answer_calling(*names: str, arguments: str = '{"expression": "2+2"}') -> ChatCompletionsWire:
    """a wire whose answer calls each of ``names`` by that exact wire name, streamed or not.

    OpenRouter's default deadline makes a plain ``ainvoke`` collect the stream, so the answer is
    scripted both ways and reads the same whichever the request asks for.

    :param names: the tool names the answer calls
    :ptype names: str
    :param arguments: the raw arguments every call carries
    :ptype arguments: str
    :return: the wire
    :rtype: ChatCompletionsWire
    """
    calls = [{"index": i, "id": f"call_{i}", "name": name, "arguments": arguments} for i, name in enumerate(names)]
    return ChatCompletionsWire(
        deltas=[tool_call_delta(*calls)],
        message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": arguments}}
                for call in calls
            ],
        },
        finish="tool_calls",
    )


class TestNameTranslatingChatOpenRouter:
    """End-to-end name-translation via the ``ChatOpenRouter`` subclass.

    Every answer comes from a scripted chat-completions API through the real
    ``openrouter`` SDK, so a name is asserted as the caller receives it.
    """

    def test_factory_returns_translating_subclass(self) -> None:
        """The factory builds the translating subclass, not vanilla
        ``ChatOpenRouter``. The class name carries ``Translating`` so a
        debugger / log line shows the active behaviour.
        """
        model = create_openrouter_chat("deepseek/deepseek-chat-v3-0324", "sk-test")
        assert "Translating" in type(model).__name__

    @pytest.mark.asyncio
    async def test_bind_tools_teaches_the_model_its_wire_names(self) -> None:
        """``bind_tools`` records each dotted tool's wire name on the model, so
        an answer calling ``threetears_calculator`` reaches the caller as the
        canonical ``threetears.calculator`` -- and the tool went out under its
        wire name.
        """
        wire = _answer_calling("threetears_calculator")
        model = openrouter_model(wire)

        result = await model.bind_tools([_DottedTool()]).ainvoke("what is 2+2?")

        assert [tool["function"]["name"] for tool in wire.bodies[-1]["tools"]] == ["threetears_calculator"]
        assert result.tool_calls[0]["name"] == "threetears.calculator"

    @pytest.mark.asyncio
    async def test_a_finished_answers_tool_calls_are_untranslated(self) -> None:
        """A finished ``AIMessage`` with underscored tool-call names
        gets its names rewritten back to the canonical dotted form.
        """
        model = openrouter_model(_answer_calling("threetears_calculator"))
        model.bind_tools([_DottedTool()])

        result = await model.ainvoke("what is 2+2?")

        assert result.tool_calls == [
            {
                "name": "threetears.calculator",
                "args": {"expression": "2+2"},
                "id": "call_0",
                "type": "tool_call",
            }
        ]

    @pytest.mark.asyncio
    async def test_a_streamed_answers_tool_call_chunks_are_untranslated(self) -> None:
        """Streaming chunks carry partial ``tool_call_chunks``; the name
        field arrives once at the start of each call. The reverse
        translation rewrites that first chunk so consumers accumulating
        tool calls see canonical names from the start.
        """
        wire = ChatCompletionsWire(
            deltas=[
                tool_call_delta({"index": 0, "id": "call_1", "name": "threetears_calculator", "arguments": ""}),
                tool_call_delta({"index": 0, "id": None, "name": None, "arguments": '{"expression": "2+2"}'}),
            ],
            finish="tool_calls",
        )
        model = openrouter_model(wire)
        model.bind_tools([_DottedTool()])

        named = [
            call_chunk["name"]
            async for chunk in model.astream("what is 2+2?")
            for call_chunk in chunk.tool_call_chunks
            if call_chunk["name"]
        ]

        assert named == ["threetears.calculator"]

    @pytest.mark.asyncio
    async def test_a_malformed_calls_invalid_tool_calls_are_untranslated(self) -> None:
        """Malformed tool calls land in ``invalid_tool_calls``
        and the consumer / 3tears-agents both inspect them when ``tool_calls``
        is empty. Translate those names too so the recovery code sees
        canonical form.
        """
        # arguments no JSON parser can repair (a truncated ``{partial`` is repaired to ``{}``
        # when the answer is streamed, and so is not malformed at all)
        model = openrouter_model(_answer_calling("threetears_calculator", arguments="not json"))
        model.bind_tools([_DottedTool()])

        result = await model.ainvoke("what is 2+2?")

        assert result.tool_calls == []
        assert [call["name"] for call in result.invalid_tool_calls] == ["threetears.calculator"]

    @pytest.mark.asyncio
    async def test_ainvoke_untranslates_when_aggregating_from_astream(self) -> None:
        """``ainvoke`` un-translates tool-call names even when it aggregates
        internally from the provider's stream.

        Regression for the converged-loop tool-dispatch failure
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
        ``if kwargs.get("stream"): return True``), forcing the stream
        aggregation path this test must exercise. Without the override the
        returned name stays ``threetears_calculator`` and this fails.
        """
        # The wire form: the model called the tool by its mangled
        # (underscored) name; un-translation has NOT happened yet.
        wire = ChatCompletionsWire(
            deltas=[tool_call_delta({"index": 0, "id": "call_1", "name": "threetears_calculator", "arguments": "{}"})],
            finish="tool_calls",
        )
        model = openrouter_model(wire)
        model.bind_tools([_DottedTool()])

        result = await model.ainvoke("hi", stream=True)

        assert wire.bodies[-1]["stream"] is True
        # The aggregated message's tool call carries the canonical dotted
        # name, not the underscored wire form.
        assert result.tool_calls, "expected an aggregated tool call"
        assert result.tool_calls[0]["name"] == "threetears.calculator"

    @pytest.mark.asyncio
    async def test_agenerate_untranslates_tool_names(self) -> None:
        """``agenerate`` (the batch chokepoint behind ``ainvoke``/``abatch``)
        un-translates tool-call names on every generated message — covering a
        direct ``agenerate`` caller that would otherwise see the wire form.
        """
        from langchain_core.outputs import ChatGeneration, LLMResult
        from langchain_openrouter import ChatOpenRouter

        async def _fake_super_agenerate(self: Any, messages: Any, *args: Any, **kwargs: Any) -> LLMResult:
            del self, messages, args, kwargs
            msg = AIMessage(
                content="",
                tool_calls=[
                    {"name": "threetears_calculator", "args": {}, "id": "c1"},
                ],
            )
            # agenerate returns LLMResult with NESTED generations (list[list]).
            return LLMResult(generations=[[ChatGeneration(message=msg)]])

        model = create_openrouter_chat("deepseek/deepseek-chat-v3-0324", "sk-test")
        model.bind_tools([_DottedTool()])

        original = ChatOpenRouter.agenerate
        try:
            ChatOpenRouter.agenerate = _fake_super_agenerate  # type: ignore[method-assign]
            result = await model.agenerate([[AIMessage(content="hi")]])
        finally:
            ChatOpenRouter.agenerate = original  # type: ignore[method-assign]

        name = result.generations[0][0].message.tool_calls[0]["name"]
        assert name == "threetears.calculator"

    @pytest.mark.asyncio
    async def test_astream_drops_invalid_tool_calls_with_junk_names(self) -> None:
        """``astream`` drops ``invalid_tool_calls`` entries whose names
        fail the canonical regex.

        Regression test for prod incident on 2026-05-19 (conv
        ``019e3e26-9870-7a03-8f04-8cc6a4f5f418``): the model emitted a
        tool call whose ``function.name`` carried an embedded
        XML-attribute fragment (``memory_recall" name="memory_recall``).
        That value landed in ``invalid_tool_calls``, passed through
        the wrapper unfiltered, and reached the consumer's dispatch layer
        where it was persisted as an unrecoverable invocation. The
        wrapper now drops those entries before yielding the chunk.
        """
        # Two malformed calls in one delta: one with a junk name, one
        # plausibly recoverable.
        wire = ChatCompletionsWire(
            deltas=[
                tool_call_delta(
                    {"index": 0, "id": "call_junk", "name": _JUNK_NAME, "arguments": "not json"},
                    {"index": 1, "id": "call_ok", "name": "threetears_calculator", "arguments": "not json"},
                )
            ],
            finish="tool_calls",
        )

        chunks: list[AIMessageChunk] = [chunk async for chunk in openrouter_model(wire).astream("hi")]

        # The chunk carrying invalid_tool_calls should keep only the
        # well-named entry; the junk name must be dropped.
        carrier_chunks = [c for c in chunks if c.invalid_tool_calls]
        assert len(carrier_chunks) == 1
        kept = carrier_chunks[0].invalid_tool_calls
        assert len(kept) == 1
        assert kept[0]["name"] == "threetears_calculator"
        assert kept[0]["id"] == "call_ok"
        assert all(call["name"] != _JUNK_NAME for call in kept)

    @pytest.mark.asyncio
    async def test_astream_keeps_nameless_streaming_continuation(self) -> None:
        """``astream`` keeps ``name=None`` invalid_tool_calls — they are not junk.

        Regression for prod conv ``019ecdfd-0b17-7b40-b27b-6c4508f4ec3b``
        (2026-06-16): every DeepSeek tool turn logged dozens of
        ``dropped invalid_tool_calls entry with junk name: None`` WARNINGs.
        Those entries are normal streaming continuation fragments (only the
        first delta carries the name; the rest accumulate by index in
        ``tool_call_chunks`` and merge into a valid tool call). The wrapper
        must keep them — dropping was a false positive (and harmless to args,
        since the merge re-derives from ``tool_call_chunks``), but the
        per-chunk log storm was the real cost.
        """
        # The first delta names the call; the continuation carries no name and
        # arguments that do not parse on their own.
        wire = ChatCompletionsWire(
            deltas=[
                tool_call_delta(
                    {"index": 0, "id": "call_1", "name": "threetears_calculator", "arguments": '{"expression":'}
                ),
                tool_call_delta({"index": 0, "id": None, "name": None, "arguments": ' "2+2"}'}),
            ],
            finish="tool_calls",
        )

        chunks: list[AIMessageChunk] = [chunk async for chunk in openrouter_model(wire).astream("hi")]

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
        wire = ChatCompletionsWire(
            message={
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_junk", "type": "function", "function": {"name": _JUNK_NAME, "arguments": "not json"}},
                    {
                        "id": "call_ok",
                        "type": "function",
                        "function": {"name": "threetears_calculator", "arguments": "not json"},
                    },
                ],
            },
            finish="tool_calls",
        )
        model = openrouter_model(wire)
        # the provider's own ``_agenerate`` runs only when no deadline is set; with one the
        # answer comes from the stream (``test_agenerate_under_a_deadline_drops_junk_names_too``)
        model.request_timeout = None

        result = await model.ainvoke("hi")

        assert wire.bodies[-1].get("stream") is not True
        kept = result.invalid_tool_calls
        assert len(kept) == 1
        assert kept[0]["name"] == "threetears_calculator"
        assert all(call["name"] != _JUNK_NAME for call in kept)

    @pytest.mark.asyncio
    async def test_agenerate_under_a_deadline_drops_junk_names_too(self) -> None:
        """with a deadline ``_agenerate`` collects the stream; the junk filter still applies."""
        wire = ChatCompletionsWire(
            deltas=[
                tool_call_delta(
                    {"index": 0, "id": "call_junk", "name": _JUNK_NAME, "arguments": "not json"},
                    {"index": 1, "id": "call_ok", "name": "threetears_calculator", "arguments": "not json"},
                )
            ],
            finish="tool_calls",
        )
        model = openrouter_model(wire)
        assert model.call_deadline_s() == 120

        result = await model.ainvoke("hi")

        assert wire.bodies[-1]["stream"] is True
        assert [call["name"] for call in result.invalid_tool_calls] == ["threetears_calculator"]

    @pytest.mark.asyncio
    async def test_no_tools_bound_means_no_translation(self) -> None:
        """Without any prior ``bind_tools`` call the model has no translation
        map, so even an underscored name that a dotted tool WOULD mangle to
        reaches the caller unchanged.
        """
        model = openrouter_model(_answer_calling("threetears_calculator", "external_tool"))

        result = await model.ainvoke("hi")

        # No bind_tools means no translation map; the names stay as-is.
        assert [call["name"] for call in result.tool_calls] == ["threetears_calculator", "external_tool"]

    @pytest.mark.asyncio
    async def test_an_unmatched_name_passes_through(self) -> None:
        """A tool-call name not in the reverse map (e.g. from a tool
        that was already underscored, or an LLM hallucination) passes
        through unchanged.
        """
        model = openrouter_model(_answer_calling("some_other_tool"))
        model.bind_tools([_DottedTool()])

        result = await model.ainvoke("hi")

        assert [call["name"] for call in result.tool_calls] == ["some_other_tool"]

    @pytest.mark.asyncio
    async def test_rebind_replaces_reverse_map(self) -> None:
        """A second ``bind_tools`` call with a different tool set
        replaces the reverse map wholesale. Otherwise stale entries
        from a prior bind would translate names that don't belong to
        the current bind.
        """

        class _OtherTool(_DottedTool):
            name: str = "threetears.web_search"

        model = openrouter_model(_answer_calling("threetears_web_search", "threetears_calculator"))
        model.bind_tools([_DottedTool()])
        model.bind_tools([_OtherTool()])

        result = await model.ainvoke("hi")

        # the current bind's name is untranslated; the earlier bind's is not
        assert [call["name"] for call in result.tool_calls] == ["threetears.web_search", "threetears_calculator"]

    @pytest.mark.asyncio
    async def test_astream_translates_aimessage_chunk_tool_calls(self) -> None:
        """``astream`` must translate ``tool_call_chunks[i]["name"]`` on
        every yielded ``AIMessageChunk``.

        The wrapper used to override ``_astream`` and translate the
        nested ``ChatGenerationChunk.message``. That broke LangGraph's
        ``astream_events(version="v2")`` event tap -- 190 chunks would
        reach the consumer's ``async for`` loop but zero
        ``on_chat_model_stream`` callbacks would fire -- observed in
        prod conv ``019e1f3d`` on 2026-05-13. Fixed by moving the
        translation off ``_astream`` and onto ``astream`` (the public
        Runnable method), so ``BaseChatModel.astream``'s callback wiring
        runs unchanged against the parent's untouched ``_astream``
        output and we post-process the AIMessageChunks after they're
        yielded. This test pins the new contract.
        """
        # One delta carrying the wire-form tool_call name the LLM emitted.
        wire = ChatCompletionsWire(
            deltas=[tool_call_delta({"index": 0, "id": "call_1", "name": "threetears_calculator", "arguments": ""})],
            finish="tool_calls",
        )
        model = openrouter_model(wire)
        model.bind_tools([_DottedTool()])

        chunks: list[AIMessageChunk] = [chunk async for chunk in model.astream("hi")]

        # ``BaseChatModel.astream`` auto-yields a final empty chunk
        # with ``chunk_position="last"`` after the source iterator
        # completes. Filter for the tool-call-carrying chunk so the
        # assertion stays robust to that framework behavior.
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

        Regression test for prod conv ``019e1f3d``: the previous
        ``_astream`` override silently dropped callback events even
        when the chunk iteration itself worked, leaving event-driven
        UIs (WS streaming) with the saved DB content but a blank live
        stream. Pinning this contract here means a future refactor
        that re-introduces an ``_astream`` override (or any other
        change that breaks the callback chain) fails CI loudly
        instead of shipping silently and surfacing only as a prod
        incident.
        """
        model = openrouter_model(ChatCompletionsWire(deltas=text_deltas("hello ", "world", "!")))

        stream_event_count = 0
        collected_text = ""
        async for event in model.astream_events("hi", version="v2"):
            if event["event"] == "on_chat_model_stream":
                stream_event_count += 1
                collected_text += event["data"]["chunk"].content

        # ``BaseChatModel.astream`` adds a final empty chunk with
        # ``chunk_position="last"`` after the source iterator finishes,
        # producing one extra ``on_chat_model_stream`` event. Three
        # streamed deltas → at least 3 events, and ``collected_text`` is
        # robust to that empty tail.
        assert stream_event_count >= 3, (
            f"Expected >=3 on_chat_model_stream events (one per streamed"
            f" delta plus the framework's final empty chunk); got"
            f" {stream_event_count}. The wrapper is breaking the"
            f" callback chain that drives astream_events(v2) — exactly"
            f" the 2026-05-13 production fingerprint (chunks delivered to"
            f" consumer, zero stream events emitted)."
        )
        assert collected_text == "hello world!", (
            "Stream events fired but their chunks did not carry the"
            f" provider's content as-streamed; got {collected_text!r}."
        )

    @pytest.mark.asyncio
    async def test_astream_events_survives_with_config_callbacks(self) -> None:
        """``with_config(callbacks=[...])`` must not strip the event_streamer.

        Production failure mode from 2026-05-13 (prod conv
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
        ``tokens_dispatched_count == 0`` fingerprint production hit.

        The previous regression test (above) only exercises the bare
        wrapper instance, so it passed CI while production was broken.
        This test mirrors what ``create_chat_model`` actually does --
        wrap the model with ``with_config(callbacks=[...])`` -- so the
        contextvar-vs-bound-callbacks merge path is on the test surface.
        """
        from langchain_core.callbacks import AsyncCallbackHandler

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
                if token:
                    self.token_seen += 1

        bound_cb = _RecordingCallback()
        model = openrouter_model(ChatCompletionsWire(deltas=text_deltas("hello ", "world", "!")))
        # Mirror ``threetears.models.factory.create_chat_model``'s
        # ``model.with_config(callbacks=[...])`` step.
        bound_model = model.with_config(callbacks=[bound_cb])

        stream_event_count = 0
        async for event in bound_model.astream_events("hi", version="v2"):
            if event["event"] == "on_chat_model_stream":
                stream_event_count += 1

        # Event_streamer must still see chat-model-stream events even
        # though the bound list of callbacks is also present.
        assert stream_event_count >= 3, (
            "with_config-bound list callbacks REPLACED the contextvar's"
            " event_streamer manager — `on_chat_model_stream` events"
            f" never reached astream_events. Got {stream_event_count}"
            " events (need >=3 from three streamed deltas plus framework's"
            " trailing empty chunk). This is the 2026-05-13"
            " production fingerprint — fix the wrapper's `astream`"
            " override (do not forward `config` verbatim; pre-merge it"
            " with the contextvar via merge_configs)."
        )
        # And the bound callbacks must STILL fire — the fix can't
        # silently drop UsageTracker / CircuitBreaker either.
        assert bound_cb.start_seen >= 1, (
            "Bound callback's on_chat_model_start never fired —"
            " the fix dropped the with_config list when preserving the"
            " contextvar manager. Both must propagate."
        )
        assert bound_cb.token_seen >= 3, (
            "Bound callback's on_llm_new_token fired"
            f" {bound_cb.token_seen} times for text — expected >=3 (one per"
            " streamed delta). The fix silently dropped the list of bound"
            " handlers somewhere in the merge."
        )


class TestVanillaChatAnthropicBaseline:
    """Baseline confirming vanilla ``ChatAnthropic`` -- with no wrapper
    in the inheritance chain -- emits ``on_chat_model_stream`` events
    correctly. Comparison case for the wrapper tests above. If THIS
    test fails the bug isn't in our wrapper; it's in the LangChain
    framework or the scripted wire. If this passes and the
    wrapper test fails, the wrapper is the culprit.

    Anthropic-direct is the second provider Pace asked us to verify on
    2026-05-13 before cutting the 3tears bump that ships the wrapper
    fix.
    """

    @pytest.mark.asyncio
    async def test_anthropic_direct_emits_on_chat_model_stream(self) -> None:
        """Vanilla ``ChatAnthropic.astream_events(v2)`` emits one
        ``on_chat_model_stream`` per streamed delta (plus the framework's
        bracketing empty chunks). No wrapper subclassing involved.
        """
        from langchain_anthropic import ChatAnthropic

        wire = AnthropicMessagesWire(blocks=[TextBlock(["anthropic ", "direct ", "streams"])])
        with wire.serve() as url:
            model = ChatAnthropic(
                model=DEFAULT_CHAT_MODEL,  # type: ignore[call-arg]
                anthropic_api_key="sk-test",  # type: ignore[arg-type]
                base_url=url,
                max_retries=0,
            )
            stream_event_count = 0
            collected_text = ""
            async for event in model.astream_events("hi", version="v2"):
                if event["event"] == "on_chat_model_stream":
                    stream_event_count += 1
                    collected_text += event["data"]["chunk"].content

        assert stream_event_count >= 3, (
            f"Vanilla ChatAnthropic.astream_events(v2) did not emit a"
            f" stream event per delta (got {stream_event_count}). This"
            f" baseline failing means the framework is broken or the"
            f" scripted wire is wrong -- look there before"
            f" suspecting the OpenRouter wrapper."
        )
        assert collected_text == "anthropic direct streams"


# -- forward (outbound) name translation -------------------------------------


class TestForwardTranslateMessage:
    """``forward_translate_message`` mangles dotted tool-call names on an
    outbound message, non-mutatingly.

    Forward mirror of ``reverse_translate_message``: the reverse pass
    un-translates a provider RESPONSE in place, this pass mangles the
    conversation history that is about to be SENT and must NOT corrupt
    the caller's message objects (they stay canonical for dispatch /
    logging / persistence), so it returns a shallow copy.
    """

    def test_mangles_dotted_tool_calls_name(self) -> None:
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "threetears.web_search", "args": {"q": "x"}, "id": "c1"}],
        )
        out = forward_translate_message(msg)
        assert out.tool_calls[0]["name"] == "threetears_web_search"

    def test_does_not_mutate_caller_message(self) -> None:
        """The returned message is a copy; the caller's AIMessage keeps the
        canonical dotted name so dispatch / logging still match the registry.
        """
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "threetears.web_search", "args": {"q": "x"}, "id": "c1"}],
        )
        out = forward_translate_message(msg)
        assert out is not msg
        assert msg.tool_calls[0]["name"] == "threetears.web_search"

    def test_dotless_name_returns_same_object(self) -> None:
        """No dotted name means no rename, so the original object is returned
        (no needless copy).
        """
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "plain_tool", "args": {}, "id": "c1"}],
        )
        assert forward_translate_message(msg) is msg

    def test_mangles_invalid_tool_calls_name(self) -> None:
        """A dotted hallucination that landed in ``invalid_tool_calls`` (e.g.
        ``functions.web_search``) is mangled too — it would 400 the turn
        unchanged.
        """
        msg = AIMessage(
            content="",
            invalid_tool_calls=[
                {"name": "functions.web_search", "args": "{bad", "id": "c1", "error": "x"},
            ],
        )
        out = forward_translate_message(msg)
        assert out.invalid_tool_calls[0]["name"] == "functions_web_search"

    def test_mangles_tool_call_chunks_name(self) -> None:
        chunk = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"name": "threetears.calculator", "args": "", "id": "c1", "index": 0},
            ],
        )
        out = forward_translate_message(chunk)
        assert out.tool_call_chunks[0]["name"] == "threetears_calculator"

    def test_round_trips_with_reverse(self) -> None:
        """A forward-mangled bound-tool name un-maps via the per-bind reverse
        map — the outbound and inbound translations are symmetric.
        """
        _, reverse_map = build_name_translation([_DottedTool()])
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "threetears.calculator", "args": {}, "id": "c1"}],
        )
        wire = forward_translate_message(msg)
        assert wire.tool_calls[0]["name"] == "threetears_calculator"
        reverse_translate_message(wire, reverse_map)
        assert wire.tool_calls[0]["name"] == "threetears.calculator"


class TestForwardTranslateInput:
    """``forward_translate_input`` applies the forward mangle across a
    message-sequence input and leaves non-list inputs alone."""

    def test_non_list_passes_through(self) -> None:
        assert forward_translate_input("hi") == "hi"

    def test_translates_list_of_messages(self) -> None:
        original = [
            SystemMessage(content="sys"),
            AIMessage(
                content="",
                tool_calls=[{"name": "threetears.web_search", "args": {}, "id": "c1"}],
            ),
        ]
        out = forward_translate_input(original)
        assert out is not original
        assert out[1].tool_calls[0]["name"] == "threetears_web_search"
        # caller's list + messages untouched
        assert original[1].tool_calls[0]["name"] == "threetears.web_search"

    def test_returns_same_list_when_no_translation(self) -> None:
        original = [SystemMessage(content="sys"), HumanMessage(content="hi")]
        assert forward_translate_input(original) is original


def _outbound_with_dotted_call() -> list[Any]:
    """a history whose prior round called a tool by its canonical dotted name.

    :return: the messages
    :rtype: list[Any]
    """
    return [
        SystemMessage(content="sys"),
        AIMessage(
            content="",
            tool_calls=[{"name": "threetears.web_search", "args": {"q": "x"}, "id": "c1"}],
        ),
        ToolMessage(content="hit", tool_call_id="c1"),
    ]


class TestOpenRouterForwardTranslation:
    """The wrapper forward-translates dotted tool-call names on the OUTBOUND
    ``messages`` before the provider call.

    Root cause (chunk 03): a prior round's ``AIMessage`` carrying
    ``tool_calls`` with the canonical dotted name is re-sent on the next
    round. OpenRouter routes to backends (Bedrock / OpenAI-compat) whose
    ``^[a-zA-Z0-9_-]`` validator rejects the dot, 400-ing the turn. The
    reverse pass un-translates responses but nothing mangled the dotted
    name back to wire form on the way OUT. These tests pin the forward
    direction on both the streaming and non-streaming code paths, on the
    request body the SDK actually sent.
    """

    @pytest.mark.asyncio
    async def test_astream_forward_translates_outbound_dotted_names(self) -> None:
        """``astream`` sends the wire (underscored) name; the caller's message
        keeps the canonical dotted name."""
        wire = ChatCompletionsWire()
        outbound = _outbound_with_dotted_call()

        async for _ in openrouter_model(wire).astream(outbound):
            pass

        assert wire.sent_tool_call_names() == ["threetears_web_search"]
        # The caller's original AIMessage keeps the canonical dotted name.
        assert outbound[1].tool_calls[0]["name"] == "threetears.web_search"

    @pytest.mark.asyncio
    async def test_agenerate_forward_translates_outbound_dotted_names(self) -> None:
        """``_agenerate`` (the non-streaming path) mangles the outbound names too.

        Through ``agenerate``, which does not forward-translate itself, so the
        name on the wire is the one ``_agenerate`` sent.
        """
        wire = ChatCompletionsWire()
        outbound = _outbound_with_dotted_call()
        model = openrouter_model(wire)
        # the provider's own ``_agenerate`` runs only when no deadline is set
        model.request_timeout = None

        await model.agenerate([outbound])

        assert wire.bodies[-1].get("stream") is not True
        assert wire.sent_tool_call_names() == ["threetears_web_search"]
        assert outbound[1].tool_calls[0]["name"] == "threetears.web_search"

    @pytest.mark.asyncio
    async def test_agenerate_under_a_deadline_forward_translates_too(self) -> None:
        """with a deadline ``_agenerate`` sends through the stream, names mangled the same way."""
        wire = ChatCompletionsWire()
        outbound = _outbound_with_dotted_call()

        result = await openrouter_model(wire).agenerate([outbound])

        assert wire.bodies[-1]["stream"] is True
        assert result.generations[0][0].message.content == "ok"
        assert wire.sent_tool_call_names() == ["threetears_web_search"]
        assert outbound[1].tool_calls[0]["name"] == "threetears.web_search"


from threetears.models.errors import ModelCallTimeout, is_provider_error  # noqa: E402


class TestTheTimeoutHolds:
    """The deadline lives on the shared mixin; OpenRouter sets it from its timeout.

    The SDK applies the timeout to each read, and OpenRouter keeps a call open with
    keep-alives, so a stalled upstream ran as long as it liked: one call took 218 s
    against a 120 s timeout (metallm, 2026-09-26). Here the timeout holds, on silence:
    a non-streamed call collects the stream too (``test_openrouter_deadline_is_on_silence``).
    The silences are the scripted API's own, sent through the real SDK."""

    @staticmethod
    def _model(wire: ChatCompletionsWire, timeout_ms: int | None) -> Any:
        model = openrouter_model(wire)
        model.request_timeout = timeout_ms
        return model

    @pytest.mark.asyncio
    async def test_a_call_that_never_answers_ends_at_the_timeout(self) -> None:
        wire = ChatCompletionsWire(deltas=text_deltas("late"), silence_after=0, silence_s=5)
        with pytest.raises(ModelCallTimeout):
            await self._model(wire, 50).ainvoke([HumanMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_a_stream_that_goes_quiet_ends_at_the_timeout(self) -> None:
        wire = ChatCompletionsWire(deltas=text_deltas("Hel", "lo"), silence_after=1, silence_s=5)
        seen: list[str] = []
        with pytest.raises(ModelCallTimeout):
            async for chunk in self._model(wire, 50).astream([HumanMessage(content="hi")]):
                seen.append(str(chunk.content))
        # the stream's own empty bracketing chunks aside, only what came before the silence
        assert [text for text in seen if text] == ["Hel"]

    @pytest.mark.asyncio
    async def test_a_long_stream_that_keeps_arriving_finishes(self) -> None:
        """The limit is on silence, not length: ten chunks 30 ms apart -- 300 ms in all -- outlast a
        150 ms timeout. The gap sits well inside the timeout: at 30 ms against 50 ms, scheduler
        jitter on a loaded machine was enough to trip it."""
        wire = ChatCompletionsWire(deltas=text_deltas(*(str(i) for i in range(10))), gap_s=0.03)
        seen = [str(c.content) async for c in self._model(wire, 150).astream([HumanMessage(content="hi")])]
        assert "".join(seen) == "0123456789"

    @pytest.mark.asyncio
    async def test_no_timeout_set_means_no_deadline(self) -> None:
        wire = ChatCompletionsWire(answer_delay_s=0.1)
        result = await self._model(wire, None).ainvoke([HumanMessage(content="hi")])
        assert wire.bodies[-1].get("stream") is not True
        assert result.content == "ok"


def test_the_other_wrappers_keep_no_extra_deadline() -> None:
    """OpenAI's and Anthropic's SDKs were not reported to stall; only OpenRouter sets one.

    Asked of the models their public factories build, with a request timeout set on each, so a
    wrapper that started deriving a deadline from it would show here.
    """
    from .provider_wire import anthropic_model, openai_model

    openai = openai_model(ChatCompletionsWire(), request_timeout=30)
    anthropic = anthropic_model("http://127.0.0.1:9", default_request_timeout=30)

    assert openai.call_deadline_s() is None
    assert anthropic.call_deadline_s() is None


class TestAProviderFailureIsNamedAsOne:
    """A caller that catches everything a model call raised can tell the provider's failure
    from its own bug, and the deadline above from any other timeout."""

    def test_the_whole_call_deadline_is_a_provider_failure(self) -> None:
        assert is_provider_error(ModelCallTimeout("no answer"))

    def test_an_open_circuit_is_a_provider_failure(self) -> None:
        """The breaker refuses a provider that keeps failing: an outage, raised from threetears' own package."""
        from threetears.models.circuit_breaker import CircuitOpenError

        assert is_provider_error(CircuitOpenError("openrouter", 30.0))

    def test_an_sdk_error_and_the_openrouter_value_error_are(self) -> None:
        class _SdkError(Exception):
            pass

        _SdkError.__module__ = "openai._exceptions"
        assert is_provider_error(_SdkError("429"))
        assert is_provider_error(ValueError("OpenRouter API error: rate limited"))

    @pytest.mark.parametrize("exc", [KeyError("reply"), ValueError("bad field"), TimeoutError()])
    def test_a_bug_or_the_callers_own_timeout_is_not(self, exc: Exception) -> None:
        assert not is_provider_error(exc)


class TestOpenRouterToolChoice:
    """A ``tool_choice`` naming a dotted tool names it by the wire name the tool was bound under."""

    def test_a_named_tool_is_chosen_by_its_wire_name(self) -> None:
        model = create_openrouter_chat("deepseek/deepseek-chat-v3-0324", "sk-test")
        bound = model.bind_tools([_DottedTool()], tool_choice="threetears.calculator")
        choice = bound.kwargs["tool_choice"]  # type: ignore[attr-defined]
        assert "threetears.calculator" not in str(choice)
        assert "threetears_calculator" in str(choice)

    def test_required_passes_through(self) -> None:
        model = create_openrouter_chat("deepseek/deepseek-chat-v3-0324", "sk-test")
        bound = model.bind_tools([_DottedTool()], tool_choice="required")
        assert bound.kwargs["tool_choice"] == "required"  # type: ignore[attr-defined]
