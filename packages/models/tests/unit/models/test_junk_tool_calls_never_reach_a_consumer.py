"""A tool call whose name fails the canonical regex never reaches anyone downstream of a chat model.

The 2026-05-19 incident (conv ``019e3e26-9870-7a03-8f04-8cc6a4f5f418``): a model emitted a tool call
named ``memory_recall" name="memory_recall`` (an XML attribute leaked into the name), and it was
dispatched and persisted. The first filter cleared ``invalid_tool_calls`` on each streamed chunk,
which protected a finished ``ainvoke`` and nothing else that streams:

- the name also rides each chunk's ``tool_call_chunks``, so a consumer that adds the chunks up --
  every streaming caller does -- got the junk call back;
- Anthropic sends a call's name and its arguments in separate events, so the chunk carrying the
  name has empty arguments, parses as a valid ``tool_calls`` entry, and was never looked at;
- ``astream_events``, a streaming ``ainvoke``'s callbacks, and the aggregate handed to
  ``on_chat_model_end`` all see chunks before the public ``astream`` does.

Every answer here comes from a scripted provider API through the real provider SDK
(:mod:`.provider_wire`), and every assertion is on what a consumer receives.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from .provider_wire import (
    AnthropicMessagesWire,
    ChatCompletionsWire,
    TextBlock,
    ToolUseBlock,
    anthropic_model,
    openai_model,
    openrouter_model,
    text_deltas,
    tool_call_delta,
)
from .translation_helpers import DottedTool

#: the XML-attribute-leak tool name from the 2026-05-19 prod incident
_JUNK_NAME = 'memory_recall" name="memory_recall'


def _summed(chunks: list[AIMessageChunk]) -> AIMessageChunk:
    """the chunks added up, as a streaming consumer accumulates them.

    :param chunks: the chunks in the order received
    :ptype chunks: list[AIMessageChunk]
    :return: their sum
    :rtype: AIMessageChunk
    """
    total = chunks[0]
    for chunk in chunks[1:]:
        total = total + chunk
    return total


def _names_anywhere(message: AIMessage) -> list[Any]:
    """every tool name a message carries, in any field a consumer could read one from.

    :param message: a finished message or a chunk
    :ptype message: AIMessage
    :return: the names, from ``tool_calls``, ``invalid_tool_calls``, ``tool_call_chunks`` and
        content blocks
    :rtype: list[Any]
    """
    names = [call.get("name") for call in message.tool_calls]
    names += [call.get("name") for call in message.invalid_tool_calls]
    names += [call.get("name") for call in getattr(message, "tool_call_chunks", None) or []]
    if isinstance(message.content, list):
        names += [block.get("name") for block in message.content if isinstance(block, dict) and "name" in block]
    return names


def _junk_split_across_deltas() -> ChatCompletionsWire:
    """a chat-completions answer streaming a junk call and a valid one, each split over deltas.

    The junk call's arguments are valid JSON once summed, so the sum would hand it to dispatch as
    a well-formed ``tool_calls`` entry.

    :return: the scripted API
    :rtype: ChatCompletionsWire
    """
    return ChatCompletionsWire(
        deltas=[
            tool_call_delta({"index": 0, "id": "call_junk", "name": _JUNK_NAME, "arguments": ""}),
            tool_call_delta({"index": 0, "id": None, "name": None, "arguments": '{"query": '}),
            tool_call_delta({"index": 0, "id": None, "name": None, "arguments": '"x"}'}),
            tool_call_delta({"index": 1, "id": "call_ok", "name": "threetears_calculator", "arguments": ""}),
            tool_call_delta({"index": 1, "id": None, "name": None, "arguments": '{"expression": '}),
            tool_call_delta({"index": 1, "id": None, "name": None, "arguments": '"2+2"}'}),
        ],
        finish="tool_calls",
    )


def _anthropic_junk_answer() -> AnthropicMessagesWire:
    """an Anthropic answer with a junk ``tool_use`` block beside a valid one, input streamed in parts.

    :return: the scripted API
    :rtype: AnthropicMessagesWire
    """
    return AnthropicMessagesWire(
        blocks=[
            TextBlock(["let me ", "check"]),
            ToolUseBlock("toolu_junk", _JUNK_NAME, ['{"query": ', '"x"}']),
            ToolUseBlock("toolu_ok", "threetears_calculator", ['{"expression": ', '"2+2"}']),
        ],
        stop_reason="tool_use",
    )


class TestASummedStreamCarriesNoJunkCall:
    """``astream``: what a consumer adds up holds the valid call and nothing of the junk one."""

    @pytest.mark.asyncio
    async def test_openai(self) -> None:
        model = openai_model(_junk_split_across_deltas())
        model.bind_tools([DottedTool()])

        chunks = [chunk async for chunk in model.astream("hi")]
        total = _summed(chunks)

        assert _JUNK_NAME not in _names_anywhere(total)
        assert all(_JUNK_NAME not in _names_anywhere(chunk) for chunk in chunks)
        assert total.tool_calls == [
            {"name": "threetears.calculator", "args": {"expression": "2+2"}, "id": "call_ok", "type": "tool_call"}
        ]
        assert total.invalid_tool_calls == []

    @pytest.mark.asyncio
    async def test_openrouter(self) -> None:
        model = openrouter_model(_junk_split_across_deltas())
        model.bind_tools([DottedTool()])

        total = _summed([chunk async for chunk in model.astream("hi")])

        assert _JUNK_NAME not in _names_anywhere(total)
        assert [call["name"] for call in total.tool_calls] == ["threetears.calculator"]
        assert total.tool_calls[0]["args"] == {"expression": "2+2"}

    @pytest.mark.asyncio
    async def test_anthropic_name_and_arguments_in_separate_events(self) -> None:
        wire = _anthropic_junk_answer()
        with wire.serve() as url:
            # through the binding, so the tools go out and the answer keeps its content blocks
            chunks = [chunk async for chunk in anthropic_model(url).bind_tools([DottedTool()]).astream("hi")]
        total = _summed(chunks)

        assert _JUNK_NAME not in _names_anywhere(total)
        assert all(_JUNK_NAME not in _names_anywhere(chunk) for chunk in chunks)
        assert [call["name"] for call in total.tool_calls] == ["threetears.calculator"]
        assert total.tool_calls[0]["args"] == {"expression": "2+2"}
        assert total.tool_calls[0]["id"] == "toolu_ok"
        # the junk call's content blocks went with it; the text and the valid call's block stayed
        assert isinstance(total.content, list)
        assert "toolu_junk" not in [block.get("id") for block in total.content if isinstance(block, dict)]
        texts = [block["text"] for block in total.content if isinstance(block, dict) and block.get("type") == "text"]
        assert "".join(texts) == "let me check"

    def test_a_sync_stream_too(self) -> None:
        wire = _anthropic_junk_answer()
        with wire.serve() as url:
            chunks = list(anthropic_model(url).bind_tools([DottedTool()]).stream("hi"))
        total = _summed(chunks)

        assert _JUNK_NAME not in _names_anywhere(total)
        assert [call["id"] for call in total.tool_calls] == ["toolu_ok"]


class TestAValidCallStreamsUnheld:
    """A call named on its first fragment is released at once, fragment by fragment."""

    @pytest.mark.asyncio
    async def test_each_fragment_arrives_as_its_own_chunk_and_the_sum_is_whole(self) -> None:
        wire = ChatCompletionsWire(
            deltas=[
                tool_call_delta({"index": 0, "id": "call_1", "name": "threetears_calculator", "arguments": ""}),
                tool_call_delta({"index": 0, "id": None, "name": None, "arguments": '{"expression"'}),
                tool_call_delta({"index": 0, "id": None, "name": None, "arguments": ': "2+2"}'}),
            ],
            finish="tool_calls",
        )
        model = openai_model(wire)
        model.bind_tools([DottedTool()])

        chunks = [chunk async for chunk in model.astream("hi")]
        carrying = [chunk for chunk in chunks if chunk.tool_call_chunks]

        # one chunk per wire delta, in wire order: nothing was held back and released in a lump
        assert [chunk.tool_call_chunks[0]["args"] for chunk in carrying] == ["", '{"expression"', ': "2+2"}']
        assert carrying[0].tool_call_chunks[0]["name"] == "threetears.calculator"
        assert _summed(chunks).tool_calls == [
            {"name": "threetears.calculator", "args": {"expression": "2+2"}, "id": "call_1", "type": "tool_call"}
        ]


class TestTextIsNeverHeld:
    """Text streams in wire order, even while a tool call is waiting for its name."""

    @pytest.mark.asyncio
    async def test_text_around_a_named_call_keeps_its_place(self) -> None:
        wire = ChatCompletionsWire(
            deltas=[
                *text_deltas("let me ", "check"),
                tool_call_delta({"index": 0, "id": "call_1", "name": "threetears_calculator", "arguments": "{}"}),
                {"content": " done"},
            ],
            finish="tool_calls",
        )

        chunks = [chunk async for chunk in openai_model(wire).astream("hi")]
        kinds = [
            "tool" if chunk.tool_call_chunks else chunk.content
            for chunk in chunks
            if chunk.content or chunk.tool_call_chunks
        ]

        assert kinds == ["let me ", "check", "tool", " done"]

    @pytest.mark.asyncio
    async def test_text_overtakes_a_call_still_waiting_for_its_name(self) -> None:
        """A fragment that arrives before its call's name is held; text behind it is not."""
        wire = ChatCompletionsWire(
            deltas=[
                tool_call_delta({"index": 0, "id": "call_1", "name": None, "arguments": '{"expression": '}),
                {"content": "thinking"},
                tool_call_delta({"index": 0, "id": None, "name": "threetears_calculator", "arguments": '"2+2"}'}),
            ],
            finish="tool_calls",
        )

        chunks = [chunk async for chunk in openai_model(wire).astream("hi")]
        kinds = [
            "tool" if chunk.tool_call_chunks else chunk.content
            for chunk in chunks
            if chunk.content or chunk.tool_call_chunks
        ]

        assert kinds == ["thinking", "tool", "tool"]
        assert _summed(chunks).tool_calls == [
            {"name": "threetears_calculator", "args": {"expression": "2+2"}, "id": "call_1", "type": "tool_call"}
        ]

    @pytest.mark.asyncio
    async def test_a_held_call_whose_name_turns_out_junk_is_dropped_whole(self) -> None:
        wire = ChatCompletionsWire(
            deltas=[
                tool_call_delta({"index": 0, "id": "call_junk", "name": None, "arguments": '{"query": '}),
                {"content": "thinking"},
                tool_call_delta({"index": 0, "id": None, "name": _JUNK_NAME, "arguments": '"x"}'}),
            ],
            finish="tool_calls",
        )

        chunks = [chunk async for chunk in openai_model(wire).astream("hi")]
        total = _summed(chunks)

        assert total.tool_call_chunks == []
        assert total.tool_calls == []
        assert total.invalid_tool_calls == []
        assert total.content == "thinking"

    @pytest.mark.asyncio
    async def test_a_call_never_named_is_released_when_the_stream_ends(self) -> None:
        """A nameless fragment is not junk; held for a name that never came, it still goes out."""
        wire = ChatCompletionsWire(
            deltas=[
                tool_call_delta({"index": 0, "id": "call_1", "name": None, "arguments": '{"expression": "2+2"}'}),
                {"content": "done"},
            ],
            finish="tool_calls",
        )

        chunks = [chunk async for chunk in openai_model(wire).astream("hi")]
        kinds = [
            "tool" if chunk.tool_call_chunks else chunk.content
            for chunk in chunks
            if chunk.content or chunk.tool_call_chunks
        ]

        assert kinds == ["done", "tool"]
        assert _summed(chunks).tool_call_chunks[0]["args"] == '{"expression": "2+2"}'

    @pytest.mark.asyncio
    async def test_a_name_completed_into_junk_after_release_stops_there(self) -> None:
        """A name split over fragments, valid until its last one: what went out under the valid
        prefix stays out, the rest of the call is dropped, and the junk name itself never leaves.
        """
        wire = ChatCompletionsWire(
            deltas=[
                tool_call_delta({"index": 0, "id": "call_1", "name": "memory_recall", "arguments": '{"query": '}),
                tool_call_delta({"index": 0, "id": None, "name": '" name="memory_recall', "arguments": '"x"}'}),
            ],
            finish="tool_calls",
        )

        total = _summed([chunk async for chunk in openai_model(wire).astream("hi")])

        assert _JUNK_NAME not in _names_anywhere(total)
        assert [(call["name"], call["args"]) for call in total.tool_call_chunks] == [("memory_recall", '{"query": ')]


class TestEventsAndCallbacksSeeNoJunkCall:
    """Every surface fed from inside the model run: stream events, callbacks, the end aggregate."""

    @pytest.mark.asyncio
    async def test_astream_events(self) -> None:
        model = openai_model(_junk_split_across_deltas())
        model.bind_tools([DottedTool()])

        streamed: list[AIMessageChunk] = []
        ended: list[Any] = []
        async for event in model.astream_events("hi", version="v2"):
            if event["event"] == "on_chat_model_stream":
                streamed.append(event["data"]["chunk"])
            elif event["event"] == "on_chat_model_end":
                ended.append(event["data"]["output"])

        assert streamed, "no stream events: the callback chain is broken"
        assert all(_JUNK_NAME not in _names_anywhere(chunk) for chunk in streamed)
        assert _JUNK_NAME not in _names_anywhere(_summed(streamed))
        assert len(ended) == 1
        assert _JUNK_NAME not in _names_anywhere(ended[0])
        assert [call["id"] for call in ended[0].tool_calls] == ["call_ok"]

    @pytest.mark.asyncio
    async def test_astream_events_over_anthropic(self) -> None:
        wire = _anthropic_junk_answer()
        streamed: list[AIMessageChunk] = []
        with wire.serve() as url:
            async for event in anthropic_model(url).bind_tools([DottedTool()]).astream_events("hi", version="v2"):
                if event["event"] == "on_chat_model_stream":
                    streamed.append(event["data"]["chunk"])

        assert all(_JUNK_NAME not in _names_anywhere(chunk) for chunk in streamed)
        assert [call["id"] for call in _summed(streamed).tool_calls] == ["toolu_ok"]

    @pytest.mark.asyncio
    async def test_a_streaming_ainvoke_feeds_its_callbacks_no_junk_chunk(self) -> None:
        seen: list[Any] = []

        class _Recorder(AsyncCallbackHandler):
            async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
                seen.append(kwargs["chunk"].message)

        model = openai_model(_junk_split_across_deltas())
        result = await model.ainvoke("hi", stream=True, config={"callbacks": [_Recorder()]})

        assert seen, "no tokens reached the callback"
        assert all(_JUNK_NAME not in _names_anywhere(message) for message in seen)
        assert _JUNK_NAME not in _names_anywhere(result)
        assert [call["id"] for call in result.tool_calls] == ["call_ok"]

    @pytest.mark.asyncio
    async def test_the_callbacks_of_a_call_collected_under_a_deadline(self) -> None:
        """Under a deadline the run manager rides into the stream itself, where the SDK reports
        each chunk; the callbacks still get every released chunk and no junk one.
        """
        tokens: list[Any] = []

        class _Recorder(AsyncCallbackHandler):
            async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
                tokens.append(kwargs["chunk"].message)

        wire = _junk_split_across_deltas()
        model = openrouter_model(wire)
        assert model.call_deadline_s() is not None

        result = await model.ainvoke("hi", config={"callbacks": [_Recorder()]})

        assert wire.bodies[-1]["stream"] is True
        assert all(_JUNK_NAME not in _names_anywhere(message) for message in tokens)
        assert [fragment["args"] for message in tokens for fragment in message.tool_call_chunks] == [
            "",
            '{"expression": ',
            '"2+2"}',
        ]
        assert [call["id"] for call in result.tool_calls] == ["call_ok"]

    def test_a_sync_streaming_invoke_feeds_its_callbacks_no_junk_chunk(self) -> None:
        seen: list[Any] = []

        class _Recorder(BaseCallbackHandler):
            def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
                seen.append(kwargs["chunk"].message)

        wire = _anthropic_junk_answer()
        with wire.serve() as url:
            result = (
                anthropic_model(url)
                .bind_tools([DottedTool()])
                .invoke("hi", stream=True, config={"callbacks": [_Recorder()]})
            )

        assert seen
        assert all(_JUNK_NAME not in _names_anywhere(message) for message in seen)
        assert [call["id"] for call in result.tool_calls] == ["toolu_ok"]


class TestAFinishedAnswerCarriesNoJunkCall:
    """A non-streamed answer: a junk name with well-formed arguments is a ``tool_calls`` entry."""

    @pytest.mark.asyncio
    async def test_anthropic_tool_use_with_parsed_input(self) -> None:
        wire = AnthropicMessagesWire(
            blocks=[
                ToolUseBlock("toolu_junk", _JUNK_NAME, ['{"query": "x"}']),
                ToolUseBlock("toolu_ok", "threetears_calculator", ['{"expression": "2+2"}']),
            ],
            stop_reason="tool_use",
        )
        with wire.serve() as url:
            model = anthropic_model(url)
            result = await model.bind_tools([DottedTool()]).ainvoke("hi")

        assert wire.bodies[-1].get("stream") is not True
        assert _JUNK_NAME not in _names_anywhere(result)
        assert [call["name"] for call in result.tool_calls] == ["threetears.calculator"]
        assert [block["id"] for block in result.content if isinstance(block, dict)] == ["toolu_ok"]

    @pytest.mark.asyncio
    async def test_chat_completions_tool_call_with_parsed_arguments(self) -> None:
        wire = ChatCompletionsWire(
            message={
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_junk",
                        "type": "function",
                        "function": {"name": _JUNK_NAME, "arguments": json.dumps({"query": "x"})},
                    },
                    {
                        "id": "call_ok",
                        "type": "function",
                        "function": {"name": "threetears_calculator", "arguments": json.dumps({"expression": "2+2"})},
                    },
                ],
            },
            finish="tool_calls",
        )
        model = openai_model(wire)

        generated = await model.agenerate([[HumanMessage(content="hi")]])
        message = generated.generations[0][0].message

        assert _JUNK_NAME not in _names_anywhere(message)
        assert [call["id"] for call in message.tool_calls] == ["call_ok"]
