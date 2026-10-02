"""A tool call named in a content block carries the same name as the message's ``tool_calls``.

Anthropic carries every tool call twice: as a ``tool_use`` block in ``content`` and as a
``tool_calls`` entry. The wrappers translated the second back to the canonical dotted name and left
the first on the wire name, so a message said ``threetears.calculator`` in one place and
``threetears_calculator`` in the other. Going out again, LangChain prefers ``tool_calls`` for a
block whose id it finds there and sends the block's own name otherwise, so the forward translation
covers blocks too. Asserted through the real Anthropic SDK against a scripted Messages API.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from threetears.models.tool_name_translation import forward_translate_message, reverse_translate_message

from .provider_wire import AnthropicMessagesWire, TextBlock, ToolUseBlock, anthropic_model
from .translation_helpers import DottedTool


def _tool_block_names(message: AIMessage) -> list[Any]:
    """the names on a message's tool-call content blocks.

    :param message: the message
    :ptype message: AIMessage
    :return: the names, in content order
    :rtype: list[Any]
    """
    content = message.content if isinstance(message.content, list) else []
    return [block["name"] for block in content if isinstance(block, dict) and block.get("type") == "tool_use"]


def _calls_the_calculator() -> AnthropicMessagesWire:
    """an answer calling the calculator by its wire name, with some text first.

    :return: the scripted API
    :rtype: AnthropicMessagesWire
    """
    return AnthropicMessagesWire(
        blocks=[TextBlock(["adding"]), ToolUseBlock("toolu_1", "threetears_calculator", ['{"expression": "2+2"}'])],
        stop_reason="tool_use",
    )


class TestContentAndToolCallsNameTheSameTool:
    @pytest.mark.asyncio
    async def test_a_finished_answer(self) -> None:
        wire = _calls_the_calculator()
        with wire.serve() as url:
            result = await anthropic_model(url).bind_tools([DottedTool()]).ainvoke("hi")

        assert wire.bodies[-1].get("stream") is not True
        assert [call["name"] for call in result.tool_calls] == ["threetears.calculator"]
        assert _tool_block_names(result) == ["threetears.calculator"]

    @pytest.mark.asyncio
    async def test_a_streamed_answer(self) -> None:
        wire = _calls_the_calculator()
        with wire.serve() as url:
            chunks = [chunk async for chunk in anthropic_model(url).bind_tools([DottedTool()]).astream("hi")]
        total: AIMessageChunk = chunks[0]
        for chunk in chunks[1:]:
            total = total + chunk

        assert [call["name"] for call in total.tool_calls] == ["threetears.calculator"]
        assert _tool_block_names(total) == ["threetears.calculator"]


class TestTheNextTurnSendsTheWireName:
    @pytest.mark.asyncio
    async def test_a_two_turn_round_trip(self) -> None:
        """Turn one's answer goes back as history on turn two, under the wire name only."""
        wire = _calls_the_calculator()
        with wire.serve() as url:
            bound = anthropic_model(url).bind_tools([DottedTool()])
            first = await bound.ainvoke("what is 2+2?")
            history = [
                HumanMessage(content="what is 2+2?"),
                first,
                ToolMessage(content="4", tool_call_id=first.tool_calls[0]["id"]),
            ]
            wire.blocks = [TextBlock(["4"])]
            wire.stop_reason = "end_turn"
            second = await bound.ainvoke(history)

        assert wire.sent_tool_use_names() == ["threetears_calculator"]
        assert second.content == "4"
        # the caller's history keeps the canonical name in both places
        assert _tool_block_names(first) == ["threetears.calculator"]
        assert first.tool_calls[0]["name"] == "threetears.calculator"

    @pytest.mark.asyncio
    async def test_a_block_with_no_matching_tool_call_is_sent_by_its_wire_name(self) -> None:
        """LangChain sends a ``tool_use`` block's own name when no ``tool_calls`` entry shares its
        id, as for history rebuilt from stored content."""
        history = [
            HumanMessage(content="what is 2+2?"),
            AIMessage(
                content=[{"type": "tool_use", "id": "toolu_1", "name": "threetears.calculator", "input": {}}],
            ),
            ToolMessage(content="4", tool_call_id="toolu_1"),
        ]
        wire = AnthropicMessagesWire()
        with wire.serve() as url:
            await anthropic_model(url).ainvoke(history)

        assert wire.sent_tool_use_names() == ["threetears_calculator"]
        assert _tool_block_names(history[1]) == ["threetears.calculator"]  # type: ignore[arg-type]


class TestEveryToolCallBlockShape:
    """The other providers' tool-call blocks: OpenAI Responses' ``function_call``, LangChain's own."""

    @pytest.mark.parametrize("block_type", ["tool_use", "function_call", "tool_call", "tool_call_chunk"])
    def test_reverse_then_forward(self, block_type: str) -> None:
        message = AIMessage(
            content=[{"type": "text", "text": "x"}, {"type": block_type, "name": "threetears_calculator", "id": "c1"}]
        )

        reverse_translate_message(message, {"threetears_calculator": "threetears.calculator"})
        sent = forward_translate_message(message)

        assert message.content[1]["name"] == "threetears.calculator"  # type: ignore[index]
        assert sent.content[1]["name"] == "threetears_calculator"
        assert message.content[1]["name"] == "threetears.calculator", "the forward pass mutated history"  # type: ignore[index]

    def test_a_block_that_is_not_a_tool_call_keeps_its_name(self) -> None:
        """A server tool (Anthropic's own web search) is not one the caller bound or names."""
        block = {"type": "server_tool_use", "name": "web.search", "id": "s1"}
        message = AIMessage(content=[block])

        reverse_translate_message(message, {"web_search": "web.search"})

        assert forward_translate_message(message) is message
        assert message.content[0]["name"] == "web.search"  # type: ignore[index]
