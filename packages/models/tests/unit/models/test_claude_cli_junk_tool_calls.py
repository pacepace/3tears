"""A subscription model hands back no tool call whose name is junk, invoked or streamed.

The subscription backend turns each of the model's ``tool_use`` blocks into a ``tool_calls`` entry
under the name the caller bound. A block whose name is not a tool name (the 2026-05-19 XML leak,
``memory_recall" name="memory_recall``) became a call too, and the caller dispatched it. The CLI's
answer is scripted at its door (:mod:`.claude_cli_recorder`); everything after it is the model's
own code.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("langchain_claude_code")
pytest.importorskip("claude_agent_sdk")

from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock  # noqa: E402

from threetears.models import DEFAULT_CHAT_MODEL  # noqa: E402

from .claude_cli_recorder import recording_cli, subscription_model  # noqa: E402
from .translation_helpers import DottedTool  # noqa: E402

#: the XML-attribute-leak tool name from the 2026-05-19 prod incident
_JUNK_NAME = 'memory_recall" name="memory_recall'

#: the prefix the CLI puts on a tool the caller bound
_BOUND = "mcp__langchain-tools__"


def _asks_for(*blocks: ToolUseBlock) -> list[Any]:
    """what the CLI sends when the model asks for tools at one turn: the tool uses, then the turn limit.

    :param blocks: the model's tool-use blocks
    :ptype blocks: ToolUseBlock
    :return: the SDK messages
    :rtype: list[Any]
    """
    return [
        AssistantMessage(content=[TextBlock(text="checking"), *blocks], model=DEFAULT_CHAT_MODEL),
        ResultMessage(
            subtype="error_max_turns", duration_ms=1, duration_api_ms=1, is_error=True, num_turns=2, session_id="s"
        ),
    ]


_VALID = ToolUseBlock(id="toolu_ok", name=f"{_BOUND}threetears_calculator", input={"expression": "2+2"})


@pytest.mark.parametrize(
    "junk_name",
    [_JUNK_NAME, f"{_BOUND}{_JUNK_NAME}"],
    ids=["unbound", "under-the-bound-prefix"],
)
class TestNoJunkCallIsHandedBack:
    """Whether the model wrote the junk name bare or under the bound-tool prefix."""

    @pytest.mark.asyncio
    async def test_ainvoke(self, junk_name: str) -> None:
        junk = ToolUseBlock(id="toolu_junk", name=junk_name, input={"query": "x"})
        with recording_cli(answer=_asks_for(junk, _VALID)):
            result = await subscription_model().bind_tools([DottedTool()]).ainvoke("hi")

        assert [(call["name"], call["id"]) for call in result.tool_calls] == [("threetears.calculator", "toolu_ok")]
        assert result.invalid_tool_calls == []

    @pytest.mark.asyncio
    async def test_astream(self, junk_name: str) -> None:
        junk = ToolUseBlock(id="toolu_junk", name=junk_name, input={"query": "x"})
        with recording_cli(answer=_asks_for(junk, _VALID)):
            chunks = [chunk async for chunk in subscription_model().bind_tools([DottedTool()]).astream("hi")]
        total = chunks[0]
        for chunk in chunks[1:]:
            total = total + chunk

        assert all(fragment["name"] != junk_name for chunk in chunks for fragment in chunk.tool_call_chunks)
        assert [(call["name"], call["id"]) for call in total.tool_calls] == [("threetears.calculator", "toolu_ok")]
        assert total.invalid_tool_calls == []


@pytest.mark.asyncio
async def test_a_turn_that_asked_only_for_a_junk_call_ends_without_one() -> None:
    """The turn still ended to hand a call back, so it is not a failed call; it hands back none."""
    with recording_cli(answer=_asks_for(ToolUseBlock(id="toolu_junk", name=_JUNK_NAME, input={}))):
        result = await subscription_model().bind_tools([DottedTool()]).ainvoke("hi")

    assert result.tool_calls == []
    assert result.content == "checking"
