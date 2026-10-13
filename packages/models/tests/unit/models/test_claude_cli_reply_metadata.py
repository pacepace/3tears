"""What a subscription reply says about itself: how it ended, invoked or streamed.

The CLI's ``ResultMessage`` carries the last model turn's ``stop_reason`` (the Messages API's) and
the query loop's ``terminal_reason``. The model used to drop both and say ``finish_reason="stop"``
for every call that handed no tool calls back, so a reply the output cap cut short read as
finished. The CLI's answer is scripted at its door (:mod:`.claude_cli_recorder`); everything after
it is the model's own code.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("langchain_claude_code")
pytest.importorskip("claude_agent_sdk")

from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock  # noqa: E402

from threetears.models import DEFAULT_CHAT_MODEL  # noqa: E402

from .claude_cli_recorder import recording_cli, subscription_model  # noqa: E402

_PATHS = ["ainvoke", "astream"]


def _result(**fields: Any) -> ResultMessage:
    """a result the CLI sends for a call that did not fail.

    :param fields: ``ResultMessage`` fields to set beyond the required ones
    :ptype fields: Any
    :return: the result
    :rtype: ResultMessage
    """
    return ResultMessage(
        subtype="success", duration_ms=1, duration_api_ms=1, is_error=False, num_turns=1, session_id="s", **fields
    )


def _reply(text: str = "Tea is from", *, model: str = DEFAULT_CHAT_MODEL) -> AssistantMessage:
    """one model turn's text, as the CLI sends it.

    :param text: the turn's text
    :ptype text: str
    :param model: the model the CLI names on the turn
    :ptype model: str
    :return: the assistant message
    :rtype: AssistantMessage
    """
    return AssistantMessage(content=[TextBlock(text=text)], model=model)


async def _ask(answer: list[Any], how: str, **bind: Any) -> Any:
    """one call on a subscription model the CLI answers with ``answer``; streams aggregated.

    :param answer: what the CLI sends
    :ptype answer: list[Any]
    :param how: ``ainvoke`` or ``astream``
    :ptype how: str
    :param bind: kwargs bound to the model first
    :ptype bind: Any
    :return: the reply
    :rtype: Any
    """
    with recording_cli(answer=answer):
        model: Any = subscription_model()
        if bind:
            model = model.bind(**bind)
        if how == "ainvoke":
            return await model.ainvoke("where is tea from?")
        merged: Any = None
        async for chunk in model.astream("where is tea from?"):
            merged = chunk if merged is None else merged + chunk
        return merged


@pytest.mark.parametrize("how", _PATHS)
class TestHowTheReplyEnded:
    async def test_a_reply_the_output_cap_cut_short_does_not_read_as_finished(self, how: str) -> None:
        reply = await _ask([_reply(), _result(stop_reason="max_tokens", terminal_reason="completed")], how)

        assert reply.response_metadata["finish_reason"] == "length"
        assert reply.response_metadata["stop_reason"] == "max_tokens"
        assert reply.response_metadata["terminal_reason"] == "completed"

    async def test_a_reply_the_context_window_cut_short_does_not_read_as_finished(self, how: str) -> None:
        reply = await _ask([_reply(), _result(stop_reason="model_context_window_exceeded")], how)

        assert reply.response_metadata["finish_reason"] == "length"

    async def test_a_refusal_reads_as_filtered(self, how: str) -> None:
        reply = await _ask([_reply("I can't help with that."), _result(stop_reason="refusal")], how)

        assert reply.response_metadata["finish_reason"] == "content_filter"

    @pytest.mark.parametrize("stop_reason", ["end_turn", "stop_sequence"])
    async def test_a_normal_end_still_reads_stop(self, how: str, stop_reason: str) -> None:
        reply = await _ask([_reply(), _result(stop_reason=stop_reason, terminal_reason="completed")], how)

        assert reply.response_metadata["finish_reason"] == "stop"
        assert reply.response_metadata["stop_reason"] == stop_reason

    async def test_a_cli_that_reports_neither_reason_still_reads_stop_and_says_it_reported_none(self, how: str) -> None:
        reply = await _ask([_reply(), _result()], how)

        assert reply.response_metadata["finish_reason"] == "stop"
        assert reply.response_metadata["stop_reason"] is None
        assert reply.response_metadata["terminal_reason"] is None

    async def test_a_stop_reason_this_module_does_not_know_is_passed_on_not_read_as_stop(self, how: str) -> None:
        reply = await _ask([_reply(), _result(stop_reason="pause_turn", terminal_reason="completed")], how)

        assert reply.response_metadata["finish_reason"] == "pause_turn"

    async def test_a_loop_that_did_not_complete_says_why(self, how: str) -> None:
        reply = await _ask([_reply(), _result(stop_reason="end_turn", terminal_reason="aborted_streaming")], how)

        assert reply.response_metadata["finish_reason"] == "aborted_streaming"

    async def test_a_structured_answer_reads_stop_though_its_last_turn_was_the_tool_call(self, how: str) -> None:
        """the model answers a schema by calling ``StructuredOutput``; the API calls that turn ``tool_use``."""
        output_config = {"format": {"type": "json_schema", "schema": {"type": "object"}}}
        answer = [_reply("thinking"), _result(stop_reason="tool_use", structured_output={"ok": True})]
        reply = await _ask(answer, how, output_config=output_config)

        assert reply.content == '{"ok": true}'
        assert reply.response_metadata["finish_reason"] == "stop"
        assert reply.response_metadata["stop_reason"] == "tool_use"
