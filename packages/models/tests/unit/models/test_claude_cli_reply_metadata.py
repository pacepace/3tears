"""What a subscription reply says about itself, invoked or streamed: how it ended, and which model served it.

The CLI's ``ResultMessage`` carries the last model turn's ``stop_reason`` (the Messages API's) and
the query loop's ``terminal_reason``. The model used to drop both and say ``finish_reason="stop"``
for every call that handed no tool calls back, so a reply the output cap cut short read as
finished. Each ``AssistantMessage`` names the model that wrote it, and the result's ``model_usage``
is keyed by model; the model dropped both, so a ``fallback_model`` switch was invisible. The CLI's
answer is scripted at its door (:mod:`.claude_cli_recorder`); everything after it is the model's
own code. The result's ``usage`` carries the thinking share of the output as
``output_tokens_details.thinking_tokens``; the model dropped it, so a subscription reply reported no
reasoning tokens even when the model thought.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("langchain_claude_code")
pytest.importorskip("claude_agent_sdk")

from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock  # noqa: E402

from threetears.models import DEFAULT_CHAT_MODEL, DEFAULT_FAST_MODEL  # noqa: E402

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


_FALLBACK = DEFAULT_FAST_MODEL


def _usage_of(*models: str) -> dict[str, Any]:
    """a result's ``model_usage``, one entry per model, as the CLI sends it.

    :param models: the models the call used
    :ptype models: str
    :return: the per-model usage
    :rtype: dict[str, Any]
    """
    entry = {
        "inputTokens": 1,
        "outputTokens": 1,
        "cacheReadInputTokens": 0,
        "cacheCreationInputTokens": 0,
        "webSearchRequests": 0,
        "costUSD": 0.0,
        "contextWindow": 200000,
        "maxOutputTokens": 32000,
    }
    return {model: dict(entry) for model in models}


@pytest.mark.parametrize("how", _PATHS)
class TestWhichModelServedTheReply:
    async def test_the_model_the_cli_named_on_the_turn_is_the_model_name(self, how: str) -> None:
        reply = await _ask([_reply(model=DEFAULT_CHAT_MODEL), _result()], how)

        assert reply.response_metadata["model_name"] == DEFAULT_CHAT_MODEL

    async def test_a_fallback_switch_is_visible(self, how: str) -> None:
        """asked for the default model, answered by the fallback: the reply names the fallback."""
        answer = [_reply(model=_FALLBACK), _result(model_usage=_usage_of(_FALLBACK))]
        with recording_cli(answer=answer):
            model: Any = subscription_model(fallback_model=_FALLBACK)
            if how == "ainvoke":
                reply = await model.ainvoke("where is tea from?")
            else:
                reply = None
                async for chunk in model.astream("where is tea from?"):
                    reply = chunk if reply is None else reply + chunk

        assert reply.response_metadata["model_name"] == _FALLBACK
        assert reply.response_metadata["model_name"] != DEFAULT_CHAT_MODEL

    async def test_the_last_turn_names_it_when_turns_differ(self, how: str) -> None:
        answer = [_reply("first", model=DEFAULT_CHAT_MODEL), _reply("second", model=_FALLBACK), _result()]
        reply = await _ask(answer, how)

        assert reply.response_metadata["model_name"] == _FALLBACK

    async def test_a_message_the_cli_wrote_itself_names_no_model(self, how: str) -> None:
        answer = [_reply(model=DEFAULT_CHAT_MODEL), _reply("(no content)", model="<synthetic>"), _result()]
        reply = await _ask(answer, how)

        assert reply.response_metadata["model_name"] == DEFAULT_CHAT_MODEL

    async def test_with_no_turn_naming_one_the_results_only_usage_key_names_it(self, how: str) -> None:
        answer = [_reply(model="<synthetic>"), _result(model_usage=_usage_of(_FALLBACK))]
        reply = await _ask(answer, how)

        assert reply.response_metadata["model_name"] == _FALLBACK

    async def test_several_usage_keys_and_no_turn_name_none_never_the_model_asked_for(self, how: str) -> None:
        answer = [_reply(model="<synthetic>"), _result(model_usage=_usage_of(DEFAULT_CHAT_MODEL, _FALLBACK))]
        reply = await _ask(answer, how)

        assert "model_name" not in reply.response_metadata


_USAGE = {"input_tokens": 10, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "output_tokens": 300}


@pytest.mark.parametrize("how", _PATHS)
class TestTheReasoningShareOfTheOutput:
    async def test_the_clis_thinking_tokens_are_the_reasoning_tokens(self, how: str) -> None:
        usage = {**_USAGE, "output_tokens_details": {"thinking_tokens": 120}}
        reply = await _ask([_reply(), _result(usage=usage)], how)

        assert reply.usage_metadata["output_token_details"] == {"reasoning": 120}
        assert reply.usage_metadata["output_tokens"] == 300, "thinking is a share of the output, not added to it"

    async def test_a_count_of_zero_the_cli_sent_is_passed_on(self, how: str) -> None:
        usage = {**_USAGE, "output_tokens_details": {"thinking_tokens": 0}}
        reply = await _ask([_reply(), _result(usage=usage)], how)

        assert reply.usage_metadata["output_token_details"] == {"reasoning": 0}

    @pytest.mark.parametrize(
        "details",
        [None, {}, {"thinking_tokens": None}],
        ids=["no-details", "empty-details", "null-count"],
    )
    async def test_a_cli_that_sends_no_count_reports_none_never_zero(
        self, how: str, details: dict[str, Any] | None
    ) -> None:
        usage = dict(_USAGE) if details is None else {**_USAGE, "output_tokens_details": details}
        reply = await _ask([_reply(), _result(usage=usage)], how)

        assert "output_token_details" not in reply.usage_metadata
        assert reply.usage_metadata["output_tokens"] == 300
