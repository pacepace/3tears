"""A subscription model honours structured output the way the anthropic provider asks for it.

A subscription token resolves to the CLI backend under the SAME provider type, so a caller asks for
a shape with ``structured_output_kwargs("anthropic", ...)`` -- ``output_config`` -- and cannot tell
which backend it holds. The Agent SDK has no ``output_config`` and dropped it without a word: the
CLI never saw the schema, the model answered in prose, and every structured call on a subscription
model failed (found live, 2026-09-21: every turn on a subscription decision model refused).

Only the SDK subprocess boundary is faked. What the fake sends is what the CLI was measured to send
at ``--max-turns 1`` with ``--json-schema``: the model's prose, a ``StructuredOutput`` tool use, and a
``success`` result carrying ``structured_output``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage

pytest.importorskip("langchain_claude_code")
pytest.importorskip("claude_agent_sdk")

from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock  # noqa: E402

from threetears.models import DEFAULT_CHAT_MODEL  # noqa: E402
from threetears.models.claude_cli_pool import launch_key  # noqa: E402
from threetears.models.providers._claude_cli import create_subscription_chat  # noqa: E402
from threetears.models.providers.structured_output import structured_output_kwargs  # noqa: E402

_TOKEN = "sk-ant-oat01-faketokenfortest"

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"ask": {"type": "string"}, "kept_back": {"type": "array", "items": {"type": "string"}}},
    "required": ["ask", "kept_back"],
    "additionalProperties": False,
}

_ANSWER = {"ask": "What's the weather in Dublin tomorrow?", "kept_back": ["hey love, how are you?"]}

_PROSE = "I appreciate the warm greeting! However, I don't have access to real-time weather data."


@pytest.fixture(autouse=True)
def _one_off_cli_per_call() -> Any:
    """The one-off client path, chosen explicitly; the pooled path is pinned in its own module."""
    from threetears.models import claude_cli_pool

    claude_cli_pool.configure_claude_cli_pool(enabled=False)
    yield
    claude_cli_pool.configure_claude_cli_pool(enabled=True)


def _tripwire_init(self: Any, *_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("the REAL ClaudeSDKClient.__init__ ran; a patch binding was missed")


# parity-exempt: implements only the calls this backend makes -- see test_claude_cli_tool_calls_handed_back.py
class _FakeSDKClient:
    """Stands in for ``ClaudeSDKClient``: records the options, answers with ``replies``."""

    replies: list[Any] = []
    options: list[Any] = []

    def __init__(self, options: Any) -> None:
        _FakeSDKClient.options.append(options)

    async def __aenter__(self) -> "_FakeSDKClient":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None

    async def query(self, prompt: str) -> None:
        return None

    async def receive_response(self) -> AsyncIterator[Any]:
        for msg in _FakeSDKClient.replies:
            yield msg


@contextmanager
def _no_real_sdk_calls():
    with (
        patch("claude_agent_sdk.ClaudeSDKClient", _FakeSDKClient),
        patch("langchain_claude_code.claude_chat_model.ClaudeSDKClient", _FakeSDKClient),
        patch("claude_agent_sdk.client.ClaudeSDKClient.__init__", _tripwire_init),
    ):
        yield


@pytest.fixture(autouse=True)
def _fresh_fake() -> Any:
    _FakeSDKClient.replies = []
    _FakeSDKClient.options = []
    yield


def _cli_answers(structured_output: Any) -> list[Any]:
    """What the CLI sends for a structured call, as measured: prose, the tool use, the result."""
    return [
        AssistantMessage(content=[TextBlock(text=_PROSE)], model=DEFAULT_CHAT_MODEL),
        AssistantMessage(
            content=[ToolUseBlock(id="toolu_1", name="StructuredOutput", input=dict(structured_output or {}))],
            model=DEFAULT_CHAT_MODEL,
        ),
        ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=2,
            session_id="fake-session",
            structured_output=structured_output,
        ),
    ]


def _bound(model: Any) -> Any:
    """Bound exactly as a caller binds any anthropic model for a shape."""
    return model.bind(**structured_output_kwargs("anthropic", _SCHEMA, name="in_split"))


async def test_the_schema_reaches_the_cli() -> None:
    with _no_real_sdk_calls():
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, _TOKEN)
        _FakeSDKClient.replies = _cli_answers(_ANSWER)
        await _bound(model).ainvoke([HumanMessage(content="hi")])

    output_format = _FakeSDKClient.options[0].output_format
    assert output_format is not None, "the schema was dropped before it reached the CLI"
    assert output_format["type"] == "json_schema"
    assert output_format["schema"]["properties"].keys() == _SCHEMA["properties"].keys()


async def test_the_answer_is_the_structured_output_not_the_prose() -> None:
    with _no_real_sdk_calls():
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, _TOKEN)
        _FakeSDKClient.replies = _cli_answers(_ANSWER)
        message = await _bound(model).ainvoke([HumanMessage(content="hi")])

    import json

    assert json.loads(message.content) == _ANSWER
    assert message.tool_calls == [], "the CLI's own StructuredOutput call is not the caller's to run"
    assert message.response_metadata["finish_reason"] == "stop"
    assert message.response_metadata["is_error"] is False


async def test_streaming_yields_the_structured_output_and_none_of_the_prose() -> None:
    with _no_real_sdk_calls():
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, _TOKEN)
        _FakeSDKClient.replies = _cli_answers(_ANSWER)
        chunks = [chunk async for chunk in _bound(model).astream([HumanMessage(content="hi")])]

    import json

    merged = chunks[0]
    for chunk in chunks[1:]:
        merged = merged + chunk
    assert json.loads(merged.content) == _ANSWER
    assert merged.tool_calls == []


@pytest.mark.parametrize("streamed", [False, True])
async def test_with_no_structured_answer_the_caller_gets_the_prose(streamed: bool) -> None:
    """So the caller's failure names what the model said, instead of parsing an empty string."""
    with _no_real_sdk_calls():
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, _TOKEN)
        _FakeSDKClient.replies = _cli_answers(None)
        if streamed:
            merged: Any = None
            async for chunk in _bound(model).astream([HumanMessage(content="hi")]):
                merged = chunk if merged is None else merged + chunk
            content = merged.content
        else:
            content = (await _bound(model).ainvoke([HumanMessage(content="hi")])).content

    assert content == _PROSE


async def test_an_output_config_it_cannot_honour_is_refused_not_dropped() -> None:
    with _no_real_sdk_calls():
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, _TOKEN)
        _FakeSDKClient.replies = _cli_answers(_ANSWER)
        with pytest.raises(ValueError, match="json_schema"):
            await model.bind(output_config={"effort": "low"}).ainvoke([HumanMessage(content="hi")])


async def test_a_key_the_sdk_has_no_option_for_is_named() -> None:
    with _no_real_sdk_calls(), patch("threetears.models.providers._claude_cli._logger") as logger:
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, _TOKEN)
        _FakeSDKClient.replies = _cli_answers(_ANSWER)
        await model.bind(temperature=0.2).ainvoke([HumanMessage(content="hi")])

    [call] = logger.warning.call_args_list
    assert call.kwargs["extra"]["extra_data"]["dropped"] == ["temperature"]


async def test_a_structured_call_never_shares_a_cli_launched_without_its_schema() -> None:
    """``--json-schema`` is a launch flag, so a pooled CLI started without it cannot answer with it."""
    with _no_real_sdk_calls():
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, _TOKEN)
        _FakeSDKClient.replies = _cli_answers(_ANSWER)
        await model.ainvoke([HumanMessage(content="hi")])
        _FakeSDKClient.replies = _cli_answers(_ANSWER)
        await _bound(model).ainvoke([HumanMessage(content="hi")])

    plain, structured = _FakeSDKClient.options
    assert launch_key(plain, _TOKEN) != launch_key(structured, _TOKEN)
