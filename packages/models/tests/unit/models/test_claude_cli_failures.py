"""A subscription call that fails raises, and never answers with the failure's text.

The CLI reports a failure as data: at the subscription's usage limit it sends a synthetic
assistant message carrying ``error="rate_limit"`` and the notice as its text ("You've hit your
session limit · resets 1:10am (UTC)"), then a result with ``is_error``. The model returned that
notice as ordinary ``AIMessage`` content, so metallm stored it as a draft and handed it to its
agent as knowledge, and the circuit breaker counted the call a success.

A failure now raises :class:`~threetears.models.errors.ModelRateLimitError` (a limit, with its reset
time when the notice gives one) or :class:`~threetears.models.errors.ModelProviderError` (anything
else), before any of its text is yielded. A call that asked for tools still ends on
``error_max_turns`` without failing: that is the designed end of such a call.

Only the Claude Agent SDK subprocess boundary is faked, with what the CLI sends.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("langchain_claude_code")
pytest.importorskip("claude_agent_sdk")

from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.tools import BaseTool  # noqa: E402

from threetears.models import DEFAULT_CHAT_MODEL, claude_cli_pool  # noqa: E402
from threetears.models.circuit_breaker import CircuitBreaker, CircuitState  # noqa: E402
from threetears.models.errors import ModelProviderError, ModelRateLimitError  # noqa: E402
from threetears.models.factory import create_chat_model  # noqa: E402
from threetears.models.providers._claude_cli import create_subscription_chat  # noqa: E402

TOKEN = "sk-ant-oat01-faketokenfortest"
_NOTICE = "You've hit your session limit · resets 1:10am (UTC)"


@pytest.fixture(autouse=True)
def _one_off_cli_per_call() -> Iterator[None]:
    """drive the one-off-client path, which the fake client stands in for."""
    claude_cli_pool.configure_claude_cli_pool(enabled=False)
    yield
    claude_cli_pool.configure_claude_cli_pool(enabled=True)


# parity-exempt: stands in for ClaudeSDKClient's query/receive_response only, driven by a per-test script
class _FakeSDKClient:
    """Stands in for ``claude_agent_sdk.ClaudeSDKClient``: answers each call from ``script``."""

    script: Callable[[], list[Any]] | None = None

    def __init__(self, options: Any) -> None:
        self._options = options

    async def __aenter__(self) -> _FakeSDKClient:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None

    async def query(self, prompt: str) -> None:
        return None

    async def receive_response(self) -> AsyncIterator[Any]:
        assert _FakeSDKClient.script is not None, "test must set _FakeSDKClient.script"
        for msg in _FakeSDKClient.script():
            yield msg


def _tripwire_init(self: Any, *_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("the REAL ClaudeSDKClient ran; a patch binding was missed")


@contextmanager
def _fake_cli(*messages: Any) -> Iterator[None]:
    """every binding of the SDK client patched to answer with ``messages``, for as long as the block runs.

    :param messages: what the CLI sends for each call
    :ptype messages: Any
    :return: a context in which no real CLI can start
    :rtype: Iterator[None]
    """
    _FakeSDKClient.script = lambda: list(messages)
    with (
        patch("claude_agent_sdk.ClaudeSDKClient", _FakeSDKClient),
        patch("langchain_claude_code.claude_chat_model.ClaudeSDKClient", _FakeSDKClient),
        patch("claude_agent_sdk.client.ClaudeSDKClient.__init__", _tripwire_init),
    ):
        yield
    _FakeSDKClient.script = None


def _assistant(text: str, *, error: Any = None, blocks: list[Any] | None = None) -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock(text=text), *(blocks or [])] if text else list(blocks or []),
        model=DEFAULT_CHAT_MODEL,
        error=error,
    )


def _result(
    *,
    is_error: bool = False,
    subtype: str = "success",
    result: str | None = None,
    status: int | None = None,
    structured_output: Any = None,
) -> ResultMessage:
    return ResultMessage(
        subtype=subtype,
        duration_ms=1,
        duration_api_ms=1,
        is_error=is_error,
        num_turns=1,
        session_id="fake-session",
        result=result,
        api_error_status=status,
        structured_output=structured_output,
    )


def _session_limit() -> tuple[Any, ...]:
    """what the CLI sends at the subscription's usage limit.

    :return: the synthetic assistant message, then the error result
    :rtype: tuple[Any, ...]
    """
    return (_assistant(_NOTICE, error="rate_limit"), _result(is_error=True, result=_NOTICE, status=429))


async def _streamed(model: Any, received: list[Any]) -> None:
    """stream one call, keeping every chunk that arrives before anything raises.

    :param model: the model
    :ptype model: Any
    :param received: where each chunk goes
    :ptype received: list[Any]
    :return: nothing
    :rtype: None
    """
    async for chunk in model.astream([HumanMessage(content="hi")]):
        received.append(chunk)


class TestTheSessionLimit:
    async def test_invoking_raises_a_rate_limit_carrying_the_reset(self) -> None:
        with _fake_cli(*_session_limit()):
            model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
            with pytest.raises(ModelRateLimitError) as raised:
                await model.ainvoke([HumanMessage(content="hi")])

        assert raised.value.resets == "1:10am (UTC)"
        assert raised.value.detail == _NOTICE
        assert raised.value.reason == "rate_limit"

    async def test_streaming_raises_before_the_notice_reaches_the_person(self) -> None:
        received: list[Any] = []
        with _fake_cli(*_session_limit()):
            model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
            with pytest.raises(ModelRateLimitError) as raised:
                await _streamed(model, received)

        assert raised.value.resets == "1:10am (UTC)"
        assert received == [], "a chunk was yielded for a call that failed"

    async def test_a_limit_reported_only_on_the_result_is_still_a_rate_limit(self) -> None:
        """a CLI that sends no flagged assistant message still says 429 on the result."""
        with _fake_cli(_assistant(_NOTICE), _result(is_error=True, result=_NOTICE, status=429)):
            model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
            with pytest.raises(ModelRateLimitError) as invoked:
                await model.ainvoke([HumanMessage(content="hi")])
            with pytest.raises(ModelRateLimitError):
                await _streamed(model, [])

        assert invoked.value.resets == "1:10am (UTC)"
        assert invoked.value.status == 429


class TestOtherFailures:
    async def test_another_flagged_assistant_message_is_a_provider_failure_not_a_limit(self) -> None:
        received: list[Any] = []
        messages = (_assistant("API Error: 500", error="server_error"), _result(is_error=True, result="API Error"))
        with _fake_cli(*messages):
            model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
            with pytest.raises(ModelProviderError) as invoked:
                await model.ainvoke([HumanMessage(content="hi")])
            with pytest.raises(ModelProviderError) as streamed:
                await _streamed(model, received)

        for raised in (invoked.value, streamed.value):
            assert not isinstance(raised, ModelRateLimitError)
            assert raised.reason == "server_error"
            assert raised.detail == "API Error: 500"
        assert received == []

    async def test_an_error_result_raises_with_what_the_result_said(self) -> None:
        messages = (_assistant("partial"), _result(is_error=True, subtype="error_during_execution", status=500))
        with _fake_cli(*messages):
            model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
            with pytest.raises(ModelProviderError) as invoked:
                await model.ainvoke([HumanMessage(content="hi")])

        assert not isinstance(invoked.value, ModelRateLimitError)
        assert invoked.value.reason == "error_during_execution"
        assert invoked.value.status == 500

    async def test_the_turn_limit_without_a_tool_call_is_a_failure(self) -> None:
        with _fake_cli(_assistant("hm"), _result(is_error=True, subtype="error_max_turns")):
            model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
            with pytest.raises(ModelProviderError):
                await model.ainvoke([HumanMessage(content="hi")])
            with pytest.raises(ModelProviderError):
                await _streamed(model, [])


class _Write(BaseTool):
    """a bound tool the model asks for."""

    name: str = "threetears.write"
    description: str = "writes"

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError


class TestWhatIsNotAFailure:
    async def test_a_call_that_asked_for_tools_still_ends_on_the_turn_limit_and_hands_them_back(self) -> None:
        asks = _assistant(
            "", blocks=[ToolUseBlock(id="tu-1", name="mcp__langchain-tools__threetears_write", input={"p": "a"})]
        )
        with _fake_cli(asks, _result(is_error=True, subtype="error_max_turns")):
            bound = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN).bind_tools([_Write()])
            invoked = await bound.ainvoke([HumanMessage(content="write a")])
            merged: Any = None
            async for chunk in bound.astream([HumanMessage(content="write a")]):
                merged = chunk if merged is None else merged + chunk

        for message in (invoked, merged):
            assert [(c["name"], c["args"]) for c in message.tool_calls] == [("threetears.write", {"p": "a"})]
            assert message.response_metadata["finish_reason"] == "tool_calls"

    async def test_a_normal_result_is_unchanged(self) -> None:
        with _fake_cli(_assistant("Tea is from China."), _result()):
            model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
            invoked = await model.ainvoke([HumanMessage(content="hi")])
            streamed: list[Any] = []
            await _streamed(model, streamed)

        assert invoked.content == "Tea is from China."
        assert invoked.response_metadata["is_error"] is False
        assert invoked.response_metadata["finish_reason"] == "stop"
        assert "".join(str(c.content) for c in streamed) == "Tea is from China."

    async def test_a_structured_answer_is_the_answer_whatever_the_result_flag_says(self) -> None:
        """the CLI delivered the shape asked for; raising would throw the answer away."""
        output_config = {"format": {"type": "json_schema", "schema": {"type": "object"}}}
        with _fake_cli(_assistant("thinking"), _result(is_error=True, structured_output={"ok": True})):
            model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN).bind(output_config=output_config)
            invoked = await model.ainvoke([HumanMessage(content="hi")])

        assert invoked.content == '{"ok": true}'


class TestTheCircuitBreaker:
    """the breaker the factory attaches sees a failed subscription call as it sees a failed API call."""

    async def test_a_session_limit_counts_as_a_failure(self) -> None:
        breaker = CircuitBreaker("anthropic", failure_threshold=1)
        with _fake_cli(*_session_limit()):
            model = create_chat_model(DEFAULT_CHAT_MODEL, api_key=TOKEN, breaker=breaker)
            with pytest.raises(ModelRateLimitError):
                await model.ainvoke([HumanMessage(content="hi")])

        assert breaker.failure_count == 1
        assert breaker.state is CircuitState.OPEN

    async def test_a_streamed_session_limit_counts_as_a_failure(self) -> None:
        breaker = CircuitBreaker("anthropic", failure_threshold=1)
        with _fake_cli(*_session_limit()):
            model = create_chat_model(DEFAULT_CHAT_MODEL, api_key=TOKEN, breaker=breaker)
            with pytest.raises(ModelRateLimitError):
                async for _chunk in model.astream([HumanMessage(content="hi")]):
                    pass

        assert breaker.state is CircuitState.OPEN

    async def test_a_normal_answer_counts_as_a_success(self) -> None:
        breaker = CircuitBreaker("anthropic", failure_threshold=2)
        breaker.record_failure()
        with _fake_cli(_assistant("fine"), _result()):
            model = create_chat_model(DEFAULT_CHAT_MODEL, api_key=TOKEN, breaker=breaker)
            await model.ainvoke([HumanMessage(content="hi")])

        assert breaker.failure_count == 0
