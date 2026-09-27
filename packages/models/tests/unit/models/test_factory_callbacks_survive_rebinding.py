"""The callbacks ``create_chat_model`` attaches still run after a caller re-binds the model.

The factory attaches its usage tracker and circuit breaker with ``with_config(callbacks=...)``,
which returns a ``RunnableBinding``. ``bind_tools`` and ``with_structured_output`` are the chat
model's own methods, reached through the binding's attribute proxy, and they return a new runnable
built from the bare model: the callbacks were left behind. So on every tool-bound or structured
call -- the calls an agent makes -- nothing was metered and the breaker never saw a failure
(metallm confirmed it with a probe).

These drive real provider models at their transport: the Anthropic API route through an
``httpx`` mock transport, the subscription route through a faked Agent SDK client.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from anthropic import InternalServerError
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from threetears.models import DEFAULT_CHAT_MODEL
from threetears.models.circuit_breaker import CircuitBreaker
from threetears.models.factory import create_chat_model
from threetears.models.tracking import UsageAuditSink, UsageRecord, UsageTracker

API_KEY = "sk-ant-api03-faketestkey"


class _Sink(UsageAuditSink):
    """keeps every usage record the tracker hands over."""

    def __init__(self) -> None:
        self.records: list[UsageRecord] = []

    async def record(self, record: UsageRecord) -> None:
        self.records.append(record)


class _Answer(BaseModel):
    """the shape a structured call asks for."""

    text: str


class _Write(BaseTool):
    """a tool a caller binds."""

    name: str = "threetears.write"
    description: str = "writes a file"

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError


def _message(content: list[dict[str, Any]], stop_reason: str) -> dict[str, Any]:
    """an Anthropic Messages API response body.

    :param content: the content blocks
    :ptype content: list[dict[str, Any]]
    :param stop_reason: why the model stopped
    :ptype stop_reason: str
    :return: the response body
    :rtype: dict[str, Any]
    """
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": DEFAULT_CHAT_MODEL,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 11, "output_tokens": 7},
    }


@contextmanager
def _anthropic_answers(*responses: httpx.Response) -> Iterator[list[dict[str, Any]]]:
    """the Anthropic API answering each request with the next response.

    :param responses: the responses, in order
    :ptype responses: httpx.Response
    :return: the request bodies the API received
    :rtype: Iterator[list[dict[str, Any]]]
    """
    queue = list(responses)
    received: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(json.loads(request.content))
        return queue.pop(0)

    def client(*, base_url: str | None, **_kwargs: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=base_url or "https://api.anthropic.com", transport=httpx.MockTransport(handler)
        )

    with patch("langchain_anthropic.chat_models._get_default_async_httpx_client", client):
        yield received


def _instrumented(
    *, provider_kwargs: dict[str, Any] | None = None, api_key: str = API_KEY
) -> tuple[Any, CircuitBreaker, _Sink]:
    """a factory-built model with a breaker and a tracker the test can read.

    :param provider_kwargs: extra provider arguments
    :ptype provider_kwargs: dict[str, Any] | None
    :param api_key: the credential
    :ptype api_key: str
    :return: the model, its breaker and its tracker's sink
    :rtype: tuple[Any, CircuitBreaker, _Sink]
    """
    breaker = CircuitBreaker("anthropic", failure_threshold=10)
    sink = _Sink()
    model = create_chat_model(
        DEFAULT_CHAT_MODEL,
        api_key=api_key,
        breaker=breaker,
        tracker=UsageTracker(audit_sink=sink),
        **(provider_kwargs or {"max_retries": 0}),
    )
    return model, breaker, sink


async def _settle() -> None:
    """the tracker hands records to its sinks on background tasks."""
    for _ in range(5):
        await asyncio.sleep(0)


_SERVER_ERROR = httpx.Response(500, json={"type": "error", "error": {"type": "api_error", "message": "boom"}})


class TestTheApiRoute:
    async def test_a_tool_bound_call_is_metered_and_its_failure_reaches_the_breaker(self) -> None:
        model, breaker, sink = _instrumented()
        bound = model.bind_tools([_Write()])
        tool_use = _message(
            [{"type": "tool_use", "id": "toolu_1", "name": "threetears_write", "input": {}}], "tool_use"
        )
        with _anthropic_answers(_SERVER_ERROR, httpx.Response(200, json=tool_use)):
            with pytest.raises(InternalServerError):
                await bound.ainvoke([HumanMessage(content="write it")])
            assert breaker.failure_count == 1, "the breaker never saw the tool-bound call fail"

            message = await bound.ainvoke([HumanMessage(content="write it")])
        await _settle()

        assert [call["name"] for call in message.tool_calls] == ["threetears.write"]
        assert breaker.failure_count == 0, "the breaker never saw the tool-bound call succeed"
        assert [(r.input_tokens, r.output_tokens) for r in sink.records] == [(11, 7)]

    async def test_a_structured_call_is_metered_and_its_failure_reaches_the_breaker(self) -> None:
        model, breaker, sink = _instrumented()
        structured = model.with_structured_output(_Answer)
        answer = _message(
            [{"type": "tool_use", "id": "toolu_1", "name": "_Answer", "input": {"text": "hi"}}], "tool_use"
        )
        with _anthropic_answers(_SERVER_ERROR, httpx.Response(200, json=answer)):
            with pytest.raises(InternalServerError):
                await structured.ainvoke([HumanMessage(content="answer")])
            assert breaker.failure_count == 1, "the breaker never saw the structured call fail"

            parsed = await structured.ainvoke([HumanMessage(content="answer")])
        await _settle()

        assert parsed == _Answer(text="hi")
        assert breaker.failure_count == 0
        assert len(sink.records) == 1

    async def test_a_bound_then_rebound_model_keeps_them_too(self) -> None:
        model, breaker, _sink = _instrumented()
        bound = model.bind_tools([_Write()]).bind(temperature=0)
        with _anthropic_answers(_SERVER_ERROR):
            with pytest.raises(InternalServerError):
                await bound.ainvoke([HumanMessage(content="write it")])

        assert breaker.failure_count == 1

    async def test_arguments_bound_before_the_tools_still_reach_the_provider(self) -> None:
        model, breaker, _sink = _instrumented()
        bound = model.bind(max_tokens=77).bind_tools([_Write()])
        text = _message([{"type": "text", "text": "hi"}], "end_turn")
        with _anthropic_answers(httpx.Response(200, json=text)) as received:
            await bound.ainvoke([HumanMessage(content="hi")])

        assert received[0]["max_tokens"] == 77
        assert [tool["name"] for tool in received[0]["tools"]] == ["threetears_write"]

    async def test_callbacks_a_caller_passes_run_beside_them(self) -> None:
        from langchain_core.callbacks import AsyncCallbackHandler

        seen: list[str] = []

        class _Spy(AsyncCallbackHandler):
            async def on_llm_end(self, *args: Any, **kwargs: Any) -> None:
                seen.append("caller")

        model, _breaker, sink = _instrumented()
        text = _message([{"type": "text", "text": "hi"}], "end_turn")
        with _anthropic_answers(httpx.Response(200, json=text)):
            await model.bind_tools([_Write()]).ainvoke([HumanMessage(content="hi")], config={"callbacks": [_Spy()]})
        await _settle()

        assert seen == ["caller"]
        assert len(sink.records) == 1

    async def test_callbacks_added_with_attach_callbacks_survive_binding_tools_too(self) -> None:
        from langchain_core.callbacks import AsyncCallbackHandler

        from threetears.models import attach_callbacks

        seen: list[str] = []

        class _Spy(AsyncCallbackHandler):
            async def on_llm_end(self, *args: Any, **kwargs: Any) -> None:
                seen.append("attached")

        model, _breaker, sink = _instrumented()
        text = _message([{"type": "text", "text": "hi"}], "end_turn")
        with _anthropic_answers(httpx.Response(200, json=text)):
            await attach_callbacks(model, _Spy()).bind_tools([_Write()]).ainvoke([HumanMessage(content="hi")])
        await _settle()

        assert seen == ["attached"]
        assert len(sink.records) == 1


class TestTheSubscriptionRoute:
    """the route metallm hit: a session limit on a tool-bound call must trip the breaker."""

    @pytest.fixture(autouse=True)
    def _one_off_cli(self) -> Iterator[None]:
        pytest.importorskip("claude_agent_sdk")
        from threetears.models import claude_cli_pool

        claude_cli_pool.configure_claude_cli_pool(enabled=False)
        yield
        claude_cli_pool.configure_claude_cli_pool(enabled=True)

    async def test_a_session_limit_on_a_tool_bound_call_reaches_the_breaker(self) -> None:
        pytest.importorskip("langchain_claude_code")
        from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

        from threetears.models.errors import ModelRateLimitError

        notice = "You've hit your session limit · resets 1:10am (UTC)"
        script = [
            AssistantMessage(content=[TextBlock(text=notice)], model=DEFAULT_CHAT_MODEL, error="rate_limit"),
            ResultMessage(
                subtype="success",
                duration_ms=1,
                duration_api_ms=1,
                is_error=True,
                num_turns=1,
                session_id="s",
                result=notice,
                api_error_status=429,
            ),
        ]

        # parity-exempt: stands in for ClaudeSDKClient's query/receive_response only, answering from one script
        class _FakeSDKClient:
            def __init__(self, options: Any) -> None:
                self.options = options

            async def __aenter__(self) -> _FakeSDKClient:
                return self

            async def __aexit__(self, *_exc: Any) -> None:
                return None

            async def query(self, prompt: str) -> None:
                return None

            async def receive_response(self) -> AsyncIterator[Any]:
                for msg in script:
                    yield msg

        with (
            patch("claude_agent_sdk.ClaudeSDKClient", _FakeSDKClient),
            patch("langchain_claude_code.claude_chat_model.ClaudeSDKClient", _FakeSDKClient),
        ):
            model, breaker, _sink = _instrumented(provider_kwargs={}, api_key="sk-ant-oat01-faketokenfortest")
            bound = model.bind_tools([_Write()])
            with pytest.raises(ModelRateLimitError):
                await bound.ainvoke([HumanMessage(content="write it")])

        assert breaker.failure_count == 1
