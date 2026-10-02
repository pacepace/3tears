"""An OpenRouter call's deadline limits silence, not length, on the non-streamed path too.

``_agenerate`` held a plain ``ainvoke`` to the deadline as a whole-call limit ("no answer within
120.0 s"), while a streamed call was limited only by the gap between chunks. A reasoning model
writing for longer than the deadline was cut off: memory extraction, resolution and dream all call
``ainvoke``. With a deadline, ``_agenerate`` now collects the answer from the stream.

The equality tests drive the real ``openrouter`` SDK over an ``httpx.MockTransport``: the same
answer, sent once as a JSON response (the old path, no deadline) and once as server-sent events
(the new path), must come back the same.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from langchain_core.messages import HumanMessage

from threetears.models.errors import ModelCallTimeout
from threetears.models.providers.openrouter import create_openrouter_chat

from .provider_wire import ChatCompletionsWire, openrouter_model, text_deltas

_MODEL = "deepseek/deepseek-v4-pro"
_SERVED_BY = "DeepInfra"
_BASE = {"id": "gen-1790000000-abc", "provider": _SERVED_BY, "model": _MODEL, "created": 1790000000}
_USAGE = {"prompt_tokens": 40, "completion_tokens": 9, "total_tokens": 49}
_CALL = {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"q": "tides"}'}}
_SCHEMA = {
    "title": "Verdict",
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def _response(message: dict[str, Any], finish: str) -> dict[str, Any]:
    """one answer as OpenRouter's non-streamed JSON response."""
    return {
        **_BASE,
        "object": "chat.completion",
        "system_fingerprint": None,
        "choices": [{"index": 0, "finish_reason": finish, "logprobs": None, "message": message}],
        "usage": _USAGE,
    }


def _events(deltas: list[dict[str, Any]], finish: str) -> bytes:
    """the same answer as server-sent events: one chunk per delta, the last one finishing, then usage."""
    chunks = [
        {
            **_BASE,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish if i == len(deltas) - 1 else None}],
        }
        for i, delta in enumerate(deltas)
    ]
    chunks.append({**_BASE, "object": "chat.completion.chunk", "choices": [], "usage": _USAGE})
    return ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()


_ANSWERS: dict[str, tuple[dict[str, Any], list[dict[str, Any]], str]] = {
    "plain": (
        {"role": "assistant", "content": "High tide is at 6."},
        [{"role": "assistant", "content": "High tide "}, {"content": "is at 6."}],
        "stop",
    ),
    "tool call": (
        {"role": "assistant", "content": "", "tool_calls": [_CALL]},
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"index": 0, "id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": ""}}
                ],
            },
            {"tool_calls": [{"index": 0, "function": {"arguments": '{"q": '}}]},
            {"tool_calls": [{"index": 0, "function": {"arguments": '"tides"}'}}]},
        ],
        "tool_calls",
    ),
    "structured": (
        {"role": "assistant", "content": '{"answer": "yes"}'},
        [{"role": "assistant", "content": '{"answer": '}, {"content": '"yes"}'}],
        "stop",
    ),
}


class _Wire:
    """OpenRouter's end of the SDK: answers JSON or events, as the request asks."""

    def __init__(self, answer: str) -> None:
        self._message, self._deltas, self._finish = _ANSWERS[answer]
        self.streamed: list[bool] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.streamed.append(bool(body.get("stream")))
        if body.get("stream"):
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=_events(self._deltas, self._finish)
            )
        return httpx.Response(200, json=_response(self._message, self._finish))


def _model(wire: _Wire, *, deadline_s: int | None) -> Any:
    import openrouter

    transport = httpx.MockTransport(wire.handle)
    sdk = openrouter.OpenRouter(
        api_key="sk-or-test",
        client=httpx.Client(transport=transport),
        async_client=httpx.AsyncClient(transport=transport),
    )
    model = create_openrouter_chat(_MODEL, "sk-or-test", max_retries=0, client=sdk)
    model.request_timeout = deadline_s * 1000 if deadline_s else None
    return model


def _tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "look it up",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        },
    }


async def _ask(answer: str, *, deadline_s: int | None) -> tuple[Any, list[bool]]:
    wire = _Wire(answer)
    model = _model(wire, deadline_s=deadline_s)
    if answer == "tool call":
        result = await model.bind_tools([_tool()]).ainvoke("when is high tide?")
    elif answer == "structured":
        result = await model.with_structured_output(_SCHEMA, method="json_schema").ainvoke("is it?")
    else:
        result = await model.ainvoke("when is high tide?")
    return result, wire.streamed


#: response_metadata keys that name the wire format itself: ``object`` is the SDK's own record
#: type (``chat.completion`` vs ``chat.completion.chunk``), and only the JSON response carries a
#: ``logprobs`` of None.
_WIRE_FORMAT_KEYS = {"object", "logprobs"}


class TestTheStreamedAnswerIsTheSameAnswer:
    @pytest.mark.parametrize("answer", ["plain", "tool call"])
    async def test_the_message_matches_the_one_the_json_path_returned(self, answer: str) -> None:
        before, before_wire = await _ask(answer, deadline_s=None)
        after, after_wire = await _ask(answer, deadline_s=120)

        assert (before_wire, after_wire) == ([False], [True])
        assert after.content == before.content
        assert after.tool_calls == before.tool_calls
        assert after.invalid_tool_calls == before.invalid_tool_calls
        assert after.usage_metadata == before.usage_metadata
        assert after.response_metadata["provider"] == _SERVED_BY
        assert after.response_metadata["finish_reason"] == before.response_metadata["finish_reason"]
        strip = lambda meta: {k: v for k, v in meta.items() if k not in _WIRE_FORMAT_KEYS}  # noqa: E731
        assert strip(after.response_metadata) == strip(before.response_metadata)

    async def test_a_structured_answer_parses_the_same(self) -> None:
        before, _ = await _ask("structured", deadline_s=None)
        after, after_wire = await _ask("structured", deadline_s=120)
        assert after_wire == [True]
        assert after == before == {"answer": "yes"}


class TestTheDeadlineIsOnSilence:
    """the deadline measured against a real stream's pacing, served by a scripted OpenRouter.

    the wire paces the server-sent events itself, so the whole stack under the mixin -- the SDK's
    event parsing, LangChain's chunking -- runs as it does live, and only the timing is scripted.
    """

    @staticmethod
    def _deadline_model(wire: ChatCompletionsWire, deadline_ms: int) -> Any:
        model = openrouter_model(wire, _MODEL)
        model.request_timeout = deadline_ms
        return model

    async def test_an_answer_longer_than_the_deadline_with_no_long_gap_arrives(self) -> None:
        # 300 ms of answer against a 150 ms deadline; no gap is longer than 30 ms
        wire = ChatCompletionsWire(deltas=text_deltas(*(str(i) for i in range(10))), gap_s=0.03)
        answer = await self._deadline_model(wire, 150).ainvoke([HumanMessage(content="hi")])
        assert answer.content == "0123456789"

    async def test_an_answer_that_goes_quiet_for_the_deadline_times_out(self) -> None:
        wire = ChatCompletionsWire(deltas=text_deltas("Hel", "lo"), silence_after=1, silence_s=5.0)
        with pytest.raises(ModelCallTimeout, match="no chunk within 0.05 s"):
            await self._deadline_model(wire, 50).ainvoke([HumanMessage(content="hi")])

    async def test_an_answer_that_never_starts_times_out(self) -> None:
        wire = ChatCompletionsWire(deltas=text_deltas("late"), silence_after=0, silence_s=5.0)
        with pytest.raises(ModelCallTimeout):
            await self._deadline_model(wire, 50).ainvoke([HumanMessage(content="hi")])
