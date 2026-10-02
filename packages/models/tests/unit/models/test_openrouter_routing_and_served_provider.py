"""OpenRouter at the wire: a structured ask keeps the model's routing, and the served provider is read.

Every test drives the real ``openrouter`` SDK over an ``httpx.MockTransport``: the request body is
the one the SDK serialised, and the response is parsed by the SDK's own ``ChatResult`` /
``ChatStreamChunk`` models, from the shape OpenRouter sends (``provider`` names the upstream that
served the call, on the response and on every stream chunk).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from langchain_core.messages import HumanMessage
import pytest

from threetears.models.providers.openrouter import create_openrouter_chat
from threetears.models.providers.structured_output import structured_output_kwargs
from threetears.models.tracking import UsageTracker, UsageTrackingCallback

_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
_ROUTING = {"only": ["anthropic", "amazon-bedrock"], "ignore": ["deepinfra"]}
_SERVED_BY = "Amazon Bedrock"
_MODEL = "anthropic/claude-sonnet-4.5"


def _response() -> dict[str, Any]:
    """a non-streamed OpenRouter chat response, as the API sends it.

    :return: the response body
    :rtype: dict[str, Any]
    """
    return {
        "id": "gen-1758000000-abc",
        "provider": _SERVED_BY,
        "model": _MODEL,
        "object": "chat.completion",
        "created": 1758000000,
        "system_fingerprint": None,
        "choices": [
            {
                "logprobs": None,
                "finish_reason": "stop",
                "native_finish_reason": "end_turn",
                "index": 0,
                "message": {"role": "assistant", "content": '{"answer": "yes"}', "refusal": None},
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
    }


def _stream() -> bytes:
    """a streamed OpenRouter chat response, as the API sends it: ``provider`` on every chunk.

    :return: the server-sent events body
    :rtype: bytes
    """
    base = {"id": "gen-1758000000-abc", "provider": _SERVED_BY, "model": _MODEL, "object": "chat.completion.chunk"}
    chunks = [
        {
            **base,
            "created": 1758000000,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": '{"answer": '}, "finish_reason": None}],
        },
        {
            **base,
            "created": 1758000000,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": '"yes"}'},
                    "finish_reason": "stop",
                    "native_finish_reason": "end_turn",
                }
            ],
        },
        {
            **base,
            "created": 1758000000,
            "choices": [],
            "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
        },
    ]
    events = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    return events.encode()


class _Wire:
    """the OpenRouter API end of the SDK's http clients: records each request body, answers it."""

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        """answer one chat request, streamed or not, as OpenRouter does.

        :param request: the request the SDK sent
        :ptype request: httpx.Request
        :return: the response
        :rtype: httpx.Response
        """
        body = json.loads(request.content)
        self.bodies.append(body)
        if body.get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_stream())
        return httpx.Response(200, json=_response())


def _model(wire: _Wire, **settings: Any) -> Any:
    """an OpenRouter chat model from the factory whose SDK talks to ``wire``.

    :param wire: the scripted API
    :ptype wire: _Wire
    :param settings: further factory keyword arguments (routing, callbacks)
    :ptype settings: Any
    :return: the chat model
    :rtype: Any
    """
    import openrouter

    transport = httpx.MockTransport(wire.handle)
    sdk = openrouter.OpenRouter(
        api_key="sk-or-test",
        client=httpx.Client(transport=transport),
        async_client=httpx.AsyncClient(transport=transport),
    )
    return create_openrouter_chat(_MODEL, "sk-or-test", max_retries=0, client=sdk, **settings)


async def _ask(model: Any, how: str) -> Any:
    """one ask on ``model`` by ``how``; the final message (streams aggregated).

    :param model: the chat model or binding
    :ptype model: Any
    :param how: ``ainvoke``, ``invoke``, ``astream`` or ``stream``
    :ptype how: str
    :return: the answer
    :rtype: Any
    """
    if how == "ainvoke":
        answer = await model.ainvoke("is it?")
    elif how == "invoke":
        answer = model.invoke("is it?")
    elif how == "astream":
        answer = None
        async for chunk in model.astream("is it?"):
            answer = chunk if answer is None else answer + chunk
    else:
        answer = None
        for chunk in model.stream("is it?"):
            answer = chunk if answer is None else answer + chunk
    return answer


_PATHS = ["ainvoke", "invoke", "astream", "stream"]


class TestAStructuredAskKeepsTheModelsRouting:
    """``ChatOpenRouter`` sends ``{**params, **kwargs}``, so the structured directive's bound
    ``provider={"require_parameters": True}`` used to REPLACE the model's routing: every structured
    ask went out with no ``only`` and no ``ignore`` at all."""

    @pytest.mark.parametrize("how", _PATHS)
    async def test_the_routing_lists_and_require_parameters_both_go_out(self, how: str) -> None:
        wire = _Wire()
        model = _model(wire, openrouter_provider=dict(_ROUTING))
        await _ask(model.bind(**structured_output_kwargs("openrouter", _SCHEMA)), how)
        [body] = wire.bodies
        assert body["provider"] == {**_ROUTING, "require_parameters": True}, "the model's routing was dropped"
        assert body["response_format"]["type"] == "json_schema"

    @pytest.mark.parametrize("how", _PATHS)
    async def test_a_model_with_no_routing_sends_just_require_parameters(self, how: str) -> None:
        wire = _Wire()
        await _ask(_model(wire).bind(**structured_output_kwargs("openrouter", _SCHEMA)), how)
        [body] = wire.bodies
        assert body["provider"] == {"require_parameters": True}

    async def test_a_plain_ask_sends_the_routing_untouched(self) -> None:
        wire = _Wire()
        await _ask(_model(wire, openrouter_provider=dict(_ROUTING)), "ainvoke")
        [body] = wire.bodies
        assert body["provider"] == _ROUTING

    async def test_the_models_routing_is_not_mutated(self) -> None:
        wire = _Wire()
        model = _model(wire, openrouter_provider=dict(_ROUTING))
        await _ask(model.bind(**structured_output_kwargs("openrouter", _SCHEMA)), "ainvoke")
        assert model.openrouter_provider == _ROUTING


class TestTheServedProviderIsRead:
    """The SDK's ``ChatResult`` / ``ChatStreamChunk`` did not declare ``provider``, so parsing dropped
    it before ``langchain-openrouter`` could copy it anywhere."""

    async def test_a_non_streamed_answer_names_it_in_response_metadata(self) -> None:
        answer = await _ask(_model(_Wire()), "ainvoke")
        assert answer.response_metadata.get("provider") == _SERVED_BY

    async def test_a_streamed_answer_names_it_on_the_final_chunks_metadata(self) -> None:
        """the final chunk's generation info reaches a streaming caller as its ``response_metadata``."""
        model = _model(_Wire())
        metas = [chunk.response_metadata async for chunk in model.astream([HumanMessage(content="is it?")])]
        finals = [meta for meta in metas if meta.get("finish_reason")]
        assert [meta.get("provider") for meta in finals] == [_SERVED_BY]

    @pytest.mark.parametrize("how", ["ainvoke", "astream"])
    async def test_the_per_call_log_line_carries_it(self, how: str, caplog: pytest.LogCaptureFixture) -> None:
        callback = UsageTrackingCallback(tracker=UsageTracker(), model_name=_MODEL, provider_name="openrouter")
        with caplog.at_level(logging.INFO, logger="threetears.models.tracking"):
            await _ask(_model(_Wire(), callbacks=[callback]), how)
        [line] = [r for r in caplog.records if r.getMessage() == "LLM call completed"]
        assert line.extra_data["served_provider"] == _SERVED_BY  # type: ignore[attr-defined]
        assert line.levelno == logging.INFO


class TestDeclaringTheFieldIsSafe:
    def test_declaring_twice_changes_nothing(self) -> None:
        from openrouter.components.chatresult import ChatResult

        from threetears.models.providers.openrouter import declare_served_provider

        declare_served_provider()
        fields = dict(ChatResult.model_fields)
        declare_served_provider()
        assert ChatResult.model_fields == fields
        assert "provider" in fields

    def test_a_class_that_moved_fails_loud(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        import types

        from threetears.models.providers.openrouter import declare_served_provider

        monkeypatch.setitem(sys.modules, "openrouter.components.chatstreamchunk", types.ModuleType("moved"))
        with pytest.raises(ImportError):
            declare_served_provider()
