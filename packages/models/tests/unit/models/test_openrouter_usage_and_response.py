"""What OpenRouter sends and LangChain drops reaches the answer's ``response_metadata``, per call.

OpenRouter's reply carries a ``usage`` block -- ``is_byok`` (whether the call ran on the caller's own
provider key), ``server_tool_use_details`` (billed), token-detail counts, each sent or not -- and
response-level fields (``openrouter_metadata``, ``service_tier``, ``system_fingerprint``). The
``openrouter`` SDK parses them all. ``langchain-openrouter`` 0.2.9 keeps ``cost`` and ``cost_details``,
flattens the counts into ``usage_metadata`` with ``or 0``, and drops the rest. Under BYOK ``cost`` is
only OpenRouter's fee, so without ``is_byok`` nobody could tell a whole cost from a fee.

The factory's model puts the usage as sent under ``openrouter_usage`` and the response-level fields
under ``openrouter_response``. Every test drives the real ``openrouter`` SDK over an
``httpx.MockTransport``: the reply is parsed by the SDK's own models and turned into a message by
``langchain-openrouter``'s own code, exactly as a live one would be. The wire answers each request by
the question it carries, so two concurrent calls on one model can be given different replies.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from langchain_core.messages import BaseMessage
from langchain_openrouter import ChatOpenRouter

from threetears.models.providers.openrouter import create_openrouter_chat

_MODEL = "anthropic/claude-sonnet-4.5"
_COST = 0.00021
_COST_DETAILS = {
    "upstream_inference_cost": 0.0042,
    "upstream_inference_prompt_cost": 0.003,
    "upstream_inference_completions_cost": 0.0012,
}
_COUNTS = {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}

#: each question's ``usage`` block, exactly as the wire sends it; ``None`` sends no usage at all.
_USAGE: dict[str, dict[str, Any] | None] = {
    # BYOK, with a billed server tool and only some detail counts: cached 0 sent, the rest not
    "byok": {
        **_COUNTS,
        "cost": _COST,
        "cost_details": dict(_COST_DETAILS),
        "is_byok": True,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 3},
        "server_tool_use_details": {"web_search_requests": 2},
    },
    # OpenRouter's own key; no detail blocks sent at all
    "own key": {**_COUNTS, "cost": _COST, "cost_details": dict(_COST_DETAILS), "is_byok": False},
    # OpenRouter did not say
    "unsaid": {**_COUNTS, "cost": _COST, "cost_details": dict(_COST_DETAILS)},
    # a sent null is kept as sent
    "null count": {**_COUNTS, "cost": _COST, "prompt_tokens_details": {"cached_tokens": 4, "audio_tokens": None}},
    "no usage": None,
    "bare": {**_COUNTS, "cost": _COST},
}

_ROUTER_METADATA = {
    "attempt": 1,
    "endpoints": {"available": [], "total": 0},
    "is_byok": True,
    "region": None,
    "requested": _MODEL,
    "strategy": "direct",
    "summary": "served by the first endpoint",
}
#: the response-level fields every question but ``bare`` is answered with.
_RESPONSE = {"openrouter_metadata": _ROUTER_METADATA, "service_tier": "default", "system_fingerprint": "fp_7f3a"}


@dataclass
class _Wire:
    """OpenRouter's end of the SDK: answers each request by the question it carries.

    A streamed answer sends ``service_tier`` and ``system_fingerprint`` on every chunk and
    ``openrouter_metadata`` on the last one with ``usage``, as a router reports it once routing is
    settled. ``bare`` sends none of them (a non-streamed ``bare`` sends ``system_fingerprint: null``,
    which the SDK requires).

    :ivar gaps: per question, the pause before each server-sent event of a streamed answer
        (three content events, the usage event, then ``[DONE]``); none when absent
    :ivar sent: every streamed event as it left, ``(question, what)``, in the order they left
    :ivar streamed: whether each request asked for a stream, in order
    """

    gaps: dict[str, list[float]] = field(default_factory=dict)
    sent: list[tuple[str, str]] = field(default_factory=list)
    streamed: list[bool] = field(default_factory=list)

    def handle(self, request: httpx.Request) -> httpx.Response:
        """answer one chat request, streamed or not.

        :param request: the request the SDK sent
        :ptype request: httpx.Request
        :return: the response
        :rtype: httpx.Response
        """
        body = json.loads(request.content)
        question = body["messages"][-1]["content"]
        self.streamed.append(bool(body.get("stream")))
        base = {"id": f"gen-{question}", "provider": "Anthropic", "model": _MODEL, "created": 1790000000}
        if body.get("stream"):
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=self._events(question, base)
            )
        answer: dict[str, Any] = {
            **base,
            "object": "chat.completion",
            "system_fingerprint": None,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "logprobs": None,
                    "message": {"role": "assistant", "content": f"answer to {question}"},
                }
            ],
        }
        if question != "bare":
            answer.update(_RESPONSE)
        if (usage := _USAGE[question]) is not None:
            answer["usage"] = usage
        return httpx.Response(200, json=answer)

    @staticmethod
    def _event_list(question: str, base: dict[str, Any]) -> list[tuple[str, bytes]]:
        """the streamed answer's server-sent events, each named: three content chunks, usage, done.

        :param question: the question answered
        :ptype question: str
        :param base: the fields every chunk carries
        :ptype base: dict[str, Any]
        :return: ``(what, event)`` in order
        :rtype: list[tuple[str, bytes]]
        """
        every = {} if question == "bare" else {k: _RESPONSE[k] for k in ("service_tier", "system_fingerprint")}
        deltas = [
            ("role", {"role": "assistant", "content": "answer "}),
            ("text", {"content": "to "}),
            ("finish", {"content": question}),
        ]
        events = []
        for what, delta in deltas:
            choice = {
                "index": 0,
                "delta": delta,
                "finish_reason": "stop" if what == "finish" else None,
                "logprobs": None,
            }
            chunk = {**base, **every, "object": "chat.completion.chunk", "choices": [choice]}
            events.append((what, f"data: {json.dumps(chunk)}\n\n".encode()))
        tail: dict[str, Any] = {**base, **every, "object": "chat.completion.chunk", "choices": []}
        if question != "bare":
            tail["openrouter_metadata"] = _ROUTER_METADATA
        if (usage := _USAGE[question]) is not None:
            tail["usage"] = usage
        events.append(("usage", f"data: {json.dumps(tail)}\n\n".encode()))
        events.append(("done", b"data: [DONE]\n\n"))
        return events

    def _events(self, question: str, base: dict[str, Any]) -> bytes | AsyncIterator[bytes]:
        """the streamed answer: all at once (a sync client can read only that), or paced by ``gaps``.

        :param question: the question answered
        :ptype question: str
        :param base: the fields every chunk carries
        :ptype base: dict[str, Any]
        :return: the body, or its paced events
        :rtype: bytes | AsyncIterator[bytes]
        """
        events = self._event_list(question, base)
        gaps = self.gaps.get(question)
        if gaps is None:
            self.sent.extend((question, what) for what, _ in events)
            return b"".join(event for _, event in events)
        return self._paced(question, events, gaps)

    async def _paced(self, question: str, events: list[tuple[str, bytes]], gaps: list[float]) -> AsyncIterator[bytes]:
        for (what, event), gap in zip(events, gaps, strict=True):
            await asyncio.sleep(gap)
            self.sent.append((question, what))
            yield event


def _sdk(wire: _Wire) -> Any:
    """the ``openrouter`` SDK client, talking to ``wire``.

    :param wire: the scripted API
    :ptype wire: _Wire
    :return: the SDK client
    :rtype: Any
    """
    import openrouter

    transport = httpx.MockTransport(wire.handle)
    return openrouter.OpenRouter(
        api_key="sk-or-test",
        client=httpx.Client(transport=transport),
        async_client=httpx.AsyncClient(transport=transport),
    )


def _model(wire: _Wire, *, deadline: bool = True) -> Any:
    """an OpenRouter chat model from the factory, whose SDK talks to ``wire``.

    :param wire: the scripted API
    :ptype wire: _Wire
    :param deadline: keep the factory's call deadline (an ``ainvoke`` then streams); without it an
        ``ainvoke`` takes LangChain's non-streamed path
    :ptype deadline: bool
    :return: the chat model
    :rtype: Any
    """
    model = create_openrouter_chat(_MODEL, "sk-or-test", max_retries=0, client=_sdk(wire))
    if not deadline:
        model.request_timeout = None
    return model


async def _ask(model: Any, how: str, question: str) -> BaseMessage:
    """one ask on ``model`` by ``how``; the final message (a stream aggregated).

    :param model: the chat model
    :ptype model: Any
    :param how: ``ainvoke``, ``invoke``, ``astream`` or ``stream``
    :ptype how: str
    :param question: the question, which picks the wire's reply
    :ptype question: str
    :return: the answer
    :rtype: BaseMessage
    """
    if how == "ainvoke":
        answer: Any = await model.ainvoke(question)
    elif how == "invoke":
        answer = model.invoke(question)
    elif how == "astream":
        answer = None
        async for chunk in model.astream(question):
            answer = chunk if answer is None else answer + chunk
    else:
        answer = None
        for chunk in model.stream(question):
            answer = chunk if answer is None else answer + chunk
    assert isinstance(answer, BaseMessage)
    return answer


#: every way a call reaches OpenRouter, and whether the wire sees a streamed request for it:
#: ``ainvoke`` under the factory's deadline collects the stream; without the deadline it takes
#: LangChain's non-streamed ``_agenerate``; a sync ``invoke`` is always LangChain's ``_generate``.
_PATHS = [
    pytest.param("astream", True, True, id="astream"),
    pytest.param("ainvoke", True, True, id="ainvoke-under-deadline"),
    pytest.param("stream", True, True, id="stream"),
    pytest.param("ainvoke", False, False, id="ainvoke-no-deadline"),
    pytest.param("invoke", True, False, id="invoke"),
]


class TestTheUsageArrivesAsSent:
    @pytest.mark.parametrize(("how", "deadline", "streamed"), _PATHS)
    @pytest.mark.parametrize("question", ["byok", "own key", "unsaid", "null count"])
    async def test_openrouter_usage_is_the_usage_block_exactly(
        self, how: str, deadline: bool, streamed: bool, question: str
    ) -> None:
        wire = _Wire()
        answer = await _ask(_model(wire, deadline=deadline), how, question)
        assert wire.streamed == [streamed]
        assert answer.response_metadata["openrouter_usage"] == _USAGE[question]

    @pytest.mark.parametrize(("how", "deadline", "streamed"), _PATHS)
    @pytest.mark.parametrize(("question", "said"), [("byok", True), ("own key", False)])
    async def test_is_byok_is_read_from_it(
        self, how: str, deadline: bool, streamed: bool, question: str, said: bool
    ) -> None:
        answer = await _ask(_model(_Wire(), deadline=deadline), how, question)
        assert answer.response_metadata["openrouter_usage"]["is_byok"] is said
        assert "is_byok" not in answer.response_metadata, "one key, openrouter_usage; no top-level copy"

    @pytest.mark.parametrize(("how", "deadline", "streamed"), _PATHS)
    async def test_an_unsaid_is_byok_is_absent_not_false(self, how: str, deadline: bool, streamed: bool) -> None:
        answer = await _ask(_model(_Wire(), deadline=deadline), how, "unsaid")
        assert "is_byok" not in answer.response_metadata["openrouter_usage"]

    @pytest.mark.parametrize(("how", "deadline", "streamed"), _PATHS)
    async def test_a_count_openrouter_omitted_is_absent_not_zero(
        self, how: str, deadline: bool, streamed: bool
    ) -> None:
        """LangChain's ``usage_metadata`` reads an omitted count as 0; ``openrouter_usage`` does not."""
        usage = (await _ask(_model(_Wire(), deadline=deadline), how, "byok")).response_metadata["openrouter_usage"]
        assert usage["prompt_tokens_details"] == {"cached_tokens": 0}
        assert usage["completion_tokens_details"] == {"reasoning_tokens": 3}
        own = (await _ask(_model(_Wire(), deadline=deadline), how, "own key")).response_metadata["openrouter_usage"]
        assert "prompt_tokens_details" not in own
        assert "completion_tokens_details" not in own

    @pytest.mark.parametrize(("how", "deadline", "streamed"), _PATHS)
    async def test_billed_server_tool_use_survives(self, how: str, deadline: bool, streamed: bool) -> None:
        answer = await _ask(_model(_Wire(), deadline=deadline), how, "byok")
        assert answer.response_metadata["openrouter_usage"]["server_tool_use_details"] == {"web_search_requests": 2}

    @pytest.mark.parametrize(("how", "deadline", "streamed"), _PATHS)
    async def test_a_reply_without_usage_has_no_key(self, how: str, deadline: bool, streamed: bool) -> None:
        wire = _Wire()
        answer = await _ask(_model(wire, deadline=deadline), how, "no usage")
        assert wire.streamed == [streamed]
        assert "openrouter_usage" not in answer.response_metadata

    async def test_on_a_stream_it_rides_the_usage_chunk_beside_the_cost(self) -> None:
        chunks = [chunk async for chunk in _model(_Wire()).astream("byok")]
        carrying = [chunk for chunk in chunks if "openrouter_usage" in chunk.response_metadata]
        assert len(carrying) == 1
        [usage_chunk] = carrying
        assert usage_chunk.usage_metadata is not None
        assert usage_chunk.response_metadata["cost"] == _COST
        assert usage_chunk.response_metadata["openrouter_response"] == _RESPONSE


class TestTheResponseFieldsArrive:
    @pytest.mark.parametrize(("how", "deadline", "streamed"), _PATHS)
    async def test_openrouter_response_carries_what_langchain_drops(
        self, how: str, deadline: bool, streamed: bool
    ) -> None:
        answer = await _ask(_model(_Wire(), deadline=deadline), how, "byok")
        assert answer.response_metadata["openrouter_response"] == _RESPONSE

    @pytest.mark.parametrize(("how", "deadline", "streamed"), _PATHS)
    async def test_none_sent_leaves_no_key(self, how: str, deadline: bool, streamed: bool) -> None:
        """``bare`` sends none of them; its non-streamed ``system_fingerprint: null`` is no value."""
        answer = await _ask(_model(_Wire(), deadline=deadline), how, "bare")
        assert "openrouter_response" not in answer.response_metadata
        assert answer.response_metadata["openrouter_usage"] == _USAGE["bare"]

    @pytest.mark.parametrize(("how", "deadline", "streamed"), _PATHS)
    async def test_they_arrive_on_a_reply_without_usage(self, how: str, deadline: bool, streamed: bool) -> None:
        """a stream with no usage chunk carries them on a trailing empty chunk, not nowhere."""
        answer = await _ask(_model(_Wire(), deadline=deadline), how, "no usage")
        assert answer.response_metadata["openrouter_response"] == _RESPONSE
        assert answer.content == "answer to no usage"

    async def test_the_fields_langchain_carries_are_not_repeated(self) -> None:
        response = (await _ask(_model(_Wire()), "astream", "byok")).response_metadata["openrouter_response"]
        assert not {"id", "model", "provider", "cost", "cost_details", "usage"} & set(response)


class TestLangChainsOwnMetadataIsUnchanged:
    """``cost``, ``cost_details`` and every other key arrive exactly as ``langchain-openrouter`` alone delivers them."""

    @staticmethod
    def _plain(wire: _Wire) -> ChatOpenRouter:
        return ChatOpenRouter(
            model=_MODEL, api_key="sk-or-test", client=_sdk(wire), max_retries=0, app_title=None, app_url=None
        )

    @pytest.mark.parametrize("how", ["astream", "ainvoke", "stream", "invoke"])
    async def test_everything_but_the_two_keys_matches_langchains_own(self, how: str) -> None:
        ours = await _ask(_model(_Wire(), deadline=False), how, "byok")
        theirs = await _ask(self._plain(_Wire()), how, "byok")
        assert ours.response_metadata["cost"] == theirs.response_metadata["cost"] == _COST
        assert ours.response_metadata["cost_details"] == theirs.response_metadata["cost_details"] == _COST_DETAILS
        added = {"openrouter_usage", "openrouter_response"}
        assert not added & set(theirs.response_metadata), "langchain-openrouter writes these keys itself now"
        assert {k: v for k, v in ours.response_metadata.items() if k not in added} == theirs.response_metadata
        assert ours.usage_metadata == theirs.usage_metadata


class TestConcurrentCallsKeepTheirOwnUsage:
    """two calls on ONE model instance, their streams interleaved on the wire, each its own usage."""

    #: byok's events leave at 0, 20, 40, 60 (usage), 80 ms; own key's at 10, 30, 50, 70 (usage), 75 ms.
    _GAPS = {"byok": [0.0, 0.02, 0.02, 0.02, 0.02], "own key": [0.01, 0.02, 0.02, 0.02, 0.005]}

    @pytest.mark.parametrize(("how", "deadline"), [("astream", True), ("ainvoke", True), ("ainvoke", False)])
    async def test_each_call_carries_its_own(self, how: str, deadline: bool) -> None:
        wire = _Wire(gaps=dict(self._GAPS))
        model = _model(wire, deadline=deadline)
        byok, own = await asyncio.gather(_ask(model, how, "byok"), _ask(model, how, "own key"))
        assert byok.response_metadata["openrouter_usage"] == _USAGE["byok"]
        assert own.response_metadata["openrouter_usage"] == _USAGE["own key"]
        if deadline:
            # the two streams really did interleave: each usage chunk left while the other call was open
            order = [f"{q}:{what}" for q, what in wire.sent]
            assert order.index("own key:role") < order.index("byok:usage") < order.index("own key:usage")
            assert order.index("own key:usage") < order.index("byok:done")

    async def test_no_usage_beside_usage_stays_without(self) -> None:
        wire = _Wire(gaps={"byok": [0.0, 0.02, 0.02, 0.0, 0.02], "no usage": [0.01, 0.02, 0.02, 0.0, 0.02]})
        model = _model(wire)
        byok, bare = await asyncio.gather(_ask(model, "astream", "byok"), _ask(model, "astream", "no usage"))
        assert byok.response_metadata["openrouter_usage"] == _USAGE["byok"]
        assert "openrouter_usage" not in bare.response_metadata


class TestAMovedSdkFailsAtBuild:
    """a structural change the wrap depends on raises when the model is built, by name."""

    @pytest.mark.parametrize("missing", ["send", "send_async"])
    def test_a_chat_without_a_send_method_raises(self, missing: str) -> None:
        sdk = _sdk(_Wire())

        class _ChatMissingASend:
            def __getattr__(self, name: str) -> Any:
                if name == missing:
                    raise AttributeError(name)
                return getattr(sdk.chat, name)

        class _ClientWithAMovedChat:
            chat = _ChatMissingASend()

        with pytest.raises(TypeError, match=f"no callable `{missing}`"):
            create_openrouter_chat(_MODEL, "sk-or-test", client=_ClientWithAMovedChat())

    def test_a_client_without_chat_raises(self) -> None:
        class _ClientWithNoChat:
            pass

        with pytest.raises(TypeError, match="has no `chat`"):
            create_openrouter_chat(_MODEL, "sk-or-test", client=_ClientWithNoChat())

    def test_the_sdk_clients_other_surface_is_still_reached(self) -> None:
        sdk = _sdk(_Wire())
        model = create_openrouter_chat(_MODEL, "sk-or-test", client=sdk)
        assert model.client.models is sdk.models
