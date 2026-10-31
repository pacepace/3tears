"""A tool call's provider-specific fields survive the round trip through an OpenAI-compatible model.

Gemini 3 returns an opaque thought signature with each function call and refuses the next request
unless the signature comes back unchanged on that call. On Google's OpenAI-compatible endpoint it
rides a non-standard top-level field of the tool call -- ``extra_content.google.thought_signature``
(https://ai.google.dev/gemini-api/docs/thought-signatures, "OpenAI compatibility") -- and only the
FIRST of several parallel calls carries one. LangChain's OpenAI integration drops every such field
in both directions, so these suites drive the real SDK and LangChain parse through the scripted
chat-completions wire and assert on the bytes sent back.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from threetears.models.providers import TOOL_CALL_PROVIDER_EXTRA_KEY

from .provider_wire import ChatCompletionsWire, openai_model

_SIG_1 = "CtYBAdHtim9+opaque/signature+one=="
_SIG_2 = "CtYBAdHtim9+opaque/signature+two=="


def _google(signature: str) -> dict[str, Any]:
    """the extra field Gemini's OpenAI-compatible endpoint puts on a signed tool call.

    :param signature: the opaque signature
    :ptype signature: str
    :return: the field, as the endpoint sends it
    :rtype: dict[str, Any]
    """
    return {"extra_content": {"google": {"thought_signature": signature}}}


def _parallel_calls_message() -> dict[str, Any]:
    """a non-streamed Gemini answer with two parallel calls; only the first is signed.

    :return: the choice's message
    :rtype: dict[str, Any]
    """
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                **_google(_SIG_1),
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
            },
        ],
    }


def _fragment(index: int, *, call_id: str | None, name: str | None, arguments: str, **extra: Any) -> dict[str, Any]:
    """one streamed tool-call fragment, with any provider fields beside the standard ones.

    :param index: the call's index
    :ptype index: int
    :param call_id: the call's id, sent on its first fragment only
    :ptype call_id: str | None
    :param name: the call's name, sent on its first fragment only
    :ptype name: str | None
    :param arguments: this fragment's argument text
    :ptype arguments: str
    :param extra: provider fields on this fragment
    :ptype extra: Any
    :return: the fragment
    :rtype: dict[str, Any]
    """
    function: dict[str, Any] = {"arguments": arguments}
    if name is not None:
        function["name"] = name
    fragment: dict[str, Any] = {"index": index, "function": function, **extra}
    if call_id is not None:
        fragment["id"] = call_id
        fragment["type"] = "function"
    return fragment


def _delta(*fragments: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant", "content": "", "tool_calls": list(fragments)}


def _streamed_parallel_calls() -> list[dict[str, Any]]:
    """Gemini's streamed shape: each call whole in one fragment, the signature on the first call.

    :return: the deltas
    :rtype: list[dict[str, Any]]
    """
    return [
        _delta(_fragment(0, call_id="call_1", name="get_weather", arguments='{"city": "Paris"}', **_google(_SIG_1))),
        _delta(_fragment(1, call_id="call_2", name="get_weather", arguments='{"city": "London"}')),
    ]


def _sync_streaming_model(deltas: list[dict[str, Any]]) -> Any:
    """an OpenAI chat model whose sync client receives ``deltas`` as one server-sent-event body.

    :param deltas: the deltas to stream
    :ptype deltas: list[dict[str, Any]]
    :return: the chat model
    :rtype: Any
    """
    from threetears.models.providers.openai import create_openai_chat

    def handle(request: httpx.Request) -> httpx.Response:
        events = []
        for i, delta in enumerate(deltas):
            finish = "tool_calls" if i == len(deltas) - 1 else None
            chunk = {
                "id": "gen-1",
                "object": "chat.completion.chunk",
                "created": 1790000000,
                "model": "gemini-3.5-flash-lite",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": None}],
            }
            events.append(f"data: {json.dumps(chunk)}\n\n")
        events.append("data: [DONE]\n\n")
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content="".join(events).encode())

    return create_openai_chat(
        "gemini-3.5-flash-lite",
        "sk-test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    )


def _summed(chunks: list[AIMessageChunk]) -> AIMessageChunk:
    total = chunks[0]
    for chunk in chunks[1:]:
        total = total + chunk
    return total


def _sent_assistant_tool_calls(wire: ChatCompletionsWire) -> list[dict[str, Any]]:
    return [
        call
        for message in wire.bodies[-1]["messages"]
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
    ]


def _history(answer: AIMessage) -> list[Any]:
    """the next turn's input: the question, the model's answer, and one result per call.

    :param answer: the model's first answer
    :ptype answer: AIMessage
    :return: the messages
    :rtype: list[Any]
    """
    results = [ToolMessage(content="sunny", tool_call_id=call["id"]) for call in answer.tool_calls]
    return [HumanMessage(content="weather?"), answer, *results]


class TestANonStreamedAnswerKeepsEachSignature:
    """a non-streamed answer carries each signed call's fields under the key, by call id."""

    @pytest.mark.asyncio
    async def test_ainvoke(self) -> None:
        wire = ChatCompletionsWire(message=_parallel_calls_message(), finish="tool_calls")
        answer = await openai_model(wire, "gemini-3.5-flash-lite").ainvoke("weather?")
        assert answer.additional_kwargs[TOOL_CALL_PROVIDER_EXTRA_KEY] == {"call_1": _google(_SIG_1)}
        assert [call["id"] for call in answer.tool_calls] == ["call_1", "call_2"]

    def test_invoke(self) -> None:
        wire = ChatCompletionsWire(message=_parallel_calls_message(), finish="tool_calls")
        answer = openai_model(wire, "gemini-3.5-flash-lite").invoke("weather?")
        assert answer.additional_kwargs[TOOL_CALL_PROVIDER_EXTRA_KEY] == {"call_1": _google(_SIG_1)}

    @pytest.mark.asyncio
    async def test_every_signed_call_keeps_its_own(self) -> None:
        message = _parallel_calls_message()
        message["tool_calls"][1].update(_google(_SIG_2))
        wire = ChatCompletionsWire(message=message, finish="tool_calls")
        answer = await openai_model(wire, "gemini-3.5-flash-lite").ainvoke("weather?")
        assert answer.additional_kwargs[TOOL_CALL_PROVIDER_EXTRA_KEY] == {
            "call_1": _google(_SIG_1),
            "call_2": _google(_SIG_2),
        }

    @pytest.mark.asyncio
    async def test_an_answer_without_extras_has_no_key(self) -> None:
        message = _parallel_calls_message()
        del message["tool_calls"][0]["extra_content"]
        wire = ChatCompletionsWire(message=message, finish="tool_calls")
        answer = await openai_model(wire).ainvoke("weather?")
        assert TOOL_CALL_PROVIDER_EXTRA_KEY not in answer.additional_kwargs


class TestAStreamedAnswerCarriesOneCopy:
    """summing a streamed answer yields each signature exactly once, under its call's id."""

    @pytest.mark.asyncio
    async def test_astream_summed(self) -> None:
        wire = ChatCompletionsWire(deltas=_streamed_parallel_calls(), finish="tool_calls")
        chunks = [chunk async for chunk in openai_model(wire, "gemini-3.5-flash-lite").astream("weather?")]
        assert _summed(chunks).additional_kwargs[TOOL_CALL_PROVIDER_EXTRA_KEY] == {"call_1": _google(_SIG_1)}

    @pytest.mark.asyncio
    async def test_no_chunk_carries_any_other_bookkeeping(self) -> None:
        wire = ChatCompletionsWire(deltas=_streamed_parallel_calls(), finish="tool_calls")
        chunks = [chunk async for chunk in openai_model(wire, "gemini-3.5-flash-lite").astream("weather?")]
        for chunk in chunks:
            assert set(chunk.additional_kwargs) <= {TOOL_CALL_PROVIDER_EXTRA_KEY}

    @pytest.mark.asyncio
    async def test_a_signature_repeated_on_a_later_fragment_is_not_doubled(self) -> None:
        # LangChain's merge concatenates equal strings, so a second copy would corrupt the signature.
        deltas = [
            _delta(_fragment(0, call_id="call_1", name="get_weather", arguments='{"city": ', **_google(_SIG_1))),
            _delta(_fragment(0, call_id=None, name=None, arguments='"Paris"}', **_google(_SIG_1))),
        ]
        wire = ChatCompletionsWire(deltas=deltas, finish="tool_calls")
        chunks = [chunk async for chunk in openai_model(wire, "gemini-3.5-flash-lite").astream("weather?")]
        summed = _summed(chunks)
        assert summed.additional_kwargs[TOOL_CALL_PROVIDER_EXTRA_KEY] == {"call_1": _google(_SIG_1)}
        assert summed.tool_call_chunks[0]["args"] == '{"city": "Paris"}'

    @pytest.mark.asyncio
    async def test_a_signature_on_a_continuation_fragment_is_keyed_by_the_calls_id(self) -> None:
        deltas = [
            _delta(_fragment(0, call_id="call_1", name="get_weather", arguments='{"city": ')),
            _delta(_fragment(0, call_id=None, name=None, arguments='"Paris"}', **_google(_SIG_1))),
        ]
        wire = ChatCompletionsWire(deltas=deltas, finish="tool_calls")
        chunks = [chunk async for chunk in openai_model(wire, "gemini-3.5-flash-lite").astream("weather?")]
        assert _summed(chunks).additional_kwargs[TOOL_CALL_PROVIDER_EXTRA_KEY] == {"call_1": _google(_SIG_1)}

    @pytest.mark.asyncio
    async def test_ainvoke_on_a_streaming_model(self) -> None:
        wire = ChatCompletionsWire(deltas=_streamed_parallel_calls(), finish="tool_calls")
        model = openai_model(wire, "gemini-3.5-flash-lite", streaming=True)
        answer = await model.ainvoke("weather?")
        assert answer.additional_kwargs[TOOL_CALL_PROVIDER_EXTRA_KEY] == {"call_1": _google(_SIG_1)}

    def test_sync_stream_summed(self) -> None:
        # the scripted wire streams through an async body, which a sync client cannot read
        chunks = list(_sync_streaming_model(_streamed_parallel_calls()).stream("weather?"))
        assert _summed(chunks).additional_kwargs[TOOL_CALL_PROVIDER_EXTRA_KEY] == {"call_1": _google(_SIG_1)}

    @pytest.mark.asyncio
    async def test_stream_events_see_the_resolved_key_only(self) -> None:
        wire = ChatCompletionsWire(deltas=_streamed_parallel_calls(), finish="tool_calls")
        model = openai_model(wire, "gemini-3.5-flash-lite")
        streamed: list[AIMessageChunk] = []
        async for event in model.astream_events("weather?", version="v2"):
            if event["event"] == "on_chat_model_stream":
                streamed.append(event["data"]["chunk"])
        assert streamed
        for chunk in streamed:
            assert set(chunk.additional_kwargs) <= {TOOL_CALL_PROVIDER_EXTRA_KEY}
        assert _summed(streamed).additional_kwargs[TOOL_CALL_PROVIDER_EXTRA_KEY] == {"call_1": _google(_SIG_1)}


class TestTheNextRequestSendsTheSignatureBack:
    """the fields come back verbatim on the matching assistant tool call of the next request."""

    @pytest.mark.asyncio
    async def test_after_a_non_streamed_answer(self) -> None:
        wire = ChatCompletionsWire(message=_parallel_calls_message(), finish="tool_calls")
        model = openai_model(wire, "gemini-3.5-flash-lite")
        answer = await model.ainvoke("weather?")
        wire.message = {"role": "assistant", "content": "Sunny in both."}
        wire.finish = "stop"
        await model.ainvoke(_history(answer))
        sent = _sent_assistant_tool_calls(wire)
        assert sent[0]["extra_content"] == {"google": {"thought_signature": _SIG_1}}
        assert "extra_content" not in sent[1]
        assert [call["id"] for call in sent] == ["call_1", "call_2"]

    @pytest.mark.asyncio
    async def test_after_a_streamed_answer(self) -> None:
        wire = ChatCompletionsWire(deltas=_streamed_parallel_calls(), finish="tool_calls")
        model = openai_model(wire, "gemini-3.5-flash-lite")
        chunks = [chunk async for chunk in model.astream("weather?")]
        summed = _summed(chunks)
        answer = AIMessage(
            content=summed.content,
            tool_calls=summed.tool_calls,
            additional_kwargs=summed.additional_kwargs,
            id=summed.id,
        )
        wire.deltas = [{"role": "assistant", "content": "Sunny."}]
        wire.finish = "stop"
        await model.ainvoke(_history(answer))
        sent = _sent_assistant_tool_calls(wire)
        assert sent[0]["extra_content"] == {"google": {"thought_signature": _SIG_1}}

    @pytest.mark.asyncio
    async def test_a_dotted_tool_name_is_still_translated(self) -> None:
        @tool("weather.lookup")
        def lookup(city: str) -> str:
            """look up the weather.

            :param city: the city
            :ptype city: str
            :return: the weather
            :rtype: str
            """
            return "sunny"

        message = _parallel_calls_message()
        for call in message["tool_calls"]:
            call["function"]["name"] = "weather_lookup"
        wire = ChatCompletionsWire(message=message, finish="tool_calls")
        bound = openai_model(wire, "gemini-3.5-flash-lite").bind_tools([lookup])
        answer = await bound.ainvoke("weather?")
        assert [call["name"] for call in answer.tool_calls] == ["weather.lookup", "weather.lookup"]
        wire.message = {"role": "assistant", "content": "Sunny."}
        wire.finish = "stop"
        await bound.ainvoke(_history(answer))
        sent = _sent_assistant_tool_calls(wire)
        assert [call["function"]["name"] for call in sent] == ["weather_lookup", "weather_lookup"]
        assert sent[0]["extra_content"] == {"google": {"thought_signature": _SIG_1}}

    def test_sync_invoke(self) -> None:
        wire = ChatCompletionsWire(message=_parallel_calls_message(), finish="tool_calls")
        model = openai_model(wire, "gemini-3.5-flash-lite")
        answer = model.invoke("weather?")
        wire.message = {"role": "assistant", "content": "Sunny."}
        wire.finish = "stop"
        model.invoke(_history(answer))
        assert _sent_assistant_tool_calls(wire)[0]["extra_content"] == {"google": {"thought_signature": _SIG_1}}

    @pytest.mark.asyncio
    async def test_the_key_itself_never_goes_on_the_wire(self) -> None:
        wire = ChatCompletionsWire(message=_parallel_calls_message(), finish="tool_calls")
        model = openai_model(wire, "gemini-3.5-flash-lite")
        answer = await model.ainvoke("weather?")
        wire.message = {"role": "assistant", "content": "Sunny."}
        wire.finish = "stop"
        await model.ainvoke(_history(answer))
        assert TOOL_CALL_PROVIDER_EXTRA_KEY not in json.dumps(wire.bodies[-1])


def _vanilla_body(history: list[Any]) -> dict[str, Any]:
    """the request body LangChain's own ``ChatOpenAI`` sends for ``history``.

    :param history: the messages
    :ptype history: list[Any]
    :return: the body
    :rtype: dict[str, Any]
    """
    wire = ChatCompletionsWire()
    transport = httpx.MockTransport(wire.handle)
    ChatOpenAI(model="gpt-4o", api_key="sk-test", max_retries=0, http_client=httpx.Client(transport=transport)).invoke(
        history
    )
    return wire.bodies[-1]


class TestARequestWithoutExtrasIsUnchanged:
    """with nothing to send back, the body is exactly what LangChain's own ChatOpenAI sends."""

    def test_tool_calls_without_extras(self) -> None:
        answer = AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "get_weather", "args": {"city": "Paris"}}],
        )
        history = _history(answer)
        wire = ChatCompletionsWire()
        openai_model(wire).invoke(history)
        assert json.dumps(wire.bodies[-1], sort_keys=False) == json.dumps(_vanilla_body(history), sort_keys=False)

    def test_extras_for_a_call_the_message_does_not_carry(self) -> None:
        answer = AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "get_weather", "args": {"city": "Paris"}}],
            additional_kwargs={TOOL_CALL_PROVIDER_EXTRA_KEY: {"call_9": _google(_SIG_1)}},
        )
        history = _history(answer)
        wire = ChatCompletionsWire()
        openai_model(wire).invoke(history)
        plain = _history(AIMessage(content="", tool_calls=answer.tool_calls))
        assert json.dumps(wire.bodies[-1]) == json.dumps(_vanilla_body(plain))


class TestWhatCannotBeSentBackIsReported:
    """fields that cannot reach the next request are logged, so a later 400 can be traced here."""

    @pytest.mark.asyncio
    async def test_a_conflicting_repeat_keeps_the_first_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        deltas = [
            _delta(_fragment(0, call_id="call_1", name="get_weather", arguments='{"city": ', **_google(_SIG_1))),
            _delta(_fragment(0, call_id=None, name=None, arguments='"Paris"}', **_google(_SIG_2))),
        ]
        wire = ChatCompletionsWire(deltas=deltas, finish="tool_calls")
        with caplog.at_level("WARNING"):
            chunks = [chunk async for chunk in openai_model(wire, "gemini-3.5-flash-lite").astream("weather?")]
        assert _summed(chunks).additional_kwargs[TOOL_CALL_PROVIDER_EXTRA_KEY] == {"call_1": _google(_SIG_1)}
        assert any("twice with different values" in record.getMessage() for record in caplog.records)

    @pytest.mark.asyncio
    async def test_a_streamed_call_that_never_gets_an_id_is_dropped_with_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        deltas = [_delta(_fragment(0, call_id=None, name="get_weather", arguments="{}", **_google(_SIG_1)))]
        wire = ChatCompletionsWire(deltas=deltas, finish="tool_calls")
        with caplog.at_level("WARNING"):
            chunks = [chunk async for chunk in openai_model(wire, "gemini-3.5-flash-lite").astream("weather?")]
        assert TOOL_CALL_PROVIDER_EXTRA_KEY not in _summed(chunks).additional_kwargs
        assert any("never an id" in record.getMessage() for record in caplog.records)

    @pytest.mark.asyncio
    async def test_a_non_streamed_call_without_an_id_is_reported(self, caplog: pytest.LogCaptureFixture) -> None:
        message = _parallel_calls_message()
        del message["tool_calls"][0]["id"]
        wire = ChatCompletionsWire(message=message, finish="tool_calls")
        with caplog.at_level("WARNING"):
            answer = await openai_model(wire, "gemini-3.5-flash-lite").ainvoke("weather?")
        assert TOOL_CALL_PROVIDER_EXTRA_KEY not in answer.additional_kwargs
        assert any("but no id" in record.getMessage() for record in caplog.records)

    def test_kept_fields_matching_no_request_call_are_reported(self, caplog: pytest.LogCaptureFixture) -> None:
        answer = AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "get_weather", "args": {"city": "Paris"}}],
            additional_kwargs={TOOL_CALL_PROVIDER_EXTRA_KEY: {"call_9": _google(_SIG_1)}},
        )
        wire = ChatCompletionsWire()
        with caplog.at_level("WARNING"):
            openai_model(wire).invoke(_history(answer))
        assert any("match no tool call" in record.getMessage() for record in caplog.records)


class TestAStructuredOutputStreamCarriesOneCopy:
    """a stream with ``response_format`` ends on a chunk LangChain rebuilds from the whole completion.

    With the locked openai SDK that completion does not carry the fields (its accumulator keeps only the
    declared ones), so the repeat cannot happen today. The resolver still sends a pre-keyed chunk through
    the same emit-once check; this pins the outcome against an SDK whose accumulator starts keeping them.
    """

    @pytest.mark.asyncio
    async def test_astream_with_response_format(self) -> None:
        wire = ChatCompletionsWire(deltas=_streamed_parallel_calls(), finish="tool_calls")
        model = openai_model(wire, "gemini-3.5-flash-lite")
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "weather", "schema": {"type": "object", "properties": {}}, "strict": False},
        }
        chunks = [chunk async for chunk in model.astream("weather?", response_format=response_format)]
        assert _summed(chunks).additional_kwargs[TOOL_CALL_PROVIDER_EXTRA_KEY] == {"call_1": _google(_SIG_1)}
