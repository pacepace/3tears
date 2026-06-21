"""unit tests for :mod:`threetears.langgraph.streaming`.

covers wire-event serialization, the :func:`parse_stream_event`
discriminated-union parser, and the full :class:`StreamingResponse`
lifecycle (idempotent terminals, mutual exclusion, error semantics,
and the :meth:`StreamingResponse.run_graph` consume loop).

test doubles substitute the :class:`StreamTransport` Protocol with
a recording fake so assertions inspect every published event without
any NATS or websocket plumbing.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from uuid import UUID

import pytest
from langgraph.types import Command
from uuid_utils import uuid7

from threetears.langgraph.streaming import (
    NOSTREAM_TAG,
    StreamEndEvent,
    StreamErrorEvent,
    StreamingResponse,
    StreamingResponseError,
    StreamStartEvent,
    StreamTokenEvent,
    StreamTransport,
    ToolCallEndEvent,
    ToolCallProgressEvent,
    ToolCallStartEvent,
    parse_stream_event,
)


def _new_uuid() -> UUID:
    """build a fresh UUID v7 for test wiring.

    :return: time-ordered UUID
    :rtype: UUID
    """
    return UUID(str(uuid7()))


class _RecordingTransport:
    """:class:`StreamTransport` test double recording every publish.

    captures payload bytes and parses each into the discriminated
    union for inspection. tests assert on the parsed event sequence,
    not raw bytes, so changes to envelope shapes that do not break
    the wire compile here.
    """

    def __init__(self) -> None:
        """initialize empty recording lists.

        :return: nothing
        :rtype: None
        """
        self.payloads: list[bytes] = []
        self.publish_failure: Exception | None = None

    async def publish(self, payload: bytes) -> None:
        """record one publish call (or raise the configured failure).

        :param payload: serialized envelope bytes
        :ptype payload: bytes
        :return: nothing
        :rtype: None
        """
        if self.publish_failure is not None:
            raise self.publish_failure
        self.payloads.append(payload)

    @property
    def events(self) -> list[Any]:
        """return parsed events for assertion-friendly inspection.

        :return: parsed envelope list
        :rtype: list[Any]
        """
        return [parse_stream_event(p) for p in self.payloads]


class _MockChunk:
    """LLM streaming chunk stand-in carrying a string ``content``.

    matches the duck-typed shape :func:`_extract_token` consults on
    ``on_chat_model_stream`` events.
    """

    def __init__(self, content: str) -> None:
        """initialize the mock chunk.

        :param content: token text
        :ptype content: str
        :return: nothing
        :rtype: None
        """
        self.content = content


class _StubGraph:
    """compiled-graph test double exposing a programmable event stream.

    matches the duck-typed surface
    :meth:`StreamingResponse.run_graph` consumes
    (``compiled_graph.astream_events(state, config=..., version="v2")``).
    """

    def __init__(self, events: list[Any] | BaseException) -> None:
        """capture either a list of events (with optional exception sentinels) or a bare exception.

        :param events: events to yield, with optional exception sentinels,
            or a bare exception to raise immediately on the first iteration
        :ptype events: list[Any] | BaseException
        :return: nothing
        :rtype: None
        """
        self._events = events

    def astream_events(
        self,
        state: dict[str, Any],
        *,
        config: dict[str, Any],
        version: str,
    ) -> Any:
        """return an async iterator over the configured events.

        :param state: incoming state (unused; required by the contract)
        :ptype state: dict[str, Any]
        :param config: incoming config (unused; required by the contract)
        :ptype config: dict[str, Any]
        :param version: astream_events version literal ("v2")
        :ptype version: str
        :return: async generator of event dicts
        :rtype: Any
        """
        events = self._events

        async def _gen() -> Any:
            """yield each configured event in order, raising on exception sentinel."""
            if isinstance(events, BaseException):
                raise events
            for evt in events:
                if isinstance(evt, BaseException):
                    raise evt
                yield evt

        return _gen()


# ---------------------------------------------------------------------------
# Wire serialization tests
# ---------------------------------------------------------------------------


class TestStreamStartEventSerialization:
    """:class:`StreamStartEvent` serializes with stable shape."""

    def test_type_discriminator(self) -> None:
        """``type`` field renders the literal ``stream_start``."""
        evt = StreamStartEvent(
            correlation_id=_new_uuid(),
            conversation_id=_new_uuid(),
        )
        data = json.loads(evt.model_dump_json())
        assert data["type"] == "stream_start"

    def test_correlation_id_renders_as_string(self) -> None:
        """correlation_id renders as the canonical UUID string."""
        cid = _new_uuid()
        evt = StreamStartEvent(correlation_id=cid, conversation_id=_new_uuid())
        data = json.loads(evt.model_dump_json())
        assert data["correlation_id"] == str(cid)


class TestStreamTokenEventSerialization:
    """:class:`StreamTokenEvent` serializes with stable shape."""

    def test_type_and_token(self) -> None:
        """``type`` is ``stream_token`` and ``token`` round-trips verbatim."""
        evt = StreamTokenEvent(correlation_id=_new_uuid(), token="hello")
        data = json.loads(evt.model_dump_json())
        assert data["type"] == "stream_token"
        assert data["token"] == "hello"


class TestStreamEndEventSerialization:
    """:class:`StreamEndEvent` serializes with stable shape."""

    def test_type_and_content(self) -> None:
        """``type`` is ``stream_end`` and ``content`` round-trips verbatim."""
        evt = StreamEndEvent(
            correlation_id=_new_uuid(),
            content="full response",
            duration_ms=42,
        )
        data = json.loads(evt.model_dump_json())
        assert data["type"] == "stream_end"
        assert data["content"] == "full response"
        assert data["duration_ms"] == 42


class TestStreamErrorEventSerialization:
    """:class:`StreamErrorEvent` serializes with the new failure shape."""

    def test_type_code_message(self) -> None:
        """``type`` is ``stream_error`` and code/message round-trip."""
        evt = StreamErrorEvent(
            correlation_id=_new_uuid(),
            code="AGENT_FAILED",
            message="upstream gateway error",
            duration_ms=1234,
        )
        data = json.loads(evt.model_dump_json())
        assert data["type"] == "stream_error"
        assert data["code"] == "AGENT_FAILED"
        assert data["message"] == "upstream gateway error"
        assert data["duration_ms"] == 1234


class TestParseStreamEvent:
    """:func:`parse_stream_event` dispatches through the discriminator."""

    def test_parses_stream_start(self) -> None:
        """``type=stream_start`` payload returns :class:`StreamStartEvent`."""
        cid = _new_uuid()
        conv = _new_uuid()
        payload = (
            StreamStartEvent(
                correlation_id=cid,
                conversation_id=conv,
            )
            .model_dump_json()
            .encode("utf-8")
        )
        evt = parse_stream_event(payload)
        assert isinstance(evt, StreamStartEvent)
        assert evt.correlation_id == cid
        assert evt.conversation_id == conv

    def test_parses_stream_token(self) -> None:
        """``type=stream_token`` payload returns :class:`StreamTokenEvent`."""
        cid = _new_uuid()
        payload = (
            StreamTokenEvent(
                correlation_id=cid,
                token="hi",
            )
            .model_dump_json()
            .encode("utf-8")
        )
        evt = parse_stream_event(payload)
        assert isinstance(evt, StreamTokenEvent)
        assert evt.token == "hi"

    def test_parses_stream_end(self) -> None:
        """``type=stream_end`` payload returns :class:`StreamEndEvent`."""
        cid = _new_uuid()
        payload = (
            StreamEndEvent(
                correlation_id=cid,
                content="done",
            )
            .model_dump_json()
            .encode("utf-8")
        )
        evt = parse_stream_event(payload)
        assert isinstance(evt, StreamEndEvent)
        assert evt.content == "done"

    def test_parses_stream_error(self) -> None:
        """``type=stream_error`` payload returns :class:`StreamErrorEvent`."""
        cid = _new_uuid()
        payload = (
            StreamErrorEvent(
                correlation_id=cid,
                code="X",
                message="boom",
            )
            .model_dump_json()
            .encode("utf-8")
        )
        evt = parse_stream_event(payload)
        assert isinstance(evt, StreamErrorEvent)
        assert evt.code == "X"
        assert evt.message == "boom"

    def test_parses_tool_call_start(self) -> None:
        """``type=tool_call_start`` payload returns :class:`ToolCallStartEvent`."""
        payload = (
            ToolCallStartEvent(
                correlation_id=_new_uuid(),
                tool_name="search",
                arguments_summary="q=...",
            )
            .model_dump_json()
            .encode("utf-8")
        )
        evt = parse_stream_event(payload)
        assert isinstance(evt, ToolCallStartEvent)
        assert evt.tool_name == "search"

    def test_parses_tool_call_end(self) -> None:
        """``type=tool_call_end`` payload returns :class:`ToolCallEndEvent`."""
        payload = (
            ToolCallEndEvent(
                correlation_id=_new_uuid(),
                tool_name="search",
                success=True,
                elapsed_ms=12,
            )
            .model_dump_json()
            .encode("utf-8")
        )
        evt = parse_stream_event(payload)
        assert isinstance(evt, ToolCallEndEvent)
        assert evt.success is True
        assert evt.elapsed_ms == 12

    def test_parses_tool_call_progress(self) -> None:
        """``type=tool_call_progress`` payload returns :class:`ToolCallProgressEvent`."""
        payload = (
            ToolCallProgressEvent(
                correlation_id=_new_uuid(),
                tool_name="slow",
                elapsed_ms=2000,
                sequence=2,
            )
            .model_dump_json()
            .encode("utf-8")
        )
        evt = parse_stream_event(payload)
        assert isinstance(evt, ToolCallProgressEvent)
        assert evt.sequence == 2

    def test_rejects_unknown_type(self) -> None:
        """unknown ``type`` raises :class:`pydantic.ValidationError`."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            parse_stream_event(b'{"type": "unknown", "correlation_id": "x"}')


# ---------------------------------------------------------------------------
# StreamingResponse lifecycle tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStreamingResponseLifecycle:
    """:class:`StreamingResponse` enforces start -> emit* -> terminal order."""

    async def test_start_publishes_stream_start_once(self) -> None:
        """:meth:`start` emits exactly one :class:`StreamStartEvent`."""
        transport = _RecordingTransport()
        cid = _new_uuid()
        conv = _new_uuid()
        stream = StreamingResponse(
            transport=transport,
            correlation_id=cid,
            conversation_id=conv,
        )
        await stream.start()
        await stream.start()  # idempotent
        assert len(transport.events) == 1
        evt = transport.events[0]
        assert isinstance(evt, StreamStartEvent)
        assert evt.correlation_id == cid
        assert evt.conversation_id == conv

    async def test_emit_token_accumulates_and_publishes(self) -> None:
        """tokens emit in order and accumulate on the running buffer."""
        transport = _RecordingTransport()
        stream = StreamingResponse(
            transport=transport,
            correlation_id=_new_uuid(),
            conversation_id=_new_uuid(),
        )
        await stream.start()
        await stream.emit_token("Hello")
        await stream.emit_token(", world")
        token_events = [e for e in transport.events if isinstance(e, StreamTokenEvent)]
        assert [e.token for e in token_events] == ["Hello", ", world"]
        assert stream.accumulated_content == "Hello, world"

    async def test_emit_token_drops_empty_token(self) -> None:
        """empty-string tokens never reach the wire."""
        transport = _RecordingTransport()
        stream = StreamingResponse(
            transport=transport,
            correlation_id=_new_uuid(),
            conversation_id=_new_uuid(),
        )
        await stream.start()
        await stream.emit_token("")
        token_events = [e for e in transport.events if isinstance(e, StreamTokenEvent)]
        assert token_events == []

    async def test_emit_keepalive_publishes_turn_level_progress(self) -> None:
        """turn-level keepalive emits progress with EMPTY tool_name + monotonic seq (LRT-02)."""
        transport = _RecordingTransport()
        stream = StreamingResponse(
            transport=transport,
            correlation_id=_new_uuid(),
            conversation_id=_new_uuid(),
        )
        await stream.start()
        await stream.emit_keepalive(elapsed_ms=10000)
        await stream.emit_keepalive(elapsed_ms=20000)
        keepalives = [e for e in transport.events if isinstance(e, ToolCallProgressEvent)]
        # empty tool_name marks the turn-level keepalive (vs a per-tool progress).
        assert [e.tool_name for e in keepalives] == ["", ""]
        assert [e.sequence for e in keepalives] == [1, 2]
        assert [e.elapsed_ms for e in keepalives] == [10000, 20000]

    async def test_emit_keepalive_after_terminal_is_noop(self) -> None:
        """a keepalive after the stream closes is silently dropped (no late wire write)."""
        transport = _RecordingTransport()
        stream = StreamingResponse(
            transport=transport,
            correlation_id=_new_uuid(),
            conversation_id=_new_uuid(),
        )
        await stream.start()
        await stream.end()
        await stream.emit_keepalive(elapsed_ms=5000)
        keepalives = [e for e in transport.events if isinstance(e, ToolCallProgressEvent)]
        assert keepalives == []

    async def test_end_publishes_stream_end_with_accumulated_content(self) -> None:
        """:meth:`end` carries the running accumulated content."""
        transport = _RecordingTransport()
        stream = StreamingResponse(
            transport=transport,
            correlation_id=_new_uuid(),
            conversation_id=_new_uuid(),
        )
        await stream.start()
        await stream.emit_token("part1")
        await stream.emit_token(" part2")
        await stream.end()
        end_events = [e for e in transport.events if isinstance(e, StreamEndEvent)]
        assert len(end_events) == 1
        assert end_events[0].content == "part1 part2"
        assert stream.closed is True

    async def test_replace_content_sets_accumulated_buffer(self) -> None:
        """:meth:`replace_content` makes the terminal carry the new text.

        a draft that already streamed can be superseded by a revision or a
        blocked-query re-run; resetting the buffer makes
        :class:`StreamEndEvent.content` authoritative rather than stale.
        """
        transport = _RecordingTransport()
        stream = StreamingResponse(
            transport=transport,
            correlation_id=_new_uuid(),
            conversation_id=_new_uuid(),
        )
        await stream.start()
        await stream.emit_token("stale draft")
        stream.replace_content("revised answer")
        await stream.emit_token(" + footer")
        await stream.end()
        end_events = [e for e in transport.events if isinstance(e, StreamEndEvent)]
        assert end_events[0].content == "revised answer + footer"

    async def test_replace_content_clears_for_rerun(self) -> None:
        """:meth:`replace_content` with ``""`` clears ahead of a re-run."""
        transport = _RecordingTransport()
        stream = StreamingResponse(
            transport=transport,
            correlation_id=_new_uuid(),
            conversation_id=_new_uuid(),
        )
        await stream.start()
        await stream.emit_token("blocked draft")
        stream.replace_content("")
        await stream.emit_token("corrected answer")
        assert stream.accumulated_content == "corrected answer"

    async def test_replace_content_noop_when_closed(self) -> None:
        """:meth:`replace_content` no-ops after a terminal has fired."""
        transport = _RecordingTransport()
        stream = StreamingResponse(
            transport=transport,
            correlation_id=_new_uuid(),
            conversation_id=_new_uuid(),
        )
        await stream.start()
        await stream.emit_token("final")
        await stream.end()
        stream.replace_content("too late")
        assert stream.accumulated_content == "final"

    async def test_end_is_idempotent(self) -> None:
        """second :meth:`end` call no-ops without a wire emit."""
        transport = _RecordingTransport()
        stream = StreamingResponse(
            transport=transport,
            correlation_id=_new_uuid(),
            conversation_id=_new_uuid(),
        )
        await stream.start()
        await stream.end()
        await stream.end()
        end_events = [e for e in transport.events if isinstance(e, StreamEndEvent)]
        assert len(end_events) == 1

    async def test_error_publishes_stream_error(self) -> None:
        """:meth:`error` emits one :class:`StreamErrorEvent`."""
        transport = _RecordingTransport()
        stream = StreamingResponse(
            transport=transport,
            correlation_id=_new_uuid(),
            conversation_id=_new_uuid(),
        )
        await stream.start()
        await stream.error(code="AGENT_FAILED", message="boom")
        err_events = [e for e in transport.events if isinstance(e, StreamErrorEvent)]
        assert len(err_events) == 1
        assert err_events[0].code == "AGENT_FAILED"
        assert err_events[0].message == "boom"
        assert stream.closed is True

    async def test_error_is_idempotent(self) -> None:
        """second :meth:`error` call no-ops without a wire emit."""
        transport = _RecordingTransport()
        stream = StreamingResponse(
            transport=transport,
            correlation_id=_new_uuid(),
            conversation_id=_new_uuid(),
        )
        await stream.start()
        await stream.error(code="X", message="m")
        await stream.error(code="X", message="m")
        err_events = [e for e in transport.events if isinstance(e, StreamErrorEvent)]
        assert len(err_events) == 1

    async def test_end_after_error_raises(self) -> None:
        """:meth:`end` after :meth:`error` raises the lifecycle error."""
        transport = _RecordingTransport()
        stream = StreamingResponse(
            transport=transport,
            correlation_id=_new_uuid(),
            conversation_id=_new_uuid(),
        )
        await stream.start()
        await stream.error(code="X", message="m")
        with pytest.raises(StreamingResponseError):
            await stream.end()

    async def test_error_after_end_raises(self) -> None:
        """:meth:`error` after :meth:`end` raises the lifecycle error."""
        transport = _RecordingTransport()
        stream = StreamingResponse(
            transport=transport,
            correlation_id=_new_uuid(),
            conversation_id=_new_uuid(),
        )
        await stream.start()
        await stream.end()
        with pytest.raises(StreamingResponseError):
            await stream.error(code="X", message="m")

    async def test_emits_after_terminal_no_op(self) -> None:
        """token/tool emits after a terminal silently no-op."""
        transport = _RecordingTransport()
        stream = StreamingResponse(
            transport=transport,
            correlation_id=_new_uuid(),
            conversation_id=_new_uuid(),
        )
        await stream.start()
        await stream.end()
        # these should not reach the wire and should not raise.
        await stream.emit_token("late")
        await stream.emit_tool_call_start(tool_name="t", arguments_summary="x")
        await stream.emit_tool_call_progress(tool_name="t", elapsed_ms=1, sequence=1)
        await stream.emit_tool_call_end(tool_name="t", success=True, elapsed_ms=1)
        # still only the start + end on the wire.
        kinds = [type(e).__name__ for e in transport.events]
        assert kinds == ["StreamStartEvent", "StreamEndEvent"]

    async def test_tool_call_start_progress_end_emits(self) -> None:
        """tool-call observation events round-trip in order."""
        transport = _RecordingTransport()
        stream = StreamingResponse(
            transport=transport,
            correlation_id=_new_uuid(),
            conversation_id=_new_uuid(),
        )
        await stream.start()
        await stream.emit_tool_call_start(tool_name="search", arguments_summary="q=...")
        await stream.emit_tool_call_progress(tool_name="search", elapsed_ms=1000, sequence=1)
        await stream.emit_tool_call_progress(tool_name="search", elapsed_ms=2000, sequence=2)
        await stream.emit_tool_call_end(tool_name="search", success=True, elapsed_ms=2500)
        await stream.end()

        kinds = [type(e).__name__ for e in transport.events]
        assert kinds == [
            "StreamStartEvent",
            "ToolCallStartEvent",
            "ToolCallProgressEvent",
            "ToolCallProgressEvent",
            "ToolCallEndEvent",
            "StreamEndEvent",
        ]

    async def test_duration_ms_uses_explicit_start_time(self) -> None:
        """``duration_ms`` is computed from the supplied monotonic value."""
        transport = _RecordingTransport()
        backdated = time.monotonic() - 1.5
        stream = StreamingResponse(
            transport=transport,
            correlation_id=_new_uuid(),
            conversation_id=_new_uuid(),
            start_time_monotonic=backdated,
        )
        await stream.start()
        await stream.end()
        end_events = [e for e in transport.events if isinstance(e, StreamEndEvent)]
        assert end_events[0].duration_ms >= 1400
        assert end_events[0].duration_ms < 3000

    async def test_publish_failure_does_not_break_lifecycle(self) -> None:
        """a transport error during emit is swallowed; lifecycle continues.

        regression guard: a wire failure mid-stream must not stop the
        terminal from firing on the same instance. callers depend on
        the terminal as the lifecycle exit signal.
        """
        transport = _RecordingTransport()
        stream = StreamingResponse(
            transport=transport,
            correlation_id=_new_uuid(),
            conversation_id=_new_uuid(),
        )
        await stream.start()
        # fail one token publish, then keep going.
        transport.publish_failure = RuntimeError("nats down")
        await stream.emit_token("dropped")
        transport.publish_failure = None
        await stream.emit_token("survived")
        await stream.end()

        token_events = [e for e in transport.events if isinstance(e, StreamTokenEvent)]
        # the failed publish never reached transport.payloads, but the
        # accumulated content includes both tokens.
        assert [e.token for e in token_events] == ["survived"]
        assert stream.accumulated_content == "droppedsurvived"
        end_events = [e for e in transport.events if isinstance(e, StreamEndEvent)]
        assert len(end_events) == 1


# ---------------------------------------------------------------------------
# StreamingResponse.run_graph tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStreamingResponseRunGraph:
    """:meth:`StreamingResponse.run_graph` consume loop semantics."""

    async def test_emits_start_tokens_end_in_order(self) -> None:
        """on success path: stream_start, tokens, stream_end, return state."""
        transport = _RecordingTransport()
        stream = StreamingResponse(
            transport=transport,
            correlation_id=_new_uuid(),
            conversation_id=_new_uuid(),
        )
        graph = _StubGraph(
            [
                {"event": "on_chat_model_stream", "data": {"chunk": _MockChunk("Hi")}},
                {"event": "on_chat_model_stream", "data": {"chunk": _MockChunk(" there")}},
                {"event": "on_chain_end", "data": {"output": {"summary": "ok"}}},
            ]
        )
        result = await stream.run_graph(graph, {"messages": []}, {})

        kinds = [type(e).__name__ for e in transport.events]
        assert kinds == [
            "StreamStartEvent",
            "StreamTokenEvent",
            "StreamTokenEvent",
            "StreamEndEvent",
        ]
        end_evt = [e for e in transport.events if isinstance(e, StreamEndEvent)][0]
        assert end_evt.content == "Hi there"
        assert result["summary"] == "ok"

    async def test_run_graph_accepts_a_command_resume_without_crashing(self) -> None:
        """``run_graph`` drives a resume with a ``Command`` (not a dict) — the streaming twin of
        ``ainvoke(Command(resume=...))`` for a checkpointed graph paused at an ``interrupt``.

        Regression for the HITL-resume crash: ``dict(Command)`` raises
        ``TypeError: 'Command' object is not iterable``, so the old ``final_state = dict(state)``
        crashed on EVERY confirm-mode resume. The guard seeds ``{}`` for a non-mapping state; the
        authoritative result still comes from the merged ``on_chain_end`` output.
        """
        transport = _RecordingTransport()
        stream = StreamingResponse(
            transport=transport,
            correlation_id=_new_uuid(),
            conversation_id=_new_uuid(),
        )
        graph = _StubGraph(
            [
                {"event": "on_chat_model_stream", "data": {"chunk": _MockChunk("done")}},
                {"event": "on_chain_end", "data": {"output": {"applied": True}}},
            ]
        )

        result = await stream.run_graph(graph, Command(resume="approve"), {})

        assert result["applied"] is True, "the resume's on_chain_end output is the authoritative result"
        kinds = [type(e).__name__ for e in transport.events]
        assert kinds == ["StreamStartEvent", "StreamTokenEvent", "StreamEndEvent"]

    async def test_drops_tokens_tagged_nostream(self) -> None:
        """``on_chat_model_stream`` events tagged NOSTREAM_TAG are dropped.

        regression guard: an internal model call inside a node (an
        adversarial reviewer refuting a draft) emits chat-model tokens that
        the run loop would otherwise accumulate verbatim into the
        user-facing answer. tagging the call NOSTREAM_TAG must keep its
        tokens out of the stream while the untagged answer tokens pass.
        """
        transport = _RecordingTransport()
        stream = StreamingResponse(
            transport=transport,
            correlation_id=_new_uuid(),
            conversation_id=_new_uuid(),
        )
        graph = _StubGraph(
            [
                {"event": "on_chat_model_stream", "data": {"chunk": _MockChunk("answer")}},
                {
                    "event": "on_chat_model_stream",
                    "tags": [NOSTREAM_TAG],
                    "data": {"chunk": _MockChunk("REVIEWER LEAK")},
                },
                {"event": "on_chat_model_stream", "data": {"chunk": _MockChunk(" tail")}},
            ]
        )
        await stream.run_graph(graph, {"messages": []}, {})

        end_evt = [e for e in transport.events if isinstance(e, StreamEndEvent)][0]
        assert end_evt.content == "answer tail"
        token_texts = [e.token for e in transport.events if isinstance(e, StreamTokenEvent)]
        assert "REVIEWER LEAK" not in token_texts

    async def test_emits_stream_error_on_graph_exception(self) -> None:
        """graph exception fires :class:`StreamErrorEvent`, not empty end.

        regression guard for the UX bug:
        ``StreamEndEvent(content="")`` looks indistinguishable from a
        successful empty response. the new contract puts the failure
        on a typed error envelope so consumers branch on it.
        """
        transport = _RecordingTransport()
        stream = StreamingResponse(
            transport=transport,
            correlation_id=_new_uuid(),
            conversation_id=_new_uuid(),
        )
        graph = _StubGraph(
            [
                {"event": "on_chat_model_stream", "data": {"chunk": _MockChunk("partial")}},
                RuntimeError("LLM exploded"),
            ]
        )

        with pytest.raises(RuntimeError, match="LLM exploded"):
            await stream.run_graph(graph, {"messages": []}, {})

        # exactly one terminal: stream_error, NOT stream_end.
        end_events = [e for e in transport.events if isinstance(e, StreamEndEvent)]
        err_events = [e for e in transport.events if isinstance(e, StreamErrorEvent)]
        assert end_events == []
        assert len(err_events) == 1
        assert err_events[0].code == "AGENT_FAILED"
        assert "LLM exploded" in err_events[0].message

    async def test_emits_stream_error_on_immediate_failure(self) -> None:
        """graph that raises before yielding still fires error terminal."""
        transport = _RecordingTransport()
        stream = StreamingResponse(
            transport=transport,
            correlation_id=_new_uuid(),
            conversation_id=_new_uuid(),
        )
        graph = _StubGraph(RuntimeError("dead"))

        with pytest.raises(RuntimeError):
            await stream.run_graph(graph, {}, {})

        kinds = [type(e).__name__ for e in transport.events]
        assert kinds == ["StreamStartEvent", "StreamErrorEvent"]

    async def test_emits_stream_error_on_cancelled(self) -> None:
        """asyncio cancellation fires error terminal then re-raises."""
        transport = _RecordingTransport()
        stream = StreamingResponse(
            transport=transport,
            correlation_id=_new_uuid(),
            conversation_id=_new_uuid(),
        )
        graph = _StubGraph(
            [
                {"event": "on_chat_model_stream", "data": {"chunk": _MockChunk("partial")}},
                asyncio.CancelledError(),
            ]
        )

        with pytest.raises(asyncio.CancelledError):
            await stream.run_graph(graph, {}, {})

        err_events = [e for e in transport.events if isinstance(e, StreamErrorEvent)]
        assert len(err_events) == 1
        assert err_events[0].code == "AGENT_CANCELLED"

    async def test_handles_v2_chunk_under_data_chunk_key(self) -> None:
        """LangGraph v2 chunk shape ``data["chunk"]`` extracts cleanly."""
        transport = _RecordingTransport()
        stream = StreamingResponse(
            transport=transport,
            correlation_id=_new_uuid(),
            conversation_id=_new_uuid(),
        )
        graph = _StubGraph(
            [
                {"event": "on_chat_model_stream", "data": {"chunk": _MockChunk("a")}},
                {"event": "on_chat_model_stream", "data": {"chunk": _MockChunk("b")}},
            ]
        )
        await stream.run_graph(graph, {}, {})
        end_evt = [e for e in transport.events if isinstance(e, StreamEndEvent)][0]
        assert end_evt.content == "ab"

    async def test_handles_legacy_bare_chunk_under_data(self) -> None:
        """legacy mock shape with bare chunk under ``data`` still works."""
        transport = _RecordingTransport()
        stream = StreamingResponse(
            transport=transport,
            correlation_id=_new_uuid(),
            conversation_id=_new_uuid(),
        )
        graph = _StubGraph(
            [
                {"event": "on_chat_model_stream", "data": _MockChunk("legacy")},
            ]
        )
        await stream.run_graph(graph, {}, {})
        end_evt = [e for e in transport.events if isinstance(e, StreamEndEvent)][0]
        assert end_evt.content == "legacy"


# ---------------------------------------------------------------------------
# StreamTransport Protocol satisfaction
# ---------------------------------------------------------------------------


def test_recording_transport_satisfies_protocol() -> None:
    """:class:`_RecordingTransport` is a structural :class:`StreamTransport`."""
    t: StreamTransport = _RecordingTransport()
    assert hasattr(t, "publish")
