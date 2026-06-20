"""canonical streaming wire types and :class:`StreamingResponse` runtime.

before this module landed, the platform carried three near-identical
copies of the streaming wire vocabulary -- one in the aibots SDK
publisher, one as a hand-mirror in the hub consumer, and one as the
hub's bridge projection. every change had to land three times and
the wire shape silently drifted. this module is the single source of
truth for the wire envelopes any 3tears app sends or receives on a
streaming token channel.

the wire vocabulary is transport-agnostic: the same envelopes flow
over NATS today (aibots) and could ride a websocket tomorrow
(metallm) with no envelope changes. the integration seam is the
:class:`StreamTransport` Protocol -- a one-method
``async def publish(self, payload: bytes) -> None`` interface that
any concrete transport adapter satisfies (see the
:class:`NatsStreamTransport` adapter in ``aibots_agents.runtime``
for the reference implementation).

:class:`StreamingResponse` is the lifecycle-aware emitter that owns
``start`` -> tokens / tool-call observations -> mutually-exclusive
terminal (``end`` or ``error``). callers either drive the lifecycle
explicitly (``stream.start(); stream.emit_token(...); stream.end()``)
or call :meth:`StreamingResponse.run_graph` to consume a LangGraph
``astream_events`` v2 loop with the start/end ordering managed for
them. the convenience method routes exceptions to
:meth:`StreamingResponse.error` instead of an empty ``end`` so client
UIs render a real error event rather than an empty done.
"""

from __future__ import annotations

import asyncio
import time
from typing import Annotated, Any, Literal, Protocol, Union, runtime_checkable
from uuid import UUID

from langgraph.types import Command
from pydantic import BaseModel, Field, TypeAdapter
from threetears.observe import get_logger, traced

__all__ = [
    "StreamStartEvent",
    "StreamTokenEvent",
    "StreamEndEvent",
    "StreamErrorEvent",
    "ToolCallStartEvent",
    "ToolCallEndEvent",
    "ToolCallProgressEvent",
    "StreamEvent",
    "parse_stream_event",
    "StreamTransport",
    "StreamingResponse",
    "StreamingResponseError",
]

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Wire types -- single source of truth for the streaming envelope shapes.
# every envelope carries a Literal ``type`` discriminator so the
# :class:`pydantic.TypeAdapter` over the union below can dispatch on it
# without ambiguity. correlation_id and conversation_id are kept as
# strings on the wire to keep parse/serialize symmetrical with the
# JSON border; the in-process construction sites pass
# :class:`uuid.UUID` and rely on pydantic's UUID->str coercion.
# ---------------------------------------------------------------------------


class StreamStartEvent(BaseModel):
    """terminal lifecycle event marking the beginning of a stream.

    published exactly once per :class:`StreamingResponse` instance,
    immediately before the first token (or first tool-call observation).

    :param type: discriminator literal
    :ptype type: Literal["stream_start"]
    :param correlation_id: request trace identifier
    :ptype correlation_id: UUID
    :param conversation_id: conversation identifier
    :ptype conversation_id: UUID
    """

    type: Literal["stream_start"] = "stream_start"
    correlation_id: UUID
    conversation_id: UUID


class StreamTokenEvent(BaseModel):
    """single-token delta event published during model streaming.

    no per-token validation overhead; payload is built and serialized
    in a single :meth:`pydantic.BaseModel.model_dump_json` call per
    token (per design DQ-F3 -- per-token gold-plating is rejected).

    :param type: discriminator literal
    :ptype type: Literal["stream_token"]
    :param correlation_id: request trace identifier
    :ptype correlation_id: UUID
    :param token: token text content
    :ptype token: str
    """

    type: Literal["stream_token"] = "stream_token"
    correlation_id: UUID
    token: str


class StreamEndEvent(BaseModel):
    """successful terminal event with accumulated content.

    published exactly once per :class:`StreamingResponse` instance
    when the stream completes without error. ``content`` carries the
    full accumulated text across every preceding
    :class:`StreamTokenEvent`. ``duration_ms`` carries the
    wall-clock time from the caller-supplied
    ``start_time_monotonic`` (or from the lifecycle ``start()``
    call when none was supplied). consumers preferentially read this
    field over any dispatch-reply timing because dispatch-reply timing
    is zero on ack-early streaming flows.

    mutually exclusive with :class:`StreamErrorEvent` -- never both
    on the same correlation_id.

    :param type: discriminator literal
    :ptype type: Literal["stream_end"]
    :param correlation_id: request trace identifier
    :ptype correlation_id: UUID
    :param content: complete accumulated response content
    :ptype content: str
    :param duration_ms: wall-clock milliseconds for the stream
    :ptype duration_ms: int
    """

    type: Literal["stream_end"] = "stream_end"
    correlation_id: UUID
    content: str
    duration_ms: int = 0


class StreamErrorEvent(BaseModel):
    """failure terminal event replacing the empty-end UX bug.

    before this envelope existed, exceptions during streaming
    surfaced as a :class:`StreamEndEvent` with empty ``content`` --
    the client could not distinguish a successful empty response
    from a failure mid-stream and rendered a blank reply with no
    diagnostic. this envelope carries an explicit error code and a
    human-readable message; the consumer surface (SSE bridge,
    websocket forwarder) translates it into a true error frame.

    mutually exclusive with :class:`StreamEndEvent` -- never both
    on the same correlation_id.

    :param type: discriminator literal
    :ptype type: Literal["stream_error"]
    :param correlation_id: request trace identifier
    :ptype correlation_id: UUID
    :param code: stable error code identifier (UPPER_SNAKE_CASE)
    :ptype code: str
    :param message: human-readable failure description
    :ptype message: str
    :param duration_ms: wall-clock milliseconds elapsed before failure
    :ptype duration_ms: int
    """

    type: Literal["stream_error"] = "stream_error"
    correlation_id: UUID
    code: str
    message: str
    duration_ms: int = 0


class ToolCallStartEvent(BaseModel):
    """observation event published when a tool call begins.

    fire-and-forget observation; not a lifecycle terminal. one
    :class:`ToolCallStartEvent` is paired with one
    :class:`ToolCallEndEvent` per tool invocation, with zero or more
    :class:`ToolCallProgressEvent` heartbeats in between.

    :param type: discriminator literal
    :ptype type: Literal["tool_call_start"]
    :param correlation_id: request trace identifier
    :ptype correlation_id: UUID
    :param tool_name: name of tool being invoked
    :ptype tool_name: str
    :param arguments_summary: truncated summary of tool arguments
    :ptype arguments_summary: str
    """

    type: Literal["tool_call_start"] = "tool_call_start"
    correlation_id: UUID
    tool_name: str
    arguments_summary: str


class ToolCallEndEvent(BaseModel):
    """observation event published when a tool call completes.

    :param type: discriminator literal
    :ptype type: Literal["tool_call_end"]
    :param correlation_id: request trace identifier
    :ptype correlation_id: UUID
    :param tool_name: name of tool that was invoked
    :ptype tool_name: str
    :param success: whether tool execution succeeded
    :ptype success: bool
    :param elapsed_ms: tool execution time in milliseconds
    :ptype elapsed_ms: int
    """

    type: Literal["tool_call_end"] = "tool_call_end"
    correlation_id: UUID
    tool_name: str
    success: bool
    elapsed_ms: int


class ToolCallProgressEvent(BaseModel):
    """keepalive event published periodically during a long tool call.

    emitted between a :class:`ToolCallStartEvent` and the matching
    :class:`ToolCallEndEvent` so consumers running gap timers do not
    trip a stream-timeout error during a slow tool. carries only a
    monotonic sequence counter and elapsed time so no tool-internal
    payload leaks; the hook owns sequence semantics
    (per design DQ-F4 -- emit_tool_call_progress is a passthrough).

    :param type: discriminator literal
    :ptype type: Literal["tool_call_progress"]
    :param correlation_id: request trace identifier
    :ptype correlation_id: UUID
    :param tool_name: name of tool currently executing
    :ptype tool_name: str
    :param elapsed_ms: wall-clock milliseconds since dispatch began
    :ptype elapsed_ms: int
    :param sequence: monotonic 1-indexed counter within this tool call
    :ptype sequence: int
    """

    type: Literal["tool_call_progress"] = "tool_call_progress"
    correlation_id: UUID
    tool_name: str
    elapsed_ms: int
    sequence: int


StreamEvent = Annotated[
    Union[
        StreamStartEvent,
        StreamTokenEvent,
        StreamEndEvent,
        StreamErrorEvent,
        ToolCallStartEvent,
        ToolCallEndEvent,
        ToolCallProgressEvent,
    ],
    Field(discriminator="type"),
]
"""discriminated union of every wire event a streaming consumer parses.

dispatch is by the literal ``type`` field. callers MUST validate
inbound payloads through :func:`parse_stream_event` rather than
``model_validate_json`` on a single concrete class so an unknown
type surfaces as a :class:`pydantic.ValidationError` instead of being
silently misparsed as the first member of the union.
"""


_stream_event_adapter: TypeAdapter[StreamEvent] = TypeAdapter(StreamEvent)


def parse_stream_event(payload: bytes | str) -> StreamEvent:
    """parse a stream-event wire payload into a typed envelope.

    dispatches by the discriminator literal on ``type``. callers
    receive a concrete subclass (one of :class:`StreamStartEvent`,
    :class:`StreamTokenEvent`, :class:`StreamEndEvent`,
    :class:`StreamErrorEvent`, :class:`ToolCallStartEvent`,
    :class:`ToolCallEndEvent`, :class:`ToolCallProgressEvent`).

    :param payload: raw JSON bytes or string from the stream transport
    :ptype payload: bytes | str
    :return: typed wire-event envelope
    :rtype: StreamEvent
    :raises ValidationError: if payload does not match any known event
    """
    result: StreamEvent = _stream_event_adapter.validate_json(payload)
    return result


# ---------------------------------------------------------------------------
# Transport seam.
# ---------------------------------------------------------------------------


@runtime_checkable
class StreamTransport(Protocol):
    """publishes raw bytes to a streaming destination.

    deliberately minimal so any wire (NATS subject, websocket,
    chunked HTTP body) satisfies the contract by exposing one async
    publish method. concrete adapters bind a destination address at
    construction time and accept already-serialized payload bytes
    on every publish call (so per-token serialization happens once,
    inside :class:`StreamingResponse`, not inside the transport).

    publish failures during streaming are observability concerns,
    not lifecycle terminals -- the transport may raise but the
    :class:`StreamingResponse` lifecycle continues unchanged.
    """

    async def publish(self, payload: bytes) -> None:
        """publish one pre-serialized payload to the stream destination.

        :param payload: serialized event bytes ready for transport
        :ptype payload: bytes
        :return: nothing
        :rtype: None
        """
        ...


# ---------------------------------------------------------------------------
# Lifecycle-aware emitter.
# ---------------------------------------------------------------------------


class StreamingResponseError(RuntimeError):
    """raised on lifecycle violations of :class:`StreamingResponse`.

    a lifecycle violation is a state transition the contract forbids
    (for example, calling :meth:`StreamingResponse.emit_token` after
    :meth:`StreamingResponse.end`). idempotent transitions
    (start-after-start, end-after-end, error-after-error) are NOT
    violations and silently no-op.
    """


class StreamingResponse:
    """lifecycle-aware emitter for one streaming response.

    contract:

    - exactly one :meth:`start` per instance (idempotent; second
      call no-ops).
    - any number of :meth:`emit_token`, :meth:`emit_tool_call_start`,
      :meth:`emit_tool_call_progress`, :meth:`emit_tool_call_end`
      between :meth:`start` and the terminal.
    - exactly one terminal: :meth:`end` (success) OR :meth:`error`
      (failure). idempotent; calling the same terminal twice no-ops.
      calling the OTHER terminal after either has fired raises
      :class:`StreamingResponseError`.
    - emits AFTER a terminal silently no-op (the consumer has stopped
      reading; logging at DEBUG keeps loki readable).

    serialization is one ``model_dump_json().encode("utf-8")`` per
    emit (per design DQ-F3 -- per-token gold-plating is rejected).
    transport publish failures are caught and logged but do NOT
    propagate; the lifecycle continues and the terminal still fires.

    :ivar accumulated_content: running concatenation of every emitted
        token. exposed as a property so the terminal :meth:`end` can
        carry the full content without callers managing a buffer.
    :ivar closed: ``True`` after a terminal fires; emit methods become
        silent no-ops.
    """

    def __init__(
        self,
        *,
        transport: StreamTransport,
        correlation_id: UUID,
        conversation_id: UUID,
        start_time_monotonic: float | None = None,
    ) -> None:
        """initialize a streaming response bound to one transport target.

        :param transport: pre-bound transport adapter (one destination)
        :ptype transport: StreamTransport
        :param correlation_id: request trace identifier (carried on
            every emitted envelope)
        :ptype correlation_id: UUID
        :param conversation_id: conversation identifier (carried on
            the start envelope)
        :ptype conversation_id: UUID
        :param start_time_monotonic: optional ``time.monotonic()`` value
            from request-receipt time; used to compute ``duration_ms``
            on the terminal envelope. defaults to ``None`` which falls
            back to the moment :meth:`start` is called
        :ptype start_time_monotonic: float | None
        """
        self._transport = transport
        self._correlation_id = correlation_id
        self._conversation_id = conversation_id
        self._explicit_start_monotonic = start_time_monotonic
        self._effective_start_monotonic: float | None = None
        self._accumulated_content: str = ""
        self._started: bool = False
        self._closed: bool = False
        self._terminal_kind: str | None = None  # "end" or "error" once closed

    @property
    def closed(self) -> bool:
        """return ``True`` once a terminal envelope has fired.

        :return: terminal-fired flag
        :rtype: bool
        """
        return self._closed

    @property
    def accumulated_content(self) -> str:
        """return the running concatenation of emitted tokens.

        the terminal :meth:`end` carries this value as
        :class:`StreamEndEvent.content`. callers MAY read it before
        the terminal fires (e.g. to render in-progress content on
        retry) but the value is only guaranteed final once
        :attr:`closed` is ``True``.

        :return: accumulated content string
        :rtype: str
        """
        return self._accumulated_content

    @traced
    async def start(self) -> None:
        """publish the lifecycle :class:`StreamStartEvent` once.

        idempotent: second and later calls return without publishing.
        records the ``start_time_monotonic`` reference (caller-supplied
        or ``time.monotonic()`` at this call) so the terminal can
        compute ``duration_ms``.

        :return: nothing
        :rtype: None
        """
        if self._started:
            return
        self._started = True
        self._effective_start_monotonic = (
            self._explicit_start_monotonic if self._explicit_start_monotonic is not None else time.monotonic()
        )
        event = StreamStartEvent(
            correlation_id=self._correlation_id,
            conversation_id=self._conversation_id,
        )
        await self._publish(event)

    async def emit_token(self, token: str) -> None:
        """publish one :class:`StreamTokenEvent` and accumulate the text.

        empty tokens are silently dropped (no on-the-wire noise for
        empty model deltas). emits after a terminal silently no-op.

        :param token: token text to publish and accumulate
        :ptype token: str
        :return: nothing
        :rtype: None
        """
        if self._closed:
            return
        if not token:
            return
        self._accumulated_content += token
        event = StreamTokenEvent(
            correlation_id=self._correlation_id,
            token=token,
        )
        await self._publish(event)

    async def emit_tool_call_start(
        self,
        *,
        tool_name: str,
        arguments_summary: str,
    ) -> None:
        """publish one :class:`ToolCallStartEvent`.

        :param tool_name: name of tool being invoked
        :ptype tool_name: str
        :param arguments_summary: truncated argument summary
        :ptype arguments_summary: str
        :return: nothing
        :rtype: None
        """
        if self._closed:
            return
        event = ToolCallStartEvent(
            correlation_id=self._correlation_id,
            tool_name=tool_name,
            arguments_summary=arguments_summary,
        )
        await self._publish(event)

    async def emit_tool_call_end(
        self,
        *,
        tool_name: str,
        success: bool,
        elapsed_ms: int,
    ) -> None:
        """publish one :class:`ToolCallEndEvent`.

        :param tool_name: name of tool that was invoked
        :ptype tool_name: str
        :param success: whether tool execution succeeded
        :ptype success: bool
        :param elapsed_ms: tool execution time in milliseconds
        :ptype elapsed_ms: int
        :return: nothing
        :rtype: None
        """
        if self._closed:
            return
        event = ToolCallEndEvent(
            correlation_id=self._correlation_id,
            tool_name=tool_name,
            success=success,
            elapsed_ms=elapsed_ms,
        )
        await self._publish(event)

    async def emit_tool_call_progress(
        self,
        *,
        tool_name: str,
        elapsed_ms: int,
        sequence: int,
    ) -> None:
        """publish one :class:`ToolCallProgressEvent` (passthrough).

        the hook owns sequence semantics (per design DQ-F4); this
        method does not advance any counter and trusts the caller's
        ``sequence`` value verbatim.

        :param tool_name: name of tool currently executing
        :ptype tool_name: str
        :param elapsed_ms: milliseconds since tool dispatch began
        :ptype elapsed_ms: int
        :param sequence: caller-managed monotonic 1-indexed counter
        :ptype sequence: int
        :return: nothing
        :rtype: None
        """
        if self._closed:
            return
        event = ToolCallProgressEvent(
            correlation_id=self._correlation_id,
            tool_name=tool_name,
            elapsed_ms=elapsed_ms,
            sequence=sequence,
        )
        await self._publish(event)

    @traced
    async def end(self) -> None:
        """publish the success terminal :class:`StreamEndEvent` once.

        idempotent if a previous :meth:`end` already fired -- second
        and later calls no-op. raises
        :class:`StreamingResponseError` if a previous :meth:`error`
        already fired (lifecycle terminals are mutually exclusive).

        :return: nothing
        :rtype: None
        :raises StreamingResponseError: when ``error`` already fired
        """
        if self._closed:
            if self._terminal_kind == "end":
                return
            raise StreamingResponseError(
                "cannot call end() after error() fired on this stream",
            )
        self._closed = True
        self._terminal_kind = "end"
        duration_ms = self._compute_duration_ms()
        event = StreamEndEvent(
            correlation_id=self._correlation_id,
            content=self._accumulated_content,
            duration_ms=duration_ms,
        )
        await self._publish(event)

    @traced
    async def error(self, *, code: str, message: str) -> None:
        """publish the failure terminal :class:`StreamErrorEvent` once.

        idempotent if a previous :meth:`error` already fired -- second
        and later calls no-op. raises
        :class:`StreamingResponseError` if a previous :meth:`end`
        already fired (lifecycle terminals are mutually exclusive).

        :param code: stable error code identifier (UPPER_SNAKE_CASE)
        :ptype code: str
        :param message: human-readable failure description
        :ptype message: str
        :return: nothing
        :rtype: None
        :raises StreamingResponseError: when ``end`` already fired
        """
        if self._closed:
            if self._terminal_kind == "error":
                return
            raise StreamingResponseError(
                "cannot call error() after end() fired on this stream",
            )
        self._closed = True
        self._terminal_kind = "error"
        duration_ms = self._compute_duration_ms()
        event = StreamErrorEvent(
            correlation_id=self._correlation_id,
            code=code,
            message=message,
            duration_ms=duration_ms,
        )
        await self._publish(event)

    async def run_graph(
        self,
        compiled_graph: Any,
        state: dict[str, Any] | Command[Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """execute a LangGraph with managed start/end ordering.

        consumes ``compiled_graph.astream_events(state, config=config,
        version="v2")`` end-to-end, calling :meth:`start` before the
        first event and the appropriate terminal afterward. tokens
        (``on_chat_model_stream``) accumulate via :meth:`emit_token`;
        chain output (``on_chain_end``) merges into the returned final
        state. on exception, fires :meth:`error` with code
        ``AGENT_FAILED`` (per design DQ-F2 -- no empty-end on failure)
        and re-raises so the caller can surface the failure on
        non-stream paths (logging, metrics).

        ``state`` is any LangGraph entry payload: a state ``dict`` for a
        fresh run, OR a resume directive (e.g. ``langgraph.types.Command``)
        for a checkpointed graph paused at an ``interrupt`` -- the streaming
        twin of ``compiled.ainvoke(Command(resume=...))``. The returned
        ``final_state`` seeds from ``state`` only when it is a mapping (a
        resume directive carries no channel values to seed from); either way
        the merged ``on_chain_end`` output is the authoritative result.

        :param compiled_graph: compiled LangGraph object exposing
            ``astream_events(state, config=..., version="v2")``
        :ptype compiled_graph: Any
        :param state: initial state dict for a fresh run, or a resume
            directive (``Command``) for a paused (interrupted) graph
        :ptype state: dict[str, Any] | Command
        :param config: LangGraph runtime config dict
        :ptype config: dict[str, Any]
        :return: final state dict merged from initial state + chain
            output observed during execution
        :rtype: dict[str, Any]
        :raises Exception: re-raised after :meth:`error` fires so the
            caller still sees the failure on the synchronous path
        """
        await self.start()
        # seed only from a mapping state: a resume directive (Command) is not a
        # state dict (``dict(Command)`` raises) and carries no channel values to
        # seed -- the merged on_chain_end output below is the authoritative result.
        final_state: dict[str, Any] = dict(state) if isinstance(state, dict) else {}
        try:
            async for event in compiled_graph.astream_events(
                state,
                config=config,
                version="v2",
            ):
                event_kind = event.get("event")
                if event_kind == "on_chat_model_stream":
                    token = _extract_token(event)
                    if token:
                        await self.emit_token(token)
                elif event_kind == "on_chain_end":
                    raw_data = event.get("data", {})
                    if isinstance(raw_data, dict):
                        output = raw_data.get("output", {})
                        if isinstance(output, dict):
                            final_state.update(output)
        except asyncio.CancelledError:
            # propagate cancellation cleanly: emit error so the
            # consumer sees a terminal, then re-raise so the surrounding
            # task tree shuts down. no logging here -- cancellation is
            # not an error condition.
            await self.error(code="AGENT_CANCELLED", message="agent run cancelled")
            raise
        except Exception as exc:
            log.error(
                "streaming graph execution failed",
                extra={
                    "extra_data": {
                        "correlation_id": str(self._correlation_id),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                },
                exc_info=True,
            )
            await self.error(code="AGENT_FAILED", message=str(exc))
            raise
        else:
            await self.end()
        return final_state

    def _compute_duration_ms(self) -> int:
        """compute milliseconds elapsed since the recorded start time.

        falls back to ``0`` when :meth:`start` has not yet fired
        (defensive; the lifecycle should never reach a terminal
        without start, but the no-crash path is preferred).

        :return: duration in milliseconds
        :rtype: int
        """
        if self._effective_start_monotonic is None:
            return 0
        return int((time.monotonic() - self._effective_start_monotonic) * 1000)

    async def _publish(self, event: BaseModel) -> None:
        """serialize one envelope and hand it to the transport.

        per design DQ-F3, this is a single
        ``model_dump_json().encode("utf-8")`` per emit -- no caching,
        no per-field optimization. transport publish failures are
        logged at WARNING and do NOT propagate so the lifecycle
        continues; the terminal still fires even if the wire is down.

        :param event: typed envelope to serialize and publish
        :ptype event: BaseModel
        :return: nothing
        :rtype: None
        """
        payload = event.model_dump_json().encode("utf-8")
        try:
            await self._transport.publish(payload)
        except Exception as exc:
            log.warning(
                "stream transport publish failed",
                extra={
                    "extra_data": {
                        "correlation_id": str(self._correlation_id),
                        "event_type": getattr(event, "type", type(event).__name__),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                },
                exc_info=True,
            )


def _extract_token(event: dict[str, Any]) -> str:
    """extract the token text from an ``on_chat_model_stream`` event.

    LangGraph v2 ``astream_events`` delivers the AIMessageChunk under
    ``data["chunk"]``; older mocks (and a few test doubles still in
    the tree) hand a bare chunk under ``data``. tolerate both shapes.

    :param event: raw event dict from ``astream_events``
    :ptype event: dict[str, Any]
    :return: token text (empty string when no content present)
    :rtype: str
    """
    raw_data = event.get("data", {})
    if isinstance(raw_data, dict):
        chunk = raw_data.get("chunk")
    else:
        chunk = raw_data
    content = getattr(chunk, "content", None) if chunk is not None else None
    return content if isinstance(content, str) else ""
