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
    "NOSTREAM_TAG",
    "NO_INTERRUPT",
    "StreamStartEvent",
    "StreamTokenEvent",
    "StreamEndEvent",
    "StreamErrorEvent",
    "StreamInterruptEvent",
    "ToolCallStartEvent",
    "ToolCallEndEvent",
    "ToolCallProgressEvent",
    "StreamEvent",
    "parse_stream_event",
    "detect_interrupt",
    "render_interrupt_prompt",
    "StreamTransport",
    "StreamingResponse",
    "StreamingResponseError",
]

log = get_logger(__name__)

#: tag a node's INTERNAL model call carries (via
#: ``model.ainvoke(..., {"tags": [NOSTREAM_TAG]})``) when its tokens must
#: NOT reach the user's stream. :meth:`StreamingResponse.run_graph`
#: consumes ``astream_events`` indiscriminately -- every
#: ``on_chat_model_stream`` event accumulates into the user-facing answer
#: REGARDLESS of which node emitted it. an auxiliary model call (an
#: adversarial reviewer refuting a draft, a summarizer compacting history)
#: is not the agent's answer; without this opt-out its deliberation tokens
#: leak verbatim into what the user sees. the run loop drops any
#: ``on_chat_model_stream`` whose event ``tags`` include this marker.
NOSTREAM_TAG = "threetears:nostream"

#: sentinel distinguishing "the graph completed normally" from "the graph paused on an interrupt
#: whose payload happens to be ``None``": :func:`_detect_interrupt` returns this when there is NO
#: pending interrupt, so a genuine ``interrupt(None)`` is never mistaken for a normal completion.
_NO_INTERRUPT: Any = object()

#: public alias of :data:`_NO_INTERRUPT` so a NON-streaming consumer (the SDK's ``ainvoke`` path)
#: can compare against the SAME sentinel object the streaming detection uses -- single source of
#: truth, no second sentinel to drift. :func:`detect_interrupt` returns this when the graph
#: completed normally; identity (``is NO_INTERRUPT``) is the test, never equality.
NO_INTERRUPT: Any = _NO_INTERRUPT


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

    ``metadata`` carries turn-level metadata that is NOT part of the
    token stream itself -- it originates wherever the terminal is
    constructed (for an ack-early dispatch flow, the consumer enriches
    the terminal from the separate request-reply once the graph
    finishes). it is transport-agnostic and free-form; a consumer with
    no terminal metadata leaves it empty. aibots rides response
    metadata here (tool-call names, ``pending_grants``, ``status`` for
    the human-in-the-loop grant flow) so a streaming caller sees the
    same turn metadata a non-streaming reply carries.

    :param type: discriminator literal
    :ptype type: Literal["stream_end"]
    :param correlation_id: request trace identifier
    :ptype correlation_id: UUID
    :param content: complete accumulated response content
    :ptype content: str
    :param duration_ms: wall-clock milliseconds for the stream
    :ptype duration_ms: int
    :param conversation_id: conversation identifier, mirrored from the
        :class:`StreamStartEvent` so the terminal independently carries
        it (the start frame and the terminal are a redundant pair over a
        fire-and-forget transport; a consumer that misses one recovers
        the id from the other)
    :ptype conversation_id: UUID | None
    :param metadata: free-form turn-level metadata (empty when none)
    :ptype metadata: dict[str, Any]
    """

    type: Literal["stream_end"] = "stream_end"
    correlation_id: UUID
    content: str
    duration_ms: int = 0
    conversation_id: UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


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


class StreamInterruptEvent(BaseModel):
    """pause terminal event: the graph stopped on a human-in-the-loop ``interrupt(...)``.

    Published exactly once when the graph reaches a LangGraph dynamic ``interrupt(value)`` instead
    of running to completion -- the turn ends as a legitimate PAUSE, distinct from a success
    (:class:`StreamEndEvent`), a failure (:class:`StreamErrorEvent`), and the empty-end bug. Before
    this envelope existed an interrupted graph yielded an empty :class:`StreamEndEvent` and the turn
    silently hung (the consumer never surfaced the prompt). ``payload`` carries the value the node
    passed to ``interrupt(...)`` (e.g. an exploit-approval prompt) for the consumer to surface to
    the human; the graph resumes on a later turn via ``run_graph(Command(resume=...))`` against the
    same checkpointed thread.

    mutually exclusive with :class:`StreamEndEvent` / :class:`StreamErrorEvent` -- never two
    terminals on the same correlation_id.

    :param type: discriminator literal
    :ptype type: Literal["stream_interrupt"]
    :param correlation_id: request trace identifier
    :ptype correlation_id: UUID
    :param payload: the value the node passed to ``interrupt(...)`` (the prompt to resolve)
    :ptype payload: Any
    :param conversation_id: conversation identifier, mirrored from the start frame
    :ptype conversation_id: UUID | None
    :param duration_ms: wall-clock milliseconds elapsed before the pause
    :ptype duration_ms: int
    :param metadata: free-form turn-level metadata (empty when none)
    :ptype metadata: dict[str, Any]
    """

    type: Literal["stream_interrupt"] = "stream_interrupt"
    correlation_id: UUID
    payload: Any
    conversation_id: UUID | None = None
    duration_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    """keepalive event so consumers running gap timers do not trip.

    serves two roles, both pure liveness:

    - per-tool progress: emitted between a :class:`ToolCallStartEvent` and
      the matching :class:`ToolCallEndEvent` while a slow tool runs, carrying
      that tool's ``tool_name`` (``emit_tool_call_progress``);
    - turn-level keepalive (LRT-02): emitted on a fixed interval for the whole
      turn with an EMPTY ``tool_name``, covering the agent-node
      time-to-first-token wait and between-node gaps that the per-tool
      heartbeat cannot (``emit_keepalive``).

    carries only a monotonic sequence counter and elapsed time so no
    tool-internal payload leaks. an empty ``tool_name`` denotes the
    turn-level keepalive; a non-empty one denotes per-tool progress.

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
        StreamInterruptEvent,
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
        self._terminal_kind: str | None = None  # "end", "error", or "interrupt" once closed
        # turn-level keepalive sequence (LRT-02), distinct from the per-tool
        # heartbeat sequence the ToolCallProgressHook owns. emit_keepalive
        # advances this so a long time-to-first-token / between-node gap never
        # trips the consumer's stream-gap timer.
        self._keepalive_sequence: int = 0

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

    def replace_content(self, content: str) -> None:
        """replace the accumulated buffer so the terminal carries ``content``.

        a post-hoc node may supersede the answer that already streamed:
        an adversarial reviewer revises a challenged draft, or a blocked
        query routes back to the agent to re-run. the superseding text is
        the authoritative answer, but the draft's tokens already landed in
        :attr:`accumulated_content`. resetting the buffer makes the terminal
        :class:`StreamEndEvent.content` (and any consumer reading it, e.g.
        the non-incremental chat surface) carry the corrected answer rather
        than the stale draft. already-published per-token envelopes cannot
        be unsent -- incremental UIs saw the draft live -- but the final
        content is made correct. pass ``""`` to clear ahead of a re-run.
        a no-op once the stream has closed (the terminal already fired).

        :param content: text to set as the accumulated buffer
        :ptype content: str
        :return: nothing
        :rtype: None
        """
        if self._closed:
            return
        self._accumulated_content = content

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

    async def emit_keepalive(self, *, elapsed_ms: int) -> None:
        """publish a TURN-LEVEL keepalive (LRT-02).

        the per-tool :meth:`emit_tool_call_progress` only keeps the stream
        alive WHILE a tool runs; it cannot cover the time-to-first-token wait
        in the agent node (slow under gateway contention) or the gaps between
        nodes. this method publishes a :class:`ToolCallProgressEvent` with an
        empty ``tool_name`` -- a pure liveness signal -- so a consumer's
        stream-gap timer never trips while the turn is genuinely working. the
        caller drives it on a fixed interval for the whole turn; a dead handler
        stops emitting, so the gap timer still catches a truly-dead stream (no
        masking). the keepalive sequence is owned here, independent of any
        per-tool heartbeat sequence.

        :param elapsed_ms: wall-clock milliseconds since the turn began
        :ptype elapsed_ms: int
        :return: nothing
        :rtype: None
        """
        if self._closed:
            return
        self._keepalive_sequence += 1
        event = ToolCallProgressEvent(
            correlation_id=self._correlation_id,
            tool_name="",
            elapsed_ms=elapsed_ms,
            sequence=self._keepalive_sequence,
        )
        await self._publish(event)

    @traced
    async def end(self, *, metadata: dict[str, Any] | None = None) -> None:
        """publish the success terminal :class:`StreamEndEvent` once.

        idempotent if a previous :meth:`end` already fired -- second
        and later calls no-op. raises
        :class:`StreamingResponseError` if a previous :meth:`error`
        already fired (lifecycle terminals are mutually exclusive).

        :param metadata: optional free-form turn-level metadata to carry
            on the terminal envelope (defaults to empty)
        :ptype metadata: dict[str, Any] | None
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
            conversation_id=self._conversation_id,
            metadata=dict(metadata) if metadata else {},
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

    @traced
    async def interrupt(self, *, payload: Any, metadata: dict[str, Any] | None = None) -> None:
        """publish the PAUSE terminal :class:`StreamInterruptEvent` once.

        the graph reached a human-in-the-loop ``interrupt(...)`` instead of completing; the turn
        ends as a legitimate pause carrying the interrupt payload for the consumer to surface +
        later resume. idempotent if a previous :meth:`interrupt` already fired; raises
        :class:`StreamingResponseError` if a previous :meth:`end` / :meth:`error` already fired
        (lifecycle terminals are mutually exclusive).

        :param payload: the value the node passed to ``interrupt(...)`` (the prompt to resolve)
        :ptype payload: Any
        :param metadata: optional free-form turn-level metadata for the terminal envelope
        :ptype metadata: dict[str, Any] | None
        :return: nothing
        :rtype: None
        :raises StreamingResponseError: when a different terminal already fired
        """
        if self._closed:
            if self._terminal_kind == "interrupt":
                return
            raise StreamingResponseError(
                "cannot call interrupt() after a different terminal fired on this stream",
            )
        self._closed = True
        self._terminal_kind = "interrupt"
        duration_ms = self._compute_duration_ms()
        event = StreamInterruptEvent(
            correlation_id=self._correlation_id,
            payload=payload,
            duration_ms=duration_ms,
            conversation_id=self._conversation_id,
            metadata=dict(metadata) if metadata else {},
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
                    # drop tokens from a node's internal model call (an
                    # adversarial reviewer / summarizer) so its deliberation
                    # never leaks into the user-facing answer; the call opts
                    # out by tagging itself NOSTREAM_TAG. untagged model
                    # calls (the agent's answer) stream as before.
                    if NOSTREAM_TAG in (event.get("tags") or []):
                        continue
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
            try:
                interrupt_payload = await _detect_interrupt(compiled_graph, config)
            except Exception as exc:
                # a checkpointer-backed graph whose post-run state query FAILS must surface a real
                # error terminal, never a silent empty StreamEndEvent.
                log.error(
                    "interrupt detection failed after graph completion",
                    extra={
                        "extra_data": {
                            "correlation_id": str(self._correlation_id),
                            "error_type": type(exc).__name__,
                        }
                    },
                    exc_info=True,
                )
                await self.error(code="AGENT_FAILED", message=str(exc))
                raise
            if interrupt_payload is not _NO_INTERRUPT:
                # the graph PAUSED on a human-in-the-loop interrupt rather than completing: stash the
                # payload for the synchronous caller and end the turn as a pause, never an empty
                # StreamEndEvent (the silent-hang bug).
                final_state["__interrupt__"] = interrupt_payload
                await self.interrupt(payload=interrupt_payload)
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


async def detect_interrupt(compiled_graph: Any, config: dict[str, Any]) -> Any:
    """return the pending interrupt payload when the graph PAUSED, else :data:`NO_INTERRUPT`.

    public, transport-agnostic wrapper over the private :func:`_detect_interrupt` so a NON-streaming
    consumer (the SDK's ``compiled.ainvoke(...)`` path) detects a human-in-the-loop pause the SAME
    way the streaming :meth:`StreamingResponse.run_graph` does -- the snapshot-shape logic lives in
    ONE place (``_detect_interrupt`` / :func:`_snapshot_interrupts`), never duplicated in the SDK.
    a failing checkpointer-backed ``aget_state`` is NOT swallowed -- it propagates so the caller can
    surface a real error rather than a silent empty reply (the bug the interrupt path exists to kill).

    :param compiled_graph: the compiled LangGraph (may expose ``aget_state`` / ``checkpointer``)
    :ptype compiled_graph: Any
    :param config: the LangGraph runtime config (carries the thread id)
    :ptype config: dict[str, Any]
    :return: the interrupt payload, or :data:`NO_INTERRUPT` when the graph completed normally
    :rtype: Any
    :raises Exception: re-raises a checkpointer-backed ``aget_state`` failure (never swallowed)
    """
    return await _detect_interrupt(compiled_graph, config)


def render_interrupt_prompt(payload: Any) -> str:
    """render a minimal, generic human-facing prompt from a raw interrupt payload.

    when a graph PAUSES on a human-in-the-loop ``interrupt(...)`` every text surface that shows the
    pause to a human -- the agent's synchronous reply ``content``, a channel delivery, the hub's
    WebSocket terminal frame -- needs a string. this is the platform's GENERIC floor and lives here,
    in the shared wire module, so EVERY surface renders an interrupt IDENTICALLY (single source of
    truth, no per-consumer drift). it stringifies the payload and appends a one-line "what to do
    next" hint. RICH rendering (a structured approval card, action-specific copy) is the consuming
    agent's job -- it owns the payload shape and can render from the first-class ``interrupt`` field
    on the response metadata / :attr:`StreamInterruptEvent.payload`. ``None`` and a blank/whitespace
    payload both collapse to the hint alone so the surface never shows a literal ``"None"`` or a
    blank bubble.

    :param payload: the value a node passed to ``interrupt(...)``
    :ptype payload: Any
    :return: a human-readable prompt string
    :rtype: str
    """
    rendered = "" if payload is None else str(payload).strip()
    hint = "Reply to approve or deny to continue."
    return f"{rendered}\n\n{hint}" if rendered else hint


async def _detect_interrupt(compiled_graph: Any, config: dict[str, Any]) -> Any:
    """return the pending interrupt payload when the graph PAUSED, else :data:`_NO_INTERRUPT`.

    a graph compiled with a checkpointer surfaces a dynamic ``interrupt(value)`` as a paused
    snapshot whose ``aget_state(config)`` carries pending interrupts; the payload is the value a node
    passed to ``interrupt(...)``. graphs WITHOUT a checkpointer cannot interrupt, so detection is
    skipped (the sentinel). when a checkpointer IS present a failing ``aget_state`` is NOT swallowed
    as "no interrupt" -- it propagates so the caller surfaces an error terminal rather than a silent
    empty end (the exact bug the interrupt terminal exists to kill). a single interrupt returns its
    value directly; concurrent interrupts return the list of values.

    :param compiled_graph: the compiled LangGraph (may expose ``aget_state`` / ``checkpointer``)
    :ptype compiled_graph: Any
    :param config: the LangGraph runtime config (carries the thread id)
    :ptype config: dict[str, Any]
    :return: the interrupt payload, or :data:`_NO_INTERRUPT` when the graph completed normally
    :rtype: Any
    :raises Exception: re-raises a checkpointer-backed ``aget_state`` failure (never swallowed)
    """
    get_state = getattr(compiled_graph, "aget_state", None)
    has_checkpointer = getattr(compiled_graph, "checkpointer", None) is not None
    result: Any = _NO_INTERRUPT
    if get_state is not None and has_checkpointer:
        interrupts = _snapshot_interrupts(await get_state(config))
        if interrupts:
            values = [getattr(item, "value", item) for item in interrupts]
            result = values[0] if len(values) == 1 else values
    return result


def _snapshot_interrupts(snapshot: Any) -> list[Any]:
    """collect pending interrupts from a LangGraph StateSnapshot across version shapes.

    recent LangGraph exposes ``snapshot.interrupts``; older shapes carry them per pending task
    (``snapshot.tasks[i].interrupts``). returns an empty list when the graph is not paused.

    :param snapshot: a LangGraph ``StateSnapshot``
    :ptype snapshot: Any
    :return: the pending ``Interrupt`` objects (empty when the graph is not paused)
    :rtype: list[Any]
    """
    direct = getattr(snapshot, "interrupts", None)
    if direct:
        result = list(direct)
    else:
        result = [
            item for task in getattr(snapshot, "tasks", ()) or () for item in getattr(task, "interrupts", ()) or ()
        ]
    return result
