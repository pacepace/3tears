"""a payload-agnostic byte pipe between a caller and whichever pod owns a key.

a **generic** primitive for "open a byte stream to the pod that currently
*serves* a key, in both directions, and get the producer's bytes out the far
end unchanged". it carries bytes and interprets none of them, in exactly the
way :mod:`threetears.nats.forward` is payload-agnostic. the first consumer
relays an RFB display out of a pod that has no inbound network path; the
second wants an interactive shell into one, and has the identical problem.

design
------

- **rendezvous rides** :func:`threetears.nats.forward` verbatim. the caller
  sends an attach request to the key's forward subject and whichever pod holds
  the key replies with the concrete stream coordinates. no second owner
  election exists here.

- **the owner names its tool in readable form**, and the caller hashes it to
  derive the subject token. the readable name is what keeps a trace
  correlatable once the subject no longer is, and it costs nothing because the
  caller hashes it anyway. an owner naming a tool it does not serve gains
  nothing: its publish grant
  (:meth:`Subjects.pipe_pod_wildcard`) names the digest of its OWN tool, so a
  subject built from a false name is one it cannot publish on.

- **the stream itself rides pod-id-led subjects**, one per direction
  (:meth:`Subjects.pipe`), for the reason
  :meth:`Subjects.gateway_stream` and :meth:`Subjects.hub_stream` already
  state. the nonce is minted by the owner per attach, so a stale reconnect
  cannot land on a live stream and an application key never rides a subject.

- **a gap is a teardown, never a delivery.** a receiver that sees a sequence
  it was not expecting raises and names the stream. this is the whole reason
  core NATS is safe here: a loss is detectable, so it converts a permanently
  corrupted stream into a clean reconnect instead of a frozen screen.

- **credit replaces the backpressure a TCP socket gave away for free.**
  reading from a source and publishing is otherwise unbounded: a slow consumer
  does not slow the producer, it fills the producer's outbound buffer until the
  connection wedges. the receiver acknowledges the bytes it has actually
  CONSUMED, and a sender whose unacknowledged bytes reach the window blocks in
  :meth:`PipeStream.send` -- which stops its caller's read loop at the source,
  which is the only place the backpressure is worth anything.

  credit runs in BOTH directions even though only one of them is fat. the
  symmetric form is less code than an asymmetric one, and the thin direction
  simply never fills its window.

wire framing
------------

every frame is a fixed 5-byte header and a body::

    byte 0      tag
    bytes 1..4  uint32 big-endian sequence
    bytes 5..   body

the tags, and what each one answers for a receiver:

- ``0x00`` data -- body is the producer's bytes, verbatim. the sequence is the
  monotonic per-direction counter, and it is what makes a gap detectable.
- ``0x01`` credit -- the sequence field carries the highest sequence the peer
  has fully consumed, and the body is empty. cumulative rather than
  incremental, so a lost acknowledgement is repaired by the next one instead
  of stranding the window. a credit frame does NOT consume a data sequence: it
  rides the acknowledging side's own outbound subject, which is why an
  acknowledgement needs no subject of its own.
- ``0x02`` close -- graceful half-close. it DOES consume the next data
  sequence, so a final data frame lost in flight makes the close arrive with
  the wrong sequence and tear the stream down rather than presenting a
  truncated stream as a complete one.
- ``0x03`` error -- a stream-level failure the peer should surface rather than
  discover as silence. body is UTF-8 JSON ``{"type": ..., "message": ...}``,
  the same shape :mod:`threetears.nats.forward` frames a handler raise in.
  it carries no data sequence, because a receiver raises on it either way.
- ``0x04`` ready -- the caller's "I have subscribed". core NATS delivers
  nothing to a subject nobody is listening on yet, so an owner that started
  producing on the reply's heels would lose its opening bytes. the owner waits
  for this frame before invoking its handler.

**versioning is negotiated at attach, not carried per frame.** the version
cannot change mid-stream, and a byte on every frame of a continuous
multi-megabyte stream is a real cost for a value that is constant. the caller
names the highest version it speaks, the owner replies with the version it
chose (never higher than the caller's), and either side that cannot speak the
chosen version refuses before a byte moves. adding a tag is therefore a
version bump, and an unknown tag is refused rather than skipped -- which is
what keeps a newer peer's frame from being silently misparsed by an older one.

**the error model has three layers, and they are deliberately distinct.** an
attach that finds no owner raises :class:`threetears.nats.NoOwnerError` from
the forward primitive underneath. a malformed frame, an unknown tag or a
version that cannot be agreed raises :class:`PipeProtocolError`. once a stream
is live, a fault detected locally (a gap) raises
:class:`PipeSequenceGapError` and a fault the PEER reported raises
:class:`PipeRemoteError`, so a consumer reading a traceback can tell which end
failed without correlating two logs. every one of them is terminal for the
stream: there is no resynchronisation, because the payloads this carries have
none either.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import struct
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Final, Literal, Protocol

from threetears.observe import get_logger

from threetears.nats.errors import NatsClientError
from threetears.nats.forward import forward, serve_owner
from threetears.nats.subjects import PipeDirection, Subject, Subjects

if TYPE_CHECKING:
    from threetears.nats.client import NatsClient, Subscription
    from threetears.nats.transport import IncomingMessage, RawMessageCallback

__all__ = [
    "DEFAULT_ATTACH_TIMEOUT",
    "DEFAULT_CREDIT_WINDOW_BYTES",
    "DEFAULT_MAX_CHUNK_BYTES",
    "DEFAULT_READY_TIMEOUT",
    "PIPE_PROTOCOL_VERSION",
    "PipeEndpoint",
    "PipeError",
    "PipeProtocolError",
    "PipeRemoteError",
    "PipeSequenceGapError",
    "PipeStream",
    "PipeStreamHandler",
    "PipeTransport",
    "attach_pipe",
    "open_pipe",
    "serve_pipe",
]


log = get_logger(__name__)


#: the framing this module speaks. bumped when a tag is added or a field
#: changes width; negotiated once per stream in the attach exchange rather
#: than carried on every frame.
PIPE_PROTOCOL_VERSION: Final[int] = 1

#: the oldest framing this module can still speak. equal to
#: :data:`PIPE_PROTOCOL_VERSION` while only one version exists; the two are
#: separate names so a later version can keep talking to a deployed peer
#: without the negotiation having to be invented at that moment.
MIN_PIPE_PROTOCOL_VERSION: Final[int] = 1

#: largest body a single data frame carries. the sender splits anything larger.
#: 64 KiB sits under even an untuned NATS ``max_payload`` of 1 MB with room for
#: the header and the subject, so the pipe works on a broker nobody has tuned;
#: a deployment with a raised limit can negotiate upward at attach.
DEFAULT_MAX_CHUNK_BYTES: Final[int] = 64 * 1024

#: unacknowledged bytes a sender may have outstanding before it blocks.
#:
#: chosen against a measurement rather than taste.
#: ``test_credit_window_sizing_measurements`` in the integration suite reports
#: the two numbers that bound it: the throughput a sender sustains, and the
#: round-trip latency of a credit acknowledgement. their product is the
#: bandwidth-delay product, which is the window BELOW which a sender stalls
#: waiting for credit it has already earned.
#:
#: measured over a testcontainer broker on a developer machine, repeatedly,
#: because both numbers move by a factor of several between runs on the SAME
#: machine. what is durable is therefore the method and the margin, not a
#: range: throughput came out in the tens of MB/s against acknowledgement
#: round trips of a few milliseconds, and the worst PAIRING observed gave a
#: bandwidth-delay product near 190 KB. this value clears that worst case
#: several times over, which is the margin a link slower and jitterier than a
#: local container needs. the same volatility is why the test asserts only
#: that both numbers are sane: pinning either would fail on the next run, let
#: alone the next machine.
#:
#: it is bounded from ABOVE too, and that bound was read rather than assumed:
#: nats-py drops a message and reports a slow consumer once a subscription's
#: pending bytes reach ``DEFAULT_SUB_PENDING_BYTES_LIMIT`` (128 MiB in
#: ``nats/aio/subscription.py``), which :meth:`NatsClient.subscribe` does not
#: override. the window caps what a peer can have in flight, so keeping it two
#: orders of magnitude under that limit means this pipe's own traffic cannot be
#: what trips it.
#:
#: what it has NOT been sized against is real traffic: a synthetic producer
#: emitting fixed chunks is not a framebuffer, so treat this as a defensible
#: starting value rather than a tuned one until a real workload has been
#: measured through it.
DEFAULT_CREDIT_WINDOW_BYTES: Final[int] = 1024 * 1024

#: how long :func:`attach_pipe` waits for an owner to answer. matches
#: :data:`threetears.nats.DEFAULT_FORWARD_TIMEOUT`'s reasoning -- a missing
#: owner should surface fast so the caller can retry.
DEFAULT_ATTACH_TIMEOUT: Final[timedelta] = timedelta(seconds=5)

#: how long an owner waits for the caller's ready frame before abandoning the
#: stream. longer than the attach timeout because the caller has to subscribe
#: and flush after it receives the reply, and a stream abandoned early is a
#: session a human loses.
DEFAULT_READY_TIMEOUT: Final[timedelta] = timedelta(seconds=30)


#: frame tag: body is the producer's bytes, verbatim.
_TAG_DATA: Final[int] = 0x00

#: frame tag: sequence field is the highest sequence the peer has consumed.
_TAG_CREDIT: Final[int] = 0x01

#: frame tag: graceful half-close; consumes the next data sequence.
_TAG_CLOSE: Final[int] = 0x02

#: frame tag: stream-level failure; body is UTF-8 JSON {type, message}.
_TAG_ERROR: Final[int] = 0x03

#: frame tag: the caller has subscribed and the owner may start producing.
_TAG_READY: Final[int] = 0x04

#: bytes of entropy in a stream nonce. UNPREDICTABLE rather than sequential or
#: time-ordered: the nonce is what keeps a stale reconnect from landing on a
#: live stream, and it renders into a subject a wildcard subscriber could
#: otherwise walk, so guessing it must not be cheaper than being granted it.
#: hex-encoded, so the token is ``[0-9a-f]`` only and needs no sanitizing.
_NONCE_BYTES: Final[int] = 16

#: ``!BI`` -- one tag byte then a big-endian uint32 sequence.
_HEADER: Final[struct.Struct] = struct.Struct("!BI")

#: highest value the uint32 sequence field holds. at the default chunk size
#: that is a quarter of a petabyte in one direction of one stream, so it is
#: not a limit any consumer meets; the encoder refuses past it rather than
#: wrapping, because a wrap would present as valid ordering.
_MAX_SEQ: Final[int] = 0xFFFFFFFF


PipeSide = Literal["caller", "owner"]
"""which end of a pipe a :class:`PipeStream` is.

decides only which subject the stream publishes on and which it subscribes:
the ``owner`` publishes ``down`` and subscribes ``up``, the ``caller`` the
reverse. everything else about the two ends is identical.
"""


class PipeError(NatsClientError):
    """base class for byte-pipe conditions.

    subclasses :class:`threetears.nats.NatsClientError` so a consumer catching
    the whole wrapper surface still catches these, while a consumer that cares
    about the stream can catch this narrower base.
    """


class PipeProtocolError(PipeError):
    """raised when a frame or an attach envelope cannot be understood.

    a truncated frame, an unknown tag, or an attach exchange whose version
    could not be agreed. distinct from the two live-stream faults below
    because it means the two ends do not speak the same protocol, which no
    reconnect repairs.
    """


class PipeSequenceGapError(PipeError):
    """raised when a receiver sees a sequence it was not expecting.

    the payloads this pipe carries do not resynchronise, so a missing frame is
    a teardown rather than something to skip. the message names the stream
    (its nonce and the subject it arrived on) because a deployment runs many
    at once and the operator needs to know which session died.
    """


class PipeRemoteError(PipeError):
    """raised when the peer reported a stream-level failure.

    carries the peer's exception type name and message, the same way
    :class:`threetears.nats.ForwardedHandlerError` does, so a consumer can tell
    "the display process died" from "the transport lost a frame" without
    correlating two logs.

    :param type_name: ``type(exc).__name__`` from the peer
    :ptype type_name: str
    :param message: ``str(exc)`` from the peer
    :ptype message: str
    """

    __slots__ = ("type_name", "message")

    def __init__(self, type_name: str, message: str) -> None:
        self.type_name = type_name
        self.message = message
        super().__init__(f"pipe peer reported {type_name}: {message}")


class PipeTransport(Protocol):
    """the NATS surface a :class:`PipeStream` needs, and nothing more.

    :class:`threetears.nats.NatsClient` satisfies it structurally. it exists
    for the reason :mod:`threetears.nats.transport`'s Protocols exist: a test
    can substitute a recording double without emulating request/reply, KV or
    JetStream, and the double's parity is then checked against a surface small
    enough to be worth checking.
    """

    async def publish_raw(self, *, subject: Subject, payload: bytes) -> None:
        """publish raw bytes to a subject.

        :param subject: target subject
        :ptype subject: Subject
        :param payload: frame bytes
        :ptype payload: bytes
        :return: nothing
        :rtype: None
        """
        ...

    async def subscribe(
        self,
        subject: Subject,
        *,
        cb: RawMessageCallback,
        deadletter_on_failure: bool,
    ) -> Subscription:
        """subscribe to a subject with a raw-bytes callback.

        :param subject: subject to subscribe on
        :ptype subject: Subject
        :param cb: async callback receiving the inbound envelope
        :ptype cb: RawMessageCallback
        :param deadletter_on_failure: whether a raising callback republishes to deadletter
        :ptype deadletter_on_failure: bool
        :return: subscription handle
        :rtype: Subscription
        """
        ...

    async def unsubscribe(self, sub: Subscription) -> None:
        """drop a subscription.

        :param sub: handle returned by :meth:`subscribe`
        :ptype sub: Subscription
        :return: nothing
        :rtype: None
        """
        ...

    async def flush(self) -> None:
        """round-trip to the server so prior operations are registered.

        :return: nothing
        :rtype: None
        """
        ...


@dataclass(frozen=True, slots=True)
class PipeEndpoint:
    """the coordinates two ends agreed on at attach.

    everything needed to build both subjects and to run the flow control. the
    tool name is the READABLE registered namespace name, not its digest: the
    subject builders hash it, and the readable form is what makes a log line
    correlatable to a subject nobody can read.

    :param tool: the serving tool's registered namespace name
    :ptype tool: str
    :param pod_id: the owning pod's identifier
    :ptype pod_id: str
    :param nonce: per-attach nonce the owner minted
    :ptype nonce: str
    :param max_chunk: largest body a single data frame carries
    :ptype max_chunk: int
    :param credit: unacknowledged bytes a sender may have outstanding
    :ptype credit: int
    :param version: the framing version both ends agreed to speak
    :ptype version: int
    """

    tool: str
    pod_id: str
    nonce: str
    max_chunk: int
    credit: int
    version: int

    def __post_init__(self) -> None:
        """refuse coordinates that cannot carry a stream.

        a window smaller than one frame is not a slow pipe, it is a stopped
        one: the first frame the sender emits already exceeds what the receiver
        agreed to hold, so the receiver's own overrun guard tears the stream
        down before a byte is consumed. caught here, where both the owner
        minting the coordinates and the caller reading them pass through, so
        the misconfiguration surfaces at attach rather than as a fault on the
        first frame.

        :return: nothing
        :rtype: None
        :raises PipeProtocolError: if either limit is non-positive, or a single
            frame could not fit the window
        """
        if self.max_chunk <= 0 or self.credit <= 0:
            raise PipeProtocolError(
                f"pipe limits must be positive; got max_chunk={self.max_chunk} credit={self.credit}"
            )
        if self.max_chunk > self.credit:
            raise PipeProtocolError(
                f"pipe max_chunk {self.max_chunk} exceeds the credit window {self.credit}, "
                f"so the first frame would overrun what the receiver agreed to hold"
            )
        # ``pod_id`` and ``nonce`` are the two values a PEER supplies that are rendered into
        # subject SEGMENTS rather than hashed, and the caller subscribes what they render. That
        # makes them the one place in this module where a remote string reaches a subject
        # un-digested, so they are checked here rather than trusted.
        #
        # ``_sanitize`` is not enough on its own: it maps ``.`` to ``-`` and touches nothing
        # else, so ``*`` and ``>`` survive it. An owner answering an attach with ``pod_id="*"``
        # and ``nonce="*"`` would otherwise have the caller subscribe a WILDCARD across every
        # pod and every stream of that tool -- which the caller's own grant permits, because a
        # grant cannot tell a literal segment from a wildcard one. That is the same
        # wildcard-injection this module hashes the family to prevent, arriving by the one door
        # hashing does not cover.
        for field_name, value in (("pod_id", self.pod_id), ("nonce", self.nonce)):
            if not value:
                raise PipeProtocolError(f"pipe {field_name} must be non-empty")
            if any(c in value for c in "*> \t\r\n"):
                raise PipeProtocolError(
                    f"pipe {field_name} {value!r} carries a NATS wildcard or whitespace; it is "
                    f"rendered into a subject segment the peer then subscribes, so a wildcard "
                    f"here would widen that subscription beyond this one stream"
                )

    def subject(self, direction: PipeDirection) -> Subject:
        """render one direction's subject for these coordinates.

        :param direction: ``down`` (owner publishes) or ``up`` (caller publishes)
        :ptype direction: PipeDirection
        :return: the stream subject
        :rtype: Subject
        """
        return Subjects.pipe(self.tool, self.pod_id, self.nonce, direction)


PipeStreamHandler = Callable[["PipeStream"], Awaitable[None]]
"""owner-side handler shape for :func:`serve_pipe`.

receives the owner's end of one attached stream and drives it for as long as
the stream should live. returning half-closes the stream gracefully; raising
frames the exception to the caller as a :class:`PipeRemoteError`.
"""


# ----------------------------------------------------------------------
# framing
# ----------------------------------------------------------------------


def _encode_frame(tag: int, seq: int, body: bytes) -> bytes:
    """frame one message: tag, big-endian uint32 sequence, body.

    :param tag: one of the module's frame tags
    :ptype tag: int
    :param seq: data sequence, or the acked-through sequence on a credit frame
    :ptype seq: int
    :param body: frame body (empty for every tag but data and error)
    :ptype body: bytes
    :return: the encoded frame
    :rtype: bytes
    :raises PipeProtocolError: if the sequence does not fit the uint32 field
    """
    if not 0 <= seq <= _MAX_SEQ:
        raise PipeProtocolError(f"pipe sequence {seq} does not fit the uint32 sequence field")
    return _HEADER.pack(tag, seq) + body


def _decode_frame(frame: bytes) -> tuple[int, int, bytes]:
    """split a frame into its tag, sequence and body.

    :param frame: raw bytes as they arrived
    :ptype frame: bytes
    :return: ``(tag, sequence, body)``
    :rtype: tuple[int, int, bytes]
    :raises PipeProtocolError: if the frame is shorter than the header
    """
    if len(frame) < _HEADER.size:
        raise PipeProtocolError(f"pipe frame is {len(frame)} bytes, shorter than its {_HEADER.size}-byte header")
    tag, seq = _HEADER.unpack_from(frame, 0)
    return int(tag), int(seq), frame[_HEADER.size :]


def _encode_error_body(exc: BaseException) -> bytes:
    """render an exception into an error frame's body.

    :param exc: the exception to report to the peer
    :ptype exc: BaseException
    :return: UTF-8 JSON ``{"type": ..., "message": ...}``
    :rtype: bytes
    """
    return json.dumps({"type": type(exc).__name__, "message": str(exc)}).encode("utf-8")


def _decode_error_body(body: bytes) -> PipeRemoteError:
    """reconstruct the peer's reported failure from an error frame's body.

    a body that cannot be parsed still produces a :class:`PipeRemoteError`
    rather than a different exception: the peer said the stream failed, and
    losing that because its explanation was malformed would replace a
    diagnosable teardown with a hang.

    :param body: error frame body
    :ptype body: bytes
    :return: the reconstructed error
    :rtype: PipeRemoteError
    """
    try:
        decoded = json.loads(body.decode("utf-8"))
        return PipeRemoteError(type_name=str(decoded["type"]), message=str(decoded["message"]))
    except (UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
        return PipeRemoteError(type_name="MalformedErrorFrame", message=f"peer sent an unreadable error frame: {exc}")


# ----------------------------------------------------------------------
# attach envelopes
# ----------------------------------------------------------------------


def _encode_attach_request(*, version: int, credit: int, max_chunk: int) -> bytes:
    """render the attach request the caller forwards to the key's owner.

    :param version: highest framing version the caller speaks
    :ptype version: int
    :param credit: window the caller can buffer on its inbound direction
    :ptype credit: int
    :param max_chunk: largest data body the caller wants to receive
    :ptype max_chunk: int
    :return: UTF-8 JSON request body
    :rtype: bytes
    """
    return json.dumps({"op": "attach", "version": version, "credit": credit, "max_chunk": max_chunk}).encode("utf-8")


def _decode_attach_request(payload: bytes) -> tuple[int, int, int]:
    """read an attach request.

    :param payload: the forwarded request body
    :ptype payload: bytes
    :return: ``(version, credit, max_chunk)``
    :rtype: tuple[int, int, int]
    :raises PipeProtocolError: if the body is not a well-formed attach request
    """
    try:
        decoded = json.loads(payload.decode("utf-8"))
        if decoded["op"] != "attach":
            raise PipeProtocolError(f"pipe attach request carries op {decoded['op']!r}")
        return int(decoded["version"]), int(decoded["credit"]), int(decoded["max_chunk"])
    except (UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
        raise PipeProtocolError(f"pipe attach request is malformed: {exc}") from exc


def _encode_attach_reply(endpoint: PipeEndpoint) -> bytes:
    """render the owner's attach reply.

    :param endpoint: the coordinates the owner minted
    :ptype endpoint: PipeEndpoint
    :return: UTF-8 JSON reply body
    :rtype: bytes
    """
    return json.dumps(
        {
            "tool": endpoint.tool,
            "pod_id": endpoint.pod_id,
            "nonce": endpoint.nonce,
            "max_chunk": endpoint.max_chunk,
            "credit": endpoint.credit,
            "version": endpoint.version,
        }
    ).encode("utf-8")


def _decode_attach_reply(payload: bytes) -> PipeEndpoint:
    """read the owner's attach reply and check the version it chose.

    :param payload: the owner's reply body
    :ptype payload: bytes
    :return: the agreed coordinates
    :rtype: PipeEndpoint
    :raises PipeProtocolError: if the reply is malformed or names a version
        this module cannot speak
    """
    try:
        decoded = json.loads(payload.decode("utf-8"))
        endpoint = PipeEndpoint(
            tool=str(decoded["tool"]),
            pod_id=str(decoded["pod_id"]),
            nonce=str(decoded["nonce"]),
            max_chunk=int(decoded["max_chunk"]),
            credit=int(decoded["credit"]),
            version=int(decoded["version"]),
        )
    except (UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
        raise PipeProtocolError(f"pipe attach reply is malformed: {exc}") from exc
    if not MIN_PIPE_PROTOCOL_VERSION <= endpoint.version <= PIPE_PROTOCOL_VERSION:
        raise PipeProtocolError(
            f"owner chose pipe protocol version {endpoint.version}, "
            f"which is outside the {MIN_PIPE_PROTOCOL_VERSION}..{PIPE_PROTOCOL_VERSION} this caller speaks"
        )
    return endpoint


# ----------------------------------------------------------------------
# the stream
# ----------------------------------------------------------------------


class PipeStream:
    """one end of an open byte pipe.

    both ends are the same class; :class:`PipeSide` decides only which subject
    it publishes on and which it subscribes. a consumer sends with
    :meth:`send`, reads with :meth:`receive` until it returns ``None``, and
    lets the context manager tear the subscription down.

    **flow control is what makes this usable on a real link.** :meth:`send`
    blocks once the peer has not acknowledged ``credit`` bytes, so a producer
    loop shaped ``while chunk := await source.read(n): await stream.send(chunk)``
    stops reading its source. that is the whole point: the block has to reach
    the source, not merely the socket.

    **the inbound subscription is deliberately serial.** it is registered
    without ``max_in_flight``, so callbacks run one at a time and in order --
    :meth:`NatsClient.subscribe`'s own docstring records that setting the cap
    trades ordering away, and out-of-order frames would manufacture sequence
    gaps that never happened on the wire.

    :param nats: connected NATS surface
    :ptype nats: PipeTransport
    :param endpoint: the coordinates both ends agreed at attach
    :ptype endpoint: PipeEndpoint
    :param side: which end this is
    :ptype side: PipeSide
    """

    def __init__(self, *, nats: PipeTransport, endpoint: PipeEndpoint, side: PipeSide) -> None:
        self._nats = nats
        self._endpoint = endpoint
        self._side = side
        self._outbound = endpoint.subject("down" if side == "owner" else "up")
        self._inbound = endpoint.subject("up" if side == "owner" else "down")
        self._sub: Subscription | None = None

        # outbound: the monotonic sequence, and the frames the peer has not
        # acknowledged yet (sequence -> body size), oldest first.
        self._send_seq = 0
        # Serialises the OUTBOUND side. Two concurrent senders would otherwise interleave both
        # their sequence assignments and their publishes, and the receiver treats out-of-order
        # as unrecoverable -- so concurrent producers would manufacture a gap teardown that no
        # lost frame caused. Sequencing alone would not be enough either: this carries a BYTE
        # STREAM, so two senders' chunks arriving validly sequenced but interleaved is still a
        # corrupted payload. The lock therefore spans a whole send rather than one frame.
        self._send_lock = asyncio.Lock()
        self._unacked: deque[tuple[int, int]] = deque()
        self._unacked_bytes = 0
        self._credit = asyncio.Event()
        self._credit.set()
        self._sent_close = False

        # inbound: the sequence expected next, the frames received but not yet
        # consumed, and how much has been consumed since the last acknowledgement.
        self._expect_seq = 1
        self._inbox: asyncio.Queue[tuple[int, bytes | None]] = asyncio.Queue()
        self._queued_bytes = 0
        self._consumed_since_ack = 0
        self._consumed_seq = 0
        self._peer_closed = False

        self._peer_ready = asyncio.Event()
        self._fault: Exception | None = None

    # -- lifecycle -----------------------------------------------------

    @property
    def endpoint(self) -> PipeEndpoint:
        """the coordinates this stream runs on.

        :return: the agreed endpoint
        :rtype: PipeEndpoint
        """
        return self._endpoint

    @property
    def unacked_bytes(self) -> int:
        """bytes this end has sent that the peer has not acknowledged consuming.

        the live flow-control gauge: :meth:`send` blocks once it reaches
        ``endpoint.credit``. exposed because a consumer that wants to know
        whether it is the producer or the link that is slow has no other way
        to tell them apart.

        :return: outstanding unacknowledged bytes
        :rtype: int
        """
        return self._unacked_bytes

    async def open(self) -> None:
        """subscribe this end's inbound subject and announce readiness.

        the subscribe is flushed before returning, so the subject is
        registered server-side before anything can be published to it. the
        caller additionally sends its ready frame here, which is what tells
        the owner it is safe to start producing.

        :return: nothing
        :rtype: None
        :raises SubscribeError: if the subscription cannot be registered
        """
        self._sub = await self._nats.subscribe(
            self._inbound,
            cb=self._on_frame,
            deadletter_on_failure=False,
        )
        await self._nats.flush()
        if self._side == "caller":
            await self._emit_control(_TAG_READY, 0, b"")

    async def aclose(self) -> None:
        """drop the inbound subscription.

        does NOT half-close the stream; :meth:`send_close` does that, and a
        consumer that wants the peer to see a clean end of stream calls it
        first.

        :return: nothing
        :rtype: None
        """
        if self._sub is not None:
            await self._nats.unsubscribe(self._sub)
            self._sub = None

    async def __aenter__(self) -> PipeStream:
        """open the stream.

        :return: this stream
        :rtype: PipeStream
        """
        await self.open()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        """half-close if the stream is still healthy, then drop the subscription.

        the half-close is best-effort and its failure is logged rather than
        raised. the likeliest reason a final publish fails is that the
        connection is gone -- which is frequently also the reason the block is
        being left, so raising here would bury the informative exception under
        a transport error from the cleanup.

        :param exc_type: exception type propagating out of the block, if any
        :ptype exc_type: object
        :param exc: exception propagating out of the block, if any
        :ptype exc: object
        :param tb: traceback propagating out of the block, if any
        :ptype tb: object
        :return: nothing
        :rtype: None
        """
        if exc is None and self._fault is None and not self._sent_close:
            try:
                await self.send_close()
            except (PipeError, NatsClientError) as close_exc:
                log.info(
                    "pipe half-close failed while leaving the stream",
                    extra={"extra_data": {**self._log_fields(), "error": str(close_exc)}},
                )
        await self.aclose()

    async def wait_for_peer(self, *, timeout: timedelta = DEFAULT_READY_TIMEOUT) -> None:
        """wait until the peer has subscribed its side of the stream.

        core NATS delivers nothing to a subject with no current subscriber, so
        an owner that starts producing the moment it has replied loses whatever
        it sends before the caller's subscribe registers. the caller announces
        itself with a ready frame; this waits for it.

        :param timeout: how long to wait before giving the stream up
        :ptype timeout: timedelta
        :return: nothing
        :rtype: None
        :raises PipeError: if the peer does not announce itself in time
        :raises PipeSequenceGapError: if the stream faulted while waiting
        :raises PipeRemoteError: if the peer reported a failure while waiting
        """
        try:
            await asyncio.wait_for(self._peer_ready.wait(), timeout.total_seconds())
        except TimeoutError as exc:
            raise PipeError(
                f"pipe peer never announced itself on {self._inbound.path} "
                f"within {timeout.total_seconds():.1f}s (stream {self._endpoint.nonce})"
            ) from exc
        self._raise_if_faulted()

    # -- sending -------------------------------------------------------

    async def send(self, data: bytes) -> None:
        """send bytes to the peer, splitting and flow-controlling as needed.

        blocks while the peer has not acknowledged ``endpoint.credit`` bytes.
        that block is the mechanism: it propagates to whatever loop is feeding
        this call, which stops reading the source.

        **concurrent calls are serialised rather than rejected.** the outbound
        half holds a lock for a whole send, so a second caller waits instead of
        interleaving with the first. two unsynchronised senders would otherwise
        corrupt the stream twice over: their sequence assignments and their
        publishes can interleave independently, and a receiver treats
        out-of-order as unrecoverable -- so they would manufacture a gap
        teardown no lost frame caused, and even correctly-ordered interleaving
        would still split one sender's bytes around another's.

        :param data: bytes to send; split into ``endpoint.max_chunk`` frames
        :ptype data: bytes
        :return: nothing
        :rtype: None
        :raises PipeError: if this end has already half-closed
        :raises PipeSequenceGapError: if the stream faulted
        :raises PipeRemoteError: if the peer reported a failure
        :raises PublishError: if the underlying publish fails
        """
        async with self._send_lock:
            self._raise_if_faulted()
            if self._sent_close:
                raise PipeError(f"pipe stream {self._endpoint.nonce} was half-closed; no further bytes can be sent")
            view = memoryview(data)
            for offset in range(0, len(view), self._endpoint.max_chunk):
                chunk = bytes(view[offset : offset + self._endpoint.max_chunk])
                await self._await_credit(len(chunk))
                await self._emit_sequenced(_TAG_DATA, chunk)

    async def send_close(self) -> None:
        """half-close this end: the peer's :meth:`receive` will return ``None``.

        the close frame consumes the next data sequence, so a final data frame
        lost in flight makes it arrive with the wrong sequence and tear the
        stream down instead of presenting a truncated stream as a complete one.

        :return: nothing
        :rtype: None
        :raises PublishError: if the underlying publish fails
        """
        if self._sent_close:
            return
        self._sent_close = True
        await self._emit_sequenced(_TAG_CLOSE, b"")

    async def send_error(self, exc: BaseException) -> None:
        """report a stream-level failure to the peer.

        the peer's next :meth:`receive` (or a :meth:`send` it is blocked in)
        raises :class:`PipeRemoteError` carrying this exception's type name and
        message, so a producer whose source died reaches the far end as a
        diagnosis rather than as silence.

        :param exc: the failure to report
        :ptype exc: BaseException
        :return: nothing
        :rtype: None
        :raises PublishError: if the underlying publish fails
        """
        await self._emit_control(_TAG_ERROR, 0, _encode_error_body(exc))

    async def _await_credit(self, frame_size: int) -> None:
        """block until *frame_size* more bytes fit inside the peer's window.

        The size matters, and admitting on the OUTSTANDING total alone was an off-by-one that
        only exact-multiple windows hid. A sender that stopped at ``unacked < credit`` could
        still be one byte under the limit and then emit a whole ``max_chunk``, putting
        ``credit - 1 + max_chunk`` bytes in flight against a receiver that faults above
        ``credit``. The peer would then tear the stream down for an overrun the sender believed
        it had avoided, and the fault would name the receiving end.

        It is reachable on the real path rather than in theory: a relay reads whatever a socket
        hands it, so frames are arbitrary partial sizes and a window that is an exact multiple of
        the chunk size is the exception rather than the rule.
        """
        while True:
            self._raise_if_faulted()
            if self._unacked_bytes + frame_size <= self._endpoint.credit:
                return
            self._credit.clear()
            # re-check after clearing: an acknowledgement that landed between
            # the test above and the clear would otherwise have set an event we
            # just unset, and this call would wait for an ack already spent.
            if self._unacked_bytes + frame_size <= self._endpoint.credit:
                return
            await self._credit.wait()

    async def _emit_sequenced(self, tag: int, body: bytes) -> None:
        """publish a frame that consumes a data sequence and owes an ack."""
        self._send_seq += 1
        frame = _encode_frame(tag, self._send_seq, body)
        self._unacked.append((self._send_seq, len(body)))
        self._unacked_bytes += len(body)
        await self._nats.publish_raw(subject=self._outbound, payload=frame)

    async def _emit_control(self, tag: int, seq: int, body: bytes) -> None:
        """publish a control frame, which consumes no data sequence."""
        await self._nats.publish_raw(subject=self._outbound, payload=_encode_frame(tag, seq, body))

    # -- receiving -----------------------------------------------------

    async def receive(self) -> bytes | None:
        """return the peer's next chunk of bytes, or ``None`` once it half-closed.

        acknowledges consumption as it goes: every half window of consumed
        bytes sends the peer a credit frame, which is what lets its
        :meth:`send` proceed.

        :return: the next chunk, or ``None`` at end of stream
        :rtype: bytes | None
        :raises PipeSequenceGapError: if a frame was lost
        :raises PipeRemoteError: if the peer reported a failure
        :raises PipeProtocolError: if the peer sent something unreadable
        """
        self._raise_if_faulted()
        if self._peer_closed:
            return None
        seq, body = await self._inbox.get()
        self._raise_if_faulted()
        self._consumed_seq = seq
        if body is None:
            self._peer_closed = True
            await self._send_credit()
            return None
        self._queued_bytes -= len(body)
        self._consumed_since_ack += len(body)
        # acknowledge every half window rather than every frame: a frame-per-ack
        # doubles the message count on the fat direction for no extra headroom,
        # and a whole-window ack would stall the peer for a full round trip.
        if self._consumed_since_ack * 2 >= self._endpoint.credit:
            await self._send_credit()
        return body

    async def _send_credit(self) -> None:
        """acknowledge everything consumed so far."""
        self._consumed_since_ack = 0
        await self._emit_control(_TAG_CREDIT, self._consumed_seq, b"")

    async def _on_frame(self, msg: IncomingMessage) -> None:
        """classify one inbound frame; runs serially on the subscription."""
        try:
            tag, seq, body = _decode_frame(msg.data)
        except PipeProtocolError as exc:
            self._fail(exc)
            return

        if tag == _TAG_CREDIT:
            self._apply_credit(seq)
        elif tag == _TAG_READY:
            self._peer_ready.set()
        elif tag == _TAG_ERROR:
            self._fail(_decode_error_body(body))
        elif tag in (_TAG_DATA, _TAG_CLOSE):
            self._accept_sequenced(tag, seq, body, msg.subject)
        else:
            # refused rather than skipped: an unknown tag means the peer speaks
            # a framing this one does not, and guessing at it is how a newer
            # peer's frame gets silently misparsed.
            self._fail(
                PipeProtocolError(
                    f"pipe stream {self._endpoint.nonce} on {msg.subject} received unknown frame tag 0x{tag:02x}"
                )
            )

    def _accept_sequenced(self, tag: int, seq: int, body: bytes, subject: str) -> None:
        """queue a data or close frame, tearing the stream down on a gap."""
        if seq != self._expect_seq:
            self._fail(
                PipeSequenceGapError(
                    f"pipe stream {self._endpoint.nonce} on {subject} lost a frame: "
                    f"expected sequence {self._expect_seq}, received {seq}"
                )
            )
            return
        self._expect_seq += 1
        if tag == _TAG_CLOSE:
            self._inbox.put_nowait((seq, None))
            return
        self._queued_bytes += len(body)
        # the queue itself is unbounded, but what may sit in it is not: the
        # peer's own credit accounting caps its in-flight bytes at the window,
        # so more than that queued means the peer is not honouring credit. that
        # is a fault to surface, not a buffer to grow.
        if self._queued_bytes > self._endpoint.credit:
            self._fail(
                PipeError(
                    f"pipe stream {self._endpoint.nonce} on {subject} has {self._queued_bytes} bytes "
                    f"unconsumed, past the {self._endpoint.credit}-byte window the peer agreed to"
                )
            )
            return
        self._inbox.put_nowait((seq, body))

    def _apply_credit(self, acked_through: int) -> None:
        """release everything the peer has now consumed, and unblock a sender."""
        while self._unacked and self._unacked[0][0] <= acked_through:
            _, size = self._unacked.popleft()
            self._unacked_bytes -= size
        if self._unacked_bytes < self._endpoint.credit:
            self._credit.set()

    # -- faults --------------------------------------------------------

    def _fail(self, exc: Exception) -> None:
        """record the first fault and wake anything waiting on the stream."""
        if self._fault is not None:
            return
        self._fault = exc
        log.warning(
            "pipe stream faulted",
            extra={"extra_data": {**self._log_fields(), "error_type": type(exc).__name__, "error": str(exc)}},
        )
        # wake both halves: a consumer parked in receive() and a producer parked
        # on credit both have to learn that the stream is over.
        self._inbox.put_nowait((0, None))
        self._credit.set()

    def _raise_if_faulted(self) -> None:
        """re-raise the recorded fault, if any."""
        if self._fault is not None:
            raise self._fault

    def _log_fields(self) -> dict[str, str]:
        """the stream's identity, for a log line that has to name which one died."""
        return {
            "tool": self._endpoint.tool,
            "pod_id": self._endpoint.pod_id,
            "nonce": self._endpoint.nonce,
            "side": self._side,
            "inbound": self._inbound.path,
        }


# ----------------------------------------------------------------------
# caller side
# ----------------------------------------------------------------------


async def attach_pipe(
    nats: NatsClient,
    key: str,
    *,
    family: str | None = None,
    credit: int = DEFAULT_CREDIT_WINDOW_BYTES,
    max_chunk: int = DEFAULT_MAX_CHUNK_BYTES,
    timeout: timedelta = DEFAULT_ATTACH_TIMEOUT,
) -> PipeEndpoint:
    """ask whichever pod owns ``key`` for a stream, and return its coordinates.

    rides :func:`threetears.nats.forward` verbatim, so "no owner right now" and
    "the owner's handler raised" surface as that module's own exceptions rather
    than as pipe-specific ones. ``credit`` and ``max_chunk`` are what this
    caller can accommodate; the owner may only lower them.

    :param nats: connected :class:`threetears.nats.NatsClient`
    :ptype nats: NatsClient
    :param key: ownership key the target pod serves
    :ptype key: str
    :param family: forward family, matching the one the owner serves under
    :ptype family: str | None
    :param credit: largest window this caller can buffer on its inbound half
    :ptype credit: int
    :param max_chunk: largest data body this caller wants to receive
    :ptype max_chunk: int
    :param timeout: how long to wait for the owner's reply
    :ptype timeout: timedelta
    :return: the coordinates to open the stream on
    :rtype: PipeEndpoint
    :raises NoOwnerError: if no pod currently serves the key
    :raises ForwardedHandlerError: if the owner refused the attach
    :raises PipeProtocolError: if the reply is malformed or names an
        unspeakable version
    """
    request = _encode_attach_request(version=PIPE_PROTOCOL_VERSION, credit=credit, max_chunk=max_chunk)
    reply = await forward(nats, key, request, timeout=timeout, family=family)
    return _decode_attach_reply(reply)


@asynccontextmanager
async def open_pipe(nats: PipeTransport, endpoint: PipeEndpoint) -> AsyncIterator[PipeStream]:
    """open the caller's end of an attached stream.

    subscribes the ``down`` direction, announces readiness on ``up``, and
    half-closes on a clean exit.

    :param nats: connected NATS surface
    :ptype nats: PipeTransport
    :param endpoint: coordinates returned by :func:`attach_pipe`
    :ptype endpoint: PipeEndpoint
    :yield: the caller's end of the stream
    :rtype: AsyncIterator[PipeStream]
    """
    async with PipeStream(nats=nats, endpoint=endpoint, side="caller") as stream:
        yield stream


# ----------------------------------------------------------------------
# owner side
# ----------------------------------------------------------------------


@asynccontextmanager
async def serve_pipe(
    nats: NatsClient,
    key: str,
    *,
    tool: str,
    pod_id: str,
    handler: PipeStreamHandler,
    family: str | None = None,
    credit: int = DEFAULT_CREDIT_WINDOW_BYTES,
    max_chunk: int = DEFAULT_MAX_CHUNK_BYTES,
    ready_timeout: timedelta = DEFAULT_READY_TIMEOUT,
) -> AsyncIterator[None]:
    """serve byte-pipe attaches for ``key`` while this context is active.

    rides :func:`threetears.nats.serve_owner`, so the attach is answered by
    whichever pod holds the key and the queue group covers a lease handoff the
    same way. each attach mints its own nonce, opens the owner's end BEFORE
    replying -- the reply is what tells the caller where to subscribe, and an
    owner that replied first could be published to on a subject it had not yet
    subscribed -- and runs ``handler`` on the stream once the caller has
    announced itself.

    a handler that returns half-closes its stream; a handler that raises frames
    the exception to the caller, which sees it as :class:`PipeRemoteError`.

    intended call site: a pod that holds the lease for ``key`` enters this
    context and leaves it the moment it loses ownership.

    :param nats: connected :class:`threetears.nats.NatsClient`
    :ptype nats: NatsClient
    :param key: ownership key this pod serves
    :ptype key: str
    :param tool: this pod's registered tool namespace name, in readable form
    :ptype tool: str
    :param pod_id: this pod's identifier; it leads the stream subject's
        variable segments and must be the one this pod's grant names
    :ptype pod_id: str
    :param handler: async callback driving one attached stream
    :ptype handler: PipeStreamHandler
    :param family: forward family the attach rides, matching the caller's
    :ptype family: str | None
    :param credit: largest window this owner will agree to
    :ptype credit: int
    :param max_chunk: largest data body this owner will agree to
    :ptype max_chunk: int
    :param ready_timeout: how long a stream waits for its caller to subscribe
    :ptype ready_timeout: timedelta
    :yield: nothing, while serving
    :rtype: AsyncIterator[None]
    :raises SubscribeError: if the attach subscription cannot be registered
    """
    running: set[asyncio.Task[None]] = set()

    async def _run_stream(stream: PipeStream) -> None:
        """drive one already-open stream, framing any failure back to the caller."""
        try:
            await stream.wait_for_peer(timeout=ready_timeout)
            await handler(stream)
            await stream.send_close()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- boundary: every handler failure is framed to the caller, never swallowed
            log.info(
                "pipe handler raised; framing the error to the caller",
                extra={"extra_data": {"tool": tool, "pod_id": pod_id, "error_type": type(exc).__name__}},
            )
            try:
                await stream.send_error(exc)
            except NatsClientError as report_exc:
                # a failed report is logged rather than raised: the likeliest
                # reason it fails is the connection this stream rode being gone,
                # which is frequently also why the handler raised.
                log.info(
                    "pipe error report failed",
                    extra={"extra_data": {"tool": tool, "pod_id": pod_id, "error": str(report_exc)}},
                )
        finally:
            await stream.aclose()

    async def _on_attach(payload: bytes) -> bytes:
        """mint a stream for one attach and reply with its coordinates."""
        requested_version, requested_credit, requested_chunk = _decode_attach_request(payload)
        version = min(requested_version, PIPE_PROTOCOL_VERSION)
        if version < MIN_PIPE_PROTOCOL_VERSION:
            raise PipeProtocolError(
                f"caller speaks pipe protocol version {requested_version}; "
                f"this owner speaks {MIN_PIPE_PROTOCOL_VERSION}..{PIPE_PROTOCOL_VERSION}"
            )
        agreed_credit = min(requested_credit, credit)
        endpoint = PipeEndpoint(
            tool=tool,
            pod_id=pod_id,
            nonce=secrets.token_hex(_NONCE_BYTES),
            # the window bounds the frame as well as the two limits either side
            # asked for: a caller that wants a small window and left max_chunk
            # at its default has asked for something coherent, and the owner is
            # the party minting the coordinates, so it reconciles them here
            # rather than refusing the attach.
            max_chunk=min(requested_chunk, max_chunk, agreed_credit),
            credit=agreed_credit,
            version=version,
        )
        stream = PipeStream(nats=nats, endpoint=endpoint, side="owner")
        # subscribe before replying: the reply is what sends the caller to these
        # subjects, and core NATS drops anything published to a subject with no
        # current subscriber.
        await stream.open()
        task = asyncio.create_task(_run_stream(stream), name=f"pipe-stream:{endpoint.nonce}")
        running.add(task)
        task.add_done_callback(running.discard)
        return _encode_attach_reply(endpoint)

    async with serve_owner(nats, key, _on_attach, family=family):
        try:
            yield
        finally:
            for task in list(running):
                task.cancel()
            if running:
                await asyncio.gather(*running, return_exceptions=True)
