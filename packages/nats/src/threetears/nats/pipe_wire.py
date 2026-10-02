"""the byte pipe's wire protocol: what two pods put on the subjects between them.

:mod:`threetears.nats.pipe` runs streams; this module is the part of it that crosses the wire --
the frame layout both ends read and write, the attach envelopes the caller and the owner exchange
over :func:`threetears.nats.forward`, the coordinates that exchange agrees, and the protocol
error either end raises when the other's bytes cannot be understood. It is its own module because
it is a contract between two processes, possibly on two releases: everything here must be read
the same way by both ends, and a change to it is a protocol version bump.

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
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Final

from threetears.nats.errors import NatsClientError
from threetears.nats.subjects import PipeDirection, Subject, Subjects

__all__ = [
    "MIN_PIPE_PROTOCOL_VERSION",
    "PIPE_PROTOCOL_VERSION",
    "TAG_CLOSE",
    "TAG_CREDIT",
    "TAG_DATA",
    "TAG_ERROR",
    "TAG_READY",
    "PipeEndpoint",
    "PipeError",
    "PipeProtocolError",
    "decode_attach_reply",
    "decode_attach_request",
    "decode_frame",
    "encode_attach_reply",
    "encode_attach_request",
    "encode_frame",
]


#: the framing this module speaks. bumped when a tag is added or a field
#: changes width; negotiated once per stream in the attach exchange rather
#: than carried on every frame.
PIPE_PROTOCOL_VERSION: Final[int] = 1

#: the oldest framing this module can still speak. equal to
#: :data:`PIPE_PROTOCOL_VERSION` while only one version exists; the two are
#: separate names so a later version can keep talking to a deployed peer
#: without the negotiation having to be invented at that moment.
MIN_PIPE_PROTOCOL_VERSION: Final[int] = 1

#: frame tag: body is the producer's bytes, verbatim.
TAG_DATA: Final[int] = 0x00

#: frame tag: sequence field is the highest sequence the peer has consumed.
TAG_CREDIT: Final[int] = 0x01

#: frame tag: graceful half-close; consumes the next data sequence.
TAG_CLOSE: Final[int] = 0x02

#: frame tag: stream-level failure; body is UTF-8 JSON {type, message}.
TAG_ERROR: Final[int] = 0x03

#: frame tag: the caller has subscribed and the owner may start producing.
TAG_READY: Final[int] = 0x04

#: ``!BI`` -- one tag byte then a big-endian uint32 sequence.
_HEADER: Final[struct.Struct] = struct.Struct("!BI")

#: highest value the uint32 sequence field holds. at the default chunk size
#: that is a quarter of a petabyte in one direction of one stream, so it is
#: not a limit any consumer meets; the encoder refuses past it rather than
#: wrapping, because a wrap would present as valid ordering.
_MAX_SEQ: Final[int] = 0xFFFFFFFF


class PipeError(NatsClientError):
    """base class for byte-pipe conditions.

    subclasses :class:`threetears.nats.NatsClientError` so a consumer catching
    the whole wrapper surface still catches these, while a consumer that cares
    about the stream can catch this narrower base.
    """


class PipeProtocolError(PipeError):
    """raised when a frame or an attach envelope cannot be understood.

    a truncated frame, an unknown tag, or an attach exchange whose version
    could not be agreed. distinct from the live-stream faults in
    :mod:`threetears.nats.pipe` because it means the two ends do not speak the
    same protocol, which no reconnect repairs.
    """


@dataclass(frozen=True, slots=True)
class PipeEndpoint:
    """the coordinates two ends agreed on at attach.

    everything needed to build both subjects and to run the flow control. the
    tool value is the READABLE tool-name NODE the serving pod OWNS, not its digest
    and not a tool leaf: the subject builders hash it, and the readable form is
    what makes a log line correlatable to a subject nobody can read.

    :param tool: the tool-name NODE the serving pod owns -- ``pentest``, or the
        canonical ``tools.pentest``, which :meth:`Subjects.pipe` roots to one value.
        NOT a registered tool namespace name: the pod's grant on these subjects is
        minted at connect from the NODES on its ``tool_pods`` row, so a leaf here
        names a stream no grant covers. the field keeps its short name because it
        is carried on the attach envelope's wire.
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
        # makes them the one place in the pipe where a remote string reaches a subject
        # un-digested, so they are checked here rather than trusted.
        #
        # ``sanitize_subject_segment`` is not enough on its own: it maps ``.`` to ``-`` and touches nothing
        # else, so ``*`` and ``>`` survive it. An owner answering an attach with ``pod_id="*"``
        # and ``nonce="*"`` would otherwise have the caller subscribe a WILDCARD across every
        # pod and every stream of that tool -- which the caller's own grant permits, because a
        # grant cannot tell a literal segment from a wildcard one. That is the same
        # wildcard-injection the pipe hashes the family to prevent, arriving by the one door
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


def encode_frame(tag: int, seq: int, body: bytes) -> bytes:
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


def decode_frame(frame: bytes) -> tuple[int, int, bytes]:
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


def encode_attach_request(*, version: int, credit: int, max_chunk: int) -> bytes:
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


def decode_attach_request(payload: bytes) -> tuple[int, int, int]:
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


def encode_attach_reply(endpoint: PipeEndpoint) -> bytes:
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


def decode_attach_reply(payload: bytes) -> PipeEndpoint:
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
