"""unit tests for :mod:`threetears.nats.pipe`.

cover the wire framing round-trip and its refusals (short frame, unknown tag,
a sequence that does not fit the field), the attach envelopes and their version
negotiation, and the two live-stream properties the design rests on: a gap
raises and names the stream, and the credit window stops a producer reading its
source and lets it resume on the next acknowledgement.

the two ends are wired to each other through an in-memory subject bus rather
than a broker, so a whole stream runs here with no docker. delivery on that bus
is DIRECT -- ``publish_raw`` awaits the peer's callback -- which makes every
test deterministic without sleeps, and is strictly stronger on ordering than
the real dispatch loop. the proof that the real loop behaves the same way is
``tests/integration/test_pipe_round_trip.py``, against an actual broker.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from datetime import timedelta

import pytest

from threetears.nats import Subjects, set_default_namespace
from threetears.nats.pipe import (
    MIN_PIPE_PROTOCOL_VERSION,
    PIPE_PROTOCOL_VERSION,
    PipeEndpoint,
    PipeError,
    PipeProtocolError,
    PipeRemoteError,
    PipeSequenceGapError,
    PipeStream,
    _decode_attach_reply,
    _decode_attach_request,
    _decode_frame,
    _encode_attach_reply,
    _encode_attach_request,
    _encode_frame,
    _TAG_CLOSE,
    _TAG_CREDIT,
    _TAG_DATA,
    _TAG_ERROR,
    _TAG_READY,
)
from threetears.nats.transport import IncomingMessage


@pytest.fixture(autouse=True)
def _reset_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """each test starts from the documented default namespace."""
    monkeypatch.delenv("THREETEARS_NATS_SUBJECT_NAMESPACE", raising=False)
    set_default_namespace("3tears")


# --------------------------------------------------------------------------
# in-memory transport
# --------------------------------------------------------------------------


class _Bus:
    """an in-memory NATS subject space shared by both ends of a pipe.

    holds at most one subscriber per subject, which is all a pipe ever needs
    (each direction has exactly one publisher and one subscriber). a frame
    published to a subject nobody has subscribed is discarded, matching core
    NATS -- that is what the ready frame exists to avoid.

    ``suppress`` is the fault injector: a callback returning ``True`` for a
    frame drops it on the floor, standing in for the slow-consumer loss the
    sequence check exists to detect.
    """

    def __init__(self) -> None:
        self.subscribers: dict[str, Callable[[IncomingMessage], Awaitable[None]]] = {}
        self.suppress: Callable[[str, bytes], bool] | None = None
        self.published: list[tuple[str, bytes]] = []

    async def deliver(self, subject_path: str, payload: bytes) -> None:
        """hand one frame to the subject's subscriber, if there is one."""
        self.published.append((subject_path, payload))
        if self.suppress is not None and self.suppress(subject_path, payload):
            return
        callback = self.subscribers.get(subject_path)
        if callback is not None:
            await callback(IncomingMessage(data=payload, reply_subject=None, subject=subject_path))


# parity-with: threetears.nats.client.Subscription
class _FakeSubscription:
    """the handle :class:`_FakePipeTransport.subscribe` hands back."""

    def __init__(self, bus: _Bus, subject_path: str) -> None:
        self._bus = bus
        self._subject_path = subject_path
        self._closed = False

    def mark_closed(self) -> None:
        """record that this subscription is done."""
        self._closed = True

    async def unsubscribe(self) -> None:
        """drop the subject's subscriber from the bus."""
        self.mark_closed()
        self._bus.subscribers.pop(self._subject_path, None)


# parity-with: threetears.nats.pipe.PipeTransport
class _FakePipeTransport:
    """one end's view of the shared bus, standing in for a connected client."""

    def __init__(self, bus: _Bus) -> None:
        self._bus = bus
        self.flushes = 0

    async def publish_raw(self, *, subject: object, payload: bytes) -> None:
        """publish a frame onto the bus."""
        await self._bus.deliver(str(subject), payload)

    async def subscribe(
        self,
        subject: object,
        *,
        cb: Callable[[IncomingMessage], Awaitable[None]],
        deadletter_on_failure: bool = True,
    ) -> _FakeSubscription:
        """register this end's inbound callback on the bus."""
        assert deadletter_on_failure is False, "a byte frame is not deadletterable"
        self._bus.subscribers[str(subject)] = cb
        return _FakeSubscription(self._bus, str(subject))

    async def unsubscribe(self, sub: _FakeSubscription) -> None:
        """drop a subscription."""
        await sub.unsubscribe()

    async def flush(self) -> None:
        """record the round trip the real client makes here."""
        self.flushes += 1


def _endpoint(*, max_chunk: int = 64, credit: int = 256) -> PipeEndpoint:
    """coordinates for a stream small enough to fill deliberately."""
    return PipeEndpoint(
        tool="tools.scrape-zone_alpha.1-0-0",
        pod_id="pod-abc",
        nonce="nonce-1",
        max_chunk=max_chunk,
        credit=credit,
        version=PIPE_PROTOCOL_VERSION,
    )


async def _wired_pair(bus: _Bus, endpoint: PipeEndpoint) -> tuple[PipeStream, PipeStream]:
    """open an owner and a caller end on ``bus``, owner first so ready lands."""
    owner = PipeStream(nats=_FakePipeTransport(bus), endpoint=endpoint, side="owner")
    caller = PipeStream(nats=_FakePipeTransport(bus), endpoint=endpoint, side="caller")
    await owner.open()
    await caller.open()
    return owner, caller


async def _settle() -> None:
    """let every ready task run to its next await point."""
    for _ in range(8):
        await asyncio.sleep(0)


# --------------------------------------------------------------------------
# framing
# --------------------------------------------------------------------------


def test_frame_round_trips_tag_sequence_and_body() -> None:
    """a framed message decodes back to the tag, sequence and body it carried."""
    frame = _encode_frame(_TAG_DATA, 7, b"payload-bytes")
    assert _decode_frame(frame) == (_TAG_DATA, 7, b"payload-bytes")


def test_frame_round_trips_an_empty_body() -> None:
    """a control frame carries no body and still decodes unambiguously."""
    assert _decode_frame(_encode_frame(_TAG_CLOSE, 42, b"")) == (_TAG_CLOSE, 42, b"")


def test_frame_body_is_verbatim_for_bytes_that_look_like_a_header() -> None:
    """a body whose leading bytes resemble a header is not re-interpreted."""
    body = _encode_frame(_TAG_CREDIT, 9, b"inner")
    tag, seq, decoded = _decode_frame(_encode_frame(_TAG_DATA, 1, body))
    assert (tag, seq, decoded) == (_TAG_DATA, 1, body)


def test_frame_round_trips_the_widest_sequence_the_field_holds() -> None:
    """the top of the uint32 space encodes and decodes without truncation."""
    assert _decode_frame(_encode_frame(_TAG_DATA, 0xFFFFFFFF, b"x"))[1] == 0xFFFFFFFF


def test_frame_refuses_a_sequence_past_the_field() -> None:
    """the sequence space is refused rather than wrapped: a wrap reads as valid ordering."""
    with pytest.raises(PipeProtocolError, match="does not fit"):
        _encode_frame(_TAG_DATA, 0x1_0000_0000, b"x")


def test_frame_refuses_a_payload_shorter_than_its_header() -> None:
    """a truncated frame is refused rather than read as a zero sequence."""
    with pytest.raises(PipeProtocolError, match="shorter than"):
        _decode_frame(b"\x00\x00")


# --------------------------------------------------------------------------
# attach envelopes
# --------------------------------------------------------------------------


def test_attach_request_round_trips() -> None:
    """the caller's request decodes to the version and limits it asked for."""
    payload = _encode_attach_request(version=PIPE_PROTOCOL_VERSION, credit=4096, max_chunk=512)
    assert _decode_attach_request(payload) == (PIPE_PROTOCOL_VERSION, 4096, 512)


def test_attach_request_refuses_a_foreign_op() -> None:
    """an envelope for some other operation is refused, not part-read."""
    with pytest.raises(PipeProtocolError, match="op"):
        _decode_attach_request(b'{"op": "detach", "version": 1, "credit": 1, "max_chunk": 1}')


def test_attach_reply_round_trips_the_readable_tool_name() -> None:
    """the reply carries the tool in readable form, which the caller hashes itself."""
    endpoint = _endpoint()
    decoded = _decode_attach_reply(_encode_attach_reply(endpoint))
    assert decoded == endpoint
    assert decoded.tool == "tools.scrape-zone_alpha.1-0-0"


def test_attach_reply_refuses_a_version_this_caller_cannot_speak() -> None:
    """a version outside the supported range is refused before a byte moves."""
    ahead = _encode_attach_reply(
        PipeEndpoint(
            tool="t",
            pod_id="p",
            nonce="n",
            max_chunk=64,
            credit=256,
            version=PIPE_PROTOCOL_VERSION + 1,
        )
    )
    with pytest.raises(PipeProtocolError, match="outside"):
        _decode_attach_reply(ahead)
    assert MIN_PIPE_PROTOCOL_VERSION <= PIPE_PROTOCOL_VERSION


def test_attach_reply_refuses_a_malformed_envelope() -> None:
    """a reply missing a field is refused rather than defaulted."""
    with pytest.raises(PipeProtocolError, match="malformed"):
        _decode_attach_reply(b'{"tool": "t"}')


# --------------------------------------------------------------------------
# a stream end to end, over the in-memory bus
# --------------------------------------------------------------------------


async def test_stream_carries_bytes_and_half_closes() -> None:
    """the receiving end reconstructs what the sender sent, then sees end of stream."""
    bus = _Bus()
    owner, caller = await _wired_pair(bus, _endpoint())

    await owner.send(b"a" * 150)
    await owner.send_close()

    received = b""
    while (chunk := await caller.receive()) is not None:
        received += chunk
    assert received == b"a" * 150
    # a second read past the close stays at end of stream rather than blocking.
    assert await caller.receive() is None


async def test_stream_splits_at_the_negotiated_chunk_size() -> None:
    """a send larger than max_chunk becomes several frames, each within the limit."""
    bus = _Bus()
    endpoint = _endpoint(max_chunk=64, credit=4096)
    owner, caller = await _wired_pair(bus, endpoint)

    await owner.send(b"z" * 200)
    down = endpoint.subject("down").path
    bodies = [_decode_frame(payload)[2] for subject, payload in bus.published if subject == down]
    assert [len(b) for b in bodies] == [64, 64, 64, 8]
    assert await caller.receive() == b"z" * 64


async def test_owner_waits_for_the_callers_ready_frame() -> None:
    """an owner does not start producing until the caller has subscribed."""
    bus = _Bus()
    endpoint = _endpoint()
    owner = PipeStream(nats=_FakePipeTransport(bus), endpoint=endpoint, side="owner")
    await owner.open()

    waiting = asyncio.create_task(owner.wait_for_peer())
    await _settle()
    # nothing has announced itself yet, so the owner is still parked.
    assert not waiting.done()

    caller = PipeStream(nats=_FakePipeTransport(bus), endpoint=endpoint, side="caller")
    await caller.open()
    await _settle()
    assert waiting.done()
    await waiting


async def test_owner_gives_up_when_no_caller_ever_subscribes() -> None:
    """a caller that never arrives ends the stream instead of holding it forever."""
    bus = _Bus()
    owner = PipeStream(nats=_FakePipeTransport(bus), endpoint=_endpoint(), side="owner")
    await owner.open()
    with pytest.raises(PipeError, match="never announced itself"):
        await owner.wait_for_peer(timeout=timedelta(milliseconds=10))


# --------------------------------------------------------------------------
# a gap is a teardown
# --------------------------------------------------------------------------


async def test_a_lost_frame_raises_and_names_the_stream() -> None:
    """a dropped data frame raises at the receiver, naming the stream and the gap."""
    bus = _Bus()
    endpoint = _endpoint(max_chunk=8, credit=4096)
    owner, caller = await _wired_pair(bus, endpoint)

    down = endpoint.subject("down").path
    dropped: list[int] = []

    def _drop_second_data_frame(subject: str, payload: bytes) -> bool:
        tag, seq, _ = _decode_frame(payload)
        if subject == down and tag == _TAG_DATA and seq == 2:
            dropped.append(seq)
            return True
        return False

    bus.suppress = _drop_second_data_frame
    await owner.send(b"0" * 24)

    # the drop actually happened -- otherwise a passing raise would prove nothing.
    assert dropped == [2]
    with pytest.raises(PipeSequenceGapError) as excinfo:
        await caller.receive()
    message = str(excinfo.value)
    assert endpoint.nonce in message
    assert down in message
    assert "expected sequence 2" in message
    assert "received 3" in message


async def test_a_lost_final_frame_is_caught_by_the_close() -> None:
    """the close frame consumes a sequence, so a lost last frame is not a clean end."""
    bus = _Bus()
    endpoint = _endpoint(max_chunk=8, credit=4096)
    owner, caller = await _wired_pair(bus, endpoint)

    down = endpoint.subject("down").path
    bus.suppress = lambda subject, payload: subject == down and _decode_frame(payload)[0:2] == (_TAG_DATA, 2)
    await owner.send(b"0" * 16)
    await owner.send_close()

    with pytest.raises(PipeSequenceGapError, match="expected sequence 2"):
        await caller.receive()


async def test_an_unknown_tag_is_refused_rather_than_skipped() -> None:
    """a frame this version does not understand tears the stream down."""
    bus = _Bus()
    endpoint = _endpoint()
    _owner, caller = await _wired_pair(bus, endpoint)

    await bus.deliver(endpoint.subject("down").path, _encode_frame(0x7F, 1, b""))
    with pytest.raises(PipeProtocolError, match="unknown frame tag 0x7f"):
        await caller.receive()


async def test_a_peer_error_frame_surfaces_as_a_remote_error() -> None:
    """a stream-level failure reaches the far end as a diagnosis, not as silence."""
    bus = _Bus()
    owner, caller = await _wired_pair(bus, _endpoint())

    await owner.send_error(RuntimeError("x11vnc exited"))
    with pytest.raises(PipeRemoteError) as excinfo:
        await caller.receive()
    assert excinfo.value.type_name == "RuntimeError"
    assert excinfo.value.message == "x11vnc exited"


async def test_a_faulted_stream_refuses_further_sends() -> None:
    """once a stream has faulted, sending on it raises rather than publishing."""
    bus = _Bus()
    endpoint = _endpoint()
    owner, _caller = await _wired_pair(bus, endpoint)

    await bus.deliver(endpoint.subject("up").path, _encode_frame(0x7F, 1, b""))
    with pytest.raises(PipeProtocolError):
        await owner.send(b"more")


# --------------------------------------------------------------------------
# credit
# --------------------------------------------------------------------------


class _CountingSource:
    """a byte source that records how many times it was read.

    the instrument the backpressure assertion is made on: the credit window is
    only worth anything if the block reaches the thing producing the bytes, so
    the test watches reads of THIS rather than any counter on the client.
    """

    def __init__(self, chunk: bytes, chunks: int) -> None:
        self._chunk = chunk
        self._remaining = chunks
        self.reads = 0

    async def read(self) -> bytes:
        """return the next chunk, or empty bytes once exhausted."""
        self.reads += 1
        if self._remaining == 0:
            return b""
        self._remaining -= 1
        return self._chunk


async def test_credit_window_stops_and_resumes_the_producers_reads() -> None:
    """the producer stops reading its source at the window and resumes on an ack."""
    bus = _Bus()
    # window = 256 bytes over 64-byte chunks: four frames fill it exactly, and the
    # receiver acknowledges every half window, so two consumed frames release it.
    endpoint = _endpoint(max_chunk=64, credit=256)
    owner, caller = await _wired_pair(bus, endpoint)
    source = _CountingSource(b"q" * 64, chunks=64)

    async def _produce() -> None:
        while chunk := await source.read():
            await owner.send(chunk)

    producer = asyncio.create_task(_produce())
    await _settle()

    # four frames fit the window; the fifth read happened and its send is parked.
    assert owner.unacked_bytes == 256
    assert source.reads == 5
    stalled_at = source.reads
    await _settle()
    assert source.reads == stalled_at, "the producer read on past a full window"

    # one consumed frame is under the half window, so no acknowledgement is due
    # and the producer must still be stopped.
    assert await caller.receive() == b"q" * 64
    await _settle()
    assert source.reads == stalled_at
    assert owner.unacked_bytes == 256

    # the second consumed frame reaches the half window and releases the sender.
    assert await caller.receive() == b"q" * 64
    await _settle()
    assert owner.unacked_bytes == 256
    assert source.reads == stalled_at + 2

    producer.cancel()
    await asyncio.gather(producer, return_exceptions=True)


async def test_credit_acknowledgement_is_cumulative() -> None:
    """a later acknowledgement releases everything before it, so a lost one self-heals."""
    bus = _Bus()
    endpoint = _endpoint(max_chunk=64, credit=256)
    owner, caller = await _wired_pair(bus, endpoint)

    await owner.send(b"r" * 128)
    assert owner.unacked_bytes == 128

    up = endpoint.subject("up").path
    dropped_acks: list[bytes] = []

    def _drop_credit(subject: str, payload: bytes) -> bool:
        if subject == up and _decode_frame(payload)[0] == _TAG_CREDIT:
            dropped_acks.append(payload)
            return True
        return False

    bus.suppress = _drop_credit
    for _ in range(2):
        await caller.receive()
    # the acknowledgement was really sent and really lost.
    assert len(dropped_acks) == 1
    assert owner.unacked_bytes == 128

    # the next one carries the same cumulative high-water mark and repairs it.
    bus.suppress = None
    await owner.send(b"r" * 128)
    for _ in range(2):
        await caller.receive()
    assert owner.unacked_bytes == 0


async def test_a_peer_that_ignores_credit_faults_rather_than_buffering() -> None:
    """unconsumed bytes past the agreed window end the stream instead of growing."""
    bus = _Bus()
    endpoint = _endpoint(max_chunk=64, credit=128)
    _owner, caller = await _wired_pair(bus, endpoint)

    down = endpoint.subject("down").path
    for seq in range(1, 5):
        await bus.deliver(down, _encode_frame(_TAG_DATA, seq, b"w" * 64))

    with pytest.raises(PipeError, match="past the 128-byte window"):
        await caller.receive()


async def test_a_ready_frame_consumes_no_data_sequence() -> None:
    """control frames do not advance the data sequence either end is counting."""
    bus = _Bus()
    endpoint = _endpoint()
    owner, caller = await _wired_pair(bus, endpoint)

    # the caller's open() already sent one ready frame; a second changes nothing.
    await bus.deliver(endpoint.subject("up").path, _encode_frame(_TAG_READY, 0, b""))
    await owner.send(b"first")
    assert await caller.receive() == b"first"


async def test_an_error_frame_consumes_no_data_sequence() -> None:
    """an error frame is a teardown, so it is not part of the ordered stream."""
    bus = _Bus()
    endpoint = _endpoint()
    owner, _caller = await _wired_pair(bus, endpoint)

    await owner.send(b"one")
    await owner.send_error(RuntimeError("boom"))
    down = endpoint.subject("down").path
    tags = [_decode_frame(payload)[0:2] for subject, payload in bus.published if subject == down]
    assert tags == [(_TAG_DATA, 1), (_TAG_ERROR, 0)]


# --------------------------------------------------------------------------
# coordinates that cannot carry a stream
# --------------------------------------------------------------------------


def test_endpoint_refuses_a_frame_larger_than_the_window() -> None:
    """a window smaller than one frame is a stopped pipe, refused at attach."""
    with pytest.raises(PipeProtocolError, match="exceeds the credit window"):
        PipeEndpoint(
            tool="t",
            pod_id="p",
            nonce="n",
            max_chunk=128,
            credit=64,
            version=PIPE_PROTOCOL_VERSION,
        )


def test_endpoint_refuses_non_positive_limits() -> None:
    """a zero window or frame size cannot move a byte, and says so."""
    with pytest.raises(PipeProtocolError, match="must be positive"):
        PipeEndpoint(tool="t", pod_id="p", nonce="n", max_chunk=64, credit=0, version=PIPE_PROTOCOL_VERSION)


def test_attach_reply_refuses_an_incoherent_pair_from_the_owner() -> None:
    """the invariant holds against a peer as well as against a local caller."""
    with pytest.raises(PipeProtocolError, match="exceeds the credit window"):
        _decode_attach_reply(
            b'{"tool": "t", "pod_id": "p", "nonce": "n", "max_chunk": 128, "credit": 64, "version": 1}'
        )


# --------------------------------------------------------------------------
# stream subjects
# --------------------------------------------------------------------------


def test_pipe_subject_shape() -> None:
    """the stream subject is namespace, family, tool digest, pod, nonce, direction."""
    subject = Subjects.pipe("tools.scrape-zone_alpha.1-0-0", "pod-abc", "n0", "down")
    prefix, family, digest, pod, nonce, direction = subject.path.split(".")
    assert (prefix, family, pod, nonce, direction) == ("3tears", "pipe", "pod-abc", "n0", "down")
    assert digest == hashlib.sha256(b"tools.scrape-zone_alpha.1-0-0").hexdigest()
    assert subject.kind == "point"


def test_pipe_subject_directions_are_distinct_subjects() -> None:
    """the two halves are separate subjects, which is what lets a grant split them."""
    args = ("tools.t.1-0-0", "pod-abc", "n0")
    assert Subjects.pipe(*args, "up").path != Subjects.pipe(*args, "down").path


def test_pipe_subject_token_is_subject_safe_for_a_hostile_tool_name() -> None:
    """a registered tool name is unvalidated, so the digest is what has to hold."""
    subject = Subjects.pipe("evil tool*name>here", "pod-abc", "n0", "down")
    token = subject.path.split(".")[2]
    assert set(token) <= set("0123456789abcdef")
    assert len(token) == 64
    for illegal in (" ", "*", ">"):
        assert illegal not in subject.path


def test_pipe_subject_rejects_empty_identifiers() -> None:
    """an empty tool or nonce is a programming error, not a silent empty token."""
    with pytest.raises(ValueError, match="tool_namespace_name must be non-empty"):
        Subjects.pipe("", "pod", "n0", "down")
    with pytest.raises(ValueError, match="nonce must be non-empty"):
        Subjects.pipe("tools.t.1-0-0", "pod", "", "down")


def test_pipe_pod_grant_is_exact_on_tool_and_pod_and_direction() -> None:
    """only the nonce is wildcarded: the pod mints it long after the grant is issued."""
    grant = Subjects.pipe_pod_wildcard("tools.t.1-0-0", "pod-abc", "down")
    assert grant.path == f"3tears.pipe.{hashlib.sha256(b'tools.t.1-0-0').hexdigest()}.pod-abc.*.down"
    assert grant.kind == "pattern"


def test_pipe_pod_grant_does_not_span_a_peers_pod_or_the_other_direction() -> None:
    """the isolation is in the literal segments, so peers and halves never overlap."""
    mine = Subjects.pipe_pod_wildcard("tools.t.1-0-0", "pod-abc", "down")
    peer = Subjects.pipe_pod_wildcard("tools.t.1-0-0", "pod-xyz", "down")
    other_tool = Subjects.pipe_pod_wildcard("tools.other.1-0-0", "pod-abc", "down")
    my_other_half = Subjects.pipe_pod_wildcard("tools.t.1-0-0", "pod-abc", "up")
    assert len({mine.path, peer.path, other_tool.path, my_other_half.path}) == 4


def test_pipe_any_pod_grant_keeps_the_direction_exact() -> None:
    """a principal fronting every pod still cannot publish onto an owner's own half."""
    up = Subjects.pipe_any_pod_wildcard("up")
    assert up.path == "3tears.pipe.*.*.*.up"
    assert up.kind == "pattern"
    assert Subjects.pipe_any_pod_wildcard("down").path == "3tears.pipe.*.*.*.down"
