"""Integration test: the byte pipe against a real NATS broker.

Two NatsClient connections stand in for two pods. The owner serves attaches
for a key with :func:`serve_pipe`; the caller attaches with
:func:`attach_pipe` and opens its end with :func:`open_pipe`. What the
in-process unit tests cannot prove is here: a multi-megabyte payload really
survives fragmentation, publication, dispatch and reassembly byte for byte,
and the credit window really stops a producer's reads when the consumer at
the far end of a real connection stops acknowledging.

The backpressure assertion is on the PRODUCER'S OWN READS, deliberately. It
is not on ``NatsClient.overflow_events``: that counter increments only when
``OutboundBufferLimitError`` is raised at the publish boundary while the
client is disconnected or reconnecting with a full pending buffer
(``NatsClient._note_if_outbound_overflow``), so a healthy connected test
reads zero whether or not any credit accounting exists at all. It would pass
against a pipe with the whole mechanism deleted.

``test_credit_window_sizing_measurements`` is not a behavioural test. It
reports the two numbers that size ``DEFAULT_CREDIT_WINDOW_BYTES`` -- sustained
throughput, and the round-trip latency of a credit acknowledgement -- so that
default rests on a measurement instead of on taste. It asserts only that both
are sane, because pinning a machine-dependent number would fail on the next
machine.

Uses the canonical session-scoped ``nats_container`` fixture from
:mod:`threetears.core.testing.fixtures`; a checkout without docker skips
cleanly via that fixture's docker gate.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from datetime import timedelta

import pytest

from threetears.nats import NatsClient, Subjects, set_default_namespace
from threetears.nats.pipe import (
    DEFAULT_MAX_CHUNK_BYTES,
    PipeStream,
    attach_pipe,
    open_pipe,
    serve_pipe,
)

pytestmark = pytest.mark.integration


_NS = "pipetest"
#: the tool-name NODE the owning pod holds; both ends derive their subjects from it.
_OWNED_NODE = "scrape-zone_alpha"
_POD_ID = "pod-7f3c"


async def _connect(url: str, name: str) -> NatsClient:
    """connect a client bound to the pipe-test namespace."""
    set_default_namespace(_NS)
    return await NatsClient.connect(
        nats_url=url,
        nats_subject_namespace=_NS,
        client_name=name,
    )


class _CountingSource:
    """a byte source that records every read, so backpressure can be asserted on it.

    the credit window is only worth anything if the block reaches the thing
    producing the bytes rather than merely the socket, so this is what the slow
    consumer test watches.
    """

    def __init__(self, payload: bytes, chunk: int) -> None:
        self._payload = payload
        self._chunk = chunk
        self._offset = 0
        self.reads = 0

    async def read(self) -> bytes:
        """return the next chunk, or empty bytes once the payload is spent."""
        self.reads += 1
        window = self._payload[self._offset : self._offset + self._chunk]
        self._offset += len(window)
        return window


async def test_multi_megabyte_payload_survives_the_pipe_byte_for_byte(nats_container: str) -> None:
    """a payload far larger than one frame comes out the other end unchanged."""
    payload = os.urandom(4 * 1024 * 1024)
    digest = hashlib.sha256(payload).hexdigest()
    family = Subjects.hitl_forward_family(_OWNED_NODE)

    async def handler(stream: PipeStream) -> None:
        await stream.send(payload)

    async with await _connect(nats_container, "owner") as owner, await _connect(nats_container, "caller") as caller:
        async with serve_pipe(
            owner,
            "display-1",
            tool=_OWNED_NODE,
            pod_id=_POD_ID,
            handler=handler,
            family=family,
        ):
            endpoint = await attach_pipe(caller, "display-1", family=family)
            # the reply carries the readable NODE, and the subject the caller then derives
            # from it is the digest of its ROOTED form -- the two are not interchangeable
            # and this is where that split is exercised. rooted because the pod's grant on
            # this subject is minted at connect from the bare node on its ``tool_pods`` row
            # while the owner may hold either spelling, so ``Subjects.pipe`` normalizes both
            # onto one digest.
            assert endpoint.tool == _OWNED_NODE
            assert Subjects.pipe(endpoint.tool, endpoint.pod_id, endpoint.nonce, "down").path.split(".")[2] == (
                hashlib.sha256(Subjects.tool_provider_node(_OWNED_NODE).encode("utf-8")).hexdigest()
            )

            received = bytearray()
            async with open_pipe(caller, endpoint) as stream:
                while (chunk := await stream.receive()) is not None:
                    received += chunk

    assert len(received) == len(payload)
    assert hashlib.sha256(bytes(received)).hexdigest() == digest


async def test_bytes_flow_in_both_directions(nats_container: str) -> None:
    """the pipe is bidirectional: the caller's bytes reach the owner's handler."""
    family = Subjects.hitl_forward_family(_OWNED_NODE)
    seen: list[bytes] = []

    async def handler(stream: PipeStream) -> None:
        while (chunk := await stream.receive()) is not None:
            seen.append(chunk)
            await stream.send(b"echo:" + chunk)

    async with await _connect(nats_container, "owner") as owner, await _connect(nats_container, "caller") as caller:
        async with serve_pipe(
            owner,
            "duplex-1",
            tool=_OWNED_NODE,
            pod_id=_POD_ID,
            handler=handler,
            family=family,
        ):
            endpoint = await attach_pipe(caller, "duplex-1", family=family)
            async with open_pipe(caller, endpoint) as stream:
                await stream.send(b"keystroke")
                assert await stream.receive() == b"echo:keystroke"

    assert seen == [b"keystroke"]


async def test_a_slow_consumer_stops_and_resumes_the_producers_reads(nats_container: str) -> None:
    """a consumer that withholds acknowledgements stops the producer reading its source."""
    chunk_bytes = 64 * 1024
    # window = four frames. the payload is twenty times the window, so a producer
    # whose reads were NOT flow-controlled would drain the whole source long
    # before the consumer read a single byte.
    window = 4 * chunk_bytes
    payload = os.urandom(20 * window)
    source = _CountingSource(payload, chunk_bytes)
    family = Subjects.hitl_forward_family(_OWNED_NODE)

    async def handler(stream: PipeStream) -> None:
        while data := await source.read():
            await stream.send(data)

    async with await _connect(nats_container, "owner") as owner, await _connect(nats_container, "caller") as caller:
        async with serve_pipe(
            owner,
            "slow-1",
            tool=_OWNED_NODE,
            pod_id=_POD_ID,
            handler=handler,
            family=family,
            credit=window,
            max_chunk=chunk_bytes,
        ):
            endpoint = await attach_pipe(caller, "slow-1", family=family, credit=window, max_chunk=chunk_bytes)
            assert endpoint.credit == window

            async with open_pipe(caller, endpoint) as stream:
                # consume nothing at all, and let the producer run as far as it can.
                await asyncio.sleep(0.5)
                stalled_at = source.reads
                assert stalled_at * chunk_bytes < len(payload), "the producer drained its whole source unblocked"

                await asyncio.sleep(0.5)
                assert source.reads == stalled_at, "the producer kept reading past a full, unacknowledged window"

                # consume a half window, which is what owes the producer an
                # acknowledgement, and it must move again.
                consumed = 0
                while consumed < window // 2:
                    chunk = await stream.receive()
                    assert chunk is not None
                    consumed += len(chunk)
                await asyncio.sleep(0.5)
                assert source.reads > stalled_at, "the acknowledgement did not release the producer"

                # and having moved, it stops again at the new window boundary.
                resumed_at = source.reads
                await asyncio.sleep(0.5)
                assert source.reads == resumed_at


async def test_credit_window_sizing_measurements(nats_container: str) -> None:
    """report the throughput and acknowledgement latency that size the default window.

    The bandwidth-delay product of these two is the floor below which a sender
    stalls waiting for credit it has already earned. Recorded here rather than
    reasoned about, because the alternative is a round number nobody can defend.
    """
    payload = os.urandom(8 * 1024 * 1024)
    family = Subjects.hitl_forward_family(_OWNED_NODE)

    async def bulk_handler(stream: PipeStream) -> None:
        await stream.send(payload)

    async with await _connect(nats_container, "owner") as owner, await _connect(nats_container, "caller") as caller:
        async with serve_pipe(owner, "bulk-1", tool=_OWNED_NODE, pod_id=_POD_ID, handler=bulk_handler, family=family):
            endpoint = await attach_pipe(caller, "bulk-1", family=family)
            started = time.perf_counter()
            total = 0
            async with open_pipe(caller, endpoint) as stream:
                while (chunk := await stream.receive()) is not None:
                    total += len(chunk)
            elapsed = time.perf_counter() - started

    throughput_mb_s = (total / elapsed) / 1_000_000

    # the acknowledgement round trip, measured on its own: one frame outstanding,
    # timed from the consuming read that owes an acknowledgement to the sender
    # seeing its window released.
    ack_seconds = await _measure_ack_round_trip(nats_container, family)

    bdp_bytes = throughput_mb_s * 1_000_000 * ack_seconds
    print(  # noqa: T201 -- the measurement IS this test's output
        f"\npipe credit sizing: throughput={throughput_mb_s:.1f} MB/s "
        f"ack_round_trip={ack_seconds * 1000:.2f} ms "
        f"bandwidth_delay_product={bdp_bytes:.0f} B"
    )

    assert total == len(payload)
    assert throughput_mb_s > 0
    assert 0 < ack_seconds < 1.0


async def _measure_ack_round_trip(url: str, family: str) -> float:
    """time one credit acknowledgement from the consuming read to the released producer.

    The producer parks in :meth:`PipeStream.send` on a full window and records
    when that call returns; the consumer records when it made the read that owes
    the acknowledgement. Measured through the real blocking path rather than by
    polling ``unacked_bytes``: a poll loop on the shared event loop starves the
    peer's own flusher, which inflates the number it is trying to measure.
    """
    chunk_bytes = DEFAULT_MAX_CHUNK_BYTES
    window = 2 * chunk_bytes
    filled = asyncio.Event()
    released = asyncio.Event()
    timings: dict[str, float] = {}

    async def handler(stream: PipeStream) -> None:
        await stream.send(b"m" * window)  # fills the window exactly
        filled.set()
        await stream.send(b"m" * chunk_bytes)  # parks until the window is released
        timings["released"] = time.perf_counter()
        released.set()

    async with await _connect(url, "ack-owner") as owner, await _connect(url, "ack-caller") as caller:
        async with serve_pipe(
            owner,
            "ack-1",
            tool=_OWNED_NODE,
            pod_id=_POD_ID,
            handler=handler,
            family=family,
            credit=window,
            max_chunk=chunk_bytes,
        ):
            endpoint = await attach_pipe(caller, "ack-1", family=family, credit=window, max_chunk=chunk_bytes)
            async with open_pipe(caller, endpoint) as stream:
                await asyncio.wait_for(filled.wait(), timeout=timedelta(seconds=5).total_seconds())
                await asyncio.sleep(0.25)  # quiesce so the clock times the ack, not the fill
                started = time.perf_counter()
                await stream.receive()
                await asyncio.wait_for(released.wait(), timeout=timedelta(seconds=5).total_seconds())
                return timings["released"] - started
