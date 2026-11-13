"""a large publish or request leaves its payload uncopied on the caller's event loop, through real nats-py.

nats-py's ``publish`` builds each ``PUB`` as one ``bytes``: the command line, the payload, CRLF, concatenated on the
event loop. A 12 MB Arrow copy's request then held the loop for the whole copy (4.6-7.2 ms on a pool container).
:class:`~threetears.nats.NatsClient` now publishes a payload from ``UNJOINED_FROM_BYTES`` as three pending items the
socket sends from memoryviews.

Every test here drives a real nats-py client through ``NatsClient``'s public methods against :mod:`.byte_server`,
which reads on threads of its own and answers each request with the size and sha256 of what it read, so the bytes
are proven whole and the server's side never shows up in what the test measures.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import tracemalloc
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta

import pytest

from threetears.nats import NatsClient, Subject, set_default_namespace
from threetears.nats.errors import PayloadTooLargeError, PublishError

from .byte_server import ByteServer, byte_server

#: the size of the Arrow copy that showed the stall
_COPY_BYTES = 12 * 1024 * 1024

#: large enough that the old one-``bytes`` copy holds a loop well past a turn on any host
_TIMED_BYTES = 64 * 1024 * 1024

_REQUEST_TIMEOUT = timedelta(seconds=30)


@asynccontextmanager
async def _connected(name: str) -> AsyncIterator[tuple[NatsClient, ByteServer]]:
    """a NatsClient connected to a running :func:`byte_server`.

    :param name: the client's name
    :ptype name: str
    :return: the client and its server
    :rtype: AsyncIterator[tuple[NatsClient, ByteServer]]
    """
    set_default_namespace("3tears")
    with byte_server() as server:
        client = await NatsClient.connect(
            nats_url=server.url, nats_subject_namespace="3tears", client_name=name, verify_jetstream=False
        )
        try:
            yield client, server
        finally:
            await client.shutdown(drain_timeout=timedelta(seconds=2))


def _payload(size: int) -> bytes:
    """``size`` bytes no two runs of which can pass for each other by length alone.

    :param size: bytes wanted
    :ptype size: int
    :return: the payload
    :rtype: bytes
    """
    return (hashlib.sha256(str(size).encode()).digest() * (size // 32 + 1))[:size]


async def test_a_large_request_reaches_the_server_whole() -> None:
    """the server reads exactly the request's bytes, and its answer comes back through the request mux."""
    payload = _payload(_COPY_BYTES)
    async with _connected("large-request") as (client, server):
        reply = await client.request_raw(subject=Subject.raw("3tears.copy"), payload=payload, timeout=_REQUEST_TIMEOUT)
    assert reply == f"{len(payload)} {hashlib.sha256(payload).hexdigest()}".encode()
    assert [(subject, size) for subject, size, _ in server.answered] == [("3tears.copy", _COPY_BYTES)]


async def test_a_large_publish_and_a_small_one_keep_their_order() -> None:
    """a large publish between two small ones arrives between them, whole, with nothing interleaved."""
    large = _payload(_COPY_BYTES)
    async with _connected("large-publish") as (client, server):
        for subject, payload in (("3tears.a", b"before"), ("3tears.b", large), ("3tears.c", b"after")):
            await client.publish_raw(subject=Subject.raw(subject), payload=payload)
        await client.flush()
    assert server.answered == [
        ("3tears.a", 6, hashlib.sha256(b"before").hexdigest()),
        ("3tears.b", _COPY_BYTES, hashlib.sha256(large).hexdigest()),
        ("3tears.c", 5, hashlib.sha256(b"after").hexdigest()),
    ]


async def test_a_large_request_copies_nothing_of_its_payload() -> None:
    """no allocation near the payload's size happens while a 12 MB request is published and answered.

    nats-py's own publish allocates a ``bytes`` of the whole payload plus its command line on the loop; the
    unjoined publish allocates the command line only. The server reads into a reused buffer on its own threads, so
    the peak traced here is the client's.
    """
    payload = _payload(_COPY_BYTES)
    async with _connected("large-request-copy") as (client, _server):
        await client.flush()
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            before, _ = tracemalloc.get_traced_memory()
            await client.request_raw(subject=Subject.raw("3tears.copy"), payload=payload, timeout=_REQUEST_TIMEOUT)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    assert peak - before < _COPY_BYTES // 8, f"publishing a {_COPY_BYTES}-byte request allocated {peak - before} bytes"


async def test_a_payload_over_the_server_limit_is_still_refused() -> None:
    """the unjoined publish keeps nats-py's max-payload refusal: the caller gets the same typed error."""
    async with _connected("too-large") as (client, _server):
        limit = client.raw.max_payload
        with pytest.raises(PayloadTooLargeError):
            await client.publish_raw(subject=Subject.raw("3tears.big"), payload=bytes(limit + 1))


async def test_a_bad_subject_is_still_refused_on_the_unjoined_publish() -> None:
    """a subject holding whitespace is refused before anything is queued, as nats-py refuses it."""
    async with _connected("bad-subject") as (client, server):
        with pytest.raises(PublishError):
            await client.publish_raw(subject=Subject.raw("3tears.bad subject"), payload=_payload(_COPY_BYTES))
        await client.flush()
    assert server.answered == []


async def _worst_tick(until: asyncio.Future[bytes] | None, seconds: float) -> float:
    """the longest the loop went between 0.5 ms sleeps, until ``until`` is done or for ``seconds``.

    :param until: the work to watch; None to watch an idle window
    :ptype until: asyncio.Future[bytes] | None
    :param seconds: how long an idle window lasts
    :ptype seconds: float
    :return: the worst lag past the sleep asked for, in seconds
    :rtype: float
    """
    worst = 0.0
    end = time.perf_counter() + seconds
    while (until is not None and not until.done()) or (until is None and time.perf_counter() < end):
        start = time.perf_counter()
        await asyncio.sleep(0.0005)
        worst = max(worst, time.perf_counter() - start - 0.0005)
    return worst


@pytest.mark.loop_timing
async def test_a_large_request_holds_the_loop_no_longer_than_its_idle_jitter_and_a_turn() -> None:
    """each round: idle window, a 64 MB request, idle window; the request's worst tick against the idle ones.

    The judgement is the Arrow copy's: 5 ms (CPython's switch interval) plus twice the worse idle window either side.
    nats-py's one-``bytes`` copy of 64 MB holds the loop for the whole memcpy in one tick.
    """
    payload = _payload(_TIMED_BYTES)
    rounds: list[tuple[float, float, float]] = []
    async with _connected("large-request-loop") as (client, _server):
        await client.flush()
        for _ in range(3):
            before = await _worst_tick(None, 0.2)
            request = asyncio.ensure_future(
                client.request_raw(subject=Subject.raw("3tears.copy"), payload=payload, timeout=_REQUEST_TIMEOUT)
            )
            during = await _worst_tick(request, 0.0)
            await request
            after = await _worst_tick(None, 0.2)
            rounds.append((before, during, after))
    held = [
        f"round {index}: {during * 1000:.1f} ms against idle {max(before, after) * 1000:.1f} ms"
        for index, (before, during, after) in enumerate(rounds)
        if during >= 0.005 + 2 * max(before, after)
    ]
    assert not held, held
