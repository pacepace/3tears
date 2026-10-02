"""Integration test: a ping that times out must not stop the connection from reading.

nats-py pairs each PONG with the oldest outstanding PING's future (``Client._pongs``). When
``Client.flush`` times out it cancels its future but leaves it in that queue; the PONG that
arrives afterwards pops the cancelled future, ``set_result`` raises ``InvalidStateError`` in
``_process_pong``, and ``_read_loop``'s catch-all logs "nats: encountered error" and exits. The
connection still reports itself connected, and never reads another byte: every later ping,
request reply and subscription delivery is lost, and nothing reconnects it.

:meth:`NatsClient.ping` is a health probe with a caller's timeout, so a slow PONG under load was
enough to do this to a live pod. Here the timeout is shorter than any real round trip, so the PONG
always arrives after it, deterministically.

Gated on docker: a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import nats.errors
import pytest

from threetears.nats import IncomingMessage, NatsClient, Subject, set_default_namespace

pytestmark = pytest.mark.integration

#: far below any round trip, so the PONG always arrives after the ping gave up.
_EXPIRED_BEFORE_ANY_PONG = 1e-9


async def test_a_ping_that_times_out_leaves_the_connection_reading(nats_container: str) -> None:
    """after a timed-out ping, pings, requests and subscriptions all still work."""
    set_default_namespace("pingtest")
    async with await NatsClient.connect(
        nats_url=nats_container,
        nats_subject_namespace="pingtest",
        client_name="ping-timeout",
    ) as nc:
        received: list[bytes] = []

        async def _on_msg(msg: IncomingMessage) -> None:
            received.append(bytes(msg.data))

        async def _echo(msg: IncomingMessage) -> None:
            assert msg.reply_subject is not None
            await nc.publish_raw_reply(reply_subject=msg.reply_subject, payload=b"pong:" + bytes(msg.data))

        await nc.subscribe(Subject.raw("pingtest.events"), cb=_on_msg)
        await nc.subscribe(Subject.raw("pingtest.echo"), cb=_echo)
        await nc.flush()

        assert await nc.ping(timeout=_EXPIRED_BEFORE_ANY_PONG) is False
        # the late PONG arrives now; give the read loop time to process it
        await asyncio.sleep(0.5)

        assert nc.is_connected
        assert await nc.ping(timeout=2.0) is True, "the read loop stopped after a timed-out ping"
        reply = await nc.request_raw(subject=Subject.raw("pingtest.echo"), payload=b"1", timeout=timedelta(seconds=2))
        assert reply == b"pong:1"
        await nc.publish_raw(subject=Subject.raw("pingtest.events"), payload=b"after")
        for _ in range(40):
            if received:
                break
            await asyncio.sleep(0.05)
        assert received == [b"after"]

        # and a flush that times out is as harmless
        with pytest.raises(nats.errors.FlushTimeoutError):
            await nc.flush(timeout=_EXPIRED_BEFORE_ANY_PONG)
        await asyncio.sleep(0.5)
        assert await nc.ping(timeout=2.0) is True, "the read loop stopped after a timed-out flush"
