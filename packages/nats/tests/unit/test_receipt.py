"""unit tests for :mod:`threetears.nats._receipt`.

A subscription takes each message off the connection the moment it arrives and dates
it there, so a request/reply server can hold a request to the deadline its caller
set. Taking messages eagerly moves the backlog out of nats-py's pending queue and into
this one, so this one has to carry the bound nats-py's did: past it the client stops
taking messages, and they wait on the connection where nats-py's own slow-consumer
limit applies.
"""

from __future__ import annotations

import asyncio

import pytest

from threetears.nats._receipt import ReceiptBacklog  # noqa: SLF001 - module is private by design; this is its test


# parity-exempt: the backlog reads only ``.data`` off a nats-py Msg, so the stand-in carries only that
class _FakeMsg:
    """the one attribute of a nats-py ``Msg`` the backlog reads."""

    def __init__(self, data: bytes) -> None:
        self.data = data


async def _parked(awaitable: asyncio.Future[object] | asyncio.Task[object]) -> bool:
    """report whether ``awaitable`` is still waiting after the loop has had a chance to run it.

    :param awaitable: the task to check
    :ptype awaitable: asyncio.Future[object] | asyncio.Task[object]
    :return: True when it has not finished
    :rtype: bool
    """
    for _ in range(5):
        await asyncio.sleep(0)
    return not awaitable.done()


@pytest.mark.asyncio
async def test_messages_come_out_in_arrival_order_with_their_arrival_times() -> None:
    backlog = ReceiptBacklog(msgs_limit=10, bytes_limit=1024)
    first, second = _FakeMsg(b"a"), _FakeMsg(b"b")

    await backlog.put(first, received=1.0)
    await backlog.put(second, received=2.0)

    assert await backlog.get() == (first, 1.0)
    assert await backlog.get() == (second, 2.0)


@pytest.mark.asyncio
async def test_a_closed_backlog_hands_out_what_it_holds_then_ends() -> None:
    """messages already taken off the connection were received, so they are still delivered."""
    backlog = ReceiptBacklog(msgs_limit=10, bytes_limit=1024)
    held = _FakeMsg(b"a")
    await backlog.put(held, received=1.0)

    await backlog.close()

    assert await backlog.get() == (held, 1.0)
    assert await backlog.get() is None


@pytest.mark.asyncio
async def test_a_reader_waiting_on_an_empty_backlog_ends_when_it_closes() -> None:
    backlog = ReceiptBacklog(msgs_limit=10, bytes_limit=1024)
    reader = asyncio.ensure_future(backlog.get())
    assert await _parked(reader)

    await backlog.close()

    assert await asyncio.wait_for(reader, timeout=1.0) is None


@pytest.mark.asyncio
async def test_the_message_count_bound_stops_taking_messages_until_one_leaves() -> None:
    backlog = ReceiptBacklog(msgs_limit=2, bytes_limit=1024)
    await backlog.put(_FakeMsg(b"a"), received=1.0)
    await backlog.put(_FakeMsg(b"b"), received=2.0)

    third = asyncio.ensure_future(backlog.put(_FakeMsg(b"c"), received=3.0))
    assert await _parked(third), "a full backlog took a message past its count bound"

    await backlog.get()
    await asyncio.wait_for(third, timeout=1.0)


@pytest.mark.asyncio
async def test_the_byte_bound_stops_taking_messages_until_bytes_leave() -> None:
    backlog = ReceiptBacklog(msgs_limit=100, bytes_limit=8)
    await backlog.put(_FakeMsg(b"12345678"), received=1.0)

    next_put = asyncio.ensure_future(backlog.put(_FakeMsg(b"x"), received=2.0))
    assert await _parked(next_put), "a full backlog took a message past its byte bound"

    await backlog.get()
    await asyncio.wait_for(next_put, timeout=1.0)


def test_a_bound_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="msgs_limit"):
        ReceiptBacklog(msgs_limit=0, bytes_limit=1)
    with pytest.raises(ValueError, match="bytes_limit"):
        ReceiptBacklog(msgs_limit=1, bytes_limit=0)
