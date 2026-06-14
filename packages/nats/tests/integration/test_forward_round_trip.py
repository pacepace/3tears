"""Integration test: owner-routed forward against a real NATS broker.

Two NatsClient connections stand in for two pods: pod A serves a key
via :func:`serve_owner`; pod B forwards a request to that key via
:func:`forward`. Exercises the real request/reply path, the queue-group
single-dispatch under a brief two-owner overlap, the no-owner timeout,
and the handler-raise -> ForwardedHandlerError propagation that the
in-process unit tests cannot prove end to end.

Uses the canonical session-scoped ``nats_container`` fixture from
:mod:`threetears.core.testing.fixtures`; a checkout without docker skips
cleanly via that fixture's docker gate.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from threetears.nats import (
    ForwardedHandlerError,
    NatsClient,
    NoOwnerError,
    forward,
    serve_owner,
    set_default_namespace,
)

pytestmark = pytest.mark.integration


_NS = "fwdtest"


async def _connect(url: str, name: str) -> NatsClient:
    """connect a client bound to the forward-test namespace."""
    set_default_namespace(_NS)
    return await NatsClient.connect(
        nats_url=url,
        nats_subject_namespace=_NS,
        client_name=name,
    )


async def test_forward_round_trip_owner_handler_runs(nats_container: str) -> None:
    """pod B's forward reaches pod A's handler and returns its reply bytes."""
    async with await _connect(nats_container, "owner-A") as a, await _connect(nats_container, "caller-B") as b:
        seen: list[bytes] = []

        async def handler(payload: bytes) -> bytes:
            seen.append(payload)
            return b"reply:" + payload

        async with serve_owner(a, "key-1", handler):
            reply = await forward(b, "key-1", b"hello", timeout=timedelta(seconds=3))

        # the owner's handler actually ran (proves routing, not a local echo)...
        assert seen == [b"hello"]
        # ...and the caller received the handler's reply bytes verbatim.
        assert reply == b"reply:hello"


async def test_forward_propagates_handler_exception(nats_container: str) -> None:
    """a handler that raises surfaces to the caller as ForwardedHandlerError."""

    class _CasConflict(Exception):
        pass

    async with await _connect(nats_container, "owner-A") as a, await _connect(nats_container, "caller-B") as b:

        async def handler(_payload: bytes) -> bytes:
            raise _CasConflict("expected seq 7, got 9")

        async with serve_owner(a, "key-err", handler):
            with pytest.raises(ForwardedHandlerError) as excinfo:
                await forward(b, "key-err", b"x", timeout=timedelta(seconds=3))

        # the original type name + message crossed the wire intact, so the
        # consumer can map it back onto its own typed exception.
        assert excinfo.value.type_name == "_CasConflict"
        assert excinfo.value.message == "expected seq 7, got 9"


async def test_forward_no_owner_raises_no_owner_error(nats_container: str) -> None:
    """forwarding to a key nobody serves raises NoOwnerError within the timeout."""
    async with await _connect(nats_container, "caller-B") as b:
        with pytest.raises(NoOwnerError):
            await forward(b, "unserved-key", b"x", timeout=timedelta(milliseconds=500))


async def test_forward_after_owner_stops_raises_no_owner_error(nats_container: str) -> None:
    """once the owner exits its serve context, a later forward is NoOwnerError."""
    async with await _connect(nats_container, "owner-A") as a, await _connect(nats_container, "caller-B") as b:

        async def handler(payload: bytes) -> bytes:
            return b"ok"

        async with serve_owner(a, "key-stop", handler):
            assert await forward(b, "key-stop", b"x", timeout=timedelta(seconds=3)) == b"ok"

        # context exited -> subscription dropped -> no owner serves the key.
        with pytest.raises(NoOwnerError):
            await forward(b, "key-stop", b"x", timeout=timedelta(milliseconds=500))


async def test_two_owners_queue_group_dispatches_to_exactly_one(nats_container: str) -> None:
    """a brief two-owner overlap (handoff window) still dispatches each request once."""
    async with (
        await _connect(nats_container, "owner-A1") as a1,
        await _connect(nats_container, "owner-A2") as a2,
        await _connect(nats_container, "caller-B") as b,
    ):
        handled_by: list[str] = []

        def make_handler(name: str):
            async def handler(payload: bytes) -> bytes:
                handled_by.append(name)
                return name.encode("utf-8")

            return handler

        # both pods believe they own the same key simultaneously.
        async with serve_owner(a1, "contended", make_handler("a1")):
            async with serve_owner(a2, "contended", make_handler("a2")):
                replies = [await forward(b, "contended", b"x", timeout=timedelta(seconds=3)) for _ in range(8)]

        # every request was handled exactly once (no duplicate dispatch),
        # by one of the two owners.
        assert len(handled_by) == 8
        assert all(r in (b"a1", b"a2") for r in replies)
