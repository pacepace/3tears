"""A transaction whose caller is cancelled while it waits for ``l3.tx.begin``'s answer leaves no session open.

The hub opens the session when it reads the begin; a caller cancelled before the answer arrives never learned the
``tx_id``, so nothing ended it and the hub held it (its pooled connection with it) until its idle sweep rolled it back
a minute later. The election replay of 2026-11-21 left one so for nearly every failed load: the load's failure
cancelled its other states, one of them waiting on a begin. The begin now runs to its answer and the session it
opened is rolled back.
"""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest
from threetears.nats import Subject

from threetears.core.backends.nats_proxy import NatsProxyL3Backend

_TX_ID = "019d9a00-0000-7000-8000-0000000000cc"


class _SlowBegin:
    """a hub that reads each request at once and answers ``tx.begin`` only when told to."""

    def __init__(self, *, begin_reply: dict[str, Any] | None = None) -> None:
        self.subjects: list[str] = []
        self.payloads: list[dict[str, Any]] = []
        self.begin_read = asyncio.Event()
        self.answer_begin = asyncio.Event()
        self.begin_reply = begin_reply or {"success": True, "tx_id": _TX_ID}

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta | None = None) -> bytes:
        del timeout
        self.subjects.append(subject.path)
        self.payloads.append(json.loads(payload))
        if subject.path.endswith("tx.begin"):
            self.begin_read.set()
            await self.answer_begin.wait()
            return json.dumps(self.begin_reply).encode()
        return json.dumps({"success": True}).encode()

    def backend(self) -> NatsProxyL3Backend:
        nats = MagicMock()
        nats.request_raw = self.request_raw
        nats.max_payload = 1024 * 1024
        return NatsProxyL3Backend(
            nats_client=nats, namespace_prefix="test", agent_id="agent-1", identity_token=lambda: "token"
        )


async def _cancelled_in_begin(hub: _SlowBegin) -> None:
    backend = hub.backend()
    entered = False

    async def caller() -> None:
        nonlocal entered
        async with backend.acquire() as conn, conn.transaction():
            entered = True

    task = asyncio.create_task(caller())
    await hub.begin_read.wait()
    task.cancel()
    # the hub answers after the caller was cancelled
    hub.answer_begin.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    for _ in range(20):
        await asyncio.sleep(0)
    assert not entered


async def test_a_session_begun_for_a_cancelled_caller_is_rolled_back() -> None:
    hub = _SlowBegin()
    await _cancelled_in_begin(hub)
    assert hub.subjects == ["test.l3.tx.begin", "test.l3.tx.rollback"]
    assert hub.payloads[1]["tx_id"] == _TX_ID


async def test_a_begin_the_hub_refused_is_not_rolled_back() -> None:
    hub = _SlowBegin(begin_reply={"success": False, "error_code": "BUSY", "error_message": "busy"})
    await _cancelled_in_begin(hub)
    assert hub.subjects == ["test.l3.tx.begin"]
