"""The audit persister's message handling and startup, without a broker or a database.

Delivery is at-least-once, so the handler's three outcomes are the contract: a valid event is
persisted THEN acked; an event that will never parse is acked and dropped (never redelivered
forever); a database fault RAISES so the pull consumer retries and finally dead-letters it.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from threetears.agent.audit import AuditEvent
from threetears.agent.audit.persist import AUDIT_STREAM_NAME, handle_audit_message, start_audit_persister


class _Msg:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.acked = False

    async def ack(self) -> None:
        self.acked = True


class _Db:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.executed: list[tuple[Any, ...]] = []

    async def execute(self, sql: str, *args: Any) -> str:
        if self.fail:
            raise ConnectionError("database down")
        self.executed.append((sql, *args))
        return "INSERT 0 1"

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        return []


def _payload() -> bytes:
    event = AuditEvent(
        id=uuid.uuid7(),
        timestamp=datetime.now(UTC),
        event_type="admin.user.create",
        action="user.create",
        correlation_id=uuid.uuid4(),
    )
    return event.model_dump_json().encode()


async def test_a_valid_event_is_persisted_then_acked() -> None:
    db, msg = _Db(), _Msg(_payload())
    await handle_audit_message(db, msg)
    assert len(db.executed) == 1 and msg.acked


async def test_an_event_that_will_never_parse_is_acked_and_dropped() -> None:
    db, msg = _Db(), _Msg(b"not an envelope")
    await handle_audit_message(db, msg)
    assert db.executed == [] and msg.acked


async def test_a_database_fault_raises_so_the_consumer_retries() -> None:
    db, msg = _Db(fail=True), _Msg(_payload())
    with pytest.raises(ConnectionError):
        await handle_audit_message(db, msg)
    assert not msg.acked


class _Consumer:
    def __init__(self) -> None:
        self.stopped = False

    async def run(self) -> None:
        import asyncio

        await asyncio.Event().wait()

    async def stop(self) -> None:
        self.stopped = True


class _Nats:
    def __init__(self) -> None:
        self.streams: list[dict[str, Any]] = []
        self.subscriptions: list[dict[str, Any]] = []
        self.consumer = _Consumer()

    async def ensure_jetstream_stream(self, **kwargs: Any) -> str:
        self.streams.append(kwargs)
        return f"ns-{kwargs['name']}"

    async def jetstream_pull_subscribe(self, **kwargs: Any) -> _Consumer:
        self.subscriptions.append(kwargs)
        return self.consumer


async def test_start_ensures_a_file_backed_stream_with_the_dead_letter_and_stops_cleanly() -> None:
    from threetears.nats import Subjects, set_default_namespace

    set_default_namespace("unitaudit")
    nats = _Nats()
    handle = await start_audit_persister(nats, _Db(), durable="app-audit-persist")
    try:
        [stream] = nats.streams
        assert stream["name"] == AUDIT_STREAM_NAME and stream["storage"] == "file"
        assert Subjects.audit_deadletter().path in stream["subjects"]
        [sub] = nats.subscriptions
        assert sub["durable"] == "app-audit-persist"
        assert sub["dead_letter_subject"] == Subjects.audit_deadletter()
    finally:
        await handle.stop()
    assert nats.consumer.stopped
