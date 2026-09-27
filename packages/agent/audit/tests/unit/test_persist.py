"""The audit persister's message handling and startup, without a broker or a database.

Delivery is at-least-once, so the handler's three outcomes are the contract: a valid event is
persisted THEN acked; an event that will never parse is acked and dropped (never redelivered
forever); a database fault RAISES so the pull consumer retries and finally dead-letters it.
"""

from __future__ import annotations

import json
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


async def test_start_ensures_a_memory_stream_by_default_with_the_dead_letter_and_stops_cleanly() -> None:
    from threetears.nats import Subjects, set_default_namespace

    set_default_namespace("unitaudit")
    nats = _Nats()
    handle = await start_audit_persister(nats, _Db(), durable="app-audit-persist")
    try:
        [stream] = nats.streams
        assert stream["name"] == AUDIT_STREAM_NAME and stream["storage"] == "memory"
        assert Subjects.audit_deadletter().path in stream["subjects"]
        [sub] = nats.subscriptions
        assert sub["durable"] == "app-audit-persist"
        assert sub["dead_letter_subject"] == Subjects.audit_deadletter()
        assert sub["stream"] == "ns-audit", "the ensured stream is named, sparing a stream-names lookup"
    finally:
        await handle.stop()
    assert nats.consumer.stopped


async def test_stop_cancels_the_task_even_when_the_consumer_stop_fails() -> None:
    import asyncio

    from threetears.agent.audit.persist import AuditPersisterHandle

    class _Broken:
        async def stop(self) -> None:
            raise ConnectionError("connection already closed")

    task = asyncio.create_task(asyncio.Event().wait())
    with pytest.raises(ConnectionError):
        await AuditPersisterHandle(_Broken(), task).stop()
    await asyncio.sleep(0)
    assert task.cancelled() or task.done()


async def test_an_undecodable_event_is_logged_where_it_sits_in_the_stream(caplog: pytest.LogCaptureFixture) -> None:
    """an operator told an audit record is missing matches it to the drop by subject and stream sequence;
    the exception's text stays out, because it can echo the event's personal content."""

    class _Sequence:
        stream = 4211

    class _Metadata:
        sequence = _Sequence()

    msg = _Msg(b'{"email": "someone@example.test"')
    msg.subject = "unitaudit.audit.admin.user.create"  # type: ignore[attr-defined]
    msg.metadata = _Metadata()  # type: ignore[attr-defined]

    with caplog.at_level("WARNING"):
        await handle_audit_message(_Db(), msg)

    [record] = [r for r in caplog.records if "undecodable" in r.getMessage()]
    extra = record.__dict__["extra_data"]
    assert extra["subject"] == "unitaudit.audit.admin.user.create"
    assert extra["stream_sequence"] == 4211
    assert "someone@example.test" not in str(extra)


def test_an_existing_table_gains_every_column_the_insert_names() -> None:
    """an insert naming a column the table lacks fails for every event, and every event dead-letters.

    ``outcome`` is NOT NULL with a default, so it is added with that default; the key and the four
    required fields are what any audit table already has.
    """
    from threetears.agent.audit.persist import AUDIT_EVENTS_DDL

    added = {
        statement.removeprefix("ALTER TABLE audit_events ADD COLUMN IF NOT EXISTS ").split(" ", 1)[0]
        for statement in AUDIT_EVENTS_DDL
        if statement.startswith("ALTER TABLE audit_events ADD COLUMN")
    }
    inserted = {
        "timestamp", "event_type", "action", "outcome", "actor_user_id", "acting_as_principal_id",
        "calling_agent_id", "owner_agent_id", "customer_id", "resource_namespace_id",
        "resource_namespace_type", "correlation_id", "conversation_id", "details", "ip_address",
    }  # fmt: skip
    required = {"timestamp", "event_type", "action", "correlation_id"}
    assert added, "the DDL must carry its migration"
    assert added == inserted - required
    assert (
        "ALTER TABLE audit_events ADD COLUMN IF NOT EXISTS outcome TEXT NOT NULL DEFAULT 'success'" in AUDIT_EVENTS_DDL
    )


class _ErasureDb:
    """answers the erasure's keyset reads with one batch of stored rows and records its updates."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._batches = [rows, []]
        self.updates: list[tuple[Any, ...]] = []

    async def execute(self, sql: str, *args: Any) -> str:
        self.updates.append(args)
        return "UPDATE 1"

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        return self._batches.pop(0)


async def test_erasure_applies_the_platform_rule_with_nothing_injected() -> None:
    """the production path, run where it cannot skip: the platform's rule, and the one answer type."""
    from threetears.agent.audit import ANONYMIZED_MARKER, AuditAnonymization
    from threetears.agent.audit.persist import anonymize_audit_rows

    user_id = str(uuid.uuid4())
    row_id = uuid.uuid4()
    db = _ErasureDb(
        [
            {
                "id": row_id,
                "event_type": "admin.user.create",
                "details": '{"user_id": "%s", "email": "someone@example.test"}' % user_id,
                "ip_address": "203.0.113.9",
            }
        ]
    )

    result = await anonymize_audit_rows(db, actor_user_ids=[uuid.uuid4()])

    assert result == AuditAnonymization(rows_matched=1, rows_changed=1)
    [(details, address, updated_id)] = db.updates
    assert updated_id == row_id
    assert address is None, "an address is removed"
    stored = json.loads(details)
    assert stored == {"user_id": user_id, "email": ANONYMIZED_MARKER}


async def test_erasure_takes_no_replacement_for_the_rule() -> None:
    """a deployment that could hand in its own scrub is the per-consumer divergence the one rule ends."""
    from threetears.agent.audit.persist import anonymize_audit_rows

    with pytest.raises(TypeError):
        await anonymize_audit_rows(  # type: ignore[call-arg]
            _ErasureDb([]), actor_user_ids=[uuid.uuid4()], anonymize=lambda details, **_: dict(details)
        )
