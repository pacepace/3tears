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
from threetears.agent.audit.persist import (
    AUDIT_STREAM_NAME,
    handle_audit_message,
    persist_audit_event,
    start_audit_persister,
)


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


def _stored_details(db: _Db) -> dict[str, Any]:
    """the ``details`` text the insert bound, parsed.

    :param db: the recording database after one insert
    :ptype db: _Db
    :return: the stored details
    :rtype: dict[str, Any]
    """
    ((_sql, *args),) = db.executed
    (details,) = [arg for arg in args if isinstance(arg, str) and arg.startswith("{")]
    stored: dict[str, Any] = json.loads(details)
    return stored


async def test_a_datetime_in_details_is_stored_in_the_one_form() -> None:
    """a direct caller's in-process details store the instant as every tier does, not pydantic's ``Z``."""
    db = _Db()
    when = datetime(2026, 10, 1, 12, 30, tzinfo=UTC)
    event = AuditEvent(
        id=uuid.uuid7(),
        timestamp=when,
        event_type="admin.user.create",
        action="user.create",
        correlation_id=uuid.uuid4(),
        details={"date_expires": when, "nested": [{"at": when}], "user_id": uuid.UUID(int=7)},
    )

    await persist_audit_event(db, event)

    assert _stored_details(db) == {
        "date_expires": "2026-10-01T12:30:00.000000+00:00",
        "nested": [{"at": "2026-10-01T12:30:00.000000+00:00"}],
        "user_id": str(uuid.UUID(int=7)),
    }


async def test_a_naive_datetime_in_details_is_refused_naming_it() -> None:
    event = AuditEvent(
        id=uuid.uuid7(),
        timestamp=datetime.now(UTC),
        event_type="admin.user.create",
        action="user.create",
        correlation_id=uuid.uuid4(),
        details={"date_expires": datetime(2026, 10, 1, 12, 30)},
    )
    with pytest.raises(ValueError, match="naive datetime in 'details.date_expires'"):
        await persist_audit_event(_Db(), event)


async def test_an_instant_that_arrived_over_the_wire_is_stored_in_the_one_form() -> None:
    """the consumer path's details are JSON text; pydantic's ``Z`` spelling of an instant is re-spelled once, here.

    a producer's ``model_dump_json`` writes ``2026-10-01T12:30:00Z`` (no fraction, ``Z``); every other
    tier stores ``2026-10-01T12:30:00.000000+00:00``. the persister is the one point the wire's text
    becomes stored data, so it is where the two spellings of one instant become one.
    """
    db = _Db()
    event = AuditEvent(
        id=uuid.uuid7(),
        timestamp=datetime.now(UTC),
        event_type="admin.user.create",
        action="user.create",
        correlation_id=uuid.uuid4(),
        details={
            "date_expires": datetime(2026, 10, 1, 12, 30, tzinfo=UTC),
            "nested": [{"at": "2026-10-01T14:30:00.25+02:00"}],
        },
    )
    wire = event.model_dump_json().encode()
    assert b'"2026-10-01T12:30:00Z"' in wire, "the producer's wire spelling this test exists for changed"

    await handle_audit_message(db, _Msg(wire))

    assert _stored_details(db) == {
        "date_expires": "2026-10-01T12:30:00.000000+00:00",
        "nested": [{"at": "2026-10-01T12:30:00.250000+00:00"}],
    }


async def test_text_that_only_resembles_an_instant_is_stored_as_it_arrived() -> None:
    """only a full date-time naming its offset is an instant; a date, a naive time or prose is kept verbatim."""
    db = _Db()
    kept = {
        "date_of_birth": "2026-10-01",
        "local_wall_clock": "2026-10-01T12:30:00",
        "note": "renewed 2026-10-01T12:30:00Z by hand",
        "spaced": "2026-10-01 12:30:00+00:00",
        "count": 3,
        "flag": None,
    }
    event = AuditEvent(
        id=uuid.uuid7(),
        timestamp=datetime.now(UTC),
        event_type="admin.user.create",
        action="user.create",
        correlation_id=uuid.uuid4(),
        details=kept,
    )

    await handle_audit_message(db, _Msg(event.model_dump_json().encode()))

    assert _stored_details(db) == kept


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
        assert stream["max_age_seconds"] is None, "no age limit unless one is asked for"
        assert Subjects.audit_deadletter().path in stream["subjects"]
        [sub] = nats.subscriptions
        assert sub["durable"] == "app-audit-persist"
        assert sub["dead_letter_subject"] == Subjects.audit_deadletter()
        assert sub["stream"] == "ns-audit", "the ensured stream is named, sparing a stream-names lookup"
    finally:
        await handle.stop()
    assert nats.consumer.stopped


async def test_the_age_limit_reaches_the_stream_declaration() -> None:
    from threetears.nats import set_default_namespace

    set_default_namespace("unitaudit")
    nats = _Nats()
    handle = await start_audit_persister(nats, _Db(), durable="app-audit-persist", max_age_seconds=86_400.0)
    try:
        [stream] = nats.streams
        assert stream["max_age_seconds"] == 86_400.0
    finally:
        await handle.stop()


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
