"""Integration: the audit persister's table, idempotency, prune and erasure on a real Postgres.

A deployment with no hub owns its audit table, so 3tears ships the persister the hub keeps for
itself. These run the SQL against the real column types: JSONB details written and read back as an
object, the two idempotency anchors the envelope documents, an age-based prune, and erasure that
rewrites only ``details`` and ``ip_address`` while every row and id survives.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest

from threetears.agent.audit import AuditEvent
from threetears.agent.audit.persist import (
    anonymize_audit_rows,
    ensure_audit_events_table,
    persist_audit_event,
    prune_audit_events,
)

pytestmark = pytest.mark.integration

_MARKER = "[anonymized]"


@pytest.fixture
async def db(db_container: str) -> AsyncIterator[asyncpg.Pool]:
    schema = f"audit_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(db_container)
    try:
        await admin.execute(f"CREATE SCHEMA {schema}")
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(db_container, min_size=1, max_size=2, server_settings={"search_path": schema})
    assert pool is not None
    await ensure_audit_events_table(pool)
    await ensure_audit_events_table(pool)  # idempotent
    try:
        yield pool
    finally:
        await pool.close()


def _event(**overrides: Any) -> AuditEvent:
    fields: dict[str, Any] = {
        "id": uuid.uuid7(),
        "timestamp": datetime.now(UTC),
        "event_type": "admin.user.create",
        "actor_user_id": uuid.uuid4(),
        "customer_id": uuid.uuid4(),
        "action": "user.create",
        "correlation_id": uuid.uuid4(),
        "details": {"user_id": str(uuid.uuid4()), "email": "a@b.test"},
    }
    fields.update(overrides)
    return AuditEvent(**fields)


async def test_an_event_round_trips_with_every_envelope_field(db: asyncpg.Pool) -> None:
    event = _event(acting_as_principal_id=uuid.uuid4(), conversation_id=uuid.uuid4())
    await persist_audit_event(db, event)
    row = await db.fetchrow("SELECT *, details::text AS details_text FROM audit_events WHERE id = $1", event.id)
    assert row is not None
    assert row["acting_as_principal_id"] == event.acting_as_principal_id
    assert row["conversation_id"] == event.conversation_id
    assert json.loads(row["details_text"]) == event.details, "details read back as a JSON object"


async def test_redelivery_collapses_to_one_row_by_id_and_by_correlation(db: asyncpg.Pool) -> None:
    event = _event()
    await persist_audit_event(db, event)
    await persist_audit_event(db, event)
    reemitted = event.model_copy(update={"id": uuid.uuid7()})  # same logical event, new envelope id
    await persist_audit_event(db, reemitted)
    assert await db.fetchval("SELECT count(*) FROM audit_events") == 1


async def test_prune_removes_only_rows_older_than_the_window(db: asyncpg.Pool) -> None:
    now = datetime.now(UTC)
    old = _event(timestamp=now - timedelta(days=40))
    fresh = _event(timestamp=now - timedelta(days=1))
    await persist_audit_event(db, old)
    await persist_audit_event(db, fresh)
    assert await prune_audit_events(db, older_than=timedelta(days=30), now=now) == 1
    assert [r["id"] for r in await db.fetch("SELECT id FROM audit_events")] == [fresh.id]


def _anonymize(details: Mapping[str, Any], *, event_type: str) -> dict[str, Any]:
    """A stand-in with the 0.55.0 contract's shape: every key kept, unsafe values masked."""
    safe = {"user_id"}
    return {k: (v if k in safe or v is None else _MARKER) for k, v in details.items()}


async def test_erasure_rewrites_details_and_ip_but_keeps_every_row_and_id(db: asyncpg.Pool) -> None:
    actor = uuid.uuid4()
    mine = [_event(actor_user_id=actor) for _ in range(3)]
    theirs = _event()
    for event in [*mine, theirs]:
        await persist_audit_event(db, event, ip_address="203.0.113.7")
    before = {r["id"]: dict(r) for r in await db.fetch("SELECT * FROM audit_events")}

    result = await anonymize_audit_rows(
        db, actor_user_ids=[actor], batch_size=2, anonymize=_anonymize, anonymize_ip=lambda _ip: None, marker=_MARKER
    )
    assert (result.rows_matched, result.rows_changed) == (3, 3)

    after = {r["id"]: r for r in await db.fetch("SELECT *, details::text AS details_text FROM audit_events")}
    assert set(after) == set(before), "erasure never deletes a row"
    for event in mine:
        row = after[event.id]
        assert json.loads(row["details_text"]) == {"user_id": event.details["user_id"], "email": _MARKER}
        assert row["ip_address"] is None
        assert (row["actor_user_id"], row["correlation_id"], row["event_type"]) == (
            actor,
            event.correlation_id,
            event.event_type,
        ), "ids and event fields never change"
    assert after[theirs.id]["ip_address"] == "203.0.113.7", "another actor's rows are untouched"

    again = await anonymize_audit_rows(
        db, actor_user_ids=[actor], anonymize=_anonymize, anonymize_ip=lambda _ip: None, marker=_MARKER
    )
    assert (again.rows_matched, again.rows_changed) == (3, 0), "idempotent"


async def test_erasure_handles_every_stored_details_shape(db: asyncpg.Pool) -> None:
    actor = uuid.uuid4()
    rows = {
        "string-held object": json.dumps(json.dumps({"email": "x@y.test"})),
        "non-object": json.dumps(["x@y.test"]),
        "json null": "null",
    }
    ids: dict[str, uuid.UUID] = {}
    for label, raw in rows.items():
        event = _event(actor_user_id=actor)
        await persist_audit_event(db, event)
        await db.execute("UPDATE audit_events SET details = $1::text::jsonb WHERE id = $2", raw, event.id)
        ids[label] = event.id
    await anonymize_audit_rows(
        db, actor_user_ids=[actor], anonymize=_anonymize, anonymize_ip=lambda _ip: None, marker=_MARKER
    )
    stored = {r["id"]: json.loads(r["d"]) for r in await db.fetch("SELECT id, details::text AS d FROM audit_events")}
    assert json.loads(stored[ids["string-held object"]]) == {"email": _MARKER}, "judged as the object, kept string-held"
    assert stored[ids["non-object"]] == _MARKER
    assert stored[ids["json null"]] is None
