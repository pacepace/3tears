"""the audit writes moved onto the keyed helpers leave the table the statements they replaced left.

``prune_audit_events`` once sent ``DELETE ... WHERE id IN (SELECT id ... WHERE timestamp < $1 LIMIT $2)`` until a
short page; it now pages through ``idx_audit_events_time_asc`` with :func:`~threetears.core.keyed.delete_before`.
``anonymize_audit_rows`` once wrote each row with a hand-written ``UPDATE ... WHERE id = $3 AND (... IS DISTINCT
FROM ...)``; it now sends :func:`~threetears.core.keyed.update_keyed` with that guard. Each is compared with what the
hand-written statements (as they stood at b628f0a6b) left: the table after, every column, and the counts reported.

Runs on PostgreSQL and YugabyteDB (the session's ``db_container``), a schema of its own.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest

from threetears.agent.audit import ANONYMIZED_MARKER, persist
from threetears.agent.audit.persist import anonymize_audit_rows, ensure_audit_events_table, prune_audit_events
from threetears.core.keyed import AnyOf, Fragment, Selected, Sort, keyset_page
from threetears.core.testing.same_results import Seeded, assert_same_results

pytestmark = pytest.mark.integration

NS = uuid.UUID("7a0f8c1e-3d2b-4e6a-9c51-1b2d3e4f5a60")
ACTORS = (uuid.uuid5(NS, "actor-0"), uuid.uuid5(NS, "actor-1"))
T0 = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)
PAGE = 6
_STATE = "SELECT id, timestamp, event_type, details::text AS details, ip_address, actor_user_id FROM audit_events"


@pytest.fixture
async def db(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    schema = f"audit_same_{uuid.uuid4().hex[:10]}"
    conn = await asyncpg.connect(db_container)
    await conn.execute(f"CREATE SCHEMA {schema}")
    await conn.execute(f"SET search_path TO {schema}")
    await ensure_audit_events_table(conn)
    try:
        yield conn
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


def _row(seed: Seeded) -> Mapping[str, Any]:
    """an event of its tenant's actor at its tie, its details naming a person on even rows and already erased on odd
    ones, its address NULL for a NULL seed."""
    details = {"email": "a@b.test", "note": "kept"} if seed.index % 2 == 0 else {"email": ANONYMIZED_MARKER}
    return {
        "id": uuid.uuid5(NS, f"event-{seed.index}"),
        "timestamp": T0 + timedelta(minutes=seed.tie),
        "event_type": "admin.user.create",
        "action": "user.create",
        "correlation_id": uuid.uuid5(NS, "correlation"),
        "details": json.dumps(details),
        "ip_address": None if seed.null else "203.0.113.7",
        "actor_user_id": ACTORS[seed.tenant],
    }


async def test_prune_leaves_the_hand_written_table(db: asyncpg.Connection) -> None:
    """Flips if the paged prune deletes a row the hand-written loop kept (one at the cutoff or after) or keeps one it
    deleted, across pages."""
    cutoff = T0 + timedelta(minutes=PAGE)
    deleted: list[int] = []

    async def helper(c: Any) -> Any:
        deleted.append(await prune_audit_events(c, older_than=timedelta(0), now=cutoff, batch_size=PAGE - 1))
        return []

    # the hand-written loop's statement, its page wide enough to be its whole loop in one statement
    await assert_same_results(
        db,
        table="audit_events",
        row=_row,
        old_sql="DELETE FROM audit_events WHERE id IN (SELECT id FROM audit_events WHERE timestamp < $1 LIMIT $2)",
        params=[cutoff, 10_000],
        helper=helper,
        page_size=PAGE,
        state_sql=_STATE,
    )
    assert any(deleted), deleted


async def _anonymize_by_hand(db: Any, actors: list[uuid.UUID], batch_size: int) -> tuple[int, int]:
    """``anonymize_audit_rows`` as it was at b628f0a6b: the same page read, the hand-written UPDATE per row."""
    matched = changed = 0
    after: uuid.UUID | None = None
    while True:
        rows = (
            await keyset_page(
                db,
                "audit_events",
                columns=("id", "event_type", Selected("details::text", "details"), "ip_address"),
                by={"actor_user_id": AnyOf(actors, cast="uuid")},
                where=[Fragment("($1::uuid IS NULL OR id > $1)", (after,))],
                order=[Sort("id")],
                limit=batch_size,
                records=True,
            )
        ).rows
        if not rows:
            break
        for row in rows:
            matched += 1
            details = persist._anonymize_stored(row["details"], event_type=row["event_type"])  # noqa: SLF001 - the rule the pass applies
            address = persist.anonymize_ip(row["ip_address"])
            status = await db.execute(
                "UPDATE audit_events SET details = $1::text::jsonb, ip_address = $2 WHERE id = $3 "
                "AND (details IS DISTINCT FROM $1::text::jsonb OR ip_address IS DISTINCT FROM $2)",
                details,
                address,
                row["id"],
            )
            tail = str(status).rsplit(" ", 1)[-1]
            changed += int(tail) if tail.isdigit() else 0
        after = rows[-1]["id"]
    return matched, changed


async def _run_seeded(db: asyncpg.Connection, rows: list[Mapping[str, Any]], run: Any) -> tuple[Any, list[Any]]:
    """seed ``rows`` in a transaction, run ``run`` twice (the second pass finds nothing to change), read the table,
    roll back: the two passes' counts and the table after."""
    transaction = db.transaction()
    await transaction.start()
    try:
        await db.execute("DELETE FROM audit_events")
        for row in rows:
            await db.execute(
                "INSERT INTO audit_events (id, timestamp, event_type, action, correlation_id, details, ip_address,"
                " actor_user_id) VALUES ($1, $2, $3, $4, $5, $6::text::jsonb, $7, $8)",
                *row.values(),
            )
        counts = (await run(db), await run(db))
        state = sorted(tuple(r) for r in await db.fetch(_STATE))
    finally:
        await transaction.rollback()
    return counts, state


@pytest.mark.parametrize("rows", [0, 1, PAGE - 1, PAGE, 3 * PAGE + 1])
async def test_anonymize_leaves_the_hand_written_table_and_counts(db: asyncpg.Connection, rows: int) -> None:
    """Flips if the keyed update writes a row the hand-written UPDATE did not (an already-erased one, another actor's),
    misses one, writes other values, or counts rows matched or changed otherwise (on a first pass and on a second
    that changes nothing), across page boundaries."""
    seeded = [
        _row(Seeded(index=i, case="anonymize", tenant=int(i % 3 == 2), null=i % 4 == 3, tie=i)) for i in range(rows)
    ]

    async def old(c: Any) -> Any:
        return await _anonymize_by_hand(c, [ACTORS[0]], PAGE - 1)

    async def new(c: Any) -> Any:
        done = await anonymize_audit_rows(c, actor_user_ids=[ACTORS[0]], batch_size=PAGE - 1)
        return (done.rows_matched, done.rows_changed)

    assert await _run_seeded(db, seeded, new) == await _run_seeded(db, seeded, old)
