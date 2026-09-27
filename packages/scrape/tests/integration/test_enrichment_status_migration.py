"""Real-Postgres check of v013's translation of pre-0.55.0 enrichment rows.

Before 0.55.0 a failed enrichment pass stored ``enrichment_notes = {}``, the same value as a
pass whose model had nothing to add. v013 adds ``enrichment_status`` / ``enrichment_failure``
and translates existing rows once. The translation is SQL over JSONB, which no offline test
can execute, so it is proved here against a real database.

Guarded by ``@pytest.mark.integration`` like the rest of this directory.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import asyncpg
import pytest
from threetears.core.data.migrations import ConnectionSession

from threetears.scrape.migrations import (
    LEGACY_EMPTY_ENRICHMENT_FAILURE,
    apply_migrations,
    v013_extraction_enrichment_status,
)

pytestmark = pytest.mark.integration


@pytest.fixture
async def pg_pool(db_container: str) -> AsyncIterator[asyncpg.Pool]:
    """A plain asyncpg pool with every 3tears-scrape migration applied."""
    pool: asyncpg.Pool = await asyncpg.create_pool(db_container, min_size=1, max_size=2)
    try:
        await apply_migrations(pool)
        yield pool
    finally:
        await pool.close()


async def _insert_legacy(conn: asyncpg.Connection, target_id: str, notes_sql: str) -> str:
    """Write a row as a pre-v013 writer left it: notes set, no status, no failure."""
    row_id = uuid.uuid4().hex
    await conn.execute(
        f"INSERT INTO scrape_extractions (id, target_id, enrichment_notes) VALUES ($1, $2, {notes_sql})",
        row_id,
        target_id,
    )
    return row_id


async def test_legacy_rows_are_translated_once_and_replay_changes_nothing(pg_pool: asyncpg.Pool) -> None:
    target_id = f"warn_legacy_{uuid.uuid4().hex[:8]}"
    async with pg_pool.acquire() as conn:
        never_ran = await _insert_legacy(conn, target_id, "NULL")
        had_notes = await _insert_legacy(conn, target_id, """'{"context": "q3"}'::jsonb""")
        empty = await _insert_legacy(conn, target_id, "'{}'::jsonb")
        empty_double_encoded = await _insert_legacy(conn, target_id, """'"{}"'::jsonb""")

        await v013_extraction_enrichment_status(ConnectionSession(conn))  # type: ignore[arg-type]
        first = {
            r["id"]: dict(r)
            for r in await conn.fetch(
                "SELECT id, enrichment_notes, enrichment_status, enrichment_failure "
                "FROM scrape_extractions WHERE target_id = $1",
                target_id,
            )
        }

        await v013_extraction_enrichment_status(ConnectionSession(conn))  # type: ignore[arg-type]
        second = {
            r["id"]: dict(r)
            for r in await conn.fetch(
                "SELECT id, enrichment_notes, enrichment_status, enrichment_failure "
                "FROM scrape_extractions WHERE target_id = $1",
                target_id,
            )
        }

    assert first[never_ran]["enrichment_status"] is None
    assert first[never_ran]["enrichment_notes"] is None
    assert first[never_ran]["enrichment_failure"] is None

    assert first[had_notes]["enrichment_status"] == "enriched"
    assert first[had_notes]["enrichment_notes"] is not None
    assert first[had_notes]["enrichment_failure"] is None

    for ambiguous in (empty, empty_double_encoded):
        assert first[ambiguous]["enrichment_status"] == "failed"
        assert first[ambiguous]["enrichment_notes"] is None
        assert first[ambiguous]["enrichment_failure"] == LEGACY_EMPTY_ENRICHMENT_FAILURE

    assert second == first
