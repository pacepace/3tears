"""Integration: a changed row under a NULL key part is read in full by read_changed, through its key's index.

Phase 2 named the wanted keys by equality, so a key with a NULL part matched no row: it was listed changed and never
read. It now reads such keys a NULL pattern at a time, each NULL part pinned ``IS NULL`` beside the batch of the
rest. Here, on PostgreSQL (the session's ``db_container``) or YugabyteDB (``THREETEARS_TEST_POSTGRES_URL``), for the
held keys given as a mapping and as L3's Arrow table: every changed row is read in full, and each phase-2 statement
holding an ``IS NULL`` pin reads the ``(state, county, id)`` index by an ``Index Cond`` on ``county``, no table read whole.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any

import asyncpg
import pyarrow as pa  # type: ignore[import-untyped]
import pytest

from threetears.core.keyed import ROW_HASH_COLUMN, HashColumn, Sort, dialect_of, read_changed, row_hash_sql
from threetears.core.testing.plans import PlanRecorder

pytestmark = pytest.mark.integration

_HASHED = (HashColumn("name"),)
_ROWS = 2000


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    """``nk``: PA's rows by (state, county, id), every tenth county NULL, its key index analyzed."""
    schema = f"nullkey_{uuid.uuid4().hex[:10]}"
    connection = await asyncpg.connect(db_container)
    try:
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        await connection.execute(f'SET search_path TO "{schema}"')
        await connection.execute("CREATE TABLE nk (state TEXT NOT NULL, county TEXT, id BIGINT NOT NULL, name TEXT)")
        await connection.execute("CREATE INDEX nk_key ON nk (state ASC, county ASC, id ASC)")
        await connection.execute(
            "INSERT INTO nk SELECT s, CASE WHEN i % 10 = 0 THEN NULL ELSE 'c' || (i % 37) END, i, 'n' || i "
            "FROM generate_series(1, $1) i, unnest(ARRAY['PA', 'OH']) s",
            _ROWS,
        )
        await connection.execute("ANALYZE nk")
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


async def _held(conn: asyncpg.Connection) -> dict[tuple[Any, ...], str | None]:
    hashed = row_hash_sql(_HASHED, await dialect_of(conn))
    rows = await conn.fetch(f"SELECT state, county, id, {hashed} AS h FROM nk AS t WHERE state = 'PA'")  # noqa: S608
    return {(row["state"], row["county"], row["id"]): row["h"] for row in rows}


def _as_table(held: Mapping[tuple[Any, ...], str | None]) -> pa.Table:
    keys = list(held)
    return pa.table(
        {
            "state": pa.array([key[0] for key in keys], pa.string()),
            "county": pa.array([key[1] for key in keys], pa.string()),
            "id": pa.array([key[2] for key in keys], pa.int64()),
            ROW_HASH_COLUMN: pa.array(list(held.values()), pa.string()),
        }
    )


def _nodes(node: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    yield node
    for child in node.get("Plans", ()):
        yield from _nodes(child)


@pytest.mark.parametrize("form", ["mapping", "arrow"])
async def test_a_changed_row_under_a_null_key_part_is_read_in_full_through_the_key_index(
    conn: asyncpg.Connection, form: str
) -> None:
    """Flips if a changed row whose county is NULL is named changed but not read in full, or a phase-2 statement
    pinning ``county IS NULL`` reads anything but a keyed range of the key index."""
    held = await _held(conn)
    await conn.execute("UPDATE nk SET name = 'moved' WHERE state = 'PA' AND id IN (20, 21, 500)")
    recorder = PlanRecorder(conn)
    changed = await read_changed(
        recorder,
        "nk",
        key=["state", "county", Sort("id", cast="bigint")],
        columns=["state", "county", "id", "name"],
        held=held if form == "mapping" else _as_table(held),
        hash_columns=_HASHED,
        by={"state": "PA"},
    )
    assert sorted((row["county"], row["id"], row["name"]) for row in changed.rows) == [
        (None, 20, "moved"),
        (None, 500, "moved"),
        ("c21", 21, "moved"),
    ]
    assert changed.gone_keys == []
    pinned = [plan for plan in recorder.plans if '"county" IS NULL' in plan.sql]
    assert pinned, [plan.sql for plan in recorder.plans]
    for plan in pinned:
        nodes = list(_nodes(plan.plan))
        assert not [node for node in nodes if node["Node Type"] == "Seq Scan"], json.dumps(plan.plan)[:3000]
        reads = [node for node in nodes if "Index Name" in node]
        assert reads and all(read["Index Name"] == "nk_key" for read in reads), json.dumps(plan.plan)[:3000]
        assert any("county" in str(read.get("Index Cond", "")) for read in reads), json.dumps(plan.plan)[:3000]
