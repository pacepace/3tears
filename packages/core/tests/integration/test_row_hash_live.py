"""Integration: a row's hash (:func:`row_hash_sql`) is the same text on PostgreSQL (or YugabyteDB) and DuckDB for the
same values, and :func:`read_changed` on a live database leaves a follower holding what a full read gives.

Every kind reports hashes (text, integer, numeric, boolean, date, timestamp), with NULLs, empty and unicode text,
negative and zero numerics, and fractional seconds, is written to a table on each database and hashed there; the
hashes must match row for row, so a hash stored from one is compared with the other's and never recomputed. On the
session's ``db_container`` (PostgreSQL), or YugabyteDB through ``THREETEARS_TEST_POSTGRES_URL``; a schema of its own.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import asyncpg
import duckdb
import pytest

from threetears.core.keyed import (
    ROW_HASH_COLUMN,
    Dialect,
    HashColumn,
    HashKind,
    KeyBatch,
    Sort,
    dialect_of,
    read_changed,
    read_keyed,
    row_hash_selected,
    row_hash_sql,
)
from threetears.core.testing.same_results import Seeded, assert_same_results

pytestmark = pytest.mark.integration

_HASHED = (
    HashColumn("name"),
    HashColumn("votes", HashKind.INTEGER),
    HashColumn("share", HashKind.NUMERIC, scale=4),
    HashColumn("called", HashKind.BOOLEAN),
    HashColumn("day", HashKind.DATE),
    HashColumn("stamp", HashKind.TIMESTAMP),
)

_ROWS: Sequence[tuple[Any, ...]] = (
    (1, "Smith", 1200, Decimal("0.5"), True, date(2024, 11, 5), datetime(2024, 11, 5, 20, 0, 0)),
    (2, "", 0, Decimal("0"), False, date(1, 1, 2), datetime(2024, 11, 5, 20, 0, 0, 500000)),
    (3, None, None, None, None, None, None),
    (4, "Peña ✓ 選挙", -7, Decimal("-0.0005"), True, date(2024, 2, 29), datetime(1999, 12, 31, 23, 59, 59, 999999)),
    (5, "~", 9_007_199_254_740_993, Decimal("12345.6789"), None, date(2024, 11, 5), datetime(2024, 1, 1, 0, 0, 0, 1)),
    (6, "a:b", 1, Decimal("1.2300"), False, None, datetime(2024, 11, 5, 9, 5, 3, 120000)),
)

_DDL = (
    "CREATE TABLE results (id BIGINT PRIMARY KEY, name VARCHAR, votes BIGINT, share DECIMAL(12, 4), "
    "called BOOLEAN, day DATE, stamp TIMESTAMP)"
)


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    schema = f"rowhash_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(db_container)
    try:
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        await connection.execute(f'SET search_path TO "{schema}"')
        await connection.execute(_DDL)
        await connection.executemany("INSERT INTO results VALUES ($1, $2, $3, $4, $5, $6, $7)", list(_ROWS))
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


async def test_a_row_hashes_alike_on_the_database_and_on_duckdb(conn: asyncpg.Connection) -> None:
    """Flips if any kind's canonical text, a NULL's mark or the md5 differs between this database and DuckDB for
    the same values: a hash stored from one would read as changed on the other."""
    dialect = await dialect_of(conn)
    assert dialect in {Dialect.POSTGRES, Dialect.YUGABYTE}
    live = {
        row["id"]: row["h"]
        for row in await conn.fetch(f"SELECT id, {row_hash_sql(_HASHED, dialect)} AS h FROM results AS t")
    }
    duck = duckdb.connect()
    duck.execute(_DDL)
    duck.executemany("INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?)", [list(row) for row in _ROWS])
    local = dict(duck.execute(f"SELECT id, {row_hash_sql(_HASHED, Dialect.DUCKDB)} AS h FROM results AS t").fetchall())
    duck.close()
    assert live == local
    assert len(set(live.values())) == len(_ROWS)


async def test_only_changed_rows_are_read_again_on_the_database(conn: asyncpg.Connection) -> None:
    """Flips if read_changed on a live database reads a row whose stored hash is current, misses a changed or new
    row, or misses a gone key."""
    hashed = row_hash_selected(_HASHED, await dialect_of(conn))
    stored = await read_keyed(
        conn,
        "results",
        keys=KeyBatch(("id",), [(row[0],) for row in _ROWS], casts=("bigint",)),
        columns=["id", hashed],
    )
    held: dict[tuple[Any, ...], str | None] = {(row["id"],): row[ROW_HASH_COLUMN] for row in stored}
    held[(2,)] = None
    await conn.execute("UPDATE results SET votes = votes + 1 WHERE id = 1")
    await conn.execute("UPDATE results SET name = 'x' WHERE id = 3")
    await conn.execute("DELETE FROM results WHERE id = 6")
    await conn.execute("INSERT INTO results (id) VALUES (7)")
    changed = await read_changed(
        conn,
        "results",
        key=[Sort("id", cast="bigint")],
        columns=["id", "name", "votes"],
        held=held,
        hash_columns=_HASHED,
        page=2,
    )
    assert sorted(row["id"] for row in changed.rows) == [1, 2, 3, 7]
    assert changed.gone_keys == [(6,)]


async def test_the_rows_after_applying_the_changes_are_a_full_read_s_on_the_database(
    conn: asyncpg.Connection,
) -> None:
    """Flips if a follower's copy after applying read_changed differs from a plain full read of the scope, over the
    harness's seeds and both tenants, on a live database."""
    await conn.execute(
        "CREATE TABLE seeded (tenant TEXT, id BIGINT, name TEXT, votes BIGINT, PRIMARY KEY (tenant, id))"
    )
    hashed = (HashColumn("name"), HashColumn("votes", HashKind.INTEGER))
    columns = ["tenant", "id", "name", "votes"]

    def row(seed: Seeded) -> dict[str, Any]:
        return {
            "tenant": f"t{seed.tenant}",
            "id": seed.index,
            "name": None if seed.null else f"n{seed.tie}",
            "votes": seed.tie,
        }

    async def follow(db: Any) -> Sequence[Mapping[str, Any]]:
        copy = await read_keyed(
            db,
            "seeded",
            keys=KeyBatch(columns=("tenant",), values=[("t0",)]),
            columns=[*columns, row_hash_selected(hashed, await dialect_of(db))],
        )
        held: dict[tuple[Any, ...], str | None] = {
            ("t0", found["id"]): (None if found["id"] % 3 == 0 else found[ROW_HASH_COLUMN]) for found in copy
        }
        held[("t0", 10_000)] = "f" * 32
        rows = {("t0", found["id"]): found for found in copy}
        rows[("t0", 10_000)] = {"tenant": "t0", "id": 10_000, "name": "gone", "votes": 0}
        changed = await read_changed(
            db,
            "seeded",
            key=["tenant", Sort("id", cast="bigint")],
            columns=columns,
            held=held,
            hash_columns=hashed,
            by={"tenant": "t0"},
            page=5,
        )
        for gone in changed.gone_keys:
            del rows[gone]
        rows.update({("t0", found["id"]): found for found in changed.rows})
        return list(rows.values())

    await assert_same_results(
        conn,
        table="seeded",
        row=row,
        old_sql="SELECT tenant, id, name, votes FROM seeded WHERE tenant = $1",
        params=["t0"],
        helper=follow,
        columns=columns,
        ordered=False,
    )


async def test_a_typed_key_held_as_text_reads_only_its_changed_row_on_the_database(conn: asyncpg.Connection) -> None:
    """Flips if a key held as a follower's text (an integer id's digits, a zoned time) and read by a warehouse that
    holds it typed (integer, timestamp without time zone) reads unchanged rows again, names held keys gone, or binds
    the typed key as text (Postgres refuses integer = text)."""
    await conn.execute(
        "CREATE TABLE typed (state VARCHAR, cid INTEGER, stamp TIMESTAMP WITHOUT TIME ZONE, name VARCHAR, "
        "PRIMARY KEY (state, cid, stamp))"
    )
    await conn.execute(
        "INSERT INTO typed SELECT 'PA', 7000 + i, TIMESTAMP '2026-11-03 22:30:00' + make_interval(secs => i), "
        "'n' || i FROM generate_series(0, 5) i"
    )
    hashed = (HashColumn("cid", HashKind.INTEGER), HashColumn("stamp", HashKind.TIMESTAMP), HashColumn("name"))
    sql = f"SELECT state, cid, stamp, {row_hash_sql(hashed, await dialect_of(conn))} AS h FROM typed AS t"
    held: dict[tuple[Any, ...], str | None] = {
        (row["state"], str(row["cid"]), f"{row['stamp'].isoformat(sep=' ')}+00:00"): row["h"]
        for row in await conn.fetch(sql)
    }

    async def changed() -> Any:
        return await read_changed(
            conn,
            "typed",
            key=["state", "cid", "stamp"],
            columns=["state", "cid", "stamp", "name"],
            held=held,
            hash_columns=hashed,
            by={"state": "PA"},
            page=2,
        )

    unchanged = await changed()
    assert (unchanged.rows, unchanged.gone_keys) == ([], [])
    await conn.execute("UPDATE typed SET name = 'moved' WHERE cid = 7002")
    await conn.execute("DELETE FROM typed WHERE cid = 7004")
    moved = await changed()
    assert [(row["cid"], row["name"]) for row in moved.rows] == [(7002, "moved")]
    assert moved.gone_keys == [("PA", "7004", "2026-11-03 22:30:04+00:00")]
