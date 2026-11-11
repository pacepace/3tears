"""Integration: a row's hash (:func:`row_hash_sql`) is the same text on PostgreSQL (or YugabyteDB) and DuckDB for the
same values, and :func:`read_changed` on a live database leaves a follower holding what a full read gives.

Every kind reports hashes (text, integer, numeric, boolean, date, timestamp), with NULLs, empty and unicode text,
negative and zero numerics, and fractional seconds, is written to a table on each database and hashed there; the
hashes must match row for row, so a hash stored from one is compared with the other's and never recomputed. On the
session's ``db_container`` (PostgreSQL), or YugabyteDB through ``THREETEARS_TEST_POSTGRES_URL``; a schema of its own.
"""

from __future__ import annotations

import base64
import hashlib
import json
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
    group_hashes,
    read_changed,
    read_keyed,
    row_hash_selected,
    row_hash_sql,
)
from threetears.core.testing.same_results import Seeded, assert_same_results
from threetears.core.keyed import InlineDuckDB
from threetears.interfaces.keyed import HashEncoding

from packages.core.tests.unit.test_keyed_changed import LANE_EDGES, grouped_same_results, lane_packings

pytestmark = pytest.mark.integration


class _RowCapped:
    """the test connection behind a transport that states its row cap (``rows_per_statement``), as a rail does."""

    binds_native_arrays = True

    def __init__(self, conn: asyncpg.Connection, dialect: Dialect, *, rows_per_statement: int) -> None:
        self._conn = conn
        self.sql_dialect = dialect
        self.rows_per_statement = rows_per_statement

    async def fetch(self, query: str, *params: Any) -> list[Any]:
        return list(await self._conn.fetch(query, *params))


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


class _Kept:
    """a connection that keeps every reply it hands on (``group_hashes``' phase 0 rows, as the wire carries them)."""

    binds_native_arrays = True

    def __init__(self, conn: Any, dialect: Dialect) -> None:
        self._conn = conn
        self.sql_dialect = dialect
        self.replies: list[dict[str, Any]] = []

    async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
        rows = [dict(row) for row in await self._conn.fetch(query, *params)]
        self.replies.extend(rows)
        return rows


async def test_a_hash_is_the_same_unpadded_base64_on_the_database_duckdb_python_and_in_redshift_s_spelling(
    conn: asyncpg.Connection,
) -> None:
    """Flips if a row's hash or a group's hash, as a read answers it, is not byte-identical on this database, on
    DuckDB and from Python's md5 of the same canonical text (22 characters of unpadded base64, the md5's 128 bits),
    or Redshift's is spelled other than the form checked live there against Python
    (``RTRIM(FROM_VARBYTE(FROM_HEX(md5(x)), 'base64'), '=')``; ``TO_VARCHAR`` refuses): a hash stored from one
    engine would read as changed on another."""
    dialect = await dialect_of(conn)
    text = row_hash_sql(_HASHED, dialect, encoding=HashEncoding.HEX).removeprefix("md5(")[:-1]
    live = {
        row["id"]: (row["text"], row["h"])
        for row in await conn.fetch(
            f"SELECT id, {text} AS text, {row_hash_sql(_HASHED, dialect)} AS h FROM results AS t"
        )
    }
    duck = duckdb.connect()
    try:
        duck.execute(_DDL)
        duck.executemany("INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?)", [list(row) for row in _ROWS])
        local = dict(
            duck.execute(f"SELECT id, {row_hash_sql(_HASHED, Dialect.DUCKDB)} AS h FROM results AS t").fetchall()
        )
        for found, (canonical, hashed) in live.items():
            spelled = base64.b64encode(hashlib.md5(canonical.encode()).digest()).decode().rstrip("=")  # noqa: S324
            assert hashed == local[found] == spelled and len(hashed) == 22, (found, hashed, local[found], spelled)
        hexed = row_hash_sql(_HASHED, Dialect.REDSHIFT, encoding=HashEncoding.HEX)
        assert row_hash_sql(_HASHED, Dialect.REDSHIFT) == f"RTRIM(FROM_VARBYTE(FROM_HEX({hexed}), 'base64'), '=')"

        # a group's hash, as phase 0 answers it, over scopes and counties (a NULL county among them)
        grouped = "CREATE TABLE scoped_b64 (state VARCHAR, county VARCHAR, id BIGINT, name VARCHAR, votes BIGINT)"
        await conn.execute(grouped)
        duck.execute(grouped)
        rows = [
            (state, None if i % 4 == 0 else f"c{i % 4}", i, None if i % 5 == 0 else f"n{i} ✓", i * 7_000_003)
            for i in range(40)
            for state in ("OH", "PA")
        ]
        await conn.executemany("INSERT INTO scoped_b64 VALUES ($1, $2, $3, $4, $5)", rows)
        duck.executemany("INSERT INTO scoped_b64 VALUES (?, ?, ?, ?, ?)", rows)
        hashed_columns = (HashColumn("name"), HashColumn("votes", HashKind.INTEGER))
        answered = []
        for reader in (_Kept(conn, dialect), _Kept(InlineDuckDB(duck), Dialect.DUCKDB)):
            hashes = await group_hashes(
                reader, "scoped_b64", scope="state", scopes=["OH", "PA"], group="county", hash_columns=hashed_columns
            )
            answered.append(
                (
                    hashes,
                    sorted((row["scope_value"], str(row["group_value"]), row["group_hash"]) for row in reader.replies),
                )
            )
        (on_db, db_replies), (on_duck, duck_replies) = answered
        assert on_db == on_duck and db_replies == duck_replies and len(db_replies) == 8
        assert all(len(group_hash) == 22 for _, _, group_hash in db_replies)
    finally:
        duck.close()


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


async def test_a_group_hash_is_the_same_from_stored_hashes_and_from_sql_on_the_database(
    conn: asyncpg.Connection,
) -> None:
    """Flips if the group hash this database computes (phase 0) differs from the one computed from the row hashes
    stored from it: every unchanged group would read as moved, phase 1 and more than one statement."""
    dialect = await dialect_of(conn)
    await conn.execute(
        "CREATE TABLE grouped (state TEXT, county TEXT, id BIGINT, name TEXT, votes BIGINT, "
        "PRIMARY KEY (state, county, id))"
    )
    await conn.execute(
        "INSERT INTO grouped SELECT 'PA', 'c' || (i / 10), i, CASE WHEN i % 7 = 0 THEN NULL ELSE 'n' || i END, "
        "i * 7 FROM generate_series(0, 59) i"
    )
    hashed = (HashColumn("name"), HashColumn("votes", HashKind.INTEGER))
    held: dict[tuple[Any, ...], str | None] = {
        (row["state"], row["county"], row["id"]): row["h"]
        for row in await conn.fetch(f"SELECT state, county, id, {row_hash_sql(hashed, dialect)} AS h FROM grouped AS t")
    }

    class _Counted:
        binds_native_arrays = True
        sql_dialect = dialect

        def __init__(self) -> None:
            self.statements = 0

        async def fetch(self, query: str, *params: Any) -> Any:
            self.statements += 1
            return await conn.fetch(query, *params)

    counted = _Counted()
    changed = await read_changed(
        counted,
        "grouped",
        key=["state", "county", Sort("id", cast="bigint")],
        columns=["state", "county", "id", "name", "votes"],
        held=held,
        hash_columns=hashed,
        by={"state": "PA"},
        group="county",
    )
    assert (changed.rows, changed.gone_keys, counted.statements) == ([], [], 1)
    await conn.execute("UPDATE grouped SET votes = -1 WHERE id = 33")
    moved = await read_changed(
        counted,
        "grouped",
        key=["state", "county", Sort("id", cast="bigint")],
        columns=["state", "county", "id", "name", "votes"],
        held=held,
        hash_columns=hashed,
        by={"state": "PA"},
        group="county",
    )
    assert [row["id"] for row in moved.rows] == [33]


async def test_the_rows_after_applying_a_grouped_read_are_a_full_read_s_on_the_database(
    conn: asyncpg.Connection,
) -> None:
    """Flips if a follower's copy after read_changed(group=...) (phases 0, 1 and 2) differs from a plain full read of
    the scope on a live database: a group with one row changed, one with a row added, one with a row removed, and a
    held group gone."""
    await conn.execute(
        "CREATE TABLE seeded_groups (tenant TEXT, grp BIGINT, id BIGINT, name TEXT, votes BIGINT, "
        "PRIMARY KEY (tenant, grp, id))"
    )
    await assert_same_results(conn, **grouped_same_results(await dialect_of(conn)))


async def test_every_scope_s_group_hashes_in_one_statement_match_each_scope_s_own_phase_zero_on_the_database(
    conn: asyncpg.Connection,
) -> None:
    """Flips if group_hashes on this database (one statement, paged by scope then group across a NULL group) answers
    a scope's groups other than read_changed's own phase 0 for that scope: a state handed its slice would read an
    unchanged group as moved or miss a moved one."""
    dialect = await dialect_of(conn)
    await conn.execute("CREATE TABLE scoped (state TEXT, county TEXT, id BIGINT, name TEXT, votes BIGINT)")
    await conn.execute(
        "INSERT INTO scoped SELECT s, CASE WHEN i % 4 = 0 THEN NULL ELSE 'c' || (i % 4) END, i, 'n' || i, i "
        "FROM generate_series(0, 11) i, (VALUES ('OH'), ('PA'), ('WV')) v(s)"
    )
    hashed = (HashColumn("name"), HashColumn("votes", HashKind.INTEGER))
    held: dict[tuple[Any, ...], str | None] = {
        (row["state"], row["county"], row["id"]): row["h"]
        for row in await conn.fetch(f"SELECT state, county, id, {row_hash_sql(hashed, dialect)} AS h FROM scoped AS t")
    }
    # two rows of two counties (a key holding a NULL county is no key a phase-2 read can bind)
    await conn.execute("UPDATE scoped SET votes = -1 WHERE state = 'PA' AND id IN (5, 6)")
    sliced = await group_hashes(
        conn, "scoped", scope="state", scopes=["PA", "OH", "WV", "XX"], group="county", hash_columns=hashed
    )
    # a transport answering 3 rows a statement: 2 groups a page
    paged = await group_hashes(
        _RowCapped(conn, dialect, rows_per_statement=3),
        "scoped",
        scope="state",
        scopes=["PA", "OH", "WV", "XX"],
        group="county",
        hash_columns=hashed,
    )
    assert paged == sliced and sliced["XX"] == {}
    for state in ("PA", "OH", "WV"):
        scope_held = {key: value for key, value in held.items() if key[0] == state}
        options: dict[str, Any] = {
            "key": ["state", "county", Sort("id", cast="bigint")],
            "columns": ["state", "county", "id", "name", "votes"],
            "held": scope_held,
            "hash_columns": hashed,
            "by": {"state": state},
            "group": "county",
        }
        own = await read_changed(conn, "scoped", **options)
        handed = await read_changed(conn, "scoped", groups=sliced[state], **options)
        assert (handed.rows, handed.gone_keys, handed.changed_keys) == (own.rows, own.gone_keys, own.changed_keys)
        assert sorted(row["id"] for row in handed.rows) == ([5, 6] if state == "PA" else [])


async def test_phase_zero_hashes_each_row_once_on_the_database(conn: asyncpg.Connection) -> None:
    """Flips if the database's plan of phase 0 (group_hashes) computes a row's hash anywhere but once a row below the
    aggregate: an unfenced inner select is pulled up and the md5 inlined into each of the four lane sums, the scan
    handing the aggregate the hashed columns and every row hashed four times (a broad upload's phase 0 went from ~2 s
    to ~15-20 s on the warehouse stand-in). Fenced, the table's scan outputs the row's one md5 and none of the hashed
    columns, so nothing above it can hash again (the nodes above print only references to that md5)."""
    dialect = await dialect_of(conn)
    await conn.execute("CREATE TABLE scoped (state TEXT, county TEXT, id BIGINT, name TEXT, votes BIGINT)")
    await conn.execute(
        "INSERT INTO scoped SELECT s, 'c' || (i % 4), i, 'n' || i, i FROM generate_series(0, 11) i, "
        "(VALUES ('OH'), ('PA')) v(s)"
    )
    hashed = (HashColumn("name"), HashColumn("votes", HashKind.INTEGER))

    class _Explained:
        binds_native_arrays = True
        sql_dialect = dialect

        def __init__(self) -> None:
            self.plans: list[dict[str, Any]] = []

        async def fetch(self, query: str, *params: Any) -> Any:
            [explained] = await conn.fetch(f"EXPLAIN (VERBOSE, FORMAT JSON) {query}", *params)
            self.plans.append(json.loads(explained[0])[0]["Plan"])
            return await conn.fetch(query, *params)

    def _scans(node: dict[str, Any]) -> list[dict[str, Any]]:
        own = [node] if node.get("Relation Name") == "scoped" else []
        return own + [scan for child in node.get("Plans", ()) for scan in _scans(child)]

    explained = _Explained()
    answered = await group_hashes(
        explained, "scoped", scope="state", scopes=["OH", "PA"], group="county", hash_columns=hashed
    )
    assert sorted(answered["PA"], key=str) == ["c0", "c1", "c2", "c3"]
    [plan] = explained.plans
    [scan] = _scans(plan)
    outputs = list(scan["Output"])
    assert sum("md5(" in output for output in outputs) == 1, outputs
    assert not {"t.name", "t.votes", "name", "votes"} & set(outputs), outputs


async def test_a_lane_is_its_value_mod_2_32_as_8_hex_digits_at_every_edge_on_the_database(
    conn: asyncpg.Connection,
) -> None:
    """Flips if this database packs a lane value outside [0, 2**32) (negative, 2**32, past it) other than as
    ``value mod 2**32`` in 8 hex digits, as Python and DuckDB do: LPAD cuts from the right, so an unreduced value
    would keep its high digits only."""
    packings = lane_packings(await dialect_of(conn))
    assert len(packings) == 4 and len(set(packings)) == 1
    for value in LANE_EDGES:
        found = await conn.fetchval(f"SELECT {packings[0].format(f'CAST({value} AS BIGINT)')}")
        assert found.lower() == f"{value % 2**32:08x}", (value, found)
