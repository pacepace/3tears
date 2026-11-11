"""changed rows by hash (:func:`read_changed`, :func:`row_hash_sql`): phase 1 reads the scope's keys and hashes,
phase 2 reads by key only the rows whose hash differs from the held one; on an in-process DuckDB."""

from __future__ import annotations

import json

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import duckdb
import pytest

from threetears.core.keyed import (
    ROW_HASH_COLUMN,
    ChangedRows,
    Dialect,
    Fragment,
    HashColumn,
    HashKind,
    InlineDuckDB,
    Sort,
    read_changed,
    read_keyed,
    KeyBatch,
    dialect_of,
    key_text,
    row_hash_selected,
    row_hash_sql,
    run_inline,
)
from threetears.core.testing.same_results import Seeded, assert_same_results

_HASHED = (
    HashColumn("name"),
    HashColumn("votes", HashKind.INTEGER),
    HashColumn("share", HashKind.NUMERIC, scale=4),
    HashColumn("called", HashKind.BOOLEAN),
    HashColumn("day", HashKind.DATE),
    HashColumn("stamp", HashKind.TIMESTAMP),
)

_COLUMNS = ("state", "geo", "name", "votes", "share", "called", "day", "stamp")


@pytest.fixture
def conn() -> Iterator[duckdb.DuckDBPyConnection]:
    db = duckdb.connect()
    db.execute(
        "CREATE TABLE results (state VARCHAR, geo BIGINT, name VARCHAR, votes BIGINT, share DECIMAL(9, 4), "
        "called BOOLEAN, day DATE, stamp TIMESTAMP, PRIMARY KEY (state, geo))"
    )
    db.execute(
        "INSERT INTO results SELECT CASE WHEN i % 3 = 0 THEN 'PA' ELSE 'OH' END, i, 'n' || i, i * 10, i / 1000, "
        "i % 2 = 0, DATE '2024-11-05', TIMESTAMP '2024-11-05 20:00:00' + to_microseconds(i) FROM range(60) r(i)"
    )
    yield db
    db.close()


class _Counted:
    """an inline DuckDB that counts the statements it is sent."""

    binds_native_arrays = False
    sql_dialect = Dialect.DUCKDB

    def __init__(self, db: duckdb.DuckDBPyConnection) -> None:
        self._db = InlineDuckDB(db)
        self.statements: list[str] = []

        self.bytes = 0

    async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
        self.statements.append(query)
        rows = await self._db.fetch(query, *params)
        self.bytes += len(json.dumps(rows, default=str, separators=(",", ":")))
        return rows


def _held(db: duckdb.DuckDBPyConnection, state: str) -> dict[tuple[Any, ...], str]:
    sql = f"SELECT state, geo, {row_hash_sql(_HASHED, Dialect.DUCKDB)} AS h FROM results AS t WHERE state = $1"
    return {(row[0], row[1]): row[2] for row in db.execute(sql, [state]).fetchall()}


def _changed(db: Any, held: Mapping[tuple[Any, ...], str], state: str = "PA", **options: Any) -> ChangedRows:
    return run_inline(
        read_changed(
            db,
            "results",
            key=["state", Sort("geo", cast="bigint")],
            columns=list(_COLUMNS),
            held=held,
            hash_columns=_HASHED,
            by={"state": state},
            **options,
        )
    )


def test_the_hash_marks_null_apart_and_never_joins_two_rows_alike(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if a NULL hashes as an empty text, if two rows whose texts only move between columns hash alike
    (``('a', 'bc')`` vs ``('ab', 'c')``), or the hash is not 32 lowercase hex digits."""
    columns = (HashColumn("x"), HashColumn("y"))
    rows = [("a", "bc"), ("ab", "c"), ("", None), (None, ""), (None, None), ("~", None)]
    conn.execute("CREATE TABLE pairs (x VARCHAR, y VARCHAR)")
    conn.executemany("INSERT INTO pairs VALUES (?, ?)", rows)
    hashes = [h for (h,) in conn.execute(f"SELECT {row_hash_sql(columns, Dialect.DUCKDB)} FROM pairs AS t").fetchall()]
    assert len(set(hashes)) == len(rows), hashes
    assert all(len(h) == 32 and h == h.lower() and int(h, 16) >= 0 for h in hashes)


def test_each_kind_is_written_as_its_canonical_text(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if a kind's canonical text moves (a numeric at its column's own scale, a timestamp without its six
    fractional digits, a boolean as ``true``): the hashes stored beside rows would all differ from the next read."""
    expression = row_hash_sql(_HASHED, Dialect.DUCKDB)
    assert expression.startswith("md5(") and "'~'" in expression
    text = expression.removeprefix("md5(")[:-1]
    (found,) = conn.execute(f"SELECT {text} FROM results AS t WHERE geo = 3").fetchone() or (None,)
    assert found == "2:n32:306:0.00301:f10:2024-11-0526:2024-11-05 20:00:00.000003"


def test_the_postgres_and_redshift_spellings_name_the_same_canonical_forms() -> None:
    """Flips if a dialect's spelling leaves the forms the others write (to_char's US microseconds, DECIMAL(38, s)),
    or Redshift's text is cast to a bounded VARCHAR (256 characters) that would cut a long value."""
    postgres = row_hash_sql(_HASHED, Dialect.POSTGRES)
    assert postgres == row_hash_sql(_HASHED, Dialect.YUGABYTE)
    assert "to_char(t.\"stamp\", 'YYYY-MM-DD HH24:MI:SS.US')" in postgres
    assert "to_char(t.\"day\", 'YYYY-MM-DD')" in postgres
    assert 'CAST(CAST(t."share" AS DECIMAL(38, 4)) AS VARCHAR)' in postgres
    redshift = row_hash_sql(_HASHED, Dialect.REDSHIFT)
    assert 'CAST(t."name" AS VARCHAR)' not in redshift and 'LENGTH(t."name")' in redshift
    assert row_hash_sql([HashColumn("a")], Dialect.POSTGRES, alias=None) == (
        "md5(COALESCE(CAST(LENGTH(CAST(\"a\" AS VARCHAR)) AS VARCHAR) || ':' || CAST(\"a\" AS VARCHAR), '~'))"
    )
    selected = row_hash_selected(_HASHED, Dialect.POSTGRES)
    assert (selected.expression, selected.alias) == (postgres, ROW_HASH_COLUMN)


def test_a_hash_refuses_no_column_a_column_twice_and_a_numeric_without_scale() -> None:
    """Flips if a hash over nothing, a column named twice, a numeric with no scale (its text would follow each
    database's own scale) or a scale on another kind is taken."""
    with pytest.raises(ValueError, match="at least one column"):
        row_hash_sql([], Dialect.DUCKDB)
    with pytest.raises(ValueError, match="each column once"):
        row_hash_sql([HashColumn("a"), HashColumn("a", HashKind.INTEGER)], Dialect.DUCKDB)
    with pytest.raises(ValueError, match="names its scale"):
        HashColumn("a", HashKind.NUMERIC)
    with pytest.raises(ValueError, match="only a hashed numeric"):
        HashColumn("a", HashKind.INTEGER, scale=2)


def test_only_the_changed_and_new_rows_are_read_in_full_and_gone_keys_named(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if phase 2 reads a row whose hash the caller holds unchanged, misses a changed or new row, names a
    held key the scope still holds as gone, or misses one that left; a row carries its own hash."""
    held = _held(conn, "PA")
    conn.execute("UPDATE results SET votes = votes + 1 WHERE geo IN (3, 6)")
    conn.execute("UPDATE results SET called = NULL WHERE geo = 9")
    conn.execute("DELETE FROM results WHERE geo = 12")
    conn.execute("INSERT INTO results VALUES ('PA', 900, NULL, NULL, NULL, NULL, NULL, NULL)")
    conn.execute("UPDATE results SET votes = -1 WHERE geo = 1")  # OH: outside the scope
    counted = _Counted(conn)
    changed = _changed(counted, held)
    assert sorted(row["geo"] for row in changed.rows) == [3, 6, 9, 900]
    assert changed.gone_keys == [("PA", 12)]
    assert len(counted.statements) == 2
    now = _held(conn, "PA")
    assert all(row[ROW_HASH_COLUMN] == now[(row["state"], row["geo"])] for row in changed.rows)


def test_a_scope_nothing_changed_in_is_one_statement(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if a scope whose every hash matches reads a row in full."""
    counted = _Counted(conn)
    changed = _changed(counted, _held(conn, "PA"))
    assert (changed.rows, changed.gone_keys, len(counted.statements)) == ([], [], 1)


def test_the_scope_s_filters_bound_both_phases(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if ``where=`` is dropped from either phase: a held key outside it would read as changed or gone."""
    held = {key: h for key, h in _held(conn, "PA").items() if key[1] >= 30}
    conn.execute("UPDATE results SET votes = 0")
    changed = _changed(InlineDuckDB(conn), held, where=[Fragment('t."geo" >= $1', (30,))])
    assert sorted(row["geo"] for row in changed.rows) == [geo for geo in range(30, 60) if geo % 3 == 0]
    assert changed.gone_keys == []


def test_phase_one_pages_by_the_key(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if phase 1 reads the scope in one statement past its page, or loses a key at a page's edge."""
    counted = _Counted(conn)
    changed = _changed(counted, {}, page=7)
    assert sorted(row["geo"] for row in changed.rows) == list(range(0, 60, 3))
    assert sum(" LIMIT 8" in statement for statement in counted.statements) == 3


class _Warehouse:
    """a Redshift reader's double: phase 1 answers two keys' hashes, phase 2 one row; every statement kept."""

    binds_native_arrays = False
    sql_dialect = Dialect.REDSHIFT

    def __init__(self) -> None:
        self.asked: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
        self.asked.append((query, params))
        if " LIMIT " in query:
            return [
                {"state": "PA", "geo": 3, ROW_HASH_COLUMN: "a" * 32},
                {"state": "PA", "geo": 6, ROW_HASH_COLUMN: "b" * 32},
            ]
        return [{"state": "PA", "geo": 6, "name": "n6", ROW_HASH_COLUMN: "b" * 32}]


def test_a_warehouse_reads_its_hash_in_its_own_spelling_and_lists_its_changed_keys() -> None:
    """Flips if a Redshift reader's hash is spelled other than Redshift's (strftime is DuckDB's), its phase 2 binds the
    keys as arrays (it has none), or a key held with no hash (a row stored before hashes) is not read again."""
    warehouse = _Warehouse()
    changed = run_inline(
        read_changed(
            warehouse,
            "results",
            key=["state", Sort("geo", cast="bigint")],
            columns=["state", "geo", "name"],
            held={("PA", 3): "a" * 32, ("PA", 6): None},
            hash_columns=_HASHED,
            by={"state": "PA"},
        )
    )
    assert [row["geo"] for row in changed.rows] == [6] and changed.gone_keys == []
    (scope, _), (keyed, params) = warehouse.asked
    assert row_hash_sql(_HASHED, Dialect.REDSHIFT) in scope and "strftime" not in scope
    assert "unnest" not in keyed and "::text[]" not in keyed and '(t."state", t."geo") IN (($1, $2))' in keyed
    assert "PA" in params and 6 in params


def test_the_dialect_is_the_reader_s_own() -> None:
    """Flips if a reader's named dialect is overridden, an in-process DuckDB is not DuckDB's, or a reader naming an
    engine no hash is spelled for is taken."""

    class _Named:
        sql_dialect = "snowflake"

    assert run_inline(dialect_of(_Warehouse())) is Dialect.REDSHIFT
    assert run_inline(dialect_of(InlineDuckDB(duckdb.connect()))) is Dialect.DUCKDB
    with pytest.raises(ValueError, match="snowflake"):
        run_inline(dialect_of(_Named()))


def test_the_rows_after_applying_the_changes_are_a_full_read_s(conn: duckdb.DuckDBPyConnection) -> None:
    """Flips if what a follower holds after applying read_changed's rows and gone keys to its stale copy differs
    from a plain full read of the scope, over empty, one, many, NULL and tied seeds and both tenants."""
    conn.execute(
        "CREATE TABLE seeded (tenant VARCHAR, id BIGINT, name VARCHAR, votes BIGINT, PRIMARY KEY (tenant, id))"
    )

    def row(seed: Seeded) -> dict[str, Any]:
        return {
            "tenant": f"t{seed.tenant}",
            "id": seed.index,
            "name": None if seed.null else f"n{seed.tie}",
            "votes": seed.tie,
        }

    hashed = (HashColumn("name"), HashColumn("votes", HashKind.INTEGER))
    columns = ["tenant", "id", "name", "votes"]

    async def follow(db: Any) -> Sequence[Mapping[str, Any]]:
        # the follower's stale copy: every row, a third of them held under a stale hash, and one key it holds that
        # the scope no longer does
        copy = await read_keyed(
            db,
            "seeded",
            keys=KeyBatch(columns=("tenant",), values=[("t0",)]),
            columns=[*columns, row_hash_selected(hashed, Dialect.DUCKDB)],
        )
        held = {("t0", found["id"]): ("0" * 32 if found["id"] % 3 == 0 else found[ROW_HASH_COLUMN]) for found in copy}
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
        assert ("t0", 10_000) in changed.gone_keys
        for gone in changed.gone_keys:
            del rows[gone]
        rows.update({("t0", found["id"]): found for found in changed.rows})
        return list(rows.values())

    run_inline(
        assert_same_results(
            InlineDuckDB(conn),
            table="seeded",
            row=row,
            old_sql="SELECT tenant, id, name, votes FROM seeded WHERE tenant = $1",
            params=["t0"],
            helper=follow,
            columns=columns,
            ordered=False,
        )
    )


#: a wide row's text columns: with the key, 77 columns of about 2.4 KB of JSON, as reports' candidate rows are
_WIDE = tuple(f"c{index:02d}" for index in range(75))


def test_a_large_scope_of_wide_rows_with_few_changes_is_a_few_statements_and_a_few_percent_of_the_bytes() -> None:
    """Flips if 124,000 wide rows in scope with 3,000 changed take more than phase 1's three pages and phase 2's
    three key batches, or carry more than 10% of a full read's bytes (every column of every row)."""
    db = duckdb.connect()
    db.execute(
        f"CREATE TABLE wide (state VARCHAR, geo BIGINT, {', '.join(f'{c} VARCHAR' for c in _WIDE)}, "
        "PRIMARY KEY (state, geo))"
    )
    values = ", ".join(f"'{c}-' || lpad(CAST(i AS VARCHAR), 24, '0')" for c in _WIDE)
    db.execute(f"INSERT INTO wide SELECT 'PA', i, {values} FROM range(124000) r(i)")
    hashed = [HashColumn(c) for c in _WIDE]
    held = dict(
        (
            (("PA", geo), h)
            for geo, h in db.execute(
                f"SELECT geo, {row_hash_sql(hashed, Dialect.DUCKDB)} FROM wide AS t WHERE state = 'PA'"
            ).fetchall()
        )
    )
    db.execute("UPDATE wide SET c40 = 'moved' WHERE geo < 3000")
    counted = _Counted(db)
    changed = run_inline(
        read_changed(
            counted,
            "wide",
            key=["state", Sort("geo", cast="bigint")],
            columns=["state", "geo", *_WIDE],
            held=held,
            hash_columns=hashed,
            by={"state": "PA"},
        )
    )
    assert len(changed.rows) == 3000 and changed.gone_keys == []
    assert len(counted.statements) == 6, [statement[:80] for statement in counted.statements]
    (full,) = db.execute("SELECT sum(strlen(CAST(to_json(w) AS VARCHAR))) FROM wide w").fetchone() or (0,)
    print(f"read_changed bytes {counted.bytes:,} of a full read's {full:,}: {counted.bytes / full:.1%}")
    assert full / 124000 > 2300
    assert counted.bytes <= 0.10 * full, (counted.bytes, full)
    db.close()


def test_a_key_value_s_canonical_text_is_the_same_held_as_text_or_answered_typed() -> None:
    """Flips if a key value held in one form (text, a zoned time) and answered in another (an integer, a naive time)
    give different texts, or a value that is no value of its kind is taken."""
    assert key_text("7001", HashKind.INTEGER) == key_text(7001, HashKind.INTEGER) == "7001"
    naive = datetime(2026, 11, 3, 22, 30)  # noqa: DTZ001 - the warehouse's naive time
    for held in (
        "2026-11-03 22:30:00",
        "2026-11-03T22:30:00+00:00",
        "2026-11-03 17:30:00-05:00",
        datetime(2026, 11, 3, 17, 30, tzinfo=timezone(timedelta(hours=-5))),
    ):
        assert key_text(held, HashKind.TIMESTAMP) == key_text(naive, HashKind.TIMESTAMP) == "2026-11-03 22:30:00.000000"
    assert key_text("2024-11-05", HashKind.DATE) == key_text(date(2024, 11, 5), HashKind.DATE) == "2024-11-05"
    assert key_text("true", HashKind.BOOLEAN) == key_text(True, HashKind.BOOLEAN) == "t"
    assert key_text("1.5", HashKind.NUMERIC, 2) == key_text(Decimal("1.50"), HashKind.NUMERIC, 2) == "1.50"
    assert key_text(None, HashKind.INTEGER) is None
    for value, kind in (
        ("x", HashKind.INTEGER),
        (1.5, HashKind.INTEGER),
        ("maybe", HashKind.BOOLEAN),
        (naive, HashKind.DATE),
        ("soon", HashKind.TIMESTAMP),
    ):
        with pytest.raises(ValueError):
            key_text(value, kind)


@pytest.fixture
def typed() -> Iterator[duckdb.DuckDBPyConnection]:
    """a warehouse table keyed by an integer id and a time (no zone), holding them typed."""
    db = duckdb.connect()
    db.execute(
        "CREATE TABLE typed (state VARCHAR, cid INTEGER, stamp TIMESTAMP, name VARCHAR, PRIMARY KEY (state, cid, stamp))"
    )
    db.execute(
        "INSERT INTO typed SELECT 'PA', 7000 + i, TIMESTAMP '2026-11-03 22:30:00' + to_seconds(i), 'n' || i "
        "FROM range(6) r(i)"
    )
    yield db
    db.close()


_TYPED_HASH = (
    HashColumn("cid", HashKind.INTEGER),
    HashColumn("stamp", HashKind.TIMESTAMP),
    HashColumn("name"),
)


def _held_as_text(db: duckdb.DuckDBPyConnection) -> dict[tuple[Any, ...], str]:
    """each row's key as a follower holds it (its own text: the id's digits, the time zoned in UTC) onto its hash."""
    sql = f"SELECT state, cid, stamp, {row_hash_sql(_TYPED_HASH, Dialect.DUCKDB)} FROM typed AS t"
    return {
        (state, str(cid), f"{stamp.isoformat(sep=' ')}+00:00"): h for state, cid, stamp, h in db.execute(sql).fetchall()
    }


def _typed_changed(db: Any, held: Mapping[tuple[Any, ...], str | None], **options: Any) -> ChangedRows:
    return run_inline(
        read_changed(
            db,
            "typed",
            key=["state", "cid", "stamp"],
            columns=["state", "cid", "stamp", "name"],
            held=held,
            hash_columns=_TYPED_HASH,
            by={"state": "PA"},
            **options,
        )
    )


def test_a_typed_key_held_as_text_reads_no_false_change_and_no_false_gone(typed: duckdb.DuckDBPyConnection) -> None:
    """Flips if a key held as text (an integer id's digits, a zoned time) is compared with the warehouse's typed
    answer as it is: every row would read as changed and every held key as gone."""
    held = _held_as_text(typed)
    counted = _Counted(typed)
    changed = _typed_changed(counted, held)
    assert (changed.rows, changed.gone_keys, len(counted.statements)) == ([], [], 1)


def test_a_typed_key_s_changed_row_is_read_and_its_gone_key_named_as_held(typed: duckdb.DuckDBPyConnection) -> None:
    """Flips if a changed row under a typed key is missed or another read with it, a row that left is not named
    gone in the form it is held in, or phase 2 binds the typed key as text (an integer = text comparison)."""
    held = _held_as_text(typed)
    typed.execute("UPDATE typed SET name = 'moved' WHERE cid = 7002")
    typed.execute("DELETE FROM typed WHERE cid = 7004")
    counted = _Counted(typed)
    changed = _typed_changed(counted, held, page=2)
    assert [(row["cid"], row["name"]) for row in changed.rows] == [(7002, "moved")]
    assert changed.gone_keys == [("PA", "7004", "2026-11-03 22:30:04+00:00")]
    keyed = counted.statements[-1]
    assert "::text[]::bigint[]" in keyed and "::text[]::timestamp[]" in keyed, keyed
    assert all("::text::bigint" in statement for statement in counted.statements[1:-1]), "pages cast the cursor"


def test_a_key_the_hash_does_not_cover_is_typed_by_key_kinds(typed: duckdb.DuckDBPyConnection) -> None:
    """Flips if ``key_kinds`` does not type a key column the hash leaves out."""
    hashed = (HashColumn("name"),)
    sql = f"SELECT state, cid, stamp, {row_hash_sql(hashed, Dialect.DUCKDB)} FROM typed AS t"
    held = {(s, str(c), f"{a.isoformat(sep=' ')}+00:00"): h for s, c, a, h in typed.execute(sql).fetchall()}
    changed = run_inline(
        read_changed(
            InlineDuckDB(typed),
            "typed",
            key=["state", "cid", "stamp"],
            columns=["state", "cid", "stamp", "name"],
            held=held,
            hash_columns=hashed,
            by={"state": "PA"},
            key_kinds=[HashColumn("cid", HashKind.INTEGER), HashColumn("stamp", HashKind.TIMESTAMP)],
        )
    )
    assert (changed.rows, changed.gone_keys) == ([], [])
