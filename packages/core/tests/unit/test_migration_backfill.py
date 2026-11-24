"""A migration's backfill: the rows read a page at a time by key, each page's writes keyed updates, never one UPDATE of
every row."""

from __future__ import annotations

from typing import Any

from threetears.core.keyed import Fragment
from threetears.core.data.migrations.backfill import backfill_pages


class _Held:
    """a table held here: answers each keyset page from its rows in key order, keeps every statement, and applies
    nothing (a backfill's count is the database's answer)."""

    def __init__(self, rows: list[dict[str, Any]], key: tuple[str, ...]) -> None:
        self.rows = sorted(rows, key=lambda row: tuple(row[c] for c in key))
        self.key = key
        self.statements: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        self.statements.append((sql, params))
        if sql.startswith("SELECT version()"):
            return [{"v": "PostgreSQL 16.4"}]
        limit = int(sql.rsplit("LIMIT ", 1)[1])
        if "> (" not in sql:
            return self.rows[:limit]
        cursor = tuple(params[-len(self.key) :])
        return [row for row in self.rows if tuple(row[c] for c in self.key) > cursor][:limit]

    async def execute(self, sql: str, *params: Any) -> str:
        self.statements.append((sql, params))
        return f"UPDATE {len(params[-1])}"


def _writes(held: _Held) -> list[tuple[str, tuple[Any, ...]]]:
    return [(sql, params) for sql, params in held.statements if sql.startswith("UPDATE")]


async def test_each_page_writes_its_rows_of_one_value_by_their_keys() -> None:
    """Flips if a backfill writes a table's rows by anything but their keys, or a row ``fill`` leaves alone."""
    rows = [{"id": f"r{n:02d}", "status": None if n % 3 else "done", "notes": n % 2} for n in range(10)]
    held = _Held(rows, ("id",))

    written = await backfill_pages(
        held,
        "t",
        ("id",),
        columns=("status", "notes"),
        fill=lambda row: None if row["status"] is not None else {"status": "odd" if row["notes"] else "even"},
        page_size=4,
    )

    reads = [sql for sql, _ in held.statements if sql.startswith("SELECT")]
    assert reads[0] == 'SELECT "id", "status", "notes" FROM t ORDER BY "id" LIMIT 5'
    assert len(reads) == 3, reads
    for sql, params in _writes(held):
        assert sql == 'UPDATE "t" SET "status" = $1 WHERE "id" = ANY($2::text[])', sql
        named = set(params[1])
        assert all(rows[int(key[1:])]["status"] is None for key in named), "a filled row was written again"
    every = sorted(key for _, params in _writes(held) for key in params[1])
    assert every == [row["id"] for row in rows if row["status"] is None]
    assert written == len(every)
    first_page_writes = _writes(held)[:2]
    assert {params[0] for _, params in first_page_writes} == {"odd", "even"}, "one write per value a page holds"


async def test_a_fragment_value_is_computed_from_the_row_s_own_columns() -> None:
    """Flips if an expression value (``LEFT("content", 1000)``) is bound as a literal rather than written as SQL."""
    held = _Held([{"a": 1, "b": 2, "long_desc": ""}], ("a", "b"))

    await backfill_pages(
        held,
        "s.items",
        ("a", "b"),
        columns=("long_desc",),
        fill=lambda row: {"long_desc": Fragment('LEFT("content", 1000)')} if row["long_desc"] == "" else None,
        casts=("bigint", "bigint"),
    )

    [(sql, params)] = _writes(held)
    assert sql == (
        'UPDATE "s"."items" SET "long_desc" = LEFT("content", 1000) '
        'WHERE ("a", "b") IN (SELECT unnest($1::text[]::bigint[]), unnest($2::text[]::bigint[]))'
    )
    assert params == (["1"], ["2"])


async def test_a_replay_over_filled_rows_writes_nothing() -> None:
    """Flips if a backfill writes a row whose ``fill`` answers nothing (a replay on recovery must write nothing)."""
    held = _Held([{"id": f"r{n}", "done": True} for n in range(7)], ("id",))

    written = await backfill_pages(held, "t", ("id",), columns=("done",), fill=lambda row: None, page_size=3)

    assert written == 0
    assert _writes(held) == []
