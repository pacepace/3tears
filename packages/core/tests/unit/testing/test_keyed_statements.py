"""the keyed-statement decoder answers what the helpers render the way PostgreSQL would.

An in-memory store built on :func:`~threetears.core.testing.keyed_statements.decode` stands in for a pool in unit
tests of code that calls the keyed helpers; these tests drive the real helpers into such a store and check each
answer, and that a statement the decoder cannot read is refused rather than matched.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from threetears.core.keyed import (
    AnyOf,
    Fragment,
    KeyBatch,
    Sort,
    counts_by,
    delete_keyed,
    keyset_page,
    read_keyed,
    update_keyed,
)
from threetears.core.testing.keyed_statements import decode, keyed_fetch

_T0 = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
_NOW = datetime(2026, 11, 10, 13, 0, tzinfo=UTC)


# parity-with: asyncpg.Pool (fetch / execute -- the seam the keyed helpers drive)
class _Store:
    """one table of row dicts, answering each helper statement through the decoder."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.statements: list[str] = []

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        self.statements.append(sql)
        statement = decode(sql, args)
        if statement.kind == "update":
            return statement.update(self.rows, now=_NOW)
        if statement.kind == "delete":
            return statement.delete(self.rows)
        if statement.kind == "count":
            return statement.count(self.rows)
        return statement.select(self.rows)

    async def execute(self, sql: str, *args: Any) -> str:
        self.statements.append(sql)
        statement = decode(sql, args)
        if statement.kind == "update":
            return f"UPDATE {len(statement.update(self.rows, now=_NOW))}"
        assert statement.kind == "delete", sql
        return f"DELETE {len(statement.delete(self.rows))}"


def _rows() -> tuple[uuid.UUID, uuid.UUID, list[dict[str, Any]]]:
    tenant, other = uuid.uuid4(), uuid.uuid4()
    rows = [
        {"tenant": tenant, "item": uuid.uuid4(), "status": "active", "due": _T0 + timedelta(minutes=3), "n": 1},
        {"tenant": tenant, "item": uuid.uuid4(), "status": "active", "due": None, "n": 2},
        {"tenant": tenant, "item": uuid.uuid4(), "status": "paused", "due": _T0 + timedelta(minutes=1), "n": 3},
        {"tenant": other, "item": uuid.uuid4(), "status": "active", "due": _T0 + timedelta(minutes=2), "n": 4},
    ]
    return tenant, other, rows


@pytest.mark.asyncio
async def test_a_keyed_read_answers_the_rows_its_keys_and_tenant_name() -> None:
    tenant, other, rows = _rows()
    store = _Store(rows)
    wanted = [rows[0]["item"], rows[3]["item"]]

    found = await read_keyed(
        store,
        "items",
        keys=KeyBatch(("item",), [(k,) for k in wanted], ("uuid",)),
        columns=["item", "n"],
        by={"tenant": tenant},
    )

    assert found == [{"item": rows[0]["item"], "n": 1}], "the other tenant's key answered, or the tenant's own did not"
    assert other != tenant


@pytest.mark.asyncio
async def test_a_page_is_matched_ordered_nulls_last_and_limited() -> None:
    tenant, _, rows = _rows()
    store = _Store(rows)

    page = await keyset_page(
        store,
        "items",
        columns=["item", "due", "n"],
        by={"tenant": tenant},
        order=[Sort("due", cast="timestamptz", nullable=True), Sort("item", cast="uuid")],
        limit=2,
    )

    assert [row["n"] for row in page.rows] == [3, 1], "not in due order, or the NULL due sorted first"


@pytest.mark.asyncio
async def test_a_where_fragment_bounds_the_page() -> None:
    _, _, rows = _rows()
    store = _Store(rows)

    page = await keyset_page(
        store,
        "items",
        columns=["due", "n"],
        by={"status": "active"},
        order=[Sort("due", cast="timestamptz")],
        limit=10,
        where=[Fragment('t."due" IS NOT NULL AND t."due" <= $1', (_T0 + timedelta(minutes=2),))],
    )

    assert [row["n"] for row in page.rows] == [4]


@pytest.mark.asyncio
async def test_an_update_sets_its_values_only_on_the_rows_it_names_and_counts_them() -> None:
    tenant, _, rows = _rows()
    store = _Store(rows)

    missed = await update_keyed(
        store, "items", values={"status": "paused"}, by={"tenant": tenant, "item": rows[0]["item"], "status": "paused"}
    )
    hit = await update_keyed(
        store,
        "items",
        values={"status": "paused", "due": None, "stamped": Fragment("now()")},
        by={"tenant": tenant, "item": rows[0]["item"]},
        where=[Fragment(""""status" != 'expired'""")],
        returning=["item"],
    )

    assert missed.count == 0
    assert hit.count == 1 and hit.rows == [{"item": rows[0]["item"]}]
    assert rows[0]["status"] == "paused" and rows[0]["due"] is None and rows[0]["stamped"] == _NOW
    assert rows[1]["status"] == "active", "an update reached a row it did not name"


@pytest.mark.asyncio
async def test_a_delete_removes_only_the_rows_it_names() -> None:
    tenant, _, rows = _rows()
    store = _Store(rows)
    gone = rows[1]["item"]

    await delete_keyed(store, "items", by={"tenant": tenant, "item": gone})

    assert [row["item"] for row in rows if row["item"] == gone] == []
    assert len(rows) == 3


@pytest.mark.asyncio
async def test_a_group_count_counts_each_value_within_its_pins() -> None:
    tenant, other, rows = _rows()
    store = _Store(rows)

    counted = await counts_by(
        store, "items", column="tenant", values=AnyOf([tenant, other], "uuid"), by={"status": "active"}
    )

    assert counted == {tenant: 2, other: 1}


def test_a_statement_the_helpers_do_not_write_is_refused() -> None:
    with pytest.raises(ValueError, match="not a statement"):
        decode("INSERT INTO items (n) VALUES ($1)", [1])
    with pytest.raises(ValueError, match="condition the decoder cannot read"):
        decode('SELECT t."n" FROM "items" AS t WHERE t."n" LIKE $1', ["x%"])
    with pytest.raises(ValueError, match="SET value"):
        decode('UPDATE "items" SET "n" = n + 1 WHERE "item" = $1', [1]).update([{"item": 1, "n": 0}])


async def test_keyed_fetch_answers_each_read_by_key_in_turn_and_hands_every_other_statement_on() -> None:
    """Flips if keyed_fetch answers a keyed page, another table's read or a hand-written statement from its queue,
    or answers the reads by key out of turn."""
    first, second = {"id": uuid.UUID(int=1), "name": "a"}, {"id": uuid.UUID(int=2), "name": "b"}
    handed: list[str] = []

    async def rest(sql: str, *_args: Any) -> list[dict[str, Any]]:
        handed.append(sql)
        return []

    fetch = keyed_fetch(first, None, second, tables={"widgets"}, rest=rest)

    # parity-with: asyncpg.Pool (fetch -- the seam the keyed helpers drive)
    class _Pool:
        async def fetch(self, sql: str, *args: Any) -> Any:
            return await fetch(sql, *args)

    pool = _Pool()
    keys = KeyBatch(columns=("id",), values=[(first["id"],)], casts=("uuid",))
    assert await read_keyed(pool, "widgets", keys=keys, columns=("id", "name")) == [first]
    await keyset_page(pool, "widgets", columns=("id", "name"), by={"tenant": 1}, order=(Sort("name"),), limit=5)
    assert await read_keyed(pool, "gadgets", keys=keys, columns=("id",)) == []
    assert await pool.fetch("SELECT 1") == []
    assert await read_keyed(pool, "widgets", keys=keys, columns=("id", "name")) == []
    assert await read_keyed(pool, "widgets", keys=keys, columns=("id", "name")) == [second]
    assert [sql.split(" FROM ")[-1][:10] for sql in handed[:2]] == ['"widgets" ', '"gadgets" '], handed
    assert handed[2] == "SELECT 1"
    with pytest.raises(AssertionError, match="past the queued answers"):
        await read_keyed(pool, "widgets", keys=keys, columns=("id", "name"))
    assert fetch.await_count == 7
