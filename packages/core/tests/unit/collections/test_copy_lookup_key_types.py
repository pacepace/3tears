"""a keyed read of the copy compares each key column by its type's value, as the copy's SQL read by key did.

The copy's read by key (``FollowedTable.held_key``) cast both sides to the column's type in SQL: a ``NUMERIC`` key the
copy holds as the rail's text (``1.50``) matched a change record's ``1.5``. Its move to Arrow's hash kernels in a query
worker (``CopyQueries.lookup``, chunk 3T-7) matched the text exactly, so such a key missed its row and the change kept
a stale one. The contract pinned here, over a version of the copy its backend applied and published as shared Arrow
(``shared_copy.SharedCopyBackend``, its ``canonical`` key columns named by ``copy_reads.canonical_keys``), through the
follower's reads (``copy_reads.QueriedReads.rows_of_keys``):

- a ``NUMERIC`` key held as text is found whatever its spelling (``1.5``, ``1.50``, ``01.5``), and the row comes back
  as the copy holds it, in the one spelling the copy stores (``1.5``);
- a UUID key held as text is found in either case, and held lower-cased;
- an integer key the copy holds typed is found by its text; an instant key by any ISO spelling of it;
- a key of another value is not found;
- a lookup spells only its asked keys, never the held column: its cost is its keys', not the part's rows.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from threetears.core.collections import copy_queries
from threetears.core.collections.copy_follower import FollowedTable
from threetears.core.collections.copy_parts import PartRule
from threetears.core.collections.copy_queries import CopyQueries
from threetears.core.collections.copy_reads import QueriedReads, canonical_keys
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    DATETIMETZ_TYPE,
    NUMERIC_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
)
from threetears.core.collections.shared_copy import ROWS, ChangePage, SharedCopyBackend
from threetears.core.shared_arrow import SharedArrowStore, read_version

_UUID = "6f1c2a9e-4b7d-4c1e-9a2f-0d3b5e7c9a11"

#: each table, its key, its columns' type tags, and the rows the copy holds (as the rail answers them: a NUMERIC, a
#: UUID and an instant as text, an integer typed)
_TABLES = {
    "prices": (("state", "price"), {"state": STRING_TYPE, "price": NUMERIC_TYPE, "note": STRING_TYPE}),
    "people": (("person_id",), {"person_id": UUID_TYPE, "note": STRING_TYPE}),
    "races": (("race_id",), {"race_id": BIGINT_TYPE, "note": STRING_TYPE}),
    "snapshots": (("taken_at",), {"taken_at": DATETIMETZ_TYPE, "note": STRING_TYPE}),
}

_ROWS = {
    "prices": pa.table(
        {
            "state": pa.array(["VA", "VA", "IL"], pa.string()),
            "price": pa.array(["1.50", "10", "1.50"], pa.string()),
            "note": pa.array(["va-1.5", "va-10", "il-1.5"], pa.string()),
        }
    ),
    "people": pa.table({"person_id": pa.array([_UUID], pa.string()), "note": pa.array(["one"], pa.string())}),
    "races": pa.table({"race_id": pa.array([7, 70], pa.int64()), "note": pa.array(["seven", "seventy"], pa.string())}),
    "snapshots": pa.table(
        {
            "taken_at": pa.array(["2026-11-03T23:59:00.000000+00:00"], pa.string()),
            "note": pa.array(["night"], pa.string()),
        }
    ),
}


def _table(name: str) -> FollowedTable:
    key, kinds = _TABLES[name]
    columns = tuple(
        Column(name=column, column_type=kind, precision=12, scale=3)
        if kind == NUMERIC_TYPE
        else Column(name=column, column_type=kind)
        for column, kind in kinds.items()
    )
    return FollowedTable(name=name, key=key, columns=columns)


@pytest.fixture
def store(tmp_path: Path) -> SharedArrowStore:
    return SharedArrowStore(tmp_path / "shm", max_bytes=1 << 28)


@pytest.fixture
async def reads(store: SharedArrowStore) -> AsyncIterator[QueriedReads]:
    tables = [_table(name) for name in _TABLES]
    backend = SharedCopyBackend(
        store,
        {name: rows.schema for name, rows in _ROWS.items()},
        {table.name: PartRule(table.name, tuple(table.key)) for table in tables},
        canonical=canonical_keys(tables),
        instants={"snapshots": {"taken_at"}},
    )
    try:
        await backend.apply([ChangePage(ROWS, name, rows) for name, rows in _ROWS.items()], mark=1)
    finally:
        await backend.close()
    pool = CopyQueries(store, workers=1)
    try:
        yield QueriedReads(pool)
    finally:
        await pool.close()


def _held(store: SharedArrowStore, table: str, column: str) -> list[str]:
    current = store.current
    assert current is not None
    with read_version(store.root, current.name) as mapped:
        return sorted(mapped.tables[table][column].to_pylist())


async def _found(reads: QueriedReads, name: str, keys: dict[str, list[str]]) -> list[str]:
    """the notes of the rows the copy holds of ``keys`` (each key column as text, as a change record names it)."""
    table = _table(name)
    asked = pa.table({column: pa.array(values, pa.string()) for column, values in keys.items()})
    rows = await reads.rows_of_keys(table, list(keys), asked, [*table.key, "note"])
    return sorted(rows.column("note").to_pylist())


async def test_a_numeric_key_is_found_whatever_its_spelling(reads: QueriedReads) -> None:
    """Flips if a NUMERIC key the copy holds as ``1.50`` is matched by its text: a change record's ``1.5`` (or
    ``01.5``, or ``1.500``) misses the row, and the change keeps a stale one."""
    for spelled in ("1.5", "1.50", "01.5", "1.500"):
        assert await _found(reads, "prices", {"state": ["VA"], "price": [spelled]}) == ["va-1.5"], spelled
    assert await _found(reads, "prices", {"state": ["VA", "IL"], "price": ["1.5", "1.5"]}) == ["il-1.5", "va-1.5"]
    assert await _found(reads, "prices", {"state": ["VA"], "price": ["10.0"]}) == ["va-10"]
    assert await _found(reads, "prices", {"state": ["VA"], "price": ["1.05"]}) == []


async def test_a_found_numeric_row_is_the_row_the_copy_holds(reads: QueriedReads) -> None:
    """the row comes back as the copy holds it (the rail's ``1.50`` stored as ``1.5``), never in the spelling it was
    asked by (``1.500``)."""
    table = _table("prices")
    asked = pa.table({"state": pa.array(["VA"], pa.string()), "price": pa.array(["1.500"], pa.string())})
    rows = await reads.rows_of_keys(table, ["state", "price"], asked, ["state", "price", "note"])
    assert rows.to_pylist() == [{"state": "VA", "price": "1.5", "note": "va-1.5"}]


async def test_the_copy_stores_each_canonical_key_in_one_spelling(reads: QueriedReads, store: SharedArrowStore) -> None:
    """Flips if the apply keeps the rail's spelling: the copy holds ``1.50`` as ``1.5``, a UUID lower-cased, once, as
    the change is applied; a column not named canonical (``note``) as it came."""
    assert _held(store, "prices", "price") == ["1.5", "1.5", "10"]
    assert _held(store, "people", "person_id") == [_UUID]
    assert _held(store, "prices", "note") == ["il-1.5", "va-1.5", "va-10"]


def test_a_lookup_respells_only_its_asked_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flips if a lookup respells the held column: every text kernel the comparison runs (``replace_substring_regex``
    for a NUMERIC, ``utf8_lower`` for a UUID) sees the asked keys alone, never the part's 200,000 rows, so a lookup's
    cost is its keys'."""
    count = 200_000
    held = pa.table(
        {
            "state": pa.array(["VA"] * count, pa.string()),
            "price": pc.cast(pa.array(np.arange(count)), pa.string()),
            "person_id": pa.array([f"{n:08d}-0000-0000-0000-000000000000" for n in range(count)], pa.string()),
        }
    )
    asked = pa.table(
        {
            "state": pa.array(["VA", "VA"], pa.string()),
            "price": pa.array(["7.00", "+12"], pa.string()),
            "person_id": pa.array(["00000007-0000-0000-0000-000000000000"] * 2, pa.string()),
        }
    )
    seen: list[int] = []
    for kernel in ("replace_substring_regex", "utf8_lower"):
        original = getattr(pc, kernel)

        def spy(values: object, *args: object, _original: object = original, **kwargs: object) -> object:
            seen.append(len(values))  # type: ignore[arg-type]
            return _original(values, *args, **kwargs)  # type: ignore[operator]

        monkeypatch.setattr(pc, kernel, spy)
    found = copy_queries.rows_found(held, asked, ("state", "price"), {"price": "numeric"})
    assert found["price"].to_pylist() == ["7", "12"]
    found = copy_queries.rows_found(held, asked.slice(0, 1), ("state", "person_id"), {"person_id": "uuid"})
    assert found.num_rows == 1
    assert seen and max(seen) <= asked.num_rows, seen


async def test_a_uuid_key_is_found_in_either_case(reads: QueriedReads) -> None:
    assert await _found(reads, "people", {"person_id": [_UUID.upper()]}) == ["one"]
    assert await _found(reads, "people", {"person_id": [_UUID]}) == ["one"]


async def test_an_integer_key_given_as_text_is_found(reads: QueriedReads) -> None:
    assert await _found(reads, "races", {"race_id": ["7"]}) == ["seven"]
    assert await _found(reads, "races", {"race_id": ["70", "8"]}) == ["seventy"]


async def test_an_instant_key_is_found_by_any_spelling(reads: QueriedReads) -> None:
    for spelled in ("2026-11-03T23:59:00Z", "2026-11-03 23:59:00+00", "2026-11-03T23:59:00.000000+00:00"):
        assert await _found(reads, "snapshots", {"taken_at": [spelled]}) == ["night"], spelled
