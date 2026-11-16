"""Integration (measurement): the first reads of a ring slot switched back into right after its emptying.

The keeper empties the idle note table with batched deletes (never TRUNCATE). Each deleted note leaves debris: a
tombstone on YugabyteDB until a compaction, a dead row (and dead index entries) on Postgres until a vacuum. Nothing
checks that the debris is gone before the ring switches back into the slot. This fills one slot to the switch size,
empties it as the keeper does, writes the notes a slot holds just after a switch into it, and times the slot's reads
against two never-filled slots holding the same notes (the second one the A/A gap).

A measurement, not a gate: it asserts only that every read answers right, and runs only when
``THREETEARS_MEASURE_REUSED_SLOT=1`` (it fills 393,216 notes, minutes on YugabyteDB). Run it under the measure lock
through the integration runner; ``R1_OUT`` names a JSON file for the table, ``R1_LATER`` (seconds, comma-separated)
measures again after each wait, ``R1_FILL`` changes the fill. Its result is recorded at the code, above
``_CURRENT_COUNT_SECONDS`` in ``threetears.core.collections.change_note_keeper``. Since the keeper vacuums an emptied slot
on Postgres (Pace 2026-11-15), ``vacuum_s`` is that VACUUM's seconds; the result with it is recorded above
``ChangeNoteKeeper._vacuum_emptied``.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest

from threetears.core.collections.change_note_keeper import ChangeNoteKeeper
from threetears.core.collections.change_notes import NotedKeysPage, NotedRows, noted_keys_page, noted_rows_since
from threetears.core.collections.epoch_change import ChangedRow, EpochChange
from threetears.core.ddl.change_notes import RING_TABLE, note_table
from threetears.core.testing.note_ring import install_note_ring

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("THREETEARS_MEASURE_REUSED_SLOT") != "1",
        reason="a measurement: THREETEARS_MEASURE_REUSED_SLOT=1",
    ),
]

#: a slot's notes at its switch: the 64 MiB log's restore budget (75 %) over two, over 64 bytes a note
_FILL = int(os.environ.get("R1_FILL", "393216"))
_TABLES = 2000
_NEW = 20_000
_ASKED = "results_asked"
_COMMIT = 5000
_RUNS = 9


class _NoLog:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(name)


class _Timed:
    """the keeper's pool, each statement it sends timed (the reused slot's VACUUM on Postgres, Pace 2026-11-15)."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
        self.took: list[tuple[str, float]] = []

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        began = time.perf_counter()
        try:
            return await self._pool.fetch(query, *args)
        finally:
            self.took.append((query, time.perf_counter() - began))

    async def execute(self, query: str, *args: Any) -> str:
        began = time.perf_counter()
        try:
            return await self._pool.execute(query, *args)
        finally:
            self.took.append((query, time.perf_counter() - began))

    def acquire(self) -> Any:
        return self._pool.acquire()

    async def fetchval(self, query: str, *args: Any) -> Any:
        return await self._pool.fetchval(query, *args)


@dataclass
class _Slot:
    name: str
    schema: str
    pool: asyncpg.Pool
    keeper: ChangeNoteKeeper


async def _ring(url: str, name: str, *, fill: bool, night: datetime) -> tuple[_Slot, dict[str, float]]:
    schema = f"r1_{name}_{uuid.uuid4().hex[:6]}"
    pool = await asyncpg.create_pool(url, min_size=1, max_size=2, command_timeout=3600)
    assert pool is not None
    await pool.execute(f"CREATE SCHEMA {schema}")
    await install_note_ring(pool, schema)
    timed = _Timed(pool)
    keeper = ChangeNoteKeeper(
        timed, notes_schema=schema, log=_NoLog(), empty_pause=timedelta(0), empty_budget=timedelta(hours=2)
    )
    notes = f"{schema}.{note_table(0)}"
    emptied: dict[str, float] = {}
    if fill:
        # the slot's last tenure: a switch's worth of notes, the same rows noted again and again through the night
        await pool.execute(
            f"INSERT INTO {notes} (table_name, keys, key_text, noted_at) "  # noqa: S608
            f"SELECT 'table_' || (g % {_TABLES}), ARRAY[(g % 50000)::text], (g % 50000)::text, "
            "$1::timestamptz + make_interval(secs => g * 0.02) FROM generate_series(1, $2) g",
            night,
            _FILL,
        )
        await pool.execute(
            f"INSERT INTO {notes} (table_name, keys, key_text, noted_at) "  # noqa: S608
            "SELECT $1, ARRAY['old-' || g], 'old-' || g, $2::timestamptz + make_interval(secs => g) "
            "FROM generate_series(1, 2000) g",
            _ASKED,
            night,
        )
        await pool.execute(f"ANALYZE {notes}")
        began = time.perf_counter()
        done, deleted = await keeper.empty_slot(0)
        emptied["emptied_s"] = round(time.perf_counter() - began, 3)
        assert done and deleted == _FILL + 2000
        emptied["vacuum_s"] = round(sum(took for query, took in timed.took if query.startswith("VACUUM")), 3)
    return _Slot(name=name, schema=schema, pool=pool, keeper=keeper), emptied


async def _refill(slot: _Slot, now: datetime) -> tuple[datetime, tuple[datetime, datetime]]:
    """the notes a slot holds a few minutes after the ring switched into it, identical in every slot."""
    notes = f"{slot.schema}.{note_table(0)}"
    start = now - timedelta(minutes=20)
    await slot.pool.execute(
        f"INSERT INTO {notes} (table_name, keys, key_text, noted_at) "  # noqa: S608
        f"SELECT 'table_' || (g % {_TABLES}), ARRAY[g::text], g::text, "
        "$1::timestamptz + make_interval(secs => g * 0.05) FROM generate_series(1, $2) g",
        start,
        _NEW,
    )
    since = now - timedelta(minutes=1)
    await slot.pool.execute(
        f"INSERT INTO {notes} (table_name, keys, key_text, noted_at) "  # noqa: S608
        "SELECT $1, ARRAY['new-' || g], 'new-' || g, $2::timestamptz + make_interval(secs => g * 0.001) "
        "FROM generate_series(1, 500) g",
        _ASKED,
        since,
    )
    bracket_from = now - timedelta(seconds=30)
    await slot.pool.execute(
        f"INSERT INTO {notes} (table_name, keys, key_text, noted_at) "  # noqa: S608
        "SELECT $1, ARRAY['c-' || g], 'c-' || g, $2::timestamptz + make_interval(secs => g * 0.0002) "
        "FROM generate_series(1, $3) g",
        _ASKED,
        bracket_from,
        _COMMIT,
    )
    await slot.pool.execute(
        f"UPDATE {slot.schema}.{RING_TABLE} SET current_ring = 0, switch_no = 3, switched_at = $1, "  # noqa: S608
        "previous_switched_at = $2, emptied = FALSE WHERE id = 1",
        now - timedelta(minutes=25),
        now - timedelta(hours=2),
    )
    return since, (bracket_from, bracket_from + timedelta(seconds=2))


def _records() -> list[EpochChange]:
    keyed = [
        EpochChange(table=f"table_{g % _TABLES}", token=f"t{g}", rows=[ChangedRow(ids=[str(g)])]) for g in range(1, 33)
    ]
    rowless = [EpochChange(table=f"table_{g}", token=f"u{g}", keys_unknown=True) for g in range(32)]
    return keyed + rowless


def _reads(slot: _Slot, since: datetime, bracket: tuple[datetime, datetime]) -> dict[str, Callable[[], Awaitable[Any]]]:
    notes = f"{slot.schema}.{note_table(0)}"
    records = _records()

    async def rows_since() -> None:
        found = await noted_rows_since(slot.pool, notes_schema=slot.schema, table=_ASKED, since=since, cap=10_000)
        assert isinstance(found, NotedRows) and len(found.rows) == 500 + _COMMIT, found

    async def keys_page() -> None:
        page = await noted_keys_page(
            slot.pool,
            notes_schema=slot.schema,
            table=_ASKED,
            tenant_token=None,
            noted_between=bracket,
            cursor=None,
            limit=5000,
            max_bytes=1 << 30,
        )
        assert isinstance(page, NotedKeysPage) and len(page.keys) == _COMMIT, page

    async def scrub_look() -> None:
        ring = await slot.keeper.ring()
        found = await slot.keeper.notes_found(ring, records)
        assert len(found[0].any_note) == 32, found[0].any_note

    async def notes_full() -> None:
        # the keeper's count every twentieth of the fence (_current_notes_full), at the switch's note count
        n = await slot.pool.fetchval(
            f"SELECT count(*) FROM (SELECT 1 FROM {notes} LIMIT $1) n",  # noqa: S608
            _FILL,
        )
        assert n == min(_FILL, _NEW + 500 + _COMMIT)

    async def full_scan() -> None:
        # the restore's read of the coverage, the current slot's half
        n = await slot.pool.fetchval(f"SELECT count(*), max(noted_at) FROM {notes}")  # noqa: S608
        assert n == _NEW + 500 + _COMMIT

    return {
        "noted_rows_since": rows_since,
        "noted_keys_page": keys_page,
        "scrub_notes_found": scrub_look,
        "notes_full_count": notes_full,
        "full_scan_count": full_scan,
    }


async def _timed(call: Callable[[], Awaitable[Any]]) -> float:
    began = time.perf_counter()
    await call()
    return time.perf_counter() - began


async def _measure(slots: list[_Slot], reads: list[dict[str, Callable[[], Awaitable[Any]]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for kind in reads[0]:
        first = [await _timed(r[kind]) for r in reads]
        best = [float("inf")] * len(slots)
        for _ in range(_RUNS):
            for i, r in enumerate(reads):
                best[i] = min(best[i], await _timed(r[kind]))
        result[kind] = {
            "first_ms": {s.name: round(t * 1000, 2) for s, t in zip(slots, first, strict=True)},
            "best_ms": {s.name: round(t * 1000, 2) for s, t in zip(slots, best, strict=True)},
            "reused_over_fresh": round(best[0] / best[1], 2),
            "aa_fresh2_over_fresh": round(best[2] / best[1], 2),
        }
    return result


@pytest.fixture
async def slots(db_container: str) -> AsyncIterator[tuple[list[_Slot], dict[str, Any]]]:
    night = datetime.now(UTC) - timedelta(hours=6)
    made: list[_Slot] = []
    try:
        reused, emptied = await _ring(db_container, "reused", fill=True, night=night)
        made.append(reused)
        for name in ("fresh", "fresh2"):
            slot, _ = await _ring(db_container, name, fill=False, night=night)
            made.append(slot)
        yield made, emptied
    finally:
        for slot in made:
            await slot.pool.execute(f"DROP SCHEMA IF EXISTS {slot.schema} CASCADE")
            await slot.pool.close()


async def test_measure_a_reused_slots_first_reads(slots: tuple[list[_Slot], dict[str, Any]]) -> None:
    made, info = slots
    now = await made[0].pool.fetchval("SELECT now()")
    marks = [await _refill(slot, now) for slot in made]
    reads = [_reads(slot, since, bracket) for slot, (since, bracket) in zip(made, marks, strict=True)]
    out: dict[str, Any] = {
        "engine": "yb" if "-YB-" in await made[0].pool.fetchval("SELECT version()") else "pg",
        "fill": _FILL,
        **info,
    }
    out["at_once"] = await _measure(made, reads)
    for wait in [int(w) for w in os.environ.get("R1_LATER", "").split(",") if w]:
        await asyncio.sleep(wait)
        out[f"after_{wait}s"] = await _measure(made, reads)
    path = os.environ.get("R1_OUT")
    if path:
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
