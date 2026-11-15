"""Integration: a restore's keyed notes, read as ``COPY`` text and folded in a worker process, put back the changes the
old fold on the event loop put back from the same notes.

The keeper's restore (``_StreamedChanges._stream``) reads the keyed notes with ``COPY ... TO STDOUT`` CSV and hands
them to :mod:`threetears.core.change_note_fold` a page at a time (L1, 2026-11-15). This seeds the ring with awkward
notes -- text keys holding quotes, commas, backslashes and line ends, two tenants, keys noted with several
invalidation columns, a table past a chunk, tables loaded whole -- and holds every restored change, its records read
as a follower reads them, against the old fold's (each table and tenant's distinct rows, cut into chunks only where the
key changes) over the same notes read as rows.

Against the session ``db_container``; set ``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL to run it there.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterable, AsyncIterator, Sequence
from itertools import groupby
from typing import Any

import asyncpg
import pytest
from threetears.nats.change_log import ChangeLogState

from threetears.core.collections.change_notes import ChangeNoteKeeper, NotedChange
from threetears.core.collections.epoch_change import EpochChange, read_epoch_change_body
from threetears.core.ddl.change_notes import RING_TABLE, note_table
from threetears.core.testing.note_ring import install_note_ring

pytestmark = pytest.mark.integration

_LOST = "lost-log"
_RESTORED = "restored-log"

#: the restore's chunk (change_notes._RESTORE_CHUNK_ROWS)
_CHUNK_ROWS = 5000

#: one table's distinct keys: past two chunks
_BIG = 12_345


class _Log:
    """a lost change log that keeps every change a restore hands it."""

    def __init__(self) -> None:
        self.declared = False
        self.changes: list[NotedChange] = []

    async def log_state(self) -> ChangeLogState | None:
        if not self.declared:
            return None
        return ChangeLogState(first_seq=0, last_seq=0, bytes=0, max_bytes=1 << 40, incarnation=_RESTORED)

    async def epoch_at(self, seq: int) -> EpochChange | None:
        raise AssertionError(f"a restore reads no record of the lost log ({seq})")

    async def start_log(self) -> str:
        raise AssertionError("a restore declares its log through begin_restore")

    async def begin_restore(self, lost: str) -> str | None:
        self.declared = True
        return _RESTORED

    async def finish_restore(
        self,
        changes: Sequence[NotedChange] | AsyncIterable[NotedChange],
        *,
        incarnation: str,
        lost: str,
        covered_from: int | None,
    ) -> int:
        assert isinstance(changes, AsyncIterable)
        async for change in changes:
            self.changes.append(change)
        return len(self.changes)


@pytest.fixture
async def ring(db_container: str) -> AsyncIterator[tuple[asyncpg.Pool, str]]:
    schema = f"fold_{uuid.uuid4().hex[:8]}"
    pool = await asyncpg.create_pool(db_container, min_size=1, max_size=2, command_timeout=600)
    assert pool is not None
    try:
        await pool.execute(f"CREATE SCHEMA {schema}")
        await install_note_ring(pool, schema)
        yield pool, schema
    finally:
        await pool.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await pool.close()


def _awkward_notes() -> list[tuple[str, str | None, list[str], str | None]]:
    """keyed notes as (table, tenant, keys, columns as JSON text), several of each row."""
    notes: list[tuple[str, str | None, list[str], str | None]] = []
    for table in ("plain", 'quote"d,comma', "line\nend", "back\\slash", "ünï"):
        for tenant in (None, "agent_1", 'te"n,\nant'):
            for key in range(40):
                keys = [f"k{key:03d}", 'x"y,z\n\\w' if key % 4 == 0 else "", "ü"]
                notes.append((table, tenant, keys, None))
                notes.append((table, tenant, keys, None))
                if key % 3 == 0:
                    notes.append((table, tenant, keys, json.dumps({"owner": f'o{key}\n"', "gone": None})))
                    notes.append((table, tenant, keys, json.dumps({"owner": f"p{key}", "gone": "x"})))
    notes.extend(("big", None, [f"b{key:06d}"], None) for key in range(_BIG))
    # every third key of the big table noted with two column sets: chunks cut only where the key changes
    notes.extend(("big", None, [f"b{key:06d}"], json.dumps({"c": "1"})) for key in range(0, _BIG, 3))
    return notes


#: tables (and the tenant) a whole-table note covers: their keyed notes are not restored
_LOADED = [("plain", "agent_1"), ("ünï", None)]


async def _seeded(pool: asyncpg.Pool, schema: str) -> None:
    notes = f"{schema}.{note_table(0)}"
    rows = [(table, keys, "\x1f".join(keys), columns, tenant) for table, tenant, keys, columns in _awkward_notes()]
    await pool.executemany(
        f"INSERT INTO {notes} (table_name, keys, key_text, columns, tenant) VALUES ($1, $2, $3, $4::jsonb, $5)",  # noqa: S608
        rows,
    )
    await pool.executemany(
        f"INSERT INTO {notes} (table_name, tenant) VALUES ($1, $2)",  # noqa: S608 - test identifiers
        _LOADED,
    )


def _reference(rows: Sequence[Any]) -> list[tuple[str, str | None, list[Any]]]:
    """the old fold over the notes read as rows: byte order, each distinct (key, columns) once, each table and
    tenant's rows cut into chunks only where the key changes once a chunk holds the chunk's rows."""

    def ordered(row: Any) -> tuple[bytes, bool, bytes, bytes]:
        # the read's ORDER BY ... COLLATE "C": byte order, a NULL tenant last
        tenant = row["tenant"]
        return (row["table_name"].encode(), tenant is None, (tenant or "").encode(), row["key_text"].encode())

    covered = {(table, tenant) for table, tenant in _LOADED}
    kept = sorted(
        {
            (row["table_name"], row["tenant"], row["key_text"], tuple(row["keys"]), row["columns"]): row
            for row in rows
            if (row["table_name"], row["tenant"]) not in covered and (row["table_name"], None) not in covered
        }.values(),
        key=ordered,
    )
    changes: list[tuple[str, str | None, list[Any]]] = []
    for (table, tenant), group in groupby(kept, key=lambda row: (row["table_name"], row["tenant"])):
        held: list[Any] = []
        last: str | None = None
        for row in group:
            if row["key_text"] != last and len(held) >= _CHUNK_ROWS:
                changes.append((table, tenant, sorted(held)))
                held = []
            last = row["key_text"]
            columns = None if row["columns"] is None else json.loads(row["columns"])
            held.append((list(row["keys"]), json.dumps(columns, sort_keys=True)))
        changes.append((table, tenant, sorted(held)))
    return changes


def _restored(changes: Sequence[NotedChange]) -> list[tuple[str, str | None, list[Any]]]:
    """each keyed change's rows, read from its records as a follower reads them (sorted within the change: the order
    of one key's column sets is the database's)."""
    restored = []
    for change in changes:
        if change.whole_table:
            continue
        assert change.bodies and not change.rows
        rows = [
            (row.ids, json.dumps(row.columns, sort_keys=True))
            for body in change.bodies
            for row in read_epoch_change_body(body, change.table, "t:1").rows
        ]
        restored.append((change.table, change.tenant, sorted(rows)))
    return restored


async def test_a_restore_folded_in_its_worker_puts_back_the_old_folds_changes(
    ring: tuple[asyncpg.Pool, str],
) -> None:
    """Flips if a restore's worker fold names other rows, other columns or tenants, another order of tables, cuts
    its chunks elsewhere than the old fold, or drops a whole-table change; on the database's own COPY text."""
    pool, schema = ring
    await _seeded(pool, schema)
    await pool.execute(f"UPDATE {schema}.{RING_TABLE} SET log_incarnation = $1", _LOST)  # noqa: S608
    log = _Log()
    assert (await ChangeNoteKeeper(pool, notes_schema=schema, log=log).run_pass()).restored == _RESTORED
    rows = await pool.fetch(
        f"SELECT table_name, tenant, key_text, keys, columns::text AS columns FROM {schema}.{note_table(0)} "  # noqa: S608
        "WHERE key_text IS NOT NULL"
    )
    expected = _reference(rows)
    assert _restored(log.changes) == expected
    whole = sorted((change.table, change.tenant or "") for change in log.changes if change.whole_table)
    assert whole == sorted((table, tenant or "") for table, tenant in _LOADED)
    # the big table came back in several chunks, each cut where the old fold cut it
    assert sum(1 for table, _, _ in expected if table == "big") > 2
