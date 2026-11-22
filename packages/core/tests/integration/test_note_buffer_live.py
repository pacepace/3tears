"""Integration: a writer's buffered change notes are exactly the trigger's, written by one insert at its commit.

A writer that sets its transaction's change-note buffer (``note_buffer_open_sql``) has each
row's note written into its connection's buffer by the table's own trigger, and moves them into the ring with one
statement before its COMMIT (``note_flush_sql``). This checks, on the database the run names (Postgres by default; set
``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL to run it there):

- a copy, an upsert moving an invalidation column and a delete leave the very notes an unflagged write leaves, note
  for note, and the same one write mark;
- a write that sets no flag is noted row by row, as always;
- a flagged transaction that commits without its flush is refused (the guard's deferred key), and nothing it wrote
  commits;
- a restore after a NATS loss puts back from them what it puts back from a plain write's;
- a copy follower reading a commit's notes bracket finds the same keys, keyed, never the whole table;
- a trigger body from before the buffer (an older installer's, during a rollout) still notes a flagged write once;
- on a table shaped as a pod's bulk table, a 1000-row copy and upsert take at most a set multiple of the bare write
  (timed runs belong under measure-lock).
"""

from __future__ import annotations

import datetime as dt
import re
import time
import uuid
import json
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import asyncpg
import pytest
from threetears.nats.change_log import ChangeLogState

from threetears.core.collections.change_note_keeper import ChangeNoteKeeper
from threetears.core.collections.change_notes import NotedChange, noted_keys_page
from threetears.core.collections.epoch_change import EpochChange, read_epoch_change_body
from threetears.core.ddl.change_notes import (
    NOTE_RING_SIZE,
    NOTE_WHOLE_SETTING,
    RING_TABLE,
    note_buffer_open_sql,
    note_flush_sql,
    note_table,
    note_trigger_sql,
    trigger_function_body,
    note_function_name,
)
from threetears.core.ddl.write_marks import WRITE_MARKS_TABLE, write_mark_trigger_sql, write_marks_ddl
from threetears.core.tenancy import TENANT_BY_SCHEMA
from threetears.core.testing.calibrate import budget_problem
from threetears.core.testing.note_ring import install_note_ring

pytestmark = pytest.mark.integration

_TABLE = "results"
#: the buffer's guard, as a refused COMMIT's error names it
_GUARD = "threetears_note_buffer_flushed"
_KEY = ("race", "geo")
_ROWS = 300


@dataclass
class _Db:
    conn: asyncpg.Connection
    schema: str


async def _give_triggers(
    conn: asyncpg.Connection,
    schema: str,
    table: str,
    keys: tuple[str, ...],
    columns: tuple[str, ...],
    body: str | None = None,
) -> None:
    """the table's note function, its note triggers and its write-mark trigger, as the installers give them."""
    kinds = {
        str(row["c"]): str(row["t"])
        for row in await conn.fetch(
            "SELECT column_name::text c, data_type::text t FROM information_schema.columns "
            "WHERE table_schema = $1 AND table_name = $2",
            schema,
            table,
        )
    }
    wanted = body or trigger_function_body(
        notes_schema=schema, keys=keys, columns=columns, tenancy=TENANT_BY_SCHEMA, kinds=kinds
    )
    await conn.execute(
        f'CREATE OR REPLACE FUNCTION "{schema}"."{note_function_name(schema, table)}"() RETURNS trigger '
        f"LANGUAGE plpgsql AS $note${wanted}$note$"
    )
    present = {
        str(row["tgname"])
        for row in await conn.fetch(
            f"SELECT tgname::text FROM pg_trigger WHERE tgrelid = '{schema}.{table}'::regclass AND NOT tgisinternal"
        )
    }
    for name, statement in note_trigger_sql(notes_schema=schema, table_schema=schema, table=table).items():
        if name not in present:
            await conn.execute(statement)
    if "threetears_collection_write_mark" not in present:
        await conn.execute(write_mark_trigger_sql(marks_schema=schema, table_schema=schema, table=table))


@pytest.fixture
async def db(db_container: str) -> AsyncIterator[_Db]:
    """a noted table with one invalidation column, its ring and its write-mark queue in its schema."""
    schema = f"buffer_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_container, command_timeout=600)
    try:
        await conn.execute(f"CREATE SCHEMA {schema}")
        await install_note_ring(conn, schema)
        for statement in write_marks_ddl(schema):
            await conn.execute(statement)
        await conn.execute(
            f"CREATE TABLE {schema}.{_TABLE} (race TEXT, geo BIGINT, state TEXT, votes BIGINT, PRIMARY KEY (race, geo))"
        )
        await _give_triggers(conn, schema, _TABLE, _KEY, ("state",))
        yield _Db(conn=conn, schema=schema)
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


async def _clear(db: _Db) -> None:
    """an empty table, ring and queue (DELETE, so the truncate trigger leaves no note)."""
    await db.conn.execute(f"DELETE FROM {db.schema}.{_TABLE}")
    for index in range(NOTE_RING_SIZE):
        await db.conn.execute(f'DELETE FROM {db.schema}."{note_table(index)}"')
    await db.conn.execute(f'DELETE FROM {db.schema}."{WRITE_MARKS_TABLE}"')


async def _ring(db: _Db) -> tuple[list[tuple[Any, ...]], int]:
    """every note the ring holds, as what it says (table, keys, key text, columns, tenant), and the marks queued."""
    notes: list[tuple[Any, ...]] = []
    for index in range(NOTE_RING_SIZE):
        notes += [
            (row["table_name"], tuple(row["keys"] or ()), row["key_text"], row["columns"], row["tenant"])
            for row in await db.conn.fetch(
                f'SELECT table_name, keys, key_text, columns::text AS columns, tenant FROM {db.schema}."{note_table(index)}"'
            )
        ]
    marks = await db.conn.fetchval(f'SELECT count(*) FROM {db.schema}."{WRITE_MARKS_TABLE}"')
    return sorted(notes, key=repr), int(marks)


def _rows(state: str) -> list[tuple[Any, ...]]:
    return [(f"race-{n % 7}", n, f"{state}{n % 5}", n * 10) for n in range(_ROWS)]


async def _seed(db: _Db) -> None:
    """the table holding the rows the upsert and the delete change; then an empty ring and queue."""
    await _clear(db)
    await db.conn.copy_records_to_table(_TABLE, schema_name=db.schema, records=_rows("A"))
    for index in range(NOTE_RING_SIZE):
        await db.conn.execute(f'DELETE FROM {db.schema}."{note_table(index)}"')
    await db.conn.execute(f'DELETE FROM {db.schema}."{WRITE_MARKS_TABLE}"')


async def _copy(db: _Db) -> None:
    await db.conn.copy_records_to_table(_TABLE, schema_name=db.schema, records=_rows("A"))


async def _upsert(db: _Db) -> None:
    await db.conn.executemany(
        f"INSERT INTO {db.schema}.{_TABLE} (race, geo, state, votes) VALUES ($1, $2, $3, $4) "
        "ON CONFLICT (race, geo) DO UPDATE SET state = EXCLUDED.state, votes = EXCLUDED.votes",
        _rows("B"),
    )


async def _delete(db: _Db) -> None:
    await db.conn.execute(
        f"DELETE FROM {db.schema}.{_TABLE} WHERE (race, geo) IN (SELECT * FROM unnest($1::text[], $2::bigint[]))",
        [f"race-{n % 7}" for n in range(0, _ROWS, 2)],
        list(range(0, _ROWS, 2)),
    )


_WRITES: dict[str, tuple[bool, Callable[[_Db], Awaitable[None]]]] = {
    "copy": (False, _copy),
    "upsert": (True, _upsert),
    "delete": (True, _delete),
}


async def _written(
    db: _Db, write: Callable[[_Db], Awaitable[None]], *, seeded: bool, flagged: bool
) -> tuple[list[tuple[Any, ...]], int]:
    """the notes and marks one transaction of ``write`` leaves, flagged and flushed or as a plain write."""
    if seeded:
        await _seed(db)
    else:
        await _clear(db)
    async with db.conn.transaction():
        if flagged:
            await db.conn.execute(note_buffer_open_sql(db.schema))
        await write(db)
        if flagged:
            await db.conn.fetchval(note_flush_sql(db.schema))
    return await _ring(db)


@pytest.mark.parametrize("kind", sorted(_WRITES))
async def test_a_buffered_write_leaves_exactly_the_notes_and_mark_of_a_plain_one(db: _Db, kind: str) -> None:
    """Flips if a flagged write's notes differ from the trigger's own in any note (a key, its text, a column, the
    tenant), in number (one missing or written twice), or if its write marks differ."""
    seeded, write = _WRITES[kind]
    plain = await _written(db, write, seeded=seeded, flagged=False)
    buffered = await _written(db, write, seeded=seeded, flagged=True)
    notes, marks = plain
    # the plain write is today's: a note a changed row (and the old row's too where the upsert moved ``state``)
    assert len(notes) == {"copy": _ROWS, "upsert": 2 * _ROWS, "delete": _ROWS // 2}[kind]
    assert all(note[2] is not None for note in notes), "a keyed write leaves only keyed notes"
    assert marks == 1
    assert buffered == plain


async def test_a_write_with_no_flag_is_noted_row_by_row(db: _Db) -> None:
    """Flips if a write that never opened the buffer leaves fewer notes than rows (the trigger skipping without a
    writer to flush), or leaves any in the buffer."""
    await _clear(db)
    # the connection's buffer made by an earlier flagged transaction; this one sets no flag
    async with db.conn.transaction():
        await db.conn.execute(note_buffer_open_sql(db.schema))
        await db.conn.fetchval(note_flush_sql(db.schema))
    async with db.conn.transaction():
        await _copy(db)
    notes, marks = await _ring(db)
    assert len(notes) == _ROWS
    assert marks == 1
    assert await db.conn.fetchval("SELECT count(*) FROM pg_temp.threetears_note_buffer") == 0


async def test_a_flagged_transaction_cannot_commit_without_its_flush(db: _Db) -> None:
    """Flips if a transaction that buffered notes and never flushed them commits: its rows would then be in the table
    with no note, and a restore would leave them out."""
    await _clear(db)
    with pytest.raises(asyncpg.ForeignKeyViolationError, match=_GUARD):
        async with db.conn.transaction():
            await db.conn.execute(note_buffer_open_sql(db.schema))
            await _copy(db)
    assert await db.conn.fetchval(f"SELECT count(*) FROM {db.schema}.{_TABLE}") == 0
    assert (await _ring(db))[0] == []
    # the connection is usable after, and its next flagged transaction notes and commits
    assert (await _written(db, _copy, seeded=False, flagged=True))[0] != []


async def test_a_flagged_write_under_an_older_trigger_body_is_noted_once(db: _Db) -> None:
    """Flips if a writer that flags its transaction loses or doubles notes on a table whose trigger function an older
    installer wrote (no buffer branch), as a rollout leaves some tables for a while."""
    body = trigger_function_body(
        notes_schema=db.schema, keys=_KEY, columns=("state",), tenancy=TENANT_BY_SCHEMA, kinds=None
    )
    # an older body: every note straight into the ring, as the trigger wrote it before the buffer
    older = re.sub(
        r"IF current_setting\('threetears\.note_buffer', true\) = '1' THEN\n.*?\n    ELSE\n      (PERFORM .*?;)\n    END IF;",
        r"\1",
        body,
        flags=re.DOTALL,
    )
    assert "note_buffer" not in older and older.count("threetears_change_note_put") == 3
    plain = await _written(db, _upsert, seeded=True, flagged=False)
    await _give_triggers(db.conn, db.schema, _TABLE, _KEY, ("state",), body=older)
    assert await _written(db, _upsert, seeded=True, flagged=True) == plain


class _LostLog:
    """a lost change log that keeps every change a restore hands it (as test_change_note_restore_fold_live's)."""

    #: names no database on its records: the keeper reads this at construction (EpochGenerationSource.database)
    database = None

    def __init__(self) -> None:
        self.declared = False
        self.changes: list[NotedChange] = []

    async def log_state(self) -> ChangeLogState | None:
        if not self.declared:
            return None
        return ChangeLogState(first_seq=0, last_seq=0, bytes=0, max_bytes=1 << 40, incarnation="restored-log")

    async def epoch_at(self, seq: int) -> EpochChange | None:
        raise AssertionError(f"a restore reads no record of the lost log ({seq})")

    async def start_log(self) -> str:
        raise AssertionError("a restore declares its log through begin_restore")

    async def begin_restore(self, lost: str) -> str | None:
        self.declared = True
        return "restored-log"

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

    async def put_back(self, changes: Any, *, incarnation: str, lost: str) -> int:
        raise AssertionError("a restore labels the log it fills; nothing is put back beside it")


async def _restored(db: _Db, db_container: str) -> list[tuple[str, str | None, bool, list[Any]]]:
    """what a restore after a NATS loss puts back from the ring as it is: each change's table, tenant, whether whole,
    and its rows as a follower reads them."""
    await db.conn.execute(f"UPDATE {db.schema}.{RING_TABLE} SET log_incarnation = 'lost-log'")  # noqa: S608
    pool = await asyncpg.create_pool(db_container, min_size=1, max_size=2, command_timeout=600)
    assert pool is not None
    try:
        log = _LostLog()
        assert (await ChangeNoteKeeper(pool, notes_schema=db.schema, log=log).run_pass()).restored == "restored-log"
    finally:
        await pool.close()
    restored = []
    for change in log.changes:
        rows = sorted(
            (row.ids, json.dumps(row.columns, sort_keys=True))
            for body in change.bodies
            for row in read_epoch_change_body(body, change.table, "t:1").rows
        )
        restored.append((change.table, change.tenant, change.whole_table, rows))
    return sorted(restored, key=repr)


async def test_a_restore_after_a_nats_loss_puts_back_a_buffered_write_as_a_plain_one(
    db: _Db, db_container: str
) -> None:
    """Flips if a restore from a flagged write's notes puts back other rows, other columns or a whole-table change
    where the plain write's notes put back its rows: the restore must be today's, row for row."""
    plain_notes = await _written(db, _upsert, seeded=True, flagged=False)
    plain = await _restored(db, db_container)
    await db.conn.execute(f"UPDATE {db.schema}.{RING_TABLE} SET log_incarnation = NULL")  # noqa: S608
    assert await _written(db, _upsert, seeded=True, flagged=True) == plain_notes
    buffered = await _restored(db, db_container)
    assert [whole for _, _, whole, _ in plain] == [False]
    # the upsert's every row, under its new and its old ``state``
    assert len(plain[0][3]) == 2 * _ROWS
    assert buffered == plain


async def _noted_keys(db: _Db, *, flagged: bool) -> tuple[list[tuple[str, ...]], bool]:
    """the keys a copy follower reads from a commit's notes bracket (the broker's ``l3.noted_keys``), read just before
    its COMMIT on the committing connection as the broker reads it, after the flush."""
    await _seed(db)
    async with db.conn.transaction():
        if flagged:
            await db.conn.execute(note_buffer_open_sql(db.schema))
        await _upsert(db)
        if flagged:
            await db.conn.fetchval(note_flush_sql(db.schema))
        bracket = await db.conn.fetchrow("SELECT now() AS since, clock_timestamp() AS until")
    keys: list[tuple[str, ...]] = []
    whole = False
    cursor = None
    while True:
        page = await noted_keys_page(
            db.conn,
            notes_schema=db.schema,
            table=_TABLE,
            tenant_token=f"s-{db.schema}",
            noted_between=(bracket["since"], bracket["until"]),
            cursor=cursor,
            limit=100,
            max_bytes=1 << 20,
        )
        assert not isinstance(page, str), page
        keys += [tuple(key) for key in page.keys]
        whole = whole or page.whole_table
        if page.next is None:
            return sorted(set(keys)), whole
        cursor = page.next


async def test_followers_catch_up_keyed_from_a_buffered_write(db: _Db) -> None:
    """Flips if a copy follower reading a flagged commit's notes bracket finds other keys than the plain commit's, or
    is told the table changed whole: it must evict exactly the written rows, never drop or read the table."""
    plain, plain_whole = await _noted_keys(db, flagged=False)
    buffered, buffered_whole = await _noted_keys(db, flagged=True)
    assert not plain_whole and not buffered_whole
    assert len(plain) == _ROWS
    assert buffered == plain


# -- the speed budget: a pod's bulk table, shaped as the reports pod's latest_results_candidates_by_id

#: its columns (34), its 7-column text key and its by-state index, as its pod declares them
_POD_COLUMNS = (
    "election_type text, race_type text, office_key text, candidate text, candidate_party_major text, "
    "precincts_reporting bigint, precincts_total bigint, metric_desc text, metric numeric(38,12), state text, "
    "fipscode text, incumbent boolean, uncontested text, winner text, winnerdatetime timestamptz, racecallstatus text, "
    "pollclosingtime timestamptz, race_full text, district_full text, join_geo_level text, join_state_code text, "
    "join_district text, join_geo_id text, cook_rating text, candidate_id text, first_name text, source_version bigint, "
    "date_created timestamptz, date_updated timestamptz, seatname text, expected_turnout numeric(38,12), "
    "tabulationstatus text, metric_full numeric(38,20), expected_turnout_full numeric(38,20)"
)
_POD_KEY = ("office_key", "candidate", "metric_desc", "join_geo_level", "join_district", "join_geo_id", "candidate_id")
_POD_ROWS = 1000

#: the most times the bare write a 1000-row copy or upsert with its notes and mark may take. WHY 5 (2026-11-21,
#: YugabyteDB 2026.1, this table, median of 5): a copy 29 ms bare, 471 ms with a ring insert a note (16x), 103 ms
#: buffered (3.5x); an upsert moving an invalidation column 203 ms bare, 1402 ms (6.9x), 684 ms (3.4x). The note a row
#: into the ring is over it at either; buffered is under it with room for a run's noise
_WRITE_BUDGET = 5.0

#: timed runs of each side
_RUNS = 5


def _pod_rows(state: str, version: int) -> list[tuple[Any, ...]]:
    """the table's rows, ``join_state_code`` as given (the upsert moves it), in its columns' order."""
    now = dt.datetime(2026, 11, 3, tzinfo=dt.UTC)
    return [
        (
            "general",
            "G",
            f"race-{n // 50}",
            f"cand-{n}",
            "DEM",
            n,
            n + 10,
            "votes",
            Decimal(n) * Decimal("1.5"),
            f"S{n % 50}",
            f"{n % 3000:05d}",
            n % 2 == 0,
            "N",
            "",
            now,
            "open",
            now,
            f"Race {n // 50}",
            "D1",
            "county",
            f"{state}{n % 50}",
            "",
            f"{n % 3000:05d}",
            "toss",
            str(n),
            f"first-{n}",
            version,
            now,
            now,
            "seat",
            Decimal(n) * Decimal("2.5"),
            "active",
            Decimal(n) * Decimal("1.5"),
            Decimal(n) * Decimal("2.5"),
        )
        for n in range(_POD_ROWS)
    ]


@pytest.mark.loop_timing
@pytest.mark.yugabyte_only
async def test_a_pod_tables_buffered_copy_and_upsert_stay_within_the_budget(db: _Db) -> None:
    """Flips if a 1000-row copy, or a 1000-row upsert moving an invalidation column, on a pod's bulk table takes more
    than _WRITE_BUDGET times the same write on the table with no trigger: the note a row inserted into the ring (the
    trigger without its writer's buffer) is 16x and 6.9x on YugabyteDB."""
    conn, schema = db.conn, db.schema
    names = [column.split()[0] for column in _POD_COLUMNS.split(", ")]
    for table, noted in (("bare_pod", False), ("noted_pod", True)):
        await conn.execute(f"CREATE TABLE {schema}.{table} ({_POD_COLUMNS}, PRIMARY KEY ({', '.join(_POD_KEY)}))")
        await conn.execute(f"CREATE INDEX {table}_by_state ON {schema}.{table} (join_state_code, join_geo_id)")
        if noted:
            # a pod table as the hub declares it: its key, the tenant by its schema, no invalidation column; the
            # upsert's moved column is declared too, so it leaves the old row's note as well
            await _give_triggers(conn, schema, table, _POD_KEY, ("join_state_code",))
    # the hub's upsert of more than a few rows: copied into a stage (a temp table), merged by one statement
    await conn.execute(f"CREATE TEMP TABLE pod_stage (LIKE {schema}.bare_pod)")
    merge = (
        "WITH staged AS (DELETE FROM pod_stage RETURNING *) INSERT INTO {table} SELECT * FROM staged ON CONFLICT ("
        + ", ".join(_POD_KEY)
        + ") DO UPDATE SET "
        + ", ".join(f"{name} = EXCLUDED.{name}" for name in names if name not in _POD_KEY)
    )

    async def reset(table: str, seeded: bool) -> None:
        # untimed, and noted whole (one note), so neither side's setup pays a note a row
        async with conn.transaction():
            await conn.execute("SELECT set_config($1, '1', true)", NOTE_WHOLE_SETTING)
            await conn.execute(f"DELETE FROM {schema}.{table}")
            if seeded:
                await conn.copy_records_to_table(table, schema_name=schema, records=_pod_rows("S", 1))

    async def copy(table: str, flagged: bool) -> float:
        await reset(table, seeded=False)
        began = time.perf_counter()
        async with conn.transaction():
            if flagged:
                await conn.execute(note_buffer_open_sql(schema))
            await conn.copy_records_to_table(table, schema_name=schema, records=rows)
            if flagged:
                await conn.fetchval(note_flush_sql(schema))
        return time.perf_counter() - began

    async def upsert(table: str, flagged: bool) -> float:
        await reset(table, seeded=True)
        began = time.perf_counter()
        async with conn.transaction():
            if flagged:
                await conn.execute(note_buffer_open_sql(schema))
            await conn.copy_records_to_table("pod_stage", records=moved)
            await conn.execute(merge.format(table=f"{schema}.{table}"))
            if flagged:
                await conn.fetchval(note_flush_sql(schema))
        return time.perf_counter() - began

    rows, moved = _pod_rows("S", 1), _pod_rows("T", 2)
    for what, write in (("copy", copy), ("upsert", upsert)):
        # turn about, best of each: a busy moment lands on both sides alike
        bare = path = float("inf")
        for _ in range(_RUNS):
            bare = min(bare, await write("bare_pod", False))
            path = min(path, await write("noted_pod", True))
        problem = budget_problem(f"a 1000-row {what} with its notes", path, bare, factor=_WRITE_BUDGET)
        assert problem is None, problem
