"""A lost change log is rebuilt from the versioned tables' own versions since X, never the ring; the scrub proves X held.

Append-only L3 chunk 5 (``design-append-only-l3.md`` decision 3). The keeper's restore finds every versioned table by
the restore declaration its ``(scope, epoch)`` index carries (``TableSchema.create_statements`` writes it), the scopes a
stamped write touched since X from the scope epochs table's ``last_epoch``, and reads each scope's versions, gone
markers and run markers with ``epoch >= X``, X = (T - M) << 15 from the newest mark at or before the ring's floor. The
pool here is a stand-in answering exactly the statements a restore sends (the ring holds no note, so whatever comes
back came from the tables), each one recorded: every read of the versioned table binds its scope and its epoch floor,
and the no-whole-reads rule judges each keyed. The live plan check is ``test_table_restore_plans_live``.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from threetears.core.collections.change_note_keeper import ChangeNoteKeeper
from threetears.core.collections.epoch_change import ChangedRow, EpochChange
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column, TableSchema
from threetears.core.collections.scope_epochs import scope_epochs_schema
from threetears.core.ddl.change_notes import MARKS_TABLE, RING_TABLE
from threetears.core.ddl.row_epoch import (
    BUCKET_COLUMN,
    RESTORE_BUCKETS,
    ROW_STAMPED,
    bucket_epoch_index_name,
    restore_floor,
    row_versions,
)
from threetears.core.ddl.run_markers import RUN_MARKERS_TABLE
from threetears.core.eviction import Declared, Policy
from threetears.core.tenancy import TENANT_BY_SCHEMA, schema_tenant
from threetears.core.testing.kv import FakeNatsClient
from threetears.enforcement.no_whole_reads.schema import collect_schema
from threetears.enforcement.no_whole_reads.sql import Reader, judge_sql
from threetears.epoch import EPOCH_CHANGES_BUCKET, EpochGenerationSource
from threetears.nats import set_default_namespace

_SCHEMA = "rpt"
_TABLE = "restore_results"
_EPOCHS = "restore_epochs"
_SCOPE = "join_state_code"
_RING = "ring"

#: the versioned table a consumer declares: its key, its scope, the scope epochs table naming its states
_RESULTS = TableSchema(
    name=_TABLE,
    primary_key=(_SCOPE, "run_id", "candidate"),
    columns=[
        Column(_SCOPE, STRING_TYPE),
        Column("run_id", STRING_TYPE),
        Column("candidate", STRING_TYPE),
        Column("votes", BIGINT_TYPE),
    ],
    tenancy=TENANT_BY_SCHEMA,
    row_epoch=row_versions(_SCOPE, epochs=_EPOCHS, scope_prefix="state:"),
    l1_eviction=Declared(Policy.EVICTABLE, "a test's versioned table"),
)

#: the restore read the (scope, epoch) index names as its purpose
_READ = "packages/core/src/threetears/core/collections/table_restore.py::_scope_versions"

_STAMPED_TABLE = "restore_runs"

#: a stamped table a consumer declares (``lsf_runs``' shape): one row a key, updated in place
_RUNS = TableSchema(
    name=_STAMPED_TABLE,
    primary_key=(_SCOPE, "run_id"),
    columns=[Column(_SCOPE, STRING_TYPE), Column("run_id", STRING_TYPE), Column("rows", BIGINT_TYPE)],
    tenancy=TENANT_BY_SCHEMA,
    row_epoch=ROW_STAMPED,
    l1_eviction=Declared(Policy.EVICTABLE, "a test's stamped table"),
)

#: the restore read the (bucket, epoch) index names as its purpose
_STAMPED_READ = "packages/core/src/threetears/core/collections/table_restore.py::_bucket_rows"


@pytest.fixture(autouse=True)
def _namespace() -> None:
    set_default_namespace("tablerestore")


def _declaration(schema: TableSchema = _RESULTS) -> str:
    """the restore declaration the table's DDL gives its restore index, as the catalog then holds it."""
    [comment] = [statement for statement in schema.create_statements(yugabyte=True) if statement.startswith("COMMENT")]
    found = re.search(r"IS '(.*)'$", comment)
    assert found is not None, comment
    return found.group(1).replace("''", "'")


class _Conn:
    """the ring stream's connection: its transaction, and a ring that holds no note."""

    def __init__(self, pool: _Pool) -> None:
        self._pool = pool

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        yield

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        self._pool.sent.append((query, args))
        return []

    async def copy_from_query(self, query: str, *args: Any, output: Any, **_: Any) -> str:
        self._pool.sent.append((query, args))
        return "COPY 0"


class _Pool:
    """the L3 a restore reads: the ring row and its marks, the catalog's one versioned table, its scope epochs, its
    versions and its run markers. Every statement is recorded with its arguments."""

    def __init__(self, lost: str, now: datetime, marks: list[dict[str, Any]]) -> None:
        self.sent: list[tuple[str, tuple[Any, ...]]] = []
        self.ring: dict[str, Any] = {
            "current_ring": 0,
            "switch_no": 3,
            "switched_at": now - timedelta(minutes=5),
            "switched_seq": None,
            "floor": f"{lost}:0",
            "fence_seq": None,
            "emptied": True,
            "log_incarnation": lost,
            "fence_seconds": 600.0,
            "wait_seconds": 600.0,
            "now": now,
        }
        self.marks = marks
        #: (scope, last_epoch) of the scope epochs table
        self.scopes: list[tuple[str, int | None]] = []
        #: (state, run, candidate, epoch, gone) of the versioned table
        self.versions: list[tuple[str, str, str, int, bool]] = []
        #: (scope, prefix, epoch) of the run-marker table, all of the versioned table
        self.run_markers: list[tuple[str, str, int]] = []
        #: the tables the owner's scrub judges as its own
        self.own: list[str] = []
        #: the catalog's restore declarations: (table, declaration)
        self.catalog: list[tuple[str, str]] = [(_TABLE, _declaration())]
        #: (state, run, bucket, epoch) of the stamped table
        self.stamped: list[tuple[str, str, int, int]] = []

    @staticmethod
    def _floor(args: tuple[Any, ...]) -> int:
        # the epoch floor is the one bound epoch-sized integer (a LIMIT is small)
        return next(arg for arg in args if isinstance(arg, int) and arg > 1 << 40)

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        self.sent.append((query, args))
        if f'"{RING_TABLE}"' in query and query.startswith("SELECT"):
            return [dict(self.ring)]
        if "pg_description" in query:
            return [
                {"table_schema": _SCHEMA, "table_name": table, "declared": declared, "run_markers": True}
                for table, declared in self.catalog
            ]
        if MARKS_TABLE in query:
            return [mark for mark in self.marks if mark["log_incarnation"] in args]
        if f'"{_EPOCHS}"' in query:
            floor = self._floor(args)
            return [
                {"scope": scope, "last_epoch": last}
                for scope, last in sorted(self.scopes, key=lambda each: (each[1] or 0, each[0]))
                if last is not None and last >= floor
            ]
        if RUN_MARKERS_TABLE in query:
            floor = self._floor(args)
            scopes = next(arg for arg in args if isinstance(arg, list))
            return [
                {"scope": scope, "prefix": prefix, "epoch": epoch}
                for scope, prefix, epoch in sorted(self.run_markers)
                if scope in scopes and epoch >= floor
            ]
        if query.startswith("SELECT count(*) AS notes"):
            return [{"notes": 0}]
        if "coalesce(sum(g.notes)" in query:
            return [{"notes": 0, "tables": 0, "bytes": 0}]
        if "table_name" in query and "pg_trigger" in query:
            return [{"table_name": table} for table in self.own]
        return []

    async def execute(self, query: str, *args: Any) -> str:
        self.sent.append((query, args))
        if query.startswith("UPDATE") and f'"{RING_TABLE}"' in query:
            assignments = query.split(" SET ", 1)[1].split(" WHERE ", 1)[0]
            for assignment in assignments.split(", "):
                name, _, placeholder = assignment.partition(" = $")
                self.ring[name] = args[int(placeholder) - 1]
        return "OK"

    async def copy_from_query(self, query: str, *args: Any, output: Any, **_: Any) -> str:
        """a scope's versions since the floor, as the restore's fold reads them: the table, the tenant, the key."""
        self.sent.append((query, args))
        floor = self._floor(args)
        tenant = schema_tenant(_SCHEMA)
        if f'"{_STAMPED_TABLE}"' in query:
            bucket = next(arg for arg in args if isinstance(arg, int) and arg < RESTORE_BUCKETS)
            for state, run, held, epoch in self.stamped:
                if held == bucket and epoch >= floor:
                    fields = (json.dumps(_STAMPED_TABLE), json.dumps(tenant), json.dumps([state, run]))
                    line = ",".join('"' + field.replace('"', '""') + '"' for field in fields) + ",\n"
                    await output(line.encode())
            return "COPY"
        state = next(arg for arg in args if isinstance(arg, str))
        for scope, run, candidate, epoch, _gone in self.versions:
            if scope == state and epoch >= floor:
                fields = (json.dumps(_TABLE), json.dumps(tenant), json.dumps([scope, run, candidate]))
                line = ",".join('"' + field.replace('"', '""') + '"' for field in fields) + ",\n"
                await output(line.encode())
        return "COPY"

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[_Conn]:
        yield _Conn(self)


async def _lost_log() -> tuple[FakeNatsClient, EpochGenerationSource, str]:
    """a log the keeper saw, then lost as a broker restart loses it."""
    bus = FakeNatsClient()
    source = EpochGenerationSource(bus)
    lost = await source.start_log()
    bus.change_log(EPOCH_CHANGES_BUCKET).vanish()
    return bus, source, lost


async def _restored_records(source: EpochGenerationSource) -> list[EpochChange]:
    state = await source.log_state()
    assert state is not None
    found = [await source.epoch_at(seq) for seq in range(1, state.last_seq + 1)]
    return [record for record in found if record is not None]


def _versioned_reads(pool: _Pool) -> list[tuple[str, tuple[Any, ...]]]:
    return [(query, args) for query, args in pool.sent if f'"{_TABLE}"' in query]


async def test_a_lost_log_is_rebuilt_from_the_versions_since_x_every_change_heard_by_key(tmp_path: Path) -> None:
    """Flips if a restore puts a versioned table's changes back from anything but its own versions (the ring holds no
    note here, so the old code restored none of them), misses a keyed version, a gone marker or a run marker stamped at
    X or later, reads a scope no stamped write touched since X, names no ``s_min`` on a record, or reads the table by
    anything but its scope and the epoch floor (the statement spy, and the no-whole-reads rule's judgement)."""
    _bus, source, lost = await _lost_log()
    now = datetime.now(UTC)
    mark_at = now - timedelta(minutes=1)
    floor = restore_floor(int(mark_at.timestamp() * 1000))
    pool = _Pool(lost, now, [{"log_incarnation": lost, "seq": 0, "at": mark_at, "id": 1}])
    pool.scopes = [
        ("*", floor + 50),
        ("state:CA", floor + 20),
        ("state:TX", floor - 10),  # last written before X: none of its versions can be at X or later
        ("race:ca-gov", floor + 30),  # another prefix: names none of this table's scopes
        ("state:NV", None),
    ]
    pool.versions = [
        ("CA", "r1", "ann", floor + 5, False),
        ("CA", "r1", "bob", floor + 9, True),  # a gone marker
        ("CA", "r1", "cal", floor - 100, False),  # before X: no record after the mark names it
        ("TX", "r1", "dee", floor - 50, False),
    ]
    pool.run_markers = [("CA", "CA\x1frun7", floor + 20), ("CA", "CA\x1frun1", floor - 1)]
    keeper = ChangeNoteKeeper(pool, notes_schema=_RING, log=source)

    done = await keeper.run_pass()

    assert done.restored is not None
    state = await source.log_state()
    assert state is not None and state.continues == lost, state
    assert state.covered_from == 0, "the newest mark at or before the ring's floor vouches for its floor"
    records = [record for record in await _restored_records(source) if record.table == _TABLE]
    keyed = sorted(tuple(row.ids) for record in records for row in record.rows)
    assert keyed == [("CA", "r1", "ann"), ("CA", "r1", "bob")], "every version and gone marker since X, by key"
    [whole] = [record for record in records if record.whole_table]
    assert whole.prefixes == [{_SCOPE: "CA", "run_id": "run7"}], "the run deleted since X, as its prefix"
    assert {record.s_min for record in records} == {floor}, "every rebuilt record names the restore's floor"

    reads = _versioned_reads(pool)
    assert len(reads) == 1, "one statement for the one scope a stamped write touched since X"
    [(query, args)] = reads
    assert f't."{_SCOPE}" = $1' in query and 't."epoch" >= $2' in query, query
    assert args == ("CA", floor), args
    statement = re.sub(r"\$\d+", "$1", query.replace('"rpt".', "").replace("rpt.", ""))
    (tmp_path / "src").mkdir()
    ddl = "\n".join(
        [
            *_RESULTS.create_statements(yugabyte=False)[:1],
            f"-- purpose: {_READ}",
            f"CREATE INDEX x ON {_TABLE} ({_SCOPE}, epoch);",
        ]
    )
    (tmp_path / "src" / "v001.sql").write_text(ddl.replace('"', ""), encoding="utf-8")
    catalog = collect_schema(tmp_path, (tmp_path / "src",))
    assert judge_sql(statement, Reader(catalog, frozenset({_READ}))) == [], "the restore's read is keyed"


async def test_a_restore_with_no_mark_vouches_for_nothing() -> None:
    """Flips if a restore with versioned tables and no mark of the lost log labels the new log as continuing it: no
    epoch floor can be named, so every follower must be behind, never carried over past what was missed."""
    _bus, source, lost = await _lost_log()
    pool = _Pool(lost, datetime.now(UTC), [])
    keeper = ChangeNoteKeeper(pool, notes_schema=_RING, log=source)

    await keeper.run_pass()

    state = await source.log_state()
    assert state is not None and state.continues == lost
    assert state.covered_from is None
    assert _versioned_reads(pool) == []


async def test_the_scrub_reports_a_record_whose_lowest_stamp_is_below_its_mark_s_floor() -> None:
    """Flips if the scrub does not count and report a record after a keeper mark whose ``s_min`` is below the mark's
    restore floor X (a restore from that mark would miss what it wrote below X), or counts one that held."""
    bus = FakeNatsClient()
    source = EpochGenerationSource(bus)
    incarnation = await source.start_log()
    now = datetime.now(UTC)
    mark_at = now - timedelta(minutes=1)
    floor = restore_floor(int(mark_at.timestamp() * 1000))
    held = ChangedRow(ids=["CA", "r1", "ann"])
    await source.advance_with_change(_TABLE, [held], s_min=floor - (1 << 15))  # before the mark: not judged
    pool = _Pool(incarnation, now, [{"log_incarnation": incarnation, "seq": 1, "at": mark_at, "id": 1}])
    pool.own = [_TABLE]
    await source.advance_with_change(_TABLE, [held], s_min=floor + 1)
    await source.advance_with_change(_TABLE, [ChangedRow(ids=["CA", "r1", "bob"])], scopes=None, s_min=floor - 1)
    keeper = ChangeNoteKeeper(pool, notes_schema=_RING, log=source, scrub_every=timedelta(0))

    await keeper.run_pass()  # the first pass's scrub takes the head the next one checks up to
    scrubbed = (await keeper.run_pass()).scrub

    assert scrubbed.checked == 3
    assert scrubbed.margin_violations == 1


def test_a_versioned_table_declares_its_scope_epoch_index_and_its_restore_declaration() -> None:
    """Flips if a versioned table's DDL lacks the (scope, epoch) index a restore reads it by, or the comment the keeper
    finds it by in the catalog, or the scope epochs table lacks its last_epoch index."""
    statements = _RESULTS.create_statements(yugabyte=True)
    assert any(f"{_TABLE}_scope_epoch ON" in each and f"({_SCOPE}, epoch)" in each for each in statements), statements
    declared = json.loads(_declaration())
    assert declared["scope"] == _SCOPE and declared["epochs"] == _EPOCHS and declared["scope_prefix"] == "state:"
    assert declared["keys"] == [_SCOPE, "run_id", "candidate"] and declared["tenancy"] == "schema"
    epochs = scope_epochs_schema(_EPOCHS).create_statements(yugabyte=True)
    assert any(f"{_EPOCHS}_last_epoch" in each and "last_epoch ASC, scope" in each for each in epochs), epochs


async def test_a_lost_log_is_rebuilt_from_a_stamped_table_s_rows_since_x_a_bucket_at_a_time(tmp_path: Path) -> None:
    """Flips if a restore does not put a stamped table's in-place changes back from its own rows (the ring holds no
    note here, so the old code restored none of them), misses a row stamped at X or later in any bucket, reads one
    stamped before X, names no ``s_min`` on a record, or reads the table by anything but one statement a bucket, each
    binding its bucket and the epoch floor (the statement spy, and the no-whole-reads rule's judgement)."""
    _bus, source, lost = await _lost_log()
    now = datetime.now(UTC)
    mark_at = now - timedelta(minutes=1)
    floor = restore_floor(int(mark_at.timestamp() * 1000))
    pool = _Pool(lost, now, [{"log_incarnation": lost, "seq": 0, "at": mark_at, "id": 1}])
    pool.catalog = [(_STAMPED_TABLE, _declaration(_RUNS))]
    # a row in every bucket stamped since X, and one in every bucket stamped before it
    pool.stamped = [
        *((f"S{bucket:02d}", "new", bucket, floor + bucket) for bucket in range(RESTORE_BUCKETS)),
        *((f"S{bucket:02d}", "old", bucket, floor - 1 - bucket) for bucket in range(RESTORE_BUCKETS)),
    ]
    keeper = ChangeNoteKeeper(pool, notes_schema=_RING, log=source)

    done = await keeper.run_pass()

    assert done.restored is not None
    state = await source.log_state()
    assert state is not None and state.continues == lost and state.covered_from == 0, state
    records = [record for record in await _restored_records(source) if record.table == _STAMPED_TABLE]
    keyed = sorted(tuple(row.ids) for record in records for row in record.rows)
    assert keyed == [(f"S{bucket:02d}", "new") for bucket in range(RESTORE_BUCKETS)], "every in-place change since X"
    assert records and {record.s_min for record in records} == {floor}, "every rebuilt record names the restore's floor"

    reads = [(query, args) for query, args in pool.sent if f'"{_STAMPED_TABLE}"' in query]
    assert len(reads) == RESTORE_BUCKETS == 16, "one statement a bucket"
    assert sorted(args for _, args in reads) == [(bucket, floor) for bucket in range(RESTORE_BUCKETS)]
    (tmp_path / "src").mkdir()
    ddl = "\n".join(
        [
            *_RUNS.create_statements(yugabyte=False)[:1],
            f"-- purpose: {_STAMPED_READ}",
            f"CREATE INDEX x ON {_STAMPED_TABLE} ({BUCKET_COLUMN}, epoch);",
        ]
    )
    (tmp_path / "src" / "v001.sql").write_text(ddl.replace('"', ""), encoding="utf-8")
    catalog = collect_schema(tmp_path, (tmp_path / "src",))
    for query, _args in reads:
        assert f't."{BUCKET_COLUMN}" = $1' in query and 't."epoch" >= $2' in query, query
        statement = re.sub(r"\$\d+", "$1", query.replace('"rpt".', "").replace("rpt.", ""))
        assert judge_sql(statement, Reader(catalog, frozenset({_STAMPED_READ}))) == [], "each bucket's read is keyed"


def test_a_stamped_table_declares_its_bucket_epoch_index_and_its_restore_declaration() -> None:
    """Flips if a stamped table's L3 DDL lacks the stored bucket column derived from its key's hash, the (bucket, epoch)
    index a restore reads it by (split a tablet a bucket on YugabyteDB, plain on Postgres), or the comment the keeper
    finds it by in the catalog; or if a cache's table carries the bucket column or its index."""
    index = bucket_epoch_index_name(_STAMPED_TABLE)
    on_yugabyte = _RUNS.create_statements(yugabyte=True)
    on_postgres = _RUNS.create_statements(yugabyte=False)
    for statements in (on_yugabyte, on_postgres):
        create = statements[0]
        assert f"{BUCKET_COLUMN} SMALLINT GENERATED ALWAYS AS (" in create and ") STORED" in create, create
        assert 'hashtext("join_state_code")' in create and 'hashtext("run_id")' in create and "& 15" in create, create
    [yb_index] = [each for each in on_yugabyte if f"{index} ON" in each]
    splits = ", ".join(f"({value})" for value in range(1, 16))
    assert f"({BUCKET_COLUMN} ASC, epoch) SPLIT AT VALUES ({splits})" in yb_index, yb_index
    [pg_index] = [each for each in on_postgres if f"{index} ON" in each]
    assert pg_index.endswith(f"({BUCKET_COLUMN} ASC, epoch)"), pg_index
    declared = json.loads(_declaration(_RUNS))
    assert declared["kind"] == "stamped" and declared["bucket"] == BUCKET_COLUMN and declared["buckets"] == 16
    assert declared["keys"] == [_SCOPE, "run_id"] and declared["tenancy"] == "schema"
    assert any(each.startswith(f'COMMENT ON INDEX "{index}"') for each in on_postgres), on_postgres
    cache = _RUNS.to_sqlalchemy_table(sa.MetaData())
    assert BUCKET_COLUMN not in cache.c and not any(each.name == index for each in cache.indexes)
