"""A lost change log is rebuilt from the tables' own versions since X, with no ring; the scrub proves X held.

Append-only L3 chunks 5 and 7 (``design-append-only-l3.md`` decision 3). The log keeper's restore finds every versioned
table by the restore declaration its ``(scope, epoch)`` index carries (``TableSchema.create_statements`` writes it), the
scopes a stamped write touched since X from the scope epochs table's ``last_epoch``, and reads each scope's versions,
gone markers and run markers with ``epoch >= X``, X = (T - M) << 15 from the lost log's oldest mark. The log it restores
is the one its newest mark names. The pool here is a stand-in answering exactly the statements a keeper sends (its
marks, the database's clock, the catalog, the tables), each one recorded: every read of the versioned table binds its
scope and its epoch floor, and the no-whole-reads rule judges each keyed. The live plan check is
``test_table_restore_plans_live``.

**Two reads a scope.** A restore sizes the changes it would put back against the log's restore budget before it appends
any (so a part too big is cut at a later mark, never appended and then abandoned), then reads them again to append: the
sizing pass and the restore each read every touched scope once, in the restore's worker, holding nothing.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import sqlalchemy as sa

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.epoch_change import ChangedRow, EpochChange
from threetears.core.collections.generation import advance_for_write
from threetears.core.collections.log_keeper import LogKeeper
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
    collection_for_schema,
    stamped_values,
)
from threetears.core.collections.scope_epochs import scope_epochs_schema
from threetears.core.ddl.keeper_marks import MARKS_TABLE
from threetears.core.ddl.row_epoch import (
    BUCKET_COLUMN,
    EPOCH_COLUMN,
    RESTORE_BUCKETS,
    ROW_STAMPED,
    bucket_epoch_index_name,
    restore_floor,
    row_versions,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.ddl.run_markers import RUN_MARKERS_TABLE
from threetears.core.entities.base import BaseEntity
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
    """the keeper's one connection (its install's): the DDL lock and the catalog, answering nothing."""

    def __init__(self, pool: _Pool) -> None:
        self._pool = pool

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        yield

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        self._pool.sent.append((query, args))
        return []

    async def execute(self, query: str, *args: Any) -> str:
        self._pool.sent.append((query, args))
        return "OK"


class _Pool:
    """the L3 a keeper reads: its marks and the database's clock, the catalog's restored tables, their scope epochs,
    versions and run markers. Every statement is recorded with its arguments."""

    def __init__(self, lost: str, now: datetime, marks: list[dict[str, Any]]) -> None:
        self.sent: list[tuple[str, tuple[Any, ...]]] = []
        self.now = now
        self.lost = lost
        #: the keeper's marks: log_incarnation, seq, at, id
        self.marks = marks
        #: (scope, last_epoch) of the scope epochs table
        self.scopes: list[tuple[str, int | None]] = []
        #: (state, run, candidate, epoch, gone) of the versioned table
        self.versions: list[tuple[str, str, str, int, bool]] = []
        #: (scope, prefix, epoch) of the run-marker table, all of the versioned table
        self.run_markers: list[tuple[str, str, int]] = []
        #: the base tables of the keeper's schema, which an older writer's record (naming no database) is judged by
        self.own: list[str] = []
        #: the catalog's restore declarations: (table, declaration)
        self.catalog: list[tuple[str, str]] = [(_TABLE, _declaration())]
        #: (state, run, bucket, epoch) of the stamped table
        self.stamped: list[tuple[str, str, int, int]] = []

    @staticmethod
    def _floor(args: tuple[Any, ...]) -> int:
        # the epoch floor is the one bound epoch-sized integer (a LIMIT is small)
        return next(arg for arg in args if isinstance(arg, int) and arg > 1 << 40)

    def _marks(self, query: str, args: tuple[Any, ...]) -> list[dict[str, Any]]:
        """the marks statements the keeper sends: its newest mark, a log's marks past a position, a window's anchor,
        and the window's sweep."""
        if "DELETE" in query:
            before = next(arg for arg in args if isinstance(arg, datetime))
            gone = [mark for mark in self.marks if mark["at"] < before]
            self.marks = [mark for mark in self.marks if mark not in gone]
            return [{"id": mark["id"]} for mark in gone]
        if 'ORDER BY t."at" DESC' in query:
            return sorted(self.marks, key=lambda mark: (mark["at"], str(mark["id"])), reverse=True)[:1]
        incarnation, position = args[0], args[1]
        held = [mark for mark in self.marks if mark["log_incarnation"] == incarnation]
        if 't."seq" < $2' in query:
            return sorted((mark for mark in held if mark["seq"] < position), key=lambda mark: mark["seq"], reverse=True)
        return sorted((mark for mark in held if mark["seq"] > position), key=lambda mark: mark["seq"])

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        self.sent.append((query, args))
        if query.startswith("SELECT now()"):
            return [{"now": self.now}]
        if "pg_description" in query:
            return [
                {"table_schema": _SCHEMA, "table_name": table, "declared": declared, "run_markers": True}
                for table, declared in self.catalog
            ]
        if MARKS_TABLE in query:
            return self._marks(query, args)
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
        if "information_schema.tables" in query:
            return [{"table_name": table} for table in self.own]
        return []

    async def execute(self, query: str, *args: Any) -> str:
        self.sent.append((query, args))
        if query.startswith("INSERT INTO") and MARKS_TABLE in query:
            incarnation, seq, at = args
            self.marks.append({"log_incarnation": incarnation, "seq": seq, "at": at, "id": len(self.marks) + 100})
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


def _keeper(pool: _Pool, source: EpochGenerationSource, **kwargs: Any) -> LogKeeper:
    return LogKeeper(pool, schema=_RING, log=source, **kwargs)


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
    note: it carries none), misses a keyed version, a gone marker or a run marker stamped at
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
    keeper = _keeper(pool, source)

    done = await keeper.run_pass()

    assert done.restored is not None
    state = await source.log_state()
    assert state is not None and state.continues == lost, state
    assert state.covered_from == 0, "the lost log's oldest mark vouches for every record after it"
    records = [record for record in await _restored_records(source) if record.table == _TABLE]
    keyed = sorted(tuple(row.ids) for record in records for row in record.rows)
    assert keyed == [("CA", "r1", "ann"), ("CA", "r1", "bob")], "every version and gone marker since X, by key"
    [whole] = [record for record in records if record.whole_table]
    assert whole.prefixes == [{_SCOPE: "CA", "run_id": "run7"}], "the run deleted since X, as its prefix"
    assert {record.s_min for record in records} == {floor}, "every rebuilt record names the restore's floor"

    reads = _versioned_reads(pool)
    assert len(reads) == 2 and reads[0] == reads[1], (
        "one statement for the one scope touched since X: sized, then put back"
    )
    (query, args) = reads[0]
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


async def test_a_keeper_with_no_mark_knows_no_log_and_starts_a_fresh_one() -> None:
    """Flips if a keeper whose marks name no log (a database's first start) labels a log as continuing one, reads a
    table, or leaves the log it declared unmarked: its newest mark is what it knows of the log, so the next pass would
    take the log it declared for one it did not know."""
    bus = FakeNatsClient()
    source = EpochGenerationSource(bus)
    pool = _Pool("none", datetime.now(UTC), [])
    keeper = _keeper(pool, source)

    done = await keeper.run_pass()

    state = await source.log_state()
    assert state is not None and done.started == state.incarnation and state.continues is None, (done, state)
    assert done.marked and [(mark["log_incarnation"], mark["seq"]) for mark in pool.marks] == [(state.incarnation, 0)]
    assert _versioned_reads(pool) == []
    again = await keeper.run_pass()
    assert again.adopted is None and again.started is None, "the newest mark names the log it declared"


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
    keeper = _keeper(pool, source, scrub_every=timedelta(0))

    await keeper.run_pass()  # the first pass's scrub takes the head the next one checks up to
    scrubbed = (await keeper.run_pass()).scrub

    assert scrubbed.checked == 3
    assert scrubbed.margin_violations == 1
    assert scrubbed.missing == 0, "a versioned table's records are read back by a restore"


class _Run(BaseEntity):
    primary_key_field = "run_id"


async def test_a_one_statement_stamped_write_s_record_names_its_own_stamp_and_the_scrub_judges_it() -> None:
    """Flips if the epoch record of a stamped table's write that joined no transaction (one statement, its own commit,
    stamped from the process clock) names no ``s_min``, or one other than the write's own stamp: the restore scrub
    could not check its margin. Both stamped draws: the store's stamped upsert and a keyed update's values. An advance
    after them, with no write of its own, names none."""
    bus = FakeNatsClient()
    source = EpochGenerationSource(bus)
    incarnation = await source.start_log()
    await source.advance_with_change(_STAMPED_TABLE, [ChangedRow(ids=["CA", "r0"])])  # seq 1: the mark's position
    registry = CollectionRegistry()
    store: Any = SqlL3Backend(MagicMock())
    registry.configure(l3_pool=store)
    runs = collection_for_schema(_RUNS, entity_class=_Run)(registry, DefaultCoreConfig(), None)
    sent: list[tuple[Any, ...]] = []

    async def execute(_sql: str, *params: Any) -> str:
        sent.append(params)
        return "INSERT 0 1"

    store.execute = execute
    rows = [ChangedRow(ids=["CA", "r1"])]

    await runs.save_to_store({_SCOPE: "CA", "run_id": "r1", "rows": 1})
    [params] = sent
    [upserted] = [value for value in params if isinstance(value, int) and value > 1 << 40]
    token = await advance_for_write(source, _STAMPED_TABLE, rows, schema=_SCHEMA, tenancy=TENANT_BY_SCHEMA)
    assert token is not None
    record = await source.change(_STAMPED_TABLE, token)
    assert record is not None and record.s_min == upserted, "the upsert's record names its own stamp"

    [updated] = stamped_values(_RUNS, {"rows": 2})[EPOCH_COLUMN].params
    token = await advance_for_write(source, _STAMPED_TABLE, rows, schema=_SCHEMA, tenancy=TENANT_BY_SCHEMA)
    assert token is not None
    record = await source.change(_STAMPED_TABLE, token)
    assert record is not None and record.s_min == updated, "the keyed update's record names its own stamp"

    token = await advance_for_write(source, _STAMPED_TABLE, rows, schema=_SCHEMA, tenancy=TENANT_BY_SCHEMA)
    assert token is not None
    record = await source.change(_STAMPED_TABLE, token)
    assert record is not None and record.s_min is None, "a stamp is named once, by the advance after its write"

    # a mark whose floor is above both stamps: the scrub judges the two stamped records, and both fall below it
    now = datetime.now(UTC)
    pool = _Pool(
        incarnation, now, [{"log_incarnation": incarnation, "seq": 1, "at": now + timedelta(hours=1), "id": 1}]
    )
    pool.own = [_STAMPED_TABLE]
    keeper = _keeper(pool, source, scrub_every=timedelta(0))
    await keeper.run_pass()  # the first pass's scrub takes the head the next one checks up to
    scrubbed = (await keeper.run_pass()).scrub
    assert scrubbed.checked == 4
    assert scrubbed.margin_violations == 2


def test_a_versioned_table_declares_its_scope_epoch_index_and_its_restore_declaration() -> None:
    """Flips if a versioned table's DDL lacks the index led by (scope, epoch) a restore reads it by (its one
    (scope, epoch, key...) index, which cleanup pages by), or the comment the keeper finds it by in the catalog, or the
    scope epochs table lacks its last_epoch index."""
    statements = _RESULTS.create_statements(yugabyte=True)
    assert any(
        f"{_TABLE}_scope_epoch ON" in each and f"({_SCOPE}, epoch, run_id, candidate)" in each for each in statements
    ), statements
    declared = json.loads(_declaration())
    assert declared["scope"] == _SCOPE and declared["epochs"] == _EPOCHS and declared["scope_prefix"] == "state:"
    assert declared["keys"] == [_SCOPE, "run_id", "candidate"] and declared["tenancy"] == "schema"
    epochs = scope_epochs_schema(_EPOCHS).create_statements(yugabyte=True)
    assert any(f"{_EPOCHS}_last_epoch" in each and "last_epoch ASC, scope" in each for each in epochs), epochs


async def test_a_lost_log_is_rebuilt_from_a_stamped_table_s_rows_since_x_a_bucket_at_a_time(tmp_path: Path) -> None:
    """Flips if a restore does not put a stamped table's in-place changes back from its own rows (the ring holds no
    note: it carries none), misses a row stamped at X or later in any bucket, reads one
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
    keeper = _keeper(pool, source)

    done = await keeper.run_pass()

    assert done.restored is not None
    state = await source.log_state()
    assert state is not None and state.continues == lost and state.covered_from == 0, state
    records = [record for record in await _restored_records(source) if record.table == _STAMPED_TABLE]
    keyed = sorted(tuple(row.ids) for record in records for row in record.rows)
    assert keyed == [(f"S{bucket:02d}", "new") for bucket in range(RESTORE_BUCKETS)], "every in-place change since X"
    assert records and {record.s_min for record in records} == {floor}, "every rebuilt record names the restore's floor"

    reads = [(query, args) for query, args in pool.sent if f'"{_STAMPED_TABLE}"' in query]
    assert len(reads) == 2 * RESTORE_BUCKETS == 32, "one statement a bucket: sized, then put back"
    assert reads[:RESTORE_BUCKETS] == reads[RESTORE_BUCKETS:]
    assert sorted(args for _, args in reads[:RESTORE_BUCKETS]) == [(bucket, floor) for bucket in range(RESTORE_BUCKETS)]
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


async def test_a_dropped_log_is_put_back_from_the_tables_with_no_ring() -> None:
    """Flips if a log lost after the keeper marked it is not put back from the tables (the version written after the
    mark, labelled as continuing the lost log from that mark), if any statement the keeper sends names the change-note
    ring (its tables, its ring row), or if the keeper does not take the restored log as its own (its newest mark names
    it, so the next pass neither restores again nor takes it for a log it did not know)."""
    bus = FakeNatsClient()
    source = EpochGenerationSource(bus)
    now = datetime.now(UTC)
    pool = _Pool("none", now, [])
    keeper = _keeper(pool, source)
    started = (await keeper.run_pass()).started
    assert started is not None
    floor = restore_floor(int(now.timestamp() * 1000))
    pool.scopes = [("state:CA", floor + 10)]
    pool.versions = [("CA", "r1", "ann", floor + 5, False), ("CA", "r1", "old", floor - 5, False)]
    await source.advance_with_change(_TABLE, [ChangedRow(ids=["CA", "r1", "ann"])], s_min=floor + 5)
    bus.change_log(EPOCH_CHANGES_BUCKET).vanish()
    pool.now = now + timedelta(seconds=1)

    done = await keeper.run_pass()

    assert done.restored is not None and done.marked, done
    state = await source.log_state()
    assert state is not None and state.incarnation == done.restored, state
    assert state.continues == started and state.covered_from == 0, state
    records = [record for record in await _restored_records(source) if record.table == _TABLE]
    assert sorted(tuple(row.ids) for record in records for row in record.rows) == [("CA", "r1", "ann")]
    ring = [query for query, _ in pool.sent if "threetears_change_notes_" in query or "change_note_ring" in query]
    assert ring == [], ring
    assert pool.marks[-1]["log_incarnation"] == done.restored, "the keeper took the restored log as its own"
    after = await keeper.run_pass()
    assert after.restored is None and after.adopted is None and after.started is None, after


async def test_the_scrub_reports_a_record_of_a_table_a_restore_cannot_read_back() -> None:
    """Flips if the scrub does not count and report a record of this database's table that carries no epoch (no restore
    declaration names it, so a restore after a NATS loss would not put it back), or counts one of a versioned table."""
    bus = FakeNatsClient()
    source = EpochGenerationSource(bus)
    incarnation = await source.start_log()
    now = datetime.now(UTC)
    pool = _Pool(incarnation, now, [{"log_incarnation": incarnation, "seq": 0, "at": now, "id": 1}])
    pool.own = [_TABLE, "plain_rows"]
    await source.advance_with_change(_TABLE, [ChangedRow(ids=["CA", "r1", "ann"])])
    await source.advance_with_change("plain_rows", [ChangedRow(ids=["a"])])
    keeper = _keeper(pool, source, scrub_every=timedelta(0))

    await keeper.run_pass()  # the first pass's scrub takes the head the next one checks up to
    scrubbed = (await keeper.run_pass()).scrub

    assert scrubbed.checked == 2 and scrubbed.missing == 1, scrubbed


async def test_a_keeper_beside_puts_its_own_tables_back_into_the_owner_s_restore() -> None:
    """Flips if a keeper beside the log's owner (a second database's) does not put its own tables' changes back into
    the restore the owner declared (from its own mark of the lost log, naming its database, labelling nothing), or does
    not take the restored log as its own afterwards, so a later pass would put them back again."""
    bus = FakeNatsClient()
    owner = EpochGenerationSource(bus)
    lost = await owner.start_log()
    beside = EpochGenerationSource(bus, database=_RING)
    now = datetime.now(UTC)
    mark_at = now - timedelta(minutes=1)
    floor = restore_floor(int(mark_at.timestamp() * 1000))
    pool = _Pool(lost, now, [{"log_incarnation": lost, "seq": 0, "at": mark_at, "id": 1}])
    pool.scopes = [("state:CA", floor + 20)]
    pool.versions = [("CA", "r1", "ann", floor + 5, False)]
    bus.change_log(EPOCH_CHANGES_BUCKET).vanish()
    restored = await owner.begin_restore(lost)
    assert restored is not None
    keeper = LogKeeper(pool, schema=_RING, log=beside, keeps_log=False)

    done = await keeper.run_pass()

    assert done.put_back == restored and done.marked, done
    state = await owner.log_state()
    assert state is not None and state.continues is None, "a keeper beside labels nothing: the owner does"
    records = [record for record in await _restored_records(owner) if record.table == _TABLE]
    assert sorted(tuple(row.ids) for record in records for row in record.rows) == [("CA", "r1", "ann")]
    assert {record.database for record in records} == {_RING}, "its records name its own database"
    assert pool.marks[-1]["log_incarnation"] == restored
    await owner.finish_restore((), incarnation=restored, lost=lost, covered_from=0)
    again = await keeper.run_pass()
    assert again.put_back is None, again
