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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import sqlalchemy as sa

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.epoch_change import ChangedRow
from threetears.core.collections.generation import advance_for_write
from threetears.core.collections.log_keeper import LogKeeper
from threetears.core.collections.quiet_cleanup import log_window_floor
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    TableSchema,
    collection_for_schema,
    stamped_values,
)
from threetears.core.collections.scope_epochs import scope_epochs_schema
from threetears.core.ddl.row_epoch import (
    BUCKET_COLUMN,
    EPOCH_COLUMN,
    RESTORE_BUCKETS,
    ROW_STAMPED,
    bucket_epoch_index_name,
    cleanup_floor,
    restore_floor,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.ddl.tenant_moves import TENANT_MOVES_TABLE, moved_epoch_column, moved_from_column
from threetears.core.entities.base import BaseEntity
from threetears.core.eviction import Declared, Policy
from threetears.core.tenancy import TENANT_BY_SCHEMA, tenant_column, value_tenant
from threetears.core.testing.kv import FakeNatsClient
from threetears.enforcement.no_whole_reads.schema import collect_schema
from threetears.enforcement.no_whole_reads.sql import Reader, judge_sql
from threetears.epoch import EPOCH_CHANGES_BUCKET, EpochGenerationReader, EpochGenerationSource
from threetears.nats import set_default_namespace

from packages.core.tests.unit.collections.restore_pool import (
    declaration as _declaration,
    EPOCHS as _EPOCHS,
    lost_log as _lost_log,
    restored_records as _restored_records,
    RestorePool as _Pool,
    RESULTS as _RESULTS,
    RING as _RING,
    RUNS as _RUNS,
    SCHEMA as _SCHEMA,
    SCOPE as _SCOPE,
    STAMPED_TABLE as _STAMPED_TABLE,
    TABLE as _TABLE,
)

#: the restore read the (scope, epoch) index names as its purpose
_READ = "packages/core/src/threetears/core/collections/table_restore.py::_scope_versions"

#: the restore read the (bucket, epoch) index names as its purpose
_STAMPED_READ = "packages/core/src/threetears/core/collections/table_restore.py::_bucket_rows"


@pytest.fixture(autouse=True)
def _namespace() -> None:
    set_default_namespace("tablerestore")


def _keeper(pool: _Pool, source: EpochGenerationSource, **kwargs: Any) -> LogKeeper:
    return LogKeeper(pool, schema=_RING, log=source, **kwargs)


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


_MOVING_TABLE = "moving_runs"
_MOVING_FROM = moved_from_column("customer_id")
_MOVING_EPOCH = moved_epoch_column("customer_id")
_LEFT = "0193a2b4-0000-7000-8000-00000000000a"
_JOINED = "0193a2b4-0000-7000-8000-00000000000b"

#: a stamped table whose tenant column a write may change in place (``TableSchema.moveable``, tenant_moves)
_MOVING = TableSchema(
    name=_MOVING_TABLE,
    primary_key="id",
    columns=[Column("id", STRING_TYPE), Column("customer_id", UUID_TYPE, nullable=True), Column("rows", BIGINT_TYPE)],
    tenancy=tenant_column("customer_id"),
    row_epoch=ROW_STAMPED,
    moveable=("customer_id",),
    l1_eviction=Declared(Policy.EVICTABLE, "a test's stamped table whose tenant moves"),
)


async def test_a_restore_reads_back_each_moved_row_for_the_tenant_it_left() -> None:
    """Flips if a restore of a lost log reads a moveable stamped table's rows back only for their current tenant (the
    tenant a row left since X keeps it), reads the moves other than by the moves table's key a bucket (the table, the
    bucket, the epoch from X), or reads a row whose move is older than X back for the tenant it left long ago."""
    bus, source, lost = await _lost_log()
    now = datetime.now(UTC)
    mark_at = now - timedelta(minutes=1)
    floor = restore_floor(int(mark_at.timestamp() * 1000))
    pool = _Pool(lost, now, [{"log_incarnation": lost, "seq": 0, "at": mark_at, "id": 1}])
    pool.catalog = [(_MOVING_TABLE, _declaration(_MOVING))]
    # (key, bucket, epoch, current tenant, tenant it left, its move's epoch)
    rows: list[tuple[str, int, int, str, str | None, int | None]] = [
        ("moved", 3, floor + 10, _JOINED, _LEFT, floor + 10),
        ("moved-long-ago", 5, floor + 20, _JOINED, _LEFT, floor - 1),
        ("never-moved", 7, floor + 30, _JOINED, None, None),
    ]

    async def copy_from_query(query: str, *args: Any, output: Any, **_: Any) -> str:
        pool.sent.append((query, args))
        moves = TENANT_MOVES_TABLE in query
        # a bucket read binds the bucket; a moves read the table, then the bucket as its scope's text
        bucket = int(args[1]) if moves else next(arg for arg in args if isinstance(arg, int) and arg < RESTORE_BUCKETS)
        for key, held, epoch, tenant, left, moved_at in rows:
            named: str | None = tenant
            if moves:
                # the moves table holds a row a move, at the move's epoch
                if held != bucket or moved_at is None or moved_at < floor:
                    continue
                named = left
            elif held != bucket or epoch < floor:
                continue
            held_by = None if named is None else value_tenant(named)
            fields = (json.dumps(_MOVING_TABLE), json.dumps(held_by), json.dumps([key]))
            await output((",".join('"' + field.replace('"', '""') + '"' for field in fields) + ",\n").encode())
        return "COPY"

    pool.copy_from_query = copy_from_query  # type: ignore[method-assign]
    keeper = _keeper(pool, source)

    done = await keeper.run_pass()

    assert done.restored is not None
    records = [record for record in await _restored_records(source) if record.table == _MOVING_TABLE]
    # each record's tenant is the key it went under: the log's entries, by their sequence (the token's last part)
    entries = await bus.change_log(EPOCH_CHANGES_BUCKET).records_after(f"*.{_MOVING_TABLE}.e", 0, 1000)
    tenants = {entry.seq: entry.key.split(".", 1)[0] for entry in entries}
    heard = sorted(
        (tenants[int(record.token.rpartition(":")[2])], row.ids[0]) for record in records for row in record.rows
    )
    assert heard == sorted(
        [
            (value_tenant(_JOINED), "moved"),
            (value_tenant(_LEFT), "moved"),
            (value_tenant(_JOINED), "moved-long-ago"),
            (value_tenant(_JOINED), "never-moved"),
        ]
    ), heard
    moves = [(query, args) for query, args in pool.sent if TENANT_MOVES_TABLE in query]
    plain = [query for query, _ in pool.sent if f'"{_MOVING_TABLE}"' in query]
    # the keeper reads the tables once to size the restore and once to restore: a moves read beside each bucket read
    assert plain and len(moves) == len(plain) and len(moves) % RESTORE_BUCKETS == 0, "one moves read a bucket"
    for query, args in moves:
        assert 't."table_name" = $1' in query and 't."scope" = $2' in query and 't."epoch" >= $3' in query, query
        assert args[0] == _MOVING_TABLE and args[2] == floor, args
        assert _MOVING_FROM not in query and _MOVING_EPOCH not in query, "the row's last move is not what is read"


async def test_the_owner_publishes_the_cleanup_floor_a_pod_reads_in_the_log_s_state() -> None:
    """Flips if the log's owner does not publish the window's cleanup floor F at its mark, publishes it again where it
    did not move, or if a pod's reader (no read by sequence: ``ChangeLog.at``) cannot take F from the log's state: a pod's
    quiet-time cleanup would then remove nothing. A log whose owner published none (an older hub) tells a pod no floor."""
    bus = FakeNatsClient()
    source = EpochGenerationSource(bus)
    incarnation = await source.start_log()
    pod = EpochGenerationReader(bus)
    s_min = (int(datetime.now(UTC).timestamp() * 1000) - 60_000) << 15
    await source.advance_with_change(_TABLE, [ChangedRow(ids=["CA", "r1", "ann"])], s_min=s_min)
    log = bus.change_log(EPOCH_CHANGES_BUCKET)
    log.granted_reads = ()
    assert await log_window_floor(pod) is None, "no owner published a floor: an older hub's log"
    log.granted_reads = None

    pool = _Pool(incarnation, datetime.now(UTC), [])
    keeper = _keeper(pool, source, mark_every=timedelta(microseconds=1))
    assert (await keeper.run_pass()).marked

    log.granted_reads = ()  # the pod's grant: no read by sequence
    state = await pod.log_state()
    assert state is not None and state.cleanup_floor == cleanup_floor(s_min), state
    assert await log_window_floor(pod) == cleanup_floor(s_min)
    log.granted_reads = None
    assert await log_window_floor(source) == cleanup_floor(s_min), "the owner's own F, read by sequence"

    updates: list[int] = []
    publish = log.label_cleanup_floor

    async def counted(floor: int) -> Any:
        updates.append(floor)
        return await publish(floor)

    log.label_cleanup_floor = counted  # type: ignore[method-assign]
    assert (await keeper.run_pass()).marked
    assert updates == [], "F did not move: no stream update"
