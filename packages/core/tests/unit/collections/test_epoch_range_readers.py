"""The ring's readers read the tables' own versions in an epoch range: the restore's read, narrowed (chunk 7).

Append-only L3 (``design-append-only-l3.md`` decision 3). Each reader that read the change-note ring until 0.95 reads
what a table changed in a bracket from the table itself: a versioned table's changed scopes (its scope epochs'
``last_epoch``), each one range of its ``(scope, epoch, key...)`` index between the bracket's floor and ceiling; a
stamped table's 16 ``(epoch_bucket, epoch)`` ranges. What each one does with it is what it did with the notes:

- the writers' backstop (``lost_writes.announce_window``) names a dead writer's rows, gone markers among them, and
  re-records a run deleted in the window whole by its prefix; more rows than one record names are recorded keys
  unknown with the window as their bracket;
- the keys-unknown pages (``noted_keys_page``, the hub's ``l3.noted_keys`` door, ``PoolNotedKeysReader``) page every
  key of a bracket, a tenant's own, across pages bounded by rows and by bytes, ``whole_table`` for a run deleted in it;
- the delta writer's lost-file recovery (``DeltaFolds._noted_keys``) writes every key of the lost record's bracket.

Each read is keyed: the statement spy holds every data read of a versioned table to its scope and the bracket's
bounds, and the no-whole-reads rule judges it keyed. The database is ``epoch_range_db``: DuckDB for the data, the
catalog from the declarations.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]

from packages.core.tests.unit.collections.epoch_range_db import (
    EPOCHS,
    RESULTS,
    SCHEMA,
    SCOPE,
    STAMPED,
    VERSIONED,
    RangeDb,
)
from threetears.core.collections.change_notes import PoolNotedKeysReader, noted_keys_page
from threetears.core.collections.delta_fold import DeltaFolds
from threetears.core.collections.epoch_change import ChangedRow, EpochChange
from threetears.core.collections.lost_writes import _MAX_NOTED_ROWS, announce_window
from threetears.core.collections.noted_keys import NotedKeysPage, noted_key_pages
from threetears.core.ddl.row_epoch import RESTORE_BUCKETS
from threetears.core.tenancy import schema_tenant
from threetears.enforcement.no_whole_reads.schema import collect_schema
from threetears.enforcement.no_whole_reads.sql import Reader, judge_sql

#: the bracket every test reads: a lost write's window, on the database's clock
_SINCE = datetime(2026, 11, 3, 20, 0, tzinfo=UTC)
_UNTIL = _SINCE + timedelta(seconds=40)
#: the bracket's epochs: the first stamp of its start's millisecond to the last of its end's (``wall_ms << 15``)
_FLOOR = int(_SINCE.timestamp() * 1000) << 15
_CEILING = ((int(_UNTIL.timestamp() * 1000) + 1) << 15) - 1

#: the platform schema's tenant: every row of the two tables, which are schema-tenancy tables in it
_TENANT = schema_tenant(SCHEMA)

#: the read the (scope, epoch) index names as its purpose (the restore's, which these readers share)
_READ = "packages/core/src/threetears/core/collections/epoch_reads.py::_range_page"


class _Source:
    """a recording source: each advance's table, rows, keys-unknown flag and what it named beside them."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    async def current(self, table_name: str) -> str:
        return "inc:0"

    async def advance(self, table_name: str) -> str:
        raise AssertionError("a recording source is advanced with its change")

    async def advance_with_change(
        self,
        table_name: str,
        rows: Sequence[ChangedRow] | None,
        *,
        keys_unknown: bool = False,
        tenant: str | None = None,
        noted_between: tuple[datetime, datetime] | None = None,
        prefixes: list[dict[str, str]] | None = None,
        partition: dict[str, str] | None = None,
    ) -> str:
        self.records.append(
            {
                "table": table_name,
                "rows": None if rows is None else sorted((tuple(row.ids), row.tenant) for row in rows),
                "keys_unknown": keys_unknown,
                "tenant": tenant,
                "bracket": noted_between,
                "prefixes": prefixes,
            }
        )
        return f"inc:{len(self.records)}"


def _filled() -> RangeDb:
    """the window's writes and their neighbours: versions and a gone marker in CA inside it, one before and one after
    it, a version in TX only before it, a run of CA deleted inside it and one before it; the stamped table's rows in
    two buckets inside it, one outside."""
    db = RangeDb()
    db.version("CA", "r1", "ann", _FLOOR + 5)
    db.version("CA", "r1", "bob", _FLOOR + 9, gone=True)
    db.version("CA", "r1", "ann", _FLOOR + 11)  # a second version of one key: named once
    db.version("CA", "r1", "cal", _FLOOR - 1)  # before the window
    db.version("CA", "r1", "dee", _CEILING + 1)  # after it
    db.version("TX", "r1", "eve", _FLOOR - 100)  # a scope no write in the window touched
    db.run_marker("CA", "run7", _FLOOR + 20)
    db.run_marker("CA", "run1", _FLOOR - 2)
    db.stamped("CA", "r2", 3, _FLOOR + 7)
    db.stamped("NV", "r3", 11, _FLOOR + 8, gone=True)
    db.stamped("NV", "r4", 11, _FLOOR - 8)
    return db


def _judged_keyed(tmp_path: Path, query: str) -> list[Any]:
    """the no-whole-reads rule's findings on a versioned table's read, against its (scope, epoch, key) index."""
    statement = re.sub(r"\$\d+", "$1", query.replace(f'"{SCHEMA}".', "").replace(f"{SCHEMA}.", ""))
    (tmp_path / "src").mkdir(exist_ok=True)
    ddl = "\n".join(
        [
            *RESULTS.create_statements(yugabyte=False)[:1],
            f"-- purpose: {_READ}",
            f"CREATE INDEX x ON {VERSIONED} ({SCOPE}, epoch, run_id, candidate);",
        ]
    )
    (tmp_path / "src" / "v001.sql").write_text(ddl.replace('"', ""), encoding="utf-8")
    catalog = collect_schema(tmp_path, (tmp_path / "src",))
    return judge_sql(statement, Reader(catalog, frozenset({_READ})))


def _assert_bounded(db: RangeDb, tmp_path: Path) -> None:
    """every data read of the versioned table binds a scope and the bracket's floor and ceiling, and is keyed; the
    stamped table is read a bucket at a time."""
    versioned = db.data_reads(VERSIONED)
    assert versioned, "the versioned table was read"
    for query, args in versioned:
        assert f't."{SCOPE}" = $1' in query, query
        assert _FLOOR in args and _CEILING in args, args
        assert judge_sql is not None and _judged_keyed(tmp_path, query) == [], query
    stamped = db.data_reads(STAMPED)
    for query, args in stamped:
        assert 't."epoch_bucket" = $1' in query and _FLOOR in args and _CEILING in args, (query, args)
    epochs = db.data_reads(EPOCHS)
    assert epochs and all('t."last_epoch" >= $1' in query for query, _ in epochs), epochs


async def test_the_backstop_announces_a_dead_writer_s_rows_from_the_tables_own_versions(tmp_path: Path) -> None:
    """Flips if a dead writer's lost advance is announced from anything but the tables' versions in its window (the
    old backstop read the ring's notes and the trigger catalog, and a versioned table carries neither): a version or
    gone marker inside the window not named, one outside it named, a key named twice, a run deleted inside the window
    not re-recorded whole by its prefix (or one before it re-recorded), the stamped table's rows in the window not
    named, or a read not bounded to its scope (or bucket) and the window."""
    db = _filled()
    source = _Source()

    done = await announce_window(
        db, generation_source=source, notes_schema=SCHEMA, since=_SINCE, until=_UNTIL, why="recovery"
    )

    assert set(done.advanced) == {VERSIONED, STAMPED} and not done.failed, done
    by_table: dict[str, list[dict[str, Any]]] = {}
    for record in source.records:
        by_table.setdefault(record["table"], []).append(record)
    [keyed, whole] = by_table[VERSIONED]
    assert keyed["rows"] == [(("CA", "r1", "ann"), _TENANT), (("CA", "r1", "bob"), _TENANT)], keyed
    assert not keyed["keys_unknown"]
    assert whole["rows"] is None and whole["prefixes"] == [{SCOPE: "CA", "run_id": "run7"}], whole
    assert whole["tenant"] == _TENANT
    [stamped] = by_table[STAMPED]
    assert stamped["rows"] == [(("CA", "r2"), _TENANT), (("NV", "r3"), _TENANT)], stamped
    assert len(db.data_reads(STAMPED)) == RESTORE_BUCKETS, "the stamped table's 16 ranges, one statement each"
    _assert_bounded(db, tmp_path)


async def test_more_rows_than_a_record_names_are_announced_keys_unknown_and_paged_back_exactly(tmp_path: Path) -> None:
    """Flips if a window of more rows than one record names is not recorded keys unknown with the window as its
    bracket, or if paging that bracket back (the follower's ``l3.noted_keys`` read) misses a key, names one outside it,
    splits a page past its row bound, or stops before the last page."""
    db = RangeDb()
    db.bulk_versions("CA", "big", _MAX_NOTED_ROWS + 1, _FLOOR + 1)
    db.version("CA", "small", "late", _CEILING + 5)
    source = _Source()

    await announce_window(db, generation_source=source, notes_schema=SCHEMA, since=_SINCE, until=_UNTIL, why="recovery")

    [record] = [record for record in source.records if record["table"] == VERSIONED]
    assert record["keys_unknown"] and record["bracket"] == (_SINCE, _UNTIL) and record["tenant"] == _TENANT, record
    reader = PoolNotedKeysReader(db, notes_schema=SCHEMA, tenant_token=_TENANT)
    keys: set[tuple[str, ...]] = set()
    pages = 0
    async for page in noted_key_pages(reader, VERSIONED, (_SINCE, _UNTIL)):
        assert isinstance(page, NotedKeysPage), page
        assert len(page.keys) <= 10_000
        pages += 1
        keys.update(page.keys)
    assert pages == 2, pages
    assert keys == {("CA", "big", f"c{index}") for index in range(_MAX_NOTED_ROWS + 1)}
    _assert_bounded(db, tmp_path)


async def test_a_keys_unknown_page_walks_every_range_by_row_and_byte_bounds_exactly(tmp_path: Path) -> None:
    """Flips if a keys-unknown page reads anything but the tables' versions in its bracket (the old read paged the
    ring's notes): a key inside the bracket missed across pages cut by rows and by bytes, one outside it named, a run
    deleted in it not flagged ``whole_table`` on the first page, another tenant's reader shown a key, or a table that
    carries no epoch answered anything but ``notes_unavailable``."""
    db = _filled()
    seen: list[tuple[str, ...]] = []
    cursor = None
    first = True
    while True:
        page = await noted_keys_page(
            db,
            notes_schema=SCHEMA,
            table=VERSIONED,
            tenant_token=_TENANT,
            noted_between=(_SINCE, _UNTIL),
            cursor=cursor,
            limit=2,
            max_bytes=1 << 20,
        )
        assert isinstance(page, NotedKeysPage), page
        assert page.whole_table is first, "a run deleted in the bracket flags the first page alone"
        first = False
        assert len(page.keys) <= 2
        seen.extend(page.keys)
        if page.next is None:
            break
        cursor = page.next
    assert sorted(seen) == [("CA", "r1", "ann"), ("CA", "r1", "ann"), ("CA", "r1", "bob")], seen

    # a byte bound of one key a page: every key still comes, one a page
    seen = []
    cursor = None
    while True:
        page = await noted_keys_page(
            db,
            notes_schema=SCHEMA,
            table=STAMPED,
            tenant_token=_TENANT,
            noted_between=(_SINCE, _UNTIL),
            cursor=cursor,
            limit=100,
            max_bytes=1,
        )
        assert isinstance(page, NotedKeysPage) and len(page.keys) <= 1, page
        seen.extend(page.keys)
        if page.next is None:
            break
        cursor = page.next
    assert sorted(seen) == [("CA", "r2"), ("NV", "r3")], seen

    other = await noted_keys_page(
        db,
        notes_schema=SCHEMA,
        table=VERSIONED,
        tenant_token=schema_tenant("someone_else"),
        noted_between=(_SINCE, _UNTIL),
        cursor=None,
        limit=100,
        max_bytes=1 << 20,
    )
    assert isinstance(other, NotedKeysPage) and other.keys == () and not other.whole_table, other
    absent = await noted_keys_page(
        db,
        notes_schema=SCHEMA,
        table="no_epoch_here",
        tenant_token=_TENANT,
        noted_between=(_SINCE, _UNTIL),
        cursor=None,
        limit=100,
        max_bytes=1 << 20,
    )
    assert absent == "notes_unavailable"
    _assert_bounded(db, tmp_path)


class _Sink:
    def __init__(self) -> None:
        self.batches: list[pa.Table] = []

    def write(self, batch: pa.Table) -> None:
        self.batches.append(batch)


async def test_a_lost_delta_file_s_keys_are_recovered_from_the_versions_in_its_bracket(tmp_path: Path) -> None:
    """Flips if the delta writer's recovery of a lost file of a commit too big to list reads its keys from anything
    but the table's versions in the record's bracket (it paged the ring's notes): a key missed, one outside the
    bracket written, or the run deleted in it not answered ``whole_table``."""
    db = RangeDb()
    db.version("CA", "r1", "ann", _FLOOR + 5)
    db.version("CA", "r1", "bob", _FLOOR + 6, gone=True)
    db.version("CA", "r1", "cal", _CEILING + 6)
    folds = SimpleNamespace(_notes=PoolNotedKeysReader(db, notes_schema=SCHEMA, tenant_token=_TENANT))
    table = SimpleNamespace(name=VERSIONED, key=(SCOPE, "run_id", "candidate"))
    record = EpochChange(table=VERSIONED, token="inc:1", keys_unknown=True, noted_between=(_SINCE, _UNTIL))
    sink = _Sink()

    count = await DeltaFolds._noted_keys(folds, table, record, sink)  # type: ignore[arg-type]  # noqa: SLF001

    assert count == 2, count
    written = pa.concat_tables(sink.batches)
    assert written.column_names == [SCOPE, "run_id", "candidate"]
    assert sorted(written.column("candidate").to_pylist()) == ["ann", "bob"]

    db.run_marker("CA", "run9", _FLOOR + 7)
    assert await DeltaFolds._noted_keys(folds, table, record, _Sink()) == "whole_table"  # type: ignore[arg-type]  # noqa: SLF001
    _assert_bounded(db, tmp_path)
