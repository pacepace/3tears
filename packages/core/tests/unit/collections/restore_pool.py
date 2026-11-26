"""The stand-in L3 a log keeper restores a lost change log from, shared by the table-restore tests.

The pool answers exactly the statements a keeper sends (its marks, the database's clock, the catalog, the tables), each
one recorded; the versioned table and the stamped table are a consumer's shapes. ``test_table_restore`` and
``test_touched_scopes`` read it.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any


from threetears.core.collections.epoch_change import EpochChange
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
)
from threetears.core.ddl.keeper_marks import MARKS_TABLE
from threetears.core.ddl.row_epoch import (
    RESTORE_BUCKETS,
    ROW_STAMPED,
    row_versions,
)
from threetears.core.ddl.run_markers import RUN_MARKERS_TABLE, TABLE_RESET_SCOPE
from threetears.core.eviction import Declared, Policy
from threetears.core.tenancy import TENANT_BY_SCHEMA, schema_tenant
from threetears.core.testing.kv import FakeNatsClient
from threetears.epoch import EPOCH_CHANGES_BUCKET, EpochGenerationSource

SCHEMA = "rpt"
TABLE = "restore_results"
EPOCHS = "restore_epochs"
SCOPE = "join_state_code"
RING = "ring"

#: the versioned table a consumer declares: its key, its scope, the scope epochs table naming its states
RESULTS = TableSchema(
    name=TABLE,
    primary_key=(SCOPE, "run_id", "candidate"),
    columns=[
        Column(SCOPE, STRING_TYPE),
        Column("run_id", STRING_TYPE),
        Column("candidate", STRING_TYPE),
        Column("votes", BIGINT_TYPE),
    ],
    tenancy=TENANT_BY_SCHEMA,
    row_epoch=row_versions(SCOPE, epochs=EPOCHS, scope_prefix="state:"),
    l1_eviction=Declared(Policy.EVICTABLE, "a test's versioned table"),
)

STAMPED_TABLE = "restore_runs"

#: a stamped table a consumer declares (``lsf_runs``' shape): one row a key, updated in place
RUNS = TableSchema(
    name=STAMPED_TABLE,
    primary_key=(SCOPE, "run_id"),
    columns=[Column(SCOPE, STRING_TYPE), Column("run_id", STRING_TYPE), Column("rows", BIGINT_TYPE)],
    tenancy=TENANT_BY_SCHEMA,
    row_epoch=ROW_STAMPED,
    l1_eviction=Declared(Policy.EVICTABLE, "a test's stamped table"),
)


def declaration(schema: TableSchema = RESULTS) -> str:
    """the restore declaration the table's DDL gives its restore index, as the catalog then holds it."""
    [comment] = [statement for statement in schema.create_statements(yugabyte=True) if statement.startswith("COMMENT")]
    found = re.search(r"IS '(.*)'$", comment)
    assert found is not None, comment
    return found.group(1).replace("''", "'")


class _Conn:
    """the keeper's one connection (its install's): the DDL lock and the catalog, answering nothing."""

    def __init__(self, pool: RestorePool) -> None:
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


class RestorePool:
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
        self.catalog: list[tuple[str, str]] = [(TABLE, declaration())]
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
                {"table_schema": SCHEMA, "table_name": table, "declared": declared, "run_markers": True}
                for table, declared in self.catalog
            ]
        if MARKS_TABLE in query:
            return self._marks(query, args)
        if f'"{EPOCHS}"' in query:
            floor = self._floor(args)
            return [
                {"scope": scope, "last_epoch": last}
                for scope, last in sorted(self.scopes, key=lambda each: (each[1] or 0, each[0]))
                if last is not None and last >= floor
            ]
        if RUN_MARKERS_TABLE in query:
            floor = self._floor(args)
            if TABLE_RESET_SCOPE in args:
                # a table's reset markers, by (table_name, scope): none here
                return []
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
        tenant = schema_tenant(SCHEMA)
        if f'"{STAMPED_TABLE}"' in query:
            bucket = next(arg for arg in args if isinstance(arg, int) and arg < RESTORE_BUCKETS)
            for state, run, held, epoch in self.stamped:
                if held == bucket and epoch >= floor:
                    fields = (json.dumps(STAMPED_TABLE), json.dumps(tenant), json.dumps([state, run]))
                    line = ",".join('"' + field.replace('"', '""') + '"' for field in fields) + ",\n"
                    await output(line.encode())
            return "COPY"
        state = next(arg for arg in args if isinstance(arg, str))
        for scope, run, candidate, epoch, _gone in self.versions:
            if scope == state and epoch >= floor:
                fields = (json.dumps(TABLE), json.dumps(tenant), json.dumps([scope, run, candidate]))
                line = ",".join('"' + field.replace('"', '""') + '"' for field in fields) + ",\n"
                await output(line.encode())
        return "COPY"

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[_Conn]:
        yield _Conn(self)


async def lost_log() -> tuple[FakeNatsClient, EpochGenerationSource, str]:
    """a log the keeper saw, then lost as a broker restart loses it."""
    bus = FakeNatsClient()
    source = EpochGenerationSource(bus)
    lost = await source.start_log()
    bus.change_log(EPOCH_CHANGES_BUCKET).vanish()
    return bus, source, lost


async def restored_records(source: EpochGenerationSource) -> list[EpochChange]:
    state = await source.log_state()
    assert state is not None
    found = [await source.epoch_at(seq) for seq in range(1, state.last_seq + 1)]
    return [record for record in found if record is not None]
