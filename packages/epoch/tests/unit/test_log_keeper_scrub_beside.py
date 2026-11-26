"""a keeper beside the log another keeper keeps scrubs its own tables by their keys, never the log by sequence.

A reports tool pod runs a keeper beside the hub's (``keeps_log=False``). A read of the change log by sequence
(``$JS.API.STREAM.MSG.GET``) carries its subject in the request body, where no grant can narrow it, so a pod's grant
never names it (:mod:`threetears.nats.subject_permissions`): on a real broker every such read of its scrub was refused
and waited out its 5 s deadline (local burst replay, 2026-11-24). What this holds, on the in-memory log with its read
by sequence refused as the broker refuses it: the keeper beside's scrub never reads by sequence, and still finds its own
table's records a restore could not read back (a table carrying no epoch), skipping every other database's record.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from threetears.core.collections.epoch_change import ChangedRow
from threetears.core.collections.log_keeper import LogKeeper
from threetears.core.collections.schema_backed import STRING_TYPE, Column, TableSchema
from threetears.core.ddl.keeper_marks import MARKS_TABLE
from threetears.core.ddl.row_epoch import row_versions
from threetears.core.eviction import Declared, Policy
from threetears.core.tenancy import TENANT_BY_SCHEMA
from threetears.core.testing.kv import FakeNatsClient
from threetears.epoch import EPOCH_CHANGES_BUCKET, EpochGenerationSource
from threetears.nats.change_log import ChangeLogEntry
from threetears.nats.errors import KvReadRefusedError

#: the keeper beside's table that carries no epoch: a restore cannot read it back
_MINE = "beside_scrub_reports"
#: its table a restore reads back: versioned, declared in the catalog
_VERSIONED = TableSchema(
    name="beside_scrub_results",
    primary_key=("state", "id"),
    columns=[Column("state", STRING_TYPE), Column("id", STRING_TYPE)],
    tenancy=TENANT_BY_SCHEMA,
    row_epoch=row_versions("state", epochs="beside_scrub_epochs", scope_prefix="state:"),
    l1_eviction=Declared(Policy.EVICTABLE, "a test's versioned table"),
)
#: the keeper beside's schema, and the database its records name
_RING = "reports"


def _declaration() -> str:
    """the restore declaration the versioned table's DDL gives its restore index, as the catalog then holds it."""
    [comment] = [each for each in _VERSIONED.create_statements(yugabyte=True) if each.startswith("COMMENT")]
    return comment.split(" IS '", 1)[1].rstrip("'").replace("''", "'")


class _KeeperPool:
    """the keeper beside's database: its marks (one of the log at its start), its schema's two tables, the catalog's
    one restore declaration, and the database's clock."""

    def __init__(self, incarnation: str) -> None:
        at = datetime(2026, 11, 24, tzinfo=UTC)
        self.now = at + timedelta(minutes=1)
        self.marks: list[dict[str, Any]] = [{"log_incarnation": incarnation, "seq": 0, "at": at, "id": "m0"}]

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        if query.startswith("SELECT now()"):
            return [{"now": self.now}]
        if MARKS_TABLE in query:
            if 'ORDER BY t."at" DESC' in query:
                return sorted(self.marks, key=lambda mark: mark["at"], reverse=True)[:1]
            if 't."seq" < $2' in query:
                return []
            return [mark for mark in self.marks if mark["log_incarnation"] == args[0] and mark["seq"] > args[1]]
        if "information_schema.tables" in query:
            return [{"table_name": _MINE}, {"table_name": _VERSIONED.name}]
        if "pg_description" in query:
            return [
                {"table_schema": _RING, "table_name": _VERSIONED.name, "declared": _declaration(), "run_markers": False}
            ]
        return []

    async def execute(self, query: str, *args: Any) -> str:
        assert query.startswith("INSERT INTO") and MARKS_TABLE in query, f"a scrub writes nothing but a mark: {query}"
        incarnation, seq, at = args
        self.marks.append({"log_incarnation": incarnation, "seq": seq, "at": at, "id": f"m{len(self.marks)}"})
        return "INSERT 0 1"

    def acquire(self) -> Any:
        raise AssertionError("a scrub shares no session")


async def test_its_scrub_never_reads_the_log_by_sequence_and_still_finds_its_own_unrestorable_records(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Flips if the keeper beside's scrub reads the shared log by sequence (a pod's grant never names that read, so on a
    real broker each one is refused and waits out its deadline), or if it stops finding its own table's records a
    restore could not read back, counts its versioned table's, or judges another database's record, of another table
    or of the same name."""
    bus = FakeNatsClient()
    owners = EpochGenerationSource(bus, database="platform")
    incarnation = await owners.start_log()
    source = EpochGenerationSource(bus, database=_RING)
    by_sequence: list[int] = []

    async def refused(seq: int) -> ChangeLogEntry | None:
        by_sequence.append(seq)
        subject = f"$JS.API.STREAM.MSG.GET.KV_{EPOCH_CHANGES_BUCKET}"
        raise KvReadRefusedError(f"permissions violation for subject {subject}", subject=subject)

    monkeypatch.setattr(bus.change_log(EPOCH_CHANGES_BUCKET), "at", refused)
    keeper = LogKeeper(_KeeperPool(incarnation), schema=_RING, log=source, scrub_every=timedelta(0), keeps_log=False)
    await keeper.run_pass()  # the scrub checks up to the head one pass ago

    await source.advance_with_change(_MINE, [ChangedRow(ids=["first"])])
    await source.advance_with_change(_VERSIONED.name, [ChangedRow(ids=["CA", "read-back"])])
    await source.advance_with_change(_MINE, [ChangedRow(ids=["second"])])
    # the owner database's epochs: another table, and its own table of the same name
    await owners.advance_with_change(f"{_MINE}_theirs", [ChangedRow(ids=["the-owners"])])
    await owners.advance_with_change(_MINE, [ChangedRow(ids=["the-owners-of-the-name"])])
    state = await source.log_state()
    assert state is not None
    await keeper.run_pass()

    with caplog.at_level(logging.ERROR):
        scrub = (await keeper.run_pass()).scrub
    said = [r.__dict__["extra_data"]["table"] for r in caplog.records if "cannot read back" in r.getMessage()]
    assert by_sequence == [], "a keeper beside never reads the shared log by sequence"
    assert said == [_MINE, _MINE], "its own table's records are said; no other database's record is"
    assert (scrub.checked, scrub.missing, scrub.others) == (3, 2, 2)
    assert scrub.position == f"{incarnation}:{state.last_seq}"
