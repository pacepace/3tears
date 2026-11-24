"""one change log, two databases that note a table of the same name: each record names the database it was written in.

The hub's platform database and identity-core's both advance through one change log, and the log names a record's
table by its bare name: both note ``coordination_revocations`` (3tears' coordination tables, migrated into each
database's own schema), so a record of one is a record of the other's table too (RC9, 2026-11-16). What these hold,
on the in-memory log (the rings and the scrubs on a real database are pinned in
``tests/integration/test_change_notes_live.py``, ``TestATableNamedInTwoDatabases``):

- a source built with its database names it on every record it appends, the restore's included;
- a reader built with its database walks only its own database's records of a table and those naming none (an
  older writer's), so the other database's write never evicts, drops or makes it behind; it is current on the log's
  head all the same;
- a reader naming no database (a pod, a reader built before) walks every record, as before;
- a reader built before the field reads a record that carries it (it ignores a field it does not know);
- a keyed record assembled in the restore's worker carries its database as one written whole does;
- a ring beside another keeper's log is refused without a database, and a keeper whose log names a database other
  than its ring's schema is refused, both at construction.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from threetears.core.collections.change_note_keeper import ChangeNoteKeeper
from threetears.core.collections.change_notes import NotedChange
from threetears.core.collections.epoch_change import (
    ChangedRow,
    epoch_change_body,
    keyed_record_form,
    read_epoch_change_body,
)
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.testing.kv import FakeNatsClient
from threetears.epoch import EPOCH_CHANGES_BUCKET, EpochGenerationReader, EpochGenerationSource

#: the name both databases give a table they each note
_SHARED = "databases_test_revocations"

#: a pool the keeper's constructor holds and never uses (it does no I/O)
_NO_POOL: Any = object()


async def _walk(reader: EpochGenerationReader, after: str) -> list[str]:
    head = await reader.read(_SHARED)
    assert head is not None
    walk = await reader.changes_after(_SHARED, after, head)
    assert walk.complete, walk
    walked: list[str] = []
    for change in walk.changes:
        walked.extend([row.ids[0] for row in change.rows] or ["*" if change.whole_table else "?"])
    return walked


class TestARecordNamesItsDatabase:
    async def test_every_append_and_every_restored_record_names_the_sources_database(self) -> None:
        bus = FakeNatsClient()
        hub = EpochGenerationSource(bus, database="platform")
        await hub.start_log()
        written = await hub.advance_with_change(_SHARED, [ChangedRow(ids=["k1"])])
        record = await EpochGenerationReader(bus).change(_SHARED, written)
        assert record is not None and record.database == "platform"

        lost = (await hub.log_state()).incarnation  # type: ignore[union-attr]
        bus.change_log(EPOCH_CHANGES_BUCKET).vanish()
        restored = await hub.begin_restore(lost)
        assert restored is not None
        await hub.finish_restore(
            [NotedChange(table=_SHARED, rows=(ChangedRow(ids=["k2"]),)), NotedChange(table=_SHARED, whole_table=True)],
            incarnation=restored,
            lost=lost,
            covered_from=0,
        )
        head = await hub.read(_SHARED)
        assert head is not None
        walk = await EpochGenerationReader(bus).changes_after(_SHARED, f"{restored}:0", head)
        assert [change.database for change in walk.changes] == ["platform", "platform"]

    async def test_a_source_naming_none_writes_as_before(self) -> None:
        bus = FakeNatsClient()
        source = EpochGenerationSource(bus)
        await source.start_log()
        written = await source.advance_with_change(_SHARED, [ChangedRow(ids=["k1"])])
        raw = (await bus.change_log(EPOCH_CHANGES_BUCKET).at(int(written.rpartition(":")[2]))).value  # type: ignore[union-attr]
        assert "database" not in json.loads(raw), "an unnamed record is byte for byte an older writer's"


class TestAFollowerWalksOnlyItsOwnDatabasesRecords:
    async def test_the_other_databases_rows_whole_table_and_keys_unknown_never_reach_it(self) -> None:
        """Flips if identity-core's follower of its own ``coordination_revocations`` applies the hub's record of the
        hub's table of that name: a hub truncate drops identity's whole cache, a hub key evicts identity's row of the
        same key, and a hub keys-unknown record is paged from identity's ring (behind when its bracket is older than
        that ring's coverage)."""
        bus = FakeNatsClient()
        hub = EpochGenerationSource(bus, database="platform")
        identity = EpochGenerationSource(bus, database="identity")
        await hub.start_log()
        reader = EpochGenerationReader(bus, database="identity")
        follower = CollectionRegistry()
        follower.follow_generation(_SHARED)
        heard: list[str] = []
        dropped: list[bool] = []
        follower.register_derived_cache(
            _SHARED,
            on_row=lambda message: heard.append(str(message.ids[0])),
            on_table_dropped=lambda: dropped.append(True),
            on_partition=lambda _prefix: dropped.append(True),
        )
        start = await reader.read(_SHARED)
        assert start is not None
        await follower.catch_up_generation(_SHARED, start, changes=reader)

        await identity.advance_with_change(_SHARED, [ChangedRow(ids=["mine-1"])])
        await hub.advance_with_change(_SHARED, [ChangedRow(ids=["theirs"])])
        await hub.advance_with_change(_SHARED, None)
        long_ago = datetime.now(tz=UTC) - timedelta(days=1)
        await hub.advance_with_change(
            _SHARED, None, keys_unknown=True, noted_between=(long_ago, long_ago + timedelta(seconds=1))
        )
        await identity.advance_with_change(_SHARED, [ChangedRow(ids=["mine-2"])])
        last = await hub.advance_with_change(_SHARED, [ChangedRow(ids=["theirs-last"])])

        assert await _walk(reader, start) == ["mine-1", "mine-2"]
        await follower.catch_up_generation(_SHARED, await reader.read(_SHARED), changes=reader)
        assert heard == ["mine-1", "mine-2"], "only its own database's rows"
        assert not dropped, "the other database's truncate never drops this one's table"
        assert not follower.behind_on(_SHARED), "the other database's keys-unknown record is never paged from here"
        assert follower.generation_marks.recorded(_SHARED) == last, "current on the log's head all the same"

    async def test_an_older_writers_record_naming_no_database_is_walked_by_every_follower(self) -> None:
        bus = FakeNatsClient()
        older = EpochGenerationSource(bus)
        await older.start_log()
        start = await EpochGenerationReader(bus).read(_SHARED)
        assert start is not None
        await older.advance_with_change(_SHARED, [ChangedRow(ids=["unnamed"])])
        await EpochGenerationSource(bus, database="platform").advance_with_change(_SHARED, [ChangedRow(ids=["hubs"])])
        assert await _walk(EpochGenerationReader(bus, database="identity"), start) == ["unnamed"]
        assert await _walk(EpochGenerationReader(bus, database="platform"), start) == ["unnamed", "hubs"]
        assert await _walk(EpochGenerationReader(bus), start) == ["unnamed", "hubs"], (
            "a reader naming no database (a pod's, an older one) walks every record, as before"
        )


class TestAnOlderReaderReadsANewRecord:
    def test_a_field_the_reader_does_not_know_is_ignored(self) -> None:
        """Flips if the stored body refuses a field it does not know: a reader built before ``database`` (every
        replica not yet upgraded, during a rolling upgrade) would then fail on every new record."""
        body = json.loads(epoch_change_body(_SHARED, [ChangedRow(ids=["k"])]))
        body["a_field_added_later"] = "x"
        record = read_epoch_change_body(json.dumps(body).encode(), _SHARED, "inc:1")
        assert [row.ids for row in record.rows] == [["k"]] and record.database is None


class TestARestoresWorkerAssemblesTheSameForm:
    def test_a_keyed_record_assembled_from_its_form_names_the_database(self) -> None:
        form = keyed_record_form("identity")
        assembled = form.head + json.dumps(_SHARED).encode() + form.middle
        assembled += form.row_head + json.dumps(["k"]).encode() + form.row_tail + form.tail
        assert assembled == epoch_change_body(_SHARED, [ChangedRow(ids=["k"])], database="identity")
        assert read_epoch_change_body(assembled, _SHARED, "inc:1").database == "identity"


class TestAKeeperKnowsItsDatabase:
    def test_a_ring_beside_the_log_is_refused_without_one(self) -> None:
        """Flips if a second database's ring starts with no name on its writes: its records and the owner's of a
        table they both note could not be told apart, by either scrub or by any follower."""
        unnamed = EpochGenerationSource(FakeNatsClient())
        with pytest.raises(ValueError, match="names its database"):
            ChangeNoteKeeper(_NO_POOL, notes_schema="identity", log=unnamed, keeps_log=False)
        ChangeNoteKeeper(_NO_POOL, notes_schema="platform", log=unnamed)  # the owner of a log no one shares, as before

    def test_its_log_names_its_rings_schema(self) -> None:
        named = EpochGenerationSource(FakeNatsClient(), database="platform")
        with pytest.raises(ValueError, match="ring's schema"):
            ChangeNoteKeeper(_NO_POOL, notes_schema="identity", log=named, keeps_log=False)
        with pytest.raises(ValueError, match="ring's schema"):
            ChangeNoteKeeper(_NO_POOL, notes_schema="identity", log=named)
        ChangeNoteKeeper(_NO_POOL, notes_schema="platform", log=named, keeps_log=False)
