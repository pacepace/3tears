"""A pod reads one epoch's record by the table's keys, never by the log's sequence.

A message get by sequence (``$JS.API.STREAM.MSG.GET``) carries its subject in the request body, where no grant can
narrow it, so the hub grants no pod that verb on the shared change log. A local replay's writer pod asked for an
epoch's record so while recreating a lost delta file (``DeltaRecreator`` -> ``change`` -> ``NatsChangeLog.at``); every
try was refused and waited out its 5 s deadline. The fake change log refuses a read by sequence on any connection a
test grants keys to, as the broker does; these hold:

- a writer recreating a lost delta file on a pod's grant reads its record by key and recreates the file;
- ``change`` answers the table's record at the epoch on a pod's grant, ``None`` past the log's head, and refuses
  an epoch that is another table's record.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from threetears.core.collections.copy_follower import FollowedTable
from threetears.core.collections.delta_recreate import DeltaRecreator
from threetears.core.collections.epoch_change import ChangedRow
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column
from threetears.core.exceptions import GenerationUnavailableError
from threetears.core.tenancy import epoch_change_filter
from threetears.core.testing.delta_files import read_delta_file
from threetears.core.testing.kv import FakeNatsClient
from threetears.epoch import EPOCH_CHANGES_BUCKET, EpochGenerationSource
from threetears.nats.errors import KvReadRefusedError

_TABLE = "recreate_by_key_results"
_OTHER = "recreate_by_key_tallies"
_FILE = f"{_TABLE}/0123456789abcdef0123456789abcdef.arrow"
_COLUMNS = (Column(name="id", column_type=STRING_TYPE), Column(name="votes", column_type=BIGINT_TYPE))


class _GoneL3:
    """the writer's L3 after the keys were deleted: holds none of them, so each is recreated as a delete marker."""

    def __init__(self) -> None:
        self.statements = 0

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        del sql, params
        self.statements += 1
        return []


async def _pod_source() -> tuple[FakeNatsClient, EpochGenerationSource]:
    bus = FakeNatsClient()
    source = EpochGenerationSource(bus)
    await source.start_log()
    return bus, source


def _grant_the_tables_keys(bus: FakeNatsClient) -> None:
    # a pod's grant: the table's keys, which a named consumer's create subject names; never the message get
    bus.change_log(EPOCH_CHANGES_BUCKET).granted_reads = (epoch_change_filter(_TABLE),)


async def test_the_fake_refuses_a_read_by_sequence_on_a_pods_grant_as_the_broker_does() -> None:
    """Flips if the double lets a pod's connection read by sequence: the test below would then prove nothing."""
    bus, source = await _pod_source()
    written = await source.advance_with_change(_TABLE, [ChangedRow(ids=["k1"])])
    _grant_the_tables_keys(bus)
    with pytest.raises(KvReadRefusedError, match="record by sequence was refused"):
        await bus.change_log(EPOCH_CHANGES_BUCKET).at(int(written.rpartition(":")[2]))


async def test_a_writer_recreating_a_lost_delta_file_on_a_pods_grant_reads_its_record_by_key(tmp_path: Path) -> None:
    """Flips if the recreation reads the epoch's record by the log's sequence (refused on every pod: the copies
    waited while each try ran out its deadline) rather than by the table's keys: the file is recreated from L3, a key
    L3 no longer holds a delete marker."""
    bus, source = await _pod_source()
    await source.advance_with_change(_OTHER, [ChangedRow(ids=["t1"])])
    epoch = await source.advance_with_change(_TABLE, [ChangedRow(ids=["k1"]), ChangedRow(ids=["k2"])], delta_file=_FILE)
    await source.advance_with_change(_OTHER, [ChangedRow(ids=["t2"])])
    _grant_the_tables_keys(bus)
    l3 = _GoneL3()
    table = FollowedTable(name=_TABLE, key=("id",), columns=_COLUMNS, delta_files=True)
    recreator = DeltaRecreator([table], l3, source, root=tmp_path)

    assert await recreator.recreate(_TABLE, _FILE, epoch) == "recreated"

    assert l3.statements == 1, "the record's keys were read from L3 by key, once"
    recreated = read_delta_file(tmp_path / _FILE, table=_TABLE)
    assert [(change.row["id"], change.deleted) for change in recreated.changes] == [("k1", True), ("k2", True)]


async def test_change_on_a_pods_grant_answers_the_tables_record_and_refuses_another_tables_epoch() -> None:
    """Flips if ``change`` reads by sequence, answers a record of another table, or answers anything but ``None``
    for an epoch past the log's head."""
    bus, source = await _pod_source()
    other = await source.advance_with_change(_OTHER, [ChangedRow(ids=["t1"])])
    mine = await source.advance_with_change(_TABLE, [ChangedRow(ids=["k1"])], delta_file=_FILE)
    await source.advance_with_change(_TABLE, [ChangedRow(ids=["k2"])])
    _grant_the_tables_keys(bus)

    record = await source.change(_TABLE, mine)
    assert record is not None and record.delta_file == _FILE and [row.ids for row in record.rows] == [["k1"]]
    with pytest.raises(GenerationUnavailableError, match="not one of"):
        await source.change(_TABLE, other)
    incarnation = mine.rpartition(":")[0]
    assert await source.change(_TABLE, f"{incarnation}:99") is None, "past the log's head: the follower is behind"
