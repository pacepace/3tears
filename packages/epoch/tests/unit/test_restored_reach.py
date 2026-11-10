"""A restored whole-table record keeps the reach its change notes name: a load's key prefixes, an operator's reset.

The keeper folds a table's whole-table notes into one :class:`NotedChange`; the restore appends it as the table's
record. Before the notes named their reach, every such record came back plain whole, and a follower of the restored
log read the whole table for a load that reached one state. These pin what the restore appends for each kind.
"""

from __future__ import annotations

import pytest

from threetears.core.collections.change_notes import NotedChange
from threetears.core.collections.epoch_change import RESET_REASON, EpochChange
from threetears.core.testing.kv import FakeNatsClient
from threetears.epoch import EPOCH_CHANGES_BUCKET, EpochGenerationSource
from threetears.nats import set_default_namespace

_TABLE = "lsf_rows"


@pytest.fixture(autouse=True)
def _namespace() -> None:
    set_default_namespace("restoredreach")


async def _restored(change: NotedChange) -> list[EpochChange]:
    """what a restore of ``change`` appends to the new log, record by record."""
    bus = FakeNatsClient()
    source = EpochGenerationSource(bus)
    await source.start_log()
    lost = (await source.log_state()).incarnation  # type: ignore[union-attr]
    bus.change_log(EPOCH_CHANGES_BUCKET).vanish()
    restored = await source.begin_restore(lost)
    assert restored is not None
    await source.finish_restore([change], incarnation=restored, lost=lost, covered_from=0)
    state = await source.log_state()
    assert state is not None
    records = [await source.epoch_at(seq) for seq in range(1, state.last_seq + 1)]
    return [record for record in records if record is not None and record.table == _TABLE]


class TestARestoredWholeRecordKeepsItsReach:
    async def test_a_load_s_prefixes_come_back_as_its_prefixes(self) -> None:
        prefixes = ({"race": "r1"}, {"race": "r2"})
        [record] = await _restored(NotedChange(table=_TABLE, whole_table=True, prefixes=prefixes))
        assert record.whole_table
        assert record.prefixes == [{"race": "r1"}, {"race": "r2"}], "the load's prefixes, never the plain whole table"
        assert record.reason is None

    async def test_one_prefix_is_also_named_as_the_older_partition_as_the_writer_names_it(self) -> None:
        [record] = await _restored(NotedChange(table=_TABLE, whole_table=True, prefixes=({"race": "r1"},)))
        assert record.prefixes == [{"race": "r1"}]
        assert record.partition == {"race": "r1"}

    async def test_a_reset_comes_back_as_a_reset(self) -> None:
        [record] = await _restored(NotedChange(table=_TABLE, whole_table=True, reason=RESET_REASON))
        assert record.whole_table
        assert record.reason == RESET_REASON, "a reset, never a plain whole record"
        assert not record.prefixes

    async def test_a_whole_note_naming_no_reach_comes_back_plain_whole(self) -> None:
        [record] = await _restored(NotedChange(table=_TABLE, whole_table=True))
        assert record.whole_table and not record.prefixes and record.reason is None


class TestANotedChangeNamesReachOnlyAsAWholeTable:
    def test_prefixes_or_a_reason_without_the_whole_table_are_refused(self) -> None:
        with pytest.raises(ValueError, match="whole-table"):
            NotedChange(table=_TABLE, prefixes=({"race": "r1"},))
        with pytest.raises(ValueError, match="whole-table"):
            NotedChange(table=_TABLE, reason=RESET_REASON)

    def test_a_reset_names_no_prefixes_and_a_reason_is_reset_only(self) -> None:
        with pytest.raises(ValueError, match="never key prefixes"):
            NotedChange(table=_TABLE, whole_table=True, prefixes=({"race": "r1"},), reason=RESET_REASON)
        with pytest.raises(ValueError, match="reason"):
            NotedChange(table=_TABLE, whole_table=True, reason="tidy")
