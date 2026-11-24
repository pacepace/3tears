"""The change log's records name their commit's delta file (``design-pod-delta-files.md``, decision 4).

Every record a write appends for a table names the file, a commit too large for one record included, so a follower
that reaches any of them has the whole commit; a write naming none appends records byte for byte as before.
"""

from __future__ import annotations

import json

from threetears.core.collections.epoch_change import ChangedRow
from threetears.core.testing.kv import FakeNatsClient
from threetears.epoch import EPOCH_CHANGES_BUCKET, EpochGenerationReader, EpochGenerationSource

_TABLE = "delta_files_test_results"
_FILE = f"{_TABLE}/0123456789abcdef0123456789abcdef.arrow"


async def test_every_record_of_a_write_names_its_delta_file() -> None:
    """Flips if a record of an opted-in table's commit, the earlier runs of a large one included, does not name the
    commit's delta file."""
    bus = FakeNatsClient()
    source = EpochGenerationSource(bus, database="platform")
    await source.start_log()
    start = await source.read(_TABLE) or f"{(await source.log_state()).incarnation}:0"  # type: ignore[union-attr]
    rows = [ChangedRow(ids=[f"{i:06d}" + "k" * 200]) for i in range(3_000)]
    head = await source.advance_with_change(_TABLE, rows, delta_file=_FILE)
    walk = await EpochGenerationReader(bus).changes_after(_TABLE, start, head)
    assert walk.complete and len(walk.changes) > 1
    assert {change.delta_file for change in walk.changes} == {_FILE}


async def test_a_write_naming_no_delta_file_appends_as_before() -> None:
    """Flips if a table that does not opt in gets a delta file field in its records."""
    bus = FakeNatsClient()
    source = EpochGenerationSource(bus)
    await source.start_log()
    written = await source.advance_with_change(_TABLE, [ChangedRow(ids=["k1"])])
    raw = (await bus.change_log(EPOCH_CHANGES_BUCKET).at(int(written.rpartition(":")[2]))).value  # type: ignore[union-attr]
    assert "delta_file" not in json.loads(raw)
    record = await EpochGenerationReader(bus).change(_TABLE, written)
    assert record is not None and record.delta_file is None


async def test_a_loads_intent_record_names_its_file_unconfirmed_beside_its_loading_writer() -> None:
    """Flips if the log's source drops the loading writer's beat key from a load's intent record (the writer could then
    not tell a crashed hub from one still loading), or stores the file as confirmed."""
    bus = FakeNatsClient()
    source = EpochGenerationSource(bus)
    await source.start_log()
    writer = "write-ledger.platform.hub-0195f3"
    written = await source.advance_with_change(
        _TABLE, None, prefixes=[{"state": "VA"}], unconfirmed_delta_file=_FILE, unconfirmed_writer=writer
    )
    record = await EpochGenerationReader(bus).change(_TABLE, written)
    assert record is not None
    assert (record.delta_file, record.delta_file_unconfirmed, record.unconfirmed_writer) == (_FILE, True, writer)
