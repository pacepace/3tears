"""a ring beside the log another keeper keeps scrubs its own tables by their keys, never the log by sequence.

A reports tool pod runs a keeper beside the hub's (``keeps_log=False``). A read of the change log by sequence
(``$JS.API.STREAM.MSG.GET``) carries its subject in the request body, where no grant can narrow it, so a pod's grant
never names it (:mod:`threetears.nats.subject_permissions`): on a real broker every such read of its scrub was refused
and waited out its 5 s deadline (local burst replay, 2026-11-24). What this holds, on the in-memory log with its read
by sequence refused as the broker refuses it: the ring beside's scrub never reads by sequence, and still finds its own
table's epoch whose note is missing, skipping every other database's epoch.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from threetears.core.collections.change_note_keeper import ChangeNoteKeeper
from threetears.core.collections.epoch_change import ChangedRow
from threetears.core.ddl.change_notes import ring_read_sql
from threetears.core.testing.kv import FakeNatsClient
from threetears.epoch import EPOCH_CHANGES_BUCKET, EpochGenerationSource
from threetears.nats.change_log import ChangeLogEntry
from threetears.nats.errors import KvReadRefusedError

#: the table the ring beside notes, and the one key of it whose note its ring holds
_MINE = "beside_scrub_reports"
_NOTED = "noted"
#: the ring beside's schema, and the database its records name
_RING = "reports"


class _RingPool:
    """the ring beside's database: its ring fenced at the log's start, past its wait and emptied, so a pass only
    scrubs; it triggers :data:`_MINE` into its ring, whose notes hold :data:`_NOTED` only."""

    def __init__(self, incarnation: str) -> None:
        at = datetime(2026, 11, 24, tzinfo=UTC)
        self._ring = {
            "current_ring": 0,
            "switch_no": 1,
            "switched_at": at,
            "switched_seq": f"{incarnation}:0",
            "floor": f"{incarnation}:0",
            "fence_seq": f"{incarnation}:0",
            "emptied": True,
            "log_incarnation": incarnation,
            "now": at + timedelta(minutes=1),
            "fence_seconds": 600.0,
            "wait_seconds": 600.0,
        }

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        if query == ring_read_sql(_RING):
            return [self._ring]
        if query.startswith("SELECT count(*) AS notes"):
            return [{"notes": 0}]
        if "FROM pg_trigger" in query:
            return [{"table_name": _MINE}]
        if query.startswith("SELECT DISTINCT table_name, key_text FROM"):
            return [{"table_name": _MINE, "key_text": _NOTED}]
        return []

    async def execute(self, query: str, *args: Any) -> str:
        raise AssertionError(f"a scrub writes nothing: {query}")

    def acquire(self) -> Any:
        raise AssertionError("a scrub shares no session")


async def test_its_scrub_never_reads_the_log_by_sequence_and_still_finds_its_own_missing_note(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Flips if the ring beside's scrub reads the shared log by sequence (a pod's grant never names that read, so on a
    real broker each one is refused and waits out its deadline), or if it stops finding its own table's epoch whose
    note is missing, or judges another database's epoch, of another table or of the same name."""
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
    keeper = ChangeNoteKeeper(
        _RingPool(incarnation), notes_schema=_RING, log=source, scrub_every=timedelta(0), keeps_log=False
    )
    await keeper.run_pass()  # the scrub checks up to the head one pass ago

    await source.advance_with_change(_MINE, [ChangedRow(ids=[_NOTED])])
    # an epoch of its own table whose write left no note: appended with no L3 write behind it
    await source.advance_with_change(_MINE, [ChangedRow(ids=["unnoted"])])
    # the owner database's epochs: another table, and its own table of the same name
    await owners.advance_with_change(f"{_MINE}_theirs", [ChangedRow(ids=["the-owners"])])
    await owners.advance_with_change(_MINE, [ChangedRow(ids=["the-owners-of-the-name"])])
    state = await source.log_state()
    assert state is not None
    await keeper.run_pass()

    with caplog.at_level(logging.ERROR):
        scrub = (await keeper.run_pass()).scrub
    said = [r.__dict__["extra_data"]["missing"] for r in caplog.records if "no change note" in r.getMessage()]
    assert by_sequence == [], "a ring beside never reads the shared log by sequence"
    assert said == ["unnoted"], "its own table's missing note is said; no other database's epoch is"
    assert (scrub.checked, scrub.missing, scrub.others) == (2, 1, 2)
    assert scrub.position == f"{incarnation}:{state.last_seq}"
