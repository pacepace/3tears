"""applying a large commit's change records never holds the event loop in one turn.

A commit of 600,000 keys is 150 change records of 4,000 keys each (a record holds at most 256 KiB). The registry
applied every record of a walk in one turn, and each record's rows as a row message a key even when the copy, told
the record's keys as one Arrow batch, was the table's only cache: a 2.3 s turn (the loop-stall guard, 2026-11-25).
These run such a walk through ``catch_up_generation``; the loop-stall guard (``threetears.core.testing.loop_stall``)
is the check, failing the test naming what held the loop. Records are built on a thread: a test's own setup belongs
off the loop.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("pyarrow")

from threetears.core.collections import CollectionRegistry  # noqa: E402
from threetears.core.collections.epoch_change import ChangedRow, EpochChange, EpochWalk, walk_by_count  # noqa: E402

_TABLE = "votes"
#: keys a record names: a record holds at most 256 KiB, about sixty bytes a named row
_RECORD_KEYS = 4_000


class _Log:
    """the table's write generation and its change records, one commit's keys ``_RECORD_KEYS`` a record."""

    def __init__(self) -> None:
        self.count = 1
        self.records: dict[str, EpochChange] = {}

    def commit(self, keys: int) -> None:
        for start in range(0, keys, _RECORD_KEYS):
            self.count += 1
            token = f"inc:{self.count}"
            rows = [ChangedRow(ids=[str(i)]) for i in range(start, min(start + _RECORD_KEYS, keys))]
            self.records[token] = EpochChange(table=_TABLE, token=token, rows=rows)

    async def change(self, table_name: str, token: str) -> EpochChange | None:
        return self.records.get(token)

    async def changes_after(self, table_name: str, after: str, upto: str) -> EpochWalk:
        return await walk_by_count(self, table_name, after, upto)


def _ignored(*_: Any) -> None:
    return None


async def _caught_up(registry: CollectionRegistry, keys: int) -> None:
    """follow the table from its first epoch, then walk one commit of ``keys`` keys."""
    log = _Log()
    registry.follow_generation(_TABLE)
    await registry.catch_up_generation(_TABLE, "inc:1", changes=log)
    await asyncio.to_thread(log.commit, keys)
    await registry.catch_up_generation(_TABLE, f"inc:{log.count}", changes=log)


async def test_a_copy_alone_is_told_each_record_as_one_batch_and_no_row_message_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flips if a record's rows are applied as a row message a key when the copy (told the record's keys as one
    batch) is the table's only cache, or if a walk of 150 records is applied in one turn of the loop."""
    registry = CollectionRegistry()
    told: list[int] = []
    rows: list[Any] = []
    registry.register_derived_cache(
        _TABLE,
        on_row=rows.append,
        on_table_dropped=_ignored,
        on_partition=_ignored,
        on_keys=lambda keys, epoch: told.append(keys.num_rows),
    )
    drops: list[str] = []
    monkeypatch.setattr(registry, "drop_local_scans", drops.append)
    await _caught_up(registry, 600_000)
    assert told == [_RECORD_KEYS] * 150
    assert rows == []
    # the table's cached scans still drop, once a record
    assert drops == [_TABLE] * 150


async def test_a_cache_taking_row_messages_hears_every_key_between_turns() -> None:
    """Flips if a record's rows are applied as row messages in one turn, or the records of a walk are: the loop
    answers between runs of them."""
    registry = CollectionRegistry()
    heard = [0]

    def on_row(message: Any) -> None:
        heard[0] += 1

    registry.register_derived_cache(_TABLE, on_row=on_row, on_table_dropped=_ignored, on_partition=_ignored)
    await _caught_up(registry, 200_000)
    assert heard[0] == 200_000
