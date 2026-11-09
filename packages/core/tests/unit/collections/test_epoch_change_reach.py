"""A whole-table epoch names its reach: a load's or a clear's key prefixes, or an operator's reset (3tears 0.95.0).

The contract this pins:

- a record names key prefixes or a reason only as a whole-table record; the reason is ``"reset"`` or none; a reset
  names no prefixes; each prefix names a column;
- the stored body carries both, and an older body without them reads as before;
- ``advance_recording`` hands them to a source that takes them, names exactly one prefix as the older ``partition``
  too, and hands a source built before them neither;
- a registry tells a derived cache each prefix (``on_partition``), a reset as a reset (``on_table_reset``), and counts
  every whole-table drop by why (``collection_whole_table_drops_total``): an older writer's whole record apart.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from prometheus_client import REGISTRY

from threetears.core.collections import CollectionRegistry
from threetears.core.collections.epoch_change import (
    RESET_REASON,
    ChangedRow,
    EpochChange,
    advance_recording,
    epoch_change_body,
    read_epoch_change_body,
)

_TABLE = "races"


def _drops(why: str) -> float:
    return (
        REGISTRY.get_sample_value("collection_whole_table_drops_total", {"table": _TABLE, "why": why}) or 0.0
    )


def test_a_reach_is_named_only_on_a_whole_table_record() -> None:
    with pytest.raises(ValueError, match="whole-table"):
        EpochChange(table=_TABLE, token="i:1", rows=[ChangedRow(ids=["a"])], prefixes=[{"office_key": "a"}])
    with pytest.raises(ValueError, match="whole-table"):
        EpochChange(table=_TABLE, token="i:1", keys_unknown=True, reason=RESET_REASON)


def test_a_reason_is_reset_or_none_and_a_reset_names_no_prefix() -> None:
    with pytest.raises(ValueError, match="reason"):
        EpochChange(table=_TABLE, token="i:1", whole_table=True, reason="load")
    with pytest.raises(ValueError, match="never key prefixes"):
        EpochChange(table=_TABLE, token="i:1", whole_table=True, reason=RESET_REASON, prefixes=[{"k": "a"}])
    with pytest.raises(ValueError, match="at least one"):
        EpochChange(table=_TABLE, token="i:1", whole_table=True, prefixes=[{}])


def test_the_stored_body_carries_the_reach_and_an_older_body_reads_as_before() -> None:
    body = epoch_change_body(_TABLE, None, prefixes=[{"office_key": "a"}, {"office_key": "b"}])
    record = read_epoch_change_body(body, _TABLE, "i:3")
    assert record.whole_table and record.prefixes == [{"office_key": "a"}, {"office_key": "b"}]
    reset = read_epoch_change_body(epoch_change_body(_TABLE, None, reason=RESET_REASON), _TABLE, "i:4")
    assert reset.reason == RESET_REASON and reset.prefixes is None
    older = read_epoch_change_body(b'{"table": "races", "whole_table": true}', _TABLE, "i:5")
    assert older.whole_table and older.prefixes is None and older.reason is None


class _Recording:
    """a source that records changes, taking the reach."""

    def __init__(self) -> None:
        self.asked: list[dict[str, Any]] = []

    async def current(self, table_name: str) -> str:
        return "i:0"

    async def advance(self, table_name: str) -> str:
        return "i:1"

    async def advance_with_change(
        self,
        table_name: str,
        rows: Sequence[ChangedRow] | None,
        *,
        keys_unknown: bool = False,
        tenant: str | None = None,
        partition: Mapping[str, str] | None = None,
        prefixes: Sequence[Mapping[str, str]] | None = None,
        reason: str | None = None,
    ) -> str:
        self.asked.append({"partition": partition, "prefixes": prefixes, "reason": reason})
        return "i:1"


class _Older:
    """a source built before the reach: it takes neither."""

    def __init__(self) -> None:
        self.asked: list[dict[str, Any]] = []

    async def current(self, table_name: str) -> str:
        return "i:0"

    async def advance(self, table_name: str) -> str:
        return "i:1"

    async def advance_with_change(
        self,
        table_name: str,
        rows: Sequence[ChangedRow] | None,
        *,
        keys_unknown: bool = False,
        tenant: str | None = None,
        partition: Mapping[str, str] | None = None,
    ) -> str:
        self.asked.append({"partition": partition})
        return "i:1"


async def test_the_reach_reaches_a_source_that_takes_it_and_one_prefix_is_the_older_partition_too() -> None:
    source = _Recording()
    await advance_recording(source, _TABLE, None, prefixes=[{"office_key": "a"}, {"office_key": "b"}])
    await advance_recording(source, _TABLE, None, prefixes=[{"office_key": "a"}])
    await advance_recording(source, _TABLE, None, reason=RESET_REASON)
    assert source.asked == [
        {"partition": None, "prefixes": [{"office_key": "a"}, {"office_key": "b"}], "reason": None},
        {"partition": {"office_key": "a"}, "prefixes": [{"office_key": "a"}], "reason": None},
        {"partition": None, "prefixes": None, "reason": RESET_REASON},
    ]


async def test_a_source_built_before_the_reach_is_handed_none_of_it() -> None:
    source = _Older()
    await advance_recording(source, _TABLE, None, prefixes=[{"office_key": "a"}, {"office_key": "b"}])
    await advance_recording(source, _TABLE, None, reason=RESET_REASON)
    # several prefixes are no one partition: an older follower reads the record as the whole table, never misled
    assert source.asked == [{"partition": None}, {"partition": None}]


async def test_a_registry_tells_each_prefix_and_a_reset_and_counts_only_whole_drops() -> None:
    registry = CollectionRegistry()
    heard: list[Any] = []
    registry.register_derived_cache(
        _TABLE,
        on_row=lambda message: None,
        on_table_dropped=lambda: heard.append("dropped"),
        on_partition=lambda prefix: heard.append(dict(prefix)),
        on_table_reset=lambda: heard.append("reset"),
    )
    resets, records = _drops("reset"), _drops("whole_record")
    await registry.apply_epoch_change(
        EpochChange(table=_TABLE, token="i:1", whole_table=True, prefixes=[{"office_key": "a"}, {"office_key": "b"}])
    )
    await registry.apply_epoch_change(EpochChange(table=_TABLE, token="i:2", whole_table=True, reason=RESET_REASON))
    await registry.apply_epoch_change(EpochChange(table=_TABLE, token="i:3", whole_table=True))
    assert heard == [{"office_key": "a"}, {"office_key": "b"}, "reset", "dropped"]
    assert (_drops("reset"), _drops("whole_record")) == (resets + 1, records + 1)
