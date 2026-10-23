"""Each key read counts the tier that answered it: L1 (this process), L2 (the shared cache), L3 (the store).

Pods here are those of ``test_table_drop_and_derived_caches``: each its own registry and L1, sharing one bus
(L2) and one L3 dict.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from threetears.core.collections.tier_reads import (
    ReadTier,
    log_tier_reads,
    record_tier_read,
    take_tier_reads,
)

from .test_table_drop_and_derived_caches import _TABLE, _Lossy, _Pod, _save, _Source


@pytest.fixture(autouse=True)
def _fresh_counts() -> None:
    take_tier_reads()


def _pods(bus: _Lossy, source: _Source) -> tuple[_Pod, _Pod, dict[str, dict[str, Any]]]:
    rows: dict[str, dict[str, Any]] = {}
    return _Pod(bus, rows, source, scope="one"), _Pod(bus, rows, source, scope="one"), rows


@pytest.mark.asyncio
async def test_a_row_only_the_store_holds_is_read_from_l3_then_from_l1() -> None:
    bus, source = _Lossy(), _Source()
    pod, _, rows = _pods(bus, source)
    rows["r1"] = {"id": "r1", "member_id": "m1"}

    assert await pod.members.ensure("r1") is not None
    assert take_tier_reads() == {_TABLE: {"l3": 1}}

    assert await pod.members.ensure("r1") is not None
    assert take_tier_reads() == {_TABLE: {"l1": 1}}


@pytest.mark.asyncio
async def test_a_row_another_pod_saved_is_read_from_l2() -> None:
    bus, source = _Lossy(), _Source()
    writer, reader, _ = _pods(bus, source)
    await _save(writer, "r1", "m1")
    take_tier_reads()

    assert await reader.members.ensure("r1") is not None
    assert take_tier_reads() == {_TABLE: {"l2": 1}}


def test_a_take_starts_the_counts_again() -> None:
    record_tier_read("t", ReadTier.L1)
    record_tier_read("t", ReadTier.L1)
    record_tier_read("t", ReadTier.L3)
    assert take_tier_reads() == {"t": {"l1": 2, "l3": 1}}
    assert take_tier_reads() == {}


def test_the_log_line_names_each_tables_reads_and_nothing_is_logged_when_none_happened(
    caplog: pytest.LogCaptureFixture,
) -> None:
    record_tier_read("namespaces", ReadTier.L3)
    with caplog.at_level(logging.INFO, logger="threetears.core.collections.tier_reads"):
        assert log_tier_reads() == {"namespaces": {"l3": 1}}
        assert log_tier_reads() == {}
    assert [r.getMessage() for r in caplog.records].count("collection reads by tier") == 1
