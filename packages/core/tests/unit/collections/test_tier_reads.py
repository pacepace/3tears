"""The tier-read counters: what a take returns and starts again, and the one log line a period.

Which tier a collection's key read is counted against is tested with the pods of
``test_table_drop_and_derived_caches`` (``TestReadsAreCountedByTier``).
"""

from __future__ import annotations

import logging
import pytest

from threetears.core.collections.tier_reads import (
    ReadTier,
    hold_tier_reads_timer,
    log_tier_reads,
    record_tier_read,
    release_tier_reads_timer,
)


@pytest.fixture(autouse=True)
def _fresh_counts() -> None:
    log_tier_reads()


def test_a_take_starts_the_counts_again() -> None:
    record_tier_read("t", ReadTier.L1)
    record_tier_read("t", ReadTier.L1)
    record_tier_read("t", ReadTier.L3)
    assert log_tier_reads() == {"t": {"l1": 2, "l3": 1}}
    assert log_tier_reads() == {}


def test_the_log_line_names_each_tables_reads_and_nothing_is_logged_when_none_happened(
    caplog: pytest.LogCaptureFixture,
) -> None:
    record_tier_read("namespaces", ReadTier.L3)
    with caplog.at_level(logging.INFO, logger="threetears.core.collections.tier_reads"):
        assert log_tier_reads() == {"namespaces": {"l3": 1}}
        assert log_tier_reads() == {}
    assert [r.getMessage() for r in caplog.records].count("collection reads by tier") == 1


@pytest.mark.asyncio
async def test_the_process_timer_runs_once_and_the_last_release_logs_the_last_reads(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="threetears.core.collections.tier_reads"):
        hold_tier_reads_timer()
        hold_tier_reads_timer()
        record_tier_read("t", ReadTier.L2)
        await release_tier_reads_timer()
        assert "collection reads by tier" not in [r.getMessage() for r in caplog.records]
        await release_tier_reads_timer()
    assert [r.getMessage() for r in caplog.records].count("collection reads by tier") == 1
    assert log_tier_reads() == {}


@pytest.mark.asyncio
async def test_releasing_a_timer_nothing_holds_is_refused() -> None:
    with pytest.raises(RuntimeError, match="released more times than it was held"):
        await release_tier_reads_timer()
