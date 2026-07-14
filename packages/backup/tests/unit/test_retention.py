"""Unit tests for GFS retention (newest-backup-per-recent-period selection)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import SecretStr

from threetears.backup.config import BackupConfig
from threetears.backup.retention import BackupRecord, GfsRetention


def _rec(iso: str) -> BackupRecord:
    """A record at ``iso`` (date or datetime), keyed by the timestamp for readable assertions."""
    return BackupRecord(key=iso, created_at=datetime.fromisoformat(iso).replace(tzinfo=UTC))


def _keys(records: tuple[BackupRecord, ...]) -> set[str]:
    return {r.key for r in records}


def test_empty_input() -> None:
    decision = GfsRetention().select([])
    assert decision.keep == ()
    assert decision.delete == ()


def test_keeps_only_the_newest_backup_of_a_day() -> None:
    # three backups the same day -> the day keeps one representative (its newest).
    records = [_rec("2026-07-08T08:00"), _rec("2026-07-08T12:00"), _rec("2026-07-08T20:00")]
    decision = GfsRetention(daily=7, weekly=4, monthly=3).select(records)

    assert _keys(decision.keep) == {"2026-07-08T20:00"}
    assert _keys(decision.delete) == {"2026-07-08T08:00", "2026-07-08T12:00"}


def test_weekly_promotes_the_newest_of_each_recent_week() -> None:
    # ISO weeks 25 / 26 / 27; weekly=2 keeps the two most recent weeks' newest.
    records = [_rec("2026-06-15"), _rec("2026-06-22"), _rec("2026-06-29")]
    decision = GfsRetention(daily=1, weekly=2, monthly=1).select(records)

    assert _keys(decision.keep) == {"2026-06-29", "2026-06-22"}  # weeks 27 + 26
    assert _keys(decision.delete) == {"2026-06-15"}  # week 25 falls outside the 2-week window


def test_monthly_keeper_does_not_depend_on_the_first_of_month() -> None:
    # THE fragility fix: neither backup is the 1st or a Sunday, yet each month keeps its newest.
    # the old date-classification model would have tagged both "daily" and lost June entirely.
    records = [_rec("2026-06-17"), _rec("2026-07-15")]  # both Wednesdays, mid-month
    decision = GfsRetention(daily=1, weekly=1, monthly=2).select(records)

    assert _keys(decision.keep) == {"2026-06-17", "2026-07-15"}
    assert decision.delete == ()


def test_iso_week_grouping_is_year_boundary_safe() -> None:
    # 2026-12-28 and 2026-12-31 share ISO week 2026-W53; 2027-01-04 is ISO week 2027-W01.
    records = [_rec("2026-12-28"), _rec("2026-12-31"), _rec("2027-01-04")]
    decision = GfsRetention(daily=1, weekly=2, monthly=2).select(records)

    # newest of W53 is the 31st (not the 28th), plus the new-year week; the 28th is redundant.
    assert _keys(decision.keep) == {"2027-01-04", "2026-12-31"}
    assert _keys(decision.delete) == {"2026-12-28"}


def test_keep_and_delete_partition_the_input() -> None:
    records = [_rec(f"2026-07-{d:02d}") for d in range(1, 21)]
    decision = GfsRetention(daily=3, weekly=2, monthly=1).select(records)

    assert _keys(decision.keep).isdisjoint(_keys(decision.delete))
    assert _keys(decision.keep) | _keys(decision.delete) == _keys(tuple(records))
    assert len(decision.keep) + len(decision.delete) == len(records)


def test_tiers_union_without_duplicates() -> None:
    # the newest backup is simultaneously the day/week/month keeper — it appears once.
    records = [_rec("2026-07-10"), _rec("2026-07-13")]
    decision = GfsRetention(daily=7, weekly=4, monthly=3).select(records)

    kept = [r.key for r in decision.keep]
    assert len(kept) == len(set(kept))  # no record kept twice despite qualifying under many tiers
    assert set(kept) == {"2026-07-10", "2026-07-13"}


def test_select_does_not_mutate_input() -> None:
    records = [_rec("2026-07-09"), _rec("2026-07-01")]
    snapshot = list(records)
    GfsRetention().select(records)
    assert records == snapshot


def test_input_order_does_not_matter() -> None:
    policy = GfsRetention(daily=1, weekly=1, monthly=1)
    ascending = [_rec("2026-07-07T01:00"), _rec("2026-07-07T02:00"), _rec("2026-07-07T03:00")]

    keep_ascending = _keys(policy.select(ascending).keep)
    keep_descending = _keys(policy.select(list(reversed(ascending))).keep)

    assert keep_ascending == keep_descending == {"2026-07-07T03:00"}  # newest wins regardless


def test_from_config() -> None:
    config = BackupConfig(passphrase=SecretStr("pw"), retention_daily=1, retention_weekly=1, retention_monthly=1)
    policy = GfsRetention.from_config(config)
    # six backups in one ISO week + month -> daily/weekly/monthly all collapse to the newest.
    records = [_rec(f"2026-07-{d:02d}") for d in (6, 7, 8, 9, 10, 11)]
    decision = policy.select(records)
    assert _keys(decision.keep) == {"2026-07-11"}


@pytest.mark.parametrize("field", ["daily", "weekly", "monthly"])
def test_rejects_zero_retention(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        GfsRetention(**{field: 0})
