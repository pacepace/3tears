"""Unit tests for GFS retention (tier classification + keep/delete selection)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import SecretStr

from threetears.backup.config import BackupConfig
from threetears.backup.retention import BackupRecord, GfsRetention, classify_tier


def _rec(iso: str) -> BackupRecord:
    return BackupRecord(key=f"backups/{iso}.enc", created_at=datetime.fromisoformat(iso).replace(tzinfo=UTC))


@pytest.mark.parametrize(
    ("iso", "tier"),
    [
        ("2026-07-01", "monthly"),  # 1st of month
        ("2026-04-01", "monthly"),
        ("2026-07-05", "weekly"),  # Sunday
        ("2026-06-28", "weekly"),
        ("2026-07-07", "daily"),  # Tuesday
        ("2026-07-09", "daily"),
    ],
)
def test_classify_tier(iso: str, tier: str) -> None:
    assert classify_tier(date.fromisoformat(iso)) == tier


def test_keeps_newest_per_tier_prunes_the_rest() -> None:
    policy = GfsRetention(daily=2, weekly=2, monthly=3)
    records = [
        _rec("2026-07-01"),  # monthly (newest monthly)
        _rec("2026-06-01"),  # monthly
        _rec("2026-05-01"),  # monthly
        _rec("2026-04-01"),  # monthly — 4th, over the limit of 3 -> delete
        _rec("2026-07-05"),  # weekly
        _rec("2026-06-28"),  # weekly
        _rec("2026-06-21"),  # weekly — 3rd, over the limit of 2 -> delete
        _rec("2026-07-09"),  # daily (newest daily)
        _rec("2026-07-08"),  # daily
        _rec("2026-07-07"),  # daily — 3rd, over the limit of 2 -> delete
    ]

    decision = policy.select(records)
    deleted_keys = {r.key for r in decision.delete}

    assert deleted_keys == {"backups/2026-04-01.enc", "backups/2026-06-21.enc", "backups/2026-07-07.enc"}
    assert len(decision.keep) == 7
    # the kept set is exactly the input minus the deleted set
    assert {r.key for r in decision.keep} == {r.key for r in records} - deleted_keys


def test_empty_input() -> None:
    decision = GfsRetention().select([])
    assert decision.keep == ()
    assert decision.delete == ()


def test_select_does_not_mutate_input() -> None:
    records = [_rec("2026-07-09"), _rec("2026-07-01")]
    snapshot = list(records)
    GfsRetention().select(records)
    assert records == snapshot  # order + contents untouched


def test_input_order_does_not_matter() -> None:
    policy = GfsRetention(daily=1, weekly=1, monthly=1)
    ascending = [_rec("2026-07-07"), _rec("2026-07-08"), _rec("2026-07-09")]  # all daily
    descending = list(reversed(ascending))

    keep_a = {r.key for r in policy.select(ascending).keep}
    keep_d = {r.key for r in policy.select(descending).keep}

    assert keep_a == keep_d == {"backups/2026-07-09.enc"}  # newest daily wins regardless of order


def test_from_config() -> None:
    config = BackupConfig(passphrase=SecretStr("pw"), retention_daily=1, retention_weekly=5, retention_monthly=9)
    policy = GfsRetention.from_config(config)
    # 6 dailies, keep only 1 (the newest)
    dailies = [_rec(f"2026-07-{d:02d}") for d in (7, 8, 9, 10, 13, 14)]  # Tue..Mon, none a Sunday/1st
    decision = policy.select(dailies)
    assert len(decision.keep) == 1
    assert decision.keep[0].key == "backups/2026-07-14.enc"


@pytest.mark.parametrize("field", ["daily", "weekly", "monthly"])
def test_rejects_zero_retention(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        GfsRetention(**{field: 0})
