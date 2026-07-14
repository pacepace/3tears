"""Grandfather-father-son (GFS) retention for backup pruning.

Each backup is classified into exactly one tier by its calendar date -- the 1st of a month is a
*monthly*, a Sunday is a *weekly*, anything else is a *daily* (lifted from the aibots policy). The
newest N of each tier are kept; the rest are pruned. :meth:`GfsRetention.select` is pure (it takes
records, returns a keep/delete split) so the engine can dry-run a prune before deleting anything.

The record is a small generic value (:class:`BackupRecord`), not tied to any app's backup row, so
this module composes over an ``ObjectStore`` listing just as well as a database table.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime

from threetears.observe import get_logger

from threetears.backup.config import BackupConfig

__all__ = ["BackupRecord", "GfsRetention", "RetentionDecision", "classify_tier"]

log = get_logger(__name__)

MONTHLY = "monthly"
WEEKLY = "weekly"
DAILY = "daily"


@dataclass(frozen=True, slots=True)
class BackupRecord:
    """One backup as retention sees it: a key, when it was made, and its size."""

    key: str
    created_at: datetime
    size_bytes: int = 0


@dataclass(frozen=True, slots=True)
class RetentionDecision:
    """The outcome of applying a policy: what to keep, what to prune."""

    keep: tuple[BackupRecord, ...]
    delete: tuple[BackupRecord, ...]


def classify_tier(backup_date: date) -> str:
    """Classify a backup's date into its GFS tier.

    :param backup_date: the date the backup was taken.
    :return: ``"monthly"`` (1st of month), ``"weekly"`` (Sunday), else ``"daily"``.
    """
    if backup_date.day == 1:
        tier = MONTHLY
    elif backup_date.weekday() == 6:  # Sunday
        tier = WEEKLY
    else:
        tier = DAILY
    return tier


class GfsRetention:
    """A grandfather-father-son retention policy (keep newest N per tier).

    :param daily: daily backups to keep (>= 1).
    :param weekly: weekly backups to keep (>= 1).
    :param monthly: monthly backups to keep (>= 1).
    """

    def __init__(self, *, daily: int = 7, weekly: int = 4, monthly: int = 3) -> None:
        for label, value in (("daily", daily), ("weekly", weekly), ("monthly", monthly)):
            if value < 1:
                raise ValueError(f"{label} retention must be >= 1")
        self._limits = {DAILY: daily, WEEKLY: weekly, MONTHLY: monthly}

    @classmethod
    def from_config(cls, config: BackupConfig) -> GfsRetention:
        """Build the policy from a :class:`BackupConfig`."""
        return cls(
            daily=config.retention_daily,
            weekly=config.retention_weekly,
            monthly=config.retention_monthly,
        )

    def select(self, records: Iterable[BackupRecord]) -> RetentionDecision:
        """Split records into keep/delete, keeping the newest N of each tier.

        :param records: the candidate backups (any order).
        :return: the keep/delete decision (records preserved, never mutated).
        """
        ordered = sorted(records, key=lambda r: r.created_at, reverse=True)
        counts = {DAILY: 0, WEEKLY: 0, MONTHLY: 0}
        keep: list[BackupRecord] = []
        delete: list[BackupRecord] = []
        for record in ordered:
            tier = classify_tier(record.created_at.date())
            if counts[tier] < self._limits[tier]:
                counts[tier] += 1
                keep.append(record)
            else:
                delete.append(record)
        log.info(
            "gfs retention evaluated",
            extra={
                "extra_data": {
                    "keep": len(keep),
                    "delete": len(delete),
                    "daily": f"{counts[DAILY]}/{self._limits[DAILY]}",
                    "weekly": f"{counts[WEEKLY]}/{self._limits[WEEKLY]}",
                    "monthly": f"{counts[MONTHLY]}/{self._limits[MONTHLY]}",
                }
            },
        )
        return RetentionDecision(keep=tuple(keep), delete=tuple(delete))
