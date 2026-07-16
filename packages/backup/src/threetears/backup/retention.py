"""Grandfather-father-son (GFS) retention for backup pruning.

The policy keeps, for each tier, the **newest backup of each of the most recent N periods**: the
newest per calendar *day* for ``daily`` days, the newest per ISO *week* for ``weekly`` weeks, and
the newest per calendar *month* for ``monthly`` months. A backup survives if it is the
newest-in-its-period for any tier still inside that tier's window; the kept sets are unioned.

This is deliberately *period-based*, not date-classification-based. An earlier model tagged a
backup as monthly/weekly only if it happened to fall on the 1st / a Sunday -- so a schedule that
ever missed those days left a month or week with no keeper. Promoting the newest backup *within*
each period instead means every populated day/week/month keeps a representative regardless of when
in the period the backup ran.

:meth:`GfsRetention.select` is pure (records in, keep/delete split out, nothing mutated) so the
engine can dry-run a prune before deleting anything. The record is a small generic value
(:class:`BackupRecord`), so this composes over an ``ObjectStore`` listing as readily as a DB table.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime

from threetears.observe import get_logger

from threetears.backup.config import BackupConfig

__all__ = ["BackupRecord", "GfsRetention", "RetentionDecision"]

log = get_logger(__name__)

#: a period identity derived from a timestamp (calendar day, ISO week, or calendar month).
_PeriodKey = tuple[int, ...]


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


def _day_key(when: datetime) -> _PeriodKey:
    return (when.year, when.month, when.day)


def _week_key(when: datetime) -> _PeriodKey:
    iso = when.isocalendar()  # ISO week is year-boundary safe (late Dec can be week 1 of next year)
    return (iso.year, iso.week)


def _month_key(when: datetime) -> _PeriodKey:
    return (when.year, when.month)


class GfsRetention:
    """A grandfather-father-son retention policy (newest backup per recent day/week/month).

    :param daily: number of recent days to keep a backup for (>= 1).
    :param weekly: number of recent ISO weeks to keep a backup for (>= 1).
    :param monthly: number of recent months to keep a backup for (>= 1).
    """

    def __init__(self, *, daily: int = 7, weekly: int = 4, monthly: int = 3) -> None:
        for label, value in (("daily", daily), ("weekly", weekly), ("monthly", monthly)):
            if value < 1:
                raise ValueError(f"{label} retention must be >= 1")
        self._daily = daily
        self._weekly = weekly
        self._monthly = monthly

    @classmethod
    def from_config(cls, config: BackupConfig) -> GfsRetention:
        """Build the policy from a :class:`BackupConfig`."""
        return cls(
            daily=config.retention_daily,
            weekly=config.retention_weekly,
            monthly=config.retention_monthly,
        )

    def select(self, records: Iterable[BackupRecord]) -> RetentionDecision:
        """Split records into keep/delete by the newest-per-period policy.

        :param records: the candidate backups (any order).
        :return: the keep/delete decision (input records preserved, never mutated).
        """
        ordered = sorted(records, key=lambda r: (r.created_at, r.key), reverse=True)
        daily = self._period_keepers(ordered, _day_key, self._daily)
        weekly = self._period_keepers(ordered, _week_key, self._weekly)
        monthly = self._period_keepers(ordered, _month_key, self._monthly)
        keep_keys = daily | weekly | monthly

        keep = tuple(r for r in ordered if r.key in keep_keys)
        delete = tuple(r for r in ordered if r.key not in keep_keys)
        log.info(
            "gfs retention evaluated",
            extra={
                "extra_data": {
                    "keep": len(keep),
                    "delete": len(delete),
                    "daily_keepers": len(daily),
                    "weekly_keepers": len(weekly),
                    "monthly_keepers": len(monthly),
                }
            },
        )
        return RetentionDecision(keep=keep, delete=delete)

    @staticmethod
    def _period_keepers(
        ordered: list[BackupRecord],
        period_of: Callable[[datetime], _PeriodKey],
        keep_periods: int,
    ) -> set[str]:
        """Keys of the newest backup in each of the ``keep_periods`` most recent populated periods.

        ``ordered`` is newest-first, so the first record seen for a period *is* that period's newest,
        and periods are first-seen in most-recent-first order.
        """
        newest_in_period: dict[_PeriodKey, str] = {}
        for record in ordered:
            newest_in_period.setdefault(period_of(record.created_at), record.key)
        recent_periods = list(newest_in_period)[:keep_periods]
        return {newest_in_period[period] for period in recent_periods}
