"""Billing periods for a key pool: a UTC calendar month starting on an anchor day.

A key's spend is kept per period, under an account the pool names with the period id, so a new period
starts from zero with no rollover job: the first reservation in it finds nothing spent. A reservation
settles into the period it was made in because its account string already names that period.

The anchor day runs from 1 to 28, so every month has one. With the default anchor of 1 a period is a
calendar month, from 00:00 UTC on the 1st to 00:00 UTC on the 1st of the next month.

Nothing here reads the clock. Every helper takes ``now`` from its caller, which owns the clock (the pool's
injectable ``clock=``), so a rollover is tested at the boundary without waiting.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final

__all__ = ["period_bounds", "period_id"]

#: The anchor day a pool uses unless configured otherwise: periods are calendar months.
_DEFAULT_ANCHOR_DAY: Final = 1

#: The latest anchor day allowed. February has 28 days in a common year, so every month has this day.
_MAX_ANCHOR_DAY: Final = 28


def period_bounds(now: datetime, anchor_day: int = _DEFAULT_ANCHOR_DAY) -> tuple[datetime, datetime]:
    """Return the billing period ``now`` falls in, as its start and its next rollover.

    The period starts at 00:00 UTC on the anchor day of a month, and ends, exclusively, at 00:00 UTC on
    the anchor day of the following month. ``now`` equal to a rollover instant belongs to the period that
    starts there. December's period rolls over into January of the next year.

    :param now: the instant to place; must be timezone-aware, and is read in UTC
    :ptype now: datetime
    :param anchor_day: the day of the month each period starts on, from 1 to 28
    :ptype anchor_day: int
    :return: ``(start, rollover)``, both timezone-aware UTC; ``start <= now < rollover``
    :rtype: tuple[datetime, datetime]
    :raises ValueError: when ``anchor_day`` is outside 1 to 28, or ``now`` is naive
    """
    _check_anchor_day(anchor_day)
    instant = _as_utc(now)
    year, month = instant.year, instant.month
    if instant.day < anchor_day:
        year, month = _previous_month(year, month)
    start = datetime(year, month, anchor_day, tzinfo=UTC)
    next_year, next_month = _next_month(year, month)
    return start, datetime(next_year, next_month, anchor_day, tzinfo=UTC)


def period_id(now: datetime, anchor_day: int = _DEFAULT_ANCHOR_DAY) -> str:
    """Name the billing period ``now`` falls in.

    The id is stable for every instant in one period and sorts in period order for one anchor day:
    ``"2026-11"`` for anchor 1, the calendar month, and ``"2026-11-15"`` for anchor 15, naming the date
    the period starts. The two forms differ so an id never claims to be a calendar month when it isn't.

    :param now: the instant to place; must be timezone-aware, and is read in UTC
    :ptype now: datetime
    :param anchor_day: the day of the month each period starts on, from 1 to 28
    :ptype anchor_day: int
    :return: the period's id
    :rtype: str
    :raises ValueError: when ``anchor_day`` is outside 1 to 28, or ``now`` is naive
    """
    start, _ = period_bounds(now, anchor_day)
    if anchor_day == _DEFAULT_ANCHOR_DAY:
        return f"{start.year:04d}-{start.month:02d}"
    return f"{start.year:04d}-{start.month:02d}-{start.day:02d}"


def _check_anchor_day(anchor_day: int) -> None:
    # bool is an int, and True would otherwise pass as anchor 1
    if isinstance(anchor_day, bool) or not isinstance(anchor_day, int):
        raise ValueError(f"anchor_day must be an int from 1 to {_MAX_ANCHOR_DAY}, got {anchor_day!r}")
    if not 1 <= anchor_day <= _MAX_ANCHOR_DAY:
        raise ValueError(f"anchor_day must be from 1 to {_MAX_ANCHOR_DAY}, so every month has one, got {anchor_day}")


def _as_utc(now: datetime) -> datetime:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError(f"now must be timezone-aware, got the naive {now.isoformat()}")
    return now.astimezone(UTC)


def _previous_month(year: int, month: int) -> tuple[int, int]:
    return (year - 1, 12) if month == 1 else (year, month - 1)


def _next_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)
