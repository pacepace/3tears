"""Recurring meetings for the team calendar: every date and time a series falls on, and the invites it sends.

A :class:`Series` is a meeting that repeats from its first date at the same time of day, as the clock on the office
wall reads it in London. ``every`` is one of:

- ``"day"``: every day;
- ``"weekday"``: every day from Monday to Friday;
- ``"week"``: the same weekday every week;
- ``"fortnight"``: the same weekday every other week.

``count`` is how many meetings the series holds, and a date in ``skip`` (a bank holiday, an away day) is passed over
without being counted. :func:`occurrences` lists the meetings as London times. The wall-clock time holds across the
clock changes: a 09:30 stand-up is at 09:30 in winter and in summer.

The calendar server takes invites in UTC, so :func:`invites` writes each meeting as ``2026-03-02T09:30Z``. UK clocks go
forward an hour (to BST, UTC+1) at 01:00 UTC on the last Sunday of March and back (to GMT, UTC+0) at 01:00 UTC on the
last Sunday of October; :data:`LONDON` keeps those rules, so this module needs no time-zone database.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from pathlib import Path

SERIES = Path(__file__).with_name("data") / "series.jsonl"

_HOUR = timedelta(hours=1)
_ZERO = timedelta(0)


def _last_sunday(year: int, month: int) -> datetime:
    last = date(year, month + 1, 1) - timedelta(days=1)
    return datetime.combine(last - timedelta(days=(last.weekday() + 1) % 7), time(1, 0))


class _UKTime(tzinfo):
    """GMT in winter, BST (UTC+1) from 01:00 UTC on the last Sunday of March to 01:00 UTC on the last Sunday of October."""

    def _summer(self, dt: datetime) -> bool:
        # dt is a London wall-clock time. The clocks go forward at 01:00 GMT, so 01:00 to 02:00 that night never
        # happens; they go back at 02:00 BST, so 01:00 to 02:00 that night happens twice, BST first (fold 0).
        local = dt.replace(tzinfo=None)
        start = _last_sunday(dt.year, 3)
        end = _last_sunday(dt.year, 10) + (_HOUR if dt.fold == 0 else _ZERO)
        return start <= local < end

    def utcoffset(self, dt: datetime | None) -> timedelta:
        return _HOUR if dt is not None and self._summer(dt) else _ZERO

    def dst(self, dt: datetime | None) -> timedelta:
        return self.utcoffset(dt)

    def tzname(self, dt: datetime | None) -> str:
        return "BST" if self.utcoffset(dt) else "GMT"

    def fromutc(self, dt: datetime) -> datetime:
        utc = dt.replace(tzinfo=None)
        start, end = _last_sunday(utc.year, 3), _last_sunday(utc.year, 10)
        if start <= utc < end:
            return (utc + _HOUR).replace(tzinfo=self)
        return utc.replace(tzinfo=self, fold=1 if end <= utc < end + _HOUR else 0)

    def __repr__(self) -> str:
        return "LONDON"


#: London time, with the UK's clock changes.
LONDON = _UKTime()

_STEP = {
    "day": timedelta(days=1),
    "weekday": timedelta(days=1),
    "week": timedelta(weeks=1),
    "fortnight": timedelta(weeks=2),
}


@dataclass(frozen=True)
class Series:
    title: str
    first: date
    at: time
    every: str
    count: int
    skip: frozenset[date] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.every not in _STEP:
            raise ValueError(f"a series repeats every {', '.join(_STEP)}, not every {self.every!r}")
        if self.count < 1:
            raise ValueError(f"a series of {self.count} meetings")

    @classmethod
    def from_json(cls, row: dict[str, object]) -> Series:
        """A series from ``{"title", "first": "2026-03-02", "at": "09:30", "every", "count", "skip": [...]}``."""
        skip = row.get("skip") or []
        if not isinstance(skip, list):
            raise ValueError(f"skip is a list of dates, not {skip!r}")
        return cls(
            title=str(row["title"]),
            first=date.fromisoformat(str(row["first"])),
            at=time.fromisoformat(str(row["at"])),
            every=str(row["every"]),
            count=int(str(row["count"])),
            skip=frozenset(date.fromisoformat(str(day)) for day in skip),
        )


def _stepped(first: datetime, step: timedelta) -> Iterator[datetime]:
    n = 0
    while True:
        yield first + step * n
        n += 1


def occurrences(series: Series) -> list[datetime]:
    """Every meeting in ``series``, in order, as London times."""
    first = datetime.combine(series.first, series.at, tzinfo=LONDON)
    meetings: list[datetime] = []
    for when in _stepped(first, _STEP[series.every]):
        if len(meetings) == series.count:
            break
        if series.every == "weekday" and when.weekday() >= 5:
            continue
        if when.date() in series.skip:
            continue
        meetings.append(when)
    return meetings


def invites(series: Series) -> list[str]:
    """Every meeting in ``series`` as the UTC time the calendar server takes: ``2026-03-02T09:30Z``."""
    return [when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ") for when in occurrences(series)]


def load_series(path: Path = SERIES) -> list[Series]:
    """The series in a file of one JSON object a line; keys beyond :meth:`Series.from_json`'s are ignored."""
    return [
        Series.from_json(json.loads(text)) for text in path.read_text(encoding="utf-8").splitlines() if text.strip()
    ]
