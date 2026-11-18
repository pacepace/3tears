"""Next departures for the 412 bus, for the stop signs, the app and the SMS line.

The timetable gives each stop's departure times as local clock times (Brisbane, which keeps UTC+10 all year), one
list for weekdays, one for Saturdays and one for Sundays. A question is a stop and the moment it was asked:

- an ISO time with an offset (`2026-03-02T07:58:00+10:00`, or `Z` for UTC) is that moment;
- an ISO time with no offset is a local clock time at the stop.

The answer is the next departures from that stop at or after that moment, in the order they leave: a bus leaving in
the very minute asked counts. After the day's last bus the answer runs on into the next day's timetable. Each
departure is given as a local ISO time to the minute (`2026-03-02T08:04`).
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

LOCAL = timezone(timedelta(hours=10), "AEST")
DAY_TYPES = ("weekday", "weekday", "weekday", "weekday", "weekday", "saturday", "sunday")
#: How many departures an answer gives.
SHOWN = 3


def load_timetable(path: str | Path) -> dict[str, dict[str, list[str]]]:
    """The timetable file: each stop's ``weekday``, ``saturday`` and ``sunday`` lists of ``HH:MM`` times."""
    return json.loads(Path(path).read_text(encoding="utf-8"))["stops"]


def asked_at(text: str) -> datetime:
    """The moment a question was asked, as a local time at the stop."""
    when = datetime.fromisoformat(text)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(LOCAL)


def departures_on(stop_times: dict[str, list[str]], day: date) -> list[datetime]:
    """Every departure from a stop on ``day``, in order."""
    times = sorted(time.fromisoformat(t) for t in stop_times[DAY_TYPES[day.weekday()]])
    return [datetime.combine(day, t, tzinfo=LOCAL) for t in times]


def next_departures(
    timetable: dict[str, dict[str, list[str]]], stop: str, asked: str, count: int = SHOWN
) -> list[str]:
    """The next ``count`` departures from ``stop`` at or after the moment ``asked``."""
    if stop not in timetable:
        raise KeyError(f"no stop called {stop!r} on the 412")
    now = asked_at(asked).replace(second=0, microsecond=0)
    found: list[datetime] = []
    day = now.date()
    while len(found) < count and day <= now.date() + timedelta(days=7):
        found.extend(d for d in departures_on(timetable[stop], day) if d >= now)
        day += timedelta(days=1)
    return [d.strftime("%Y-%m-%dT%H:%M") for d in found[:count]]
