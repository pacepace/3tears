"""Meeting slots across our offices: the times in a UTC day when every attendee is inside their working hours.

Each attendee works in one of our office cities and has working hours in that city's local time (``"08:30-17:00"``).
A slot is a meeting of ``minutes`` length. Slots start at midnight UTC and then every ``step`` minutes through the UTC
day (every half hour unless the request says otherwise; a step divides the day evenly: 10, 15, 20 or 30 minutes). A
slot is free when, for every attendee, the meeting starts and ends within their working hours on one local day.

Local time follows each city's clocks, daylight saving included. The cities and their rules are in :data:`CITIES`:
each has a standard offset from UTC, and, where it keeps summer time, the day summer time starts and the day it ends
(a Sunday, given as a month and which Sunday of it, ``-1`` for the last). Summer time adds an hour. Sydney's summer
runs from October into the next year's April, as the southern hemisphere's does.

The changeover itself is at 01:00 to 03:00 local time on a Sunday morning, outside anyone's working hours, so a city's
offset is taken from its local calendar date.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

#: How far apart slot starts are, in minutes, unless a request says otherwise.
STEP_MINUTES = 30


@dataclass(frozen=True)
class Changeover:
    month: int
    sunday: int  # 1 for the first Sunday of the month, 2 for the second, -1 for the last


@dataclass(frozen=True)
class City:
    standard_hours: int
    summer_starts: Changeover | None = None
    summer_ends: Changeover | None = None


CITIES: dict[str, City] = {
    "London": City(0, Changeover(3, -1), Changeover(10, -1)),
    "Lisbon": City(0, Changeover(3, -1), Changeover(10, -1)),
    "Berlin": City(1, Changeover(3, -1), Changeover(10, -1)),
    "New York": City(-5, Changeover(3, 2), Changeover(11, 1)),
    "Toronto": City(-5, Changeover(3, 2), Changeover(11, 1)),
    "Sydney": City(10, Changeover(10, 1), Changeover(4, 1)),
    "Singapore": City(8),
    "Bangalore": City(5),  # half an hour more is added below
    "Sao Paulo": City(-3),
}
#: Cities whose offset is not a whole number of hours, and the extra minutes.
EXTRA_MINUTES = {"Bangalore": 30}


@dataclass(frozen=True)
class Attendee:
    name: str
    city: str
    hours: str = "09:00-17:00"

    def window(self) -> tuple[int, int]:
        """Working hours as minutes after local midnight."""
        start, end = self.hours.split("-")
        return _minutes(start), _minutes(end)


def _minutes(text: str) -> int:
    hours, minutes = text.strip().split(":")
    return int(hours) * 60 + int(minutes)


def sunday(year: int, change: Changeover) -> date:
    """The Sunday a changeover falls on in ``year``."""
    weeks = calendar.Calendar().monthdatescalendar(year, change.month)
    sundays = [week[6] for week in weeks if week[6].month == change.month]
    return sundays[change.sunday - 1] if change.sunday > 0 else sundays[change.sunday]


def in_summer_time(city: str, day: date) -> bool:
    """Whether ``city`` keeps summer time on its local date ``day``."""
    rule = CITIES[city]
    if rule.summer_starts is None or rule.summer_ends is None:
        return False
    starts, ends = sunday(day.year, rule.summer_starts), sunday(day.year, rule.summer_ends)
    return starts <= day < ends if starts < ends else not ends <= day < starts


def utc_offset(city: str, moment: datetime) -> timedelta:
    """How far ``city``'s clocks are ahead of UTC at ``moment`` (a UTC time)."""
    rule = CITIES[city]
    standard = timedelta(hours=rule.standard_hours, minutes=EXTRA_MINUTES.get(city, 0))
    local_day = (moment + standard).date()
    return standard + timedelta(hours=1) if in_summer_time(city, local_day) else standard


def local_time(city: str, moment: datetime) -> datetime:
    """``moment`` (UTC) on ``city``'s clocks, as a naive local time."""
    return (moment + utc_offset(city, moment)).replace(tzinfo=None)


def is_free(attendee: Attendee, start: datetime, minutes: int) -> bool:
    """Whether a meeting from ``start`` (UTC) for ``minutes`` sits inside the attendee's working hours."""
    local = local_time(attendee.city, start)
    begins = local.hour * 60 + local.minute
    opens, closes = attendee.window()
    return opens <= begins and begins + minutes <= closes


def free_slots(day: date, attendees: list[Attendee], minutes: int = 30, step: int = STEP_MINUTES) -> list[datetime]:
    """Every slot start in the UTC day ``day`` when all ``attendees`` are free, as UTC times, earliest first."""
    if not attendees:
        return []
    if (24 * 60) % step:
        raise ValueError(f"a step of {step} minutes does not divide the day")
    midnight = datetime.combine(day, time(), tzinfo=timezone.utc)
    found = []
    hour = 0.0
    for _ in range(24 * 60 // step):
        start = midnight + timedelta(minutes=round(hour * 60))
        if all(is_free(person, start, minutes) for person in attendees):
            found.append(start)
        hour += step / 60
    return found


def first_slot(day: date, attendees: list[Attendee], minutes: int = 30, step: int = STEP_MINUTES) -> datetime | None:
    """The earliest free slot in the UTC day ``day``, or ``None`` when there is none."""
    slots = free_slots(day, attendees, minutes, step)
    return slots[0] if slots else None


def from_request(request: dict[str, Any]) -> tuple[date, list[Attendee], int, int]:
    """A request as ``data/requests.jsonl`` stores one: its UTC day, its attendees, the meeting's length and the step."""
    attendees = [Attendee(p["name"], p["city"], p.get("hours", "09:00-17:00")) for p in request["attendees"]]
    day = date.fromisoformat(request["day"])
    return day, attendees, int(request.get("minutes", 30)), int(request.get("step", STEP_MINUTES))
