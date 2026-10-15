"""Reading the zones, dates and times a person writes, for the time tools.

A person writes "3pm", "tomorrow 9:00", "PST", "chicago" or "Oct 9". The time
tools took eight strptime formats and IANA names written exactly, so "convert
9am EST to GMT" -- the router's own example -- failed, and a time with no date
was stamped with the server's UTC day, which every evening in the Americas is
tomorrow. This module reads what people write and says how it read it, so a
result can name the zone and the day it used.

Nothing here reads the clock: a caller passes "now" in, so a test freezes it.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, timezone, tzinfo
from functools import cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

__all__ = [
    "Clock",
    "ReadError",
    "ZONE_ABBREVIATIONS",
    "When",
    "ZoneRead",
    "describe_zone",
    "parse_date",
    "parse_when",
    "read_zone",
    "system_clock",
]

#: What returns the current instant, timezone-aware. Tools take one so a test can freeze it.
Clock = Callable[[], datetime]


def system_clock() -> datetime:
    """The current instant, in UTC.

    :return: now
    :rtype: datetime
    """
    return datetime.now(UTC)


class ReadError(ValueError):
    """What a person wrote could not be read; the message says what forms are taken."""


#: Abbreviations and plain names people use for a zone, each mapped to ONE zone and
#: the words said back. A summer or winter form maps to the same place: "PST" in
#: July is read as Los Angeles, which is on PDT then, and the result says so.
#: "IST" is India: Israel (Asia/Jerusalem) and Ireland (Europe/Dublin) are named by city.
ZONE_ABBREVIATIONS: dict[str, tuple[str, str]] = {
    **dict.fromkeys(("PST", "PDT", "PT", "PACIFIC"), ("America/Los_Angeles", "US Pacific")),
    **dict.fromkeys(("MST", "MDT", "MT", "MOUNTAIN"), ("America/Denver", "US Mountain")),
    **dict.fromkeys(("CST", "CDT", "CT", "CENTRAL"), ("America/Chicago", "US Central")),
    **dict.fromkeys(("EST", "EDT", "ET", "EASTERN"), ("America/New_York", "US Eastern")),
    **dict.fromkeys(("AKST", "AKDT", "ALASKA"), ("America/Anchorage", "Alaska")),
    **dict.fromkeys(("HST", "HAWAII"), ("Pacific/Honolulu", "Hawaii")),
    **dict.fromkeys(("UTC", "Z", "ZULU"), ("UTC", "UTC")),
    "GMT": ("Etc/GMT", "GMT, which is UTC all year; Europe/London for UK time"),
    **dict.fromkeys(("BST", "UK"), ("Europe/London", "UK time")),
    **dict.fromkeys(("CET", "CEST"), ("Europe/Paris", "Central European")),
    **dict.fromkeys(("EET", "EEST"), ("Europe/Athens", "Eastern European")),
    **dict.fromkeys(("WET", "WEST"), ("Europe/Lisbon", "Western European")),
    "MSK": ("Europe/Moscow", "Moscow"),
    "IST": ("Asia/Kolkata", "India; for Israel or Ireland name the city"),
    "JST": ("Asia/Tokyo", "Japan"),
    "KST": ("Asia/Seoul", "Korea"),
    "SGT": ("Asia/Singapore", "Singapore"),
    "HKT": ("Asia/Hong_Kong", "Hong Kong"),
    **dict.fromkeys(("AEST", "AEDT"), ("Australia/Sydney", "Australian Eastern")),
    **dict.fromkeys(("ACST", "ACDT"), ("Australia/Adelaide", "Australian Central")),
    "AWST": ("Australia/Perth", "Australian Western"),
    **dict.fromkeys(("NZST", "NZDT"), ("Pacific/Auckland", "New Zealand")),
}

_ZONE_FORMS = (
    "an IANA name such as America/Chicago (any case), a city such as Tokyo or New York, "
    "an abbreviation such as PST, EST, CET, JST or IST, or an offset such as UTC+5:30"
)

_OFFSET = re.compile(r"^(?:UTC|GMT)?\s*([+-])(\d{1,2})(?::?(\d{2}))?$", re.IGNORECASE)


@cache
def _zone_names() -> dict[str, str]:
    """Every IANA name and every city part of one, lowercased, to the name."""
    names: dict[str, str] = {}
    for name in sorted(available_timezones()):
        names.setdefault(name.lower(), name)
    for name in sorted(available_timezones()):
        if "/" not in name or name.startswith(("Etc/", "SystemV/", "posix/", "right/")):
            continue
        city = name.rsplit("/", 1)[1].replace("_", " ").lower()
        # The first zone a city names wins; America/ sorts before the rest, which is
        # what a reader means by an unqualified American city.
        names.setdefault(city, name)
    return names


@dataclass(frozen=True)
class ZoneRead:
    """A zone as read from what a person wrote.

    :ivar zone: the zone
    :ivar name: its name as the result shows it
    :ivar how: how an abbreviation or a city was read, or ``""`` when written exactly
    """

    zone: tzinfo
    name: str
    how: str = ""


def read_zone(text: str | None) -> ZoneRead:
    """Read a zone a person wrote.

    :param text: an IANA name in any case, a city, an abbreviation, or a UTC offset
    :ptype text: str | None
    :return: the zone and how it was read
    :rtype: ZoneRead
    :raises ReadError: when it names no zone, with the forms that are taken
    """
    raw = (text or "").strip()
    if not raw:
        raise ReadError(f"no zone given. Write {_ZONE_FORMS}.")
    if raw.upper() in ZONE_ABBREVIATIONS:
        name, words = ZONE_ABBREVIATIONS[raw.upper()]
        how = "" if name.lower() == raw.lower() else f"{raw} read as {name} ({words})"
        return ZoneRead(ZoneInfo(name), name, how)
    offset = _OFFSET.match(raw)
    if offset and any(c.isdigit() for c in raw):
        sign, hours, minutes = offset.groups()
        delta = timedelta(hours=int(hours), minutes=int(minutes or 0))
        if delta > timedelta(hours=14):
            raise ReadError(f"{raw!r} is not an offset a zone has; offsets run from UTC-12 to UTC+14.")
        fixed = timezone(delta if sign == "+" else -delta)
        label = f"UTC{sign}{int(hours):02d}:{int(minutes or 0):02d}"
        return ZoneRead(fixed, label, f"{raw} read as the fixed offset {label}, which has no summer time")
    exact = _zone_names().get(raw.lower().replace("_", " ")) or _zone_names().get(raw.lower())
    if exact is None:
        raise ReadError(f"unknown timezone {raw!r}. Write {_ZONE_FORMS}.")
    try:
        zone = ZoneInfo(exact)
    except ZoneInfoNotFoundError, ValueError:
        raise ReadError(f"unknown timezone {raw!r}. Write {_ZONE_FORMS}.") from None
    how = "" if exact == raw else f"{raw} read as {exact}"
    return ZoneRead(zone, exact, how)


def describe_zone(moment: datetime) -> str:
    """The zone's short name and offset at one moment, e.g. ``CDT, UTC-05:00``.

    :param moment: an aware datetime
    :ptype moment: datetime
    :return: the abbreviation and the offset
    :rtype: str
    """
    offset = moment.utcoffset() or timedelta(0)
    total = int(offset.total_seconds() // 60)
    sign = "+" if total >= 0 else "-"
    hours, minutes = divmod(abs(total), 60)
    label = f"UTC{sign}{hours:02d}:{minutes:02d}"
    abbreviation = moment.tzname() or ""
    return (
        label
        if not abbreviation or abbreviation == label or abbreviation.startswith(("+", "-"))
        else f"{abbreviation}, {label}"
    )


_MONTHS: dict[str, int] = {
    **{name.lower(): i for i, name in enumerate(calendar.month_name) if name},
    **{name.lower(): i for i, name in enumerate(calendar.month_abbr) if name},
    "sept": 9,
}
_MONTH = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DAY_WORDS = {"today": 0, "tonight": 0, "tomorrow": 1, "yesterday": -1}
_WEEKDAYS = {name.lower(): i for i, name in enumerate(calendar.day_name)}

_ISO_DATE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_US_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_MONTH_FIRST = re.compile(rf"\b({_MONTH})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b", re.IGNORECASE)
_DAY_FIRST = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH})\.?(?:,?\s+(\d{{4}}))?\b", re.IGNORECASE)
_DAY_WORD = re.compile(r"\b(today|tonight|tomorrow|yesterday)\b", re.IGNORECASE)
_NEXT_WEEKDAY = re.compile(r"\b(?:(next|this|last)\s+)?(" + "|".join(_WEEKDAYS) + r")\b", re.IGNORECASE)

_CLOCK = re.compile(
    r"^(?:at\s+)?(\d{1,2})(?:[:.](\d{2}))?(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?$",
    re.IGNORECASE,
)

_DATE_FORMS = "2026-10-09, 10/09/2026, Oct 9, 9 October 2026, today, tomorrow, yesterday or next Friday"
_TIME_FORMS = "3pm, 3:30 PM, 15:00, noon or midnight"


def _make_date(year: int, month: int, day: int, written: str) -> date:
    try:
        return date(year, month, day)
    except ValueError:
        raise ReadError(f"{written!r} is not a date on the calendar.") from None


def _take_date(text: str, today: date) -> tuple[date | None, str]:
    """The date a phrase names, and the phrase with that part removed."""
    if m := _ISO_DATE.search(text):
        found = _make_date(int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(0))
    elif m := _US_DATE.search(text):
        found = _make_date(int(m.group(3)), int(m.group(1)), int(m.group(2)), m.group(0))
    elif m := _MONTH_FIRST.search(text):
        year = int(m.group(3)) if m.group(3) else today.year
        found = _make_date(year, _MONTHS[m.group(1).lower()], int(m.group(2)), m.group(0))
    elif m := _DAY_FIRST.search(text):
        year = int(m.group(3)) if m.group(3) else today.year
        found = _make_date(year, _MONTHS[m.group(2).lower()], int(m.group(1)), m.group(0))
    elif m := _DAY_WORD.search(text):
        found = today + timedelta(days=_DAY_WORDS[m.group(1).lower()])
    elif m := _NEXT_WEEKDAY.search(text):
        which = (m.group(1) or "").lower()
        target = _WEEKDAYS[m.group(2).lower()]
        ahead = (target - today.weekday()) % 7
        if which == "next" and ahead == 0:
            ahead = 7
        elif which == "last":
            ahead = ahead - 7 if ahead else -7
        found = today + timedelta(days=ahead)
    else:
        return None, text
    rest = (text[: m.start()] + " " + text[m.end() :]).strip(" ,")
    return found, rest


def parse_date(text: str, today: date) -> date:
    """Read a date a person wrote; a month and day with no year is this year.

    :param text: e.g. ``2026-12-25``, ``Dec 25``, ``25 December 2026``, ``tomorrow``, ``next friday``
    :ptype text: str
    :param today: the day "today" means, in the zone that applies
    :ptype today: date
    :return: the date
    :rtype: date
    :raises ReadError: when no date is read, or something is left over
    """
    found, rest = _take_date(text.strip(), today)
    if found is None or rest:
        raise ReadError(f"could not read {text!r} as a date. Write {_DATE_FORMS}.")
    return found


def _read_clock(text: str) -> time:
    lowered = text.strip().lower()
    if lowered in ("noon", "midday", "12 noon"):
        return time(12, 0)
    if lowered == "midnight":
        return time(0, 0)
    m = _CLOCK.match(lowered)
    if not m:
        raise ReadError(f"could not read {text!r} as a time. Write {_TIME_FORMS}.")
    hour, minute, second = int(m.group(1)), int(m.group(2) or 0), int(m.group(3) or 0)
    meridiem = (m.group(4) or "").replace(".", "")
    if meridiem:
        if not 1 <= hour <= 12:
            raise ReadError(f"{text!r}: with am or pm the hour runs from 1 to 12.")
        hour = hour % 12 + (12 if meridiem == "pm" else 0)
    if hour > 23 or minute > 59 or second > 59:
        raise ReadError(f"{text!r} is not a time of day. Write {_TIME_FORMS}.")
    return time(hour, minute, second)


@dataclass(frozen=True)
class When:
    """A moment as a person wrote it.

    :ivar day: the date written, or ``None`` when only a time was written
    :ivar clock: the time of day written, or ``None`` when only a date was written
    :ivar instant: the moment, when what was written carried its own offset (``...Z``)
    """

    day: date | None
    clock: time | None
    instant: datetime | None = None


def parse_when(text: str, today: date) -> When:
    """Read a date, a time, or both, as a person writes them.

    :param text: e.g. ``3pm``, ``tomorrow 9:00``, ``2026-10-10T14:00:00Z``, ``Oct 9 3:30 PM``
    :ptype text: str
    :param today: the day "today" and "tomorrow" count from, in the zone that applies
    :ptype today: date
    :return: what was read
    :rtype: When
    :raises ReadError: when it cannot be read, with the forms that are taken
    """
    raw = text.strip()
    try:
        iso = datetime.fromisoformat(raw[:-1] + "+00:00" if raw.endswith(("Z", "z")) else raw)
    except ValueError:
        iso = None
    if iso is not None:
        only_date = len(raw) <= 10
        if iso.tzinfo is not None:
            return When(day=iso.date(), clock=iso.timetz().replace(tzinfo=None), instant=iso)
        return When(day=iso.date(), clock=None if only_date else iso.time())
    day, rest = _take_date(raw, today)
    rest = re.sub(r"^\s*at\s+", "", rest).strip()
    clock = _read_clock(rest) if rest else None
    if day is None and clock is None:
        raise ReadError(f"could not read {text!r}. Write a time ({_TIME_FORMS}), a date ({_DATE_FORMS}), or both.")
    return When(day=day, clock=clock)
