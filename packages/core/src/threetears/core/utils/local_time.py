"""A moment, written for a reader in one zone.

A tool that shows a time shows it to someone who lives somewhere. Printed as a
bare UTC instant -- ``2026-10-10T04:11:20.123456+00:00`` -- the reader converts
it in their head, or does not, and tells a person the wrong hour. A tool takes
the zone of the person the turn is with and hands every time it prints to
:func:`render_local_time`, which says the zone in the text.

Only what is read as a time goes through here. An id that happens to hold a
time (a cursor, a version stamp) stays exactly as stored: its reader is the next
call, not a person.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

__all__ = ["parse_date_bound", "render_local_time", "resolve_zone"]


def resolve_zone(tz: tzinfo | str | None) -> tzinfo | None:
    """The zone a caller named, as a ``tzinfo``.

    ``None`` stays ``None``: the caller named no zone. A name that is not an
    IANA zone is refused rather than read as UTC, because a time shown in a zone
    nobody asked for is a wrong time that reads as a right one.

    :param tz: a ``tzinfo``, an IANA name such as ``America/Los_Angeles``, or ``None``
    :ptype tz: tzinfo | str | None
    :return: the zone, or ``None`` when none was named
    :rtype: tzinfo | None
    :raises ValueError: when the name is not a zone this system knows
    """
    if tz is None or isinstance(tz, tzinfo):
        return tz
    try:
        return ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError, OSError) as exc:
        raise ValueError(f"{tz!r} is not a time zone this system knows") from exc


def render_local_time(value: datetime, tz: tzinfo | str | None = None) -> str:
    """``value`` as a reader in ``tz`` says it, with the zone named: ``Oct 9, 2026 at 3:00 PM PDT``.

    A ``value`` with no zone of its own is taken as UTC, which is how every
    timestamp this platform stores is meant. With no ``tz`` the time is shown in
    UTC and says so.

    :param value: the moment
    :ptype value: datetime
    :param tz: the reader's zone: a ``tzinfo``, an IANA name, or ``None`` for UTC
    :ptype tz: tzinfo | str | None
    :return: the time in that zone, ending with the zone's short name
    :rtype: str
    :raises ValueError: when ``tz`` is a name that is not a zone this system knows
    """
    zone = resolve_zone(tz) or UTC
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    local = aware.astimezone(zone)
    hour = local.hour % 12 or 12
    half = "AM" if local.hour < 12 else "PM"
    name = local.tzname() or local.strftime("%z")
    return f"{local:%b} {local.day}, {local.year} at {hour}:{local.minute:02d} {half} {name}"


def parse_date_bound(label: str, raw: str | None, zone: tzinfo | None) -> tuple[datetime | None, str | None]:
    """Read a date filter. A date alone is the whole day in the reader's zone; a time with no zone is in it too.

    ``date_before`` given as a bare date runs to the end of that day, so the day
    named is included. With no zone named the day is UTC's.

    :param label: ``date_after`` or ``date_before``
    :ptype label: str
    :param raw: what the caller passed
    :ptype raw: str | None
    :param zone: the reader's zone, when the caller named one
    :ptype zone: tzinfo | None
    :return: the moment, or an error to show
    :rtype: tuple[datetime | None, str | None]
    """
    if not raw or not raw.strip():
        return None, None
    text = raw.strip()
    home = zone or UTC
    try:
        day = date.fromisoformat(text) if len(text) <= 10 else None
    except ValueError:
        day = None
    if day is not None:
        edge = time.max if label == "date_before" else time.min
        return datetime.combine(day, edge, tzinfo=home), None
    try:
        value = datetime.fromisoformat(text)
    except ValueError:
        return None, f"Invalid {label} '{raw}'. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)."
    if value.tzinfo is None:
        value = value.replace(tzinfo=home)
    return value, None
