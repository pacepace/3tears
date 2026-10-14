"""How a wake's schedule and its times are said to whoever reads a tool's result.

One place, so a listing, a change and a refusal say the same schedule the same
way: ``every 60 minutes``, ``daily at 03:00 America/Los_Angeles``. A time is
shown in the zone its caller names and says the zone; with no zone named it is
shown as stored.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, tzinfo
from typing import Any

from threetears.core.utils.local_time import render_local_time

__all__ = ["describe_schedule", "format_wake_time"]

#: The zone a schedule's clock times are in when its config names none.
_UNNAMED_ZONE = "UTC"


def format_wake_time(value: datetime | None, zone: tzinfo | None = None, *, unset: str = "-") -> str:
    """A wake's time as its reader says it: in ``zone`` and naming it, or as stored when no zone was named.

    :param value: the moment, or ``None`` when there is none
    :ptype value: datetime | None
    :param zone: the reader's zone, when the caller named one
    :ptype zone: tzinfo | None
    :param unset: what to say when there is no moment
    :ptype unset: str
    :return: the time, or ``unset``
    :rtype: str
    """
    if value is None:
        return unset
    if zone is None:
        return value.isoformat()
    return render_local_time(value, zone)


def _every(count: int, unit: str) -> str:
    return f"every {unit}" if count == 1 else f"every {count} {unit}s"


def _said(schedule_type: str, config: Mapping[str, Any]) -> str | None:
    """The schedule in words, or ``None`` for a type this does not know."""
    if schedule_type == "interval":
        seconds = int(config["seconds"])
        if seconds % 3600 == 0:
            return _every(seconds // 3600, "hour")
        if seconds % 60 == 0:
            return _every(seconds // 60, "minute")
        return _every(seconds, "second")
    if schedule_type == "every_n_hours":
        return _every(int(config["n"]), "hour")
    if schedule_type == "daily_at":
        zone = config.get("tz") or _UNNAMED_ZONE
        return f"daily at {int(config['hour']):02d}:{int(config.get('minute', 0)):02d} {zone}"
    if schedule_type == "random_within_window":
        zone = config.get("tz") or _UNNAMED_ZONE
        times = int(config.get("fires_per_day", 1))
        often = "once a day" if times == 1 else f"{times} times a day"
        return f"{often} between {int(config['start_hour']):02d}:00 and {int(config['end_hour']):02d}:00 {zone}"
    if schedule_type == "cron":
        return f"cron {config['expr']}"
    if schedule_type == "one_shot_at":
        return f"once at {config['fire_at_iso']}"
    if schedule_type == "relative_delay":
        return f"once, {config['delay']} after it was set"
    return None


def describe_schedule(schedule_type: str, schedule_config: Mapping[str, Any]) -> str:
    """A schedule in words: how often it fires, or when.

    A config this does not know how to say is shown as it is stored, after its
    type, so nothing is described wrongly.

    :param schedule_type: one of the wake schedule types
    :ptype schedule_type: str
    :param schedule_config: that type's config
    :ptype schedule_config: Mapping[str, Any]
    :return: the schedule, in a few words
    :rtype: str
    """
    try:
        said = _said(schedule_type, schedule_config)
    except KeyError, TypeError, ValueError:
        said = None  # not its type's shape: shown as stored, below
    if said is not None:
        return said
    return f"{schedule_type} {json.dumps(dict(schedule_config), sort_keys=True, default=str)}"
