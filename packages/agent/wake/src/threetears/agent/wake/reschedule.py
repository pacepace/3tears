"""Pure helper: compute the next ``next_fire_at`` for a wake schedule.

Split out from :mod:`threetears.agent.wake.tick` so the (DST-heavy,
branch-dense) reschedule logic can be unit-tested without spinning up
the tick body / dispatch callback / DB.

design notes
------------

- **Pure function.** No DB, no I/O, no APScheduler scheduler state.
  Given ``(schedule_type, schedule_config, missed_fire_policy,
  last_fired_at, now)`` it deterministically returns the next fire
  instant or ``None`` (terminal one-shot).
- **TZ-aware via stdlib ``zoneinfo``.** ``daily_at`` and
  ``random_within_window`` resolve in the schedule's configured
  timezone; DST transitions handled by ``zoneinfo`` naturally
  (spring-forward day advances by 23h not 24h, fall-back day
  advances by 25h not 24h).
- **Missed-fire policy honored** per PLACEMENT §1.7. ``'coalesce'``
  (default) fires once for a backlog and recomputes the next
  ``next_fire_at`` forward into the future. ``'catch_up'`` advances
  ``next_fire_at`` by exactly one increment so subsequent ticks fire
  once per missed interval until caught up.
- **APScheduler is an optional cron utility.** The ``'cron'`` branch
  is the only place APScheduler enters; its ``CronTrigger`` is
  imported lazily inside the branch so the rest of the platform does
  not pay an import cost for a single schedule_type. Consumers using
  cron schedules MUST install ``apscheduler`` (declared as an extra
  on the wake package). The function raises ``RuntimeError`` with a
  pointer to the install if the import fails.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, time, timedelta
from typing import Any, Final
from zoneinfo import ZoneInfo

__all__ = ["_compute_next_fire_at"]


# Default timezone for schedule types that take a ``tz`` field when
# config omits it. UTC is unambiguous + DST-stable; consumers that
# want a local timezone supply it explicitly in the schedule_config.
_DEFAULT_TZ: Final[str] = "UTC"


def _compute_next_fire_at(
    schedule_type: str,
    schedule_config: dict[str, Any],
    missed_fire_policy: str,
    last_fired_at: datetime | None,
    now: datetime,
) -> datetime | None:
    """Compute the next ``next_fire_at`` for a wake schedule.

    Returns the next fire instant as a TZ-aware UTC ``datetime``, or
    ``None`` for terminal one-shot schedules (``one_shot_at`` and
    ``relative_delay`` after their single fire).

    :param schedule_type: one of the values pinned by
        :data:`threetears.agent.wake.types.ScheduleType`
    :ptype schedule_type: str
    :param schedule_config: per-schedule-type JSONB payload; shape
        documented in :mod:`threetears.agent.wake.collections`
    :ptype schedule_config: dict[str, Any]
    :param missed_fire_policy: one of the values pinned by
        :data:`threetears.agent.wake.types.MissedFirePolicy`
        (``'coalesce'`` | ``'catch_up'``)
    :ptype missed_fire_policy: str
    :param last_fired_at: timestamp of the most recent fire, or
        ``None`` for unfired
    :ptype last_fired_at: datetime | None
    :param now: tick instant (TZ-aware)
    :ptype now: datetime
    :return: next fire instant or ``None`` for terminal one-shot
    :rtype: datetime | None
    :raises ValueError: when ``schedule_type`` is unknown or the
        config is malformed
    """
    if schedule_type == "daily_at":
        return _next_daily_at(schedule_config, missed_fire_policy, last_fired_at, now)
    if schedule_type == "every_n_hours":
        return _next_every_n_hours(schedule_config, missed_fire_policy, last_fired_at, now)
    if schedule_type == "random_within_window":
        return _next_random_within_window(schedule_config, now)
    if schedule_type == "one_shot_at":
        return _next_one_shot_at(schedule_config, last_fired_at, now)
    if schedule_type == "cron":
        return _next_cron(schedule_config, missed_fire_policy, last_fired_at, now)
    if schedule_type == "relative_delay":
        return _next_relative_delay(schedule_config, last_fired_at, now)
    if schedule_type == "interval":
        return _next_interval(schedule_config, missed_fire_policy, last_fired_at, now)
    msg = f"unknown schedule_type: {schedule_type!r}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# per-schedule-type branches
# ---------------------------------------------------------------------------


def _tz(config: dict[str, Any]) -> ZoneInfo:
    """Resolve the configured timezone (defaults to UTC)."""
    return ZoneInfo(config.get("tz", _DEFAULT_TZ))


def _ensure_utc(dt: datetime) -> datetime:
    """Coerce a TZ-aware datetime into UTC."""
    if dt.tzinfo is None:
        # treat naive datetimes as UTC for legacy callers; new code
        # should always pass TZ-aware values
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _next_daily_at(
    config: dict[str, Any],
    missed_fire_policy: str,
    last_fired_at: datetime | None,
    now: datetime,
) -> datetime:
    """``daily_at``: fire once per day at the configured wall-clock time.

    Config: ``{"hour": int, "minute": int, "tz": str}``.
    DST is handled by ``zoneinfo``: spring-forward day advances by
    23h not 24h; fall-back day advances by 25h not 24h.
    """
    hour = int(config["hour"])
    minute = int(config.get("minute", 0))
    tz = _tz(config)
    now_local = now.astimezone(tz)

    # candidate = today at HH:MM in tz
    candidate_local = datetime.combine(
        now_local.date(),
        time(hour=hour, minute=minute),
        tzinfo=tz,
    )

    if missed_fire_policy == "catch_up" and last_fired_at is not None:
        # advance from the last fire by exactly one day; lets backlogged
        # ticks catch up one fire per tick.
        last_local = last_fired_at.astimezone(tz)
        # the day after the last fire's date, at HH:MM
        next_local = datetime.combine(
            last_local.date() + timedelta(days=1),
            time(hour=hour, minute=minute),
            tzinfo=tz,
        )
        return _ensure_utc(next_local)

    # coalesce (default): jump forward to the next future fire
    if candidate_local <= now_local:
        candidate_local = datetime.combine(
            now_local.date() + timedelta(days=1),
            time(hour=hour, minute=minute),
            tzinfo=tz,
        )
    return _ensure_utc(candidate_local)


def _next_every_n_hours(
    config: dict[str, Any],
    missed_fire_policy: str,
    last_fired_at: datetime | None,
    now: datetime,
) -> datetime:
    """``every_n_hours``: fire every N hours.

    Config: ``{"n": int}``.
    """
    n = int(config["n"])
    if n <= 0:
        msg = f"every_n_hours requires positive n, got {n}"
        raise ValueError(msg)
    step = timedelta(hours=n)

    if missed_fire_policy == "catch_up" and last_fired_at is not None:
        return _ensure_utc(last_fired_at + step)

    # coalesce default: anchor on now (skip the backlog)
    return _ensure_utc(now + step)


def _next_random_within_window(
    config: dict[str, Any],
    now: datetime,
) -> datetime:
    """``random_within_window``: pick a uniform-random time-of-day
    inside the configured window for the next fire day.

    Config: ``{"start_hour": int, "end_hour": int, "tz": str,
    "fires_per_day": int}``. Overnight windows (``start_hour >
    end_hour``) wrap across midnight.
    """
    start_hour = int(config["start_hour"])
    end_hour = int(config["end_hour"])
    if start_hour == end_hour:
        msg = "random_within_window requires start_hour != end_hour"
        raise ValueError(msg)
    tz = _tz(config)
    now_local = now.astimezone(tz)

    # base candidate day: today (so we can still fire later today if
    # the window has room left) or tomorrow if today's window passed.
    if start_hour < end_hour:
        # non-wrapping: window is [start, end) on a single date
        latest_today = datetime.combine(
            now_local.date(),
            time(hour=end_hour, minute=0),
            tzinfo=tz,
        )
        candidate_date = now_local.date()
        if now_local >= latest_today:
            candidate_date = now_local.date() + timedelta(days=1)
        start_dt = datetime.combine(
            candidate_date,
            time(hour=start_hour, minute=0),
            tzinfo=tz,
        )
        end_dt = datetime.combine(
            candidate_date,
            time(hour=end_hour, minute=0),
            tzinfo=tz,
        )
    else:
        # overnight wrap: window spans midnight from start_hour on day D
        # to end_hour on day D+1. anchor on "tonight": if now is past
        # end_hour today, anchor on today's start; otherwise anchor on
        # yesterday's start (so we may still fire later today before
        # end_hour).
        if now_local.hour < end_hour:
            anchor_date = now_local.date() - timedelta(days=1)
        else:
            anchor_date = now_local.date()
        start_dt = datetime.combine(
            anchor_date,
            time(hour=start_hour, minute=0),
            tzinfo=tz,
        )
        end_dt = datetime.combine(
            anchor_date + timedelta(days=1),
            time(hour=end_hour, minute=0),
            tzinfo=tz,
        )
        # if we are already past the wrap end_dt, slide to next day
        if now_local >= end_dt:
            start_dt = start_dt + timedelta(days=1)
            end_dt = end_dt + timedelta(days=1)

    # pick uniform random instant in [max(start, now), end)
    floor_local = max(start_dt, now_local)
    span_seconds = (end_dt - floor_local).total_seconds()
    if span_seconds <= 0:
        # entire window already past (shouldn't happen with the slide
        # above, but defense in depth) -- bump to the next day
        start_dt = start_dt + timedelta(days=1)
        end_dt = end_dt + timedelta(days=1)
        floor_local = start_dt
        span_seconds = (end_dt - floor_local).total_seconds()
    offset = random.uniform(0.0, span_seconds)
    candidate_local = floor_local + timedelta(seconds=offset)
    return _ensure_utc(candidate_local)


def _next_one_shot_at(
    config: dict[str, Any],
    last_fired_at: datetime | None,
    now: datetime,
) -> datetime | None:
    """``one_shot_at``: fire once at the configured ISO instant.

    Config: ``{"fire_at_iso": str}``. After the fire (when
    ``last_fired_at`` is set), return ``None`` so the caller can flip
    the schedule to ``status='expired'``.
    """
    if last_fired_at is not None:
        return None
    iso = config["fire_at_iso"]
    fire_at = datetime.fromisoformat(iso)
    fire_at = _ensure_utc(fire_at)
    # if the configured time is already past, fire NOW (the tick will
    # pick up "due" on the next pass and we want to fire immediately
    # rather than skip a missed one-shot).
    if fire_at <= now:
        return _ensure_utc(now)
    return fire_at


def _next_cron(
    config: dict[str, Any],
    missed_fire_policy: str,
    last_fired_at: datetime | None,
    now: datetime,
) -> datetime:
    """``cron``: standard 5-field cron expression.

    Config: ``{"expr": str}``. Uses APScheduler's ``CronTrigger`` as a
    pure pure-utility import (NOT as scheduler infrastructure -- the
    tick body is APScheduler-agnostic). Raises ``RuntimeError`` with
    install guidance if APScheduler is unavailable.
    """
    try:
        from apscheduler.triggers.cron import CronTrigger  # noqa: PLC0415
    except ImportError as exc:
        msg = (
            "cron schedule_type requires apscheduler; install it via "
            "`uv add apscheduler` or pin it as an extra on the wake package"
        )
        raise RuntimeError(msg) from exc

    expr = config["expr"]
    trigger = CronTrigger.from_crontab(expr)

    if missed_fire_policy == "catch_up" and last_fired_at is not None:
        anchor = _ensure_utc(last_fired_at)
    else:
        anchor = _ensure_utc(now)

    fire_time = trigger.get_next_fire_time(previous_fire_time=anchor, now=anchor)
    if fire_time is None:
        # cron with no future fire (unlikely for standard exprs) --
        # fall back to a far-future placeholder; caller can detect via
        # subsequent ticks finding no work to do.
        msg = f"cron expression {expr!r} has no next fire time"
        raise ValueError(msg)
    return _ensure_utc(fire_time)


def _next_relative_delay(
    config: dict[str, Any],
    last_fired_at: datetime | None,
    now: datetime,
) -> datetime | None:
    """``relative_delay``: fire once after a one-shot delay from creation.

    Config: ``{"delay": str}`` where ``delay`` is ``"30m"``, ``"2h"``,
    ``"1d"``, etc. After the fire, return ``None`` so the caller can
    flip the schedule to ``status='expired'``.
    """
    if last_fired_at is not None:
        return None
    delay_str = str(config["delay"])
    delay = _parse_delay(delay_str)
    return _ensure_utc(now + delay)


def _next_interval(
    config: dict[str, Any],
    missed_fire_policy: str,
    last_fired_at: datetime | None,
    now: datetime,
) -> datetime:
    """``interval``: fire every N seconds.

    Config: ``{"seconds": int}``.
    """
    seconds = int(config["seconds"])
    if seconds <= 0:
        msg = f"interval requires positive seconds, got {seconds}"
        raise ValueError(msg)
    step = timedelta(seconds=seconds)

    if missed_fire_policy == "catch_up" and last_fired_at is not None:
        return _ensure_utc(last_fired_at + step)
    return _ensure_utc(now + step)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _parse_delay(delay: str) -> timedelta:
    """Parse a relative-delay literal like ``"30m"`` / ``"2h"`` / ``"1d"``.

    Accepts an integer suffix unit: ``s``/``m``/``h``/``d``. Raises
    ``ValueError`` for malformed input.
    """
    if not delay:
        msg = "delay must be non-empty"
        raise ValueError(msg)
    unit = delay[-1].lower()
    try:
        value = int(delay[:-1])
    except ValueError as exc:
        msg = f"delay must be <int><unit>, got {delay!r}"
        raise ValueError(msg) from exc
    if unit == "s":
        return timedelta(seconds=value)
    if unit == "m":
        return timedelta(minutes=value)
    if unit == "h":
        return timedelta(hours=value)
    if unit == "d":
        return timedelta(days=value)
    msg = f"unknown delay unit {unit!r}; expected s/m/h/d"
    raise ValueError(msg)
