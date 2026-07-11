"""Pure helper: compute the next ``next_fire_at`` for a scheduled job.

Generalized from :mod:`threetears.agent.wake.reschedule`; the math is
domain-neutral already (it operates only on a ``schedule_type``, a
config dict, a missed-fire policy, and timestamps), so the
generalization is a rename (``_compute_next_fire_at`` ->
:func:`compute_next_fire_at`, now public) plus dropping the
agent-specific module references. The branch bodies are unchanged.

design notes
------------

- **Pure function.** No DB, no I/O, no scheduler state. Given
  ``(schedule_type, schedule_config, missed_fire_policy, last_fired_at,
  now)`` it deterministically returns the next fire instant or ``None``
  (terminal one-shot).
- **TZ-aware via stdlib ``zoneinfo``.** ``daily_at`` and
  ``random_within_window`` resolve in the schedule's configured
  timezone; DST transitions handled by ``zoneinfo`` naturally
  (spring-forward day advances by 23h not 24h, fall-back day advances
  by 25h not 24h).
- **Missed-fire policy honored.** ``'coalesce'`` (default) fires once
  for a backlog and recomputes the next ``next_fire_at`` forward into
  the future. ``'catch_up'`` advances ``next_fire_at`` by exactly one
  increment *from the occurrence being fired* (``current_fire_at``, the
  claimed row's ``next_fire_at``) so subsequent ticks fire once per
  missed interval until caught up. It deliberately does NOT anchor on
  ``last_fired_at``: the store stamps ``last_fired_at = now`` on every
  claim, so anchoring there would collapse ``'catch_up'`` back into
  ``'coalesce'``.
- **APScheduler is a cron-only utility.** The ``'cron'`` branch is the
  only place APScheduler enters; its ``CronTrigger`` is imported lazily
  inside the branch so non-cron consumers pay no import cost. APScheduler
  is declared as a hard dependency (small dep, ``cron`` is part of the
  public ``ScheduleType`` surface) but the lazy import keeps the cost of
  merely importing this module at zero. The function raises
  ``RuntimeError`` with install guidance if the import fails, so
  consumers running a stripped wheel see a clear message instead of a
  bare ``ImportError``.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, time, timedelta
from typing import Any, Final
from zoneinfo import ZoneInfo

__all__ = ["compute_next_fire_at"]


# Default timezone for schedule types that take a ``tz`` field when
# config omits it. UTC is unambiguous + DST-stable; consumers that want
# a local timezone supply it explicitly in the schedule_config.
_DEFAULT_TZ: Final[str] = "UTC"


def compute_next_fire_at(
    schedule_type: str,
    schedule_config: dict[str, Any],
    missed_fire_policy: str,
    last_fired_at: datetime | None,
    now: datetime,
    current_fire_at: datetime | None = None,
) -> datetime | None:
    """Compute the next ``next_fire_at`` for a scheduled job.

    Returns the next fire instant as a TZ-aware UTC ``datetime``, or
    ``None`` for terminal one-shot schedules (``one_shot_at`` and
    ``relative_delay`` after their single fire).

    ``current_fire_at`` is the scheduled instant of the occurrence being
    fired right now (the claimed row's ``next_fire_at``), NOT the
    wall-clock ``now``. ``'catch_up'`` anchors the next occurrence on it
    so a backlog drains one interval per tick. It MUST NOT anchor on
    ``last_fired_at``: the store stamps ``last_fired_at = now`` on every
    claim, so anchoring there silently collapses ``'catch_up'`` into
    ``'coalesce'`` (a job down three intervals would fire twice, not
    once per missed interval). When ``current_fire_at`` is ``None`` (no
    occurrence in flight, e.g. the very first schedule pass) ``catch_up``
    falls through to the ``coalesce`` anchor.

    :param schedule_type: one of the values pinned by
        :data:`threetears.scheduled_jobs.types.ScheduleType`
    :ptype schedule_type: str
    :param schedule_config: per-schedule-type config payload
    :ptype schedule_config: dict[str, Any]
    :param missed_fire_policy: one of the values pinned by
        :data:`threetears.scheduled_jobs.types.MissedFirePolicy`
        (``'coalesce'`` | ``'catch_up'``)
    :ptype missed_fire_policy: str
    :param last_fired_at: timestamp of the most recent fire, or ``None``
        for unfired; drives terminal detection for ``one_shot_at`` /
        ``relative_delay`` and the per-day slot budget for
        ``random_within_window``
    :ptype last_fired_at: datetime | None
    :param now: tick instant (TZ-aware)
    :ptype now: datetime
    :param current_fire_at: scheduled instant of the occurrence being
        fired now; the ``catch_up`` anchor
    :ptype current_fire_at: datetime | None
    :return: next fire instant or ``None`` for terminal one-shot
    :rtype: datetime | None
    :raises ValueError: when ``schedule_type`` is unknown or the config
        is malformed
    :raises RuntimeError: when ``schedule_type == 'cron'`` and
        apscheduler is not installed
    """
    if schedule_type == "daily_at":
        return _next_daily_at(schedule_config, missed_fire_policy, current_fire_at, now)
    if schedule_type == "every_n_hours":
        return _next_every_n_hours(schedule_config, missed_fire_policy, current_fire_at, now)
    if schedule_type == "random_within_window":
        return _next_random_within_window(schedule_config, last_fired_at, now)
    if schedule_type == "one_shot_at":
        return _next_one_shot_at(schedule_config, last_fired_at, now)
    if schedule_type == "cron":
        return _next_cron(schedule_config, missed_fire_policy, current_fire_at, now)
    if schedule_type == "relative_delay":
        return _next_relative_delay(schedule_config, last_fired_at, now)
    if schedule_type == "interval":
        return _next_interval(schedule_config, missed_fire_policy, current_fire_at, now)
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
    current_fire_at: datetime | None,
    now: datetime,
) -> datetime:
    """``daily_at``: fire once per day at the configured wall-clock time.

    Config: ``{"hour": int, "minute": int, "tz": str}``.
    DST is handled by ``zoneinfo``: spring-forward day advances by 23h
    not 24h; fall-back day advances by 25h not 24h.
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

    if missed_fire_policy == "catch_up" and current_fire_at is not None:
        # advance from the occurrence being fired by exactly one day; lets
        # backlogged ticks catch up one fire per tick.
        current_local = current_fire_at.astimezone(tz)
        next_local = datetime.combine(
            current_local.date() + timedelta(days=1),
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
    current_fire_at: datetime | None,
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

    if missed_fire_policy == "catch_up" and current_fire_at is not None:
        # step one interval past the occurrence being fired, draining a
        # backlog one fire per tick.
        return _ensure_utc(current_fire_at + step)

    # coalesce default: anchor on now (skip the backlog)
    return _ensure_utc(now + step)


def _next_random_within_window(
    config: dict[str, Any],
    last_fired_at: datetime | None,
    now: datetime,
) -> datetime:
    """``random_within_window``: pick a uniform-random time-of-day inside
    the configured window, honoring the per-day fire budget.

    Config: ``{"start_hour": int, "end_hour": int, "tz": str,
    "fires_per_day": int}``. Overnight windows (``start_hour >
    end_hour``) wrap across midnight. ``fires_per_day`` (default ``1``)
    partitions the window into that many equal, non-overlapping slots;
    each fire lands in one slot, so the schedule fires exactly
    ``fires_per_day`` times per window and never re-fires within the slot
    it just fired in. Without the slot budget the reschedule would keep
    picking a fresh time in ``[now, end)`` on the SAME day after every
    fire and run away, firing far more often than intended.

    ``last_fired_at`` distinguishes the first pass (never fired -- the
    earliest slot still open, possibly the one ``now`` sits in) from a
    reschedule after a fire (skip the slot ``now`` occupies and take the
    next one, rolling to the next day's first slot once the day's slots
    are spent).
    """
    start_hour = int(config["start_hour"])
    end_hour = int(config["end_hour"])
    if start_hour == end_hour:
        msg = "random_within_window requires start_hour != end_hour"
        raise ValueError(msg)
    fires_per_day = int(config.get("fires_per_day", 1))
    if fires_per_day <= 0:
        msg = f"random_within_window requires positive fires_per_day, got {fires_per_day}"
        raise ValueError(msg)
    tz = _tz(config)
    now_local = now.astimezone(tz)

    start_dt, end_dt = _window_bounds(start_hour, end_hour, now_local, tz)
    slot_span = (end_dt - start_dt) / fires_per_day
    already_fired = last_fired_at is not None
    chosen = _select_slot(start_dt, slot_span, fires_per_day, now_local, already_fired=already_fired)
    if chosen is None:
        # every slot in this window is spent (or already past) -- roll to
        # the first slot of the next day's window.
        next_start = start_dt + timedelta(days=1)
        chosen = (next_start, next_start + slot_span)
    slot_start, slot_end = chosen

    # pick uniform random instant in [max(slot_start, now), slot_end)
    floor_local = max(slot_start, now_local)
    span_seconds = (slot_end - floor_local).total_seconds()
    if span_seconds <= 0:
        # defense in depth: floor collapsed onto the slot end -- fall back
        # to the whole slot rather than a zero-width span.
        floor_local = slot_start
        span_seconds = (slot_end - slot_start).total_seconds()
    offset = random.uniform(0.0, span_seconds)
    candidate_local = floor_local + timedelta(seconds=offset)
    return _ensure_utc(candidate_local)


def _window_bounds(
    start_hour: int,
    end_hour: int,
    now_local: datetime,
    tz: ZoneInfo,
) -> tuple[datetime, datetime]:
    """Resolve the ``[start, end)`` window (as TZ-aware local datetimes)
    that either contains ``now`` or is the day's upcoming window.

    Non-wrapping windows (``start_hour < end_hour``) resolve to today's
    ``[start, end)``. Overnight windows (``start_hour > end_hour``) span
    midnight: if ``now`` is before ``end_hour`` the active window opened
    yesterday, otherwise it opens today.
    """
    if start_hour < end_hour:
        start_dt = datetime.combine(now_local.date(), time(hour=start_hour), tzinfo=tz)
        end_dt = datetime.combine(now_local.date(), time(hour=end_hour), tzinfo=tz)
    else:
        # overnight wrap: window spans midnight from start_hour on day D to
        # end_hour on day D+1.
        if now_local.hour < end_hour:
            anchor_date = now_local.date() - timedelta(days=1)
        else:
            anchor_date = now_local.date()
        start_dt = datetime.combine(anchor_date, time(hour=start_hour), tzinfo=tz)
        end_dt = datetime.combine(anchor_date + timedelta(days=1), time(hour=end_hour), tzinfo=tz)
    return start_dt, end_dt


def _select_slot(
    start_dt: datetime,
    slot_span: timedelta,
    fires_per_day: int,
    now_local: datetime,
    *,
    already_fired: bool,
) -> tuple[datetime, datetime] | None:
    """Choose the ``[slot_start, slot_end)`` the next fire belongs in.

    Slots partition ``[start_dt, start_dt + fires_per_day * slot_span)``
    into ``fires_per_day`` equal spans. When ``already_fired`` is set the
    slot ``now`` sits in was just consumed, so the earliest slot whose
    START is strictly after ``now`` is chosen; otherwise (never fired)
    the earliest slot whose END is still in the future is chosen (which
    may be the slot ``now`` currently occupies). Returns ``None`` when no
    slot in this window qualifies.
    """
    chosen: tuple[datetime, datetime] | None = None
    for index in range(fires_per_day):
        slot_start = start_dt + slot_span * index
        slot_end = slot_start + slot_span
        boundary = slot_start if already_fired else slot_end
        if boundary > now_local:
            chosen = (slot_start, slot_end)
            break
    return chosen


def _next_one_shot_at(
    config: dict[str, Any],
    last_fired_at: datetime | None,
    now: datetime,
) -> datetime | None:
    """``one_shot_at``: fire once at the configured ISO instant.

    Config: ``{"fire_at_iso": str}``. After the fire (when
    ``last_fired_at`` is set), return ``None`` so the caller can flip the
    schedule to ``status='expired'``.
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
    current_fire_at: datetime | None,
    now: datetime,
) -> datetime:
    """``cron``: standard 5-field cron expression.

    Config: ``{"expr": str}``. Uses APScheduler's ``CronTrigger`` as a
    pure utility import (NOT as scheduler infrastructure -- the tick body
    is APScheduler-agnostic). APScheduler is a hard dep; the import is
    lazy here so non-cron consumers pay no import cost. Raises
    ``RuntimeError`` with install guidance if APScheduler is missing
    (e.g. stripped wheel).
    """
    try:
        from apscheduler.triggers.cron import CronTrigger  # noqa: PLC0415
    except ImportError as exc:
        msg = (
            "cron schedule_type requires apscheduler; install it via "
            "`uv add apscheduler` (it ships as a hard dep of "
            "3tears-scheduled-jobs, so this should only happen on a "
            "stripped wheel)"
        )
        raise RuntimeError(msg) from exc

    expr = config["expr"]
    # Pin the cron to the configured timezone (UTC by default), matching
    # every other schedule type's use of ``_tz(config)``. Without an
    # explicit timezone, ``from_crontab`` adopts the SYSTEM local tz, so a
    # non-UTC server fires cron schedules at the wrong wall-clock instant.
    # The fire times are stored/compared in UTC, so the cron must be
    # evaluated in a fixed zone, not the host's.
    trigger = CronTrigger.from_crontab(expr, timezone=_tz(config))

    if missed_fire_policy == "catch_up" and current_fire_at is not None:
        anchor = _ensure_utc(current_fire_at)
    else:
        anchor = _ensure_utc(now)

    fire_time = trigger.get_next_fire_time(previous_fire_time=anchor, now=anchor)
    if fire_time is None:
        # cron with no future fire (unlikely for standard exprs)
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
    ``"1d"``, etc. After the fire, return ``None`` so the caller can flip
    the schedule to ``status='expired'``.
    """
    if last_fired_at is not None:
        return None
    delay_str = str(config["delay"])
    delay = _parse_delay(delay_str)
    return _ensure_utc(now + delay)


def _next_interval(
    config: dict[str, Any],
    missed_fire_policy: str,
    current_fire_at: datetime | None,
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

    if missed_fire_policy == "catch_up" and current_fire_at is not None:
        # step one interval past the occurrence being fired, draining a
        # backlog one fire per tick.
        return _ensure_utc(current_fire_at + step)
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
