"""Unit tests for :func:`threetears.agent.wake.reschedule._compute_next_fire_at`.

Pure-function tests: pinned ``now`` via ``freezegun`` for the time-
sensitive branches, deterministic seeds via ``random.seed`` for the
``random_within_window`` branch.

Coverage:

- one case per schedule_type
- ``missed_fire_policy`` 'coalesce' vs 'catch_up'
- DST transitions (spring-forward + fall-back) for ``daily_at``
- overnight-window wrap for ``random_within_window``
- terminal one-shot semantics (``one_shot_at`` and ``relative_delay``
  return ``None`` after their first fire)
- malformed config errors
- unknown schedule_type
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from threetears.agent.wake.reschedule import _compute_next_fire_at


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    """Tiny constructor for TZ-aware UTC datetimes."""
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


# ---------------------------------------------------------------------------
# daily_at
# ---------------------------------------------------------------------------


class TestDailyAt:
    """``daily_at`` branch — fire once per day at the configured wall time."""

    def test_today_in_future(self) -> None:
        """Today's HH:MM is still in the future — return today's instance."""
        now = _utc(2026, 5, 22, 8, 0)
        result = _compute_next_fire_at(
            "daily_at",
            {"hour": 14, "minute": 0, "tz": "UTC"},
            "coalesce",
            None,
            now,
        )
        assert result == _utc(2026, 5, 22, 14, 0)

    def test_today_past_advances_to_tomorrow(self) -> None:
        """Today's HH:MM has passed — bump to tomorrow."""
        now = _utc(2026, 5, 22, 15, 0)
        result = _compute_next_fire_at(
            "daily_at",
            {"hour": 14, "minute": 0, "tz": "UTC"},
            "coalesce",
            None,
            now,
        )
        assert result == _utc(2026, 5, 23, 14, 0)

    def test_dst_spring_forward_la(self) -> None:
        """``daily_at 09:00 America/Los_Angeles`` across spring-forward advances by 23h.

        On 2026-03-08 LA, 02:00 jumps to 03:00 — the day is 23 hours
        long. ``zoneinfo`` makes this transparent: the next-day 09:00
        local instant is correctly 23 hours after the prior-day 09:00.
        """
        tz = ZoneInfo("America/Los_Angeles")
        # 2026-03-07 09:00 LA == 2026-03-07 17:00 UTC
        last_fire_utc = datetime(2026, 3, 7, 9, 0, tzinfo=tz).astimezone(UTC)
        # tick just past today's 09:00 LA so we recompute forward
        now = datetime(2026, 3, 7, 9, 1, tzinfo=tz).astimezone(UTC)
        result = _compute_next_fire_at(
            "daily_at",
            {"hour": 9, "minute": 0, "tz": "America/Los_Angeles"},
            "coalesce",
            last_fire_utc,
            now,
        )
        # next fire is 2026-03-08 09:00 LA — 23h after 2026-03-07 09:00 LA
        assert result is not None
        assert (result - last_fire_utc) == timedelta(hours=23)
        assert result.astimezone(tz) == datetime(2026, 3, 8, 9, 0, tzinfo=tz)

    def test_dst_fall_back_la(self) -> None:
        """``daily_at 09:00 America/Los_Angeles`` across fall-back advances by 25h.

        On 2026-11-01 LA, 02:00 falls back to 01:00 — the day is 25
        hours long.
        """
        tz = ZoneInfo("America/Los_Angeles")
        last_fire_utc = datetime(2026, 10, 31, 9, 0, tzinfo=tz).astimezone(UTC)
        now = datetime(2026, 10, 31, 9, 1, tzinfo=tz).astimezone(UTC)
        result = _compute_next_fire_at(
            "daily_at",
            {"hour": 9, "minute": 0, "tz": "America/Los_Angeles"},
            "coalesce",
            last_fire_utc,
            now,
        )
        assert result is not None
        assert (result - last_fire_utc) == timedelta(hours=25)
        assert result.astimezone(tz) == datetime(2026, 11, 1, 9, 0, tzinfo=tz)

    def test_catch_up_advances_one_day_from_last_fire(self) -> None:
        """``catch_up`` advances exactly one day from the last fire, ignoring backlog."""
        last_fire = _utc(2026, 5, 20, 9, 0)
        # tick is 3 days later — coalesce would jump to today/tomorrow
        # but catch_up advances by exactly one day so the next ticks
        # catch up one fire at a time.
        now = _utc(2026, 5, 23, 12, 0)
        result = _compute_next_fire_at(
            "daily_at",
            {"hour": 9, "minute": 0, "tz": "UTC"},
            "catch_up",
            last_fire,
            now,
        )
        assert result == _utc(2026, 5, 21, 9, 0)

    def test_default_tz_is_utc(self) -> None:
        """Omitting ``tz`` defaults to UTC."""
        now = _utc(2026, 5, 22, 8, 0)
        result = _compute_next_fire_at(
            "daily_at",
            {"hour": 14, "minute": 30},
            "coalesce",
            None,
            now,
        )
        assert result == _utc(2026, 5, 22, 14, 30)


# ---------------------------------------------------------------------------
# every_n_hours
# ---------------------------------------------------------------------------


class TestEveryNHours:
    """``every_n_hours`` branch."""

    def test_coalesce_anchors_on_now(self) -> None:
        now = _utc(2026, 5, 22, 10, 0)
        result = _compute_next_fire_at(
            "every_n_hours",
            {"n": 3},
            "coalesce",
            None,
            now,
        )
        assert result == _utc(2026, 5, 22, 13, 0)

    def test_catch_up_anchors_on_last_fire(self) -> None:
        last = _utc(2026, 5, 22, 4, 0)
        now = _utc(2026, 5, 22, 15, 0)  # backlog
        result = _compute_next_fire_at(
            "every_n_hours",
            {"n": 3},
            "catch_up",
            last,
            now,
        )
        assert result == _utc(2026, 5, 22, 7, 0)

    def test_zero_n_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _compute_next_fire_at(
                "every_n_hours",
                {"n": 0},
                "coalesce",
                None,
                _utc(2026, 5, 22),
            )


# ---------------------------------------------------------------------------
# random_within_window
# ---------------------------------------------------------------------------


class TestRandomWithinWindow:
    """``random_within_window`` branch — random time-of-day in [start, end)."""

    def test_non_wrap_window_picks_inside_today(self) -> None:
        random.seed(42)
        now = _utc(2026, 5, 22, 10, 0)
        result = _compute_next_fire_at(
            "random_within_window",
            {"start_hour": 9, "end_hour": 21, "tz": "UTC"},
            "coalesce",
            None,
            now,
        )
        assert result is not None
        assert _utc(2026, 5, 22, 10, 0) <= result < _utc(2026, 5, 22, 21, 0)

    def test_non_wrap_window_after_end_advances_to_tomorrow(self) -> None:
        random.seed(42)
        now = _utc(2026, 5, 22, 22, 0)
        result = _compute_next_fire_at(
            "random_within_window",
            {"start_hour": 9, "end_hour": 21, "tz": "UTC"},
            "coalesce",
            None,
            now,
        )
        assert result is not None
        assert _utc(2026, 5, 23, 9, 0) <= result < _utc(2026, 5, 23, 21, 0)

    def test_wrap_window_picks_inside_overnight(self) -> None:
        """``start_hour > end_hour`` defines an overnight window."""
        random.seed(42)
        # Window: 22:00 -> 06:00 the next day
        # Tick at 23:00 -- pick is inside the open span [now, 06:00 tomorrow)
        now = _utc(2026, 5, 22, 23, 0)
        result = _compute_next_fire_at(
            "random_within_window",
            {"start_hour": 22, "end_hour": 6, "tz": "UTC"},
            "coalesce",
            None,
            now,
        )
        assert result is not None
        assert _utc(2026, 5, 22, 23, 0) <= result < _utc(2026, 5, 23, 6, 0)

    def test_wrap_window_in_morning_uses_current_window(self) -> None:
        """At 03:00 (overnight window 22:00->06:00) we're still in the active window."""
        random.seed(42)
        now = _utc(2026, 5, 22, 3, 0)
        result = _compute_next_fire_at(
            "random_within_window",
            {"start_hour": 22, "end_hour": 6, "tz": "UTC"},
            "coalesce",
            None,
            now,
        )
        assert result is not None
        # window anchored on yesterday 22:00, so end is today 06:00
        assert _utc(2026, 5, 22, 3, 0) <= result < _utc(2026, 5, 22, 6, 0)

    def test_start_equals_end_rejected(self) -> None:
        with pytest.raises(ValueError, match="start_hour != end_hour"):
            _compute_next_fire_at(
                "random_within_window",
                {"start_hour": 10, "end_hour": 10, "tz": "UTC"},
                "coalesce",
                None,
                _utc(2026, 5, 22, 1, 0),
            )


# ---------------------------------------------------------------------------
# one_shot_at
# ---------------------------------------------------------------------------


class TestOneShotAt:
    """``one_shot_at`` branch — terminal after first fire."""

    def test_future_iso_returned(self) -> None:
        now = _utc(2026, 5, 22, 10, 0)
        result = _compute_next_fire_at(
            "one_shot_at",
            {"fire_at_iso": "2026-05-22T14:00:00+00:00"},
            "coalesce",
            None,
            now,
        )
        assert result == _utc(2026, 5, 22, 14, 0)

    def test_past_iso_first_compute_fires_now(self) -> None:
        """A past one-shot fire that hasn't fired yet fires immediately."""
        now = _utc(2026, 5, 22, 18, 0)
        result = _compute_next_fire_at(
            "one_shot_at",
            {"fire_at_iso": "2026-05-22T14:00:00+00:00"},
            "coalesce",
            None,
            now,
        )
        assert result == now

    def test_after_fire_returns_none(self) -> None:
        """After ``last_fired_at`` is set, return ``None`` (terminal)."""
        result = _compute_next_fire_at(
            "one_shot_at",
            {"fire_at_iso": "2026-05-22T14:00:00+00:00"},
            "coalesce",
            _utc(2026, 5, 22, 14, 0),
            _utc(2026, 5, 22, 14, 1),
        )
        assert result is None


# ---------------------------------------------------------------------------
# cron
# ---------------------------------------------------------------------------


class TestCron:
    """``cron`` branch — APScheduler's CronTrigger as a pure utility."""

    def test_hourly_cron(self) -> None:
        now = _utc(2026, 5, 22, 10, 30)
        result = _compute_next_fire_at(
            "cron",
            {"expr": "0 * * * *"},
            "coalesce",
            None,
            now,
        )
        assert result == _utc(2026, 5, 22, 11, 0)

    def test_every_three_hours(self) -> None:
        now = _utc(2026, 5, 22, 1, 30)
        result = _compute_next_fire_at(
            "cron",
            {"expr": "0 */3 * * *"},
            "coalesce",
            None,
            now,
        )
        # 0 */3 fires at 00, 03, 06, 09, 12, 15, 18, 21
        assert result == _utc(2026, 5, 22, 3, 0)

    def test_catch_up_anchors_on_last_fire(self) -> None:
        last = _utc(2026, 5, 22, 9, 0)
        now = _utc(2026, 5, 22, 20, 0)  # backlog of multiple fires
        result = _compute_next_fire_at(
            "cron",
            {"expr": "0 */3 * * *"},
            "catch_up",
            last,
            now,
        )
        # catch_up advances by exactly one cron step from last fire
        assert result == _utc(2026, 5, 22, 12, 0)


# ---------------------------------------------------------------------------
# relative_delay
# ---------------------------------------------------------------------------


class TestRelativeDelay:
    """``relative_delay`` branch — fire once after ``delay`` from creation."""

    def test_first_fire_in_future(self) -> None:
        now = _utc(2026, 5, 22, 10, 0)
        result = _compute_next_fire_at(
            "relative_delay",
            {"delay": "30m"},
            "coalesce",
            None,
            now,
        )
        assert result == _utc(2026, 5, 22, 10, 30)

    def test_hours_unit(self) -> None:
        now = _utc(2026, 5, 22, 10, 0)
        result = _compute_next_fire_at(
            "relative_delay",
            {"delay": "2h"},
            "coalesce",
            None,
            now,
        )
        assert result == _utc(2026, 5, 22, 12, 0)

    def test_days_unit(self) -> None:
        now = _utc(2026, 5, 22, 10, 0)
        result = _compute_next_fire_at(
            "relative_delay",
            {"delay": "3d"},
            "coalesce",
            None,
            now,
        )
        assert result == _utc(2026, 5, 25, 10, 0)

    def test_after_fire_returns_none(self) -> None:
        result = _compute_next_fire_at(
            "relative_delay",
            {"delay": "30m"},
            "coalesce",
            _utc(2026, 5, 22, 10, 30),
            _utc(2026, 5, 22, 10, 31),
        )
        assert result is None

    def test_malformed_delay_rejected(self) -> None:
        with pytest.raises(ValueError):
            _compute_next_fire_at(
                "relative_delay",
                {"delay": "abc"},
                "coalesce",
                None,
                _utc(2026, 5, 22, 10, 0),
            )

    def test_unknown_unit_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown delay unit"):
            _compute_next_fire_at(
                "relative_delay",
                {"delay": "5y"},
                "coalesce",
                None,
                _utc(2026, 5, 22, 10, 0),
            )


# ---------------------------------------------------------------------------
# interval
# ---------------------------------------------------------------------------


class TestInterval:
    """``interval`` branch — fire every N seconds."""

    def test_coalesce_anchors_on_now(self) -> None:
        now = _utc(2026, 5, 22, 10, 0)
        result = _compute_next_fire_at(
            "interval",
            {"seconds": 1800},
            "coalesce",
            None,
            now,
        )
        assert result == _utc(2026, 5, 22, 10, 30)

    def test_catch_up_anchors_on_last_fire(self) -> None:
        last = _utc(2026, 5, 22, 9, 0)
        now = _utc(2026, 5, 22, 15, 0)
        result = _compute_next_fire_at(
            "interval",
            {"seconds": 1800},
            "catch_up",
            last,
            now,
        )
        assert result == _utc(2026, 5, 22, 9, 30)

    def test_zero_seconds_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _compute_next_fire_at(
                "interval",
                {"seconds": 0},
                "coalesce",
                None,
                _utc(2026, 5, 22),
            )


# ---------------------------------------------------------------------------
# unknown schedule_type
# ---------------------------------------------------------------------------


def test_unknown_schedule_type_raises() -> None:
    """An unknown ``schedule_type`` is a programming error."""
    with pytest.raises(ValueError, match="unknown schedule_type"):
        _compute_next_fire_at(
            "lunar_cycle",
            {},
            "coalesce",
            None,
            _utc(2026, 5, 22),
        )
