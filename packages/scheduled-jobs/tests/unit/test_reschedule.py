"""Unit tests for :func:`threetears.scheduled_jobs.reschedule.compute_next_fire_at`.

Pure-function tests, mirroring agent-wake's reschedule suite (the math is
identical -- this asserts the generalization preserved every branch). One
case per schedule_type; ``coalesce`` vs ``catch_up`` missed-fire policy;
DST transitions for ``daily_at``; overnight-window wrap for
``random_within_window``; terminal one-shot semantics; malformed config /
unknown type errors.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from threetears.scheduled_jobs.reschedule import compute_next_fire_at


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    """Tiny constructor for TZ-aware UTC datetimes."""
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


class TestDailyAt:
    """``daily_at`` branch -- fire once per day at the configured wall time."""

    def test_today_in_future(self) -> None:
        """Today's HH:MM is still in the future -- return today's instance."""
        result = compute_next_fire_at(
            "daily_at", {"hour": 14, "minute": 0, "tz": "UTC"}, "coalesce", None, _utc(2026, 5, 22, 8, 0)
        )
        assert result == _utc(2026, 5, 22, 14, 0)

    def test_today_past_advances_to_tomorrow(self) -> None:
        """Today's HH:MM has passed -- bump to tomorrow."""
        result = compute_next_fire_at(
            "daily_at", {"hour": 14, "minute": 0, "tz": "UTC"}, "coalesce", None, _utc(2026, 5, 22, 15, 0)
        )
        assert result == _utc(2026, 5, 23, 14, 0)

    def test_dst_spring_forward_la(self) -> None:
        """``daily_at 09:00 America/Los_Angeles`` across spring-forward advances by 23h."""
        tz = ZoneInfo("America/Los_Angeles")
        last_fire_utc = datetime(2026, 3, 7, 9, 0, tzinfo=tz).astimezone(UTC)
        now = datetime(2026, 3, 7, 9, 1, tzinfo=tz).astimezone(UTC)
        result = compute_next_fire_at(
            "daily_at", {"hour": 9, "minute": 0, "tz": "America/Los_Angeles"}, "coalesce", last_fire_utc, now
        )
        assert result is not None
        assert (result - last_fire_utc) == timedelta(hours=23)
        assert result.astimezone(tz) == datetime(2026, 3, 8, 9, 0, tzinfo=tz)

    def test_dst_fall_back_la(self) -> None:
        """``daily_at 09:00 America/Los_Angeles`` across fall-back advances by 25h."""
        tz = ZoneInfo("America/Los_Angeles")
        last_fire_utc = datetime(2026, 10, 31, 9, 0, tzinfo=tz).astimezone(UTC)
        now = datetime(2026, 10, 31, 9, 1, tzinfo=tz).astimezone(UTC)
        result = compute_next_fire_at(
            "daily_at", {"hour": 9, "minute": 0, "tz": "America/Los_Angeles"}, "coalesce", last_fire_utc, now
        )
        assert result is not None
        assert (result - last_fire_utc) == timedelta(hours=25)
        assert result.astimezone(tz) == datetime(2026, 11, 1, 9, 0, tzinfo=tz)

    def test_catch_up_advances_one_day_from_current_fire(self) -> None:
        """``catch_up`` advances exactly one day from the occurrence being fired.

        It must anchor on ``current_fire_at`` (5th positional here), NOT
        ``last_fired_at`` and NOT ``now``: the store stamps
        ``last_fired_at = now`` on every claim, so anchoring there would
        collapse ``catch_up`` into ``coalesce``. A deliberately far-ahead
        ``last_fired_at`` proves it is ignored.
        """
        result = compute_next_fire_at(
            "daily_at",
            {"hour": 9, "minute": 0, "tz": "UTC"},
            "catch_up",
            _utc(2026, 5, 23, 12, 0),  # last_fired_at stamped to now -- must be ignored
            _utc(2026, 5, 23, 12, 0),  # now
            _utc(2026, 5, 20, 9, 0),  # current_fire_at (the occurrence being fired)
        )
        assert result == _utc(2026, 5, 21, 9, 0)

    def test_default_tz_is_utc(self) -> None:
        """Omitting ``tz`` defaults to UTC."""
        result = compute_next_fire_at("daily_at", {"hour": 14, "minute": 30}, "coalesce", None, _utc(2026, 5, 22, 8, 0))
        assert result == _utc(2026, 5, 22, 14, 30)


class TestEveryNHours:
    """``every_n_hours`` branch."""

    def test_coalesce_anchors_on_now(self) -> None:
        result = compute_next_fire_at("every_n_hours", {"n": 3}, "coalesce", None, _utc(2026, 5, 22, 10, 0))
        assert result == _utc(2026, 5, 22, 13, 0)

    def test_catch_up_anchors_on_current_fire(self) -> None:
        """``catch_up`` steps one interval past the occurrence being fired,
        ignoring the (now-stamped) ``last_fired_at`` and ``now``."""
        result = compute_next_fire_at(
            "every_n_hours",
            {"n": 3},
            "catch_up",
            _utc(2026, 5, 22, 15, 0),  # last_fired_at = now -- must be ignored
            _utc(2026, 5, 22, 15, 0),  # now
            _utc(2026, 5, 22, 4, 0),  # current_fire_at
        )
        assert result == _utc(2026, 5, 22, 7, 0)

    def test_catch_up_drains_backlog_one_interval_per_tick(self) -> None:
        """A three-hour outage of a 30-minute schedule fires each missed
        occurrence, not once. Simulate successive ticks: each tick fires
        the occurrence at ``next_fire_at`` and advances by exactly one
        interval, with ``last_fired_at`` stamped to wall-clock ``now`` (as
        the store does). Under the old ``last_fired_at`` anchor this
        collapsed to two fires; the current-fire anchor drains all six."""
        step_config = {"n": 1}  # 1-hour interval expressed via every_n_hours
        # occurrence backlog: fired at 10:00, then down until 13:00.
        next_fire = _utc(2026, 5, 22, 11, 0)
        now = _utc(2026, 5, 22, 13, 0)
        fired: list[datetime] = []
        # bound the loop so a regression cannot spin forever
        for _ in range(20):
            if next_fire > now:
                break
            fired.append(next_fire)
            computed = compute_next_fire_at(
                "every_n_hours",
                step_config,
                "catch_up",
                now,  # last_fired_at stamped to now on claim
                now,
                next_fire,  # current_fire_at = the occurrence being fired
            )
            assert computed is not None
            next_fire = computed
        # occurrences at 11:00, 12:00, 13:00 all fire (3 missed intervals).
        assert fired == [
            _utc(2026, 5, 22, 11, 0),
            _utc(2026, 5, 22, 12, 0),
            _utc(2026, 5, 22, 13, 0),
        ]

    def test_zero_n_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            compute_next_fire_at("every_n_hours", {"n": 0}, "coalesce", None, _utc(2026, 5, 22))


class TestRandomWithinWindow:
    """``random_within_window`` branch -- random time-of-day in [start, end)."""

    def test_non_wrap_window_picks_inside_today(self) -> None:
        random.seed(42)
        result = compute_next_fire_at(
            "random_within_window",
            {"start_hour": 9, "end_hour": 21, "tz": "UTC"},
            "coalesce",
            None,
            _utc(2026, 5, 22, 10, 0),
        )
        assert result is not None
        assert _utc(2026, 5, 22, 10, 0) <= result < _utc(2026, 5, 22, 21, 0)

    def test_non_wrap_window_after_end_advances_to_tomorrow(self) -> None:
        random.seed(42)
        result = compute_next_fire_at(
            "random_within_window",
            {"start_hour": 9, "end_hour": 21, "tz": "UTC"},
            "coalesce",
            None,
            _utc(2026, 5, 22, 22, 0),
        )
        assert result is not None
        assert _utc(2026, 5, 23, 9, 0) <= result < _utc(2026, 5, 23, 21, 0)

    def test_wrap_window_picks_inside_overnight(self) -> None:
        """``start_hour > end_hour`` defines an overnight window."""
        random.seed(42)
        result = compute_next_fire_at(
            "random_within_window",
            {"start_hour": 22, "end_hour": 6, "tz": "UTC"},
            "coalesce",
            None,
            _utc(2026, 5, 22, 23, 0),
        )
        assert result is not None
        assert _utc(2026, 5, 22, 23, 0) <= result < _utc(2026, 5, 23, 6, 0)

    def test_wrap_window_in_morning_uses_current_window(self) -> None:
        """At 03:00 (overnight window 22:00->06:00) we're still in the active window."""
        random.seed(42)
        result = compute_next_fire_at(
            "random_within_window",
            {"start_hour": 22, "end_hour": 6, "tz": "UTC"},
            "coalesce",
            None,
            _utc(2026, 5, 22, 3, 0),
        )
        assert result is not None
        assert _utc(2026, 5, 22, 3, 0) <= result < _utc(2026, 5, 22, 6, 0)

    def test_after_fire_advances_to_next_day_not_same_day(self) -> None:
        """With the default ``fires_per_day=1``, a reschedule AFTER a fire
        rolls to tomorrow's window -- it must NOT pick another time later
        the same day (the runaway-over-firing bug). ``last_fired_at`` set
        marks 'already fired today'."""
        random.seed(42)
        result = compute_next_fire_at(
            "random_within_window",
            {"start_hour": 9, "end_hour": 21, "tz": "UTC"},
            "coalesce",
            _utc(2026, 5, 22, 14, 30),  # last_fired_at -- fired earlier today
            _utc(2026, 5, 22, 14, 30),  # now
        )
        assert result is not None
        assert _utc(2026, 5, 23, 9, 0) <= result < _utc(2026, 5, 23, 21, 0)

    def test_fires_per_day_partitions_window_into_slots(self) -> None:
        """``fires_per_day=3`` over a 9->21 window yields three slots
        (9-13, 13-17, 17-21). After firing in slot 0 the next fire lands
        in slot 1; after slot 2 it rolls to tomorrow's slot 0."""
        cfg = {"start_hour": 9, "end_hour": 21, "tz": "UTC", "fires_per_day": 3}

        random.seed(1)
        after_slot0 = compute_next_fire_at(
            "random_within_window", cfg, "coalesce", _utc(2026, 5, 22, 10, 0), _utc(2026, 5, 22, 10, 0)
        )
        assert after_slot0 is not None
        assert _utc(2026, 5, 22, 13, 0) <= after_slot0 < _utc(2026, 5, 22, 17, 0)

        random.seed(1)
        after_slot1 = compute_next_fire_at(
            "random_within_window", cfg, "coalesce", _utc(2026, 5, 22, 14, 0), _utc(2026, 5, 22, 14, 0)
        )
        assert after_slot1 is not None
        assert _utc(2026, 5, 22, 17, 0) <= after_slot1 < _utc(2026, 5, 22, 21, 0)

        random.seed(1)
        after_slot2 = compute_next_fire_at(
            "random_within_window", cfg, "coalesce", _utc(2026, 5, 22, 18, 0), _utc(2026, 5, 22, 18, 0)
        )
        assert after_slot2 is not None
        assert _utc(2026, 5, 23, 9, 0) <= after_slot2 < _utc(2026, 5, 23, 13, 0)

    def test_never_fired_can_fire_in_current_slot_today(self) -> None:
        """A never-fired schedule (``last_fired_at=None``) fires in the
        slot ``now`` currently occupies, not tomorrow."""
        random.seed(7)
        result = compute_next_fire_at(
            "random_within_window",
            {"start_hour": 9, "end_hour": 21, "tz": "UTC", "fires_per_day": 3},
            "coalesce",
            None,
            _utc(2026, 5, 22, 14, 0),
        )
        assert result is not None
        # now sits in slot 1 (13-17); a first fire may land from now to slot end
        assert _utc(2026, 5, 22, 14, 0) <= result < _utc(2026, 5, 22, 17, 0)

    def test_zero_fires_per_day_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive fires_per_day"):
            compute_next_fire_at(
                "random_within_window",
                {"start_hour": 9, "end_hour": 21, "tz": "UTC", "fires_per_day": 0},
                "coalesce",
                None,
                _utc(2026, 5, 22, 10, 0),
            )

    def test_start_equals_end_rejected(self) -> None:
        with pytest.raises(ValueError, match="start_hour != end_hour"):
            compute_next_fire_at(
                "random_within_window",
                {"start_hour": 10, "end_hour": 10, "tz": "UTC"},
                "coalesce",
                None,
                _utc(2026, 5, 22, 1, 0),
            )


class TestOneShotAt:
    """``one_shot_at`` branch -- terminal after first fire."""

    def test_future_iso_returned(self) -> None:
        result = compute_next_fire_at(
            "one_shot_at", {"fire_at_iso": "2026-05-22T14:00:00+00:00"}, "coalesce", None, _utc(2026, 5, 22, 10, 0)
        )
        assert result == _utc(2026, 5, 22, 14, 0)

    def test_past_iso_first_compute_fires_now(self) -> None:
        """A past one-shot fire that hasn't fired yet fires immediately."""
        now = _utc(2026, 5, 22, 18, 0)
        result = compute_next_fire_at(
            "one_shot_at", {"fire_at_iso": "2026-05-22T14:00:00+00:00"}, "coalesce", None, now
        )
        assert result == now

    def test_after_fire_returns_none(self) -> None:
        """After ``last_fired_at`` is set, return ``None`` (terminal)."""
        result = compute_next_fire_at(
            "one_shot_at",
            {"fire_at_iso": "2026-05-22T14:00:00+00:00"},
            "coalesce",
            _utc(2026, 5, 22, 14, 0),
            _utc(2026, 5, 22, 14, 1),
        )
        assert result is None


class TestCron:
    """``cron`` branch -- APScheduler's CronTrigger as a pure utility."""

    def test_hourly_cron(self) -> None:
        result = compute_next_fire_at("cron", {"expr": "0 * * * *"}, "coalesce", None, _utc(2026, 5, 22, 10, 30))
        assert result == _utc(2026, 5, 22, 11, 0)

    def test_every_three_hours(self) -> None:
        result = compute_next_fire_at("cron", {"expr": "0 */3 * * *"}, "coalesce", None, _utc(2026, 5, 22, 1, 30))
        assert result == _utc(2026, 5, 22, 3, 0)

    def test_catch_up_anchors_on_current_fire(self) -> None:
        result = compute_next_fire_at(
            "cron",
            {"expr": "0 */3 * * *"},
            "catch_up",
            _utc(2026, 5, 22, 20, 0),  # last_fired_at = now -- must be ignored
            _utc(2026, 5, 22, 20, 0),  # now
            _utc(2026, 5, 22, 9, 0),  # current_fire_at
        )
        assert result == _utc(2026, 5, 22, 12, 0)


class TestRelativeDelay:
    """``relative_delay`` branch -- fire once after ``delay`` from creation."""

    def test_first_fire_in_future(self) -> None:
        result = compute_next_fire_at("relative_delay", {"delay": "30m"}, "coalesce", None, _utc(2026, 5, 22, 10, 0))
        assert result == _utc(2026, 5, 22, 10, 30)

    def test_hours_unit(self) -> None:
        result = compute_next_fire_at("relative_delay", {"delay": "2h"}, "coalesce", None, _utc(2026, 5, 22, 10, 0))
        assert result == _utc(2026, 5, 22, 12, 0)

    def test_days_unit(self) -> None:
        result = compute_next_fire_at("relative_delay", {"delay": "3d"}, "coalesce", None, _utc(2026, 5, 22, 10, 0))
        assert result == _utc(2026, 5, 25, 10, 0)

    def test_after_fire_returns_none(self) -> None:
        result = compute_next_fire_at(
            "relative_delay", {"delay": "30m"}, "coalesce", _utc(2026, 5, 22, 10, 30), _utc(2026, 5, 22, 10, 31)
        )
        assert result is None

    def test_malformed_delay_rejected(self) -> None:
        with pytest.raises(ValueError):
            compute_next_fire_at("relative_delay", {"delay": "abc"}, "coalesce", None, _utc(2026, 5, 22, 10, 0))

    def test_unknown_unit_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown delay unit"):
            compute_next_fire_at("relative_delay", {"delay": "5y"}, "coalesce", None, _utc(2026, 5, 22, 10, 0))


class TestInterval:
    """``interval`` branch -- fire every N seconds."""

    def test_coalesce_anchors_on_now(self) -> None:
        result = compute_next_fire_at("interval", {"seconds": 1800}, "coalesce", None, _utc(2026, 5, 22, 10, 0))
        assert result == _utc(2026, 5, 22, 10, 30)

    def test_catch_up_anchors_on_current_fire(self) -> None:
        result = compute_next_fire_at(
            "interval",
            {"seconds": 1800},
            "catch_up",
            _utc(2026, 5, 22, 15, 0),  # last_fired_at = now -- must be ignored
            _utc(2026, 5, 22, 15, 0),  # now
            _utc(2026, 5, 22, 9, 0),  # current_fire_at
        )
        assert result == _utc(2026, 5, 22, 9, 30)

    def test_zero_seconds_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            compute_next_fire_at("interval", {"seconds": 0}, "coalesce", None, _utc(2026, 5, 22))


def test_unknown_schedule_type_raises() -> None:
    """An unknown ``schedule_type`` is a programming error."""
    with pytest.raises(ValueError, match="unknown schedule_type"):
        compute_next_fire_at("lunar_cycle", {}, "coalesce", None, _utc(2026, 5, 22))
