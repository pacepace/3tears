"""How a wake's schedule and its times are said to whoever reads a tool's result."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from threetears.agent.wake.wording import describe_schedule, format_wake_time


class TestAScheduleInWords:
    @pytest.mark.parametrize(
        ("schedule_type", "config", "said"),
        [
            ("interval", {"seconds": 3600}, "every hour"),
            ("interval", {"seconds": 14400}, "every 4 hours"),
            ("interval", {"seconds": 5400}, "every 90 minutes"),
            ("interval", {"seconds": 60}, "every minute"),
            ("interval", {"seconds": 45}, "every 45 seconds"),
            ("every_n_hours", {"n": 1}, "every hour"),
            ("every_n_hours", {"n": 6}, "every 6 hours"),
            ("daily_at", {"hour": 3, "minute": 5, "tz": "America/Los_Angeles"}, "daily at 03:05 America/Los_Angeles"),
            ("daily_at", {"hour": 3}, "daily at 03:00 UTC"),
            (
                "random_within_window",
                {"start_hour": 9, "end_hour": 17, "tz": "Europe/Dublin", "fires_per_day": 2},
                "2 times a day between 09:00 and 17:00 Europe/Dublin",
            ),
            ("random_within_window", {"start_hour": 22, "end_hour": 2}, "once a day between 22:00 and 02:00 UTC"),
            ("cron", {"expr": "0 9 * * 1"}, "cron 0 9 * * 1"),
            ("one_shot_at", {"fire_at_iso": "2026-10-10T04:11:00+00:00"}, "once at 2026-10-10T04:11:00+00:00"),
            ("relative_delay", {"delay": "30m"}, "once, 30m after it was set"),
        ],
    )
    def test_each_type_is_said_the_way_a_person_would(
        self, schedule_type: str, config: dict[str, Any], said: str
    ) -> None:
        assert describe_schedule(schedule_type, config) == said

    @pytest.mark.parametrize(
        ("schedule_type", "config", "said"),
        [
            ("interval", {}, "interval {}"),
            ("interval", {"seconds": "soon"}, 'interval {"seconds": "soon"}'),
            ("lunar", {"phase": "full"}, 'lunar {"phase": "full"}'),
        ],
    )
    def test_a_config_it_cannot_say_is_shown_as_stored_and_never_guessed(
        self, schedule_type: str, config: dict[str, Any], said: str
    ) -> None:
        assert describe_schedule(schedule_type, config) == said


class TestAWakeTime:
    _MOMENT = datetime(2026, 10, 10, 4, 11, 20, 123456, tzinfo=UTC)

    def test_in_a_named_zone_it_is_local_and_names_the_zone(self) -> None:
        assert format_wake_time(self._MOMENT, ZoneInfo("America/Los_Angeles")) == "Oct 9, 2026 at 9:11 PM PDT"

    def test_with_no_zone_named_it_is_as_stored(self) -> None:
        assert format_wake_time(self._MOMENT) == "2026-10-10T04:11:20.123456+00:00"

    def test_no_moment_is_said_in_the_callers_word_for_it(self) -> None:
        assert format_wake_time(None) == "-" and format_wake_time(None, unset="never") == "never"
