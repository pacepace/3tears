"""A moment, written for a reader in one zone, with the zone named."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from threetears.core.utils import parse_date_bound, render_local_time, resolve_zone

_LA = "America/Los_Angeles"


class TestATimeIsShownInTheReadersZone:
    def test_a_utc_moment_is_the_evening_before_on_the_pacific_coast(self) -> None:
        moment = datetime(2026, 10, 10, 4, 11, 20, 123456, tzinfo=UTC)
        assert render_local_time(moment, _LA) == "Oct 9, 2026 at 9:11 PM PDT"

    def test_the_zone_name_follows_the_season(self) -> None:
        assert render_local_time(datetime(2026, 1, 15, 20, 0, tzinfo=UTC), _LA) == "Jan 15, 2026 at 12:00 PM PST"

    def test_with_no_zone_the_time_is_utc_and_says_so(self) -> None:
        assert render_local_time(datetime(2026, 3, 12, 14, 30, tzinfo=UTC)) == "Mar 12, 2026 at 2:30 PM UTC"

    def test_a_moment_with_no_zone_of_its_own_is_taken_as_utc(self) -> None:
        assert render_local_time(datetime(2026, 10, 10, 4, 11), _LA) == "Oct 9, 2026 at 9:11 PM PDT"

    def test_a_moment_in_another_zone_is_converted_not_relabelled(self) -> None:
        tokyo = datetime(2026, 10, 10, 13, 11, tzinfo=ZoneInfo("Asia/Tokyo"))
        assert render_local_time(tokyo, _LA) == "Oct 9, 2026 at 9:11 PM PDT"

    @pytest.mark.parametrize(("hour", "said"), [(0, "12:05 AM"), (11, "11:05 AM"), (12, "12:05 PM"), (23, "11:05 PM")])
    def test_midnight_and_noon_are_twelve(self, hour: int, said: str) -> None:
        assert render_local_time(datetime(2026, 6, 1, hour, 5, tzinfo=UTC)) == f"Jun 1, 2026 at {said} UTC"

    def test_a_tzinfo_is_taken_as_it_is(self) -> None:
        plus_two = timezone(timedelta(hours=2), "EET")
        assert render_local_time(datetime(2026, 6, 1, 10, 0, tzinfo=UTC), plus_two) == "Jun 1, 2026 at 12:00 PM EET"


class TestTheZoneACallerNames:
    def test_no_zone_stays_no_zone(self) -> None:
        assert resolve_zone(None) is None

    def test_a_name_becomes_its_zone(self) -> None:
        assert resolve_zone(_LA) == ZoneInfo(_LA)

    def test_a_tzinfo_is_returned_as_it_is(self) -> None:
        assert resolve_zone(UTC) is UTC

    @pytest.mark.parametrize("name", ["Mars/Olympus", "", "not a zone", "../etc/passwd"])
    def test_a_name_that_is_not_a_zone_is_refused_not_read_as_utc(self, name: str) -> None:
        with pytest.raises(ValueError, match="is not a time zone this system knows"):
            resolve_zone(name)
        with pytest.raises(ValueError, match="is not a time zone this system knows"):
            render_local_time(datetime(2026, 6, 1, tzinfo=UTC), name)


class TestADateFilterIsTheReadersDay:
    def test_a_bare_date_before_runs_to_the_end_of_that_day_in_the_readers_zone(self) -> None:
        moment, error = parse_date_bound("date_before", "2026-10-09", ZoneInfo(_LA))
        assert error is None
        assert moment == datetime(2026, 10, 9, 23, 59, 59, 999999, tzinfo=ZoneInfo(_LA))
        # an evening on the Pacific coast is the next day in UTC, and is still inside the day asked for
        assert datetime(2026, 10, 10, 4, 11, tzinfo=UTC) <= moment

    def test_a_bare_date_after_starts_at_the_days_first_minute(self) -> None:
        moment, error = parse_date_bound("date_after", "2026-10-09", ZoneInfo(_LA))
        assert (moment, error) == (datetime(2026, 10, 9, 0, 0, tzinfo=ZoneInfo(_LA)), None)

    def test_a_time_with_no_zone_is_in_the_readers_zone_and_is_not_moved_to_the_days_end(self) -> None:
        moment, _ = parse_date_bound("date_before", "2026-10-09T08:30:00", ZoneInfo(_LA))
        assert moment == datetime(2026, 10, 9, 8, 30, tzinfo=ZoneInfo(_LA))

    def test_a_time_that_names_its_zone_keeps_it(self) -> None:
        moment, _ = parse_date_bound("date_before", "2026-10-09T08:30:00+00:00", ZoneInfo(_LA))
        assert moment == datetime(2026, 10, 9, 8, 30, tzinfo=UTC)

    def test_with_no_zone_the_day_is_utcs(self) -> None:
        moment, _ = parse_date_bound("date_before", "2026-10-09", None)
        assert moment == datetime(2026, 10, 9, 23, 59, 59, 999999, tzinfo=UTC)

    @pytest.mark.parametrize("blank", [None, "", "   "])
    def test_nothing_given_is_no_bound(self, blank: str | None) -> None:
        assert parse_date_bound("date_after", blank, None) == (None, None)

    def test_what_is_not_a_date_is_said(self) -> None:
        moment, error = parse_date_bound("date_before", "next tuesday", None)
        assert moment is None and error is not None and "date_before" in error and "next tuesday" in error
