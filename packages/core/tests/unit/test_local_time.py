"""A moment, written for a reader in one zone, with the zone named."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from threetears.core.utils import render_local_time, resolve_zone

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
