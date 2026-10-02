"""The release heading is dated the operator's calendar day, not the UTC day."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from threetears.enforcement.release.cli import release_date

# 03:00 UTC on 2026-10-01 is still the evening of 2026-09-30 in Los Angeles: the
# off-by-a-day this guards against dated a release cut that evening as the 1st.
_EVENING_WEST_OF_UTC = datetime(2026, 10, 1, 3, 0, tzinfo=UTC)


def test_a_release_cut_in_the_evening_west_of_utc_is_dated_that_day() -> None:
    """the operator's day wins over the UTC day that has already begun."""
    assert release_date(_EVENING_WEST_OF_UTC, ZoneInfo("America/Los_Angeles")) == "2026-09-30"


def test_a_release_cut_in_utc_is_dated_the_utc_day() -> None:
    """control: the same instant read in UTC is the 1st."""
    assert release_date(_EVENING_WEST_OF_UTC, UTC) == "2026-10-01"


def test_a_zone_east_of_utc_can_be_a_day_ahead() -> None:
    """the same rule moves the date forward east of UTC."""
    late_utc = datetime(2026, 9, 30, 22, 0, tzinfo=UTC)
    assert release_date(late_utc, timezone(timedelta(hours=10))) == "2026-10-01"


def test_a_naive_instant_is_refused() -> None:
    """a naive instant has no calendar day to read."""
    with pytest.raises(ValueError, match="aware instant"):
        release_date(datetime(2026, 10, 1, 3, 0))  # noqa: DTZ001 -- the refused input
