"""tests for the key pool's billing periods: ids, bounds, and the anchor day."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from threetears.models.key_pool.period import period_bounds, period_id


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


class TestCalendarMonth:
    """the default anchor of 1: a period is a UTC calendar month."""

    def test_id_names_the_month(self) -> None:
        assert period_id(_utc(2026, 11, 26, 13)) == "2026-11"

    def test_default_anchor_is_the_first(self) -> None:
        assert period_bounds(_utc(2026, 11, 26)) == period_bounds(_utc(2026, 11, 26), 1)

    def test_bounds_are_the_month(self) -> None:
        assert period_bounds(_utc(2026, 11, 26, 13)) == (_utc(2026, 11, 1), _utc(2026, 12, 1))

    def test_rollover_instant_starts_the_new_period(self) -> None:
        assert period_id(_utc(2026, 12, 1)) == "2026-12"
        assert period_id(_utc(2026, 11, 30, 23, 59, 59)) == "2026-11"

    def test_december_rolls_into_january(self) -> None:
        assert period_bounds(_utc(2026, 12, 31, 23)) == (_utc(2026, 12, 1), _utc(2027, 1, 1))
        assert period_id(_utc(2027, 1, 1)) == "2027-01"

    def test_another_offset_is_read_in_utc(self) -> None:
        """00:30 on Dec 1 at +01:00 is still Nov 30 in UTC."""
        plus_one = timezone(timedelta(hours=1))
        assert period_id(datetime(2026, 12, 1, 0, 30, tzinfo=plus_one)) == "2026-11"


class TestAnchorDay:
    """an anchor from 2 to 28 starts each period on that day."""

    def test_id_names_the_start_date(self) -> None:
        assert period_id(_utc(2026, 11, 26), 15) == "2026-11-15"

    def test_before_the_anchor_is_the_previous_period(self) -> None:
        assert period_bounds(_utc(2026, 11, 14, 23, 59), 15) == (_utc(2026, 10, 15), _utc(2026, 11, 15))
        assert period_id(_utc(2026, 11, 14, 23, 59), 15) == "2026-10-15"

    def test_on_the_anchor_is_the_new_period(self) -> None:
        assert period_id(_utc(2026, 11, 15), 15) == "2026-11-15"

    def test_anchor_28_in_a_common_february(self) -> None:
        assert period_bounds(_utc(2027, 2, 27), 28) == (_utc(2027, 1, 28), _utc(2027, 2, 28))
        assert period_bounds(_utc(2027, 2, 28), 28) == (_utc(2027, 2, 28), _utc(2027, 3, 28))
        assert period_id(_utc(2027, 3, 1), 28) == "2027-02-28"

    def test_anchor_28_in_a_leap_february(self) -> None:
        """the 29th of a leap February falls in the period that began on the 28th."""
        assert period_bounds(_utc(2028, 2, 29, 12), 28) == (_utc(2028, 2, 28), _utc(2028, 3, 28))
        assert period_id(_utc(2028, 2, 29), 28) == "2028-02-28"

    def test_calendar_month_february_in_leap_and_common_years(self) -> None:
        assert period_bounds(_utc(2028, 2, 29, 23)) == (_utc(2028, 2, 1), _utc(2028, 3, 1))
        assert period_bounds(_utc(2027, 2, 28, 23)) == (_utc(2027, 2, 1), _utc(2027, 3, 1))

    def test_anchor_day_period_spans_december_into_january(self) -> None:
        assert period_bounds(_utc(2027, 1, 10), 20) == (_utc(2026, 12, 20), _utc(2027, 1, 20))
        assert period_bounds(_utc(2026, 12, 25), 20) == (_utc(2026, 12, 20), _utc(2027, 1, 20))
        assert period_id(_utc(2027, 1, 10), 20) == "2026-12-20"

    @pytest.mark.parametrize("anchor_day", [0, 29, 31, -1])
    def test_an_anchor_outside_1_to_28_raises(self, anchor_day: int) -> None:
        with pytest.raises(ValueError, match="anchor_day"):
            period_id(_utc(2026, 11, 26), anchor_day)
        with pytest.raises(ValueError, match="anchor_day"):
            period_bounds(_utc(2026, 11, 26), anchor_day)

    def test_a_bool_anchor_raises(self) -> None:
        with pytest.raises(ValueError, match="anchor_day"):
            period_id(_utc(2026, 11, 26), True)

    def test_a_naive_instant_raises(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            period_id(datetime(2026, 11, 26), 1)


# Hypothesis's timing checks measure the host, not the property, and the shared test pool includes a slow host.
_NO_TIMING_CHECKS = settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])

_INSTANTS = st.datetimes(
    min_value=datetime(2000, 1, 1),
    max_value=datetime(2200, 12, 31),
    timezones=st.just(UTC),
)


class TestPeriodProperties:
    """what holds for every instant and anchor."""

    @_NO_TIMING_CHECKS
    @given(now=_INSTANTS, anchor_day=st.integers(min_value=1, max_value=28))
    def test_bounds_contain_now_and_start_on_the_anchor(self, now: datetime, anchor_day: int) -> None:
        start, rollover = period_bounds(now, anchor_day)
        assert start <= now < rollover
        assert start.day == anchor_day == rollover.day
        assert start.tzinfo is UTC and rollover.tzinfo is UTC
        months_apart = (rollover.year - start.year) * 12 + rollover.month - start.month
        assert months_apart == 1

    @_NO_TIMING_CHECKS
    @given(now=_INSTANTS, anchor_day=st.integers(min_value=1, max_value=28))
    def test_the_rollover_starts_the_next_period(self, now: datetime, anchor_day: int) -> None:
        start, rollover = period_bounds(now, anchor_day)
        assert period_bounds(rollover, anchor_day)[0] == rollover
        assert period_bounds(rollover - timedelta(microseconds=1), anchor_day)[0] == start

    @_NO_TIMING_CHECKS
    @given(now=_INSTANTS, anchor_day=st.integers(min_value=1, max_value=28))
    def test_ids_are_stable_within_a_period_and_sort_in_order(self, now: datetime, anchor_day: int) -> None:
        start, rollover = period_bounds(now, anchor_day)
        current = period_id(now, anchor_day)
        assert period_id(start, anchor_day) == current
        assert period_id(rollover - timedelta(microseconds=1), anchor_day) == current
        assert period_id(rollover, anchor_day) > current
