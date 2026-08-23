"""Behaviour of the shared cache-entry age predicate.

Every case injects its clock readings. A test that slept to prove an
expiry could not exercise an hour-long window at all, and would be flaky
at any window.
"""

from __future__ import annotations

from threetears.core.cache.base import _entry_is_fresh


class TestWithinMaxAge:
    def test_an_entry_stamped_now_is_fresh(self) -> None:
        assert _entry_is_fresh(100.0, now_monotonic=100.0, max_age_seconds=30.0) is True

    def test_an_entry_inside_the_window_is_fresh(self) -> None:
        assert _entry_is_fresh(100.0, now_monotonic=129.0, max_age_seconds=30.0) is True

    def test_the_boundary_is_inclusive(self) -> None:
        """Exactly at max age still serves.

        Matches the pre-existing scan-cache behaviour this predicate was
        extracted from, so migrating that caller changed nothing.
        """
        assert _entry_is_fresh(100.0, now_monotonic=130.0, max_age_seconds=30.0) is True


class TestPastMaxAge:
    def test_an_entry_past_the_window_is_stale(self) -> None:
        assert _entry_is_fresh(100.0, now_monotonic=131.0, max_age_seconds=30.0) is False

    def test_a_long_window_is_exercised_without_waiting_for_it(self) -> None:
        assert _entry_is_fresh(0.0, now_monotonic=3601.0, max_age_seconds=3600.0) is False


class TestUnstampedEntries:
    def test_an_unstamped_entry_is_fresh(self) -> None:
        """No stamp means the row was authored locally and never pulled through.

        Expiring it would discard a local write in favour of the older
        value a pull-through would return.
        """
        assert _entry_is_fresh(None, now_monotonic=1_000_000.0, max_age_seconds=1.0) is True

    def test_an_unstamped_entry_is_fresh_at_a_zero_window(self) -> None:
        assert _entry_is_fresh(None, now_monotonic=0.0, max_age_seconds=0.0) is True


class TestDegenerateWindows:
    def test_a_zero_window_expires_anything_stamped_in_the_past(self) -> None:
        assert _entry_is_fresh(0.0, now_monotonic=0.5, max_age_seconds=0.0) is False

    def test_a_zero_window_still_serves_an_entry_stamped_this_instant(self) -> None:
        assert _entry_is_fresh(5.0, now_monotonic=5.0, max_age_seconds=0.0) is True
