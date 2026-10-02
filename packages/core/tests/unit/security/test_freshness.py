"""tests for the one issue-time check every proof verifier shares.

The contract this pins:

- the two directions are separate numbers: an issue time may be OLD by up to ``max_age`` and
  AHEAD of the verifier's clock by up to ``future_tolerance``, and neither bounds the other;
- both edges are inclusive, so the documented tolerance is usable to its last second;
- the platform's future tolerance is five seconds, and a replay guard sized for it refuses for
  ten seconds after a wipe -- the number a broker restart costs;
- a negative bound is a wiring error, not a window that refuses everything.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from threetears.core.coordination.replay_guard import CLOCK_DRIFT_ALLOWANCE
from threetears.core.security import (
    CLIENT_ISSUE_TIME_FUTURE_TOLERANCE,
    DEFAULT_PROOF_MAX_AGE,
    ISSUE_TIME_FUTURE_TOLERANCE,
    issue_time_is_fresh,
)

_NOW = 1_800_000_000


def _fresh(offset_seconds: int) -> bool:
    """whether an issue time ``offset_seconds`` from the verifier's clock is accepted by the defaults.

    :param offset_seconds: positive is ahead of the verifier's clock, negative is behind it
    :ptype offset_seconds: int
    :return: the verdict
    :rtype: bool
    """
    return issue_time_is_fresh(
        _NOW + offset_seconds,
        now=_NOW,
        max_age=DEFAULT_PROOF_MAX_AGE,
        future_tolerance=ISSUE_TIME_FUTURE_TOLERANCE,
    )


class TestThePlatformNumbers:
    """the constants are the contract: a change to either is a change to what a restart costs."""

    def test_the_future_tolerance_is_five_seconds(self) -> None:
        assert timedelta(seconds=5) == ISSUE_TIME_FUTURE_TOLERANCE

    def test_a_client_signed_proof_keeps_a_minute_and_its_guard_the_sixty_five_second_reach(self) -> None:
        # the signer is a browser or a laptop whose clock the platform does not keep, so the
        # small number is not applied to it; the cost is the longer refusal after a wipe.
        assert timedelta(seconds=60) == CLIENT_ISSUE_TIME_FUTURE_TOLERANCE
        assert CLIENT_ISSUE_TIME_FUTURE_TOLERANCE + CLOCK_DRIFT_ALLOWANCE == timedelta(seconds=65)
        assert issue_time_is_fresh(
            _NOW + 55, now=_NOW, max_age=DEFAULT_PROOF_MAX_AGE, future_tolerance=CLIENT_ISSUE_TIME_FUTURE_TOLERANCE
        )
        assert not issue_time_is_fresh(
            _NOW + 61, now=_NOW, max_age=DEFAULT_PROOF_MAX_AGE, future_tolerance=CLIENT_ISSUE_TIME_FUTURE_TOLERANCE
        )

    def test_the_past_window_is_still_sixty_seconds(self) -> None:
        # separating the directions must not shorten how long a slow request has to arrive.
        assert timedelta(seconds=60) == DEFAULT_PROOF_MAX_AGE

    def test_a_guard_sized_for_it_refuses_for_ten_seconds_after_a_wipe(self) -> None:
        assert ISSUE_TIME_FUTURE_TOLERANCE + CLOCK_DRIFT_ALLOWANCE == timedelta(seconds=10)


class TestTheFutureSide:
    """an issue time ahead of the verifier's clock is accepted only inside the small tolerance."""

    def test_four_seconds_ahead_is_accepted(self) -> None:
        assert _fresh(4) is True

    def test_the_edge_itself_is_accepted(self) -> None:
        assert _fresh(5) is True

    def test_six_seconds_ahead_is_refused(self) -> None:
        assert _fresh(6) is False

    def test_what_the_symmetric_window_used_to_admit_is_refused(self) -> None:
        # the old single leeway accepted an issue time a full minute ahead; that minute is what
        # every replay guard had to refuse for after a wipe.
        assert _fresh(55) is False
        assert _fresh(60) is False


class TestThePastSide:
    """an old proof keeps the whole window it had before the directions were separated."""

    def test_now_is_accepted(self) -> None:
        assert _fresh(0) is True

    def test_an_old_proof_inside_the_window_is_accepted(self) -> None:
        assert _fresh(-55) is True

    def test_the_edge_itself_is_accepted(self) -> None:
        assert _fresh(-60) is True

    def test_a_proof_past_the_window_is_refused(self) -> None:
        assert _fresh(-61) is False


class TestTheDirectionsAreIndependent:
    """neither bound stands in for the other."""

    def test_a_wide_past_window_does_not_widen_the_future(self) -> None:
        assert (
            issue_time_is_fresh(
                _NOW + 6, now=_NOW, max_age=timedelta(hours=1), future_tolerance=ISSUE_TIME_FUTURE_TOLERANCE
            )
            is False
        )

    def test_a_zero_future_tolerance_still_admits_the_past(self) -> None:
        assert issue_time_is_fresh(_NOW - 30, now=_NOW, max_age=DEFAULT_PROOF_MAX_AGE, future_tolerance=timedelta(0))
        assert not issue_time_is_fresh(_NOW + 1, now=_NOW, max_age=DEFAULT_PROOF_MAX_AGE, future_tolerance=timedelta(0))

    @pytest.mark.parametrize(
        ("max_age", "future_tolerance"),
        [(timedelta(seconds=-1), timedelta(seconds=5)), (timedelta(seconds=60), timedelta(seconds=-1))],
    )
    def test_a_negative_bound_is_a_wiring_error(self, max_age: timedelta, future_tolerance: timedelta) -> None:
        with pytest.raises(ValueError, match="must not be negative"):
            issue_time_is_fresh(_NOW, now=_NOW, max_age=max_age, future_tolerance=future_tolerance)
