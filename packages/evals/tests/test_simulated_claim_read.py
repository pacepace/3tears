"""A fast seeded case of each claim-read calibration claim the statistics certificate holds (``claim_read_nulls.py``).

The certificate counts thousands of seeded campaigns; this reads a few dozen, on seeds the certificate does not read, so
a change that breaks the interval or the lean test goes red in the unit suite and not only at re-certification. Each
bound below is stated with how often a correct build would trip it on other seeds.
"""

from __future__ import annotations

from packages.evals.tests.claim_read_nulls import one

#: Seeds past every index the certificate reads.
_OFFSET = 100_000


def test_kappas_lower_end_rarely_sits_above_the_campaigns_own_kappa_at_two_claims_a_cell() -> None:
    """At the certified miss rate (about 5% at two arms), more than four misses in twenty draws happens about three
    times in a thousand."""
    outcomes = [one("equal.2-arms", _OFFSET + index) for index in range(20)]

    assert sum(outcome.low_missed for outcome in outcomes) <= 4
    assert sum(outcome.high_missed for outcome in outcomes) <= 3
    assert sum(outcome.arm_low_missed for outcome in outcomes) <= 4


def test_a_fair_instrument_over_arms_whose_verdict_mixes_differ_sharply_is_rarely_read_as_leaning() -> None:
    """One arm researched (few false claims, small cells, unsettled claims supported), one did not; the instrument is
    fair on both. At the certified false-lean rate (about 5%), more than four leans in twenty draws happens about three
    times in a thousand, and so for groundedness under the draw by truth verdict."""
    assert sum(one("mixed.2-arms", _OFFSET + index).leaned for index in range(20)) <= 4
    assert sum(one("mixed-grounded.2-arms", _OFFSET + index).leaned for index in range(20)) <= 4


def test_one_arm_whose_instrument_misses_most_of_its_false_claims_is_read_as_a_lean() -> None:
    """At the certified power (about two in three, recall 0.80 against 0.45 at 400 labels), fewer than four leans in
    twelve draws happens about three times in a thousand."""
    outcomes = [one("lopsided-80-45", _OFFSET + index) for index in range(12)]

    assert sum(outcome.leaned for outcome in outcomes) >= 4
