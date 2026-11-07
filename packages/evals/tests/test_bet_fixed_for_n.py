"""The bounded test as one bet fixed for N cases, read at any prefix; and the resolution floor it implies.

A reading at k cases and one at k + 1 used to be two bets (each reshuffled by its own count, each stake tuned to it),
so stopping at the first that separated was optional stopping. With ``planned_n`` the order and every stake are fixed
by N before the first case, so each reading over the first k is a prefix of one bet. Held here: at k = N the reading
is today's, number for number, on the same pairs listed in case order; a prefix is the same bet's earlier capital
(the p never rises and the interval never widens as cases are added, since the capital's running maximum only grows);
and the floor, the smallest preference held on every case that the gate's own test separates, at 11 and at 25 cases.
"""

from __future__ import annotations

import pytest

from threetears.evals.analysis.stats import (
    bet_order,
    bounded_difference_interval,
    bounded_separation_p,
    smallest_separable_move,
)
from threetears.evals.ops.report_writer.writer_pairwise import resolution_line

#: The share range the writer gate reads on.
_RANGE = (0.0, 1.0)

#: A judge's graded lean on twelve cases, 1-5, its share (lean - 1) / 4: the pairwise tests' spread of preferences.
_LEAN = (5, 5, 4, 4, 4, 3, 5, 4, 3, 2, 5, 4)

#: Twelve shares with spread, in case order.
TWELVE = [(lean - 1) / 4 for lean in _LEAN]

#: Twenty-five shares with spread, in case order, one a clean loss.
TWENTY_FIVE = [
    1.0, 0.75, 1.0, 0.5, 1.0, 0.875, 0.25, 1.0, 0.75, 1.0, 0.625, 1.0, 0.5,
    1.0, 0.75, 1.0, 0.0, 1.0, 0.875, 0.75, 1.0, 0.5, 1.0, 0.75, 1.0,
]  # fmt: skip

#: Today's fixed-n readings on those shares (control one minus each), pinned before the bet took ``planned_n``.
PINNED = {
    12: (0.043532428078897685, (0.012729058507829905, 0.8665763814933598)),
    25: (0.00038675382448340526, (0.22633092757314444, 0.8531258325092494)),
}


def _control(shares: list[float]) -> list[float]:
    return [1.0 - share for share in shares]


def _in_bet_order(shares: list[float]) -> list[float]:
    """The shares in the order a bet fixed for their count reads them."""
    return [shares[index] for index in bet_order(len(shares))]


@pytest.mark.parametrize("shares", [TWELVE, TWENTY_FIVE], ids=["12", "25"])
def test_at_n_the_bet_fixed_for_n_reads_todays_numbers_exactly(shares: list[float]) -> None:
    n = len(shares)
    p, interval = PINNED[n]
    # Today's reading, no planned_n: the pairs in case order, reshuffled by their count.
    assert bounded_separation_p(_control(shares), shares, paired=True, value_range=_RANGE) == p
    assert bounded_difference_interval(_control(shares), shares, paired=True, value_range=_RANGE) == interval
    # The bet fixed for N = n, its pairs given in the order bet: the same p and interval, to the last bit.
    bet = _in_bet_order(shares)
    assert bounded_separation_p(_control(bet), bet, paired=True, value_range=_RANGE, planned_n=n) == p
    assert bounded_difference_interval(_control(bet), bet, paired=True, value_range=_RANGE, planned_n=n) == interval


def test_each_prefix_is_the_same_bet_so_the_reading_only_tightens_as_cases_are_added() -> None:
    bet = _in_bet_order(TWENTY_FIVE)
    n = len(bet)
    readings = [
        (
            bounded_separation_p(_control(bet[:k]), bet[:k], paired=True, value_range=_RANGE, planned_n=n),
            bounded_difference_interval(_control(bet[:k]), bet[:k], paired=True, value_range=_RANGE, planned_n=n),
        )
        for k in range(2, n + 1)
    ]
    for (p_before, before), (p_after, after) in zip(readings, readings[1:], strict=False):
        assert p_before is not None and p_after is not None and before is not None and after is not None
        assert p_after <= p_before
        assert before[0] <= after[0] and after[1] <= before[1]
    # The last prefix is the whole bet: today's reading.
    assert readings[-1] == PINNED[n]
    # A prefix is not a smaller bet: the 9-case prefix of the 25-case bet is not the 9-case bet.
    nine = bet[:9]
    alone = bounded_separation_p(_control(nine), nine, paired=True, value_range=_RANGE, planned_n=9)
    assert readings[7][0] != alone


def test_a_landslide_first_separates_at_case_seven_of_twenty_five() -> None:
    """A candidate preferred outright on every case, under the bet fixed for 25: REVIEW.md's figure."""
    separated = [
        k
        for k in range(2, 26)
        if (p := bounded_separation_p([0.0] * k, [1.0] * k, paired=True, value_range=_RANGE, planned_n=25)) is not None
        and p < 0.05
    ]
    assert separated[0] == 7


def test_a_bet_holds_no_more_cases_than_it_was_fixed_for_and_is_read_paired() -> None:
    with pytest.raises(ValueError, match="fixed for 3 cases"):
        bounded_separation_p([0.0] * 4, [1.0] * 4, paired=True, value_range=_RANGE, planned_n=3)
    with pytest.raises(ValueError, match="fixed for 3 cases"):
        bounded_difference_interval([0.0] * 4, [1.0] * 4, paired=True, value_range=_RANGE, planned_n=3)
    with pytest.raises(ValueError, match="read paired"):
        bounded_separation_p([0.0] * 4, [1.0] * 4, paired=False, value_range=_RANGE, planned_n=8)


@pytest.mark.parametrize(("cases", "share", "percent"), [(11, 0.7232, 73), (25, 0.5903, 60)])
def test_the_resolution_floor_is_the_gates_own_test_inverted(cases: int, share: float, percent: int) -> None:
    """About 0.72 at 11 cases and 0.59 at 25 (REVIEW.md's 0.6), each the smallest share held on every case that
    separates; the line rounds up, so it never claims a share the test would not separate."""
    move = smallest_separable_move(cases, value_range=_RANGE)
    assert move is not None
    floor = (1.0 + move) / 2.0
    assert floor == pytest.approx(share, abs=1e-4)

    def separates(at: float) -> bool:
        control, candidate = [1.0 - at] * cases, [at] * cases
        p = bounded_separation_p(control, candidate, paired=True, value_range=_RANGE)
        interval = bounded_difference_interval(control, candidate, paired=True, value_range=_RANGE)
        return p is not None and p < 0.05 and interval is not None and interval[0] > 0

    assert separates(floor) and not separates(floor - 1e-4)
    assert resolution_line(cases) == (
        f"At {cases} cases this run can separate only a preference of {percent}% or more, held on every case."
    )


def test_the_verdict_changes_once_across_the_move_so_bisection_finds_the_floor() -> None:
    """Over every count from 2 to 25, on a grid of 1/500 of the range, a move held on every case that separates is
    never followed by a larger one that does not: the monotonicity the bisection rests on."""
    for cases in range(2, 26):
        seen = False
        for step in range(1, 501):
            move = step / 500
            p = bounded_separation_p([0.0] * cases, [move] * cases, paired=True, value_range=_RANGE)
            interval = bounded_difference_interval([0.0] * cases, [move] * cases, paired=True, value_range=_RANGE)
            separates = p is not None and p < 0.05 and interval is not None and interval[0] > 0
            assert separates or not seen, (cases, move)
            seen = seen or separates
        floor = smallest_separable_move(cases, value_range=_RANGE)
        assert (floor is None) == (not seen), cases


def test_the_line_carries_the_last_runs_projection_when_it_is_given() -> None:
    assert resolution_line(11, projected=43) == (
        "At 11 cases this run can separate only a preference of 73% or more, held on every case; the last run on "
        "these cases projected 43 to settle."
    )
    assert resolution_line(25, projected=10_001).endswith(
        "; the last run on these cases projected more than 10,000 to settle."  # spell_count (N-6)
    )


def test_too_few_cases_separate_nothing_and_the_line_says_so() -> None:
    assert smallest_separable_move(5, value_range=_RANGE) is None
    assert resolution_line(5) == "At 5 cases this run cannot separate any preference, even one held on every case."


#: The bet's order for these counts, as it was taken: a Python whose generator moved them would move every bet.
_PINNED: dict[int, tuple[int, ...]] = {
    7: (0, 6, 4, 3, 2, 5, 1),
    12: (5, 11, 8, 6, 9, 3, 2, 1, 0, 4, 7, 10),
    25: (7, 16, 14, 15, 12, 6, 11, 1, 19, 0, 8, 18, 22, 4, 9, 17, 23, 24, 20, 2, 5, 3, 21, 10, 13),
}


@pytest.mark.parametrize("n", sorted(_PINNED))
def test_the_bet_s_order_is_its_own_derivation_the_same_each_time_and_pinned(n: int) -> None:
    """The order is derived by the package's own shuffle, not ``random.shuffle``: the same count gives the same order
    each time, a permutation of its cases, and the order pinned when it was taken."""
    first, second = bet_order(n), bet_order(n)
    assert first == second == _PINNED[n]
    assert sorted(first) == list(range(n))
