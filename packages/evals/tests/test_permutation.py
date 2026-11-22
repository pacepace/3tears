"""The stratified conditional permutation draws its exact law, and its sequential read changes a decision only within
its stated bound.

``stratified_hypergeometric_totals`` is held to the law it draws, computed here by enumerating every arrangement of
every stratum: few strata of their own margins, many strata sharing theirs, one stratum too large to tabulate, and
strata whose draw is fixed (all marked, or one group). Each is a seeded goodness-of-fit test whose chance of a false
red is under one in a thousand. ``sequential_permutation_p`` is held to its rule by the streams it is fed: when it
stops, with what p, and that the rule it follows adds at most 0.0003 to a test's false-reject rate,
computed from the binomial law over every law a valid test's true p can have.

Mutations that turn this file red (each run against a saved copy and restored from it):

- ``_totals``: dropping the last step of a counts draw, so many strata sharing margins lose marked members.
- ``_hypergeometric_table``: one member too many in a downward term's ratio, so the large stratum's law shifts.
- ``_rule``: an early look's most count one higher, so the cost bound fails.
- ``sequential_permutation_p``: stopping on reaching the count before the one that settles the read.
"""

from __future__ import annotations

import itertools
import math
import random
from collections import Counter
from collections.abc import Callable
from statistics import NormalDist

import pytest

from threetears.evals.analysis.permutation import sequential_permutation_p, stratified_hypergeometric_totals

#: The rule ``sequential_permutation_p`` states: its early looks, the chance at the level a look's count must sit
#: under to stop the read, and the most the looks add to the false-reject rate at 999 draws and a level of 5%.
EARLY_LOOKS = (99, 199, 399, 799)
EARLY_REJECT_CHANCE = 0.01
EARLY_REJECT_COST = 0.0003

Strata = list[tuple[tuple[int, ...], int]]


def _exact_law(strata: Strata) -> dict[tuple[int, ...], float]:
    """Each total's chance, by enumerating every stratum's arrangements and convolving them."""
    law: dict[tuple[int, ...], float] = {(0,) * len(strata[0][0]): 1.0}
    for sizes, marked in strata:
        whole = math.comb(sum(sizes), marked)
        step = {
            held: math.prod(math.comb(size, each) for size, each in zip(sizes, held, strict=True)) / whole
            for held in itertools.product(*(range(size + 1) for size in sizes))
            if sum(held) == marked
        }
        convolved: dict[tuple[int, ...], float] = {}
        for before, chance in law.items():
            for held, more in step.items():
                total = tuple(a + b for a, b in zip(before, held, strict=True))
                convolved[total] = convolved.get(total, 0.0) + chance * more
        law = convolved
    return law


def _fits(strata: Strata, draws: int, seed: int) -> tuple[float, float, set[tuple[int, ...]]]:
    """The chi-square of ``draws`` draws against the exact law (values expected under five times pooled), its 0.999
    critical value (Wilson-Hilferty), and any total drawn that the law cannot give."""
    law = _exact_law(strata)
    stream = stratified_hypergeometric_totals(strata, random.Random(seed))
    seen = Counter(next(stream) for _ in range(draws))
    statistic, cells, pooled_seen, pooled_expected = 0.0, 0, 0, 0.0
    for total, chance in law.items():
        expected = chance * draws
        if expected < 5:
            pooled_seen += seen[total]
            pooled_expected += expected
            continue
        statistic += (seen[total] - expected) ** 2 / expected
        cells += 1
    if pooled_expected >= 5:
        statistic += (pooled_seen - pooled_expected) ** 2 / pooled_expected
        cells += 1
    df = cells - 1
    z = NormalDist().inv_cdf(0.999)
    critical = df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3
    return statistic, critical, set(seen) - set(law)


@pytest.mark.parametrize(
    "strata",
    [
        pytest.param([((2, 2, 2), 2), ((2, 2, 2), 1), ((1, 3, 0), 2)], id="few-strata-own-margins"),
        pytest.param([((1, 1, 1), 1)] * 60 + [((1, 1, 1), 2)] * 30, id="many-strata-shared-margins"),
        pytest.param([((30, 30, 25), 20)], id="one-stratum-too-large-to-tabulate"),
        pytest.param(
            [((1, 1, 1), 1)] * 40 + [((3, 2, 1), 3), ((0, 4, 4), 4), ((5, 5, 5), 15), ((2, 0, 0), 1)],
            id="mixed-with-fixed-strata",
        ),
    ],
)
def test_the_totals_follow_the_exact_conditional_law(strata: Strata) -> None:
    statistic, critical, impossible = _fits(strata, 40_000, seed=11)
    assert not impossible, f"totals the law cannot give: {sorted(impossible)[:5]}"
    assert statistic < critical, f"chi-square {statistic:.1f} over its 0.999 critical value {critical:.1f}"


def test_every_draw_keeps_each_stratum_s_margins() -> None:
    strata: Strata = [((3, 0, 2), 2), ((1, 1, 1), 3), ((4, 4, 4), 0)] * 5 + [((200, 150, 100), 90)]
    stream = stratified_hypergeometric_totals(strata, random.Random(5))
    for _ in range(200):
        totals = next(stream)
        assert sum(totals) == sum(marked for _, marked in strata)
        assert totals[1] >= 5  # the second group's member of each of the five all-marked strata
        assert all(0 <= total <= sum(sizes[group] for sizes, _ in strata) for group, total in enumerate(totals))


def test_one_seed_reads_one_stream_whatever_the_strata_s_order() -> None:
    strata: Strata = [((2, 3), 2), ((1, 1), 1), ((5, 5), 4), ((1, 1), 1)]
    first = stratified_hypergeometric_totals(strata, random.Random(9))
    again = stratified_hypergeometric_totals(list(reversed(strata)), random.Random(9))
    assert [next(first) for _ in range(50)] == [next(again) for _ in range(50)]


@pytest.mark.parametrize(
    "strata",
    [[((1, 2), 1), ((1, 2, 3), 1)], [((2, 2), 5)], [((2, -1), 0)]],
    ids=["different-groups", "more-marked-than-members", "negative-size"],
)
def test_strata_the_law_cannot_read_are_refused(strata: Strata) -> None:
    with pytest.raises(ValueError, match="cannot mark"):
        stratified_hypergeometric_totals(strata, random.Random(1))


class _Stream:
    """Says True for each draw in ``extreme`` (counted from one), and counts the draws asked for."""

    def __init__(self, extreme: Callable[[int], bool]) -> None:
        self.extreme = extreme
        self.asked = 0

    def __call__(self) -> bool:
        self.asked += 1
        return self.extreme(self.asked)


def _read(extreme: Callable[[int], bool], draws: int = 999, level: float = 0.05) -> tuple[float, int]:
    stream = _Stream(extreme)
    return sequential_permutation_p(stream, draws=draws, level=level), stream.asked


def test_a_read_stops_once_no_later_draw_can_bring_its_p_under_the_level() -> None:
    p, asked = _read(lambda _: True)
    assert (p, asked) == (0.05, 49)
    p, asked = _read(lambda draw: draw % 3 == 0)
    assert (p, asked) == (0.05, 147)


def test_a_read_that_never_stops_is_the_full_draws_p() -> None:
    # Twenty-six draws at least as extreme, one at every early look but the last: past every early look's most, short
    # of what settles the read.
    p, asked = _read(lambda draw: draw <= 26)
    assert (p, asked) == (27 / 1000, 999)
    # Every early look is past fifty draws, so a read of fifty with none as extreme reads them all.
    p, asked = _read(lambda _: False, draws=50)
    assert (p, asked) == (1 / 51, 50)


def _rule_by_stream(draws: int = 999) -> tuple[int, dict[int, int]]:
    """The read's rule as its streams show it: the count that settles a read as no rejection, and the most count at
    each early look that stops it and rejects. A stream whose first ``k`` draws are as extreme stops at the first
    look whose most is at least ``k``."""
    settles = _read(lambda _: True, draws=draws)[1]
    stops: dict[int, int] = {}
    for front in range(settles):
        p, asked = _read(lambda draw, front=front: draw <= front, draws=draws)
        if asked < draws:
            assert p == (1 + front) / (1 + asked) < 0.05
            stops[asked] = front
    return settles, stops


def test_each_early_look_stops_on_a_count_a_true_p_at_the_level_gives_at_most_one_time_in_a_hundred() -> None:
    _, stops = _rule_by_stream()
    assert set(stops) == set(EARLY_LOOKS)
    for look, most in stops.items():
        assert _binomial_cdf(look, most, 0.05) <= EARLY_REJECT_CHANCE < _binomial_cdf(look, most + 1, 0.05)
    assert stops == {99: 0, 199: 3, 399: 10, 799: 25}


def _binomial_cdf(trials: int, at_most: int, chance: float) -> float:
    if at_most < 0:
        return 0.0
    return math.fsum(math.comb(trials, k) * chance**k * (1 - chance) ** (trials - k) for k in range(at_most + 1))


def _early_reject_cost(draws: int, settles: int, stops: dict[int, int], cells: int = 1000, top: float = 0.25) -> float:
    """An upper bound on the chance, over every law of a valid test's true p, that the read rejects where all its draws
    would not.

    Given the data, each draw is as extreme with the true p ``u``, so the count by draw ``l`` is binomial. The read
    rejects where the full draws would not only when it stops at some look ``l`` (count at most ``c``) and the full
    draws reach ``settles``; that has chance at most ``P(Bin(l, u) <= c) * P(Bin(draws - l, u) >= settles - c)``,
    summed over the looks. Over a cell of ``u`` the first factor is largest at the cell's low end and the second at
    its high end, so each cell's bound is rigorous, and past ``top`` the first factors alone bound it. A valid test's
    true p is never more likely than uniform to sit low, so the chance is at most the integral of the bound's
    decreasing envelope over a uniform ``u``."""
    width = top / cells
    bounds = [
        math.fsum(
            _binomial_cdf(look, most, cell * width)
            * (1 - _binomial_cdf(draws - look, settles - most - 1, (cell + 1) * width))
            for look, most in stops.items()
        )
        for cell in range(cells)
    ]
    beyond = math.fsum(_binomial_cdf(look, most, top) for look, most in stops.items())
    envelope, cost = beyond, beyond * (1 - top)
    for bound in reversed(bounds):
        envelope = max(envelope, bound)
        cost += envelope * width
    return cost


def test_the_early_looks_add_at_most_their_stated_cost_to_the_false_reject_rate() -> None:
    settles, stops = _rule_by_stream()
    cost = _early_reject_cost(999, settles, stops)
    assert cost <= EARLY_REJECT_COST, f"the early looks add up to {cost:.2e}, over the stated {EARLY_REJECT_COST}"


def test_where_it_does_not_stop_early_to_reject_the_read_decides_as_the_full_draws() -> None:
    for seed, chance in itertools.product(range(40), (0.02, 0.04, 0.05, 0.06, 0.1, 0.5)):
        stream = random.Random(seed)
        extreme = [stream.random() < chance for _ in range(999)]
        p, asked = _read(lambda draw, extreme=extreme: extreme[draw - 1])
        full = (1 + sum(extreme)) / 1000
        stopped_to_reject = asked < 999 and p < 0.05
        if not stopped_to_reject:
            assert (p < 0.05) == (full < 0.05), (seed, chance, p, full)
        else:
            assert p == (1 + sum(extreme[:asked])) / (1 + asked)


@pytest.mark.parametrize(("draws", "level"), [(0, 0.05), (999, 0.0), (999, 1.5)])
def test_a_read_with_no_draws_or_no_level_is_refused(draws: int, level: float) -> None:
    with pytest.raises(ValueError, match="at least one draw"):
        sequential_permutation_p(lambda: False, draws=draws, level=level)
