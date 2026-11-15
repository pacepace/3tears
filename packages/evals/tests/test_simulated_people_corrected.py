"""The ``corrected`` check of a judged contrast, checked against a known truth with fixed seeds (judges §5.3).

Each replicate draws a population of ``G`` cases, ``k`` judged answers per arm per case, a reader who sees the truth of
one drawn answer per arm, and a random sample of ``n`` labelled cases. Every row reads its figures through
:func:`~threetears.evals.analysis.people_corrected.corrected_reading`, the function the bundle's filler calls, which
calls :func:`~threetears.evals.analysis.people_corrected.prediction_powered_difference`. 4,000 replicates a row; the bars
were fixed before the run.

| Row | Judge | True people difference | Bar | Held |
|---|---|---|---|---|
| Level | Unbiased and noisy; biased equally in both arms; leaning to the candidate by about 0.15 | 0 | The corrected interval excludes 0 at most 6.5% | Every row of the grid |
| Coverage | Passes everything (a 5 on everything on 1 to 5) | 0.2 | Covers at least 93.5% | Every row of the grid |
| A rare one-sided lean | Passes every failed candidate answer on 3%, 6%, 10% of cases near the ceiling; a 5 on 5% of candidate answers on 1 to 5; G = 100, k = 3 | 0 | Covers at least 93.5% from the decision floor | n = 60 to 100; at n = 20 coverage falls to about 92% (recorded) |
| Confirmation | Every row above and the rare lean: the judge's own test reads improved and ``corrected`` is ``met`` | 0 | At most 3.5% at the decision floor | At n = 60. At n = 40 the 1-to-5 rare lean confirms about 4.6%, so 40 did not pass |
| Identical outputs on 20% of G | Unbiased | 0 and 0.2 | Covers at least 93.5% with the stratum rule | Both, both scales |
| Content-following skips | Leaning; the reader skips items where it and the judge disagree | 0 | 5%: coverage recorded; 20%: ``not_available``, ``skipped_past_limit`` | Every replicate at 20% |
| Lean test over five dimensions | Unbiased | no lean | Chance of any spurious lean recorded, at most ``1 - 0.95^5`` plus four standard errors | about 23% |
| Labels drawn by a selection rule | The 10% rare lean; the reader labels the cases where it and the judge agree | 0 | Fails: coverage far below 93.5% (recorded) | about 22% |
| A wrong variance | Unbiased; the rectifier's spread divided by G rather than n | 0 | Fails the coverage bar the shipped variance meets | about 75% against 95% |

**The decision floor** (:data:`~threetears.evals.analysis.people_corrected.PPI_DECISION_MIN_LABELLED_CASES`) is the
smallest number of labelled cases at which every row met its bar: 60. At 40 the rare lean on 1 to 5 (a 5 on one
candidate answer in twenty) confirmed a difference people do not see more often than the bar allows; at 20 the rare lean
on pass/fail broke coverage, since a sample that met no leaning case corrects nothing.
"""

from __future__ import annotations

import math
import random
from typing import Literal

import pytest

from threetears.evals.analysis.people_corrected import (
    PPI_DECISION_MIN_LABELLED_CASES,
    PairLabels,
    corrected_reading,
    prediction_powered_difference,
)
from threetears.evals.analysis.stats import t_critical_two_sided
from packages.evals.tests.simulation_support import at_most

REPLICATES = 4_000
LEVEL = 0.95
EXCLUDES_AT_MOST = 0.065
COVERS_AT_LEAST = 0.935
CONFIRMS_AT_MOST = 0.035

Scale = Literal["pass_fail", "ordinal"]
Judge = Literal["unbiased", "equal_bias", "leans", "passes_all", "rare"]

#: ``(G, n)``: populations of 20 to 150 cases, and n = 20, 40, 60, 80, 100 = G and 120.
GRID = [(20, 20), (60, 40), (60, 60), (100, 20), (100, 40), (100, 60), (100, 80), (100, 100), (150, 120)]


def _clip(score: int) -> int:
    return min(5, max(1, score))


class _Case:
    """One case's judge scores per arm, and the reader's and the judge's scores of one drawn answer per arm."""

    __slots__ = ("judge_control", "judge_candidate", "reader", "drawn")

    def __init__(self, judge_control: list[int], judge_candidate: list[int], reader: int, drawn: int) -> None:
        self.judge_control = judge_control
        self.judge_candidate = judge_candidate
        self.reader = reader
        self.drawn = drawn


def _case(rng: random.Random, scale: Scale, judge: Judge, k: int, d: float, rare: float, identical: bool) -> _Case:
    """One case. The reader sees the truth; the candidate is better by ``d`` in expectation; an identical case answers
    both arms with the same outputs, which the judge reads with its own noise each time."""
    if scale == "pass_fail":
        if judge == "rare" and rng.random() < rare:
            # A hard case both arms fail, whose failed candidate answers the judge passes: a lean in a few cases.
            return _Case([0] * k, [1] * k, 0, 1)
        p = rng.uniform(0.85, 0.97) if judge == "rare" else rng.uniform(0.2, 0.7)
        control = [int(rng.random() < p) for _ in range(k)]
        candidate = control if identical else [int(rng.random() < p + d) for _ in range(k)]

        def read(answer: int, is_candidate: bool) -> int:
            if judge in ("rare",):
                return answer
            if judge == "passes_all":
                return 1
            if judge == "equal_bias":
                return int(rng.random() >= 0.05) if answer else int(rng.random() < 0.3)
            seen = answer if rng.random() >= 0.1 else 1 - answer
            if judge == "leans" and is_candidate and not answer and rng.random() < 0.27:
                seen = 1
            return seen

    else:
        base = rng.choice((3, 4)) if judge == "rare" else rng.choice((2, 3))

        def answer(up: bool) -> int:
            noise = rng.random()
            score = base + (-1 if noise < 0.25 else 1 if noise >= 0.75 else 0)
            return _clip(score + (1 if up and rng.random() < d else 0))

        control = [answer(False) for _ in range(k)]
        candidate = control if identical else [answer(True) for _ in range(k)]

        def read(answer: int, is_candidate: bool) -> int:
            if judge == "passes_all":
                return 5
            if judge == "rare":
                return 5 if is_candidate and rng.random() < rare else answer
            noise = rng.random()
            seen = answer + (-1 if noise < 0.15 else 1 if noise >= 0.85 else 0)
            if judge == "equal_bias" and rng.random() < 0.4:
                seen += 1
            if judge == "leans" and is_candidate and rng.random() < 0.15:
                seen += 1
            return _clip(seen)

    judged_control = [read(one, False) for one in control]
    judged_candidate = [read(one, True) for one in candidate]
    return _Case(
        judged_control,
        judged_candidate,
        candidate[0] - control[0],
        judged_candidate[0] - judged_control[0],
    )


class _Replicate:
    __slots__ = ("differences", "cases", "identical", "labels")

    def __init__(
        self,
        differences: dict[str, float],
        cases: dict[str, _Case],
        identical: frozenset[str],
        labels: PairLabels,
    ) -> None:
        self.differences = differences
        self.cases = cases
        self.identical = identical
        self.labels = labels


def _replicate(
    rng: random.Random,
    *,
    scale: Scale,
    judge: Judge,
    population: int,
    k: int,
    n: int,
    d: float,
    rare: float = 0.0,
    identical_share: float = 0.0,
    skip: int = 0,
    selected: bool = False,
) -> _Replicate:
    """One population, its judge's case differences, and one reader's labels of ``n`` of its non-identical cases."""
    same = frozenset(f"c{i}" for i in range(round(identical_share * population)))
    cases = {f"c{i}": _case(rng, scale, judge, k, d, rare, f"c{i}" in same) for i in range(population)}
    differences = {name: (sum(one.judge_candidate) - sum(one.judge_control)) / k for name, one in cases.items()}
    rest = [name for name in cases if name not in same]
    if selected:
        # The reader labels the cases where it and the judge agree first: a rule that follows content.
        chosen = sorted(rest, key=lambda name: (cases[name].reader != cases[name].drawn, rng.random()))[:n]
    else:
        chosen = rng.sample(rest, n)
    skipped = 0
    pairs: dict[str, tuple[float, float]] = {}
    for name in chosen:
        one = cases[name]
        if skipped < skip and one.reader != one.drawn:
            skipped += 1
            continue
        pairs[name] = (float(one.reader), float(one.drawn))
    labels = PairLabels(
        against="people",
        against_id=None,
        read_id="sim",
        sample="random",
        answered=n,
        skipped=skipped,
        population=frozenset(cases),
        identical=same,
        pairs=pairs,
    )
    return _Replicate(differences, cases, same, labels)


def _judge_reads_improved(differences: dict[str, float]) -> bool:
    """The judge's own paired test over every case, at the nominal level: what the contrast's verdict reads."""
    values = list(differences.values())
    mean = math.fsum(values) / len(values)
    spread = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))
    return mean - t_critical_two_sided(LEVEL, len(values) - 1) * spread / math.sqrt(len(values)) > 0


class _Rates:
    """What a row measured over its replicates."""

    def __init__(self) -> None:
        self.n = 0
        self.covered = 0
        self.excluded = 0
        self.confirmed = 0
        self.judge_improved = 0

    def rate(self, count: int) -> float:
        return count / self.n


def _row(seed: str, truth: float, **kwargs: object) -> _Rates:
    """Run one row through the shipped check."""
    rng = random.Random(seed)
    rates = _Rates()
    for _ in range(REPLICATES):
        drawn = _replicate(rng, **kwargs)  # type: ignore[arg-type]
        corrected = corrected_reading([drawn.labels], drawn.differences, direction=1, confidence=LEVEL)
        assert corrected is not None and corrected.interval is not None, corrected
        low, high = corrected.interval
        improved = _judge_reads_improved(drawn.differences)
        rates.n += 1
        rates.covered += low <= truth <= high
        rates.excluded += low > 0 or high < 0
        rates.judge_improved += improved
        rates.confirmed += improved and corrected.result == "met"
    return rates


# --- the level, coverage and confirmation over the grid -------------------------------------------------------------


@pytest.mark.parametrize("judge", ["unbiased", "equal_bias", "leans"])
@pytest.mark.parametrize("scale", ["pass_fail", "ordinal"])
@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize(("population", "n"), GRID)
def test_with_no_true_difference_the_corrected_interval_excludes_zero_at_most_six_and_a_half_percent(
    judge: Judge, scale: Scale, k: int, population: int, n: int
) -> None:
    rates = _row(
        f"null|{judge}|{scale}|{k}|{population}|{n}",
        0.0,
        scale=scale,
        judge=judge,
        population=population,
        k=k,
        n=n,
        d=0.0,
    )
    assert rates.rate(rates.excluded) <= EXCLUDES_AT_MOST, rates.rate(rates.excluded)
    if n == PPI_DECISION_MIN_LABELLED_CASES:
        assert rates.rate(rates.confirmed) <= CONFIRMS_AT_MOST, rates.rate(rates.confirmed)


def test_a_leaning_judge_reads_improved_where_people_see_nothing_and_the_check_does_not_confirm_it() -> None:
    """The judge-only rate is recorded: the lean is what the check exists for."""
    rates = _row("lean-recorded", 0.0, scale="pass_fail", judge="leans", population=100, k=3, n=60, d=0.0)
    assert rates.rate(rates.judge_improved) > 0.5, rates.rate(rates.judge_improved)
    assert rates.rate(rates.confirmed) <= CONFIRMS_AT_MOST


@pytest.mark.parametrize("scale", ["pass_fail", "ordinal"])
@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize(("population", "n"), GRID)
def test_a_judge_that_passes_everything_is_corrected_to_people_s_difference(
    scale: Scale, k: int, population: int, n: int
) -> None:
    rates = _row(
        f"all|{scale}|{k}|{population}|{n}",
        0.2,
        scale=scale,
        judge="passes_all",
        population=population,
        k=k,
        n=n,
        d=0.2,
    )
    assert rates.rate(rates.covered) >= COVERS_AT_LEAST, rates.rate(rates.covered)


# --- a rare one-sided lean, and the decision floor ------------------------------------------------------------------

_RARE = [("pass_fail", 0.03), ("pass_fail", 0.06), ("pass_fail", 0.10), ("ordinal", 0.05)]


@pytest.mark.parametrize(("scale", "rare"), _RARE)
@pytest.mark.parametrize("n", [60, 80, 100])
def test_a_rare_one_sided_lean_is_covered_and_never_confirmed_from_the_decision_floor(
    scale: Scale, rare: float, n: int
) -> None:
    rates = _row(f"rare|{scale}|{rare}|{n}", 0.0, scale=scale, judge="rare", population=100, k=3, n=n, d=0.0, rare=rare)
    assert rates.rate(rates.covered) >= COVERS_AT_LEAST, rates.rate(rates.covered)
    if n == PPI_DECISION_MIN_LABELLED_CASES:
        assert rates.rate(rates.confirmed) <= CONFIRMS_AT_MOST, rates.rate(rates.confirmed)


def test_the_decision_floor_is_the_smallest_n_that_passed_and_at_least_forty() -> None:
    """At 20 the rare lean on pass/fail breaks coverage; at 40 the rare lean on 1 to 5 confirms too often; from 60
    every row above holds. So the floor is 60."""
    at_twenty = _row("floor|20", 0.0, scale="pass_fail", judge="rare", population=100, k=3, n=20, d=0.0, rare=0.10)
    assert at_twenty.rate(at_twenty.covered) < COVERS_AT_LEAST, at_twenty.rate(at_twenty.covered)
    at_forty = _row("floor|40", 0.0, scale="ordinal", judge="rare", population=100, k=3, n=40, d=0.0, rare=0.05)
    assert at_forty.rate(at_forty.confirmed) > CONFIRMS_AT_MOST, at_forty.rate(at_forty.confirmed)
    assert PPI_DECISION_MIN_LABELLED_CASES == max(40, 60)


# --- identical outputs, skips, five dimensions, a selection rule and a wrong variance ------------------------------


@pytest.mark.parametrize("scale", ["pass_fail", "ordinal"])
@pytest.mark.parametrize(("other", "truth"), [(0.0, 0.0), (0.25, 0.2)])
def test_identical_outputs_are_a_stratum_whose_people_difference_is_zero(
    scale: Scale, other: float, truth: float
) -> None:
    rates = _row(
        f"identical|{scale}|{other}",
        truth,
        scale=scale,
        judge="unbiased",
        population=100,
        k=3,
        n=PPI_DECISION_MIN_LABELLED_CASES,
        d=other,
        identical_share=0.2,
    )
    assert rates.rate(rates.covered) >= COVERS_AT_LEAST, rates.rate(rates.covered)


def test_content_following_skips_are_recorded_at_five_percent_and_stop_the_check_at_twenty() -> None:
    rng = random.Random("skips")
    n = 60
    covered = 0
    for _ in range(REPLICATES):
        drawn = _replicate(rng, scale="pass_fail", judge="leans", population=100, k=3, n=n, d=0.0, skip=3)
        corrected = corrected_reading([drawn.labels], drawn.differences, direction=1, confidence=LEVEL)
        assert corrected is not None and corrected.state is None and corrected.interval is not None
        covered += corrected.interval[0] <= 0.0 <= corrected.interval[1]
    # Recorded: removing the items the judge got wrong hides some of its lean.
    assert covered / REPLICATES >= 0.85, covered / REPLICATES
    for _ in range(200):
        drawn = _replicate(rng, scale="pass_fail", judge="leans", population=100, k=3, n=n, d=0.0, skip=12)
        corrected = corrected_reading([drawn.labels], drawn.differences, direction=1, confidence=LEVEL)
        assert corrected is not None and corrected.state is not None
        assert (corrected.state.state, corrected.state.reason, corrected.result) == (
            "not_available",
            "skipped_past_limit",
            None,
        )


def test_the_lean_test_over_five_dimensions_holds_each_test_s_level() -> None:
    rng = random.Random("five-dimensions")
    any_lean = 0
    for _ in range(REPLICATES):
        leaned = False
        for _dimension in range(5):
            drawn = _replicate(rng, scale="pass_fail", judge="unbiased", population=100, k=3, n=60, d=0.0)
            corrected = corrected_reading([drawn.labels], drawn.differences, direction=1, confidence=LEVEL)
            assert corrected is not None and corrected.lean_interval is not None
            low, high = corrected.lean_interval
            leaned |= low > 0 or high < 0
        any_lean += leaned
    family = 1 - LEVEL**5
    assert any_lean / REPLICATES <= at_most(family, REPLICATES), any_lean / REPLICATES


def test_labels_chosen_where_the_reader_and_the_judge_agree_miss_the_lean() -> None:
    """Why the sample must be random: the same estimator on selected labels covers a fraction of the time."""
    rng = random.Random("selected")
    covered = 0
    for _ in range(REPLICATES):
        drawn = _replicate(
            rng, scale="pass_fail", judge="rare", population=100, k=3, n=60, d=0.0, rare=0.10, selected=True
        )
        figures = prediction_powered_difference(
            drawn.differences, drawn.labels.pairs, identical=drawn.identical, confidence=LEVEL
        )
        assert figures is not None and figures.interval is not None
        covered += figures.interval[0] <= 0.0 <= figures.interval[1]
    assert covered / REPLICATES < COVERS_AT_LEAST - 0.5, covered / REPLICATES


def test_a_wrong_variance_fails_the_bar_the_shipped_one_meets() -> None:
    """The rectifier's spread divided by every case rather than the labelled ones: a common slip, and visible here."""
    rng = random.Random("wrong-variance")
    right = wrong = 0
    population, n = 100, 20
    critical = t_critical_two_sided(LEVEL, n - 1)
    for _ in range(REPLICATES):
        drawn = _replicate(rng, scale="pass_fail", judge="unbiased", population=population, k=3, n=n, d=0.0)
        corrected = corrected_reading([drawn.labels], drawn.differences, direction=1, confidence=LEVEL)
        assert corrected is not None and corrected.interval is not None and corrected.estimate is not None
        right += corrected.interval[0] <= 0.0 <= corrected.interval[1]
        judged = list(drawn.differences.values())
        judged_mean = math.fsum(judged) / population
        gaps = [reader - judge for reader, judge in drawn.labels.pairs.values()]
        gap_mean = math.fsum(gaps) / n
        judged_spread = math.fsum((value - judged_mean) ** 2 for value in judged) / (population - 1)
        gap_spread = math.fsum((gap - gap_mean) ** 2 for gap in gaps) / (n - 1)
        half = critical * math.sqrt(judged_spread / population + gap_spread / population)
        wrong += corrected.estimate - half <= 0.0 <= corrected.estimate + half
    assert right / REPLICATES >= COVERS_AT_LEAST, right / REPLICATES
    assert wrong / REPLICATES < COVERS_AT_LEAST, wrong / REPLICATES
