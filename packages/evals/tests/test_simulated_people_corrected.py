"""The ``corrected`` check of a judged contrast, checked against a known truth with fixed seeds (judges §5.3).

Each replicate draws a population of ``G`` cases, ``k`` judged answers per arm per case, a reader who sees the truth of
one drawn answer per arm, and a random sample of ``n`` labelled cases. Every row reads its figures through
:func:`~threetears.evals.analysis.people_corrected.corrected_reading`, the function the bundle's filler calls, which
calls :func:`~threetears.evals.analysis.people_corrected.prediction_powered_difference`. 4,000 replicates a row, and
16,000 for the rows that decide the floor; the bars were fixed before the run.

| Row | Judge | True people difference | Bar | Held |
|---|---|---|---|---|
| Level | Unbiased and noisy; biased equally in both arms; leaning to the candidate by about 0.15 | 0 | The corrected interval excludes 0 at most 6.5%; unbiased from n = 40, covers at most 97.5% | Every row of the grid |
| Coverage | Passes everything (a 5 on everything on 1 to 5) | 0.2 | Covers at least 93.5% | Every row of the grid |
| A rare one-sided lean (the deciding rows, 16,000 replicates) | Passes every failed candidate answer on 3%, 6%, 10% of cases near the ceiling; a 5 on 5% of candidate answers on 1 to 5; G = 100, k = 3 | 0 | Covers at least 93.5% and confirms at most 3.5%, each with two Monte Carlo standard errors to spare | n = 80 and 100, every row; at 60 the 1-to-5 row confirms 3.61% (table below) |
| Confirmation | Every row above: the judge's own test reads improved and the corrected interval lies above zero (``met`` from the floor) | 0 | At most 3.5% at the decision floor | Every row at n = 80 |
| Below the deciding rows | The rare lean | 0 | Fails a bar | n = 20: the 10% lean on pass/fail covers under 93.5%; n = 40: the 1-to-5 lean confirms over 3.5% |
| Identical outputs on 20% of G | Unbiased | 0 and 0.2 | Covers at least 93.5% with the stratum rule | Both, both scales |
| Outputs that vary across repeats, a fifth of cases identical on every repeat; G = 200, n = 60 | Unbiased | 0 and a shift toward each case's best output | Covers at least 93.5% when a coinciding draw is an ordinary pair; the rule that read it as the zero stratum fails | Both scales |
| Content-following skips | Leaning; the reader skips items where it and the judge disagree | 0 | 5%: coverage recorded; 20%: ``not_available``, ``skipped_past_limit`` | Every replicate at 20% |
| Lean test over five dimensions | Unbiased | no lean | Chance of any spurious lean recorded, at most ``1 - 0.95^5`` plus four standard errors | about 23% |
| Labels drawn by a selection rule | The 10% rare lean; the reader labels the cases where it and the judge agree | 0 | Fails: coverage far below 93.5% (recorded) | about 22% |
| A wrong variance, too narrow | Unbiased; the rectifier's spread divided by G rather than n | 0 | Fails the coverage bar the shipped variance meets | about 75% against 95% |
| A wrong variance, too wide | Unbiased on 1 to 5, G = 100, n = 80; the judge's difference read as two independent arms (their covariance dropped) | 0 | Covers more than 97.5%, where the shipped one holds both bars | Both |

**The decision floor** (:data:`~threetears.evals.analysis.people_corrected.PPI_DECISION_MIN_LABELLED_CASES`) is the
smallest n at which every deciding row, and every row at a larger n, covers at least 93.5% and confirms at most 3.5%
with two Monte Carlo standard errors to spare, read off :data:`DECIDING` (16,000 replicates a row):

| Row | n | Covers % (SE) | Confirms % | SE | Confirms + 2 SE | Bars |
|---|---|---|---|---|---|---|
| 1 to 5, 5% | 60 | 94.89 (0.17) | 3.61 | 0.15 | 3.90 | fails |
| pass/fail, 3%, 6%, 10% | 60 | 95.23 to 95.36 | 1.93 to 2.13 | 0.11 | 2.15 to 2.36 | hold |
| 1 to 5, 5% | 80 | 95.28 (0.17) | 2.93 | 0.13 | 3.20 | hold |
| pass/fail, 3%, 6%, 10% | 80 | 95.03 to 95.33 | 1.90 to 2.25 | 0.11 to 0.12 | 2.12 to 2.48 | hold |
| 1 to 5, 5% | 100 | 94.75 (0.18) | 2.89 | 0.13 | 3.15 | hold |
| pass/fail, 3%, 6%, 10% | 100 | 94.99 to 95.58 | 2.23 to 2.59 | 0.12 to 0.13 | 2.46 to 2.84 | hold |

So the floor is 80. Below it a rare lean on 1 to 5 (a 5 on one candidate answer in twenty) is confirmed against people
more often than the bar allows. At 80 that row still confirms 2.93%, about three standard errors above the nominal
2.5%: a lean concentrated in rare answers skews the rectifier, and the interval over-confirms under it, inside the bar
only by the slack the bar allows.
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
from packages.evals.tests.simulation_support import at_most, monte_carlo_se

REPLICATES = 4_000
LEVEL = 0.95
EXCLUDES_AT_MOST = 0.065
COVERS_AT_LEAST = 0.935
#: The upper bar on a well-behaved row's coverage: an interval far wider than its level claims fails it.
COVERS_AT_MOST = 0.975
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


def _row(seed: str, truth: float, *, replicates: int = REPLICATES, **kwargs: object) -> _Rates:
    """Run one row through the shipped check."""
    rng = random.Random(seed)
    rates = _Rates()
    for _ in range(replicates):
        drawn = _replicate(rng, **kwargs)  # type: ignore[arg-type]
        corrected = corrected_reading([drawn.labels], drawn.differences, direction=1, confidence=LEVEL)
        assert corrected is not None and corrected.interval is not None, corrected
        low, high = corrected.interval
        improved = _judge_reads_improved(drawn.differences)
        rates.n += 1
        rates.covered += low <= truth <= high
        rates.excluded += low > 0 or high < 0
        rates.judge_improved += improved
        # Confirmed: the judge reads improved and the corrected interval lies above zero, which is what ``met`` reads
        # from the decision floor; counted below it too, so a floor can be read off the rates.
        confirms = low > 0
        if corrected.labelled_cases is not None and corrected.labelled_cases >= PPI_DECISION_MIN_LABELLED_CASES:
            assert confirms == (corrected.result == "met"), corrected
        rates.confirmed += improved and confirms
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
    if judge == "unbiased" and n >= 40:
        # A well-behaved row: its interval is no wider than its level claims, so an over-wide variance fails here.
        assert 1 - rates.rate(rates.excluded) <= COVERS_AT_MOST, 1 - rates.rate(rates.excluded)
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

#: Replicates behind each deciding row: a Monte Carlo standard error of about 0.15 points at the confirmation bar.
DECIDING_REPLICATES = 16_000

#: The deciding rows, as the seeded simulation below measured them at :data:`DECIDING_REPLICATES` each:
#: ``(scale, rare share, n) -> (replicates covering the truth, replicates confirming)``. Every row re-runs on the pool
#: and must reproduce its counts exactly (``random.Random`` with a fixed string seed is the same on every platform), so
#: the floor read off this table is the simulation's, never a number typed beside it.
DECIDING: dict[tuple[Scale, float, int], tuple[int, int]] = {
    ("pass_fail", 0.03, 60): (15258, 341),
    ("pass_fail", 0.06, 60): (15236, 309),
    ("pass_fail", 0.10, 60): (15248, 328),
    ("ordinal", 0.05, 60): (15182, 577),
    ("pass_fail", 0.03, 80): (15217, 360),
    ("pass_fail", 0.06, 80): (15252, 304),
    ("pass_fail", 0.10, 80): (15204, 308),
    ("ordinal", 0.05, 80): (15244, 469),
    ("pass_fail", 0.03, 100): (15293, 356),
    ("pass_fail", 0.06, 100): (15199, 400),
    ("pass_fail", 0.10, 100): (15218, 414),
    ("ordinal", 0.05, 100): (15160, 462),
}


def _passes(covered: int, confirmed: int, replicates: int = DECIDING_REPLICATES) -> bool:
    """Whether a deciding row meets both bars with two Monte Carlo standard errors to spare."""
    cover, confirm = covered / replicates, confirmed / replicates
    return cover - 2 * monte_carlo_se(cover, replicates) >= COVERS_AT_LEAST and (
        confirm + 2 * monte_carlo_se(confirm, replicates) <= CONFIRMS_AT_MOST
    )


def _deciding_row(scale: Scale, share: float, n: int) -> _Rates:
    return _row(
        f"decide|{scale}|{share}|{n}",
        0.0,
        replicates=DECIDING_REPLICATES,
        scale=scale,
        judge="rare",
        population=100,
        k=3,
        n=n,
        d=0.0,
        rare=share,
    )


@pytest.mark.parametrize(("scale", "rare"), _RARE)
@pytest.mark.parametrize("n", [60, 80, 100])
def test_each_deciding_row_reproduces_its_counts(scale: Scale, rare: float, n: int) -> None:
    rates = _deciding_row(scale, rare, n)
    assert DECIDING.get((scale, rare, n)) == (rates.covered, rates.confirmed), (
        scale,
        rare,
        n,
        rates.covered,
        rates.confirmed,
    )


def test_the_decision_floor_is_the_smallest_n_whose_rows_all_pass_with_two_standard_errors_to_spare() -> None:
    """The floor is read off :data:`DECIDING`: the smallest n at which every rare-lean row, and every row at a larger n,
    covers at least 93.5% and confirms at most 3.5% with two Monte Carlo standard errors to spare. One constant, not a
    rule computed from the data: the failure it guards (a rare lean the sample never met) cannot be seen in a sample."""
    sizes = sorted({n for _, _, n in DECIDING})
    passing = [
        n
        for n in sizes
        if all(_passes(*DECIDING[(scale, share, size)]) for scale, share in _RARE for size in sizes if size >= n)
    ]
    assert passing, "no deciding n met both bars"
    assert PPI_DECISION_MIN_LABELLED_CASES == max(40, passing[0])


def test_below_the_deciding_rows_the_rare_lean_breaks_a_bar() -> None:
    """At 20 the lean on pass/fail breaks coverage, and at 40 the lean on 1 to 5 is confirmed too often: neither is a
    floor, whatever the deciding rows read."""
    at_twenty = _row(
        "rare|pass_fail|0.1|20", 0.0, scale="pass_fail", judge="rare", population=100, k=3, n=20, d=0.0, rare=0.10
    )
    assert at_twenty.rate(at_twenty.covered) < COVERS_AT_LEAST, at_twenty.rate(at_twenty.covered)
    at_forty = _row(
        "rare|ordinal|0.05|40", 0.0, scale="ordinal", judge="rare", population=100, k=3, n=40, d=0.0, rare=0.05
    )
    assert at_forty.rate(at_forty.confirmed) > CONFIRMS_AT_MOST, at_forty.rate(at_forty.confirmed)


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


def test_dropping_the_covariance_between_arms_widens_the_interval_past_the_upper_bar() -> None:
    """The judge's difference read as two independent arms: each arm's case means spread over the population, their
    covariance left out. A case's two arms share its difficulty, so that variance is too large and the interval too
    wide; the shipped one holds both bars on the same draws."""
    rng = random.Random("arms-covariance")
    right = wrong = 0
    population, n = 100, 80
    critical = t_critical_two_sided(LEVEL, n - 1)
    for _ in range(REPLICATES):
        drawn = _replicate(rng, scale="ordinal", judge="unbiased", population=population, k=3, n=n, d=0.0)
        corrected = corrected_reading([drawn.labels], drawn.differences, direction=1, confidence=LEVEL)
        assert corrected is not None and corrected.interval is not None and corrected.estimate is not None
        low, high = corrected.interval
        right += low <= 0.0 <= high
        sem = (high - low) / 2 / critical
        control = [math.fsum(one.judge_control) / 3 for one in drawn.cases.values()]
        candidate = [math.fsum(one.judge_candidate) / 3 for one in drawn.cases.values()]
        control_mean, candidate_mean = math.fsum(control) / population, math.fsum(candidate) / population
        covariance = math.fsum(
            (a - control_mean) * (b - candidate_mean) for a, b in zip(control, candidate, strict=True)
        ) / (population - 1)
        half = critical * math.sqrt(max(0.0, sem * sem + 2 * covariance / population))
        wrong += corrected.estimate - half <= 0.0 <= corrected.estimate + half
    assert COVERS_AT_LEAST <= right / REPLICATES <= COVERS_AT_MOST, right / REPLICATES
    assert wrong / REPLICATES > COVERS_AT_MOST, wrong / REPLICATES


# --- outputs that vary across repeats, whose drawn pair sometimes coincides ---------------------------------------


def _varying_replicate(
    rng: random.Random, *, scale: Scale, population: int, k: int, n: int, shift: float, drawn_rule: bool
) -> tuple[float, dict[str, float], PairLabels]:
    """One population whose cases hold a few distinct outputs, drawn per repeat, so a case's two drawn outputs
    sometimes coincide while its outputs vary. A fifth of the cases hold one output only: identical on every repeat.

    The control draws an output uniformly; the candidate draws the case's best with chance ``shift``, else uniformly.
    The judge reads each output's truth with unbiased noise. ``drawn_rule`` labels as the rule before this one did: a
    coinciding draw is the zero stratum. Otherwise the stratum is the cases identical on every repeat, and a coinciding
    draw is a pair the reader sees no difference in, read against the judge's mean over the candidate's repeats of that
    output less the control's drawn one, as ``pair_labels`` reads it.

    :return: The population's true difference, the judge's case differences and the reader's labels.
    """
    differences: dict[str, float] = {}
    truths: list[float] = []
    drawn_same: set[str] = set()
    every_same: set[str] = set()
    pairs_all: dict[str, tuple[float, float]] = {}
    for i in range(population):
        name = f"c{i}"
        size = 1 if rng.random() < 0.2 else 3
        if scale == "pass_fail":
            p = rng.uniform(0.3, 0.8)
            outputs = [int(rng.random() < p) for _ in range(size)]
        else:
            base = rng.choice((2, 3, 4))
            outputs = [_clip(base + rng.choice((-1, 0, 0, 1))) for _ in range(size)]
        best = max(range(size), key=lambda j: outputs[j])
        truths.append(shift * (outputs[best] - math.fsum(outputs) / size))

        def judged(j: int) -> int:
            if scale == "pass_fail":
                return outputs[j] if rng.random() >= 0.1 else 1 - outputs[j]
            noise = rng.random()
            return _clip(outputs[j] + (-1 if noise < 0.15 else 1 if noise >= 0.85 else 0))

        control = [rng.randrange(size) for _ in range(k)]
        candidate = [best if rng.random() < shift else rng.randrange(size) for _ in range(k)]
        read_control = [judged(j) for j in control]
        read_candidate = [judged(j) for j in candidate]
        differences[name] = (math.fsum(read_candidate) - math.fsum(read_control)) / k
        if size == 1:
            every_same.add(name)
        if candidate[0] == control[0]:
            drawn_same.add(name)
            same = [score for j, score in zip(candidate, read_candidate, strict=True) if j == control[0]]
            pairs_all[name] = (0.0, math.fsum(same) / len(same) - read_control[0])
        else:
            pairs_all[name] = (
                float(outputs[candidate[0]] - outputs[control[0]]),
                float(read_candidate[0] - read_control[0]),
            )
    identical = frozenset(drawn_same if drawn_rule else every_same)
    chosen = rng.sample([name for name in differences if name not in identical], n)
    labels = PairLabels(
        against="people",
        against_id=None,
        read_id="sim",
        sample="random",
        answered=n,
        skipped=0,
        population=frozenset(differences),
        identical=identical,
        pairs={name: pairs_all[name] for name in chosen},
    )
    return math.fsum(truths) / population, differences, labels


def _varying_coverage(seed: str, *, scale: Scale, shift: float, drawn_rule: bool) -> float:
    rng = random.Random(seed)
    covered = 0
    for _ in range(REPLICATES):
        truth, differences, labels = _varying_replicate(
            rng, scale=scale, population=200, k=3, n=60, shift=shift, drawn_rule=drawn_rule
        )
        corrected = corrected_reading([labels], differences, direction=1, confidence=LEVEL)
        assert corrected is not None and corrected.interval is not None
        covered += corrected.interval[0] <= truth <= corrected.interval[1]
    return covered / REPLICATES


@pytest.mark.parametrize("scale", ["pass_fail", "ordinal"])
@pytest.mark.parametrize("shift", [0.0, 0.6])
def test_outputs_that_vary_across_repeats_are_covered_when_a_coinciding_draw_is_an_ordinary_pair(
    scale: Scale, shift: float
) -> None:
    coverage = _varying_coverage(f"varying|{scale}|{shift}", scale=scale, shift=shift, drawn_rule=False)
    assert coverage >= COVERS_AT_LEAST, coverage


@pytest.mark.parametrize("scale", ["pass_fail", "ordinal"])
def test_a_coinciding_draw_read_as_the_zero_stratum_misses_a_real_difference(scale: Scale) -> None:
    """The rule before: a case whose two drawn outputs coincided weighed in at zero, though its outputs vary and its
    difference does not average zero. The estimate shrinks toward zero by the share of such draws."""
    coverage = _varying_coverage(f"varying|{scale}|0.6", scale=scale, shift=0.6, drawn_rule=True)
    assert coverage < COVERS_AT_LEAST, coverage
