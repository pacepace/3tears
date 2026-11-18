"""An engine family reads a reading on 0 to 1 by the bounded test its interval is drawn by, and holds an eval of one
check once.

**The bounded test holds α on pass/fail pairs.** The engine reads a comparison once at the count it has (no committed
order lets it be read again as cases are added): on 0 to 1, by the fixed-n plug-in bet
(``stats.bounded_separation_p`` with no mixture), its stakes tuned to the level each of the family's intervals is read
at. Its stakes are set from the values before each, so under the null its capital is a nonnegative supermartingale and
Ville's inequality bounds the chance it ever reaches 1/level: α holds at every count. Here every pass/fail sequence of
up to twelve pairs is enumerated (``engine_take_up.plug_in_separating``, held sequence for sequence to the product's p
up to seven): the chance is at most α, and at most α/3 at each look of a plan of three. From 10 to 100 pairs the
statistics certificate counts it by seeded simulation, not exactly (the plug-in's stakes follow the running variance,
so its capital is path-dependent). The paired t test the engine read by before separated fair coin pairs 5.28% of the
time at 40 cases.

**Through the product**: forty cases where one arm alone passed ten and the other two, which the t test would separate
and the bounded test does not; and seven of forty to one arm alone, which the bounded test separates beside an interval
that excludes zero. The mixture bet the engine read by before left that one not separated beside the same interval. No
engine contrast reads not separated beside an interval that excludes zero, paired or unpaired.

**One hypothesis, read once.** An eval whose every trial is graded on one check passes a trial exactly where the check
holds, so its pass rate is the check case for case; the family holds the check alone. With a second check the pass
rate is no one check's, and the family holds it beside both.

Mutations that turn this file red (each run against a saved copy and restored from it):

- ``_fixed_count_test``: returning the t-test's reading for a reading on 0 to 1, so the grade separates.
- ``_fixed_count_test``: reading the paired p by the mixture bet, so seven of forty read not separated beside an
  interval that excludes zero.
- ``_corrected_family``: not tuning the bet to α/m, so the p is not the interval's bet.
- ``_unpaired_bounded_p``: reading at the midpoint cut, so an unpaired interval excludes zero beside no separation.
- ``multiple_comparisons``: not calling ``_repeats_a_reading`` on the engine, so the pass rate joins the check.
- ``_one_check``: returning the first check of several, so a two-check eval loses its pass rate.
"""

from __future__ import annotations

import itertools
import random
from collections import Counter
from dataclasses import replace
from fractions import Fraction

import pytest

from threetears.evals.analysis import AnalysisContextBundle, ComparisonFamily, assemble_context_bundle
from threetears.evals.analysis.stats import (
    SIGNIFICANCE_ALPHA,
    bet_order,
    bounded_separation_p,
    composite_significance,
)
from threetears.evals.kernel import CampaignArm
from threetears.evals.kernel.host import MeasureRegistry
from threetears.evals.kernel.metrics import MetricDescriptor
from threetears.evals.schema import EvalResult
from threetears.evals.schema.models import GoalStateOutcome
from packages.evals.tests.engine_take_up import (
    exact_small_rate,
    exact_unpaired_rate,
    plug_in_separating,
    unpaired_bounds,
)
from packages.evals.tests.factories import (
    fixture_variant_key,
    make_aspect_campaign,
    make_eval_result,
    make_eval_run,
    minimal_aspect_design,
)
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_MEASURES, toyhost_profile
from packages.evals.tests.simulation_support import engine_separation

#: A check declared pass/fail on a merit axis, and a grade on 0 to 1 that is no check: the two shapes a 0-1 reading takes.
_CHECK = MetricDescriptor(
    name="answered_in_full",
    reader_name="Answered in full",
    data_type="numeric",
    family="mechanical",
    transferability_class="mechanical",
    attribution_scope="end_to_end",
    description="Whether the reply answered every part of the question.",
    higher_is_better=True,
    merit_axis="reliability",
    value_range=(0.0, 1.0),
    pass_fail=True,
)
_GRADE = _CHECK.model_copy(
    update={"name": "graded", "reader_name": "Graded", "description": "A grade on 0 to 1.", "pass_fail": False}
)


def _pairs(plus: int, minus: int, cases: int = 40) -> tuple[list[float], list[float]]:
    """``cases`` pass/fail pairs: ``plus`` the contrast alone passed, ``minus`` the control alone, the rest tied, half of
    them both passing."""
    tied = cases - plus - minus
    both, neither = [1.0] * (tied // 2), [0.0] * (tied - tied // 2)
    return [0.0] * plus + [1.0] * minus + both + neither, [1.0] * plus + [0.0] * minus + both + neither


#: Forty cases, ten the contrast alone passed and two the control alone: the t test separates them, the bounded one not.
_CONTROL, _CONTRAST = _pairs(10, 2)

#: Forty cases, seven the contrast alone passed and none the control: the bounded test tuned to α/2 separates them, and
#: the interval it draws at 97.5% excludes zero; the mixture bet the engine read by before did not (p 0.048).
_SWEEP = _pairs(7, 0)


def _bundle(
    sides: tuple[list[float], list[float]] = (_CONTROL, _CONTRAST),
    *,
    second_check: bool = False,
    unpaired: bool = False,
) -> AnalysisContextBundle:
    """A control and a contrast, one trial a case, carrying the check and the grade on the same answers; with
    ``second_check`` each trial is graded on a goal check too, which it always passes; ``unpaired`` runs the contrast on
    cases of its own."""
    profile = toyhost_profile()
    profile = replace(
        profile, measures=MeasureRegistry((*TOYHOST_MEASURES, _CHECK, _GRADE), families=profile.measures.families)
    )
    runs = []
    results: dict[str, list[EvalResult]] = {}
    for model, values in zip(("control-model", "contrast-model"), sides, strict=True):
        run = make_eval_run(status="completed", candidate_model=model)
        runs.append(run)
        results[run.id] = [
            make_eval_result(
                id=f"{model}-{case}",
                eval_run_id=run.id,
                scope_id=run.scope_id,
                model=model,
                test_case_id=f"tc-{case + (1000 if unpaired and model == 'contrast-model' else 0):04d}",
                goal_state_outcomes=[GoalStateOutcome(expression="filed", passed=True)] if second_check else [],
                rubric_scores=[],
                cost_usd=None,
                host_measures={"answered_in_full": value, "graded": value},
            )
            for case, value in enumerate(values)
        ]
    asked = min(run.created_at for run in runs)
    campaign = make_aspect_campaign(
        label="answers",
        run_ids=[run.id for run in runs],
        design=minimal_aspect_design(merit_axes=["reliability"]),
        question="does the contrast answer in full more often?",
        asked_at=asked,
        control=CampaignArm(
            label="control", role="control", variant_key=fixture_variant_key("control-model"), named_at=asked
        ),
        scope_id=runs[0].scope_id,
        name="fixed count",
        subject_id=runs[0].subject_snapshot.subject_id,
        subject_kind="s",
    )
    return assemble_context_bundle(campaign, storage=ToyhostStorage(runs, results), profile=profile)


def _family(bundle: AnalysisContextBundle) -> ComparisonFamily:
    (family,) = bundle.multiple_comparisons.families
    return family


# --- the bounded test's own rate ---------------------------------------------------------------------------------------


def _product_separates(diffs: tuple[int, ...], level: float) -> bool:
    control = [1.0 if d < 0 else 0.0 for d in diffs]
    contrast = [1.0 if d > 0 else 0.0 for d in diffs]
    p = bounded_separation_p(control, contrast, paired=True, value_range=(0.0, 1.0), alpha=level)
    return p is not None and p < level


@pytest.mark.parametrize("level", [SIGNIFICANCE_ALPHA, SIGNIFICANCE_ALPHA / 3], ids=["alpha", "alpha-over-3"])
def test_the_enumeration_is_the_product_s_test_sequence_for_sequence(level: float) -> None:
    """Every pass/fail sequence of up to seven pairs, read by the product's own p: the restated enumeration
    (``engine_take_up.plug_in_separating``) counts exactly the sequences it separates, by their counts each way."""
    for cases in range(2, 8):
        by_product: Counter[tuple[int, int]] = Counter()
        for diffs in itertools.product((-1, 0, 1), repeat=cases):
            order = bet_order(cases)
            if _product_separates(tuple(diffs[order.index(i)] for i in range(cases)), level):
                by_product[diffs.count(1), diffs.count(-1)] += 1
        assert plug_in_separating(cases, level) == by_product, cases


@pytest.mark.parametrize("plus", [Fraction(1, 20), Fraction(1, 4), Fraction(1, 2)], ids=["rare", "coins", "no-ties"])
def test_the_bounded_test_holds_alpha_exactly_on_every_pass_fail_sequence_up_to_twelve_pairs(plus: Fraction) -> None:
    """Enumerated in full, every sequence of up to twelve pass/fail pairs of equal mean: the chance the engine's reading
    separates is at most α at α, and at most α/3 at each look of a plan of three. Ville's inequality holds it at every
    count; from 10 to 100 pairs the statistics certificate counts it by seeded simulation (``engine.null.*``)."""
    for level in (SIGNIFICANCE_ALPHA, SIGNIFICANCE_ALPHA / 3):
        for cases in range(2, 13):
            rate = exact_small_rate(plug_in_separating(cases, level), cases, plus)
            assert rate <= level, f"{float(rate):.5f} at {cases} pairs, level {level}"


# --- the product -----------------------------------------------------------------------------------------------------


def test_the_engine_reads_a_check_and_a_grade_on_0_to_1_by_the_bounded_test_not_the_t_test() -> None:
    t_p = composite_significance(_CONTROL, _CONTRAST, paired=True).p_value
    assert t_p is not None and 2 * t_p < SIGNIFICANCE_ALPHA, "the t test would separate both, Holm over the two"

    family = _family(_bundle())
    assert family.look_alpha is None and family.family_size == 2
    level = SIGNIFICANCE_ALPHA / 2
    bounded = bounded_separation_p(_CONTROL, _CONTRAST, paired=True, value_range=(0.0, 1.0), alpha=level)
    assert bounded is not None and bounded > level
    for name in ("answered_in_full", "graded"):
        (compared,) = [comparison for comparison in family.comparisons if comparison.name == name]
        assert (compared.basis, compared.test) == ("bounded", "paired"), name
        assert compared.p_raw == pytest.approx(bounded, rel=1e-12), "the bet its interval is drawn by, tuned to α/2"
        assert compared.interval == compared.bounded_interval and compared.interval_basis == "bounded", name
        assert compared.verdict == "not_separated", name


def test_a_contrast_whose_interval_excludes_zero_is_separated_by_the_same_bet() -> None:
    """Seven of forty to the contrast alone: the bet tuned to α/2 gives p ≈ 0.022 and its 97.5% interval excludes zero,
    so both readings separate. Read by the mixture's p beside that interval, as before, they read not separated
    beside an interval that excludes zero."""
    family = _family(_bundle(_SWEEP))
    for name in ("answered_in_full", "graded"):
        (compared,) = [comparison for comparison in family.comparisons if comparison.name == name]
        assert compared.interval is not None and compared.interval[0] > 0.0, name
        assert compared.p_raw == pytest.approx(
            bounded_separation_p(*_SWEEP, paired=True, value_range=(0.0, 1.0), alpha=SIGNIFICANCE_ALPHA / 2)
        )
        assert compared.verdict == "improved", name


def _reads_not_separated_beside_an_excluding_interval(family: ComparisonFamily) -> list[str]:
    return [
        comparison.name
        for comparison in family.comparisons
        if comparison.verdict in ("not_separated", "equivalent")
        and comparison.interval is not None
        and (comparison.interval[0] > 0.0 or comparison.interval[1] < 0.0)
    ]


def test_no_engine_contrast_reads_not_separated_beside_an_interval_that_excludes_zero() -> None:
    """Seeded engine families of pass/fail checks and grades on 0 to 1, paired and unpaired, from 4 to 60 cases and
    every gap from none to wide: wherever a contrast's interval excludes zero, it is separated."""
    rng = random.Random("no-contradiction")
    excluded = 0
    for index in range(60):
        cases = rng.choice((4, 8, 12, 20, 40, 60))
        control_rate = rng.random()
        contrast_rate = min(1.0, max(0.0, control_rate + rng.choice((-0.6, -0.3, 0.0, 0.3, 0.6))))
        control = [float(rng.random() < control_rate) for _ in range(cases)]
        contrast = [float(rng.random() < contrast_rate) for _ in range(cases)]
        family = _family(_bundle((control, contrast), unpaired=index % 3 == 0))
        assert not _reads_not_separated_beside_an_excluding_interval(family), (index, cases)
        excluded += sum(
            comparison.interval is not None and (comparison.interval[0] > 0.0 or comparison.interval[1] < 0.0)
            for comparison in family.comparisons
        )
    assert excluded >= 10, "the seeds must reach intervals that exclude zero"


# --- unpaired --------------------------------------------------------------------------------------------------------


def _excludes_zero(interval: tuple[float, float] | None) -> bool:
    return interval is not None and (interval[0] > 0.0 or interval[1] < 0.0)


def test_the_unpaired_test_is_its_interval_s_own_inversion_through_the_product() -> None:
    """An unpaired engine reading on 0 to 1 separates exactly where its interval excludes zero: its p, below the level
    each interval is read at (α/m), wherever that interval's two arms' ends part, and nowhere else. One control passing
    one case of ten against a contrast passing all ten is the case the midpoint cut missed (its p 0.050 at α/2 beside a
    97.5% interval from +0.006); seeded arms of every size and gap reach both outcomes."""
    rng = random.Random("unpaired-inversion")
    sides = [([1.0] + [0.0] * 9, [1.0] * 10)]
    for _ in range(24):
        sizes = rng.choice(((6, 6), (10, 10), (12, 30), (40, 40)))
        rates = rng.random(), rng.random()
        sides.append(tuple([float(rng.random() < rate) for _ in range(size)] for size, rate in zip(sizes, rates)))  # type: ignore[arg-type]
    outcomes = set()
    for control, contrast in sides:
        family = _family(_bundle((control, contrast), unpaired=True))
        level = SIGNIFICANCE_ALPHA / family.family_size
        for compared in family.comparisons:
            if compared.p_raw is None or compared.name not in ("answered_in_full", "graded"):
                continue
            assert compared.test == "unpaired"
            restated = engine_separation(control, contrast, paired=False, value_range=(0.0, 1.0), alpha=level).p
            assert compared.p_raw == pytest.approx(restated, rel=1e-12)
            excluded = _excludes_zero(compared.interval)
            assert (compared.p_raw < level) == excluded, (
                len(control),
                len(contrast),
                compared.p_raw,
                compared.interval,
            )
            assert (compared.verdict in ("improved", "regressed")) == excluded
            outcomes.add(excluded)
    assert outcomes == {True, False}, "the seeds must reach both"


def test_the_unpaired_enumeration_is_the_restated_test_pair_for_pair() -> None:
    """Every pair of pass/fail sequences of four cases a side: the enumeration's rule (one arm's lower end above the
    other's upper end, ``engine_take_up.unpaired_bounds``) separates exactly the pairs the restated unpaired p does."""
    for level in (SIGNIFICANCE_ALPHA, SIGNIFICANCE_ALPHA / 3):
        bounds = unpaired_bounds(4, level)
        for a, (low_a, high_a) in bounds.items():
            for b, (low_b, high_b) in bounds.items():
                p = engine_separation(list(a), list(b), paired=False, value_range=(0.0, 1.0), alpha=level).p
                separated = p is not None and p < level
                assert separated == (low_b > high_a or low_a > high_b), (a, b, p)


@pytest.mark.parametrize("rate", [Fraction(1, 10), Fraction(1, 2), Fraction(9, 10)], ids=["rare", "even", "common"])
def test_the_unpaired_test_holds_alpha_exactly_on_every_pair_of_sequences_up_to_ten_a_side(rate: Fraction) -> None:
    """Enumerated in full, every pair of pass/fail sequences of up to ten cases a side, both arms passing at ``rate``:
    the chance the engine's unpaired reading separates is at most α at α, and at most α/3 at each look of a plan of
    three. From 6 to 96 a side the certificate counts it by seeded simulation (``engine.null.unpaired.*``)."""
    for level in (SIGNIFICANCE_ALPHA, SIGNIFICANCE_ALPHA / 3):
        for cases in (2, 4, 6, 8, 10):
            chance = exact_unpaired_rate(cases, level, rate)
            assert chance <= level, f"{float(chance):.6f} at {cases} a side, level {level}"


def test_an_engine_family_of_a_one_check_eval_holds_the_check_once() -> None:
    family = _family(_bundle())
    names = sorted(comparison.name for comparison in family.comparisons if comparison.p_raw is not None)
    assert names == ["answered_in_full", "graded"], "the pass rate is the check case for case: read once, as the check"
    assert not [comparison for comparison in family.comparisons if comparison.reading == "trial"]


def test_an_engine_family_of_two_checks_holds_the_pass_rate_beside_them() -> None:
    family = _family(_bundle(second_check=True))
    (passed,) = [comparison for comparison in family.comparisons if comparison.reading == "trial"]
    assert passed.name == "passed" and passed.basis == "bounded"
    assert {comparison.name for comparison in family.comparisons} >= {"answered_in_full", "graded", "passed"}
