"""An engine family reads a reading on 0 to 1 by the bounded test, and holds an eval of one check once.

**The bounded test holds α on pass/fail pairs, exactly.** The engine reads a comparison once at the count it has (no
committed order lets it be read again as cases are added): on 0 to 1, by the mixture bet over the pairs in the order
fixed by their count. On pairs of pass/fail answers its capital depends only on how many pairs went each way
(``engine_take_up.hedged_log_capital``, held here to the product's p), so the chance it separates by ``n`` pairs is a
sum over the walk of those counts, computed exactly as fractions (``engine_take_up.exact_rates``): at most α at every
count from 10 to 100, at every share of pairs one arm alone got right, and under the engine's plan of looks at α/L. The
paired t test it read by before separated fair coin pairs 5.28% of the time at 40 cases.

**Through the product**, forty cases where one arm alone passed ten and the other two: the paired t test reads p ≈ 0.019
and would separate them; the engine reads the bounded test's p, the restated rule's, and does not, on a check and on a
grade on 0 to 1 alike.

**One hypothesis, read once.** An eval whose every trial is graded on one check passes a trial exactly where the check
holds, so its pass rate is the check case for case; the family holds the check alone. With a second check the pass
rate is no one check's, and the family holds it beside both.

Mutations that turn this file red (each run against a saved copy and restored from it):

- ``_fixed_count_test``: returning the t-test's reading for a reading on 0 to 1, as before, so the grade separates and
  the check's p is the t test's.
- ``_fixed_count_test``: betting in the order the case ids sort in, so the product's p is not the restated one.
- ``multiple_comparisons``: not calling ``_repeats_a_reading`` on the engine, so the pass rate joins the check.
- ``_one_check``: returning the first check of several, so a two-check eval loses its pass rate.
"""

from __future__ import annotations

import math
import random
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
from packages.evals.tests.engine_take_up import exact_rates, hedged_log_capital
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

#: Forty cases: ten the contrast alone passed, two the control alone passed, the rest tied, half of them both passing.
_CASES = 40
_CONTROL = [0.0] * 10 + [1.0] * 2 + [1.0] * 14 + [0.0] * 14
_CONTRAST = [1.0] * 10 + [0.0] * 2 + [1.0] * 14 + [0.0] * 14


def _bundle(*, second_check: bool = False) -> AnalysisContextBundle:
    """A control and a contrast over :data:`_CASES` cases, one trial each, carrying the check and the grade on the
    same answers; with ``second_check`` each trial is graded on a goal check too, which it always passes."""
    profile = toyhost_profile()
    profile = replace(
        profile, measures=MeasureRegistry((*TOYHOST_MEASURES, _CHECK, _GRADE), families=profile.measures.families)
    )
    runs = []
    results: dict[str, list[EvalResult]] = {}
    for model, values in (("control-model", _CONTROL), ("contrast-model", _CONTRAST)):
        run = make_eval_run(status="completed", candidate_model=model)
        runs.append(run)
        results[run.id] = [
            make_eval_result(
                id=f"{model}-{case}",
                eval_run_id=run.id,
                scope_id=run.scope_id,
                model=model,
                test_case_id=f"tc-{case:02d}",
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


# --- the bounded test's own rate, exactly ----------------------------------------------------------------------------


def test_the_restated_capital_is_the_product_s_p_on_pass_fail_pairs() -> None:
    """The product's p on pass/fail pairs is one over the most the restated capital reached over its counts each way, in
    the order the pairs are bet: what lets the exact rates below stand for the product's test."""
    rng = random.Random("restated-capital")
    for _ in range(200):
        cases = rng.randint(2, 100)
        diffs = [rng.choice((-1, 0, 0, 1)) for _ in range(cases)]
        control = [1.0 if d < 0 else 0.0 for d in diffs]
        contrast = [1.0 if d > 0 else 0.0 for d in diffs]
        order = bet_order(cases)
        product = bounded_separation_p(
            [control[i] for i in order], [contrast[i] for i in order], paired=True, value_range=(0.0, 1.0), mixture=True
        )
        up = down = 0
        most = 0.0
        for i in order:
            up, down = up + (diffs[i] > 0), down + (diffs[i] < 0)
            most = max(most, hedged_log_capital(up, down))
        assert product == pytest.approx(min(1.0, math.exp(-most)), rel=1e-9)
        assert engine_separation(control, contrast, paired=True, value_range=(0.0, 1.0)).p == product


@pytest.mark.parametrize("plus", [Fraction(1, 20), Fraction(1, 4), Fraction(1, 2)], ids=["rare", "coins", "no-ties"])
def test_the_bounded_test_holds_alpha_exactly_at_every_count_from_10_to_100(plus: Fraction) -> None:
    """The exact chance the engine's reading separates pass/fail pairs of equal mean at α, read once at each count from
    10 to 100: at most α wherever each arm alone gets a pair right at ``plus`` (1/4 for two fair coins)."""
    rates = exact_rates(SIGNIFICANCE_ALPHA, 100, plus=plus)
    worst = max(rates[9:])
    assert worst <= SIGNIFICANCE_ALPHA, f"{float(worst):.5f} at {rates.index(worst) + 1} pairs"


def test_the_bounded_test_holds_alpha_exactly_over_the_engine_s_planned_looks() -> None:
    """Under a plan of three looks at 20, 40 and 60 coin pairs, each read at α/3 in the order fixed by its own count:
    each look's exact rate is at most α/3, so the chance any separates is at most their sum, at most α."""
    level = SIGNIFICANCE_ALPHA / 3
    rates = exact_rates(level, 60)
    looks = [rates[19], rates[39], rates[59]]
    assert all(rate <= level for rate in looks), [float(rate) for rate in looks]
    assert sum(looks) <= SIGNIFICANCE_ALPHA


# --- the product -----------------------------------------------------------------------------------------------------


def test_the_engine_reads_a_check_and_a_grade_on_0_to_1_by_the_bounded_test_not_the_t_test() -> None:
    t_p = composite_significance(_CONTROL, _CONTRAST, paired=True).p_value
    assert t_p is not None and 2 * t_p < SIGNIFICANCE_ALPHA, "the t test would separate both, Holm over the two"
    restated = engine_separation(_CONTROL, _CONTRAST, paired=True, value_range=(0.0, 1.0))
    assert restated.basis == "bounded" and restated.p is not None and restated.p > SIGNIFICANCE_ALPHA

    family = _family(_bundle())
    assert family.look_alpha is None and family.family_size == 2
    for name in ("answered_in_full", "graded"):
        (compared,) = [comparison for comparison in family.comparisons if comparison.name == name]
        assert (compared.basis, compared.test) == ("bounded", "paired"), name
        assert compared.p_raw == pytest.approx(restated.p, rel=1e-12), name
        assert compared.interval == compared.bounded_interval and compared.interval_basis == "bounded", name
        assert compared.verdict == "not_separated", name


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
