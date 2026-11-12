"""A case result is a miss exactly when its trial failed, and its reasons are what failed it.

``CaseResult.missed_because`` once ran a rule of its own: any directed scorer at or below 0 was a miss, so a trial the
page passed was listed among the misses, and a 1–5 judged score under the bar, which fails the trial on the page,
was no miss at all. Contracts held here (``threetears.evals.analysis.CaseResult.of``):

- a trial is a miss exactly when its state is ``failed``, as the page's trials read it;
- its reasons are each check it failed, worded as its pass rule words it at the failing value, and each judged
  dimension under the bar, with the judge's reason;
- a measure that is no check names no reason, whatever its value.
"""

from __future__ import annotations

from packages.evals.tests.factories import make_eval_result
from threetears.evals.analysis import CaseResult
from threetears.evals.kernel.metrics import MetricDescriptor
from threetears.evals.schema import RubricScore
from threetears.evals.schema.models import EvalResult, GoalStateOutcome

THRESHOLD = 3

CHECK = MetricDescriptor(
    name="top_up_never",
    data_type="boolean",
    family="mechanical",
    transferability_class="mechanical",
    attribution_scope="end_to_end",
    description="Whether the candidate never topped the customer up.",
    reader_name="Never topped up",
    higher_is_better=False,
)

#: A directed score that is no check: a grade on a range, not a pass or a fail.
SCORE = MetricDescriptor(
    name="politeness",
    data_type="numeric",
    family="mechanical",
    transferability_class="mechanical",
    attribution_scope="end_to_end",
    description="How polite the reply was.",
    reader_name="Politeness",
    higher_is_better=True,
    value_range=(0.0, 1.0),
)

CATALOG = {descriptor.name: descriptor for descriptor in (CHECK, SCORE)}


def _read(result: EvalResult) -> CaseResult:
    return CaseResult.of(
        result, case="c", given=None, expected=None, answer=None, catalog=CATALOG, rubric_threshold=THRESHOLD
    )


def _judged(score: int) -> list[RubricScore]:
    return [
        RubricScore(
            dim="conversation.tone",
            score=score,
            scale="ordinal",
            axis="capability",
            judge_temperature=0.0,
            reasoning="the reply  was\ncurt",
        )
    ]


def test_a_passed_trial_is_no_miss_whatever_a_score_that_is_no_check_reads() -> None:
    read = _read(make_eval_result(rubric_scores=_judged(4), host_measures={SCORE.name: 0.0, CHECK.name: False}))

    assert read.state == "passed"
    assert read.missed_because == [] and not read.missed


def test_a_judged_score_under_the_bar_is_a_miss_with_the_judge_s_reason() -> None:
    read = _read(make_eval_result(rubric_scores=_judged(2), host_measures={SCORE.name: 1.0}))

    assert read.state == "failed"
    assert read.missed_because == ["the judge scored conversation.tone 2 of 5, under the bar of 3: the reply was curt"]


def test_each_check_a_trial_failed_is_a_reason_in_its_pass_rule_s_words() -> None:
    read = _read(
        make_eval_result(
            goal_state_outcomes=[GoalStateOutcome(expression="state.cart.length >= 1", passed=False)],
            rubric_scores=_judged(4),
            host_measures={CHECK.name: True, SCORE.name: 1.0},
        )
    )

    assert read.state == "failed"
    assert read.missed_because == ["goal check state.cart.length >= 1 failed", "Never topped up is true"]
