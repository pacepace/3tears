"""Four campaigns of four shapes, for every test of a campaign's page: a change, a choice, a measurement, and a
change of one aspect.

A test of the page passes on at least two of them, so no rule of the page fits one campaign alone. Each is built as
a host's would be: every arm on every aspect is one run of its own trials, stored and assembled into the aspect's
bundle (:func:`~threetears.evals.analysis.assemble_context_bundle`), each aspect's own page built over it, the
campaign's readings read over every bundle, and the campaign's page over those.

The trials are written case by case rather than run, so every figure is exact and every build of a shape is the same
page, byte for byte: its ids, its times and its figures are fixed. The question was asked, and every criterion set,
before the first run (:data:`~packages.evals.tests.campaign_support.ASKED_AT`).

- :func:`s1_change`: a control and three candidates on three aspects, each arm once per case but one, which runs
  twice per case on one aspect; a floor on every rate of one aspect and a floor on one check of another; the control
  passing no case on the third.
- :func:`s2_choice`: three arms and no control on one aspect, three runs per case, priced and timed apart, held to a
  cost bar and a latency bar, no floor.
- :func:`s3_measurement`: one arm on two aspects, judged on a pass/fail and an ordinal dimension, held to a bar on
  the pass/fail one.
- :func:`s4_change`: a control and one candidate on one aspect, with no floor and no margin.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from threetears.evals.analysis import assemble_context_bundle, build_campaign_report, build_code_only_report
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.campaign_readings import CampaignReadings, campaign_readings
from threetears.evals.analysis.report.model import CampaignReport, Report
from threetears.evals.kernel.authored import AuthoredCampaignAnalysis
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.declaration import BarOverride, CampaignArm, DecisionClause, RateDrop
from threetears.evals.schema import EvalResult, RubricScore
from threetears.evals.schema.models import EvalRun, GoalStateOutcome, LatencyMetrics, RoleUsage, RubricScale
from packages.evals.tests.campaign_support import ASKED_AT, ASSEMBLED_AT, QUESTION
from packages.evals.tests.factories import (
    fixture_variant_key,
    make_aspect_campaign,
    make_eval_result,
    make_eval_run,
    minimal_aspect_design,
)
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

__all__ = [
    "CHECKS",
    "CORRECT_BAR",
    "COST_BAR",
    "FIRST_RUN_AT",
    "LATENCY_BAR",
    "SHAPES",
    "ArmTrials",
    "AspectTrials",
    "Shape",
    "s1_change",
    "s2_choice",
    "s3_measurement",
    "s4_change",
]

#: The two goal-state checks every trial of a change is graded on; a trial passes where both hold.
CHECKS = ("reply.total == case.total", "reply.cites_policy")

#: When every run was made: after the question was asked (:data:`ASKED_AT`), on one fixed instant.
FIRST_RUN_AT = "2026-10-01T00:00:00+00:00"

#: The cases every aspect runs.
CASES = 20

#: The judged dimensions a measurement is scored on, each by its scale.
JUDGED: dict[str, RubricScale] = {"answer.correct": "pass_fail", "answer.clear": "ordinal"}


@dataclass(frozen=True)
class ArmTrials:
    """One arm's trials on one aspect: how many runs per case, and which attempts pass each check.

    :ivar label: The arm, as the campaign names it; its run's model is the label too, so its variant key is the same
        on every aspect.
    :ivar k: The attempts per case.
    :ivar passing: The cases on which every attempt passes the first check, counted from the first case; the rest
        fail it on every attempt.
    :ivar second: The cases on which every attempt passes the second check, from the first case; None passes it on
        every case.
    :ivar flaky: The cases, after the passing ones, whose first attempt passes the first check and whose other
        attempts fail it, so an arm run more than once per case reads a share between nought and one.
    :ivar cost: What each reply costs, in US dollars; None leaves the arm unpriced.
    :ivar latency_ms: Each reply's turn time at the first case, rising ten milliseconds a case; None leaves it untimed.
    :ivar scores: Each judged dimension's score per case, by the dimension; empty judges nothing.
    """

    label: str
    k: int = 1
    passing: int = 0
    second: int | None = None
    flaky: int = 0
    cost: float | None = 0.002
    latency_ms: float | None = None
    scores: Mapping[str, Sequence[int]] = field(default_factory=dict)


@dataclass(frozen=True)
class AspectTrials:
    """One aspect: its label, its margin, the clauses of its floors and its bars, and every arm's trials on it, the
    control's first where the campaign names one."""

    label: str
    arms: Sequence[ArmTrials]
    margin: float | None = 10.0
    floors: Sequence[DecisionClause] = ()
    bars: Sequence[BarOverride] = ()
    checks: bool = True


@dataclass(frozen=True)
class Shape:
    """One campaign of one shape, assembled: the campaign, each aspect's bundle and own page, its readings, and its
    arms; :meth:`page` builds the campaign's page over them."""

    name: str
    campaign: EvalCampaign
    bundles: dict[str, AnalysisContextBundle]
    reports: dict[str, Report]
    readings: CampaignReadings
    arms: list[CampaignArm]

    def key(self, label: str) -> str:
        """The variant key of the arm the campaign labels ``label``."""
        return next(arm.variant_key for arm in self.arms if arm.label == label)

    def page(
        self,
        memo: AuthoredCampaignAnalysis | None = None,
        *,
        recommended: str | None = None,
    ) -> CampaignReport:
        """The campaign's page, with ``memo`` where one was written; ``recommended`` is the label of the arm the memo
        recommends, whose key the analysis recorded."""
        return build_campaign_report(
            self.campaign,
            self.readings,
            self.reports,
            memo,
            recommended_variant_key=self.key(recommended) if recommended is not None else None,
        )


def _run(aspect: str, arm: ArmTrials) -> EvalRun:
    """The arm's one run on the aspect, at its planned depth, on fixed ids and a fixed instant."""
    return make_eval_run(
        id=f"run-{aspect}-{arm.label}",
        status="completed",
        candidate_model=arm.label,
        k_runs=arm.k,
        test_case_ids=[_case(index) for index in range(CASES)],
        created_at=FIRST_RUN_AT,
    )


def _case(index: int) -> str:
    return f"tc-{index:02d}"


def _results(run: EvalRun, aspect: AspectTrials, arm: ArmTrials) -> list[EvalResult]:
    """Every attempt of the arm's run, case by case, each graded on the checks and the judged dimensions."""
    results = []
    for index in range(CASES):
        for attempt in range(1, arm.k + 1):
            first = index < arm.passing or (attempt == 1 and index < arm.passing + arm.flaky)
            second = arm.second is None or index < arm.second
            usage = (
                [RoleUsage(role="candidate", model=arm.label, cost_usd=arm.cost, call_count=1, served_model=arm.label)]
                if arm.cost is not None
                else []
            )
            results.append(
                make_eval_result(
                    id=f"{run.id}-{index:02d}-{attempt}",
                    eval_run_id=run.id,
                    scope_id=run.scope_id,
                    model=arm.label,
                    test_case_id=_case(index),
                    k_iteration=attempt,
                    scored_at=FIRST_RUN_AT,
                    goal_state_outcomes=[
                        GoalStateOutcome(expression=CHECKS[0], passed=first),
                        GoalStateOutcome(expression=CHECKS[1], passed=second),
                    ]
                    if aspect.checks
                    else [],
                    rubric_scores=[
                        RubricScore(
                            dim=dim, score=scores[index], scale=JUDGED[dim], axis="capability", judge_temperature=0.0
                        )
                        for dim, scores in arm.scores.items()
                    ],
                    cost_usd=arm.cost,
                    usage=usage,
                    latency=None
                    if arm.latency_ms is None
                    else LatencyMetrics(total_ms=arm.latency_ms + 10.0 * index + 3.0 * attempt),
                )
            )
    return results


def _assembled(
    name: str, aspects: Sequence[AspectTrials], *, control: bool, question: str
) -> tuple[EvalCampaign, dict[str, AnalysisContextBundle], list[CampaignArm]]:
    """Each aspect as a one-aspect campaign over its runs, assembled; then one campaign holding every aspect, every
    floor and every arm, the first aspect's arm the control where ``control``."""
    labels = list(dict.fromkeys(arm.label for aspect in aspects for arm in aspect.arms))
    arms = [
        CampaignArm(
            label=label,
            role="control" if control and index == 0 else "candidate",
            variant_key=fixture_variant_key(label),
            **({"named_at": ASKED_AT} if control and index == 0 else {}),
        )
        for index, label in enumerate(labels)
    ]
    held = []
    bundles: dict[str, AnalysisContextBundle] = {}
    for aspect in aspects:
        runs = [_run(aspect.label, arm) for arm in aspect.arms]
        results = {run.id: _results(run, aspect, arm) for run, arm in zip(runs, aspect.arms, strict=True)}
        one = make_aspect_campaign(
            id=f"campaign-{name}",
            label=aspect.label,
            run_ids=[run.id for run in runs],
            design=minimal_aspect_design(merit_axes=[], bars=list(aspect.bars)),
            question=question,
            asked_at=ASKED_AT,
            control=arms[0] if control else None,
            margin=aspect.margin,
            floors=aspect.floors,
            arms=arms,
            scope_id=runs[0].scope_id,
            name=name,
            subject_id=runs[0].subject_snapshot.subject_id,
            subject_kind="s",
            created_at=ASKED_AT,
        )
        held.append(one)
        bundles[aspect.label] = assemble_context_bundle(
            one, storage=ToyhostStorage(runs, results), profile=toyhost_profile()
        )
    first = held[0]
    campaign = first.model_copy(
        update={
            "aspects": [aspect for one in held for aspect in one.aspects],
            "floors": [floor for one in held for floor in one.floors],
        }
    )
    return campaign, bundles, arms


def _shape(name: str, aspects: Sequence[AspectTrials], *, control: bool, question: str = QUESTION) -> Shape:
    """One shape assembled, its readings read and each aspect's own page built with the campaign's labels."""
    campaign, bundles, arms = _assembled(name, aspects, control=control, question=question)
    readings = campaign_readings(campaign, bundles, arms=arms)
    reports = {
        label: build_code_only_report(
            bundle, assembled_at=ASSEMBLED_AT, campaign_name=name, arm_labels=readings.arm_labels()
        )
        for label, bundle in bundles.items()
    }
    return Shape(name=name, campaign=campaign, bundles=bundles, reports=reports, readings=readings, arms=arms)


def s1_change() -> Shape:
    """A change: ``production`` the control, ``v2``, ``v3`` and ``v4`` its candidates, on three aspects.

    - ``billing``: v2 falls from eighteen cases to six, breaking the floor on every rate; v3 and v4 hold near it.
    - ``tone``: v2 rises from ten to eighteen; v3 runs twice per case, its second attempt failing on four cases its
      first passes; v4 passes the first check as production does but the second on only eight cases, breaking the
      floor read on that check alone.
    - ``refunds``: production passes no case, so no drop past the margin can show; v2 passes five.
    """
    return _shape(
        "s1 change",
        [
            AspectTrials(
                "billing",
                [
                    ArmTrials("production", passing=18),
                    ArmTrials("v2", passing=6),
                    ArmTrials("v3", passing=17),
                    ArmTrials("v4", passing=19, cost=0.001),
                ],
                floors=(RateDrop(max_points=10),),
            ),
            AspectTrials(
                "tone",
                [
                    ArmTrials("production", passing=10),
                    ArmTrials("v2", passing=18),
                    ArmTrials("v3", k=2, passing=8, flaky=4),
                    ArmTrials("v4", passing=10, second=8, cost=0.001),
                ],
                floors=(RateDrop(max_points=15, reading=CHECKS[1]),),
            ),
            AspectTrials(
                "refunds",
                [
                    ArmTrials("production", passing=0),
                    ArmTrials("v2", passing=5),
                    ArmTrials("v3", passing=0),
                    ArmTrials("v4", passing=2, cost=0.001),
                ],
            ),
        ],
        control=True,
    )


#: A choice's cost bar: no reply may cost more than a tenth of a cent.
COST_BAR = BarOverride(measure_id="cost_usd", threshold=0.001, direction="lower_is_better")

#: A choice's latency bar: no turn may take more than 800 milliseconds.
LATENCY_BAR = BarOverride(measure_id="total_ms", threshold=800.0, direction="lower_is_better")


def s2_choice() -> Shape:
    """A choice: ``alpha``, ``beta`` and ``gamma``, no control, on one aspect, three runs per case, each priced and
    timed apart and held to :data:`COST_BAR` and :data:`LATENCY_BAR`. Alpha passes the most cases and costs the most;
    gamma is the fastest."""
    return _shape(
        "s2 choice",
        [
            AspectTrials(
                "answers",
                [
                    ArmTrials("alpha", k=3, passing=17, flaky=2, cost=0.004, latency_ms=900.0),
                    ArmTrials("beta", k=3, passing=12, flaky=3, cost=0.0008, latency_ms=600.0),
                    ArmTrials("gamma", k=3, passing=9, cost=0.0005, latency_ms=300.0),
                ],
                margin=None,
                bars=(COST_BAR, LATENCY_BAR),
            )
        ],
        control=False,
        question="Which arm answers within the cost and latency bars?",
    )


def _scores(passing: int, high: int) -> dict[str, list[int]]:
    """A measurement's judged scores: ``answer.correct`` passing the first ``passing`` cases, ``answer.clear`` five on
    the first ``high`` and three on the rest."""
    return {
        "answer.correct": [1 if index < passing else 0 for index in range(CASES)],
        "answer.clear": [5 if index < high else 3 for index in range(CASES)],
    }


#: A measurement's bar: the judged correctness must pass at least nine in ten cases.
CORRECT_BAR = BarOverride(measure_id="answer.correct", threshold=0.9, direction="higher_is_better")


def s3_measurement() -> Shape:
    """A measurement: ``solo`` alone on two aspects, twice per case, judged on ``answer.correct`` (pass/fail) and
    ``answer.clear`` (ordinal), and held to :data:`CORRECT_BAR` on both."""
    return _shape(
        "s3 measurement",
        [
            AspectTrials(
                "accuracy",
                [ArmTrials("solo", k=2, scores=_scores(19, 12))],
                margin=None,
                bars=(CORRECT_BAR,),
                checks=False,
            ),
            AspectTrials(
                "clarity",
                [ArmTrials("solo", k=2, scores=_scores(12, 6))],
                margin=None,
                bars=(CORRECT_BAR,),
                checks=False,
            ),
        ],
        control=False,
        question="How often is solo correct, and how clear is it?",
    )


def s4_change() -> Shape:
    """A change of one aspect: ``production`` the control and ``v2`` its one candidate on ``billing``, with no floor
    and no margin; v2 passes fourteen cases to production's twelve."""
    return _shape(
        "s4 change",
        [
            AspectTrials(
                "billing",
                [ArmTrials("production", passing=12), ArmTrials("v2", passing=14)],
                margin=None,
            )
        ],
        control=True,
    )


#: Every shape, by its short name.
SHAPES: dict[str, Callable[[], Shape]] = {
    "s1": s1_change,
    "s2": s2_choice,
    "s3": s3_measurement,
    "s4": s4_change,
}
