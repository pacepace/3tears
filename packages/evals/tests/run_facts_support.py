"""A campaign of one aspect and two arms, built from the runs and results a test writes: what a run's facts are read on.

The run facts (the unanswered calls a usage row counts, the rig a run recorded at launch) are carried from the run and
its results into the bundle, the report input and the analyst's overview. A test reads each hop from runs it wrote:
:func:`run_of` and :func:`result_of` write them, on fixed ids and one fixed instant, and :func:`campaign_over`
assembles the aspect's bundle through the toy host, then the campaign's bundle over it. The first arm is the control.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from threetears.evals.analysis import assemble_context_bundle
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.campaign_bundle import CampaignBundle, build_campaign_bundle
from threetears.evals.analysis.campaign_readings import campaign_readings
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.declaration import CampaignArm
from threetears.evals.schema.models import EvalResult, EvalRun, GoalStateOutcome, RoleUsage
from packages.evals.tests.campaign_support import ASKED_AT, QUESTION
from packages.evals.tests.factories import (
    fixture_variant_key,
    make_aspect_campaign,
    make_eval_result,
    make_eval_run,
    minimal_aspect_design,
)
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

__all__ = ["ASPECT", "CASES", "CHECK", "RAN_AT", "Trials", "aspect_bundle", "campaign_over", "result_of", "run_of"]

#: The aspect's label.
ASPECT = "answers"

#: The cases every arm runs.
CASES = 6

#: The one goal check every trial is graded on.
CHECK = "reply.total == case.total"

#: When every run was made.
RAN_AT = "2026-10-01T00:00:00+00:00"


def run_of(arm: str, **overrides: Any) -> EvalRun:
    """The arm's one run, over :data:`CASES` cases, its id ``run-<arm>``; ``overrides`` set any other field."""
    return make_eval_run(
        id=f"run-{arm}",
        status="completed",
        candidate_model=arm,
        test_case_ids=[f"tc-{index}" for index in range(CASES)],
        created_at=RAN_AT,
        **overrides,
    )


def result_of(run: EvalRun, index: int, *, usage: Sequence[RoleUsage] | None = None, **overrides: Any) -> EvalResult:
    """One trial of ``run`` on case ``index``: it passes the check on the even cases, and its candidate's usage is
    ``usage`` where given, else one priced call."""
    fields: dict[str, Any] = {
        "id": f"{run.id}-{index}",
        "eval_run_id": run.id,
        "scope_id": run.scope_id,
        "model": run.candidate_model,
        "test_case_id": f"tc-{index}",
        "scored_at": RAN_AT,
        "goal_state_outcomes": [GoalStateOutcome(expression=CHECK, passed=index % 2 == 0)],
        "rubric_scores": [],
        "cost_usd": 0.002,
        "usage": (
            list(usage)
            if usage is not None
            else [RoleUsage(role="candidate", model=run.candidate_model, cost_usd=0.002, call_count=1)]
        ),
    }
    return make_eval_result(**{**fields, **overrides})


#: Each arm's run and its results, the control's first.
Trials = Sequence[tuple[EvalRun, Sequence[EvalResult]]]


def aspect_bundle(trials: Trials) -> tuple[EvalCampaign, AnalysisContextBundle, list[CampaignArm]]:
    """The campaign of the one aspect over the runs, its bundle over them and their results, and its arms (the first
    the control)."""
    runs = [run for run, _ in trials]
    arms = [
        CampaignArm(
            label=run.candidate_model,
            role="control" if index == 0 else "candidate",
            variant_key=fixture_variant_key(run.candidate_model),
            **({"named_at": ASKED_AT} if index == 0 else {}),
        )
        for index, run in enumerate(runs)
    ]
    campaign = make_aspect_campaign(
        id="campaign-run-facts",
        label=ASPECT,
        run_ids=[run.id for run in runs],
        design=minimal_aspect_design(merit_axes=[]),
        question=QUESTION,
        asked_at=ASKED_AT,
        control=arms[0],
        margin=10.0,
        arms=arms,
        scope_id=runs[0].scope_id,
        name="run facts",
        subject_id=runs[0].subject_snapshot.subject_id,
        subject_kind="s",
        created_at=ASKED_AT,
    )
    results = {run.id: list(held) for run, held in trials}
    bundle = assemble_context_bundle(campaign, storage=ToyhostStorage(runs, results), profile=toyhost_profile())
    return campaign, bundle, arms


def campaign_over(trials: Trials) -> CampaignBundle:
    """The campaign's bundle over the one aspect's runs and their results, the first run's arm its control."""
    campaign, bundle, arms = aspect_bundle(trials)
    readings = campaign_readings(campaign, {ASPECT: bundle}, arms=arms)
    return build_campaign_bundle(campaign, [bundle], arms=arms, readings=readings)
