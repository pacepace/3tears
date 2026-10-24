"""Two arms on one aspect, each one run of its own attempts at its own planned depth, assembled for real.

What a depth test reads: the bundle the aspect assembles, the campaign's readings over it, and each arm's frontier
point. ``A`` is the control, ``B`` the candidate; their attempts are given case by case, so every figure is exact.
"""

from __future__ import annotations


from threetears.evals.analysis import assemble_context_bundle, campaign_readings
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.campaign_readings import CampaignReadings
from threetears.evals.analysis.lenses.frontier import FrontierPoint
from threetears.evals.kernel.declaration import CampaignArm
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.kernel.identity import IDENTITY_VERSION, derive_context_identity, derive_variant_identity
from threetears.evals.schema.models import EvalResult, EvalRun, RoleUsage
from packages.evals.tests.factories import (
    make_aspect_campaign,
    make_eval_run,
    make_scored_result,
    minimal_aspect_design,
)
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

__all__ = ["CASES", "ONCE", "PROFILE", "TWICE", "TwoArms"]

PROFILE = toyhost_profile()

#: The four cases every fixture arm runs.
CASES = ["tc1", "tc2", "tc3", "tc4"]


def _launched(run_id: str, model: str, k: int) -> EvalRun:
    """A run of one arm (its candidate model) at planned depth ``k``, stamped as the launch stamps it."""
    run = make_eval_run(
        id=run_id, status="completed", template_digest="digest-1", candidate_model=model, k_runs=k, test_case_ids=CASES
    )
    context = derive_context_identity(run, PROFILE)
    variant = derive_variant_identity(run=run, profile=PROFILE)
    return run.model_copy(
        update={
            "context_key": context.context_key,
            "context_components": context.context_components,
            "identity_version": IDENTITY_VERSION,
            "variant_levers": variant.levers,
        }
    )


def _key(run: EvalRun) -> str:
    return derive_variant_identity(run=run, profile=PROFILE).variant_key


def _results(run: EvalRun, outcomes: dict[str, tuple[bool, ...]]) -> list[EvalResult]:
    """One result per attempt, each case's attempts in order, every candidate call naming the model that answered."""
    usage = [
        RoleUsage(
            role="candidate", model=run.candidate_model, cost_usd=0.01, call_count=1, served_model=run.candidate_model
        )
    ]
    return [
        make_scored_result(
            test_case_id=case, run_id=run.id, k=k, goal_passes=(passed,), model=run.candidate_model
        ).model_copy(
            update={
                "variant_key": _key(run),
                "identity_version": IDENTITY_VERSION,
                "scope_id": run.scope_id,
                "usage": usage,
            }
        )
        for case, attempts in outcomes.items()
        for k, passed in enumerate(attempts, start=1)
    ]


class TwoArms:
    """Two arms on one aspect: ``A`` the control, ``B`` the candidate, each one run with its own attempts."""

    def __init__(
        self,
        a: dict[str, tuple[bool, ...]],
        b: dict[str, tuple[bool, ...]],
        *,
        k_a: int,
        k_b: int,
        profile: HostProfile = PROFILE,
    ) -> None:
        self.run_a, self.run_b = _launched("run-a", "sonnet", k_a), _launched("run-b", "haiku", k_b)
        self.key_a, self.key_b = _key(self.run_a), _key(self.run_b)
        self.results = {self.run_a.id: _results(self.run_a, a), self.run_b.id: _results(self.run_b, b)}
        self.arms = [
            CampaignArm(label="A", role="control", variant_key=self.key_a),
            CampaignArm(label="B", role="candidate", variant_key=self.key_b),
        ]
        self.campaign = make_aspect_campaign(
            label="character",
            run_ids=[self.run_a.id, self.run_b.id],
            question="Is B better on character?",
            design=minimal_aspect_design(merit_axes=[]),
            margin=10.0,
            control=self.arms[0],
            scope_id=self.run_a.scope_id,
            subject_id=self.run_a.subject_snapshot.subject_id,
            behavior="b",
        )
        self.bundle = assemble_context_bundle(
            self.campaign, storage=ToyhostStorage([self.run_a, self.run_b], self.results), profile=profile
        )

    def readings(self, bundle: AnalysisContextBundle | None = None) -> CampaignReadings:
        return campaign_readings(self.campaign, {"character": bundle or self.bundle}, arms=self.arms)

    def point(self, key: str) -> FrontierPoint:
        return next(point for point in self.bundle.frontier.subjects[0].points if point.variant_key == key)


#: Arm A run twice on every case, arm B once: the shares A 1, 1/2, 0, 1 and B 1, 0, 1, 1.
TWICE = {"tc1": (True, True), "tc2": (True, False), "tc3": (False, False), "tc4": (True, True)}
ONCE = {"tc1": (True,), "tc2": (False,), "tc3": (True,), "tc4": (True,)}
