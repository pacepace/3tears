"""What a campaign's writer is told of its kind, read off its arms (owner, 2026-10-18).

A **measurement** (one arm) has nothing to choose between, so its writer is told it recommends no arm. A **choice**
(several arms, no control) leads with code's pair readings and its missed bars, so the writer reads which arm was shown
ahead before anything else. Driven over the toy host's campaign: no control for a choice, its narrow width alone for a
measurement.
"""

from __future__ import annotations

import json
from typing import Any

from packages.evals.tests.aspect_support import with_control
from packages.evals.tests.campaign_bundle_support import arms_of, campaign_bundle
from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_NARROW, TOYHOST_WIDE, toyhost_campaign
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.toyhost_cells import arm_at, cell_at
from threetears.evals.analysis import AnalysisContextBundle, CampaignBundle, assemble_context_bundle
from threetears.evals.analysis.campaign_bundle import build_campaign_user_message
from threetears.evals.analysis.campaign_readings import BarStatus, PairReading
from threetears.evals.analysis.cells import variant_of_cell_ref


def _aspect(*, control: bool) -> AnalysisContextBundle:
    profile = toyhost_profile()
    campaign, storage = toyhost_campaign(profile=profile)
    bundle = assemble_context_bundle(campaign, storage=storage, profile=profile)
    if not control:
        return bundle
    campaign = with_control(campaign, variant_of_cell_ref(cell_at(bundle, TOYHOST_NARROW)))
    return assemble_context_bundle(campaign, storage=storage, profile=profile)


def _variant(bundle: AnalysisContextBundle | CampaignBundle, level: int) -> str:
    variant = variant_of_cell_ref(cell_at(bundle, level))
    assert variant is not None
    return variant


def _measurement() -> CampaignBundle:
    """The toy aspect as a campaign of its narrow width alone."""
    aspect = _aspect(control=False)
    (narrow,) = [arm for arm in arms_of([aspect]) if arm.variant_key == _variant(aspect, TOYHOST_NARROW)]
    return campaign_bundle(aspect, arms=[narrow])


def _choice(**fields: Any) -> CampaignBundle:
    return campaign_bundle(_aspect(control=False), **fields)


def _pair(bundle: CampaignBundle, *, ahead: int | None) -> PairReading:
    """Code's reading of the two widths on the toy aspect, ``ahead`` the width shown ahead (None: no separation)."""
    narrow, wide = (_variant(bundle, level) for level in (TOYHOST_NARROW, TOYHOST_WIDE))
    return PairReading(
        aspect=bundle.labels[0],
        first=narrow,
        first_label=arm_at(bundle, TOYHOST_NARROW),
        second=wide,
        second_label=arm_at(bundle, TOYHOST_WIDE),
        change=-20.0,
        interval=(-35.0, -5.0),
        confidence=0.95,
        p=0.01,
        p_holm=0.01,
        ahead=None if ahead is None else _variant(bundle, ahead),
        ahead_label=None if ahead is None else arm_at(bundle, ahead),
        reason="the wide width passed 20 points more trials, its whole interval above zero",
        confirmatory=True,
        first_cases=12,
        second_cases=12,
        first_passed=24,
        first_trials=36,
        second_passed=31,
        second_trials=36,
    )


def _bar(bundle: CampaignBundle, level: int, status: str = "missed") -> BarStatus:
    return BarStatus(
        aspect=bundle.labels[0],
        arm=_variant(bundle, level),
        arm_label=arm_at(bundle, level),
        measure="field_accuracy",
        bar=0.92,
        direction="higher_is_better",
        status=status,  # type: ignore[arg-type]
        value=0.85,
        interval=(0.8, 0.9),
        margin=None,
        reason="its whole interval is below 0.92",
    )


def test_one_arm_is_a_measurement_and_its_writer_is_told_so() -> None:
    bundle = _measurement()
    assert bundle.kind == "measurement"
    view = json.loads(build_campaign_user_message(bundle).split("\n", 1)[1])
    assert view["kind"] == "measurement" and "recommends no arm" in view["kind_means"]


def test_several_arms_and_no_control_are_a_choice_and_its_pairs_come_first() -> None:
    bare = _choice()
    bundle = _choice(pairs=[_pair(bare, ahead=TOYHOST_WIDE)], bars=[_bar(bare, TOYHOST_NARROW)])
    assert bundle.kind == "choice" and bundle.control() is None
    view = json.loads(build_campaign_user_message(bundle).split("\n", 1)[1])
    assert list(view)[:5] == ["campaign", "kind", "kind_means", "pairs", "bars"]
    assert view["pairs"][0]["ahead"] == arm_at(bundle, TOYHOST_WIDE)
    assert view["bars"][0]["status"] == "missed" and "bar_warning" in view
