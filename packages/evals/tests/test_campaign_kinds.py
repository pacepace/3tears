"""What a campaign's stored memo page says where its memo recommends nothing, decided by the campaign's kind.

A **measurement** (one arm) has nothing to choose between, and a **choice** (several arms, no control) may find no
clear winner; a memo recommending no arm over either is laid out (:func:`~threetears.evals.analysis.report.build_report`)
with the line its kind earns. The page states the fingerprint of the aspect's bundle it was laid out over, never the
campaign bundle's.

The memo writer that wrote such memos is retired, and its refusals of a recommendation off a campaign's kind left with
it; the memos it stored still lay out. Each memo here is built as the writer stored one (:func:`_stored_memo`), over the
toy host's campaign: its narrow width the control for a change, no control for a choice, one arm alone for a
measurement. What the writer is told of each kind is pinned in ``test_campaign_kind_view.py``.
"""

from __future__ import annotations

from packages.evals.tests.report_support import memo_analysis_over

from typing import Any

from packages.evals.tests.aspect_support import with_control
from packages.evals.tests.campaign_bundle_support import arms_of, campaign_bundle
from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_AXIS, TOYHOST_NARROW, TOYHOST_WIDE, toyhost_campaign
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.toyhost_cells import arm_at, cell_at
from threetears.evals.analysis import AnalysisContextBundle, CampaignBundle, assemble_context_bundle
from threetears.evals.analysis.campaign_readings import PairReading
from threetears.evals.analysis.cells import variant_of_cell_ref
from threetears.evals.analysis.report import build_report
from threetears.evals.analysis.report.build import analysis_bundle


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


def _memo(bundle: CampaignBundle, level: int | None) -> dict[str, Any]:
    """The toy memo: one finding on the widths' latency, recommending the arm at ``level``, or none."""
    recommendation = (
        []
        if level is None
        else [
            {
                "arm": arm_at(bundle, level),
                "reasons": ["The narrow chunk width is much faster on every document."],
                "confidence": "high",
                "rests_on": [0],
                "floors_overridden": [],
                "behind_on": [],
                "bars_missed": [],
            }
        ]
    )
    return {
        "findings": [
            {
                "title": "widening the chunk slows every document down",
                "body": "The narrow width averaged about 920 ms per document and the wide one about 1420 ms.",
                "confidence": "high",
                "axes": [TOYHOST_AXIS],
                "evidence": [
                    {"cell": cell_at(bundle, TOYHOST_NARROW), "measure_id": "total_ms", "reading": "measure"},
                    {"cell": cell_at(bundle, TOYHOST_WIDE), "measure_id": "total_ms", "reading": "measure"},
                ],
                "trials": [],
                "caveats": [],
                "invalidates": [],
                "durable": "",
            }
        ],
        "recommendation": recommendation,
        "aspect_summaries": [
            {
                "aspect": bundle.labels[0],
                "what_changed": "The wide chunk width slowed every document down.",
                "how_sure": "Sure of the direction; the gap is wide.",
                "means": "The wider one is slower by a wide margin.",
            }
        ],
        "summary": "The narrow width is much faster on every document.",
        "next": [],
    }


# --- a measurement -------------------------------------------------------------------------------------------------


def _measurement() -> CampaignBundle:
    """The toy aspect as a campaign of its narrow width alone."""
    aspect = _aspect(control=False)
    (narrow,) = [arm for arm in arms_of([aspect]) if arm.variant_key == _variant(aspect, TOYHOST_NARROW)]
    return campaign_bundle(aspect, arms=[narrow])


def test_a_measurement_s_memo_with_no_recommendation_is_laid_out_saying_there_is_no_arm_to_recommend() -> None:
    bundle = _measurement()
    analysis = memo_analysis_over(bundle, _memo(bundle, None))
    assert analysis.document is not None
    assert analysis.document.recommended() is None and analysis.recommended_variant_key is None
    report = build_report(analysis)
    assert any(
        getattr(block, "text", "") == "This campaign measures one arm, so there is no arm to recommend."
        for block in report.blocks
    )


# --- a choice ------------------------------------------------------------------------------------------------------


def _choice(**fields: Any) -> CampaignBundle:
    return campaign_bundle(_aspect(control=False), **fields)


def _unseparated(bundle: CampaignBundle) -> PairReading:
    """Code's reading of the two widths on the toy aspect, neither shown ahead."""
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
        ahead=None,
        ahead_label=None,
        reason="the wide width passed 20 points more trials, its whole interval above zero",
        confirmatory=True,
        first_cases=12,
        second_cases=12,
        first_passed=24,
        first_trials=36,
        second_passed=31,
        second_trials=36,
    )


def test_a_choice_s_memo_recommending_no_arm_is_laid_out_as_no_clear_winner() -> None:
    bundle = _choice(pairs=[_unseparated(_choice())])
    analysis = memo_analysis_over(bundle, _memo(bundle, None))
    assert analysis.document is not None and analysis.document.recommended() is None
    assert any(
        getattr(block, "text", "") == "No clear winner: the writer recommends no arm."
        for block in build_report(analysis).blocks
    )


# --- the page's fingerprint is the bundle served beside it ---------------------------------------------------------


def test_the_page_states_the_fingerprint_of_the_bundle_it_was_laid_out_over() -> None:
    bundle = campaign_bundle(_aspect(control=True))
    analysis = memo_analysis_over(bundle, _memo(bundle, TOYHOST_NARROW))
    report = build_report(analysis)
    assert report.source.bundle_fingerprint == analysis_bundle(analysis).fingerprint()
    assert report.source.bundle_fingerprint != analysis.generation.bundle_fingerprint, (
        "the generation's is the whole campaign bundle's"
    )
