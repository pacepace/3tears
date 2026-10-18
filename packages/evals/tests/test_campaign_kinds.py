"""What a campaign's memo may recommend is decided by the campaign's kind, read off its arms (owner, 2026-10-18).

A **measurement** (one arm) has nothing to choose between, so its memo recommends no arm, and one that does is
refused. A **choice** (several arms, no control) picks an arm, or recommends none when no arm stood out (no clear
winner); a pick rests on a finding and names every aspect code's pair readings show another arm ahead of it on
(``behind_on``), and every absolute bar it missed (``bars_missed``). A **change** (a control and candidates) recommends
a candidate or keeps the control, and only confirmatory evidence — an aspect asked before its runs — licenses a switch.
There is no "no control, so anything goes": that exemption is gone.

Driven through :func:`~threetears.evals.analysis.generator.generate_analysis` over the toy host's campaign: its narrow
width the control for a change, no control for a choice, one arm alone for a measurement.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from packages.evals.tests.aspect_support import with_control
from packages.evals.tests.campaign_bundle_support import arms_of, campaign_bundle
from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_NARROW, TOYHOST_WIDE, toyhost_campaign
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.toyhost_memo import (
    MODEL,
    PROMPT,
    PROMPT_ID,
    FixturedClient,
    alias_at,
    arm_at,
    cell_at,
    memo_payload,
)
from threetears.evals.analysis import AnalysisContextBundle, CampaignBundle, assemble_context_bundle, generate_analysis
from threetears.evals.analysis.campaign_bundle import build_campaign_user_message
from threetears.evals.analysis.campaign_readings import AspectReading, BarStatus, PairReading
from threetears.evals.analysis.cells import variant_of_cell_ref
from threetears.evals.analysis.errors import SoundnessRefusal
from threetears.evals.analysis.report import build_report
from threetears.evals.analysis.report.build import analysis_bundle
from threetears.evals.kernel.campaign import EvalAnalysis
from threetears.evals.schema.models import utc_now_iso


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


async def _generate(bundle: CampaignBundle, payload: dict[str, Any]) -> EvalAnalysis:
    analysis, _ = await generate_analysis(
        bundle,
        prompt=PROMPT,
        model=MODEL,
        client=FixturedClient(json.dumps(payload)),
        prompt_id=PROMPT_ID,
        bundle_assembled_at=utc_now_iso(),
        profile=toyhost_profile(),
    )
    return analysis


def _recommending(bundle: CampaignBundle, level: int | None, **fields: Any) -> dict[str, Any]:
    payload = memo_payload(bundle)
    if level is None:
        payload["recommendation"] = []
    else:
        # Another arm owes its own words on what it missed: the canned draft's are the narrow width's.
        payload["recommendation"][0] = {
            **payload["recommendation"][0],
            "arm": arm_at(bundle, level),
            "bars_missed": [],
            **fields,
        }
    return payload


# --- a measurement -------------------------------------------------------------------------------------------------


def _measurement() -> CampaignBundle:
    """The toy aspect as a campaign of its narrow width alone."""
    aspect = _aspect(control=False)
    (narrow,) = [arm for arm in arms_of([aspect]) if arm.variant_key == _variant(aspect, TOYHOST_NARROW)]
    return campaign_bundle(aspect, arms=[narrow])


class TestAMeasurementRecommendsNothing:
    def test_one_arm_is_a_measurement_and_its_writer_is_told_so(self) -> None:
        bundle = _measurement()
        assert bundle.kind == "measurement"
        view = json.loads(build_campaign_user_message(bundle).split("\n", 1)[1])
        assert view["kind"] == "measurement" and "recommends no arm" in view["kind_means"]

    async def test_a_measurement_refuses_a_recommendation(self) -> None:
        bundle = _measurement()
        with pytest.raises(SoundnessRefusal, match="this campaign measures one arm, so there is no arm to choose"):
            await _generate(bundle, _recommending(bundle, TOYHOST_NARROW))

    async def test_a_measurement_s_memo_with_no_recommendation_is_kept_and_its_page_says_so(self) -> None:
        bundle = _measurement()
        analysis = await _generate(bundle, _recommending(bundle, None))
        assert analysis.document.recommended() is None and analysis.recommended_variant_key is None
        report = build_report(analysis)
        assert any(
            getattr(block, "text", "") == "This campaign measures one arm, so there is no arm to recommend."
            for block in report.blocks
        )


# --- a choice ------------------------------------------------------------------------------------------------------


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
        ahead=None if ahead is None else arm_at(bundle, ahead),
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


class TestAChoicePicksOrFindsNoClearWinner:
    def test_several_arms_and_no_control_are_a_choice_and_its_pairs_come_first(self) -> None:
        bare = _choice()
        bundle = _choice(pairs=[_pair(bare, ahead=TOYHOST_WIDE)], bars=[_bar(bare, TOYHOST_NARROW)])
        assert bundle.kind == "choice" and bundle.control() is None
        view = json.loads(build_campaign_user_message(bundle).split("\n", 1)[1])
        assert list(view)[:5] == ["campaign", "kind", "kind_means", "pairs", "bars"]
        assert view["pairs"][0]["ahead"] == arm_at(bundle, TOYHOST_WIDE)
        assert view["bars"][0]["status"] == "missed" and "bar_warning" in view

    async def test_no_clear_winner_recommends_no_arm(self) -> None:
        bundle = _choice(pairs=[_pair(_choice(), ahead=None)])
        analysis = await _generate(bundle, _recommending(bundle, None))
        assert analysis.document.recommended() is None
        assert any(
            getattr(block, "text", "") == "No clear winner: the writer recommends no arm."
            for block in build_report(analysis).blocks
        )

    async def test_picking_the_arm_shown_ahead_is_kept_and_its_key_stored(self) -> None:
        bundle = _choice(pairs=[_pair(_choice(), ahead=TOYHOST_WIDE)])
        analysis = await _generate(bundle, _recommending(bundle, TOYHOST_WIDE))
        assert analysis.recommended_variant_key == _variant(bundle, TOYHOST_WIDE)

    async def test_picking_an_arm_shown_behind_without_saying_why_is_refused(self) -> None:
        bundle = _choice(pairs=[_pair(_choice(), ahead=TOYHOST_WIDE)])
        with pytest.raises(SoundnessRefusal) as refused:
            await _generate(bundle, _recommending(bundle, TOYHOST_NARROW))
        message = str(refused.value)
        assert f"which is shown behind another arm on ['{bundle.labels[0]}/{arm_at(bundle, TOYHOST_WIDE)}']" in message
        assert "`behind_on`" in message

    async def test_picking_an_arm_shown_behind_with_why_is_kept(self) -> None:
        bundle = _choice(pairs=[_pair(_choice(), ahead=TOYHOST_WIDE)])
        behind = {"aspect": bundle.labels[0], "ahead_arm": arm_at(bundle, TOYHOST_WIDE), "why": "It costs a third."}
        analysis = await _generate(bundle, _recommending(bundle, TOYHOST_NARROW, behind_on=[behind]))
        (said,) = analysis.document.recommendation[0].behind_on
        assert said.why == "It costs a third."

    async def test_a_behind_on_naming_no_pair_code_read_is_refused(self) -> None:
        bundle = _choice(pairs=[_pair(_choice(), ahead=None)])
        behind = {"aspect": bundle.labels[0], "ahead_arm": arm_at(bundle, TOYHOST_WIDE), "why": "It costs a third."}
        with pytest.raises(
            SoundnessRefusal, match="in `behind_on`, by aspect and arm, and code raised no such warning"
        ):
            await _generate(bundle, _recommending(bundle, TOYHOST_NARROW, behind_on=[behind]))

    async def test_a_pick_resting_on_no_finding_is_refused(self) -> None:
        bundle = _choice(pairs=[_pair(_choice(), ahead=TOYHOST_WIDE)])
        with pytest.raises(SoundnessRefusal, match="a pick of an arm rests on the findings that show it"):
            await _generate(bundle, _recommending(bundle, TOYHOST_WIDE, rests_on=[]))

    async def test_a_missed_bar_on_the_pick_must_be_addressed(self) -> None:
        bare = _choice()
        bundle = _choice(pairs=[_pair(bare, ahead=TOYHOST_WIDE)], bars=[_bar(bare, TOYHOST_WIDE)])
        key = f"{bundle.labels[0]}/field_accuracy"
        with pytest.raises(SoundnessRefusal, match=rf"which missed bar\(s\) \['{key}'\]"):
            await _generate(bundle, _recommending(bundle, TOYHOST_WIDE))
        missed = {"aspect": bundle.labels[0], "measure": "field_accuracy", "why": "The bar is next quarter's."}
        analysis = await _generate(bundle, _recommending(bundle, TOYHOST_WIDE, bars_missed=[missed]))
        assert analysis.document.recommendation[0].bars_missed[0].measure == "field_accuracy"

    async def test_a_bar_missed_by_another_arm_needs_no_word(self) -> None:
        bare = _choice()
        bundle = _choice(pairs=[_pair(bare, ahead=TOYHOST_WIDE)], bars=[_bar(bare, TOYHOST_NARROW)])
        analysis = await _generate(bundle, _recommending(bundle, TOYHOST_WIDE))
        assert analysis.document.recommendation[0].bars_missed == []


# --- a change ------------------------------------------------------------------------------------------------------


class TestAChangeAlwaysRecommends:
    async def test_a_change_recommending_nothing_is_refused(self) -> None:
        bundle = campaign_bundle(_aspect(control=True))
        assert bundle.kind == "change"
        with pytest.raises(SoundnessRefusal, match="compares candidates against a control, so it recommends one"):
            await _generate(bundle, _recommending(bundle, None))

    async def test_a_switch_shown_better_only_on_an_aspect_asked_late_is_refused(self) -> None:
        """Aspect A asked its question in time and B after its runs; the candidate is better only on B."""
        timely = _without_gains(_aspect(control=True))
        late = timely.model_copy(
            update={
                "aspect": "late",
                "reading_scope": timely.reading_scope.model_copy(update={"questions_declared": False}),
                "multiple_comparisons": _aspect(control=True).multiple_comparisons,
            }
        )
        bare = campaign_bundle(timely, late)
        bundle = campaign_bundle(timely, late, readings=[_late_better(bare)])
        payload = _recommending(bundle, TOYHOST_WIDE)
        payload["aspect_summaries"].append({**payload["aspect_summaries"][0], "aspect": "late"})
        with pytest.raises(SoundnessRefusal) as refused:
            await _generate(bundle, payload)
        message = str(refused.value)
        assert "on the aspects asked in time" in message
        assert "It is shown better only on 'late', whose question was asked after its runs" in message


def _without_gains(aspect: AnalysisContextBundle) -> AnalysisContextBundle:
    families = [
        family.model_copy(
            update={
                "comparisons": [
                    c.model_copy(update={"verdict": "not_separated"}) if c.verdict == "improved" else c
                    for c in family.comparisons
                ]
            }
        )
        for family in aspect.multiple_comparisons.families
    ]
    return aspect.model_copy(
        update={"multiple_comparisons": aspect.multiple_comparisons.model_copy(update={"families": families})}
    )


def _late_better(bundle: CampaignBundle) -> AspectReading:
    return AspectReading(
        aspect="late",
        arm=_variant(bundle, TOYHOST_WIDE),
        arm_label=arm_at(bundle, TOYHOST_WIDE),
        change=20.0,
        gain_interval=(5.0, 35.0),
        gain_confidence=0.95,
        drop_interval=None,
        drop_confidence=None,
        gain_p=0.01,
        gain_p_holm=0.01,
        drop_p=None,
        drop_p_holm=None,
        reading="better",
        margin=None,
        control_cases=12,
        arm_cases=12,
        control_passed=24,
        control_trials=36,
        arm_passed=31,
        arm_trials=36,
        projection=None,
        reason="the wide width passed 20 points more trials",
        confirmatory=False,
    )


# --- titles name arms by their labels too (#829) -------------------------------------------------------------------


async def test_a_title_naming_an_arm_by_its_person_given_label_is_held_to_its_evidence() -> None:
    aspect = _aspect(control=True)
    narrow, wide = (_variant(aspect, level) for level in (TOYHOST_NARROW, TOYHOST_WIDE))
    named = [
        arm.model_copy(update={"label": "baseline" if arm.variant_key == narrow else "fast"})
        for arm in arms_of([aspect])
    ]
    bundle = campaign_bundle(aspect, arms=named)
    payload = memo_payload(bundle)
    payload["findings"][0]["title"] = "fast slows every document down"
    payload["findings"][0]["evidence"] = [
        {"cell": alias_at(bundle, TOYHOST_NARROW), "measure_id": "total_ms", "reading": "measure"}
    ]
    assert payload["recommendation"][0]["arm"] == "baseline", wide
    with pytest.raises(SoundnessRefusal, match=r'findings\[0\]\.title names "fast"'):
        await _generate(bundle, payload)


# --- the page's fingerprint is the bundle served beside it ---------------------------------------------------------


async def test_the_page_states_the_fingerprint_of_the_bundle_it_was_laid_out_over() -> None:
    bundle = campaign_bundle(_aspect(control=True))
    analysis = await _generate(bundle, memo_payload(bundle))
    report = build_report(analysis)
    assert report.source.bundle_fingerprint == analysis_bundle(analysis).fingerprint()
    assert report.source.bundle_fingerprint != analysis.generation.bundle_fingerprint, (
        "the generation's is the whole campaign bundle's"
    )
