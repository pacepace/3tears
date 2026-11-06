"""The memo recommends one arm, pinned to its findings, and the recommendation answers to code's readings.

The report's first screen is code's, so what the generator still checks of a memo is structure: the recommendation
names an arm the campaign holds and the findings it rests on, each aspect is summarised once, a finding names only
trials the campaign holds, a campaign that asks no question recommends no candidate — which is what
printed "Keep the current model — adopted" over the G2 campaign before a recommendation could name only the
control there — and a candidate needs an upside code's readings separated. Each refusal buys the one repair round like
any other.

Driven through :func:`~threetears.evals.analysis.generator.generate_analysis` over the toy host's campaign, which
declares a question and the narrow chunk width as its control; the exploratory case is the same bundle with no question
declared before its run.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from packages.evals.tests.aspect_support import with_control
from packages.evals.tests.campaign_bundle_support import campaign_bundle
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
from threetears.evals.analysis.campaign_readings import AspectReading
from threetears.evals.analysis.cells import variant_of_cell_ref
from threetears.evals.analysis.errors import SoundnessRefusal
from threetears.evals.kernel.numbers import format_number
from threetears.evals.analysis.references import resolve_reading
from threetears.evals.kernel.campaign import EvalAnalysis
from threetears.evals.schema.models import utc_now_iso


#: A time after every run of the toy corpus: a question asked then was asked after its runs.
_AFTER_THE_RUNS = "2099-01-01T00:00:00+00:00"


def _aspect(*, asked_at: str | None = None) -> AnalysisContextBundle:
    """The toy campaign's bundle, its narrow width declared the control; ``asked_at`` re-dates its question."""
    profile = toyhost_profile()
    campaign, storage = toyhost_campaign(profile=profile)
    bundle = assemble_context_bundle(campaign, storage=storage, profile=profile)
    control = variant_of_cell_ref(cell_at(bundle, TOYHOST_NARROW))
    # Validated, not copied: a control arm is stamped with the identity version it was addressed under.
    campaign = with_control(campaign, control)
    if asked_at is not None:
        aspects = [aspect.model_copy(update={"asked_at": asked_at}) for aspect in campaign.aspects]
        campaign = campaign.model_copy(update={"declared_at": asked_at, "aspects": aspects})
    return assemble_context_bundle(campaign, storage=storage, profile=profile)


def _bundle(**fields: Any) -> CampaignBundle:
    """The toy campaign as its writer reads it."""
    return campaign_bundle(_aspect(), **fields)


def _exploratory(bundle: CampaignBundle) -> CampaignBundle:
    """The campaign as one that declares no question reads."""
    (aspect,) = bundle.aspects
    scope = aspect.reading_scope.model_copy(update={"questions_declared": False})
    return bundle.model_copy(update={"aspects": [aspect.model_copy(update={"reading_scope": scope})]})


def _without_gains(bundle: CampaignBundle) -> CampaignBundle:
    """The campaign with every comparison the wide arm improved on read as not separated: only its losses separate."""
    (aspect,) = bundle.aspects
    families = [
        family.model_copy(
            update={
                "comparisons": [
                    comparison.model_copy(update={"verdict": "not_separated"})
                    if comparison.verdict == "improved"
                    else comparison
                    for comparison in family.comparisons
                ]
            }
        )
        for family in aspect.multiple_comparisons.families
    ]
    comparisons = aspect.multiple_comparisons.model_copy(update={"families": families})
    return bundle.model_copy(update={"aspects": [aspect.model_copy(update={"multiple_comparisons": comparisons})]})


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


def _recommending(bundle: CampaignBundle, level: int) -> dict[str, Any]:
    payload = memo_payload(bundle)
    payload["recommendation"][0]["arm"] = arm_at(bundle, level)
    return payload


def _better(bundle: CampaignBundle, level: int) -> AspectReading:
    """Code's reading that the arm at ``level`` is shown better than the control on the toy aspect."""
    variant = variant_of_cell_ref(cell_at(bundle, level))
    assert variant is not None
    return AspectReading(
        aspect=bundle.labels[0],
        arm=variant,
        arm_label=arm_at(bundle, level),
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
        control_cases=2,
        arm_cases=2,
        control_passed=3,
        control_trials=6,
        arm_passed=6,
        arm_trials=6,
        projection=None,
        reason="the wide width passed every trial the narrow one did and more",
        confirmatory=True,
    )


class TestAnExploratoryCampaignKeepsTheControl:
    async def test_a_candidate_recommended_in_an_exploratory_campaign_is_refused(self) -> None:
        bundle = _exploratory(_bundle())
        with pytest.raises(SoundnessRefusal) as refused:
            await _generate(bundle, _recommending(bundle, TOYHOST_WIDE))
        message = str(refused.value)
        assert "a candidate, and this campaign asks no question of any aspect" in message
        assert f"Recommend the control, {arm_at(bundle, TOYHOST_NARROW)!r}" in message

    async def test_keeping_the_control_in_an_exploratory_campaign_is_kept(self) -> None:
        bundle = _exploratory(_bundle())
        analysis = await _generate(bundle, _recommending(bundle, TOYHOST_NARROW))
        assert analysis.document.recommendation[0].arm == arm_at(bundle, TOYHOST_NARROW)

    async def test_with_a_question_declared_before_the_run_the_control_may_be_kept(self) -> None:
        bundle = _bundle()
        assert bundle.aspects[0].reading_scope.questions_declared
        analysis = await _generate(bundle, _recommending(bundle, TOYHOST_NARROW))
        assert analysis.document.recommendation[0].arm == arm_at(bundle, TOYHOST_NARROW)


class TestACandidateNeedsASeparatedUpside:
    async def test_a_candidate_separated_only_by_its_losses_is_refused(self) -> None:
        bundle = _without_gains(_bundle())
        with pytest.raises(SoundnessRefusal) as refused:
            await _generate(bundle, _recommending(bundle, TOYHOST_WIDE))
        message = str(refused.value)
        assert f"the recommendation names {arm_at(bundle, TOYHOST_WIDE)!r}, but" in message
        assert "it separated from the control only by being worse" in message
        assert "on the aspects the question asks about" in message and "better on none of them" in message
        assert "Recommend the control" in message

    async def test_a_candidate_that_improved_on_a_comparison_is_kept_as_a_trade_off(self) -> None:
        bundle = _bundle()
        analysis = await _generate(bundle, _recommending(bundle, TOYHOST_WIDE))
        assert analysis.document.recommendation[0].arm == arm_at(bundle, TOYHOST_WIDE)

    async def test_a_candidate_on_an_aspect_asked_after_its_readings_is_kept_as_one_asked_before_them(self) -> None:
        """Owner ruling (2026-10-21): disclosed, not enforced. An aspect asked after its readings counts as asked, so
        the upside that keeps a candidate in time keeps it here too; the bundle says when each criterion was set."""
        bundle = campaign_bundle(_aspect(asked_at=_AFTER_THE_RUNS))
        (aspect,) = bundle.aspects
        assert aspect.question_declared and aspect.reading_scope.late_criteria != []
        analysis = await _generate(bundle, _recommending(bundle, TOYHOST_WIDE))
        assert analysis.document.recommendation[0].arm == arm_at(bundle, TOYHOST_WIDE)

    async def test_a_candidate_code_reads_better_is_kept(self) -> None:
        bundle = _without_gains(_bundle())
        bundle = _without_gains(_bundle(readings=[_better(bundle, TOYHOST_WIDE)]))
        analysis = await _generate(bundle, _recommending(bundle, TOYHOST_WIDE))
        assert analysis.document.recommendation[0].arm == arm_at(bundle, TOYHOST_WIDE)

    async def test_an_arm_the_campaign_does_not_hold_is_refused_naming_the_arms(self) -> None:
        bundle = _bundle()
        payload = memo_payload(bundle)
        payload["recommendation"][0]["arm"] = "the fast one"
        with pytest.raises(SoundnessRefusal) as refused:
            await _generate(bundle, payload)
        message = str(refused.value)
        assert "recommendation.arm is 'the fast one', which is no arm's label" in message
        assert all(arm.label in message for arm in bundle.arms)


class TestTheRecommendationIsPinned:
    async def test_a_switch_resting_on_no_finding_is_refused(self) -> None:
        bundle = _bundle()
        payload = _recommending(bundle, TOYHOST_WIDE)
        payload["recommendation"][0]["rests_on"] = []
        with pytest.raises(
            SoundnessRefusal, match=r"names the findings it rests on.*recommendation\.rests_on is empty"
        ):
            await _generate(bundle, payload)

    async def test_keeping_the_control_may_rest_on_no_finding(self) -> None:
        bundle = _bundle()
        payload = _recommending(bundle, TOYHOST_NARROW)
        payload["recommendation"][0]["rests_on"] = []
        analysis = await _generate(bundle, payload)
        assert analysis.document.recommendation[0].rests_on == []

    async def test_a_recommendation_resting_on_a_finding_that_is_not_there_is_refused(self) -> None:
        bundle = _bundle()
        payload = memo_payload(bundle)
        payload["recommendation"][0]["rests_on"] = [3]
        with pytest.raises(SoundnessRefusal, match=r"recommendation\.rests_on names finding position\(s\) \[3\]"):
            await _generate(bundle, payload)


class TestEachAspectIsSummarisedOnce:
    async def test_an_aspect_with_no_summary_is_refused(self) -> None:
        bundle = _bundle()
        payload = memo_payload(bundle)
        payload["aspect_summaries"] = []
        with pytest.raises(SoundnessRefusal, match=rf"aspect\(s\) \['{bundle.labels[0]}'\] have no summary"):
            await _generate(bundle, payload)

    async def test_a_summary_of_an_aspect_the_campaign_lacks_is_refused(self) -> None:
        bundle = _bundle()
        payload = memo_payload(bundle)
        payload["aspect_summaries"].append({**payload["aspect_summaries"][0], "aspect": "routing"})
        with pytest.raises(SoundnessRefusal, match=r"summarise \['routing'\], which the campaign does not hold"):
            await _generate(bundle, payload)

    async def test_an_aspect_summarised_twice_is_refused(self) -> None:
        bundle = _bundle()
        payload = memo_payload(bundle)
        payload["aspect_summaries"].append(payload["aspect_summaries"][0])
        with pytest.raises(SoundnessRefusal, match="are summarised more than once"):
            await _generate(bundle, payload)


class TestTheSummaryReadsAcrossEveryAspect:
    """Every kind's memo carries one summary across its aspects, held to the reference rules a finding's prose is."""

    @pytest.mark.parametrize("written", ["", "   \n "])
    async def test_an_empty_summary_is_refused_naming_it(self, written: str) -> None:
        bundle = _bundle()
        payload = memo_payload(bundle)
        payload["summary"] = written
        with pytest.raises(SoundnessRefusal, match=r"`summary` is empty"):
            await _generate(bundle, payload)

    async def test_a_memo_with_no_summary_is_refused_as_off_the_contract(self) -> None:
        bundle = _bundle()
        payload = memo_payload(bundle)
        del payload["summary"]
        with pytest.raises(SoundnessRefusal, match=r"does not match the analysis contract(.|\n)*summary"):
            await _generate(bundle, payload)

    async def test_a_figure_reference_in_the_summary_renders_as_code_s_number(self) -> None:
        bundle = _bundle()
        payload = memo_payload(bundle)
        narrow = alias_at(bundle, TOYHOST_NARROW)
        payload["summary"] = f"Over {{{{{narrow}|total_ms|measure|n}}}} documents the narrow width is faster."
        analysis = await _generate(bundle, payload)
        (aspect,) = analysis.aspects
        resolved = resolve_reading(aspect.decision_surface, cell_at(bundle, TOYHOST_NARROW), "total_ms", "measure")
        assert analysis.document.summary == f"Over {format_number(resolved.n)} documents the narrow width is faster."

    async def test_a_reference_in_the_summary_naming_nothing_is_refused_where_it_sits(self) -> None:
        bundle = _bundle()
        payload = memo_payload(bundle)
        payload["summary"] = "The narrow width is faster by {{c99|total_ms|measure|mean}}."
        with pytest.raises(SoundnessRefusal, match=r"summary: "):
            await _generate(bundle, payload)


async def test_a_finding_naming_a_trial_the_campaign_does_not_hold_is_refused() -> None:
    bundle = _bundle()
    payload = memo_payload(bundle)
    payload["findings"][0]["trials"] = ["T999"]
    with pytest.raises(SoundnessRefusal, match=r"findings\[0\]\.trials names 'T999'"):
        await _generate(bundle, payload)


async def test_a_finding_naming_trials_the_campaign_holds_keeps_them_and_each_aspect_s_bundle_is_stored() -> None:
    bundle = _bundle()
    payload = memo_payload(bundle)
    payload["findings"][0]["trials"] = ["T1"]
    analysis = await _generate(bundle, payload)
    assert analysis.document.findings[0].trials == ["T1"]
    assert bundle.trial("T1") == (bundle.labels[0], bundle.aspects[0].trials[0].trial_id)
    (aspect,) = analysis.aspects
    assert AnalysisContextBundle.from_dict(dict(aspect.bundle)).fingerprint() == bundle.aspects[0].fingerprint()
    assert aspect.bundle_fingerprint == bundle.aspects[0].fingerprint()
    assert (
        CampaignBundle.from_frame(analysis.campaign_bundle, [bundle.aspects[0]]).fingerprint() == bundle.fingerprint()
    )
    assert analysis.generation.bundle_fingerprint == bundle.fingerprint()


async def test_the_memo_carries_no_headline_summary_or_chart() -> None:
    bundle = _bundle()
    payload = memo_payload(bundle)
    payload["headline"] = "keep the narrow width"
    with pytest.raises(SoundnessRefusal, match="headline"):
        await _generate(bundle, payload)
