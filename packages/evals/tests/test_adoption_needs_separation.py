"""A recommended candidate is one the statistics separated from the control in its favour.

Recommending a candidate says to switch to it, so it claims that arm beat the control. Code's reading of each aspect
and the aspect's ``multiple_comparisons`` are where the bundle says whether it did, and a recommendation of a candidate
nothing separated is refused, repairably: the model gets one round to keep the control. A trade-off is still a
recommendation a reader may take — better on one reading and worse on another, or cheaper and not separated on quality
— so one ``improved`` verdict licenses it.

Driven through :func:`~threetears.evals.analysis.generator.generate_analysis` over the toy host's
campaign. Where the toy campaign declares its narrow width the control, the bundle assembles a
real family (the wide width improves accuracy and regresses cost and latency); the other verdict
mixes are that family with its verdicts replaced, the way the projection tests replace a
declaration's questions.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_NARROW, TOYHOST_WIDE, toyhost_campaign
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.toyhost_memo import (
    MODEL,
    PROMPT,
    PROMPT_ID,
    FixturedClient,
    FixturedCompletion,
    arm_at,
    cell_at,
    memo_payload,
)
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from threetears.evals.analysis.bundle.assemble import assemble_context_bundle
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, ComparisonVerdict
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.cells import variant_of_cell_ref
from threetears.evals.analysis.errors import SoundnessRefusal
from threetears.evals.analysis.generator import generate_analysis
from threetears.evals.analysis.report import build_report
from threetears.evals.analysis.report.model import DisclosureBlock, TableBlock, TextBlock
from threetears.evals.kernel.surface import GuardrailCell, GuardrailCheck, GuardrailDecision, GuardrailReadings
from threetears.evals.kernel.campaign import EvalAnalysis
from threetears.evals.schema.models import utc_now_iso
from packages.evals.tests.aspect_support import with_aspect, with_control

_UNTESTED_REASON = "fewer than two cases carry this reading on a side"


class _MemoSequence(FixturedClient):
    """Hands back one prepared memo per call, in order: the first output, then its repair."""

    def __init__(self, *bodies: str) -> None:
        super().__init__(bodies[0])
        self.bodies = list(bodies)

    async def generate(
        self, *, system: str, user: str, response_format: Any = None, tools: Any = None
    ) -> FixturedCompletion:
        self.completion = FixturedCompletion(content=self.bodies[len(self.calls)])
        return await super().generate(system=system, user=user, response_format=response_format, tools=tools)


def _toy_bundle(*, control: bool) -> AnalysisContextBundle:
    """The toy campaign's bundle, with its narrow width declared the control or with no control at all."""
    profile = toyhost_profile()
    campaign, storage = toyhost_campaign(profile=profile)
    bundle = assemble_context_bundle(campaign, storage=storage, profile=profile)
    if not control:
        return bundle
    controlled = with_control(campaign, variant_of_cell_ref(cell_at(bundle, TOYHOST_NARROW)))
    return assemble_context_bundle(controlled, storage=storage, profile=profile)


def _with_verdicts(
    bundle: AnalysisContextBundle, verdicts: dict[str, ComparisonVerdict], *, otherwise: ComparisonVerdict
) -> AnalysisContextBundle:
    """The bundle with each comparison's verdict replaced: ``verdicts`` by reading name, ``otherwise`` the rest."""
    families = []
    for family in bundle.multiple_comparisons.families:
        comparisons = []
        for comparison in family.comparisons:
            verdict = verdicts.get(comparison.name, otherwise)
            untested = verdict == "untested"
            comparisons.append(
                comparison.model_copy(
                    update={
                        "verdict": verdict,
                        "p_adjusted": None if untested else (comparison.p_adjusted or 0.5),
                        "untested_reason": _UNTESTED_REASON if untested else None,
                    }
                )
            )
        families.append(family.model_copy(update={"comparisons": comparisons}))
    return bundle.model_copy(
        update={"multiple_comparisons": bundle.multiple_comparisons.model_copy(update={"families": families})}
    )


def _memo(bundle: CampaignBundle, level: int) -> str:
    """The toy memo recommending the arm at ``level``."""
    payload = memo_payload(bundle)
    payload["recommendation"][0]["arm"] = arm_at(bundle, level)
    return json.dumps(payload)


async def _generate(bundle: AnalysisContextBundle | CampaignBundle, client: FixturedClient) -> EvalAnalysis:
    campaign = bundle if isinstance(bundle, CampaignBundle) else campaign_bundle(bundle)
    analysis, _insights = await generate_analysis(
        campaign,
        prompt=PROMPT,
        model=MODEL,
        client=client,
        prompt_id=PROMPT_ID,
        bundle_assembled_at=utc_now_iso(),
        profile=toyhost_profile(),
    )
    return analysis


async def _adopt(bundle: AnalysisContextBundle, level: int) -> EvalAnalysis:
    """Generate over a memo recommending the arm at ``level``, in one call or the refusal."""
    campaign = campaign_bundle(bundle)
    client = FixturedClient(_memo(campaign, level))
    analysis = await _generate(campaign, client)
    assert len(client.calls) == 1, "a recommendation the statistics support is accepted first time"
    return analysis


class TestAnAdoptionTheStatisticsSupportIsAccepted:
    async def test_a_trade_off_better_on_one_reading_and_worse_on_others_is_adopted(self):
        """The assembled family as it stands: the wide width improves accuracy and regresses cost and latency."""
        bundle = _toy_bundle(control=True)
        verdicts = {c.name: c.verdict for f in bundle.multiple_comparisons.families for c in f.comparisons}
        assert verdicts["field_accuracy"] == "improved" and verdicts["production_replicating_cost"] == "regressed"

        analysis = await _adopt(bundle, TOYHOST_WIDE)

        assert analysis.document.recommendation[0].arm == arm_at(campaign_bundle(bundle), TOYHOST_WIDE)

    async def test_an_arm_separated_only_on_cost_is_adopted(self):
        """Cheaper, and not separated on quality: a trade-off a reader may take."""
        bundle = _with_verdicts(
            _toy_bundle(control=True), {"production_replicating_cost": "improved"}, otherwise="not_separated"
        )
        analysis = await _adopt(bundle, TOYHOST_WIDE)
        assert analysis.document.recommendation[0].arm == arm_at(campaign_bundle(bundle), TOYHOST_WIDE)

    async def test_adopting_the_control_needs_no_separation(self):
        bundle = _with_verdicts(_toy_bundle(control=True), {}, otherwise="not_separated")
        await _adopt(bundle, TOYHOST_NARROW)


class TestAnAdoptionNoReadingSeparatedIsRefused:
    async def test_an_arm_no_reading_separated_is_refused_and_told_what_to_do(self):
        bundle = _with_verdicts(_toy_bundle(control=True), {}, otherwise="not_separated")
        with pytest.raises(SoundnessRefusal) as refused:
            await _adopt(bundle, TOYHOST_WIDE)
        message = str(refused.value)
        assert f"the recommendation names {arm_at(campaign_bundle(bundle), TOYHOST_WIDE)!r}, but" in message
        assert "no reading separated it from the control" in message
        assert f"Recommend the control, {arm_at(campaign_bundle(bundle), TOYHOST_NARROW)!r}," in message

    async def test_an_arm_separated_only_by_regressing_is_refused(self):
        """Its only separation is evidence against it: there is no upside a trade-off could buy."""
        bundle = _with_verdicts(
            _toy_bundle(control=True), {"production_replicating_cost": "regressed"}, otherwise="not_separated"
        )
        with pytest.raises(SoundnessRefusal, match=r"only by being worse \(production_replicating_cost regressed\)"):
            await _adopt(bundle, TOYHOST_WIDE)

    async def test_the_repair_round_keeps_the_control_and_the_analysis_is_stored(self):
        bundle = campaign_bundle(_with_verdicts(_toy_bundle(control=True), {}, otherwise="not_separated"))
        client = _MemoSequence(_memo(bundle, TOYHOST_WIDE), _memo(bundle, TOYHOST_NARROW))

        analysis = await _generate(bundle, client)

        assert len(client.calls) == 2
        refusal = analysis.generation.repaired_refusal
        assert analysis.generation.repair_attempts == 1 and refusal is not None
        assert f"the recommendation names {arm_at(bundle, TOYHOST_WIDE)!r}" in refusal
        assert refusal in client.calls[1]["user"]
        assert analysis.document.recommendation[0].arm == arm_at(bundle, TOYHOST_NARROW)


class TestAnArmNoTestCouldReachHasNoSeparationToAdoptOn:
    """No family, no comparison naming the arm, or only untested ones: nothing separated it, so it is refused."""

    async def test_a_campaign_naming_no_control_compared_nothing_and_its_recommendation_is_not_refused(self):
        """With no control there is no arm to keep and no switch from one: nothing could be tested either way."""
        bundle = _toy_bundle(control=False)
        assert bundle.multiple_comparisons.withheld is not None
        campaign = campaign_bundle(bundle)
        assert campaign.control() is None
        analysis = await _adopt(bundle, TOYHOST_NARROW)
        assert analysis.document.recommendation[0].arm == arm_at(campaign, TOYHOST_NARROW)

    async def test_an_arm_whose_every_comparison_is_untested_is_refused_with_why(self):
        bundle = _with_verdicts(_toy_bundle(control=True), {}, otherwise="untested")
        with pytest.raises(
            SoundnessRefusal, match=f"no test against the control was possible on any reading \\({_UNTESTED_REASON}\\)"
        ):
            await _adopt(bundle, TOYHOST_WIDE)

    async def test_an_arm_no_comparison_names_is_refused(self):
        bundle = _toy_bundle(control=True)
        families = [family.model_copy(update={"comparisons": []}) for family in bundle.multiple_comparisons.families]
        bundle = bundle.model_copy(
            update={"multiple_comparisons": bundle.multiple_comparisons.model_copy(update={"families": families})}
        )
        with pytest.raises(SoundnessRefusal, match="no comparison in `multiple_comparisons` tests it"):
            await _adopt(bundle, TOYHOST_WIDE)


# --- guardrails: a breached guardrail refuses the adoption, an undecided one is stated on it ----------------


def _with_guardrail(bundle: AnalysisContextBundle, decision: GuardrailDecision) -> AnalysisContextBundle:
    """The bundle with one judged guardrail decided ``decision`` for the wide width against the control.

    Its cells are those of the family's first comparison, so the check names the arm the memo adopts.
    """
    comparison = bundle.multiple_comparisons.families[0].comparisons[0]
    control, contrast = comparison.control, comparison.contrast
    check = GuardrailCheck(
        reading="judged",
        name="boundary.correct",
        higher_is_better=True,
        control=GuardrailCell(
            variant_key=control.variant_key, apparatus_class_id=control.apparatus_class_id, n_cases=4, mean=1.0
        ),
        contrast=GuardrailCell(
            variant_key=contrast.variant_key, apparatus_class_id=contrast.apparatus_class_id, n_cases=4, mean=0.5
        ),
        test="paired",
        delta=-0.5,
        interval=(-0.9, -0.1) if decision == "breached" else (-0.9, 0.4),
        interval_basis="t",
        margin=0.0,
        margin_declared=False,
        decision=decision,
        undecided_reason="the interval reaches both sides of 0" if decision == "undecided" else None,
    )
    guardrails = GuardrailReadings(dimensions=["boundary.correct"], checks=[check])
    return bundle.model_copy(update={"guardrails": guardrails})


class TestAGuardrailOnTheAdoptedArm:
    async def test_a_breached_guardrail_refuses_the_adoption_whatever_the_arm_gained(self):
        """The wide width improves accuracy (a real separation), and still cannot be adopted over a breach."""
        bundle = _with_guardrail(_toy_bundle(control=True), "breached")
        with pytest.raises(SoundnessRefusal) as refused:
            await _adopt(bundle, TOYHOST_WIDE)
        message = str(refused.value)
        assert "breached the guardrail boundary.correct" in message
        assert "no gain elsewhere pays for a guardrail; recommend the control" in message

    async def test_an_undecided_guardrail_lets_the_adoption_stand_and_the_report_states_it(self):
        bundle = _with_guardrail(_toy_bundle(control=True), "undecided")
        analysis = await _adopt(bundle, TOYHOST_WIDE)

        assert analysis.aspects[0].decision_surface.guardrails == bundle.guardrails, "frozen with the analysis"
        report = build_report(analysis)
        assert any(
            isinstance(b, DisclosureBlock) and "not known to be safe" in b.text and "boundary.correct" in b.text
            for b in report.blocks
        )
        assert any(isinstance(b, TableBlock) and b.name == "guardrails" for b in report.blocks)

    async def test_a_held_guardrail_adds_nothing_to_the_decision(self):
        bundle = _with_guardrail(_toy_bundle(control=True), "held").model_copy()
        held = bundle.guardrails.checks[0].model_copy(update={"interval": (-0.1, 0.4), "decision": "held"})
        bundle = bundle.model_copy(update={"guardrails": bundle.guardrails.model_copy(update={"checks": [held]})})
        report = build_report(await _adopt(bundle, TOYHOST_WIDE))
        (facts,) = [b for b in report.blocks if isinstance(b, TableBlock) and b.name == "recommendation_facts"]
        assert "Guardrails" not in {column.header for column in facts.columns}
        assert not any(
            isinstance(b, DisclosureBlock) and "boundary.correct" in b.text and "undecided" in b.text
            for b in report.blocks
        )


# --- exploratory: a finding resting on readings no declared question asked about is labelled ----------------


def _toy_bundle_asking(axes: list[str]) -> AnalysisContextBundle:
    """The toy campaign's bundle, its one question scoped to ``axes``."""
    profile = toyhost_profile()
    campaign, storage = toyhost_campaign(profile=profile)
    campaign = with_aspect(campaign, design={"merit_axes": axes})
    return assemble_context_bundle(campaign, storage=storage, profile=profile)


def _finding_scope(analysis: EvalAnalysis) -> str | None:
    report = build_report(analysis)
    (title,) = [b for b in report.blocks if isinstance(b, TextBlock) and b.role == "finding_title"]
    return {fact.name: fact.value for fact in title.facts}.get("Scope")


class TestAFindingNoQuestionAskedAboutIsExploratory:
    """The toy memo's one finding rests on the arms' latency (`total_ms`)."""

    async def test_a_question_on_quality_leaves_the_latency_finding_exploratory(self):
        bundle = _toy_bundle_asking(["quality"])
        assert "total_ms" in bundle.reading_scope.exploratory_measures
        analysis = await _generate(bundle, FixturedClient(json.dumps(memo_payload(campaign_bundle(bundle)))))
        scope = _finding_scope(analysis)
        assert scope is not None and scope.startswith("exploratory")

    async def test_a_question_on_latency_asks_about_it_and_no_label_is_added(self):
        bundle = _toy_bundle_asking(["latency"])
        analysis = await _generate(bundle, FixturedClient(json.dumps(memo_payload(campaign_bundle(bundle)))))
        assert _finding_scope(analysis) is None
        report = build_report(analysis)
        assert not [b for b in report.blocks if getattr(b, "source", None) == "scope"], "said only with no question"
