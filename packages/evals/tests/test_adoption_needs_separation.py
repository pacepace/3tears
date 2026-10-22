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
from packages.evals.tests.campaign_bundle_support import arms_of, campaign_bundle
from threetears.evals.analysis.bundle.assemble import assemble_context_bundle
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, ComparisonVerdict
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.campaign_readings import AspectReading, FloorState, FloorStatus
from threetears.evals.kernel.declaration import Floor, RateDrop
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

    async def test_a_measurement_refuses_a_recommendation(self):
        """One arm alone compares nothing: no control to keep and no arm to switch to, so nothing is recommended."""
        bundle = _toy_bundle(control=False)
        assert bundle.multiple_comparisons.withheld is not None
        (narrow,) = [
            arm for arm in arms_of([bundle]) if arm.variant_key == variant_of_cell_ref(cell_at(bundle, TOYHOST_NARROW))
        ]
        campaign = campaign_bundle(bundle, arms=[narrow])
        assert campaign.kind == "measurement"
        with pytest.raises(SoundnessRefusal, match="measures one arm, so there is no arm to choose"):
            await _generate(campaign, FixturedClient(_memo(campaign, TOYHOST_NARROW)))

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


# --- no worse: a candidate held on every floor of every asked aspect is recommendable --------------------------


def _aspect(label: str | None = None, *, asked: bool = True, guardrail: bool = False) -> AnalysisContextBundle:
    """The control-declared toy aspect, nothing separated: relabelled, asked or not, its guardrail breached or not."""
    bundle = _with_verdicts(_toy_bundle(control=True), {}, otherwise="not_separated")
    if guardrail:
        bundle = _with_guardrail(bundle, "breached")
    if label is not None:
        bundle = bundle.model_copy(update={"aspect": label})
    if asked:
        return bundle
    scope = bundle.reading_scope.model_copy(update={"questions_declared": False})
    return bundle.model_copy(update={"reading_scope": scope})


def _with_floors(
    aspects: list[AnalysisContextBundle], floors: list[tuple[str, FloorState]], *, arm: str | None = None
) -> CampaignBundle:
    """The campaign over ``aspects``, declaring one floor per (aspect label, status), each read so for the wide width.

    ``arm`` reads them for another variant key instead, so the wide width holds no reading of any.
    """
    campaign = campaign_bundle(*aspects)
    (wide,) = [candidate for candidate in campaign.arms if candidate.label == arm_at(campaign, TOYHOST_WIDE)]
    declared = [
        Floor(aspect=label, clause=RateDrop(max_points=15.0, reading=f"check_{index}"))
        for index, (label, _status) in enumerate(floors)
    ]
    statuses = [
        FloorStatus(
            floor=floor.key,
            aspect=floor.aspect,
            arm=arm or wide.variant_key,
            arm_label=wide.label,
            status=status,
            reading=f"check_{index}",
            interval=(-4.0, 3.0) if status == "held" else (-22.0, 3.0),
            reason=f"check_{index} reads {status} on its interval",
        )
        for index, (floor, (_label, status)) in enumerate(zip(declared, floors, strict=True))
    ]
    return campaign.model_copy(
        update={
            "declaration": campaign.declaration.model_copy(update={"floors": declared}),
            "readings": campaign.readings.model_copy(update={"floors": statuses}),
        }
    )


def _wide_memo(campaign: CampaignBundle) -> str:
    """The toy memo recommending the wide width, with one aspect summary per aspect."""
    payload = memo_payload(campaign)
    payload["recommendation"][0]["arm"] = arm_at(campaign, TOYHOST_WIDE)
    (summary,) = payload["aspect_summaries"]
    payload["aspect_summaries"] = [{**summary, "aspect": label} for label in campaign.labels]
    return json.dumps(payload)


async def _refusal(campaign: CampaignBundle) -> str:
    with pytest.raises(SoundnessRefusal) as refused:
        await _generate(campaign, FixturedClient(_wide_memo(campaign)))
    return str(refused.value)


class TestACandidateShownNoWorseIsRecommendable:
    """A floor is the campaign's line for how much worse a candidate may be: held on all of them answers "no worse"."""

    async def test_a_candidate_held_on_every_floor_of_every_asked_aspect_is_accepted_though_nothing_separated(self):
        first = _aspect()
        campaign = _with_floors([first, _aspect("second")], [(first.aspect, "held"), ("second", "held")])
        client = FixturedClient(_wide_memo(campaign))

        analysis = await _generate(campaign, client)

        assert len(client.calls) == 1, "a candidate shown no worse is accepted first time"
        assert analysis.document.recommendation[0].arm == arm_at(campaign, TOYHOST_WIDE)

    async def test_an_unasked_aspect_needs_no_floor_and_its_floors_are_not_read(self):
        """Only the asked aspects decide: an unasked one with no floor, or with a broken one, changes nothing."""
        first = _aspect()
        for floors in ([(first.aspect, "held")], [(first.aspect, "held"), ("lead", "broken")]):
            campaign = _with_floors([first, _aspect("lead", asked=False)], floors)
            payload = json.loads(_wide_memo(campaign))
            broken = [status.floor for status in campaign.readings.floors if status.status == "broken"]
            payload["recommendation"][0]["floors_overridden"] = [{"floor": key, "why": "A lead."} for key in broken]
            client = FixturedClient(json.dumps(payload))
            await _generate(campaign, client)
            assert len(client.calls) == 1

    async def test_held_floors_carry_the_switch_though_a_reading_is_worse_against_a_tighter_margin(self):
        """The floor is the campaign's declared line: a drop past the aspect's margin and within the floor is accepted.

        The separated-gain path accepts a trade-off with a regressed reading unaddressed, and this path does the same.
        """
        first = _aspect()
        campaign = _with_floors([first], [(first.aspect, "held")])
        worse = AspectReading(
            aspect=first.aspect,
            arm=campaign.readings.floors[0].arm,
            arm_label=arm_at(campaign, TOYHOST_WIDE),
            change=-12.0,
            gain_interval=None,
            gain_confidence=None,
            drop_interval=(-13.5, -10.5),
            drop_confidence=0.95,
            gain_p=None,
            gain_p_holm=None,
            drop_p=0.01,
            drop_p_holm=0.01,
            reading="worse",
            margin=10.0,
            control_cases=60,
            arm_cases=60,
            control_passed=150,
            control_trials=180,
            arm_passed=128,
            arm_trials=180,
            projection=None,
            reason="the wide width's drop is past the 10-point margin on its whole interval",
            confirmatory=True,
        )
        campaign = campaign.model_copy(update={"readings": campaign.readings.model_copy(update={"readings": [worse]})})
        client = FixturedClient(_wide_memo(campaign))

        analysis = await _generate(campaign, client)

        assert len(client.calls) == 1
        assert analysis.document.recommendation[0].arm == arm_at(campaign, TOYHOST_WIDE)

    async def test_one_floor_still_unclear_refuses_it_naming_that_floor_and_why(self):
        first = _aspect()
        campaign = _with_floors([first], [(first.aspect, "held"), (first.aspect, "unclear")])
        message = await _refusal(campaign)
        held, unclear = campaign.readings.floors
        assert (
            f"Nor is it shown no worse than its floors: floor {unclear.floor} is unclear ({unclear.reason})" in message
        )
        assert held.floor not in message, "only a floor not held is named"
        assert f"Recommend the control, {arm_at(campaign, TOYHOST_NARROW)!r}," in message

    async def test_every_floor_held_on_one_asked_aspect_does_not_carry_another(self):
        first = _aspect()
        campaign = _with_floors([first, _aspect("second")], [(first.aspect, "held"), ("second", "unclear")])
        message = await _refusal(campaign)
        assert "floor second/rate_drop/check_1 is unclear" in message
        assert first.aspect not in message.split("Nor is it shown no worse than its floors:")[1]

    async def test_a_broken_floor_refuses_it(self):
        first = _aspect()
        campaign = _with_floors([first], [(first.aspect, "held"), (first.aspect, "broken")])
        assert f"floor {first.aspect}/rate_drop/check_1 is broken" in await _refusal(campaign)

    async def test_an_asked_aspect_declaring_no_floor_shows_nothing_no_worse(self):
        first = _aspect()
        campaign = _with_floors([first, _aspect("second")], [(first.aspect, "held")])
        message = await _refusal(campaign)
        assert "aspect 'second' declares no floor, so nothing there shows it no worse" in message
        assert first.aspect not in message.split("Nor is it shown no worse than its floors:")[1]

    async def test_a_declared_floor_read_only_on_another_arm_is_not_held_by_this_one(self):
        first = _aspect()
        campaign = _with_floors([first], [(first.aspect, "held")], arm="some-other-candidate")
        message = await _refusal(campaign)
        assert f"floor {first.aspect}/rate_drop/check_0 was not read on it, so it is not shown held" in message
        assert "declares no floor" not in message

    async def test_a_breached_guardrail_refuses_it_whatever_its_floors_held(self):
        first = _aspect(guardrail=True)
        campaign = _with_floors([first], [(first.aspect, "held")])
        assert "breached the guardrail boundary.correct" in await _refusal(campaign)


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
