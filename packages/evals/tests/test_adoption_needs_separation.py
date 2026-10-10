"""An adopted arm is one the statistics separated from the control in its favour.

An ``adopted`` decision makes the arm it names the report's winner, so it claims that arm beat the
control. ``multiple_comparisons`` is where the bundle says whether it did, and a decision that
adopts an arm no reading separated is refused, repairably: the model gets one round to defer it
or keep the control. A trade-off is still a decision a reader may take — better on one reading
and worse on another, or cheaper and not separated on quality — so one ``improved`` verdict
licenses the adoption.

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
    alias_at,
    cell_at,
    memo_payload,
)
from threetears.evals.analysis.arms import arm_table
from threetears.evals.analysis.bundle import AnalysisContextBundle, ComparisonVerdict, assemble_context_bundle
from threetears.evals.analysis.cells import variant_of_cell_ref
from threetears.evals.analysis.errors import SoundnessRefusal
from threetears.evals.analysis.generator import generate_analysis
from threetears.evals.analysis.report import build_report
from threetears.evals.analysis.report.model import TextBlock
from threetears.evals.contracts.surface import GuardrailCell, GuardrailCheck, GuardrailDecision, GuardrailReadings
from threetears.evals.contracts.campaign import EvalAnalysis
from threetears.evals.contracts.models import utc_now_iso

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
    assert campaign.declared_design is not None, "the toy campaign declares a design; this check adds its control"
    design = campaign.declared_design.model_copy(
        update={"control": variant_of_cell_ref(cell_at(bundle, TOYHOST_NARROW))}
    )
    return assemble_context_bundle(
        campaign.model_copy(update={"declared_design": design}), storage=storage, profile=profile
    )


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


def _memo(bundle: AnalysisContextBundle, *, disposition: str, cells: list[str]) -> str:
    """The toy memo with its one decision replaced."""
    payload = memo_payload(bundle)
    payload["decisions"] = [
        {
            "proposal": "Use this chunk width in the extraction pipeline.",
            "disposition": disposition,
            "cells": cells,
            "confidence": "medium",
            "rests_on": [0],
            "revisit_when": "a run with more documents per width" if disposition == "deferred" else "",
        }
    ]
    return json.dumps(payload)


async def _generate(bundle: AnalysisContextBundle, client: FixturedClient) -> EvalAnalysis:
    analysis, _insights = await generate_analysis(
        bundle,
        prompt=PROMPT,
        model=MODEL,
        client=client,
        prompt_id=PROMPT_ID,
        bundle_assembled_at=utc_now_iso(),
        profile=toyhost_profile(),
    )
    return analysis


async def _adopt(bundle: AnalysisContextBundle, level: int, *, disposition: str = "adopted") -> EvalAnalysis:
    """Generate over a memo whose one decision names the arm at ``level``, in one call or the refusal."""
    client = FixturedClient(_memo(bundle, disposition=disposition, cells=[alias_at(bundle, level)]))
    analysis = await _generate(bundle, client)
    assert len(client.calls) == 1, "an adoption the statistics support is accepted first time"
    return analysis


def _status(analysis: EvalAnalysis, bundle: AnalysisContextBundle, level: int) -> str:
    variant = variant_of_cell_ref(cell_at(bundle, level))
    (row,) = [row for row in arm_table(analysis).rows if row.variant_key == variant]
    return row.status


class TestAnAdoptionTheStatisticsSupportIsAccepted:
    async def test_a_trade_off_better_on_one_reading_and_worse_on_others_is_adopted(self):
        """The assembled family as it stands: the wide width improves accuracy and regresses cost and latency."""
        bundle = _toy_bundle(control=True)
        verdicts = {c.name: c.verdict for f in bundle.multiple_comparisons.families for c in f.comparisons}
        assert verdicts["field_accuracy"] == "improved" and verdicts["production_replicating_cost"] == "regressed"

        analysis = await _adopt(bundle, TOYHOST_WIDE)

        assert _status(analysis, bundle, TOYHOST_WIDE) == "winner"

    async def test_an_arm_separated_only_on_cost_is_adopted(self):
        """Cheaper, and not separated on quality: a trade-off a reader may take."""
        bundle = _with_verdicts(
            _toy_bundle(control=True), {"production_replicating_cost": "improved"}, otherwise="not_separated"
        )
        assert _status(await _adopt(bundle, TOYHOST_WIDE), bundle, TOYHOST_WIDE) == "winner"

    async def test_adopting_the_control_needs_no_separation(self):
        bundle = _with_verdicts(_toy_bundle(control=True), {}, otherwise="not_separated")
        await _adopt(bundle, TOYHOST_NARROW)

    @pytest.mark.parametrize("disposition", ["deferred", "rejected"])
    async def test_a_deferred_or_rejected_decision_is_not_checked(self, disposition: str):
        bundle = _with_verdicts(_toy_bundle(control=True), {}, otherwise="not_separated")
        await _adopt(bundle, TOYHOST_WIDE, disposition=disposition)

    async def test_an_adopted_decision_naming_no_cell_is_not_checked(self):
        bundle = _toy_bundle(control=False)
        client = FixturedClient(_memo(bundle, disposition="adopted", cells=[]))
        await _generate(bundle, client)
        assert len(client.calls) == 1


class TestAnAdoptionNoReadingSeparatedIsRefused:
    async def test_an_arm_no_reading_separated_is_refused_and_told_what_to_do(self):
        bundle = _with_verdicts(_toy_bundle(control=True), {}, otherwise="not_separated")
        with pytest.raises(SoundnessRefusal) as refused:
            await _adopt(bundle, TOYHOST_WIDE)
        message = str(refused.value)
        assert "decisions[0] adopts the arm at cell" in message
        assert "no reading separated it from the control" in message
        assert "mark it `deferred` with what would settle it in `revisit_when`, or adopt the control" in message

    async def test_an_arm_separated_only_by_regressing_is_refused(self):
        """Its only separation is evidence against it: there is no upside a trade-off could buy."""
        bundle = _with_verdicts(
            _toy_bundle(control=True), {"production_replicating_cost": "regressed"}, otherwise="not_separated"
        )
        with pytest.raises(SoundnessRefusal, match=r"only by being worse \(production_replicating_cost regressed\)"):
            await _adopt(bundle, TOYHOST_WIDE)

    async def test_the_repair_round_defers_it_and_the_analysis_is_stored(self):
        bundle = _with_verdicts(_toy_bundle(control=True), {}, otherwise="not_separated")
        wide = alias_at(bundle, TOYHOST_WIDE)
        client = _MemoSequence(
            _memo(bundle, disposition="adopted", cells=[wide]), _memo(bundle, disposition="deferred", cells=[wide])
        )

        analysis = await _generate(bundle, client)

        assert len(client.calls) == 2
        refusal = analysis.generation.repaired_refusal
        assert analysis.generation.repair_attempts == 1 and refusal is not None
        assert f"decisions[0] adopts the arm at cell {wide} " in refusal, "the writer is told in its own aliases"
        assert refusal in client.calls[1]["user"]
        assert analysis.document.decisions[0].disposition == "deferred"
        assert _status(analysis, bundle, TOYHOST_WIDE) == "unresolved"


def _contradicting_memo(bundle: AnalysisContextBundle, cells: list[str], *, then_reject: bool = True) -> str:
    """The toy memo deciding twice: adopt ``cells``, then reject them (or, without ``then_reject``, defer them)."""
    payload = memo_payload(bundle)
    adopt = {
        "proposal": "Use this chunk width in the extraction pipeline.",
        "disposition": "adopted",
        "cells": cells,
        "confidence": "medium",
        "rests_on": [0],
        "revisit_when": "",
    }
    second = {
        "proposal": "Keep this chunk width out of the extraction pipeline.",
        "disposition": "rejected" if then_reject else "deferred",
        "cells": cells,
        "confidence": "medium",
        "rests_on": [0],
        "revisit_when": "" if then_reject else "a run with more documents per width",
    }
    payload["decisions"] = [adopt, second]
    return json.dumps(payload)


class TestAnArmAdoptedAndRejectedIsRefused:
    """One arm under two verdicts is a structural defect in the memo (#669), refused through the repair round."""

    async def test_the_refusal_names_both_decision_positions(self):
        """The wide width improves on accuracy, so the adoption alone would stand: the contradiction is what is refused."""
        bundle = _toy_bundle(control=True)
        with pytest.raises(SoundnessRefusal) as refused:
            await _generate(bundle, FixturedClient(_contradicting_memo(bundle, [alias_at(bundle, TOYHOST_WIDE)])))
        message = str(refused.value)
        assert "decisions[0] adopts the arm at cell" in message
        assert "and decisions[1] rejects it" in message
        assert "one arm takes one verdict" in message

    async def test_it_is_read_off_the_arm_not_the_cell_list(self):
        """Adopting the control while rejecting it is as contradictory as any other arm, needing no separation."""
        bundle = _toy_bundle(control=True)
        narrow = alias_at(bundle, TOYHOST_NARROW)
        with pytest.raises(SoundnessRefusal, match=r"decisions\[0\] adopts .* and decisions\[1\] rejects it"):
            await _generate(bundle, FixturedClient(_contradicting_memo(bundle, [narrow])))

    async def test_the_repair_round_settles_it_and_the_arm_reads_as_the_winner(self):
        bundle = _toy_bundle(control=True)
        wide = alias_at(bundle, TOYHOST_WIDE)
        client = _MemoSequence(
            _contradicting_memo(bundle, [wide]), _contradicting_memo(bundle, [wide], then_reject=False)
        )

        analysis = await _generate(bundle, client)

        assert len(client.calls) == 2
        refusal = analysis.generation.repaired_refusal
        assert refusal is not None and f"decisions[0] adopts the arm at cell {wide} " in refusal
        assert _status(analysis, bundle, TOYHOST_WIDE) == "winner"


class TestAnArmNoTestCouldReachHasNoSeparationToAdoptOn:
    """No family, no comparison naming the arm, or only untested ones: nothing separated it, so it is refused."""

    async def test_a_bundle_with_no_control_withholds_every_family_and_refuses_the_adoption(self):
        bundle = _toy_bundle(control=False)
        assert bundle.multiple_comparisons.withheld is not None
        with pytest.raises(SoundnessRefusal) as refused:
            await _adopt(bundle, TOYHOST_NARROW)
        message = str(refused.value)
        assert "the bundle tested no comparison against a control" in message
        assert "no test was possible" in message
        assert "adopt the control" not in message, "there is no control to offer"

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
        assert "mark it `rejected`, or adopt the control" in message

    async def test_an_undecided_guardrail_lets_the_adoption_stand_and_the_report_states_it(self):
        bundle = _with_guardrail(_toy_bundle(control=True), "undecided")
        analysis = await _adopt(bundle, TOYHOST_WIDE)

        assert analysis.decision_surface.guardrails == bundle.guardrails, "frozen with the analysis"
        report = build_report(analysis)
        (decision,) = [b for b in report.blocks if isinstance(b, TextBlock) and b.role == "decision"]
        facts = {fact.name: fact.value for fact in decision.facts}
        assert "undecided on boundary.correct, so not known to be safe" in facts["Guardrails"]
        assert [b.section for b in report.blocks].count("guardrails") >= 1

    async def test_a_held_guardrail_adds_nothing_to_the_decision(self):
        bundle = _with_guardrail(_toy_bundle(control=True), "held").model_copy()
        held = bundle.guardrails.checks[0].model_copy(update={"interval": (-0.1, 0.4), "decision": "held"})
        bundle = bundle.model_copy(update={"guardrails": bundle.guardrails.model_copy(update={"checks": [held]})})
        report = build_report(await _adopt(bundle, TOYHOST_WIDE))
        (decision,) = [b for b in report.blocks if isinstance(b, TextBlock) and b.role == "decision"]
        assert "Guardrails" not in {fact.name for fact in decision.facts}


# --- exploratory: a finding resting on readings no declared question asked about is labelled ----------------


def _toy_bundle_asking(axes: list[str]) -> AnalysisContextBundle:
    """The toy campaign's bundle, its one question scoped to ``axes``."""
    profile = toyhost_profile()
    campaign, storage = toyhost_campaign(profile=profile)
    assert campaign.declared_design is not None
    design = campaign.declared_design
    questions = [question.model_copy(update={"merit_axes": axes}) for question in design.questions]
    campaign = campaign.model_copy(update={"declared_design": design.model_copy(update={"questions": questions})})
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
        analysis = await _generate(bundle, FixturedClient(json.dumps(memo_payload(bundle))))
        scope = _finding_scope(analysis)
        assert scope is not None and scope.startswith("exploratory")

    async def test_a_question_on_latency_asks_about_it_and_no_label_is_added(self):
        bundle = _toy_bundle_asking(["latency"])
        analysis = await _generate(bundle, FixturedClient(json.dumps(memo_payload(bundle))))
        assert _finding_scope(analysis) is None
        report = build_report(analysis)
        assert not [b for b in report.blocks if getattr(b, "source", None) == "scope"], "said only with no question"
