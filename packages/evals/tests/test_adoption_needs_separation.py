"""How a stored memo's page lays out the guardrails on the arm it recommends, and the scope of a finding no question
asked about.

The memo writer refused a recommendation of a candidate the statistics did not separate from the control; it is
retired, and those refusals left with it. The memos it stored still lay out
(:func:`~threetears.evals.analysis.report.build_report`): an undecided guardrail on the recommended arm is frozen with
the analysis and stated on its page, a held one adds nothing, and a finding resting on readings no declared question
asked about is labelled exploratory. Each memo here is built as the writer stored one (:func:`_stored_memo`), over the
toy host's campaign with its narrow width declared the control.
"""

from __future__ import annotations

from packages.evals.tests.report_support import memo_analysis_over

from typing import Any

from packages.evals.tests.aspect_support import with_aspect, with_control
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_AXIS, TOYHOST_NARROW, TOYHOST_WIDE, toyhost_campaign
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.toyhost_cells import arm_at, cell_at
from threetears.evals.analysis.bundle.assemble import assemble_context_bundle
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.cells import variant_of_cell_ref
from threetears.evals.analysis.report import build_report
from threetears.evals.analysis.report.model import DisclosureBlock, TableBlock, TextBlock
from threetears.evals.kernel.campaign import EvalAnalysis
from threetears.evals.kernel.surface import GuardrailCell, GuardrailCheck, GuardrailDecision, GuardrailReadings


def _toy_bundle(*, control: bool) -> AnalysisContextBundle:
    """The toy campaign's bundle, with its narrow width declared the control or with no control at all."""
    profile = toyhost_profile()
    campaign, storage = toyhost_campaign(profile=profile)
    bundle = assemble_context_bundle(campaign, storage=storage, profile=profile)
    if not control:
        return bundle
    controlled = with_control(campaign, variant_of_cell_ref(cell_at(bundle, TOYHOST_NARROW)))
    return assemble_context_bundle(controlled, storage=storage, profile=profile)


def _memo(bundle: CampaignBundle, level: int) -> dict[str, Any]:
    """The toy memo: one finding on the widths' latency (``total_ms``), recommending the arm at ``level``."""
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
        "recommendation": [
            {
                "arm": arm_at(bundle, level),
                "reasons": ["The width recommended is the one to run."],
                "confidence": "high",
                "rests_on": [0],
                "floors_overridden": [],
                "behind_on": [],
                "bars_missed": [],
            }
        ],
        "aspect_summaries": [
            {
                "aspect": bundle.labels[0],
                "what_changed": "The wide chunk width slowed every document down.",
                "how_sure": "Sure of the direction; the gap is wide.",
                "means": "The wider one is slower by a wide margin.",
            }
        ],
        "summary": "The width recommended is the one to run.",
        "next": [],
    }


def _adopted(bundle: AnalysisContextBundle, level: int) -> EvalAnalysis:
    """The stored memo over ``bundle`` recommending the arm at ``level``."""
    campaign = campaign_bundle(bundle)
    return memo_analysis_over(campaign, _memo(campaign, level))


# --- guardrails: an undecided guardrail on the recommended arm is stated on its page ----------------------------


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
    def test_an_undecided_guardrail_is_frozen_with_the_analysis_and_the_report_states_it(self):
        bundle = _with_guardrail(_toy_bundle(control=True), "undecided")
        analysis = _adopted(bundle, TOYHOST_WIDE)

        assert analysis.aspects[0].decision_surface.guardrails == bundle.guardrails, "frozen with the analysis"
        report = build_report(analysis)
        assert any(
            isinstance(b, DisclosureBlock) and "not known to be safe" in b.text and "boundary.correct" in b.text
            for b in report.blocks
        )
        assert any(isinstance(b, TableBlock) and b.name == "guardrails" for b in report.blocks)

    def test_a_held_guardrail_adds_nothing_to_the_decision(self):
        bundle = _with_guardrail(_toy_bundle(control=True), "held").model_copy()
        held = bundle.guardrails.checks[0].model_copy(update={"interval": (-0.1, 0.4), "decision": "held"})
        bundle = bundle.model_copy(update={"guardrails": bundle.guardrails.model_copy(update={"checks": [held]})})
        report = build_report(_adopted(bundle, TOYHOST_WIDE))
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

    def test_a_question_on_quality_leaves_the_latency_finding_exploratory(self):
        bundle = _toy_bundle_asking(["quality"])
        assert "total_ms" in bundle.reading_scope.exploratory_measures
        scope = _finding_scope(_adopted(bundle, TOYHOST_NARROW))
        assert scope is not None and scope.startswith("exploratory")

    def test_a_question_on_latency_asks_about_it_and_no_label_is_added(self):
        bundle = _toy_bundle_asking(["latency"])
        analysis = _adopted(bundle, TOYHOST_NARROW)
        assert _finding_scope(analysis) is None
        report = build_report(analysis)
        assert not [b for b in report.blocks if getattr(b, "source", None) == "scope"], "said only with no question"
