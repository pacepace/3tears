"""How code's page lays out the guardrails on a candidate: an undecided guardrail is frozen with an analysis's aspect
and stated on the aspect's page, and a held one adds nothing.

The memo writer refused a recommendation of a candidate the statistics did not separate from the control; it is
retired, and its refusals and the memo page left with it. What stays is code's: the guardrail frozen with an aspect
(:func:`~threetears.evals.analysis.doc_analysis.aspect_analysis`) and the aspect's page
(:func:`~threetears.evals.analysis.report.build_code_only_report`), over the toy host's campaign with its narrow width
declared the control.
"""

from __future__ import annotations

from packages.evals.tests.aspect_support import with_control
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_NARROW, toyhost_campaign
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.toyhost_cells import cell_at
from threetears.evals.analysis.bundle.assemble import assemble_context_bundle
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.cells import variant_of_cell_ref
from threetears.evals.analysis.doc_analysis import aspect_analysis
from threetears.evals.analysis.report import Report, build_code_only_report
from threetears.evals.analysis.report.model import DisclosureBlock, TableBlock
from threetears.evals.kernel.surface import GuardrailCell, GuardrailCheck, GuardrailDecision, GuardrailReadings

#: When the toy aspect's page is laid out.
_ASSEMBLED_AT = "2026-10-14T19:23:41+00:00"


def _toy_bundle(*, control: bool) -> AnalysisContextBundle:
    """The toy campaign's bundle, with its narrow width declared the control or with no control at all."""
    profile = toyhost_profile()
    campaign, storage = toyhost_campaign(profile=profile)
    bundle = assemble_context_bundle(campaign, storage=storage, profile=profile)
    if not control:
        return bundle
    controlled = with_control(campaign, variant_of_cell_ref(cell_at(bundle, TOYHOST_NARROW)))
    return assemble_context_bundle(controlled, storage=storage, profile=profile)


def _page(bundle: AnalysisContextBundle) -> Report:
    """Code's page over the aspect's bundle."""
    return build_code_only_report(bundle, assembled_at=_ASSEMBLED_AT)


# --- guardrails: an undecided guardrail on a candidate is stated on its page -------------------------------------


def _with_guardrail(bundle: AnalysisContextBundle, decision: GuardrailDecision) -> AnalysisContextBundle:
    """The bundle with one judged guardrail decided ``decision`` for the wide width against the control.

    Its cells are those of the family's first comparison, so the check names the wide width.
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


class TestAGuardrailOnACandidate:
    def test_an_undecided_guardrail_is_frozen_with_the_analysis_and_the_report_states_it(self):
        bundle = _with_guardrail(_toy_bundle(control=True), "undecided")
        held = aspect_analysis(bundle, campaign_bundle(bundle).surfaces()[bundle.aspect])

        assert held.decision_surface.guardrails == bundle.guardrails, "frozen with the analysis"
        report = _page(bundle)
        # The guardrail sentence states it by the name the page prints the dimension by (verdict.guardrail_lines).
        stated = "1024tok may do worse than 256tok on the boundary correct (judged) guardrail."
        assert any(isinstance(b, DisclosureBlock) and b.text == stated for b in report.blocks)
        assert any(isinstance(b, TableBlock) and b.name == "guardrails" for b in report.blocks)

    def test_a_held_guardrail_adds_nothing_to_the_decision(self):
        bundle = _with_guardrail(_toy_bundle(control=True), "held").model_copy()
        held = bundle.guardrails.checks[0].model_copy(update={"interval": (-0.1, 0.4), "decision": "held"})
        bundle = bundle.model_copy(update={"guardrails": bundle.guardrails.model_copy(update={"checks": [held]})})
        report = _page(bundle)
        assert not any(isinstance(b, DisclosureBlock) and "boundary correct (judged)" in b.text for b in report.blocks)
