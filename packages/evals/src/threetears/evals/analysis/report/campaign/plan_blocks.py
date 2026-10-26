"""A planned analysis's first screen on the campaign's page: the plan's headline, lede, confidence and decision slots.

A page built from the plan itself (its options, charts and sections) is the report kit's renderer's; until it lands, a
planned analysis is laid out as the campaign's code-only page with the plan's first-screen text in the answer section,
where a memo's summary and recommendation sit, so a planned analysis reads end to end. Every figure key in the plan's
prose is written out against the report input the plan was written against
(:func:`~threetears.evals.analysis.report.figure_keys.render`).
"""

from __future__ import annotations

from threetears.evals.analysis.report.blocks import Fact, ReportBlock, TextBlock
from threetears.evals.analysis.report.figure_keys import render
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.report_plan import ReportPlan

__all__ = ["CONFIDENCE_FORM_WORDS", "SLOT_TITLES", "plan_first_screen_blocks"]

#: How the decision's confidence form reads on the page.
CONFIDENCE_FORM_WORDS: dict[str, str] = {
    "high": "High.",
    "high_if_immaterial": "High, if the open caveats below are immaterial.",
    "low": "Low.",
}

#: Each decision slot's lead on the page.
SLOT_TITLES: dict[str, str] = {
    "recommendation": "Recommendation",
    "why_not": "Why not the others",
    "watch": "Watch",
    "next": "Next",
    "the_trade": "The trade",
    "where_it_flips": "Where it flips",
    "our_lean": "Our lean",
    "what_settles_it": "What settles it",
}


def plan_first_screen_blocks(plan: ReportPlan, report_input: ReportInput) -> list[ReportBlock]:
    """The plan's first screen as the page's answer section: its headline, its lede, how sure it is and on which open
    caveats, then each decision slot in the mode's order.

    :param plan: The stored plan.
    :ptype plan: ReportPlan
    :param report_input: The report input it was written against, which its figure keys resolve in.
    :ptype report_input: ReportInput
    :return: The blocks, in reading order.
    :rtype: list[ReportBlock]
    :raises UnresolvableReference: A figure key no longer resolves against the input (a checked plan's always do).
    """

    def written(text: str) -> str:
        return render(text, report_input, {})

    confidence = plan.decision.confidence
    clauses = " ".join(written(clause.material_if) for clause in confidence.caveats)
    blocks: list[ReportBlock] = [
        TextBlock(section="answer", role="answer", body=written(plan.headline)),
        TextBlock(section="answer", role="summary", body=written(plan.lede)),
        TextBlock(
            section="answer",
            role="how_sure",
            body=" ".join(part for part in (CONFIDENCE_FORM_WORDS[confidence.form], clauses) if part),
        ),
    ]
    blocks += [
        TextBlock(
            section="answer",
            role="recommendation",
            body=written(slot.text),
            facts=[Fact(name="Disposition", value=SLOT_TITLES[slot.key])],
        )
        for slot in plan.decision.slots
    ]
    return blocks
