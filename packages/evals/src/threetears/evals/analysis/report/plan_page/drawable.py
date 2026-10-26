"""Whether a plan's charts draw on its own input: the page's resolvers, run when the plan is generated.

A plan the generation path accepts must render on the input it was written against: a chart the page cannot draw is
refused when the plan is generated, with a fault the planner can repair, and never stored (a stored plan that cannot
render would ask for a regeneration that writes the same plan again). So the refusal and the page are one rule:
:func:`undrawn_charts` draws each summary chart and each section's chart through
:func:`~threetears.evals.analysis.report.plan_page.resolve.resolve_chart`, the resolver the page calls, under
:func:`plan_look`, the look the page draws with, and turns each refusal into a fault at the chart's field path.
:func:`~threetears.evals.analysis.report_plan_facts.complete_plan` calls it beside the plan checks.

This module sits below the page builder and reads code's facts only by type, so the completion step that imports it
imports no builder that imports the completion step back.
"""

from __future__ import annotations

from collections.abc import Collection, Iterator
from typing import TYPE_CHECKING

from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.plan_page.look import PageLook, page_look
from threetears.evals.analysis.report.plan_page.resolve import resolve_chart
from threetears.evals.analysis.report_decision import surviving_options
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.report_plan import ChartSpec, ReportPlan, plan_recommended_arm

if TYPE_CHECKING:  # code's facts are read by type only: the completion step that computes them imports this module.
    from threetears.evals.analysis.report_plan_facts import PlanFacts

__all__ = ["plan_look", "undrawn_charts"]


def plan_look(plan: ReportPlan, report_input: ReportInput, facts: PlanFacts) -> PageLook:
    """The look a plan's page draws with: each arm's tag, name, colour and mark, the options table's order, the
    survivors and the recommended arm.

    :param plan: The plan.
    :ptype plan: ReportPlan
    :param report_input: The report input it was written against.
    :ptype report_input: ReportInput
    :param facts: Code's facts about the input.
    :ptype facts: PlanFacts
    :return: The look.
    :rtype: PageLook
    """
    return page_look(
        facts.arms,
        {arm.id: arm.color_slot for arm in plan.arms},
        order=[row.option for row in plan.options.rows],
        survivors=surviving_options(report_input),
        recommended=plan_recommended_arm(plan),
        names=facts.names,
    )


def _charts(plan: ReportPlan) -> Iterator[tuple[str, ChartSpec, bool]]:
    for index, chart in enumerate(plan.summary):
        yield f"summary[{index}]", chart, True
    for index, section in enumerate(plan.sections):
        yield f"sections[{index}].chart", section.chart, False


def undrawn_charts(
    plan: ReportPlan, report_input: ReportInput, facts: PlanFacts, *, faulted: Collection[str] = ()
) -> list[tuple[str, str]]:
    """Every chart of the plan the page cannot draw on its input, each as a fault at its field path.

    :param plan: The plan, completed.
    :ptype plan: ReportPlan
    :param report_input: The report input it was written against.
    :ptype report_input: ReportInput
    :param facts: Code's facts about the input.
    :ptype facts: PlanFacts
    :param faulted: Field paths already refused: a chart at or under one is not drawn, so a chart the checks refused
        is refused once.
    :ptype faulted: Collection[str]
    :return: Each fault as ``(field path, why)``, summary charts first, in the plan's order.
    :rtype: list[tuple[str, str]]
    """
    look = plan_look(plan, report_input, facts)
    faults: list[tuple[str, str]] = []
    for path, chart, summary in _charts(plan):
        if any(seen == path or seen.startswith((f"{path}.", f"{path}[")) for seen in faulted):
            continue
        try:
            resolve_chart(chart, report_input, look, title=chart.title, summary=summary)
        except UnresolvableReference as refused:
            faults.append((path, f"is refused by the page, which draws it from this input: {refused}"))
    return faults
