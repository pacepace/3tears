"""A stored plan's charts drawn through the chart resolvers, as the retired planned page drew them, for the tests that
hold the resolvers on every chart a plan stores (an anchor, a current run, the Kairo reference).

:func:`anchor_look` is the look the charts draw under: each arm's colour slot and the options' order from the plan,
and code's facts about the input. :func:`drawn_charts` draws every summary and section chart of a plan through
:func:`~threetears.evals.analysis.report.parts.resolve.resolve_chart` under that look, its title's figure keys
written out. :func:`charts_html` and :func:`charts_markdown` lay the drawn charts out with the writers every page
shares, so a test reads what a page would print.
"""

from __future__ import annotations

from collections.abc import Sequence

from threetears.evals.analysis.report.blocks import ChartBlock
from threetears.evals.analysis.report.figure_keys import render
from threetears.evals.analysis.report.html_parts import chart_html, escape, page_head, table_html
from threetears.evals.analysis.report.md_parts import chart_lines, inert, one_line, table_lines
from threetears.evals.analysis.report.parts.look import PageLook, page_look
from threetears.evals.analysis.report.parts.model import PageChart
from threetears.evals.analysis.report.parts.resolve import resolve_chart
from threetears.evals.analysis.report_decision import surviving_options
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_plan_facts import plan_facts
from threetears.evals.kernel.report_plan import ChartSpec, ReportPlan, plan_recommended_arm

__all__ = ["anchor_look", "charts_html", "charts_markdown", "drawn_charts"]


def anchor_look(plan: ReportPlan, report_input: ReportInput) -> PageLook:
    """The look a plan's charts draw with on ``report_input``: each arm's tag, name, colour and mark, the options'
    order, the survivors and the recommended arm.

    :param plan: The stored plan.
    :param report_input: The input it is drawn on.
    :return: The look.
    """
    facts = plan_facts(report_input)
    return page_look(
        facts.arms,
        {arm.id: arm.color_slot for arm in plan.arms},
        order=[row.option for row in plan.options.rows],
        survivors=surviving_options(report_input),
        recommended=plan_recommended_arm(plan),
        names=facts.names,
    )


def drawn_charts(plan: ReportPlan, report_input: ReportInput) -> list[tuple[ChartSpec, PageChart]]:
    """Every chart of a plan drawn on ``report_input``, summary charts first, then each section's, in the plan's order:
    each its title's figure keys written out, a section's titled by its headline where its chart has none.

    :param plan: The stored plan.
    :param report_input: The input it is drawn on.
    :return: Each chart as stored, beside it as drawn.
    :raises UnresolvableReference: A chart does not draw on the input.
    """
    look = anchor_look(plan, report_input)
    names = {arm.id: arm.name for arm in plan_facts(report_input).arms}

    def written(text: str) -> str:
        return render(text, report_input, names)

    drawn = [
        (spec, resolve_chart(spec, report_input, look, title=written(spec.title), summary=True))
        for spec in plan.summary
    ]
    drawn += [
        (
            section.chart,
            resolve_chart(
                section.chart,
                report_input,
                look,
                title=written(section.chart.title),
                fallback_title=written(section.headline),
            ),
        )
        for section in plan.sections
    ]
    return drawn


def charts_html(charts: Sequence[PageChart], *, static: bool = False) -> str:
    """The charts as one page lays them out, each its figure (or its table) and code's lines under it."""
    parts = [*page_head("charts", static=static), "<body>"]
    for index, chart in enumerate(charts):
        block = chart.chart
        drawn = (
            chart_html(block, f"chart-{index}", static=static, figcaption=bool(chart.title))
            if isinstance(block, ChartBlock)
            else table_html(block)
        )
        parts.append(drawn + "".join(f'<p class="cap">{escape(note)}</p>' for note in chart.notes))
    return "\n".join([*parts, "</body>", "</html>"]) + "\n"


def charts_markdown(charts: Sequence[PageChart]) -> str:
    """The charts as Markdown, each its lines (or its table) and code's lines under it."""
    lines: list[str] = []
    for chart in charts:
        block = chart.chart
        drawn = (
            chart_lines(block)
            if isinstance(block, ChartBlock)
            else [f"**{one_line(block.title)}**", "", *table_lines(block)]
        )
        lines += [*drawn, *(line for note in chart.notes for line in ("", inert(note))), ""]
    return "\n".join(lines)
