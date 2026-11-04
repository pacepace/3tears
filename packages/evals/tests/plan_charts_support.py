"""The charts the retired report planner's plans stored, drawn through the chart resolvers as its page drew them, for
the tests that hold the resolvers on every chart a plan stored (an anchor, a recorded run, the Kairo reference).

The plans are kept as JSON under ``fixtures/plan_charts/`` (``rN-anchor``, ``rN-run``, ``r5-reference``), never
validated as a plan: :class:`StoredCharts` reads only their charts (each a
:class:`~threetears.evals.kernel.report_plan.ChartSpec`) and the look they were drawn with (each arm's colour slot,
the options table's order, the recommended arm). :func:`anchor_look` is that look on an input, with each arm's tag
and name and every id's reader name as code reads them. :func:`drawn_charts` draws every summary and section chart
through :func:`~threetears.evals.analysis.report.parts.resolve.resolve_chart` under it, its title's figure keys
written out. :func:`charts_html` and :func:`charts_markdown` lay the drawn charts out with the writers every page
shares, so a test reads what a page would print.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from threetears.evals.analysis.report.blocks import ChartBlock
from threetears.evals.analysis.report.figure_keys import reader_names, render
from threetears.evals.analysis.report.html_parts import chart_html, escape, page_head, table_html
from threetears.evals.analysis.report.md_parts import chart_lines, inert, one_line, table_lines
from threetears.evals.analysis.report.parts.look import PageLook, page_look
from threetears.evals.analysis.report.parts.model import PageChart
from threetears.evals.analysis.report.parts.resolve import resolve_chart
from threetears.evals.analysis.report_arms import fact_arms
from threetears.evals.analysis.report_decision import surviving_options
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.report_plan import ChartSpec

__all__ = [
    "PLAN_CHARTS",
    "StoredCharts",
    "StoredSection",
    "anchor_look",
    "charts_html",
    "charts_markdown",
    "drawn_charts",
    "stored_charts",
]

#: The stored plans, as JSON: ``rN-anchor`` (hand-written), ``rN-run`` (the planner's last recorded run) and
#: ``r5-reference`` (the kit's Kairo round 2 plan).
PLAN_CHARTS = Path(__file__).resolve().parent / "fixtures" / "plan_charts"


@dataclass(frozen=True)
class StoredSection:
    """One evidence section's chart, beside the headline that titles it where the chart has no title of its own."""

    headline: str
    chart: ChartSpec


@dataclass(frozen=True)
class StoredCharts:
    """A stored plan's charts and the look they were drawn with."""

    slots: dict[str, int]
    order: list[str]
    recommended: str | None
    summary: list[ChartSpec] = field(default_factory=list)
    sections: list[StoredSection] = field(default_factory=list)

    def charts(self) -> list[ChartSpec]:
        """Every chart, summary charts first, then each section's, in the plan's order."""
        return [*self.summary, *(section.chart for section in self.sections)]

    def with_one_section(self, chart: ChartSpec) -> StoredCharts:
        """The same look with no summary chart and one section, its first section's headline, drawing ``chart``."""
        return replace(self, summary=[], sections=[StoredSection(self.sections[0].headline, chart)])


def stored_charts(name: str) -> StoredCharts:
    """A stored plan's charts (``r1-anchor``, ``r5-run``, ``r5-reference`` …).

    :param name: The fixture's name under ``fixtures/plan_charts/``, without ``.json``.
    :return: Its charts and look.
    """
    plan: dict[str, Any] = json.loads((PLAN_CHARTS / f"{name}.json").read_text(encoding="utf-8"))
    rows = plan["options"]["rows"]
    return StoredCharts(
        slots={arm["id"]: arm["color_slot"] for arm in plan["arms"]},
        order=[row["option"] for row in rows],
        recommended=next((row["option"] for row in rows if row["role"] == "recommended"), None),
        summary=[ChartSpec.model_validate(chart) for chart in plan["summary"]],
        sections=[
            StoredSection(section["headline"], ChartSpec.model_validate(section["chart"]))
            for section in plan["sections"]
        ],
    )


def anchor_look(stored: StoredCharts, report_input: ReportInput) -> PageLook:
    """The look a stored plan's charts draw with on ``report_input``: each arm's tag, name, colour and mark, the
    options' order, the survivors and the recommended arm.

    :param stored: The stored plan's charts and look.
    :param report_input: The input it is drawn on.
    :return: The look.
    """
    return page_look(
        fact_arms(report_input),
        stored.slots,
        order=stored.order,
        survivors=surviving_options(report_input),
        recommended=stored.recommended,
        names={str(kind): found for kind, found in reader_names(report_input).items()},
    )


def drawn_charts(stored: StoredCharts, report_input: ReportInput) -> list[tuple[ChartSpec, PageChart]]:
    """Every chart of a stored plan drawn on ``report_input``, summary charts first, then each section's, in the plan's
    order: each its title's figure keys written out, a section's titled by its headline where its chart has none.

    :param stored: The stored plan's charts and look.
    :param report_input: The input it is drawn on.
    :return: Each chart as stored, beside it as drawn.
    :raises UnresolvableReference: A chart does not draw on the input.
    """
    look = anchor_look(stored, report_input)
    names = {arm.id: arm.name for arm in fact_arms(report_input)}

    def written(text: str) -> str:
        return render(text, report_input, names)

    drawn = [
        (spec, resolve_chart(spec, report_input, look, title=written(spec.title), summary=True))
        for spec in stored.summary
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
        for section in stored.sections
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
