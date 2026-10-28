"""A stored chart to the chart a page draws: dispatched on the chart's kind through every resolver module's registry.

Each ``resolve_*`` module owns ``RESOLVERS``, its kinds to their resolvers, so the kinds are resolved apart:

- :mod:`~threetears.evals.analysis.report.plan_page.resolve_changes`: forest, diverging bars;
- :mod:`~threetears.evals.analysis.report.plan_page.resolve_magnitudes`: bars, lines;
- :mod:`~threetears.evals.analysis.report.plan_page.resolve_examples`: example cards;
- :mod:`~threetears.evals.analysis.report.plan_page.resolve_gates`, ``resolve_rates``, ``resolve_cells`` and
  ``resolve_table``: the rest of the kit's kinds and the plain table, each registered by its own module.

A kind no module resolves is refused, naming the kind and its source, never drawn empty; so is a kind two modules
claim, when this module is loaded.
"""

from __future__ import annotations

from collections.abc import Mapping

from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.plan_page import (
    resolve_cells,
    resolve_changes,
    resolve_examples,
    resolve_gates,
    resolve_magnitudes,
    resolve_rates,
    resolve_table,
)
from threetears.evals.analysis.report.blocks import ChartBlock
from threetears.evals.analysis.report.plan_page.charting import ChartContext, KindResolver
from threetears.evals.analysis.report.plan_page.look import PageLook
from threetears.evals.analysis.report.plan_page.model import PageChart
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_rows import config_grid
from threetears.evals.kernel.report_plan import ChartSpec

__all__ = ["resolve_chart"]


def _merged(registries: Mapping[str, Mapping[str, KindResolver]]) -> dict[str, KindResolver]:
    """Every module's resolvers, by kind.

    :raises ValueError: Two modules claim one kind.
    """
    merged: dict[str, KindResolver] = {}
    owner: dict[str, str] = {}
    for module, resolvers in registries.items():
        for kind, resolver in resolvers.items():
            if kind in merged:
                raise ValueError(f"chart kind {kind!r} is resolved by both {owner[kind]} and {module}")
            merged[kind], owner[kind] = resolver, module
    return merged


#: Every kind a page can draw, to its resolver.
_RESOLVERS: dict[str, KindResolver] = _merged(
    {
        module.__name__: module.RESOLVERS
        for module in (
            resolve_changes,
            resolve_magnitudes,
            resolve_examples,
            resolve_gates,
            resolve_rates,
            resolve_cells,
            resolve_table,
        )
    }
)


def resolve_chart(
    spec: ChartSpec,
    report_input: ReportInput,
    look: PageLook,
    *,
    title: str,
    fallback_title: str = "",
    summary: bool = False,
) -> PageChart:
    """Draw one stored chart: its kind's resolver reads its query off the input.

    :param spec: The stored chart.
    :ptype spec: ChartSpec
    :param report_input: The report input the plan was written against.
    :ptype report_input: ReportInput
    :param look: The page's look.
    :ptype look: PageLook
    :param title: The plan's title for the chart, its keys written out; empty where a headline says it.
    :ptype title: str
    :param fallback_title: What the chart's block is titled where the plan's title is empty: the section's headline.
    :ptype fallback_title: str
    :param summary: Whether it is a summary chart, whose rows follow the options table.
    :ptype summary: bool
    :return: The chart as the page draws it.
    :rtype: PageChart
    :raises UnresolvableReference: No resolver draws the chart's kind, or its resolver cannot draw it from the input.
    """
    resolver = _RESOLVERS.get(spec.kind)
    if resolver is None:
        raise UnresolvableReference(
            f"no resolver draws a {spec.kind!r} chart over {spec.query.source!r} yet ({title or fallback_title!r})"
        )
    if spec.query.source == "configs":
        # A configs chart reads the grid the input's configs make, or, where it holds none, its arms' settings.
        report_input = config_grid(report_input)
    context = ChartContext(
        spec=spec, report_input=report_input, look=look, title=title or fallback_title or spec.kind, summary=summary
    )
    drawn = resolver(context)
    # What the chart's money is drawn per, where its figures do not say it, among code's lines (#993).
    per = drawn.block.drawn_per if isinstance(drawn.block, ChartBlock) else []
    return PageChart(
        title=title, chart=drawn.block, notes=[*drawn.notes, *(line for line in per if line not in drawn.notes)]
    )
