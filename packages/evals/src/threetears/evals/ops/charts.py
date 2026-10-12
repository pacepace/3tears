"""A report's charts, drawn by a host's renderer: every chart that draws, and one refusal per chart that does not.

A report carries each chart as an intent — what the chart says — and never as a picture: how it looks is the
renderer's (:class:`~threetears.evals.analysis.viz.ChartRenderer`; the package's own is
:class:`~threetears.evals.vega.VegaRenderer`, the ``[vega]`` adapter). A surface that shows the report's
pictures used to walk the report's blocks itself and draw each, and each surface decided on its own what one
chart failing to draw did to the rest. :func:`report_charts` is that walk, once.

**One refusal per chart, and the others still drawn.** A chart fails for reasons of its own — the stored
chart could not be decided into an intent (the block already says why), or the renderer refused the intent it
was handed (a spec rule, a payload it cannot place) — so one failure says nothing about its neighbours, and a
report with one undrawable chart is not a report with none. Each failure is returned beside the drawings as a
:class:`ChartRefusal` naming its block and its reason, so a surface states it where the chart would have been
rather than dropping it: a chart missing from a report is otherwise indistinguishable from a finding that
carried none.

**What a renderer refuses is a** :class:`~threetears.evals.analysis.viz.ChartRefused` — the Vega adapter's
:class:`~threetears.evals.vega.SpecPolicyError` (a rule its drawn spec breaks) and
:class:`~threetears.evals.vega.RasterRefused` (a spec vl-convert would not convert) are two. Anything else a
renderer raises, any other ``ValueError`` included, is a defect in the renderer or the compiler behind it, not a
property of one chart, and propagates: caught as a refusal, a broken compiler read as a report of undrawable
charts.

**Only the chart blocks are read**, and of each only its intent, its type, the reason it could not be decided
and the finding it belongs to; nothing here reads a report's text or tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from threetears.evals.analysis.report import ChartBlock, Report
from threetears.evals.analysis.viz import ChartRefused, ChartRenderer, ChartType
from threetears.evals.schema.base import EvalBaseModel

#: Why a chart was not drawn: ``undrawable`` — the report could not decide the stored chart into an intent,
#: and its block says why; ``refused_by_renderer`` — the intent was decided and the renderer refused to draw it.
ChartRefusalCause = Literal["undrawable", "refused_by_renderer"]


@dataclass(frozen=True)
class DrawnChart[DrawingT]:
    """One chart of a report, drawn.

    Attributes:
        block: The chart block's position in the report's ``blocks``.
        finding: The finding the chart belongs to, by its position in the document; ``None`` for a chart code
            chose, which names no finding.
        viz_type: The chart's type.
        drawing: The renderer's drawing of the block's intent, in the renderer's own form.
    """

    block: int
    finding: int | None
    viz_type: ChartType
    drawing: DrawingT


class ChartRefusal(EvalBaseModel):
    """One chart of a report that was not drawn, and why — stated where the chart would have been."""

    block: int = Field(ge=0, description="The chart block's position in the report's blocks.")
    finding: int | None = Field(
        description="The finding the chart belongs to, by its position in the document; None for a chart code chose."
    )
    viz_type: ChartType = Field(description="The chart's type.")
    cause: ChartRefusalCause = Field(
        description="undrawable: the stored chart could not be decided into an intent; refused_by_renderer: the "
        "renderer refused the decided intent."
    )
    reason: str = Field(min_length=1, description="Why, naming the offending field or rule.")


@dataclass(frozen=True)
class ReportCharts[DrawingT]:
    """Every chart of a report: the ones drawn and the ones refused, each in report order.

    Attributes:
        drawn: Each chart the renderer drew.
        refused: Each chart that was not drawn, with why.
    """

    drawn: tuple[DrawnChart[DrawingT], ...]
    refused: tuple[ChartRefusal, ...]


def report_charts[DrawingT](report: Report, renderer: ChartRenderer[DrawingT]) -> ReportCharts[DrawingT]:
    """Draw every chart block of ``report`` with ``renderer``, refusing each chart that cannot be drawn on its own.

    Args:
        report: The report, as :func:`~threetears.evals.analysis.campaign_report` or any report builder returns it.
        renderer: The host's renderer, built for its style — the Vega adapter's
            :class:`~threetears.evals.vega.VegaRenderer`, or one of the host's own.

    Returns:
        The drawings and the refusals. Every chart block is in exactly one of the two.

    Raises:
        Exception: Whatever the renderer raises that is not a
            :class:`~threetears.evals.analysis.viz.ChartRefused` — a defect in the renderer or its compiler rather
            than a refusal of one chart.
    """
    drawn: list[DrawnChart[DrawingT]] = []
    refused: list[ChartRefusal] = []
    for position, block in enumerate(report.blocks):
        if not isinstance(block, ChartBlock):
            continue
        if block.intent is None:
            refused.append(
                ChartRefusal(
                    block=position,
                    finding=block.finding,
                    viz_type=block.viz_type,
                    cause="undrawable",
                    reason=block.error,
                )
            )
            continue
        try:
            drawing = renderer.draw(block.intent)
        except ChartRefused as exc:
            # NOSILENT: the refusal IS the answer -- one chart's failure is returned beside the others, by contract
            refused.append(
                ChartRefusal(
                    block=position,
                    finding=block.finding,
                    viz_type=block.viz_type,
                    cause="refused_by_renderer",
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        drawn.append(DrawnChart(block=position, finding=block.finding, viz_type=block.viz_type, drawing=drawing))
    return ReportCharts(drawn=tuple(drawn), refused=tuple(refused))


__all__ = ["ChartRefusal", "ChartRefusalCause", "DrawnChart", "ReportCharts", "report_charts"]
