"""The chart vocabulary: what a chart on a report page reads, and what code completes it with.

A chart is a kind over a structured query of the report input (:class:`ChartQuery`), at most one reference line and
one thing it stresses (:class:`PlannedChart`, as an analyst asks for a block). Code completes it with the reference
line's figure key, the interval level, the unit and the value axis it reads off the data (:class:`ChartSpec`,
:func:`~threetears.evals.analysis.report.block_spec.complete_chart`), and the page's resolvers draw it. A report
document's evidence block (:mod:`threetears.evals.kernel.report_doc`) names its kind and reference line by the same
words (:data:`PlanChartKind`, :data:`ReferenceKind`).

The report planner's plan, which these charts were first written for, is retired: a stored plan is read only as
verbatim JSON (:class:`~threetears.evals.kernel.campaign.EvalReportPlan`), never validated against a model here.
**Field-level constraints only**: whether a chart names things its input holds is
:func:`~threetears.evals.analysis.report.block_spec.chart_faults`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from threetears.evals.schema.prose import ModelProse

#: An id a chart names something by: never empty.
_PlanId = Annotated[str, StringConstraints(min_length=1)]

#: The kit's chart kinds (``analysis.report.charts.KIT_CHARTS``, one name each) plus a plain table: what a plan's chart
#: and a report document's evidence block (:mod:`threetears.evals.kernel.report_doc`) may be drawn as.
PlanChartKind = Literal[
    "forest",
    "bars",
    "diverging_bars",
    "lines",
    "scatter_frontier",
    "gate_scatter",
    "dot_interval",
    "range_bars",
    "gate_matrix",
    "lattice",
    "heatmap",
    "diff_heatmap",
    "confusion",
    "examples",
    "table",
]
#: The part of the report input a chart reads.
_ChartSource = Literal[
    "readings",
    "contrasts",
    "measures",
    "judged",
    "scoreboard",
    "floors",
    "strata",
    "examples",
    "tasks",
    "configs",
    "cells",
    "confusion",
    "human_pref",
    "prices",
    "bar_reads",
    "compare",
    "settle",
    "levers",
]
#: What a chart does to the values it reads: per 1,000 replies, each arm less the control, or nothing.
_ChartTransform = Literal["none", "per_1k", "arm_minus_control"]
#: What a chart repeats its panel by.
_SmallMultiplesBy = Literal["none", "arm", "aspect", "task"]
#: What a reference line is: the kit's reference kinds (``analysis.report.charts.ChartReference.kind``), on a plan's
#: chart or a report document's evidence block.
ReferenceKind = Literal["floor", "budget", "margin", "no_preference"]

class _Plan(BaseModel):
    """A closed object: the schema sent is strict, the check on return is strict, and so is a stored read."""

    model_config = ConfigDict(extra="forbid")


class ChartQuery(_Plan):
    """What a chart reads from the report input, structured so code can check it names things that exist.

    Planner-written. An empty list or string means no filter there: every arm, every measure, every stratum, no
    aspect, no limit (``limit`` 0).
    """

    source: _ChartSource = Field(description="The part of the report input the chart reads.")
    aspect: str = Field(description="The aspect id it reads, for readings and anything per aspect; else empty.")
    measures: list[str] = Field(description="Measure or judged-dimension keys, as the source spells them.")
    arms: list[str] = Field(description="Arm ids to show, in order; empty for every arm.")
    strata: list[str] = Field(description="Stratum or task ids to keep; empty for all.")
    limit: int = Field(ge=0, description="The most rows to show (strata by cases, examples); 0 for no limit.")
    transform: _ChartTransform
    small_multiples_by: _SmallMultiplesBy


class _PlannedChartReference(_Plan):
    """A reference line as the planner asks for it: what kind of line it is."""

    kind: ReferenceKind


class ChartReference(_PlannedChartReference):
    """A reference line on a chart: what kind of line it is, and the figure it sits at. The planner picks the kind;
    code fills the key from the chart's aspect, and the renderer reads the value from it and writes the label."""

    key: _PlanId = Field(description="The figure key of the line's value, e.g. aspect.margin.")


class PlannedChart(_Plan):
    """A chart as the planner chooses it: kind, title, what it reads, a reference line and the one thing it stresses."""

    kind: PlanChartKind
    title: ModelProse = Field(description="The chart's title; empty when the section headline says it.")
    query: ChartQuery
    reference: Sequence[_PlannedChartReference] = Field(max_length=1, description="At most one reference line.")
    emphasis: str = Field(description="The one arm or row bolded or ringed, by id; empty for none.")


class ChartSpec(PlannedChart):
    """A chart as stored: the planner's choice plus what code reads off the data. ``level``, ``unit`` and ``domain``
    are code-filled; an empty ``domain`` lets the renderer fit the axis."""

    reference: Sequence[ChartReference] = Field(max_length=1, description="At most one reference line.")
    level: str = Field(description="The interval level printed on the chart, e.g. 95%; empty without intervals.")
    unit: str = Field(description="The unit its rows state; empty for several (a panel each) or none.")
    domain: list[float] = Field(max_length=2, description="The value axis, [low, high], or empty.")

    @field_validator("domain")
    @classmethod
    def _domain_is_a_range_or_empty(cls, domain: list[float]) -> list[float]:
        """Refuse a domain of one number: an axis has two ends, or the renderer fits it.

        :raises ValueError: The domain holds one number.
        """
        if len(domain) == 1:
            raise ValueError("`domain` is [low, high] or empty, never one number")
        return domain


__all__ = [
    "ChartQuery",
    "ChartReference",
    "ChartSpec",
    "PlanChartKind",
    "PlannedChart",
    "ReferenceKind",
]
