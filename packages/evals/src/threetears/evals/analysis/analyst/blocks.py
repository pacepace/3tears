"""The analyst's block tools: the evidence block a claim would show, as its values table, and what blocks there are.

**A block is a chart a page could draw** (a kind over a query of the report input), and :func:`block_render` draws
it the way a planned page draws a stored chart, by the same code:
:func:`~threetears.evals.analysis.report.block_spec.complete_chart` completes it (its reference line's key, interval
level, unit and axis), :func:`~threetears.evals.analysis.report.block_spec.chart_faults` and
:func:`~threetears.evals.analysis.report.block_spec.small_multiple_faults` hold it to the plan checks' own rules, and
the page's resolver draws it (:func:`~threetears.evals.analysis.report.plan_page.resolve.resolve_chart`, under
:func:`~threetears.evals.analysis.report.block_spec.block_look`), as
:func:`~threetears.evals.analysis.report.plan_page.drawable.undrawn_charts` draws a plan's. So a block the analyst
sees drawn is one a plan may store, and a block refused here is refused there. What comes back is the block's values
table (every value the chart would draw, as text), its level, unit, axis and reference line, and code's notes: never
the picture or the kit's chart data. Every fault is refused together, each named by the parameter it sits at.

**What there is to draw** is :func:`block_catalog`: each kind and the sources it draws
(:data:`~threetears.evals.analysis.report.block_spec.BLOCK_SOURCES`, the kit's chart models in
:data:`~threetears.evals.analysis.report.charts.KIT_CHARTS`), each source's rows and the query fields that narrow
them (:mod:`~threetears.evals.analysis.report_rows`), whether the snapshot holds any, and the figure key forms
(:func:`~threetears.evals.analysis.report.figure_keys.figure_key_grammar`). No kind is chosen for the analyst: every
block is open on every eval, and what refuses one is the data.

**A refusal sends the analyst somewhere.** Most go to the catalog and the overview. A ``prices`` block on a gain axis
whose frontier yields no step (one option or none on it, or none of its steps between two arms the query keeps) goes
to ``price_tradeoff`` on that axis, which shows the frontier and why every other option is off it, and lists the gain
axes whose frontier takes a step: the axes a ``prices`` block draws.

**Money in a values table** reads as the figure keys write it: its whole dollars grouped by thousands (``$1,239``,
:func:`~threetears.evals.analysis.report.charts.even_precision`).
"""

from __future__ import annotations

import functools
import re
from typing import get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from threetears.evals.analysis.analyst.cited import CitedFigure, cite
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.report.block_spec import (
    BLOCK_SOURCES,
    INTERVAL_KINDS,
    LINE_KINDS,
    PANELLED_KINDS,
    block_look,
    chart_faults,
    complete_chart,
    line_statistics,
    small_multiple_faults,
)
from threetears.evals.analysis.report.blocks import ChartBlock, TableBlock
from threetears.evals.analysis.report.charts import KIT_CHARTS, even_precision
from threetears.evals.analysis.report.figure_keys import figure_key_grammar, price_frontier, render
from threetears.evals.analysis.report.plan_page.resolve import resolve_chart
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_plan_facts import fact_arms
from threetears.evals.analysis.report_rows import query_rows
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.kernel.report_plan import ChartQuery, PlannedChart

__all__ = [
    "BLOCK_ROWS",
    "BlockCatalog",
    "RenderedBlock",
    "block_catalog",
    "block_render",
]

#: The values table's rows one page of a rendered block holds, by default.
BLOCK_ROWS = 40

#: The most characters of cells one page holds, whatever its limit: a page stops before the row that would pass it
#: (a page holds one row at least), so a table of long cells (an example's reply) pages within what one tool result
#: may take of an analyst's context.
_PAGE_CHARS = 8_000

#: The field path code's chart rules name a block's faults under.
_BLOCK = "block"

#: What a refused block's caller does instead.
_BLOCK_REMEDY = (
    "block_catalog lists each kind, the sources it draws and the sources this snapshot holds rows of; "
    "campaign_overview lists the ids a query may name"
)

#: Where a ``prices`` block whose gain axis yields no step sends its caller: the frontier on that axis, or another axis.
_STEPLESS_REMEDY = (
    "price_tradeoff with gain_key={gain_key!r} shows the frontier on that axis and why every other option is off it; "
    "a prices block draws the steps between neighbouring frontier options its query.arms keeps (empty keeps all); or "
    "name in query.measures a gain axis whose frontier takes a step: {axes}"
)

#: The parameters a ``prices`` block's step-less refusal sits at: its gain axis, and the arms it keeps.
_STEP_PARAMETERS = frozenset({"query.measures", "query.arms"})

#: What the remedy above says where no gain axis's frontier takes a step.
_NO_STEPPED_AXIS = "none does on this snapshot, so no prices block draws"


def _stepless_prices(report_input: ReportInput, query: ChartQuery) -> tuple[str, list[str]] | None:
    """Where a ``prices`` query reads one gain axis and selects no step: the axis, and every gain axis whose frontier
    takes a step, in the input's order (judged dimensions, then measures); None for any other query, and for a key
    naming no gain axis, which the chart's measures fault names."""
    if query.source != _PRICES or len(query.measures) != 1 or query_rows(query, report_input):
        return None
    steps: dict[str, bool] = {}
    for key in dict.fromkeys([*report_input.judged, *report_input.measures]):
        try:
            steps[key] = bool(price_frontier(report_input, key).steps)
        except UnresolvableReference:
            # NOSILENT: a measure stating no direction is no gain axis, and is listed as none.
            continue
    (gain_key,) = query.measures
    if gain_key not in steps:
        return None
    return gain_key, [key for key, stepped in steps.items() if stepped]


#: The source of price steps, which reads one gain axis.
_PRICES = "prices"

#: The figure key grammar's heading for its path forms, and how each form is written under it.
_FORMS_HEADING = "Paths:"
_FORM = re.compile(r"^- `([^`]+)`")

#: What a plain table draws, which no kit chart model says.
_TABLE_READS = "a plain table: every row the query selects, a column for each figure the source holds"


class BlockRow(BaseModel):
    """One row of a block's values table: its cells as the page writes them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cells: list[str] = Field(description="The row's cells, in the columns' order, as text.")


class RenderedBlock(BaseModel):
    """A block as a page would draw it, read without the picture: its values table and what code fixed about it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    block_kind: str = Field(description="The block's kind.")
    query: ChartQuery = Field(description="What it reads, every field filled.")
    reference_line: str | None = Field(
        description="The reference line it draws, as the call named it; None for none. It sits outside the query, so "
        "a report document's evidence carries it beside the query."
    )
    emphasis: str = Field(
        description="The one arm, stratum or task it stresses, by id, as the call named it; empty for none. It sits "
        "outside the query, so a report document's evidence carries it beside the query."
    )
    title: str = Field(description="Its title, its figure keys written out; empty for none.")
    level: str = Field(description="The interval level it prints; empty where it draws no interval.")
    unit: str = Field(description="The unit its values are in; empty for several (a panel each) or none.")
    domain: str = Field(description="Its value axis, low to high, where code fixes one; empty where it is fitted.")
    reference: CitedFigure | None = Field(description="Its reference line's figure, by its key; None for no line.")
    columns: list[str] = Field(description="The values table's columns.")
    rows: list[BlockRow] = Field(description="This page of its rows.")
    total: int = Field(description="How many rows the values table has.")
    offset: int = Field(description="The rows skipped before this page.")
    next_offset: int | None = Field(description="The offset of the next page; None on the last.")
    truncated: bool = Field(description="Whether rows follow this page.")
    notes: list[str] = Field(description="Code's lines under the block: what it leaves out, what bold means.")


class BlockKind(BaseModel):
    """One kind of block: what it draws, the sources it reads, and the lines and panels it takes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str = Field(description="The kind, as block_render's block_kind names it.")
    reads: str = Field(description="What it draws.")
    sources: list[str] = Field(description="The sources it draws, by name.")
    needs_interval: bool = Field(description="Whether every row it selects must carry an interval.")
    panels: bool = Field(description="Whether it draws small multiples (query.small_multiples_by).")
    references: list[str] = Field(description="The reference lines code can key on it.")
    line_statistics: list[str] = Field(description="The statistic a floor or budget line's criterion must read.")


class BlockSource(BaseModel):
    """One source a block reads: what its rows are, the query fields that narrow them, and whether any are here."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(description="The source, as query.source names it.")
    row_is: str = Field(description="What one row is.")
    label: str = Field(description="How a row is labelled.")
    interval: bool = Field(
        description="Whether a row, or each arm's figure in it, carries an interval where the snapshot holds one; "
        "a values table prints it beside the figure, at the block's level."
    )
    filters: list[str] = Field(description="The query fields that narrow its rows; the others leave them whole.")
    holds_rows: bool = Field(description="Whether this snapshot holds any row of it.")


class BlockCatalog(BaseModel):
    """Every block a snapshot can be shown by: the kinds, the sources, the query's options and the figure key forms."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot_id: str = Field(description="The snapshot read.")
    kinds: list[BlockKind] = Field(description="Every kind, the plain table last.")
    sources: list[BlockSource] = Field(description="Every source.")
    transforms: list[str] = Field(description="What query.transform may do to the values read.")
    small_multiples: list[str] = Field(description="What query.small_multiples_by may repeat a panel by.")
    figure_key_forms: list[str] = Field(
        description="Every figure key's path form, as the figure key grammar lists it; the analyst tool's help topic "
        "figure_keys gives the grammar's rules and filters and a page per group of paths, figure_keys.<group>."
    )


#: What each source's rows are, how they are labelled and whether they carry an interval, and the query fields that
#: narrow them (:mod:`~threetears.evals.analysis.report_rows`, whose selection the page and the checks read).
_SOURCE_ROWS: dict[str, tuple[str, str, bool, tuple[str, ...]]] = {
    "readings": ("an arm's reading on the query's aspect: its change from the control", "the arm", True,
                 ("arms", "aspect")),
    "contrasts": ("a contrast of an arm against the control on one measure", "arm/measure", True,
                  ("arms", "measures", "aspect")),
    "measures": ("an arm's figure on a measure", "measure/arm", True, ("arms", "measures", "aspect")),
    "judged": ("an arm's figure on a judged dimension", "dimension/arm", True, ("arms", "measures", "aspect")),
    "configs": ("an arm's figure on a measure or judged dimension, for each arm that ran a config", "key/arm", True,
                ("arms", "measures", "aspect")),
    "scoreboard": ("an arm's scoreboard row: trials ran, usable, passed, lost, and cost", "the arm", False,
                   ("arms", "aspect")),
    "floors": ("a floor read on an arm", "floor/arm", True, ("arms", "measures", "aspect")),
    "strata": ("a stratum, with each arm's figures in it", "its id", True, ("arms", "strata", "limit")),
    "tasks": ("a task, with each arm's figures in it", "its id", True, ("arms", "strata")),
    "examples": ("a failed trial the snapshot shows", "its trial, else its case", False,
                 ("arms", "strata", "aspect", "limit")),
    "cells": ("a target-matrix cell, with each arm's figures in it", "its id", True, ("strata",)),
    "confusion": ("a confusion table, one per axis", "the axis's key", False, ("measures",)),
    "human_pref": ("a slice of human preference for the arm rated against the control", "its id", True, ("arms",)),
    "prices": ("a step along the price frontier on a gain axis (a block names one in query.measures): its price per "
               "point a month at the declared volume, else per thousand of what the cost counts, ends off its gain's "
               "interval",
               "from→to", True, ("measures", "arms")),
}  # fmt: skip


def block_render(
    report_input: ReportInput,
    *,
    block_kind: str,
    query: ChartQuery,
    reference: str | None = None,
    emphasis: str = "",
    title: str = "",
    limit: int = BLOCK_ROWS,
    offset: int = 0,
) -> RenderedBlock:
    """The block a page would draw of ``query`` as a ``block_kind``, as its values table (this module's account).

    :param report_input: The snapshot's frozen report input.
    :ptype report_input: ReportInput
    :param block_kind: The kind of block.
    :ptype block_kind: str
    :param query: What it reads.
    :ptype query: ChartQuery
    :param reference: The reference line to draw (``margin``, ``floor``, ``budget``, ``no_preference``); None for none.
    :ptype reference: str | None
    :param emphasis: The one arm, stratum or task it stresses, by id; empty for none.
    :ptype emphasis: str
    :param title: Its title, figure keys allowed; empty for none.
    :ptype title: str
    :param limit: The most values-table rows on this page; a page of long cells stops sooner, its ``next_offset``
        naming where the next starts.
    :ptype limit: int
    :param offset: The rows to skip.
    :ptype offset: int
    :return: The block.
    :rtype: RenderedBlock
    :raises RefusedInput: The block cannot be drawn on this input, naming every fault at the parameter it sits at.
    """
    try:
        planned = PlannedChart.model_validate(
            {
                "kind": block_kind,
                "title": title,
                "query": query,
                "reference": [{"kind": reference}] if reference is not None else [],
                "emphasis": emphasis,
            }
        )
    except ValidationError as invalid:
        raise RefusedInput(
            f"the block is not one a page draws: {invalid}", remedy=_BLOCK_REMEDY, valid_values_from="block_catalog"
        ) from invalid
    spec, faults = complete_chart(planned, report_input, _BLOCK)
    faults += chart_faults(_BLOCK, spec, report_input)
    faults += small_multiple_faults(_BLOCK, spec)
    drawn = None
    written = title
    arms = fact_arms(report_input)
    try:
        written = render(title, report_input, {arm.id: arm.name for arm in arms})
    except UnresolvableReference as refused:
        faults.append((f"{_BLOCK}.title", str(refused)))
    if not faults:
        try:
            drawn = resolve_chart(spec, report_input, block_look(report_input, arms), title=written)
        except UnresolvableReference as refused:
            faults.append((_BLOCK, f"is refused by the page, which draws it from this input: {refused}"))
    if drawn is None:
        said = "; ".join(f"{_parameter(path)} {why}" for path, why in faults)
        at_fault = dict.fromkeys(_parameter(path) for path, _why in faults if path != _BLOCK)
        stepless = _stepless_prices(report_input, spec.query)
        if stepless is not None and set(at_fault) <= _STEP_PARAMETERS and _BLOCK not in (p for p, _ in faults):
            (gain_key, stepped) = stepless
            raise RefusedInput(
                f"the {block_kind!r} block over {query.source!r} cannot be drawn: {said}",
                remedy=_STEPLESS_REMEDY.format(
                    gain_key=gain_key, axes=", ".join(stepped) if stepped else _NO_STEPPED_AXIS
                ),
                parameters=list(at_fault),
                valid_values=stepped,
            )
        raise RefusedInput(
            f"the {block_kind!r} block over {query.source!r} cannot be drawn: {said}",
            remedy=_BLOCK_REMEDY,
            parameters=list(at_fault),
            valid_values_from="block_catalog",
        )
    columns, rows = _values_table(drawn.chart)
    page = _page(rows[offset : offset + limit])
    after = offset + len(page)
    more = after < len(rows)
    return RenderedBlock(
        block_kind=spec.kind,
        query=spec.query,
        reference_line=spec.reference[0].kind if spec.reference else None,
        emphasis=spec.emphasis,
        title=written,
        level=spec.level,
        unit=spec.unit,
        domain=" to ".join(format_number(end) for end in spec.domain),
        reference=cite(spec.reference[0].key, report_input) if spec.reference else None,
        columns=columns,
        rows=[BlockRow(cells=cells) for cells in page],
        total=len(rows),
        offset=offset,
        next_offset=after if more else None,
        truncated=more,
        notes=list(drawn.notes),
    )


def _page(rows: list[list[str]]) -> list[list[str]]:
    """The rows one page holds: the first, and each after it while the page's cells stay within its characters."""
    held: list[list[str]] = []
    spent = 0
    for cells in rows:
        spent += sum(len(cell) for cell in cells)
        if held and spent > _PAGE_CHARS:
            break
        held.append(cells)
    return held


def _parameter(path: str) -> str:
    """A fault's field path as the parameter the analyst set: ``block.query.arms`` as ``query.arms``, the kind as
    ``block_kind``, a reference line's as ``reference``; the block as a whole as ``the block``."""
    if path == _BLOCK:
        return "the block"
    field = path.removeprefix(f"{_BLOCK}.")
    if field == "kind":
        return "block_kind"
    if field.startswith("reference["):
        return "reference"
    return field


def _values_table(block: ChartBlock | TableBlock) -> tuple[list[str], list[list[str]]]:
    """A drawn block's values: a chart's table of every value it draws, or a table's own columns and cells, each cell
    after its arm's tag where it names an arm, as the page's Markdown prints it; and in either, every number at one
    precision among the cells that read alike (:func:`~threetears.evals.analysis.report.charts.even_precision`), so
    two figures compared read at one precision."""
    if isinstance(block, ChartBlock):
        headers, body = block.values_table()
        return headers, even_precision(body)
    columns = [column.header for column in block.columns]
    rows = [
        [cell.text if cell.tag is None else f"{cell.tag} · {cell.text}" for cell in (row[c.key] for c in block.columns)]
        for row in block.rows
    ]
    return columns, even_precision(rows)


def block_catalog(report_input: ReportInput, snapshot_id: str) -> BlockCatalog:
    """Every block the snapshot can be shown by (this module's account).

    :param report_input: The snapshot's frozen report input.
    :ptype report_input: ReportInput
    :param snapshot_id: The snapshot, as the freeze named it.
    :ptype snapshot_id: str
    :return: The catalog.
    :rtype: BlockCatalog
    """
    sources: tuple[str, ...] = get_args(ChartQuery.model_fields["source"].annotation)
    return BlockCatalog(
        snapshot_id=snapshot_id,
        kinds=[
            BlockKind(
                kind=kind,
                reads=_reads(kind),
                sources=[source for source in sources if source in drawn],
                needs_interval=kind in INTERVAL_KINDS,
                panels=kind in PANELLED_KINDS,
                references=list(LINE_KINDS),
                line_statistics=sorted(line_statistics(kind)),
            )
            for kind, drawn in BLOCK_SOURCES.items()
        ],
        sources=[_source(source, report_input) for source in sources],
        transforms=list(get_args(ChartQuery.model_fields["transform"].annotation)),
        small_multiples=list(get_args(ChartQuery.model_fields["small_multiples_by"].annotation)),
        figure_key_forms=list(_figure_key_forms()),
    )


def _reads(kind: str) -> str:
    """What a kind draws: its kit chart model's first paragraph, else a plain table's."""
    model = KIT_CHARTS.get(kind)
    if model is None or not model.__doc__:
        return _TABLE_READS
    (first, *_rest) = model.__doc__.strip().split("\n\n")
    return " ".join(first.split())


def _source(source: str, report_input: ReportInput) -> BlockSource:
    row_is, label, interval, filters = _SOURCE_ROWS[source]
    return BlockSource(
        source=source,
        row_is=row_is,
        label=label,
        interval=interval,
        filters=list(filters),
        holds_rows=bool(query_rows(_whole(source), report_input)),
    )


def _whole(source: str) -> ChartQuery:
    """A query of every row of one source: no filter, no limit."""
    return ChartQuery.model_validate(
        {
            "source": source,
            "aspect": "",
            "measures": [],
            "arms": [],
            "strata": [],
            "limit": 0,
            "transform": "none",
            "small_multiples_by": "none",
        }
    )


@functools.cache
def _figure_key_forms() -> tuple[str, ...]:
    """Every path form the figure key grammar lists, in its order."""
    lines = figure_key_grammar().splitlines()
    start = lines.index(_FORMS_HEADING) + 1
    forms: list[str] = []
    for line in lines[start:]:
        found = _FORM.match(line)
        if found is None:
            break
        forms.append(found.group(1))
    return tuple(forms)
