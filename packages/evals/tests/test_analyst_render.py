"""The analyst's block tools: a block is drawn as a page draws a chart, by the page's own code, and read as its values.

Contracts held here (``analysis.report.block_spec``, ``analysis.analyst.blocks``, ``ops.analyst_blocks``,
``actions.analyst_render``):

- **every kind and source the catalog accepts**, on r1 to r5, the misfit inputs and Kairo round 2, renders or is
  refused, and never raises anything else; a line drawn on any block that renders does the same;
- **each anchor plan's charts**, completed by today's code (a stored plan keeps the level code filled when it was
  stored; only that may move), read the same values table, level, unit, axis and reference line through
  ``block_render`` as through the planned page built from the plan;
- **a values table says what the input holds**: a plain table over several measures is a row per arm with a column
  per measure; every figure's interval the input holds is printed beside it, at the block's level; a scatter names
  every point by its reader name, never by an id the input names otherwise;
- **block_render echoes** the reference line and emphasis it drew;
- **one rule**: a chart the planner's completion refuses, by its checks or by the page, is refused by
  ``block_render`` with the same reasons, each named by the parameter it sits at, every fault together;
- **the catalog** lists every kind and source a page's chart takes, each kind's sources as the plan checks hold them,
  the same on every input (no kind, shape or spine chosen for the analyst); what each source's rows are and the query
  fields that narrow them are the selection's own (a field it does not list never narrows a row, a field it lists
  narrows one somewhere); whether the snapshot holds rows is read off the data; the figure key forms are the
  grammar's;
- **a page of rows** stops at its limit, or sooner where its cells run long, and the pages read back the whole table.
"""

from __future__ import annotations

from collections import Counter
import functools
import json
import re
from copy import deepcopy
from typing import Any, get_args

import pytest

from packages.evals.tests.analyst_support import KAIRO, aspect_tagged, budget_inputs, fixture_inputs, fixture_snapshots
from packages.evals.tests.fixtures.report_plans import dataset
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, ops_fixture
from packages.evals.tests.report_plan_support import planned_reply
from threetears.evals.actions import ActionOutcome, MountedTool, analyst_tools, eval_catalogue
from threetears.evals.analysis.analyst.blocks import RenderedBlock, block_catalog, block_render
from threetears.evals.analysis.analyst.snapshot import frozen_input
from threetears.evals.analysis.errors import PlanRefusal
from threetears.evals.analysis.numbers import WHOLE_FROM, format_number
from threetears.evals.analysis.report.block_spec import BLOCK_SOURCES, complete_chart
from threetears.evals.analysis.report.blocks import ChartBlock, TableBlock
from threetears.evals.analysis.report.charts import KIT_CHARTS, even_precision
from threetears.evals.analysis.report.figure_keys import figure_key_grammar, resolve
from threetears.evals.analysis.report.plan_page.build import plan_page
from threetears.evals.analysis.report.plan_page.charting import measure_words
from threetears.evals.analysis.report.plan_page.words import ASPECT_MEASURE_TEMPLATE
from threetears.evals.analysis.report_input import Cell, Contrast, ReportInput, Stratum, Task
from threetears.evals.analysis.report_plan_checks import raised_warnings
from threetears.evals.analysis.report_plan_facts import complete_plan
from threetears.evals.analysis.report_rows import SourceRow, query_rows
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.kernel.report_plan import ChartQuery, ChartReference, ChartSpec, PlannedChart, PlannedReport

ANCHORS = ("r1", "r2", "r3", "r4", "r5")

_KINDS: tuple[str, ...] = get_args(PlannedChart.model_fields["kind"].annotation)
_SOURCES: tuple[str, ...] = get_args(ChartQuery.model_fields["source"].annotation)
_LINES: tuple[str, ...] = get_args(ChartReference.model_fields["kind"].annotation)

#: Every query field, empty: no filter, no limit.
_EMPTY: dict[str, Any] = {
    "aspect": "",
    "measures": [],
    "arms": [],
    "strata": [],
    "limit": 0,
    "transform": "none",
    "small_multiples_by": "none",
}


def _query(source: str, **fields: Any) -> ChartQuery:
    return ChartQuery.model_validate({**_EMPTY, "source": source, **fields})


@functools.cache
def _inputs() -> dict[str, ReportInput]:
    """r1 to r5, the misfits, and Kairo round 2 as its snapshot froze it."""
    inputs = dict(fixture_inputs())
    inputs[KAIRO] = frozen_input(fixture_snapshots("s")[KAIRO])
    return inputs


def _drawn_or_refused(held: ReportInput, kind: str, source: str, line: str | None = None) -> RenderedBlock | None:
    """The block drawn, or None where it is refused; anything else it raises fails the test."""
    try:
        return block_render(held, block_kind=kind, query=_query(source), reference=line)
    except RefusedInput as refused:
        assert refused.fault and "cannot be drawn" in refused.fault, refused.fault
        assert "block_catalog" in refused.remedy
        return None


# =============================================================================
# Every kind and source, on every input
# =============================================================================


@pytest.mark.parametrize("name", list(_inputs()))
def test_every_kind_and_source_the_catalog_accepts_renders_or_refuses_and_never_raises(name: str) -> None:
    held = _inputs()[name]
    catalog = block_catalog(held, "s")
    drawn: list[tuple[str, str]] = []
    for kind in catalog.kinds:
        for source in kind.sources:
            block = _drawn_or_refused(held, kind.kind, source)
            if block is None:
                continue
            drawn.append((kind.kind, source))
            assert block.total >= len(block.rows) and block.offset == 0
            assert all(len(row.cells) == len(block.columns) for row in block.rows), (kind.kind, source)
            for line in _LINES:
                lined = _drawn_or_refused(held, kind.kind, source, line)
                if lined is not None:
                    assert lined.reference is not None
                    assert lined.reference.value == resolve(lined.reference.key, held).value
    assert drawn, f"no block draws on {name}: the sweep reaches no resolver"


def test_the_sweep_meets_refusals_as_well_as_blocks() -> None:
    """The control: on r1, which holds no aspect, a forest of readings is refused and a table of its tasks drawn."""
    held = _inputs()["r1"]
    assert not held.aspects
    assert _drawn_or_refused(held, "forest", "readings") is None
    assert _drawn_or_refused(held, "table", "tasks") is not None


# =============================================================================
# The anchors: the same values through block_render as through the planned page
# =============================================================================


def _values(block: ChartBlock | TableBlock) -> tuple[list[str], list[list[str]]]:
    """A page's block as its values table: a chart's own, a table's cells as its Markdown prints them, each number at
    one precision among the cells that read alike, as block_render reads them."""
    if isinstance(block, ChartBlock):
        headers, body = block.values_table()
        return headers, even_precision(body)
    cells = [[row[c.key] for c in block.columns] for row in block.rows]
    return [c.header for c in block.columns], even_precision(
        [[cell.text if cell.tag is None else f"{cell.tag} · {cell.text}" for cell in row] for row in cells]
    )


def _completed_now(spec: ChartSpec, held: ReportInput) -> ChartSpec:
    """A stored chart completed again by the code that completes charts now: a stored plan keeps the fields code
    filled when it was stored (a rule added later changes no stored plan), and block_render completes by today's."""
    chosen = PlannedChart.model_validate(
        {
            "kind": spec.kind,
            "title": spec.title,
            "query": spec.query,
            "reference": [{"kind": line.kind} for line in spec.reference],
            "emphasis": spec.emphasis,
        }
    )
    completed, faults = complete_chart(chosen, held, "chart")
    assert not faults, faults
    # Only a code-filled field may move: what the plan chose, its line's key, its unit and its axis stay as stored.
    assert completed.model_dump(exclude={"level"}) == spec.model_dump(exclude={"level"})
    return completed


@pytest.mark.parametrize("name", ANCHORS)
def test_each_anchor_plans_charts_read_the_same_through_block_render_as_on_the_planned_page(name: str) -> None:
    stored = dataset.load_plan(dataset.HERE / name / "anchor.plan.json")
    held = dataset.report_input_of(name)
    plan = stored.model_copy(
        update={
            "summary": [_completed_now(spec, held) for spec in stored.summary],
            "sections": [
                section.model_copy(update={"chart": _completed_now(section.chart, held)}) for section in stored.sections
            ],
        }
    )
    page = plan_page(plan, held)
    charts = [*zip(plan.summary, page.summary, strict=True)]
    charts += [(section.chart, shown.chart) for section, shown in zip(plan.sections, page.sections, strict=True)]
    assert charts
    for spec, shown in charts:
        block = block_render(
            held,
            block_kind=spec.kind,
            query=spec.query,
            reference=spec.reference[0].kind if spec.reference else None,
            emphasis=spec.emphasis,
            title=spec.title,
            limit=200,
        )
        where = f"{name}: {spec.kind} over {spec.query.source}"
        assert block.next_offset is None, where
        assert (block.columns, [row.cells for row in block.rows]) == _values(shown.chart), where
        assert block.notes == shown.notes, where
        assert (block.level, block.unit) == (spec.level, spec.unit), where
        assert block.domain == (f"{spec.domain[0]:g} to {spec.domain[1]:g}" if spec.domain else ""), where
        assert (block.reference.key if block.reference else None) == (spec.reference[0].key if spec.reference else None)


def test_an_anchor_draws_a_forest_with_a_margin_line_and_an_axis_code_fixed() -> None:
    """The control for the comparison above: r5's forest of readings carries a level, an axis and its margin line."""
    plan = dataset.load_plan(dataset.HERE / "r5" / "anchor.plan.json")
    forests = [s.chart for s in plan.sections if s.chart.kind == "forest" and s.chart.query.source == "readings"]
    assert forests and forests[0].domain and forests[0].reference and forests[0].level


# =============================================================================
# One rule: what the planner's completion refuses, block_render refuses
# =============================================================================


def _probe_reply(name: str, chart: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """The anchor's reply with one probe section after its own, drawing ``chart``."""
    plan: dict[str, Any] = json.loads((dataset.HERE / name / "anchor.plan.json").read_text(encoding="utf-8"))
    rows = plan["options"]["rows"]
    recommended = next((row["option"] for row in rows if row["role"] == "recommended"), None)
    raised = raised_warnings(dataset.report_input_of(name), recommended) if recommended is not None else []
    reply = planned_reply(plan, [warning.model_dump() for warning in raised])
    probe = deepcopy(reply["sections"][-1])
    probe.update(id="probe", eyebrow="Probe", headline="A probe chart", takeaway="", caption="", caveats=[])
    reply["sections"].append(probe | {"chart": chart})
    return reply, len(reply["sections"]) - 1


#: Charts r2's planner completion refuses, each at another layer: the checks, the completion, the panels, the page.
_REFUSED = {
    "a kind that cannot draw its source, naming an arm the input lacks": {
        "kind": "lattice",
        "query": {"source": "readings", "arms": ["no-such-arm"]},
        "reference": [],
    },
    "a line no figure sits at": {"kind": "bars", "query": {"source": "measures"}, "reference": ["no_preference"]},
    "panels on a kind the kit does not panel": {
        "kind": "gate_matrix",
        "query": {"source": "tasks", "small_multiples_by": "arm"},
        "reference": [],
    },
    "a chart the page cannot draw": {"kind": "bars", "query": {"source": "scoreboard"}, "reference": []},
}


@pytest.mark.parametrize("probe", list(_REFUSED))
def test_a_chart_the_planner_completion_refuses_is_refused_by_block_render_for_the_same_reasons(probe: str) -> None:
    chart = _REFUSED[probe]
    written = {
        "kind": chart["kind"],
        "title": "",
        "query": {**_EMPTY, **chart["query"]},
        "reference": [{"kind": kind} for kind in chart["reference"]],
        "emphasis": "",
    }
    reply, index = _probe_reply("r2", written)
    with pytest.raises(PlanRefusal) as planned:
        complete_plan(PlannedReport.model_validate(reply), dataset.report_input_of("r2"))
    at = f"sections[{index}].chart"
    whys = [why for path, why in planned.value.faults if path == at or path.startswith((f"{at}.", f"{at}["))]
    assert whys, f"the probe is not refused by the planner: {planned.value.faults}"

    with pytest.raises(RefusedInput) as refused:
        block_render(
            dataset.report_input_of("r2"),
            block_kind=chart["kind"],
            query=_query(**chart["query"]),
            reference=chart["reference"][0] if chart["reference"] else None,
        )
    for why in whys:
        assert why in refused.value.fault, f"{probe}: {why!r} is not in {refused.value.fault!r}"


@pytest.mark.parametrize(
    ("arguments", "said"),
    [
        (
            {"block_kind": "lattice", "query": _query("readings", arms=["no-such-arm"])},
            ["block_kind is 'lattice', which cannot draw 'readings'", "query.arms name ['no-such-arm']"],
        ),
        (
            {"block_kind": "bars", "query": _query("measures"), "reference": "no_preference"},
            ["reference is 'no_preference', and the report input holds no figure a no_preference line sits at"],
        ),
        (
            {"block_kind": "gate_matrix", "query": _query("tasks", small_multiples_by="arm")},
            ["query.small_multiples_by is 'arm', and the kit draws no small multiples of a 'gate_matrix'"],
        ),
        (
            {"block_kind": "bars", "query": _query("scoreboard")},
            ["the block is refused by the page, which draws it from this input"],
        ),
        (
            {"block_kind": "table", "query": _query("tasks"), "emphasis": "no-such-row"},
            ["emphasis is 'no-such-row', which is no arm, option, stratum or task to stress"],
        ),
        (
            {"block_kind": "table", "query": _query("tasks"), "title": "{pass_rate.no-such-arm}"},
            ["title ", "pass_rate.no-such-arm"],
        ),
    ],
    ids=["kind and arm", "a line", "panels", "the page", "emphasis", "a title's key"],
)
def test_a_refusal_names_every_fault_at_the_parameter_it_sits_at(arguments: dict[str, Any], said: list[str]) -> None:
    with pytest.raises(RefusedInput) as refused:
        block_render(dataset.report_input_of("r2"), **arguments)
    for words in said:
        assert words in refused.value.fault


def test_a_block_naming_no_control_on_an_input_without_one_is_refused_and_drawn_on_one_with_it() -> None:
    """r2 names no control: diverging bars of its measures would draw each arm less one it does not have."""
    held = dataset.report_input_of("r2")
    assert not any(arm.role == "control" for arm in held.arms)
    with pytest.raises(RefusedInput, match="which draws each arm less the control"):
        block_render(held, block_kind="diverging_bars", query=_query("measures", measures=["latency_ms"]))


# =============================================================================
# The catalog
# =============================================================================


def test_the_catalog_lists_every_kind_and_source_a_chart_takes_and_each_kinds_sources_as_the_checks_hold_them() -> None:
    catalog = block_catalog(_inputs()["r5"], "s")
    assert [kind.kind for kind in catalog.kinds] == list(BLOCK_SOURCES) and catalog.kinds[-1].kind == "table"
    assert set(_KINDS) == {kind.kind for kind in catalog.kinds}
    assert [source.source for source in catalog.sources] == list(_SOURCES)
    for kind in catalog.kinds:
        assert set(kind.sources) == BLOCK_SOURCES[kind.kind], kind.kind
        assert kind.panels == (kind.kind in KIT_CHARTS and "panels" in KIT_CHARTS[kind.kind].model_fields)
        assert kind.reads and set(kind.references) == {"margin", "floor", "budget"}
    by_kind = {kind.kind: kind for kind in catalog.kinds}
    assert by_kind["forest"].needs_interval and not by_kind["bars"].needs_interval
    assert by_kind["range_bars"].line_statistics == ["p50", "p95"] and by_kind["bars"].line_statistics == ["mean"]
    assert set(by_kind["table"].sources) == set(_SOURCES)
    assert catalog.transforms == list(get_args(ChartQuery.model_fields["transform"].annotation))


def test_the_catalog_is_the_same_on_every_input_but_for_the_rows_each_holds() -> None:
    """No kind is chosen for the eval: the kinds and what each source is read the same everywhere; only the data's
    presence differs."""
    catalogs = {name: block_catalog(held, "s").model_dump() for name, held in _inputs().items()}
    for catalog in catalogs.values():
        for source in catalog["sources"]:
            del source["holds_rows"]
    first = next(iter(catalogs.values()))
    assert all(catalog == first for catalog in catalogs.values())
    text = json.dumps(first).lower()
    assert "spine" not in text and '"shape"' not in text


def test_the_catalog_says_which_sources_a_snapshot_holds_rows_of_from_the_data() -> None:
    r1, r5 = _inputs()["r1"], _inputs()["r5"]
    holds = {
        name: {s.source: s.holds_rows for s in block_catalog(held, "s").sources}
        for name, held in (("r1", r1), ("r5", r5))
    }
    assert not r1.aspects and holds["r1"]["readings"] is False
    assert r5.aspects[0].readings and holds["r5"]["readings"] is True
    for name, held in _inputs().items():
        for source in block_catalog(held, "s").sources:
            assert source.holds_rows == bool(query_rows(_query(source.source), held)), (name, source.source)


def _changed(held: ReportInput, source: str, field: str) -> bool | None:
    """Whether setting one query field to something the input holds changes a source's rows; None where the input
    holds nothing to set it to."""
    whole = query_rows(_query(source), held)
    keys = [row.key for row in whole if row.key] or list(held.measures)
    slices = [row.label for row in whole] if source in ("strata", "tasks", "cells") else []
    slices = slices or [s.id for s in held.strata] or [t.id for t in held.tasks]
    values: dict[str, Any] = {
        "arms": [held.arms[-1].id],
        "aspect": held.aspects[-1].id if held.aspects else None,
        "measures": keys[-1:] or None,
        "strata": slices[-1:] or None,
        "limit": 1,
    }
    if values[field] is None:
        return None
    rows = query_rows(_query(source, **{field: values[field]}), held)
    return [(row.label, row.arm, row.key) for row in rows] != [(row.label, row.arm, row.key) for row in whole]


#: The listed filters no fixture can show narrowing, and why: every fixture's aspects are read alike or one, no floor
#: read differs from another on one arm by measure alone, and every slice holds a figure for every arm.
_UNEXERCISED_FILTERS = {
    *(
        (source, "aspect")
        for source in ("readings", "measures", "judged", "configs", "scoreboard", "floors", "examples")
    ),
    ("floors", "measures"),
    ("strata", "arms"),
    ("tasks", "arms"),
    # Only r1 reading only bars holds bar reads, on one aspect and one measure.
    ("bar_reads", "aspect"),
    ("bar_reads", "measures"),
    # Two arms compared, and lever effects, are read on the arms and the one measure a query names, so a query of
    # every row selects none and these fields alone change none here; test_block_sources_analysed holds them.
    *(("compare", field) for field in ("arms", "measures", "strata")),
    ("levers", "arms"),
    ("levers", "strata"),
}

#: The sources said to carry an interval whose rows no fixture gives one: no fixture's judged or config figure states
#: its ``ci``; r1 reading only bars reads its bars on no interval; and a query of every row selects no compare or lever
#: row (test_block_sources_analysed holds theirs).
_UNEXERCISED_INTERVALS = {"judged", "configs", "bar_reads", "compare", "levers"}


def test_each_sources_filters_are_the_query_fields_that_change_its_rows_and_no_other() -> None:
    changes: dict[tuple[str, str], bool] = {}
    for name, held in _inputs().items():
        for source in block_catalog(held, "s").sources:
            for field in ("arms", "aspect", "measures", "strata", "limit"):
                changed = _changed(held, source.source, field)
                if changed is None:
                    continue
                if field not in source.filters:
                    assert not changed, f"{field} changes {source.source}'s rows on {name}; the catalog lists none"
                changes[source.source, field] = changes.get((source.source, field), False) or changed
    listed = {(s.source, field) for s in block_catalog(_inputs()["r5"], "s").sources for field in s.filters}
    unshown = {pair for pair in listed if not changes.get(pair)}
    assert unshown == _UNEXERCISED_FILTERS


def _carries(row: SourceRow) -> bool:
    """Whether a row carries an interval: its own, or one on an arm's figure in it (a slice's, a cell's)."""
    if row.interval is not None:
        return True
    per_arm = getattr(row.item, "per_arm", None)
    if not isinstance(row.item, Stratum | Task | Cell) or per_arm is None:
        return False
    return any(stat.ci is not None or stat.half_width is not None for on in per_arm.values() for stat in on.values())


def test_a_source_said_to_carry_no_interval_never_does_and_one_said_to_carries_one_somewhere() -> None:
    carried: dict[str, bool] = {}
    for held in _inputs().values():
        for source in block_catalog(held, "s").sources:
            rows = query_rows(_query(source.source), held)
            some = any(_carries(row) for row in rows)
            if not source.interval:
                assert not some, source.source
            carried[source.source] = carried.get(source.source, False) or some
    said = {source.source for source in block_catalog(_inputs()["r5"], "s").sources if source.interval}
    assert {source for source in said if not carried[source]} == _UNEXERCISED_INTERVALS


def test_the_catalogs_figure_key_forms_are_the_grammars_path_forms() -> None:
    forms = block_catalog(_inputs()["r1"], "s").figure_key_forms
    grammar = figure_key_grammar()
    assert len(forms) > 20 and "criteria.<criterion>.<field>" in forms and "aspect.<field>" in forms
    for form in forms:
        assert f"- `{form}`:" in grammar, form
    assert len(forms) == grammar.count("\n- `") - grammar.split("Filters", 1)[1].count("\n- `")


# =============================================================================
# Pages of rows
# =============================================================================


#: r5's strata, on one judged dimension.
_STRATA = _query("strata", measures=["persona.holds_character"])


def test_pages_of_rows_read_back_the_whole_values_table() -> None:
    held = _inputs()["r5"]
    whole = block_render(held, block_kind="diverging_bars", query=_STRATA, limit=200)
    assert whole.total > 5 and whole.next_offset is None and not whole.truncated
    rows, offset = [], 0
    while True:
        page = block_render(held, block_kind="diverging_bars", query=_STRATA, limit=5, offset=offset)
        assert len(page.rows) <= 5 and page.offset == offset and page.total == whole.total
        rows += page.rows
        if page.next_offset is None:
            break
        assert page.truncated and page.next_offset == offset + len(page.rows)
        offset = page.next_offset
    assert rows == whole.rows


def test_a_page_of_long_cells_stops_before_its_limit_and_names_where_the_next_starts() -> None:
    held = _inputs()[KAIRO]
    page = block_render(held, block_kind="examples", query=_query("examples"), limit=40)
    assert page.total >= 40, "the control: Kairo's pool holds a full page of examples"
    assert 0 < len(page.rows) < 40 and page.next_offset == len(page.rows) and page.truncated
    following = block_render(held, block_kind="examples", query=_query("examples"), limit=40, offset=page.next_offset)
    assert following.rows[0] != page.rows[-1]


# =============================================================================
# What a values table says
# =============================================================================


def test_a_table_over_several_measures_is_a_row_per_arm_with_a_column_per_measure() -> None:
    """Paired figures sit side by side: an arm's figures are one row, each under its measure's column."""
    name, held, keys = next(
        (name, held, keys)
        for name, held in _inputs().items()
        if len(keys := [key for key, measure in held.measures.items() if measure.per_arm]) >= 2
    )
    drawn = block_render(held, block_kind="table", query=_query("measures", measures=keys))
    arms = {arm for key in keys for arm in held.measures[key].per_arm}
    assert drawn.total == len(arms), name
    measured = [column for column in drawn.columns if not column.endswith("interval")]
    assert len(measured) == 1 + len(keys), (name, drawn.columns)


def _aspect_contrasts() -> dict[str, ReportInput]:
    """Inputs whose contrasts are read on several aspects: r4 with one arm's contrast on one measure key held on two
    (the lead aspect's untested), and the shape campaign s1, three asked aspects each contrasting every arm."""
    return {"r4 tagged": aspect_tagged(_inputs()["r4"])[0], "s1": budget_inputs()["s1"]}


@pytest.mark.parametrize("name", list(_aspect_contrasts()))
def test_a_contrasts_block_keeps_each_aspects_contrast_and_names_the_aspect_where_two_share_a_key(name: str) -> None:
    """A contrasts query naming no aspect selects every aspect's contrasts. Where two aspects hold a contrast of one
    arm on one measure key, the table and the change charts keep both, each under the measure's words naming its
    aspect, never one in place of the other; a measure whose contrasts sit on one aspect keeps its plain words. Bars
    asked for in panels by aspect draw a panel per aspect the measure's contrasts sit on, each titled by its name; a
    dot strip asked for them draws a strip per aspect, each naming it."""
    held = _aspect_contrasts()[name]
    names = {aspect.id: aspect.name for aspect in held.aspects}
    assert all(c.aspect is not None and names[c.aspect] for c in held.contrasts), "the control: every aspect is named"
    on = {
        measure: {c.aspect for c in held.contrasts if c.measure == measure}
        for measure in dict.fromkeys(c.measure for c in held.contrasts)
    }
    shared = Counter((c.arm, c.measure) for c in held.contrasts)
    assert max(shared.values()) > 1, "the control: two aspects hold one arm's contrast on one key"

    def words(contrast: Contrast) -> str:
        plain = measure_words(held, contrast.measure, kind=contrast.kind)
        assert contrast.aspect is not None
        split = len(on[contrast.measure]) > 1
        return ASPECT_MEASURE_TEMPLATE.format(measure=plain, aspect=names[contrast.aspect]) if split else plain

    table = block_render(held, block_kind="table", query=_query("contrasts"), limit=200)
    assert table.total == len(held.contrasts)
    measured = table.columns.index("Measure")
    assert Counter(row.cells[measured] for row in table.rows) == Counter(words(c) for c in held.contrasts)
    for key, aspects in on.items():
        stated = [c for c in held.contrasts if c.measure == key and c.delta is not None]
        if not stated:
            continue
        bars = block_render(held, block_kind="diverging_bars", query=_query("contrasts", measures=[key]), limit=200)
        assert bars.total == len(stated), key
        if len(aspects) > 1:
            grouped = bars.columns.index("Group")
            assert Counter(row.cells[grouped] for row in bars.rows) == Counter(words(c) for c in stated), key
        else:
            assert "Group" not in bars.columns, key
        query = _query("contrasts", measures=[key], small_multiples_by="aspect")
        panelled = block_render(held, block_kind="diverging_bars", query=query, limit=200)
        assert panelled.total == len(stated), key
        if len({c.aspect for c in stated}) > 1:
            titled = panelled.columns.index("Panel")
            assert Counter(row.cells[titled] for row in panelled.rows) == Counter(
                names[c.aspect] for c in stated if c.aspect is not None
            ), key
        strips = block_render(held, block_kind="dot_interval", query=query, limit=200)
        assert strips.total == len(stated), key
        if len(aspects) > 1:
            stripped = strips.columns.index("Panel")
            assert Counter(row.cells[stripped] for row in strips.rows) == Counter(words(c) for c in stated), key


#: The sources that read one aspect where a query names none, and a kind that panels each.
_ONE_ASPECT = (("measures", "bars"), ("judged", "bars"), ("scoreboard", "bars"), ("readings", "diverging_bars"))


def _lead(held: ReportInput) -> str:
    asked = [aspect.id for aspect in held.aspects if aspect.asked]
    return asked[0] if asked else held.aspects[0].id


def _aspects_holding(held: ReportInput, source: str) -> list[str]:
    """The aspects the input holds a source's rows on, the lead first: figures by arm (the lead's) or by aspect, a
    scoreboard (the lead's is the input's own), readings."""
    lead = _lead(held)
    if source == "readings":
        found = [aspect.id for aspect in held.aspects if aspect.readings]
    elif source == "scoreboard":
        found = [lead] * bool(held.scoreboard) + [aspect.id for aspect in held.aspects if aspect.scoreboard]
    else:
        tables = (held.measures if source == "measures" else held.judged).values()
        found = [lead] * any(table.per_arm for table in tables)
        found += [aspect.id for aspect in held.aspects if any(table.by_aspect.get(aspect.id) for table in tables)]
    return sorted(set(found), key=[lead, *(aspect.id for aspect in held.aspects)].index)


def _as_lead(held: ReportInput, source: str, aspect: str) -> ReportInput:
    """The input with one aspect's rows where a query naming none reads them: its figures by arm (an aspect holding
    none of a measure reads the lead's, as the source does), its scoreboard, or it the one aspect asked and holding
    readings."""
    data = held.model_dump(mode="json", by_alias=True, exclude_none=True)
    if source in ("measures", "judged"):
        for table in data[source].values():
            table["per_arm"] = table.get("by_aspect", {}).get(aspect, table["per_arm"])
            table["by_aspect"] = {}
    elif source == "scoreboard":
        own = next(one for one in data["aspects"] if one["id"] == aspect).get("scoreboard")
        data["scoreboard"] = own if own is not None else data["scoreboard"]
    else:
        for one in data["aspects"]:
            one["asked"] = one["id"] == aspect
            if one["id"] != aspect:
                one["readings"] = []
    return ReportInput.model_validate(data)


def _keys(held: ReportInput, source: str) -> list[list[str]]:
    """The measure sets a block reads in one unit: each measure or judged dimension alone, the scoreboard's count of
    usable trials, or all readings."""
    if source in ("measures", "judged"):
        return [[key] for key in (held.measures if source == "measures" else held.judged)]
    return [["usable"]] if source == "scoreboard" else [[]]


@pytest.mark.parametrize("name", ["s1", "s3"])
def test_panels_by_aspect_over_a_source_reading_one_aspect_are_refused_naming_a_block_per_aspect(name: str) -> None:
    """Measures, judged figures and the scoreboard read the lead aspect's rows where a query names no aspect, and
    readings the one asked aspect's, so panels by aspect would show one aspect as all of them. Where the input holds
    the rows on several, the block is refused at ``query.aspect``, listing each aspect; one block per aspect, naming
    it, draws that aspect's rows: the lead's by-arm figures, another's own, as a block naming none draws an input
    whose lead rows they are. Where the input holds them on one aspect, nothing is refused for it."""
    held = budget_inputs()[name]
    refused: set[str] = set()
    compared: set[str] = set()
    for source, kind in _ONE_ASPECT:
        aspects = _aspects_holding(held, source)
        for measures in _keys(held, source):
            query = _query(source, measures=measures, small_multiples_by="aspect")
            if len(aspects) < 2:
                try:
                    block_render(held, block_kind=kind, query=query, limit=200)
                except RefusedInput as refusal:
                    assert "one block per aspect" not in refusal.fault, (source, refusal.fault)
                continue
            with pytest.raises(RefusedInput) as caught:
                block_render(held, block_kind=kind, query=query, limit=200)
            refusal = caught.value
            assert refusal.parameters == ("query.aspect",), (source, refusal.parameters)
            assert refusal.valid_values == tuple(aspects), (source, refusal.valid_values)
            assert "query.aspect" in refusal.remedy and all(aspect in refusal.remedy for aspect in aspects)
            refused.add(source)
            for aspect in aspects:
                named = _query(source, measures=measures, aspect=aspect)
                try:
                    drawn = block_render(held, block_kind=kind, query=named, limit=200)
                except RefusedInput as other:
                    assert "one block per aspect" not in other.fault, (source, aspect, other.fault)
                    continue
                expected = block_render(
                    _as_lead(held, source, aspect), block_kind=kind, query=_query(source, measures=measures), limit=200
                )
                assert (drawn.columns, [row.cells for row in drawn.rows]) == (
                    expected.columns,
                    [row.cells for row in expected.rows],
                ), (source, measures, aspect)
                compared.add(source)
    assert refused, "the control: the campaign holds some one-aspect source's rows on several aspects"
    assert compared == refused, "the control: every refused source draws one block per aspect somewhere"


def _holds_intervals(held: ReportInput, query: ChartQuery) -> bool:
    return any(_carries(row) for row in query_rows(query, held))


#: The kinds over figures whose values table prints each figure's interval where the input holds one, and the sources.
_INTERVAL_TABLES = (("table", "tasks"), ("table", "cells"), ("table", "measures"), ("heatmap", "cells"))


@pytest.mark.parametrize(("kind", "source"), _INTERVAL_TABLES, ids=[f"{k} {s}" for k, s in _INTERVAL_TABLES])
def test_a_values_table_prints_every_interval_the_input_holds_and_its_level(kind: str, source: str) -> None:
    seen = 0
    for name, held in _inputs().items():
        query = _query(source)
        if not _holds_intervals(held, query):
            continue
        try:
            drawn = block_render(held, block_kind=kind, query=query, limit=200)
        except RefusedInput:
            continue
        cells = [cell for row in drawn.rows for cell in row.cells]
        assert drawn.level, f"{name}: a {kind} over {source} prints intervals at no level"
        assert any(" to " in cell for cell in cells), f"{name}: a {kind} over {source} prints no interval"
        seen += 1
    assert seen, f"the control: some fixture holds an interval a {kind} over {source} reads"


def test_no_scatter_names_a_point_by_an_id_the_input_names_otherwise() -> None:
    seen = 0
    for name, held in _inputs().items():
        named = {arm.id for arm in held.arms if (arm.name or arm.short_name or arm.id) != arm.id} | {
            config.arm for config in held.configs
        }
        for kind in ("gate_scatter", "scatter_frontier"):
            for source in BLOCK_SOURCES[kind]:
                drawn = _drawn_or_refused(held, kind, source)
                if drawn is None:
                    continue
                at = drawn.columns.index("Point")
                points = {row.cells[at] for row in drawn.rows}
                assert not points & named, f"{name}: {kind} over {source} names points by id: {points & named}"
                seen += 1
    assert seen, "the control: some fixture draws a scatter"


def test_cells_that_read_alike_print_their_numbers_at_one_precision_and_lose_no_digit() -> None:
    rows = [["A", "94% · $1 per 1,000 calls", "-7.6 to +1.1"], ["B", "96.4% · $1.655 per 1,000 calls", "-3 to +0.25"]]
    assert even_precision(rows) == [
        ["A", "94.0% · $1.000 per 1,000 calls", "-7.6 to +1.10"],
        ["B", "96.4% · $1.655 per 1,000 calls", "-3.0 to +0.25"],
    ]
    assert even_precision([["p95 latency", "Few-shot (12 ex.)"]]) == [["p95 latency", "Few-shot (12 ex.)"]]
    assert even_precision([["$0.000616"], ["$1"]]) == [["$0.000616"], ["$1"]], "past three decimals, nothing is padded"


#: Values across magnitudes, as a latency or cost column holds them: each reaches even_precision as format_number
#: spells it, rounded to its own significant figures (whole from WHOLE_FROM up).
_MIXED_MAGNITUDES = (1239.7, 980.53, 12.3549, 1.23456, 0.5, 94.0)


def _significant(text: str) -> int:
    """How many significant figures a number's text states: its digits less the leading zeros; a whole number's
    trailing zeros count, since format_number writes a value it rounded to the units digit whole."""
    digits = re.sub(r"\D", "", text).lstrip("0")
    return len(digits)


def test_padding_never_states_a_digit_past_the_precision_a_number_was_rounded_to() -> None:
    """The Critic's case: ``1240`` beside ``1.655`` read ``$1,240.000`` for a true 1239.7. A padded number states no
    more significant figures than format_number rounded every number to (its most, read off the spelling of a
    number below WHOLE_FROM), and a number at or above WHOLE_FROM, which the rule printed whole, gains no decimals."""
    spelled = [format_number(value) for value in _MIXED_MAGNITUDES]
    most = max(
        _significant(text) for text, value in zip(spelled, _MIXED_MAGNITUDES, strict=True) if value != int(value)
    )
    for prefix, suffix in (("$", ""), ("", " ms"), ("", "%")):
        rows = [[f"{prefix}{text}{suffix}"] for text in spelled]
        padded = [row[0] for row in even_precision(rows)]
        for value, before, after in zip(_MIXED_MAGNITUDES, spelled, padded, strict=True):
            number = after.removeprefix(prefix).removesuffix(suffix).replace(",", "")
            assert float(number) == pytest.approx(float(before)), (before, after)
            assert number.startswith(before), f"padding only appends zeros: {before} -> {after}"
            if abs(value) >= WHOLE_FROM:
                assert "." not in number, f"a number printed whole gains no decimals: {after}"
            else:
                assert _significant(number) <= max(most, _significant(before)), (
                    f"{before} padded past its figures: {after}"
                )
    assert (
        "$1,240" in even_precision([["$1240"], ["$1.655"]])[0][0]
        and "." not in even_precision([["$1240"], ["$1.655"]])[0][0]
    )
    assert even_precision([["12.35"], ["1.235"]]) == [["12.35"], ["1.235"]], (
        "12.35 holds every figure it was rounded to"
    )


def test_a_tables_paired_figures_read_at_one_precision() -> None:
    """The prototype's complaint: 98% beside 98.6% in one table reads as a different precision, not a rounder figure."""
    name, held, keys = next(
        (name, held, keys)
        for name, held in _inputs().items()
        if len(keys := [key for key, measure in held.measures.items() if measure.per_arm]) >= 2
    )
    drawn = block_render(held, block_kind="table", query=_query("measures", measures=keys), limit=200)
    for column in range(1, len(drawn.columns)):
        cells = [row.cells[column] for row in drawn.rows if row.cells[column] not in ("", "—")]
        decimals = {len(cell.split(" ")[0].split(".")[1]) if "." in cell.split(" ")[0] else 0 for cell in cells}
        assert len(decimals) <= 1 or max(decimals) > 3, (name, drawn.columns[column], cells)


def _spread(values: list[float]) -> bool:
    return len(values) > 1 and min(values) > 0 and max(values) / min(values) > 10


def test_bars_over_a_magnitude_spanning_more_than_ten_times_draw_on_a_log_axis_and_say_so() -> None:
    """One rule, read off the figures: a magnitude whose bars span more than ten times is drawn on a log axis, a
    caption says so, and bars that do not stay linear."""
    seen: set[bool] = set()
    for held in _inputs().values():
        for key, measure in held.measures.items():
            if measure.unit in (None, "pct") or measure.grade is not None:
                continue
            stats = [stat for stat in measure.per_arm.values() if stat.mean is not None]
            if not stats or any(stat.ci is not None or stat.half_width is not None for stat in stats):
                continue
            try:
                drawn = block_render(held, block_kind="bars", query=_query("measures", measures=[key]))
            except RefusedInput:
                continue
            logged = any("log scale" in note for note in drawn.notes)
            assert logged == _spread([float(stat.mean) for stat in stats if stat.mean is not None]), key
            seen.add(logged)
    held = _inputs()["r1"]
    costs = [
        float(stat.mean)
        for task in held.tasks
        for figures in task.per_arm.values()
        if (stat := figures.get("cost_per_1k")) is not None and stat.mean is not None
    ]
    if costs:
        drawn = block_render(held, block_kind="bars", query=_query("tasks", measures=["cost_per_1k"]))
        logged = any("log scale" in note for note in drawn.notes)
        assert logged == _spread(costs)
        seen.add(logged)
    assert seen == {True, False}, "the control: some fixture's bars span more than ten times and some do not"


# =============================================================================
# Through the analyst's tool
# =============================================================================


def _tool() -> MountedTool:
    (spec,) = analyst_tools()
    return eval_catalogue().mount(spec)


async def _call(arguments: dict[str, Any]) -> ActionOutcome:
    fixture = off_loop(ops_fixture)
    snapshot = fixture_snapshots(TOYHOST_SCOPE)["r2"]
    off_loop(fixture.host.eval_host.storage.save_report_snapshot, snapshot)
    return await _tool().call({**arguments, "snapshot_id": snapshot.id}, host=fixture.host, caller=CALLER)


async def test_block_render_reads_back_a_values_table_and_its_line_by_its_key() -> None:
    outcome = await _call(
        {"action": "block_render", "block_kind": "bars", "query": {"source": "measures", "measures": ["latency_ms"]}}
    )
    assert outcome.structured is not None, outcome.text
    assert outcome.structured["query"] == {**_EMPTY, "source": "measures", "measures": ["latency_ms"]}
    assert outcome.structured["columns"] and outcome.structured["rows"]
    assert outcome.text.startswith("rendered block\nblock_kind: bars\n")


async def test_block_render_echoes_the_line_and_the_emphasis_it_drew_beside_its_query() -> None:
    """A report document's evidence carries a block's line and emphasis beside its query, so the block says both."""
    emphasis = next(arm.id for arm in fixture_inputs()["r2"].arms if arm.role != "control")
    query = {"source": "measures", "measures": ["latency_ms"]}
    outcome = await _call(
        {
            "action": "block_render",
            "block_kind": "range_bars",
            "query": query,
            "reference": "budget",
            "emphasis": emphasis,
        }
    )
    assert outcome.structured is not None, outcome.text
    assert outcome.structured["reference_line"] == "budget" and outcome.structured["emphasis"] == emphasis
    assert "\nreference_line: budget\n" in outcome.text and f"\nemphasis: {emphasis}\n" in outcome.text
    plain = await _call({"action": "block_render", "block_kind": "range_bars", "query": query})
    assert plain.structured is not None, plain.text
    assert plain.structured["reference_line"] is None and plain.structured["emphasis"] == ""


async def test_block_render_refused_on_the_snapshot_names_its_faults_and_the_catalog() -> None:
    outcome = await _call({"action": "block_render", "block_kind": "lattice", "query": {"source": "readings"}})
    assert outcome.refusal is not None and outcome.refusal.kind == "validation_failed"
    assert "block_kind is 'lattice', which cannot draw 'readings'" in outcome.text
    assert "block_catalog lists each kind" in outcome.text


@pytest.mark.parametrize(
    ("arguments", "said"),
    [
        ({"block_kind": "bars", "query": {"measures": []}}, "query is not a chart query"),
        ({"block_kind": "bars", "query": {"source": "measures", "colour": "red"}}, "query is not a chart query"),
        ({"block_kind": "pie", "query": {"source": "measures"}}, "block_kind is 'pie'"),
        ({"block_kind": "bars", "query": {"source": "measures"}, "reference": "ceiling"}, "reference is 'ceiling'"),
    ],
    ids=["no source", "a field no query holds", "a kind no page draws", "a line no page draws"],
)
async def test_a_call_that_is_no_block_is_refused_as_the_call(arguments: dict[str, Any], said: str) -> None:
    outcome = await _call({"action": "block_render", **arguments})
    assert outcome.refusal is not None and outcome.refusal.kind == "validation_failed"
    assert said in outcome.text


async def test_block_catalog_reads_back_through_the_tool() -> None:
    outcome = await _call({"action": "block_catalog"})
    assert outcome.structured is not None, outcome.text
    assert [kind["kind"] for kind in outcome.structured["kinds"]] == list(BLOCK_SOURCES)
    assert "figure_key_forms:" in outcome.text
