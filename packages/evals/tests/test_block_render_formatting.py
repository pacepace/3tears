"""What a block prints, read as a reader reads it: the formatting and labels the regression prototype found wrong.

Contracts held here (``analysis.report.charts`` values tables, ``analysis.report.plan_page`` resolvers and words,
``analysis.analyst.blocks`` and ``actions.render`` pages):

- **a computed change prints as its key reads it**: a compare or lever chart's value and interval ends, in points,
  read as the figure keys spell them (one decimal), never at a float's four figures;
- **a settle table prints counts whole and leaves out what no row fills**: every size and count reads as its key
  spells it (``617``, never ``617.0``), no column but a measure's is a dash in every row, and its note names the
  column it explains, where a bar chart's names the bar;
- **a lever row's label is its own and says what it compares**: a move ends at the setting's value as the input
  states it (a declared level's name, else the value, a whole number whole), every row's label differs from every
  other's in its group, and the note speaks of interactions only where the block draws one;
- **a criterion's line is named by its criterion**: its reader name and declared kind (gate, floor, bar, guardrail)
  and the side that passes, and an overall criterion drawn on each task's figures says it is the overall line;
- **one arm-label rule**: a bar chart names an arm's row as every other chart does, its letter tag first;
- **a page of a block holds as many rows as fit in one result**, up to its limit, and says how many follow;
- **a paged result says how much of the whole it holds**: ``all N shown`` on a complete page, else its rows and how
  many follow, before the next page's call.

Every expectation is computed from the input the block reads (r1 to r5, the misfits, Kairo round 2).
"""

from __future__ import annotations

import functools
import re
from typing import Any

from pydantic import BaseModel

from packages.evals.tests.analyst_support import KAIRO, TEXT_BUDGET, fixture_inputs, fixture_snapshots
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, OpsFixture, ops_fixture
from threetears.evals.actions import ActionOutcome, MountedTool, analyst_tools, eval_catalogue
from threetears.evals.actions.render import render_cited, render_page
from threetears.evals.analysis.analyst.blocks import RenderedBlock, block_render
from threetears.evals.analysis.analyst.compare import arms_compare
from threetears.evals.analysis.analyst.levers import EffectRow, InteractionRow, lever_effects
from threetears.evals.analysis.analyst.settle import settle_cases
from threetears.evals.analysis.analyst.snapshot import frozen_input
from threetears.evals.analysis.report.figure_keys import arm_settings
from threetears.evals.analysis.report.plan_page.look import quantity
from threetears.evals.analysis.report.plan_page.charting import contrast_unit_of
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_margins import criterion_place
from threetears.evals.analysis.style import humanize
from threetears.evals.kernel import EvalReportSnapshot
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.kernel.report_plan import ChartQuery

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

#: More rows than any fixture's block holds: one page of all of them.
_EVERY = 200


def _query(source: str, **fields: Any) -> ChartQuery:
    return ChartQuery.model_validate({**_EMPTY, "source": source, **fields})


@functools.cache
def _inputs() -> dict[str, ReportInput]:
    """r1 to r5, the misfits, and Kairo round 2 as its snapshot froze it."""
    inputs = dict(fixture_inputs())
    inputs[KAIRO] = frozen_input(fixture_snapshots("s")[KAIRO])
    return inputs


def _drawn(held: ReportInput, kind: str, query: ChartQuery, reference: str | None = None) -> RenderedBlock | None:
    """The block, every row on one page, or None where the input cannot draw it."""
    try:
        return block_render(
            held, block_kind=kind, query=query, reference=reference, limit=_EVERY, fits=lambda _page: True
        )
    except RefusedInput:
        return None


def _numbers(text: str) -> list[float]:
    """The numbers a cell spells, as numbers."""
    return [float(found.replace(",", "")) for found in re.findall(r"[+-]?\d[\d,]*(?:\.\d+)?", text)]


def _ends(text: str) -> set[float]:
    """An interval's ends, as numbers: a key spells an interval of no width as its one end."""
    return set(_numbers(text))


def _column(block: RenderedBlock, name: str) -> list[str]:
    at = block.columns.index(name)
    return [row.cells[at] for row in block.rows]


def _interval_column(block: RenderedBlock) -> str:
    return next(column for column in block.columns if column.endswith("interval"))


# =============================================================================
# A computed change prints as its key reads it
# =============================================================================


def _points(held: ReportInput, key: str) -> bool:
    return quantity(contrast_unit_of(held, key), change=True).unit == "pts"


def test_a_compare_chart_in_points_prints_each_change_and_its_ends_as_their_keys_read_them() -> None:
    seen = 0
    for name, held in _inputs().items():
        arms = [arm.id for arm in held.arms]
        for key in dict.fromkeys([*held.measures, *held.judged]):
            if len(arms) < 2 or not _points(held, key):
                continue
            try:
                read = arms_compare(held, arms[:2], key, limit=_EVERY)
            except RefusedInput:
                continue
            block = _drawn(held, "diverging_bars", _query("compare", arms=arms[:2], measures=[key]))
            if block is None or not block.level:
                continue
            values, intervals = _column(block, "Value"), _column(block, _interval_column(block))
            for row, value, interval in zip(read.rows, values, intervals, strict=True):
                assert _numbers(value) == _numbers(row.difference.reads), (name, key, value, row.difference.reads)
                if row.interval is not None:
                    assert _ends(interval) == _ends(row.interval.reads), (name, key, interval)
            seen += 1
    assert seen, "the control: some fixture compares two arms in points"


def test_a_lever_chart_in_points_prints_each_effect_and_its_ends_as_their_keys_read_them() -> None:
    seen = 0
    for name, held in _inputs().items():
        for key in dict.fromkeys([*held.measures, *held.judged]):
            if not _points(held, key):
                continue
            try:
                read = lever_effects(held, key, limit=_EVERY)
            except RefusedInput:
                continue
            block = _drawn(held, "diverging_bars", _query("levers", measures=[key]))
            if block is None:
                continue
            values = _column(block, "Value")
            intervals = _column(block, _interval_column(block)) if block.level else [""] * len(values)
            for row, value, interval in zip([*read.effects, *read.interactions], values, intervals, strict=True):
                assert _numbers(value) == _numbers(row.figures[0].reads), (name, key, value, row.figures[0].reads)
                if len(row.figures) > 1 and interval:
                    assert _ends(interval) == _ends(row.figures[1].reads), (name, key, interval)
            seen += 1
    assert seen, "the control: some fixture's arms isolate a setting on a rate"


# =============================================================================
# A settle table prints counts whole and leaves out what no row fills
# =============================================================================

#: A settle table's columns that hold a size or a count, each to the figure of a settle row it prints.
_COUNTS = {
    "Needed, one-sided": "cases_needed",
    "Needed, two-sided": "cases_needed_two_sided",
    "Trials needed, one-sided": "trials_needed",
    "Trials needed, two-sided": "trials_needed_two_sided",
}


def test_a_settle_table_prints_every_size_as_its_key_reads_it_and_no_column_no_row_fills() -> None:
    seen = 0
    for name, held in _inputs().items():
        for arm in held.arms:
            try:
                rows = settle_cases(held, [arm.id], limit=_EVERY).rows
            except RefusedInput:
                continue
            block = _drawn(held, "table", _query("settle", arms=[arm.id]))
            if block is None or not rows:
                continue
            assert len(block.rows) == len(rows)
            for column, field in _COUNTS.items():
                if column not in block.columns:
                    assert all(getattr(row, field) is None for row in rows), (name, column)
                    continue
                for row, printed in zip(rows, _column(block, column), strict=True):
                    figure = getattr(row, field)
                    assert printed == (figure.reads if figure is not None else "—"), (name, column, printed)
            for at, column in enumerate(block.columns[1:], start=1):
                assert any(printed.cells[at] not in ("", "—") for printed in block.rows), (name, column)
            (note,) = block.notes
            named = note.split('"')[1] if note.count('"') == 2 else ""
            assert "bar" not in note.split('"')[0] and named in block.columns, (name, note)
            seen += 1
    assert seen, "the control: some fixture settles an arm"


def test_a_settle_bar_chart_says_each_bar_is_the_figure() -> None:
    held = _inputs()["r4"]
    arm = next(arm.id for arm in held.arms if settle_cases(held, [arm.id], limit=_EVERY).rows)
    block = _drawn(held, "bars", _query("settle", arms=[arm]))
    assert block is not None and any(note.startswith("Each bar is") for note in block.notes), block


# =============================================================================
# A lever row's label is its own and says what it compares
# =============================================================================


def _stated(held: ReportInput, setting: str, arm: str) -> str:
    """A setting's value on an arm as the input states it: its declared level's name, else the value, whole if whole."""
    value = arm_settings(held)[arm][setting]
    spelled = value if isinstance(value, str) else (str(int(value)) if float(value).is_integer() else str(value))
    factor = next((factor for factor in held.factors if factor.id == setting), None)
    for level in factor.levels if factor is not None else []:
        if spelled.casefold() in (level.id.casefold(), (level.name or level.id).casefold()):
            return level.name or level.id
    return spelled


def test_every_lever_row_is_labelled_apart_and_a_move_ends_at_the_value_the_input_states() -> None:
    seen = 0
    for name, held in _inputs().items():
        for key in dict.fromkeys([*held.measures, *held.judged]):
            try:
                read = lever_effects(held, key, limit=_EVERY)
            except RefusedInput:
                continue
            block = _drawn(held, "diverging_bars", _query("levers", measures=[key]))
            if block is None:
                continue
            grouped = "Group" in block.columns
            labels = [(row.cells[0] if grouped else "", row.cells[grouped]) for row in block.rows]
            assert len(set(labels)) == len(labels), (name, key, labels)
            for row, (_group, label) in zip(read.effects, labels, strict=False):
                assert label.endswith(f" → {_stated(held, row.setting, str(row.to_arm.value))}"), (name, label)
            interactions = bool(read.interactions)
            assert any("interaction" in note for note in block.notes) == interactions, (name, block.notes)
            seen += 1
    assert seen, "the control: some fixture's arms isolate a setting"


def test_an_interaction_names_both_moves_it_compares() -> None:
    held = _inputs()["r2"]
    read = lever_effects(held, "pass_rate", limit=_EVERY)
    assert read.interactions, "the control: r2's grid crosses its settings"
    block = _drawn(held, "table", _query("levers", measures=["pass_rate"]))
    assert block is not None
    moves = _column(block, "Move")[len(read.effects) :]
    assert len(set(moves)) == len(moves) == len(read.interactions), moves
    assert all(move.count("→") == 2 and " less " in move for move in moves), moves
    assert isinstance(read.interactions[0], InteractionRow) and isinstance(read.effects[0], EffectRow)


# =============================================================================
# A criterion's line is named by its criterion
# =============================================================================

#: The side of a criterion's line that passes, by its direction.
_SIDES = {"higher_is_better": "at least", "lower_is_better": "at most"}


def _line_rows(block: RenderedBlock, names: set[str]) -> list[str]:
    return [row.cells[0] for row in block.rows if any(row.cells[0].startswith(f"{one} ") for one in names)]


def test_a_gate_scatter_names_each_line_by_its_criterion_and_kind() -> None:
    seen = 0
    for name, held in _inputs().items():
        overall = [criterion for criterion in held.criteria if criterion_place(criterion) == "overall"]
        by_measure = {criterion.measure: criterion for criterion in overall}
        if len(by_measure) < 2:
            continue
        keys = list(by_measure)[:2]
        block = _drawn(held, "gate_scatter", _query("measures", measures=keys))
        if block is None:
            continue
        points = {row.cells[0] for row in block.rows}
        for criterion in (by_measure[key] for key in keys):
            reader = criterion.label or humanize("dimension", criterion.id)
            said = f"{reader} {criterion.kind}, {_SIDES[criterion.direction]} "
            assert any(point.startswith(said) for point in points), (name, said, points)
        assert not any(point.startswith(("floor ", "budget ")) for point in points), (name, points)
        seen += 1
    assert seen, "the control: some fixture gates two measures overall"


def test_an_overall_criterion_drawn_in_each_tasks_panel_says_it_is_the_overall_line() -> None:
    seen = 0
    for name, held in _inputs().items():
        for criterion in held.criteria:
            held_by_tasks = any(
                criterion.measure in figures for task in held.tasks for figures in task.per_arm.values()
            )
            if criterion_place(criterion) != "overall" or criterion.statistic != "mean" or not held_by_tasks:
                continue
            line = "floor" if criterion.direction == "higher_is_better" else "budget"
            block = _drawn(held, "dot_interval", _query("tasks", measures=[criterion.measure]), reference=line)
            if block is None:
                continue
            reader = criterion.label or humanize("dimension", criterion.id)
            at = block.columns.index("Row")
            lines = [row.cells[at] for row in block.rows if row.cells[at].startswith(f"{reader} {criterion.kind}, ")]
            assert lines and all(said.endswith(" overall") for said in lines), (name, criterion.id, lines)
            seen += 1
    assert seen, "the control: some fixture draws an overall criterion's line over its tasks"


# =============================================================================
# One arm-label rule
# =============================================================================


def test_a_bar_chart_names_each_arm_as_a_dot_strip_of_the_same_figures_does() -> None:
    seen = 0
    for name, held in _inputs().items():
        arms = {arm.id for arm in held.arms}
        for key, measure in held.measures.items():
            if not set(measure.per_arm) <= arms or len(measure.per_arm) < 2:
                continue
            bars = _drawn(held, "bars", _query("measures", measures=[key]))
            dots = _drawn(held, "dot_interval", _query("measures", measures=[key]))
            if bars is None or dots is None or "Panel" in bars.columns or "Panel" in dots.columns:
                continue
            # A dot strip may add a gated measure's line as a row of its own; every bar is an arm the strip names alike.
            assert {row.cells[0] for row in bars.rows} <= {row.cells[0] for row in dots.rows}, (name, key)
            seen += 1
    assert seen, "the control: some fixture draws one measure both ways"


# =============================================================================
# Pages
# =============================================================================


def _analyst_tool() -> MountedTool:
    (spec,) = analyst_tools()
    return eval_catalogue().mount(spec)


def _seeded() -> tuple[OpsFixture, dict[str, EvalReportSnapshot]]:
    fixture = ops_fixture()
    snapshots = fixture_snapshots(TOYHOST_SCOPE)
    for snapshot in snapshots.values():
        fixture.host.eval_host.storage.save_report_snapshot(snapshot)
    return fixture, snapshots


async def _call(fixture: OpsFixture, arguments: dict[str, Any]) -> ActionOutcome:
    return await _analyst_tool().call(arguments, host=fixture.host, caller=CALLER)


class _Params(BaseModel):
    """A paged call's parameters, as the tool validated them."""

    snapshot_id: str
    offset: int = 0


def _text_of(page: RenderedBlock, params: BaseModel) -> str:
    text, _structured = render_page("block_render", params, render_cited(page), page.model_dump(mode="json"))
    return text


async def test_a_page_of_example_cards_holds_as_many_as_fit_in_one_result_and_says_how_many_follow() -> None:
    fixture, snapshots = off_loop(_seeded)
    snapshot = snapshots[KAIRO]
    held = frozen_input(snapshot)
    params = _Params(snapshot_id=snapshot.id)
    every = block_render(held, block_kind="examples", query=_query("examples"), limit=_EVERY, fits=lambda _: True)
    assert every.total > 9, "the control: Kairo's pool holds more cards than one result"
    asked = {"action": "block_render", "snapshot_id": snapshot.id, "block_kind": "examples"}

    nine = _query("examples", limit=9)
    whole = block_render(held, block_kind="examples", query=nine, limit=_EVERY, fits=lambda _: True)
    page = await _call(fixture, {**asked, "query": {"source": "examples", "limit": 9}})
    assert page.structured is not None, page.text
    if len(_text_of(whole, params)) <= TEXT_BUDGET:
        assert len(page.structured["rows"]) == whole.total == 9, "nine cards that fit one result are one page"
        assert page.text.splitlines()[-1] == "all 9 shown"

    first = await _call(fixture, {**asked, "query": {"source": "examples"}})
    assert first.structured is not None and len(first.text) <= TEXT_BUDGET
    held_rows = len(first.structured["rows"])
    assert 0 < held_rows < every.total
    longer = block_render(
        held, block_kind="examples", query=_query("examples"), limit=held_rows + 1, fits=lambda _: True
    )
    assert len(_text_of(longer, params)) > TEXT_BUDGET, "the page holds every card that fits, not fewer"
    left = every.total - held_rows
    assert first.text.splitlines()[-2] == f"rows 1 to {held_rows} of {every.total} shown; {left} follow"
    assert first.text.splitlines()[-1].startswith("next: action='block_render'")


class _Listing(BaseModel):
    """A paged listing, as an analyst action returns one."""

    rows: list[str]
    total: int
    offset: int
    next_offset: int | None


def test_a_page_says_how_much_of_the_whole_it_holds() -> None:
    params = _Params(snapshot_id="s")
    whole = _Listing(rows=["a", "b", "c"], total=3, offset=0, next_offset=None)
    text, structured = render_page("x_list", params, render_cited(whole), whole.model_dump(mode="json"))
    assert text.splitlines()[-1] == "all 3 shown" and "next" not in structured
    first = _Listing(rows=["a", "b"], total=3, offset=0, next_offset=2)
    text, _structured = render_page("x_list", params, render_cited(first), first.model_dump(mode="json"))
    assert text.splitlines()[-2:] == [
        "rows 1 to 2 of 3 shown; 1 follow",
        "next: action='x_list', snapshot_id='s', offset=2",
    ]
    last = _Listing(rows=["c"], total=3, offset=2, next_offset=None)
    text, _structured = render_page("x_list", params, render_cited(last), last.model_dump(mode="json"))
    assert text.splitlines()[-1] == "rows 3 to 3 of 3 shown; none follow"
    empty = _Listing(rows=[], total=0, offset=0, next_offset=None)
    text, _structured = render_page("x_list", params, render_cited(empty), empty.model_dump(mode="json"))
    assert text.splitlines()[-1] == "none to show"


async def test_a_complete_overview_says_all_of_it_is_shown() -> None:
    fixture, snapshots = off_loop(_seeded)
    snapshot = snapshots[KAIRO]
    overview = await _call(fixture, {"action": "campaign_overview", "snapshot_id": snapshot.id})
    assert overview.structured is not None and overview.structured.get("next_offset") is None, overview.text
    assert overview.text.splitlines()[-1] == f"all {overview.structured['total']} shown"
