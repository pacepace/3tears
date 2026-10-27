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
- **a paged result says how much of the whole it holds, in what it counts**: ``all N cards shown`` on a complete page,
  else which it holds and how many follow, before the next page's call; and where it counts none (a sweep with no
  flips), that there are none to page, never that the page shows nothing.

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
from threetears.evals.analysis.report.plan_page.words import SETTLE_NO_SIZE
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

#: How a compare block says its interval may be too narrow: as a may, on what the interval counts.
NARROW_IF = "may be too narrow if these intervals count trials nested in cases"

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
                    unsized = SETTLE_NO_SIZE if row.settles == "on_the_line" and column.startswith("Needed") else "—"
                    assert printed == (figure.reads if figure is not None else unsized), (name, column, printed)
            for at, column in enumerate(block.columns[1:], start=1):
                assert any(printed.cells[at] not in ("", "—") for printed in block.rows), (name, column)
            note = block.notes[0]  # what its figure is; what would settle a row on its line follows
            named = note.split('"')[1] if note.count('"') == 2 else ""
            assert "bar" not in note.split('"')[0] and named in block.columns, (name, note)
            seen += 1
    assert seen, "the control: some fixture settles an arm"


def test_a_settle_bar_chart_says_each_bar_is_the_figure() -> None:
    held = _inputs()["r4"]
    arm = next(arm.id for arm in held.arms if settle_cases(held, [arm.id], limit=_EVERY).rows)
    block = _drawn(held, "bars", _query("settle", arms=[arm]))
    assert block is not None and any(note.startswith("Each bar is") for note in block.notes), block


def test_a_settle_row_on_its_line_says_no_size_settles_it_and_what_would() -> None:
    held = _inputs()["r1"]
    seen = 0
    for arm in held.arms:
        try:
            rows = settle_cases(held, [arm.id], limit=_EVERY).rows
        except RefusedInput:
            continue
        on_line = [at for at, row in enumerate(rows) if row.settles == "on_the_line"]
        if not on_line:
            continue
        block = _drawn(held, "table", _query("settle", arms=[arm.id]))
        assert block is not None
        for at in on_line:
            for column in ("Needed, one-sided", "Needed, two-sided"):
                assert _column(block, column)[at] == SETTLE_NO_SIZE, (arm.id, column)
            would = rows[at].what_would
            assert any(note.lower().startswith(would.lower()) for note in block.notes), block.notes
        seen += 1
    assert seen, "the control: r1's pick reads exactly its gate's line in one task"


# =============================================================================
# A compare chart says once where its interval may be too narrow
# =============================================================================


def _paired(held: ReportInput) -> tuple[str, str, str]:
    """A candidate, the control and the measure a contrast pairs them on."""
    control = next(arm.id for arm in held.arms if arm.role == "control")
    contrast = next(contrast for contrast in held.contrasts if contrast.interval is not None)
    return contrast.arm, control, contrast.measure


def test_a_compare_block_whose_interval_may_be_too_narrow_says_so_once_with_each_arms_trials_and_cases() -> None:
    held = _inputs()["r4"]
    first, second, key = _paired(held)
    read = arms_compare(held, [first, second], key, limit=_EVERY)
    narrow = [row for row in read.rows if row.may_be_narrow]
    assert narrow, "the control: r4's replies nest in fewer conversations"
    for kind in ("diverging_bars", "table"):
        block = _drawn(held, kind, _query("compare", arms=[first, second], measures=[key], strata=["overall"]))
        assert block is not None
        said = [note for note in block.notes if "too narrow" in note]
        assert len(said) == 1, (kind, block.notes)
        for figure in narrow[0].nested:
            assert figure.reads in said[0], (kind, figure.reads, said[0])
        # A may, never a fact: the paired row's unit is unstated, so the note neither calls it counting trials nor
        # says reading it on the cases would widen it.
        assert NARROW_IF in said[0], said[0]
        assert "unit is unstated" in said[0], said[0]
        assert "would be wider" not in said[0] and "counting trials" not in said[0], said[0]
        assert "observations" not in said[0], "the control: no unpaired row on this block states a unit"


def test_a_compare_block_names_what_its_intervals_count_only_where_the_input_states_it() -> None:
    held = _inputs()["r4"]
    first, second, key = _paired(held)
    read = arms_compare(held, [first, second], key, limit=_EVERY)
    unpaired = [row for row in read.rows if row.may_be_narrow and not row.counted_unstated]
    assert unpaired, "the control: r4's slices read the two arms' own spreads"
    stated = {str(figure.value) for row in unpaired for figure in row.counts if figure.value != "cases"}
    assert stated == {"observations"}, stated
    place = unpaired[0].where.removeprefix("strata.")
    block = _drawn(held, "table", _query("compare", arms=[first, second], measures=[key], strata=[place]))
    assert block is not None
    (said,) = [note for note in block.notes if "too narrow" in note]
    assert NARROW_IF in said and "count: observations" in said, said
    assert "unit is unstated" not in said, "the block holds no paired row"


def test_a_compare_block_whose_arms_ran_each_case_once_says_nothing_of_narrow_intervals() -> None:
    data = _inputs()["r4"].model_dump(mode="json", by_alias=True, exclude_none=True)
    data["scoreboard"] = {
        arm: {**row, "ran": row["cases"], "usable": row["cases"]} for arm, row in data["scoreboard"].items()
    }
    held = ReportInput.model_validate(data)
    first, second, key = _paired(held)
    assert not any(row.may_be_narrow for row in arms_compare(held, [first, second], key, limit=_EVERY).rows)
    block = _drawn(held, "diverging_bars", _query("compare", arms=[first, second], measures=[key], strata=["overall"]))
    assert block is not None and not any("too narrow" in note for note in block.notes), block.notes


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
    text, _structured = render_page(
        "block_render", params, render_cited(page), page.model_dump(mode="json"), page.page_unit
    )
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
        assert page.text.splitlines()[-1] == "all 9 cards shown"

    first = await _call(fixture, {**asked, "query": {"source": "examples"}})
    assert first.structured is not None and len(first.text) <= TEXT_BUDGET
    held_rows = len(first.structured["rows"])
    assert 0 < held_rows < every.total
    longer = block_render(
        held, block_kind="examples", query=_query("examples"), limit=held_rows + 1, fits=lambda _: True
    )
    assert len(_text_of(longer, params)) > TEXT_BUDGET, "the page holds every card that fits, not fewer"
    left = every.total - held_rows
    follow = "follows" if left == 1 else "follow"
    assert first.text.splitlines()[-2] == f"cards 1 to {held_rows} of {every.total} shown; {left} {follow}"
    assert first.text.splitlines()[-1].startswith("next: action='block_render'")


class _Listing(BaseModel):
    """A paged listing, as an analyst action returns one."""

    rows: list[str]
    total: int
    offset: int
    next_offset: int | None


def _shown(listing: _Listing, params: BaseModel) -> tuple[str, dict[str, Any]]:
    return render_page("x_list", params, render_cited(listing), listing.model_dump(mode="json"), ("flip", "flips"))


def test_a_page_says_how_much_of_the_whole_it_holds_in_what_it_counts() -> None:
    params = _Params(snapshot_id="s")
    text, structured = _shown(_Listing(rows=["a", "b", "c"], total=3, offset=0, next_offset=None), params)
    assert text.splitlines()[-1] == "all 3 flips shown" and "next" not in structured
    text, _structured = _shown(_Listing(rows=["a", "b"], total=3, offset=0, next_offset=2), params)
    assert text.splitlines()[-2:] == [
        "flips 1 to 2 of 3 shown; 1 follows",
        "next: action='x_list', snapshot_id='s', offset=2",
    ]
    text, _structured = _shown(_Listing(rows=["a"], total=3, offset=0, next_offset=1), params)
    assert text.splitlines()[-2] == "flips 1 to 1 of 3 shown; 2 follow"
    text, _structured = _shown(_Listing(rows=["c"], total=3, offset=2, next_offset=None), params)
    assert text.splitlines()[-1] == "flips 3 to 3 of 3 shown; none follow"
    text, _structured = _shown(_Listing(rows=["a"], total=1, offset=0, next_offset=None), params)
    assert text.splitlines()[-1] == "the one flip shown"
    text, _structured = _shown(_Listing(rows=[], total=3, offset=4, next_offset=None), params)
    assert text.splitlines()[-1] == "none from flip 5: 3 flips in all"
    text, _structured = _shown(_Listing(rows=[], total=0, offset=0, next_offset=None), params)
    assert text.splitlines()[-1] == "no flips; the page above is the whole result"


async def test_a_sweep_with_no_flips_says_it_has_none_to_page_and_never_that_the_page_shows_nothing() -> None:
    """r4's margin sweep on an aspect with no reading flips nowhere, and its page still holds the segment and call."""
    fixture, snapshots = off_loop(_seeded)
    held = frozen_input(snapshots["r4"])
    seen = 0
    for aspect in held.aspects:
        page = await _call(
            fixture,
            {"action": "decision_sensitivity", "snapshot_id": snapshots["r4"].id, "parameter": f"margin:{aspect.id}"},
        )
        assert page.structured is not None, page.text
        if page.structured["total"] != 0:
            continue
        seen += 1
        assert page.structured["segments"], "the control: a sweep with no flips still holds its one segment"
        assert page.text.splitlines()[-1] == "no flips; the page above is the whole result"
        assert "none to show" not in page.text
    assert seen, "the control: r4 sweeps a margin with no flips"


async def test_a_complete_overview_says_all_of_it_is_shown() -> None:
    fixture, snapshots = off_loop(_seeded)
    snapshot = snapshots[KAIRO]
    overview = await _call(fixture, {"action": "campaign_overview", "snapshot_id": snapshot.id})
    assert overview.structured is not None and overview.structured.get("next_offset") is None, overview.text
    assert overview.text.splitlines()[-1] == f"all {overview.structured['total']} rows shown"
