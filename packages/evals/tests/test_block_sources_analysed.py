"""Block sources over what the analysis tools compute, and a lattice from the arms' own settings.

Contracts held here (``analysis.report_rows`` selectors ``compare``, ``settle``, ``levers``, ``bar_reads`` and
``config_grid``; ``analysis.report.block_spec``; the plan page's resolvers):

- **compare** draws two arms' difference with its interval, every row an ``arms_compare`` row, its figures as their
  keys read them; it names two arms, in order, and one measure; a pair the function refuses is refused with its
  reason; ``strata`` keeps places;
- **settle** draws ``settle_cases`` rows: bars of how often a fresh run reads the other side, a table of every figure
  as its key reads it; ``arms``, ``measures`` and ``strata`` narrow them;
- **levers** draws ``lever_effects``' effects and interactions on one measure at one place, each with its interval;
- **bar_reads** draws an arm against its absolute bar: its value, the interval it was read on and the bar's line;
- **a lattice draws from the arms' settings** where the snapshot holds no configs rows, its tiles the same as the
  lattice of the snapshot that holds them;
- **no number is recomputed**: every number a table of these prints is the figure its key reads;
- **reader words**: a compare block names its arms and measure by reader names, never by id or key, even where its
  interval is the paired estimate a contrast states;
- **one unit per size**: a settle table sets the cases and the trials a design needs beside today's cases and trials,
  and the repeats of a case.

Every expectation is read off the analysis functions' own results on the input (r1 to r5, the misfits, the shapes).
"""

from __future__ import annotations

import functools
import re
from typing import Any

import pytest

from packages.evals.tests.analyst_support import fixture_inputs, shape_snapshots
from threetears.evals.analysis.analyst.blocks import block_render
from threetears.evals.analysis.analyst.compare import arms_compare
from threetears.evals.analysis.analyst.levers import lever_effects
from threetears.evals.analysis.analyst.settle import settle_cases
from threetears.evals.analysis.analyst.snapshot import frozen_input
from threetears.evals.analysis.report.block_spec import block_look, complete_chart
from threetears.evals.analysis.report.charts import BarsChart, DotIntervalChart
from threetears.evals.analysis.report.plan_page.charting import measure_words
from threetears.evals.analysis.report.plan_page.resolve import resolve_chart
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_plan_facts import fact_arms
from threetears.evals.analysis.report_rows import config_grid, lever_arms, place_of, query_rows
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.kernel.report_plan import ChartQuery, PlannedChart

_EMPTY: dict[str, Any] = {
    "aspect": "",
    "measures": [],
    "arms": [],
    "strata": [],
    "limit": 0,
    "transform": "none",
    "small_multiples_by": "none",
}

_EVERY = 100_000


def _numbers(text: str) -> list[float]:
    """The numbers a cell spells, read as numbers: a values table pads a column to one precision, which moves none."""
    return [float(found.replace(",", "")) for found in re.findall(r"[+-]?\d[\d,]*(?:\.\d+)?", text)]


def _query(source: str, **fields: Any) -> ChartQuery:
    return ChartQuery.model_validate({**_EMPTY, "source": source, **fields})


@functools.cache
def _inputs() -> dict[str, ReportInput]:
    held = dict(fixture_inputs())
    held |= {name: frozen_input(snapshot) for name, snapshot in shape_snapshots("s").items()}
    return held


def _data(held: ReportInput, kind: str, query: ChartQuery) -> Any:
    planned = PlannedChart.model_validate({"kind": kind, "title": "", "query": query, "reference": [], "emphasis": ""})
    spec, faults = complete_chart(planned, held, "block")
    assert not faults, faults
    return resolve_chart(spec, held, block_look(held, fact_arms(held)), title="").chart.data


# =============================================================================
# compare
# =============================================================================


def _pairs() -> list[tuple[str, str, str, str]]:
    """Each input's first two arms and the first measure arms_compare reads them on: input, arms, measure."""
    found = []
    for name, held in _inputs().items():
        arms = [arm.id for arm in held.arms]
        if len(arms) < 2:
            continue
        for key in dict.fromkeys([*held.measures, *held.judged]):
            try:
                arms_compare(held, arms[:2], key)
            except RefusedInput:
                continue
            found.append((name, arms[0], arms[1], key))
            break
    assert len(found) > 3
    return found


@pytest.mark.parametrize("kind", ["dot_interval", "diverging_bars"])
def test_a_compare_chart_draws_each_place_arms_compare_reads_with_its_interval(kind: str) -> None:
    for name, first, second, key in _pairs():
        held = _inputs()[name]
        read = arms_compare(held, [first, second], key, limit=_EVERY)
        data = _data(held, kind, _query("compare", arms=[first, second], measures=[key]))
        rows = data.rows
        assert rows is not None
        drawn = {row.id: row for row in rows}
        for row in read.rows:
            assert isinstance(row.difference.value, float)
            shown = drawn[place_of(row.where, held)]
            factor = shown.value / row.difference.value if row.difference.value else 1.0
            assert factor in (pytest.approx(1.0), pytest.approx(1000.0)), factor
            if row.interval is not None and data.level:
                assert isinstance(row.interval.value, tuple)
                assert (shown.lo, shown.hi) == pytest.approx(tuple(end * factor for end in row.interval.value))


def test_a_compare_table_prints_every_figure_as_its_key_reads_it() -> None:
    for name, first, second, key in _pairs():
        held = _inputs()[name]
        read = arms_compare(held, [first, second], key, limit=_EVERY)
        block = block_render(
            held, block_kind="table", query=_query("compare", arms=[first, second], measures=[key]), limit=_EVERY
        )
        cells = [row.cells for row in block.rows]
        assert len(cells) == len(read.rows)
        for printed, row in zip(cells, read.rows, strict=True):
            assert [_numbers(cell) for cell in printed[1:4]] == [
                _numbers(figure.reads) for figure in (row.first, row.second, row.difference)
            ]


def test_a_compare_block_names_two_arms_and_the_order_is_the_difference() -> None:
    name, first, second, key = _pairs()[0]
    held = _inputs()[name]
    with pytest.raises(RefusedInput) as refused:
        block_render(held, block_kind="dot_interval", query=_query("compare", arms=[first], measures=[key]))
    assert "query.arms" in refused.value.parameters
    ahead = query_rows(_query("compare", arms=[first, second], measures=[key]), held)
    behind = query_rows(_query("compare", arms=[second, first], measures=[key]), held)
    for one, other in zip(ahead, behind, strict=True):
        assert one.item.difference.value == pytest.approx(-other.item.difference.value)  # type: ignore[union-attr]


def test_a_compare_block_the_function_refuses_is_refused_with_its_reason() -> None:
    for held in _inputs().values():
        arms = [arm.id for arm in held.arms][:2]
        for key in dict.fromkeys([*held.measures, *held.judged]):
            try:
                arms_compare(held, arms, key)
            except RefusedInput as why:
                with pytest.raises(RefusedInput) as refused:
                    block_render(held, block_kind="table", query=_query("compare", arms=arms, measures=[key]))
                assert "arms_compare refuses it" in str(refused.value.fault)
                assert str(why.fault) in str(refused.value.fault)
                return
    pytest.fail("no pair is refused by arms_compare on any fixture")


def test_a_compare_query_keeps_the_places_its_strata_name() -> None:
    kept = 0
    for name, first, second, key in _pairs():
        held = _inputs()[name]
        whole = query_rows(_query("compare", arms=[first, second], measures=[key]), held)
        if len(whole) < 2:
            continue
        one = query_rows(_query("compare", arms=[first, second], measures=[key], strata=[whole[-1].label]), held)
        assert [row.label for row in one] == [whole[-1].label]
        overall = query_rows(_query("compare", arms=[first, second], measures=[key], strata=["overall"]), held)
        assert [row.label for row in overall] == [row.label for row in whole if row.label == "overall"]
        kept += 1
    assert kept


# =============================================================================
# settle
# =============================================================================


def _settled() -> list[tuple[str, list[str]]]:
    found = []
    for name, held in _inputs().items():
        arms = []
        for arm in held.arms:
            try:
                settle_cases(held, [arm.id])
            except RefusedInput:
                continue
            arms.append(arm.id)
        if arms:
            found.append((name, arms))
    assert found
    return found


def test_settle_bars_draw_how_often_a_fresh_run_reads_the_other_side_for_every_row_settle_cases_reads() -> None:
    for name, arms in _settled():
        held = _inputs()[name]
        expected = [
            row.wrong_side.value
            for arm in arms
            for row in settle_cases(held, [arm], limit=_EVERY).rows
            if row.wrong_side is not None
        ]
        data = _data(held, "bars", _query("settle", arms=arms))
        assert isinstance(data, BarsChart) and data.rows is not None
        assert [row.value for row in data.rows] == pytest.approx([float(value) * 100 for value in expected])


def test_a_settle_table_prints_each_figure_as_its_key_reads_it_and_its_filters_narrow_it() -> None:
    for name, arms in _settled():
        held = _inputs()[name]
        rows = settle_cases(held, [arms[0]], limit=_EVERY).rows
        block = block_render(held, block_kind="table", query=_query("settle", arms=[arms[0]]), limit=_EVERY)
        assert len(block.rows) == len(rows)
        at = block.columns.index("A fresh run reads the other side")
        for printed, row in zip(block.rows, rows, strict=True):
            assert printed.cells[at] == (row.wrong_side.reads if row.wrong_side is not None else "—")
        whole = query_rows(_query("settle"), held)
        assert {row.arm for row in whole} == set(arms)
        assert {row.arm for row in query_rows(_query("settle", arms=[arms[-1]]), held)} == {arms[-1]}


# =============================================================================
# levers
# =============================================================================


def _levered() -> list[tuple[str, str]]:
    found = []
    for name, held in _inputs().items():
        for key in dict.fromkeys([*held.measures, *held.judged]):
            try:
                read = lever_effects(held, key, limit=_EVERY)
            except RefusedInput:
                continue
            if read.interactions:
                found.append((name, key))
                break
    assert found, "no fixture's levers cross"
    return found


def test_a_levers_chart_draws_each_effect_and_interaction_with_its_interval() -> None:
    for name, key in _levered():
        held = _inputs()[name]
        read = lever_effects(held, key, limit=_EVERY)
        data = _data(held, "dot_interval", _query("levers", measures=[key]))
        assert isinstance(data, DotIntervalChart)
        rows = {row.id: row for panel in (data.panels or [data]) for row in (panel.rows or [])}
        moves = {"→".join(lever_arms(item, held)): item.figures for item in [*read.effects, *read.interactions]}
        assert set(rows) == set(moves)
        for move, held_figures in moves.items():
            row = rows[move]
            point = held_figures[0].value
            assert isinstance(point, float)
            factor = row.value / point if point else 1.0
            assert factor in (pytest.approx(1.0), pytest.approx(1000.0))
            if len(held_figures) > 1 and data.level:
                interval = held_figures[1].value
                assert isinstance(interval, tuple)
                assert (row.lo, row.hi) == pytest.approx(tuple(end * factor for end in interval))


def test_a_levers_block_reads_one_place() -> None:
    name, key = _levered()[0]
    held = _inputs()[name]
    places = [task.id for task in held.tasks][:2]
    if len(places) < 2:
        pytest.skip("the fixture holds fewer than two tasks")
    with pytest.raises(RefusedInput) as refused:
        block_render(held, block_kind="table", query=_query("levers", measures=[key], strata=places))
    assert "query.strata" in refused.value.parameters


# =============================================================================
# bar_reads
# =============================================================================


def test_a_bar_reads_strip_draws_each_arms_value_and_interval_against_its_bars_line() -> None:
    drawn = 0
    for held in _inputs().values():
        reads = [read for read in held.bars if read.value is not None]
        if not reads:
            continue
        data = _data(held, "dot_interval", _query("bar_reads"))
        assert isinstance(data, DotIntervalChart)
        panels = data.panels or [data]
        assert len(panels) == len({(read.aspect, read.measure, read.bar, read.direction) for read in reads})
        for panel in panels:
            line = panel.reference or data.reference
            assert line is not None and line.kind in ("floor", "budget")
            for row in panel.rows or []:
                read = next(
                    read
                    for read in reads
                    if read.arm == row.id
                    and read.value is not None
                    and read.bar * (row.value / read.value if read.value else 1) == pytest.approx(line.value)
                )
                assert (row.hollow is True) == (read.status == "missed")
                if read.interval is not None:
                    factor = row.value / read.value if read.value else 1.0
                    assert (row.lo, row.hi) == pytest.approx(tuple(end * factor for end in read.interval))
                drawn += 1
    assert drawn


# =============================================================================
# a lattice from the arms' own settings
# =============================================================================


def test_a_lattice_draws_from_the_arms_settings_as_the_lattice_of_the_configs_the_snapshot_holds() -> None:
    with_configs, without = _inputs()["r3"], _inputs()["r3 without its configs"]
    assert with_configs.configs and not without.configs
    drawn = []
    for key in with_configs.measures:
        try:
            drawn.append(block_render(with_configs, block_kind="lattice", query=_query("configs", measures=[key])))
        except RefusedInput:
            continue
    assert drawn, "no measure of the configs the snapshot holds draws a lattice"
    query = drawn[0].query
    held = block_render(with_configs, block_kind="lattice", query=query, limit=_EVERY)
    built = block_render(without, block_kind="lattice", query=query, limit=_EVERY)
    assert built.columns == held.columns
    assert [row.cells for row in built.rows] == [row.cells for row in held.rows]


def test_the_grid_the_settings_make_is_the_configs_the_snapshot_holds() -> None:
    with_configs, without = _inputs()["r3"], _inputs()["r3 without its configs"]
    grid = config_grid(without)
    assert [(config.arm, config.levels) for config in grid.configs] == [
        (config.arm, config.levels) for config in with_configs.configs
    ]
    assert config_grid(with_configs) is with_configs


# =============================================================================
# reader words, and sizes beside today's count in one unit
# =============================================================================


def _paired_pairs() -> list[tuple[ReportInput, str, str, str]]:
    """Each input's pairs whose overall compare row is the paired estimate a contrast states: input, arms, measure."""
    found = []
    for held in _inputs().values():
        control = next((arm.id for arm in held.arms if arm.role == "control"), None)
        if control is None:
            continue
        for contrast in held.contrasts:
            try:
                read = arms_compare(held, [control, contrast.arm], contrast.measure)
            except RefusedInput:
                continue
            if any(row.basis is not None and str(row.basis.value).startswith("paired") for row in read.rows):
                found.append((held, control, contrast.arm, contrast.measure))
    assert found, "no fixture pairs two arms by a contrast"
    return found


@pytest.mark.parametrize("kind", ["dot_interval", "table"])
def test_a_compare_block_names_its_arms_and_measure_by_reader_names_never_by_id_or_key(kind: str) -> None:
    for held, control, candidate, key in _paired_pairs():
        query = _query("compare", arms=[control, candidate], measures=[key], strata=["overall"])
        block = block_render(held, block_kind=kind, query=query)
        printed = " ".join([*block.notes, *(cell for row in block.rows for cell in row.cells)])
        assert "paired" in printed
        names = block_look(held, fact_arms(held))
        for arm in (control, candidate):
            if arm != names.arm(arm).name and re.search(r"[_.:/=]", arm):
                assert not re.search(rf"(?<![\w.]){re.escape(arm)}(?![\w.])", printed), (arm, printed)
        # An id reads as an id where it holds a mark no reader name carries (the plan checks' own rule); a plain word
        # may also sit inside a reader name.
        if re.search(r"[_.:/=]", key) and key != measure_words(held, key):
            assert key not in printed, (key, printed)


def _repeated() -> ReportInput:
    """r2 with one gate, each arm's pass rate a count of trials over a third as many cases (three repeats each)."""
    held = _inputs()["r2 with one gate"]
    data = held.model_dump(mode="json", by_alias=True, exclude_none=True)
    for stat in data["measures"]["pass_rate"]["per_arm"].values():
        stat["n"], stat["n_cases"] = 1200, 400
    return ReportInput.model_validate(data)


def test_a_settle_table_sets_each_size_beside_todays_count_in_its_own_unit() -> None:
    held = _repeated()
    arm = held.arms[0].id
    rows = settle_cases(held, [arm], limit=_EVERY).rows
    block = block_render(held, block_kind="table", query=_query("settle", arms=[arm]), limit=_EVERY)
    at = {column: block.columns.index(column) for column in block.columns}
    for printed, row in zip(block.rows, rows, strict=True):
        cells = printed.cells
        assert row.trials_needed is not None and row.repeats and row.cases_needed is not None
        assert _numbers(cells[at["Cases needed, one-sided"]]) == _numbers(row.cases_needed.reads)
        assert _numbers(cells[at["Trials needed, one-sided"]]) == _numbers(row.trials_needed.reads)
        assert _numbers(cells[at["Repeats of a case"]]) == _numbers(row.repeats[0].reads)
        stat = held.measures["pass_rate"].per_arm[arm]
        assert _numbers(cells[at["Cases today"]]) == [float(stat.n_cases or 0)]
        assert _numbers(cells[at["Trials today"]]) == [float(stat.n or 0)]
