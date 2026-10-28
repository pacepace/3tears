"""Levels as the input states them, and the renderer's labels: what a block prints is what a reader can read.

Contracts held here (``analysis.report.figure_keys`` level figures, ``analysis.report.block_spec``,
``analysis.report.charts`` values tables, ``analysis.report.plan_page`` labels and money):

- **a level prints as stated**: a pair's, a spread's, a compare's and a lever's level key reads in percent to every
  digit the number rule keeps, so two levels that differ never print alike, and a chart refused for mixing levels
  (two contrasts at different levels) names them as the keys print them;
- **a chart of readings prints the level of the interval it draws**: a forest of readings draws each one's
  ``within_interval``, at the nominal level with no correction, and prints that level, as a reading's level key does,
  never the gain interval's Holm step (``gain_confidence``): one candidate at a step past the nominal level prints the
  nominal one, candidates at different steps draw together, and a reading stating no gain interval draws; a reading
  stating no within_interval (an aspect with no margin) states no level: its level key is unstated and its bars
  print none;
- **a confusion block names every panel's own columns and its axis**: with two axes the domain panel never sits
  under the complexity labels, and with two arms each panel names its arm and its axis;
- **a check is named as the code it is**: a measure whose only name is its check's expression heads its column as
  that expression set as code, never as a bare dotted key;
- **one money spelling within a chart**: a bar's value label and the axis ticks read by one format, thousands
  grouped, an amount of ten dollars or more whole and a smaller one as the package spells money;
- **a row label too long for a narrow chart has a shorter reader name**: a price step carries its options' tags.

Every expectation is computed from the input the block reads (r1 to r5, the misfits and the shape campaigns).
"""

from __future__ import annotations

import functools
import re
from typing import Any

import pytest

from packages.evals.tests.analyst_support import fixture_inputs, shape_snapshots
from threetears.evals.analysis.analyst.blocks import block_render
from threetears.evals.analysis.analyst.snapshot import frozen_input
from threetears.evals.analysis.errors import UnresolvableReference, UnstatedFigure
from threetears.evals.kernel.numbers import format_number, group_thousands, money_form
from threetears.evals.analysis.report.block_spec import block_look, complete_chart
from threetears.evals.analysis.report.charts import BarsChart, ForestChart
from threetears.evals.analysis.report.figure_keys import format_figure, price_frontier, resolve
from threetears.evals.analysis.report.plan_page.resolve import resolve_chart
from threetears.evals.analysis.report_input import Factor, Reading, ReportInput
from threetears.evals.analysis.stats import INTERVAL_LEVEL
from threetears.evals.analysis.report_plan_facts import fact_arms
from threetears.evals.kernel.metrics import goal_check_of
from threetears.evals.analysis.style import humanize
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.kernel.report_plan import ChartQuery, PlannedChart

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
    """r1 to r5, the misfits, and the shape campaigns as their snapshots froze them."""
    held = dict(fixture_inputs())
    held |= {name: frozen_input(snapshot) for name, snapshot in shape_snapshots("s").items()}
    return held


def _stated(level: float) -> str:
    """A level, a share, as the input states it: in percent, to every digit the number rule keeps."""
    return f"{format_number(level * 100)}%"


def _drawn(held: ReportInput, kind: str, query: ChartQuery) -> Any:
    """The chart a page draws of a block, as the kit's data."""
    planned = PlannedChart.model_validate({"kind": kind, "title": "", "query": query, "reference": [], "emphasis": ""})
    spec, faults = complete_chart(planned, held, "block")
    assert not faults, faults
    return resolve_chart(spec, held, block_look(held, fact_arms(held)), title="").chart


# =============================================================================
# Levels as stated
# =============================================================================


#: The level a reading's within_interval is read at, as a chart of readings and a reading's level key print it.
_NOMINAL = f"{format_number(INTERVAL_LEVEL * 100)}%"


def _stepped_readings() -> list[tuple[str, str, Reading]]:
    """Every reading whose gain interval sits at a Holm step past the nominal level, with a within_interval: the
    input, the aspect and the reading."""
    return [
        (name, aspect.id, reading)
        for name, held in _inputs().items()
        for aspect in held.aspects
        for reading in aspect.readings
        if reading.within_interval is not None
        and reading.gain_confidence is not None
        and float(reading.gain_confidence) != INTERVAL_LEVEL
    ]


def _forest(held: ReportInput, aspect: str, arms: list[str]) -> tuple[Any, ForestChart]:
    """A forest of the arms' readings on the aspect: the analyst's block, and the chart a page draws."""
    query = _query("readings", aspect=aspect, arms=arms)
    drawn = _drawn(held, "forest", query).data
    assert isinstance(drawn, ForestChart)
    return block_render(held, block_kind="forest", query=query), drawn


def _bars(held: ReportInput, aspect: str, arms: list[str]) -> BarsChart:
    """Diverging bars of the arms' readings on the aspect, as a page draws them."""
    drawn = _drawn(held, "diverging_bars", _query("readings", aspect=aspect, arms=arms)).data
    assert isinstance(drawn, BarsChart)
    return drawn


def _with_reading(held: ReportInput, aspect_id: str, reading: Reading) -> ReportInput:
    """The input with one aspect's reading of ``reading.arm`` replaced by ``reading``."""
    aspects = [
        aspect.model_copy(update={"readings": [reading if r.arm == reading.arm else r for r in aspect.readings]})
        if aspect.id == aspect_id
        else aspect
        for aspect in held.aspects
    ]
    return held.model_copy(update={"aspects": aspects})


def _drew_within(chart: ForestChart, held: ReportInput, aspect: str) -> None:
    """Each forest row is the reading's change over its within_interval, the interval the level labels."""
    readings = {r.arm: r for a in held.aspects if a.id == aspect for r in a.readings}
    assert chart.rows
    for row in chart.rows:
        assert row.id is not None
        interval = readings[row.id].within_interval
        assert interval is not None
        assert (row.lo, row.hi) == (float(interval[0]), float(interval[1])), row.id


def test_a_readings_level_key_prints_the_nominal_level_of_its_interval_never_its_gain_level() -> None:
    found = _stepped_readings()
    assert found, "no input holds a reading whose gain interval sits past the nominal level"
    for name, aspect, reading in found:
        figure = resolve(f"aspects.{aspect}.reading.{reading.arm}.level", _inputs()[name])
        assert format_figure(figure) == _NOMINAL, (name, aspect, reading.arm)
        assert _NOMINAL != _stated(float(reading.gain_confidence or 0)), (name, aspect, reading.arm)


def test_two_pair_levels_that_differ_never_print_alike() -> None:
    printed: dict[str, set[float]] = {}
    for held in _inputs().values():
        for aspect in held.aspects:
            for pair in aspect.pairs:
                if pair.confidence is None:
                    continue
                first, second = pair.arms
                text = format_figure(resolve(f"aspects.{aspect.id}.pair.{first}.{second}.confidence", held))
                printed.setdefault(text, set()).add(round(float(pair.confidence), 12))
    assert len(printed) > 1, "every input's pairs sit at one level: the test reads no difference"
    assert all(len(levels) == 1 for levels in printed.values()), printed


def test_a_pairs_level_key_prints_the_level_as_the_input_states_it() -> None:
    found = [
        (held, aspect.id, pair)
        for held in _inputs().values()
        for aspect in held.aspects
        for pair in aspect.pairs
        if pair.confidence is not None
    ]
    assert found, "no input states a pair reading's level"
    for held, aspect, pair in found:
        first, second = pair.arms
        figure = resolve(f"aspects.{aspect}.pair.{first}.{second}.confidence", held)
        assert format_figure(figure) == _stated(float(pair.confidence))


def test_a_one_candidate_forest_prints_the_nominal_level_of_the_interval_it_draws() -> None:
    found = _stepped_readings()
    assert found, "no input holds a reading whose gain interval sits past the nominal level"
    for name, aspect, reading in found:
        held = _inputs()[name]
        block, chart = _forest(held, aspect, [reading.arm])
        assert block.level == chart.level == _NOMINAL, (name, aspect, reading.arm)
        assert block.level == format_figure(resolve(f"aspects.{aspect}.reading.{reading.arm}.level", held))
        _drew_within(chart, held, aspect)


def test_a_forest_over_readings_at_different_holm_steps_draws_at_the_one_level_it_draws() -> None:
    mixed = []
    for name, held in _inputs().items():
        for aspect in held.aspects:
            levels = {r.gain_confidence for r in aspect.readings if r.gain_confidence is not None}
            if len(levels) > 1 and all(r.within_interval is not None for r in aspect.readings):
                mixed.append((name, aspect))
    assert mixed, "no aspect's readings sit at different Holm steps"
    for name, aspect in mixed:
        held = _inputs()[name]
        block, chart = _forest(held, aspect.id, [r.arm for r in aspect.readings])
        assert block.level == chart.level == _NOMINAL, (name, aspect.id)
        assert {row.id for row in chart.rows} == {r.arm for r in aspect.readings}
        _drew_within(chart, held, aspect.id)
        bars = _bars(held, aspect.id, [r.arm for r in aspect.readings])
        assert bars.level == _NOMINAL, (name, aspect.id)


def test_a_reading_with_a_within_interval_and_no_gain_fields_draws_at_the_nominal_level() -> None:
    name, aspect_id, reading = _stepped_readings()[0]
    held = _inputs()[name]
    stripped = _with_reading(
        held, aspect_id, reading.model_copy(update={"gain_interval": None, "gain_confidence": None})
    )
    block, chart = _forest(stripped, aspect_id, [reading.arm])
    assert block.level == chart.level == _NOMINAL
    _drew_within(chart, stripped, aspect_id)
    assert format_figure(resolve(f"aspects.{aspect_id}.reading.{reading.arm}.level", stripped)) == _NOMINAL


def test_a_reading_with_gain_fields_and_no_within_interval_states_no_level() -> None:
    """An aspect with no margin: the reading states its gain interval and level, and no within_interval. No chart of
    readings draws the gain interval, so nothing labels one: the level key is unstated, and its bars print no level."""
    name, aspect_id, reading = _stepped_readings()[0]
    assert reading.gain_interval is not None and reading.gain_confidence is not None
    held = _with_reading(_inputs()[name], aspect_id, reading.model_copy(update={"within_interval": None}))
    with pytest.raises(UnstatedFigure):
        resolve(f"aspects.{aspect_id}.reading.{reading.arm}.level", held)
    block = block_render(
        held, block_kind="diverging_bars", query=_query("readings", aspect=aspect_id, arms=[reading.arm])
    )
    assert block.level == ""
    bars = _bars(held, aspect_id, [reading.arm])
    assert bars.level is None
    assert all(row.lo is None and row.hi is None for row in bars.rows or [])


def test_a_chart_refused_for_mixing_levels_names_each_level_as_its_key_prints_it() -> None:
    """Two contrasts on one measure, each at its own level: one chart carries one interval level (GENERATOR §5.2), so
    the forest is refused, naming each level as its contrast's level key prints it."""
    found = []
    for name, held in _inputs().items():
        by_measure: dict[tuple[str | None, str], list[Any]] = {}
        for contrast in held.contrasts:
            if contrast.interval is not None:
                by_measure.setdefault((contrast.aspect, contrast.measure), []).append(contrast)
        found += [(name, held, pair[:2]) for pair in by_measure.values() if len({c.arm for c in pair[:2]}) == 2]
    assert found, "no input holds two arms' contrasts with intervals on one measure"
    name, held, (first, second) = found[0]
    levels = {first.arm: "99%", second.arm: "97.5%"}
    mixed = held.model_copy(
        update={
            "contrasts": [
                c.model_copy(update={"level": levels[c.arm]}) if c in (first, second) else c for c in held.contrasts
            ]
        }
    )
    query = _query("contrasts", measures=[first.measure], arms=[first.arm, second.arm], aspect=first.aspect or "")
    planned = PlannedChart.model_validate(
        {"kind": "forest", "title": "", "query": query, "reference": [], "emphasis": ""}
    )
    _, faults = complete_chart(planned, mixed, "block")
    refusals = [why for where, why in faults if where == "block.query" and "one interval level" in why]
    assert refusals, (name, faults)
    for contrast in (first, second):
        printed = format_figure(resolve(f"contrast.{contrast.arm}.{contrast.measure}.level", mixed))
        assert printed == levels[contrast.arm]
        assert printed in refusals[0], (printed, refusals[0])
    with pytest.raises(RefusedInput):
        block_render(mixed, block_kind="forest", query=query)


# =============================================================================
# A confusion block's panels
# =============================================================================


def _confusion_inputs() -> list[tuple[ReportInput, list[str]]]:
    """Each input holding confusion tables on two axes or more, and the arms holding every axis's."""
    found = []
    for held in _inputs().values():
        if len(held.confusion) < 2:
            continue
        arms = [arm.id for arm in held.arms if all(arm.id in table.per_arm for table in held.confusion.values())]
        found.append((held, arms))
    assert found, "no input holds two confusion axes"
    return found


def _axes(held: ReportInput) -> list[Factor]:
    """Every factor and target-matrix axis the input declares, which name a confusion axis and its labels."""
    matrix = held.target_matrix
    return [*held.factors, *((matrix.rows, matrix.columns) if matrix is not None else ())]


def _axis_name(held: ReportInput, axis: str) -> str:
    factor = next((factor for factor in _axes(held) if factor.id == axis), None)
    return (factor.name if factor is not None else None) or humanize("dimension", axis)


def _label_names(held: ReportInput, axis: str) -> list[str]:
    factor = next((factor for factor in _axes(held) if factor.id == axis), None)
    names = {level.id: level.name for level in factor.levels if level.name} if factor is not None else {}
    return [names.get(label, label) for label in held.confusion[axis].labels]


@pytest.mark.parametrize("arms_drawn", [1, 2])
def test_each_confusion_panel_names_its_axis_and_arm_and_reads_each_cell_by_its_own_labels(arms_drawn: int) -> None:
    for held, arms in _confusion_inputs():
        shown = arms[:arms_drawn]
        assert len(shown) == arms_drawn
        look = block_look(held, fact_arms(held))
        block = block_render(held, block_kind="confusion", query=_query("confusion", arms=shown), limit=400)
        assert block.columns[:3] == ["Panel", "True", "Predicted"], block.columns
        for arm in shown:
            for axis, table in held.confusion.items():
                names = _label_names(held, axis)
                rows = [
                    row.cells
                    for row in block.rows
                    if _axis_name(held, axis) in row.cells[0]
                    and (arms_drawn == 1 or look.arm(arm).name in row.cells[0])
                ]
                assert len(rows) == len(names) ** 2, (arm, axis, [row.cells[0] for row in block.rows])
                matrix = table.per_arm[arm]
                for cells in rows:
                    expected, predicted = names.index(cells[1]), names.index(cells[2])
                    share = float(matrix[expected][predicted])
                    if table.unit != "row_percent":
                        share = share / sum(float(value) for value in matrix[expected]) * 100
                    assert float(cells[3].rstrip("%")) == pytest.approx(share, abs=0.051)


# =============================================================================
# A check named as the code it is
# =============================================================================


def _bare_checks() -> list[tuple[ReportInput, str, str]]:
    """Each measure whose label is its check's expression, bare: the input, the key and the expression."""
    found = []
    for held in _inputs().values():
        for key, measure in held.measures.items():
            check = goal_check_of(key)
            if check is not None and measure.label == check:
                found.append((held, key, check))
    assert found, "no input names a check by its bare expression"
    return found


def test_a_check_named_by_its_bare_expression_heads_its_column_as_code() -> None:
    for held, key, check in _bare_checks():
        block = block_render(held, block_kind="table", query=_query("measures", measures=[key]))
        assert f"`{check}`" in block.columns, block.columns
        assert check not in [column.strip("`") for column in block.columns if not column.startswith("`")]


# =============================================================================
# Money within one chart
# =============================================================================

#: A money amount as a chart's label or tick spells one: a sign, a dollar sign, digits with any thousands separators,
#: and any cents.
_MONEY = re.compile(r"^[+\-]?\$(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d+))?$")


def _price_charts() -> list[tuple[ReportInput, BarsChart]]:
    found = []
    for held in _inputs().values():
        for key in dict.fromkeys([*held.judged, *held.measures]):
            try:
                steps = price_frontier(held, key).steps
            except UnresolvableReference:
                continue
            if steps:
                chart = _drawn(held, "bars", _query("prices", measures=[key]))
                assert isinstance(chart.data, BarsChart)
                found.append((held, chart.data))
    assert found, "no input prices a step"
    return found


def test_a_money_charts_labels_and_ticks_share_one_spelling_grouped() -> None:
    """A price chart's notes and its ticks in the one form of every amount it draws (#993): whole dollars where the
    least is $10 or more, else cents and up; the kit draws its ticks to the same decimals, thousands grouped."""
    for _held, data in _price_charts():
        assert data.rows is not None
        drawn = [row.value for row in data.rows if row.value is not None]
        form = money_form(drawn)
        assert data.fmt == f"usd:{form.decimals}" and data.tick_fmt in (None, data.fmt), (data.fmt, data.tick_fmt)
        for row in data.rows:
            assert row.note is not None and row.value is not None
            spelled = _MONEY.match(row.note)
            assert spelled is not None, row.note
            digits = row.note.lstrip("+-$").split(" ")[0]
            assert digits == group_thousands(digits.replace(",", "")), row.note
            assert row.note.split(" (")[0] == form.spell(row.value), row.note


# =============================================================================
# A shorter reader name where a label runs long
# =============================================================================


def test_a_price_step_carries_its_options_tags_as_its_short_label() -> None:
    for held, data in _price_charts():
        look = block_look(held, fact_arms(held))
        assert data.rows is not None
        for row in data.rows:
            assert row.id is not None
            first, _arrow, second = row.id.partition("→")
            assert row.short_label is not None
            assert look.arm(first).tag in row.short_label and look.arm(second).tag in row.short_label
            assert len(row.short_label) < len(row.label)
