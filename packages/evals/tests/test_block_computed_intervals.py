"""Intervals the tools compute, drawn: a pass rate with a count and no stated interval is drawn with its binomial one.

Contracts held here (``analysis.separation.spread_interval``, the ``spread.<arm>.<key>.overall.interval`` figure key,
``analysis.report.block_spec`` completion, ``analysis.report.plan_page.charting.computed_interval``):

- **the key**: an arm's spread reads an interval at its level, the mean plus and minus the normal quantile times its
  standard error, a rate's ends held to none to all; a rate of none or all reads none (a zero-width interval would
  read as certain);
- **the block**: a dot strip, bars or a table of arms' figures none of which states an interval, where some is a
  pass/fail rate with a count, is completed at the computed intervals' level, marked computed, draws each one as its
  key reads it, and says so in a note; a chart whose figures state their own interval draws theirs, as before;
- **a stored chart draws as it was stored**: one completed before (its level empty) draws no computed interval;
- **repeats are said**: where a case ran more than once, the note says the interval is read on the distinct cases, at
  the widest spread the repeats allow.

Every expectation is computed from the input the block reads (r1 to r5, the misfits and the shape campaigns).
"""

from __future__ import annotations

import functools
from statistics import NormalDist
from typing import Any

import pytest

from packages.evals.tests.analyst_support import fixture_inputs, shape_snapshots
from threetears.evals.analysis.analyst.blocks import block_render
from threetears.evals.analysis.analyst.snapshot import frozen_input
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.kernel.numbers import format_number
from threetears.evals.analysis.report.block_spec import block_look, complete_chart
from threetears.evals.analysis.report.charts import DotIntervalChart
from threetears.evals.analysis.report.figure_keys import resolve
from threetears.evals.analysis.report.plan_page.charting import computed_interval
from threetears.evals.analysis.report.plan_page.resolve import resolve_chart
from threetears.evals.analysis.report.plan_page.words import COMPUTED_NOTE, COMPUTED_REPEATS_NOTE
from threetears.evals.analysis.report_input import ReportInput, Stat
from threetears.evals.analysis.report_plan_facts import fact_arms
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

#: How a binomial spread's basis begins, as its ``spread.`` key reads it.
BINOMIAL = "a pass/fail rate's binomial spread"

#: The kinds that draw an arm's figure with an interval code may compute.
_KINDS = ("dot_interval", "bars", "table")


def _query(source: str, **fields: Any) -> ChartQuery:
    return ChartQuery.model_validate({**_EMPTY, "source": source, **fields})


@functools.cache
def _inputs() -> dict[str, ReportInput]:
    held = dict(fixture_inputs())
    held |= {name: frozen_input(snapshot) for name, snapshot in shape_snapshots("s").items()}
    return held


def _bare(stat: Stat) -> bool:
    """Whether a figure states a mean and no spread of its own."""
    return stat.mean is not None and stat.ci is None and stat.half_width is None and stat.sem is None


def _read(key: str, held: ReportInput) -> Any:
    try:
        return resolve(key, held)
    except UnresolvableReference:
        return None


def _computed() -> list[tuple[str, str, list[str]]]:
    """Each measure or judged dimension no arm's overall figure states a spread of, with the arms whose spread reads a
    binomial interval, the one a chart computes: the input, the key and those arms. An overall that is the plain mean
    of its cells reads the spread derived from theirs instead (``cell_mean``), which a chart does not draw as computed."""
    found = []
    for name, held in _inputs().items():
        for key, table in [*held.measures.items(), *held.judged.items()]:
            stats = table.per_arm
            if not stats or not all(_bare(stat) for stat in stats.values()):
                continue
            arms = [
                arm
                for arm in stats
                if _read(f"spread.{arm}.{key}.overall.interval", held) is not None
                and str(resolve(f"spread.{arm}.{key}.overall.basis", held).value).startswith(BINOMIAL)
            ]
            if arms:
                found.append((name, key, arms))
    return found


def test_the_fixtures_hold_rates_with_a_count_and_no_stated_interval() -> None:
    names = {name for name, _key, _arms in _computed()}
    assert "r2 with one gate" in names, names


def test_an_overall_its_cells_give_an_interval_draws_none_computed() -> None:
    """r3's balanced accuracy is the plain mean of its cells, and its spread is read off theirs (``cell_mean``): a
    chart of it computes no interval, since its note would call it binomial; its interval is cited by its key."""
    held = _inputs()["r3"]
    for arm, stat in held.measures["balanced_accuracy"].per_arm.items():
        basis = resolve(f"spread.{arm}.balanced_accuracy.overall.basis", held).value
        assert isinstance(basis, str) and not basis.startswith(BINOMIAL), "the control: its spread reads its cells"
        assert computed_interval(held, "balanced_accuracy", arm, stat) is None


def test_no_interval_is_computed_on_a_figure_that_is_no_share_of_counted_trials() -> None:
    """r3 grades its balanced accuracy by class label, but it is the mean of each class's recall: no binomial spread
    describes it, so no interval is computed for it, however its arms' trials are counted. Only a share of counted
    trials is (a pass/fail measure, a plain accuracy or match)."""
    held = _inputs()["r3 without its configs"]
    shares = {"accuracy", "match"}
    derived = [key for key, measure in held.measures.items() if measure.grade == "class_label" and key not in shares]
    assert derived, "the control: r3 grades a derived mean by class label"
    for name, key, _arms in _computed():
        measure = _inputs()[name].measures.get(key)
        assert measure is None or measure.grade == "pass_fail" or key in shares, (name, key)


def test_an_arms_spread_reads_its_interval_as_its_mean_plus_and_minus_the_quantile_times_its_error() -> None:
    read = 0
    for held in _inputs().values():
        for key, table in [*held.measures.items(), *held.judged.items()]:
            for arm, stat in table.per_arm.items():
                prefix = f"spread.{arm}.{key}.overall"
                interval = _read(f"{prefix}.interval", held)
                if interval is None:
                    continue
                mean = resolve(f"{prefix}.mean", held).value
                level = resolve(f"{prefix}.level", held).value
                se = resolve(f"{prefix}.se", held)
                assert isinstance(mean, float) and isinstance(level, float) and isinstance(se.value, float)
                # se reads in points for a rate on the share scale; the interval in the mean's own units.
                error = se.value / 100 if se.scale == "points" and interval.scale == "share" else se.value
                half = NormalDist().inv_cdf(0.5 + level / 2) * error
                low, high = mean - half, mean + half
                if interval.scale in ("share", "percent"):
                    top = 1.0 if interval.scale == "share" else 100.0
                    low, high = max(low, 0.0), min(high, top)
                assert interval.value == pytest.approx((low, high)), prefix
                assert resolve(f"{prefix}.low", held).value == pytest.approx(low)
                assert resolve(f"{prefix}.high", held).value == pytest.approx(high)
                assert stat.mean is not None
                read += 1
    assert read


@pytest.mark.parametrize("end", ["none", "all"])
def test_a_rate_of_none_or_all_with_no_stated_spread_reads_no_interval_and_draws_none(end: str) -> None:
    name, key, arms = _computed()[0]
    held = _inputs()[name]
    table = held.measures[key] if key in held.measures else held.judged[key]
    arm = arms[0]
    scale = resolve(f"spread.{arm}.{key}.overall.mean", held).scale
    top = 100.0 if scale == "percent" else 1.0
    data = held.model_dump(mode="json", by_alias=True)
    where = "measures" if key in held.measures else "judged"
    data[where][key]["per_arm"][arm]["mean"] = 0.0 if end == "none" else top
    changed = ReportInput.model_validate(data)
    assert _bare(changed.measures[key].per_arm[arm] if key in changed.measures else changed.judged[key].per_arm[arm])
    assert table.per_arm[arm].mean not in (0.0, top)
    assert _read(f"spread.{arm}.{key}.overall.interval", changed) is None
    block = block_render(changed, block_kind="table", query=_query(where, measures=[key], arms=arms), limit=400)
    interval_at = next(index for index, column in enumerate(block.columns) if column.endswith("interval"))
    shown = next(
        row.cells for row in block.rows if block_look(changed, fact_arms(changed)).arm(arm).name in row.cells[0]
    )
    assert shown[interval_at] in ("", "—"), shown


@pytest.mark.parametrize("kind", _KINDS)
def test_a_block_of_rates_stating_no_interval_draws_the_one_each_count_gives_at_its_level(kind: str) -> None:
    for name, key, arms in _computed():
        held = _inputs()[name]
        block = block_render(
            held,
            block_kind=kind,
            query=_query("measures" if key in held.measures else "judged", measures=[key], arms=arms),
            limit=400,
        )
        levels = {resolve(f"spread.{arm}.{key}.overall.level", held).value for arm in arms}
        assert len(levels) == 1
        (level,) = levels
        assert isinstance(level, float)
        assert block.level == f"{format_number(level * 100)}% computed", (name, key)
        assert COMPUTED_NOTE in block.notes
        header = next(column for column in block.columns if column.endswith("interval"))
        assert header.startswith(block.level)
        where = block.columns.index(header)
        printed = [row.cells[where] for row in block.rows if row.cells[where] not in ("", "—")]
        expected = [resolve(f"spread.{arm}.{key}.overall.interval", held) for arm in arms]
        assert len(printed) == len(expected), (name, key, printed)
        for cell, figure in zip(printed, expected, strict=True):
            assert isinstance(figure.value, tuple)
            factor = 100.0 if figure.scale == "share" else 1.0
            ends = [float(end.rstrip("%").replace(",", "")) for end in cell.split(" to ")]
            assert ends == pytest.approx([end * factor for end in figure.value], abs=0.051), (cell, figure.value)


def test_a_block_of_figures_stating_their_own_interval_draws_theirs_and_no_computed_one() -> None:
    drawn = 0
    for held in _inputs().values():
        for key, table in held.measures.items():
            stats = table.per_arm
            if table.level and stats and all(stat.ci is not None for stat in stats.values()):
                block = block_render(held, block_kind="dot_interval", query=_query("measures", measures=[key]))
                assert block.level == table.level
                assert COMPUTED_NOTE not in block.notes
                drawn += 1
    assert drawn


def test_a_chart_stored_before_its_intervals_were_computed_draws_as_it_was_stored() -> None:
    for name, key, arms in _computed():
        held = _inputs()[name]
        query = _query("measures" if key in held.measures else "judged", measures=[key], arms=arms)
        planned = PlannedChart.model_validate(
            {"kind": "dot_interval", "title": "", "query": query, "reference": [], "emphasis": ""}
        )
        spec, faults = complete_chart(planned, held, "chart")
        assert not faults
        stored = spec.model_copy(update={"level": ""})
        chart = resolve_chart(stored, held, block_look(held, fact_arms(held)), title="").chart
        assert isinstance(chart.data, DotIntervalChart)
        rows = [row for panel in (chart.data.panels or [chart.data]) for row in (panel.rows or [])]
        assert rows and all(row.lo is None and row.hi is None for row in rows)


def test_a_computed_interval_on_repeated_cases_says_it_is_read_on_the_distinct_cases() -> None:
    name, key, arms = next((name, key, arms) for name, key, arms in _computed() if key in _inputs()[name].measures)
    held = _inputs()[name]
    plain = block_render(held, block_kind="dot_interval", query=_query("measures", measures=[key], arms=arms))
    assert COMPUTED_NOTE in plain.notes and COMPUTED_REPEATS_NOTE not in plain.notes
    data = held.model_dump(mode="json", by_alias=True, exclude_none=True)
    for arm in arms:
        stat = data["measures"][key]["per_arm"][arm]
        stat["n"], stat["n_cases"] = 1200, 400
    repeated = ReportInput.model_validate(data)
    block = block_render(repeated, block_kind="dot_interval", query=_query("measures", measures=[key], arms=arms))
    assert block.notes[-2:] == [COMPUTED_NOTE, COMPUTED_REPEATS_NOTE]
