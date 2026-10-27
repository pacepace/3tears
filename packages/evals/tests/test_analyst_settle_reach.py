"""How far settle_cases and arms_compare reach: a slice's stated cases, a repeated design, a margin or floor, and an
assumed count.

Contracts held here (``threetears.evals.analysis.separation``, the ``spread.``, ``settle.`` and ``assumed.`` families
of ``analysis.report.figure_keys``, ``analysis.analyst.settle`` and ``analysis.analyst.compare``):

- **a rate with no count of its own** reads the slice's stated cases, and its basis says so;
- **on a repeated design a size counts cases**, each run as often as today, and ``trials_needed`` is the trials that
  takes (s3's bars here; r2 with three trials a case in ``test_analyst_separation``); a design that repeats no case
  states no trials;
- **a candidate's reading against its margin, or a floor read on it**: the change from the control against minus the
  margin, on the reading's own interval at its level with each arm's cases (a floor on another rate: the two arms'
  rates on that measure, as independent arms); with one arm and nothing named, every criterion, margin and floor
  reading it; ``against`` names one, and a wrong one is refused naming those that read;
- **an assumed count** is a what-if keyed ``assumed.<cases>.`` (or ``assumed.share.``): a slice whose figure states no
  count is read as holding it, a place whose figures state their count is read as they state it, and an assumption
  that fills nothing is refused.

Every expectation is computed from the input's own figures.
"""

from __future__ import annotations

import math
from functools import cache
from statistics import NormalDist
from typing import Any

import pytest

from packages.evals.tests.analyst_support import budget_inputs, fixture_inputs
from packages.evals.tests.fixtures.report_plans.misfits import gates_in_each_task, reference_data
from threetears.evals.analysis.analyst.compare import arms_compare
from threetears.evals.analysis.analyst.settle import SettleRow, settle_cases
from threetears.evals.analysis.errors import UnresolvableReference, UnstatedFigure
from threetears.evals.analysis.report.figure_keys import resolve
from threetears.evals.analysis.report_decision import dominant_options
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.separation import level_share
from threetears.evals.kernel.errors import RefusedInput

_NORMAL = NormalDist()

#: Every page a test reads whole asks for this many rows.
WHOLE = 200


def _number(value: object) -> float:
    assert isinstance(value, int | float), value
    return float(value)


def _least(per_case: float, distance: float, z: float) -> int:
    return math.ceil((z * per_case / abs(distance)) ** 2)


# --- a slice's stated cases -----------------------------------------------------------------------------------------


@cache
def _tasks_counted() -> ReportInput:
    """r2 with each task stating its cases, read off the scoreboard's cases and the task's traffic share."""
    data = reference_data("r2")
    cases = next(iter(data["scoreboard"].values()))["cases"]
    for task in data["tasks"]:
        task["n_cases"] = round(cases * task["share"])
    return ReportInput.model_validate(data)


def test_a_rate_with_no_count_reads_its_tasks_stated_cases_and_its_basis_says_so() -> None:
    held = _tasks_counted()
    task = held.tasks[0]
    arm = next(iter(task.per_arm))
    stat = task.per_arm[arm]["pass_rate"]
    assert stat.mean is not None and stat.n is None and stat.n_cases is None and task.n_cases is not None
    prefix = f"spread.{arm}.pass_rate.task.{task.id}"
    share = float(stat.mean) / 100
    assert _number(resolve(f"{prefix}.se", held).value) == pytest.approx(
        math.sqrt(share * (1 - share) / task.n_cases) * 100
    )
    assert resolve(f"{prefix}.n", held).value == task.n_cases
    basis = resolve(f"{prefix}.basis", held).value
    assert isinstance(basis, str) and "binomial" in basis and "the task's stated cases" in basis


# --- a repeated design ----------------------------------------------------------------------------------------------


def test_s3s_bar_rows_size_cases_and_the_trials_they_take() -> None:
    held = budget_inputs()["s3"]
    arm = held.arms[0].id
    checked = 0
    for row in settle_cases(held, [arm], limit=WHOLE).rows:
        if row.cases_needed is None or not row.against.startswith("criterion "):
            continue
        criterion = next(c for c in held.criteria if f"criterion {c.id}" == row.against)
        stat = (held.judged.get(criterion.measure) or held.measures[criterion.measure]).per_arm[arm]
        assert stat.n is not None and stat.n_cases is not None and stat.n_cases < stat.n, "s3 repeats its cases"
        assert row.trials_needed is not None
        assert row.trials_needed.value == math.ceil(_number(row.cases_needed.value) * stat.n / stat.n_cases - 1e-9)
        checked += 1
    assert checked


def test_a_design_that_repeats_no_case_states_no_trials() -> None:
    held = fixture_inputs()["r2"]
    (pick,) = dominant_options(held)
    gate = next(criterion for criterion in held.criteria if criterion.measure == "pass_rate")
    (row,) = settle_cases(held, [pick], criterion=gate.id).rows
    assert row.cases_needed is not None and row.trials_needed is None and not row.repeats
    with pytest.raises(UnstatedFigure, match="no count of trials"):
        resolve(f"settle.criteria.{gate.id}.{pick}.overall.trials_needed", held)


# --- margins and floors ---------------------------------------------------------------------------------------------


def _margin_inputs() -> dict[str, ReportInput]:
    held = {name: budget_inputs()[name] for name in ("s1", "s2", "s3")}
    return held | dict(fixture_inputs())


#: Every input a margin or floor is settled on.
MARGIN_INPUTS = _margin_inputs()


def _harmonic(counts: list[int]) -> float:
    return len(counts) / sum(1 / count for count in counts)


@pytest.mark.parametrize("name", sorted(MARGIN_INPUTS))
def test_a_readings_margin_is_settled_on_the_readings_own_interval_and_cases(name: str) -> None:
    held = MARGIN_INPUTS[name]
    for aspect in held.aspects:
        for reading in aspect.readings:
            margin = reading.margin if reading.margin is not None else aspect.margin
            if margin is None:
                continue
            prefix = f"settle.margin.{aspect.id}.{reading.arm}"
            assert resolve(f"{prefix}.line", held).value == -float(margin)
            level = level_share(reading.gain_confidence)
            if reading.gain_interval is None or level is None:
                with pytest.raises(UnstatedFigure):
                    resolve(f"{prefix}.cases_needed", held)
                continue
            low, high = reading.gain_interval
            se = (float(high) - float(low)) / 2 / _NORMAL.inv_cdf(0.5 + level / 2)
            distance = float(reading.change) + float(margin)
            assert resolve(f"{prefix}.distance", held).value == pytest.approx(distance)
            assert resolve(f"{prefix}.change", held).value == pytest.approx(float(reading.change))
            if distance == 0 or not reading.cases:
                continue
            per_case = se * math.sqrt(_harmonic(reading.cases))
            assert resolve(f"{prefix}.cases_needed", held).value == _least(per_case, distance, _NORMAL.inv_cdf(level))
            assert resolve(f"{prefix}.wrong_side", held).value == pytest.approx(_NORMAL.cdf(-abs(distance) / se))


def test_some_fixture_settles_a_margin() -> None:
    assert any(
        reading.gain_interval is not None and (reading.margin is not None or aspect.margin is not None)
        for held in MARGIN_INPUTS.values()
        for aspect in held.aspects
        for reading in aspect.readings
    )


@pytest.mark.parametrize("name", sorted(MARGIN_INPUTS))
def test_a_floor_on_the_pass_rate_reads_as_the_readings_margin_at_the_floors_own(name: str) -> None:
    held = MARGIN_INPUTS[name]
    for read in held.floors:
        if read.reading is not None or read.margin is None:
            continue
        aspect = next((aspect for aspect in held.aspects if aspect.id == read.aspect), None)
        reading = next((r for r in aspect.readings if r.arm == read.arm), None) if aspect is not None else None
        if reading is None or reading.gain_interval is None:
            continue
        prefix = f"settle.floor.{read.aspect}.{read.floor}.{read.arm}"
        assert resolve(f"{prefix}.line", held).value == -float(read.margin)
        assert resolve(f"{prefix}.distance", held).value == pytest.approx(float(reading.change) + float(read.margin))


def test_a_floor_on_a_check_reads_the_two_arms_rates_on_its_measure() -> None:
    held = budget_inputs()["s1"]
    control = next(arm.id for arm in held.arms if arm.role == "control")
    checked = 0
    for read in held.floors:
        if read.reading is None or read.margin is None:
            continue
        key = next(key for key in held.measures if key.endswith(read.reading))
        table = held.measures[key]
        figures = table.by_aspect.get(read.aspect, table.per_arm)
        mine, theirs = figures[read.arm].mean, figures[control].mean
        assert mine is not None and theirs is not None
        factor = 100.0 if resolve(f"measure.{key}.{control}.mean", held).scale == "share" else 1.0
        change = factor * (float(mine) - float(theirs))
        prefix = f"settle.floor.{read.aspect}.{read.floor}.{read.arm}"
        assert resolve(f"{prefix}.change", held).value == pytest.approx(change)
        assert resolve(f"{prefix}.distance", held).value == pytest.approx(change + float(read.margin))
        checked += 1
    assert checked, "the control: s1 holds a floor on a check"


def test_one_arm_with_nothing_named_reads_every_margin_and_floor_and_against_names_one() -> None:
    held = budget_inputs()["s1"]
    arm = next(arm.id for arm in held.arms if any(read.arm == arm.id for read in held.floors))
    margins = [
        f"margin:{aspect.id}"
        for aspect in held.aspects
        if any(r.arm == arm and (r.margin is not None or aspect.margin is not None) for r in aspect.readings)
    ]
    floors = [
        f"floor:{read.aspect}/{read.floor}" for read in held.floors if read.arm == arm and read.margin is not None
    ]
    rows = settle_cases(held, [arm], limit=WHOLE).rows
    criteria = [f"criterion {criterion.id}" for criterion in held.criteria]
    assert [row.against for row in rows if row.against not in criteria] == margins + floors
    for against in margins + floors:
        (row,) = settle_cases(held, [arm], against=against).rows
        assert row.against == against and row.line is not None
        assert row.line.key.startswith("settle.margin." if against.startswith("margin:") else "settle.floor.")


@pytest.mark.parametrize(
    ("named", "says"),
    [
        ({"against": "margin:no-such"}, "names no margin or floor"),
        ({"against": "margin:billing", "criterion": "x"}, "not both"),
        ({"against": "margin:billing", "line": 0.5, "measure": "cost_usd"}, "not both"),
    ],
)
def test_against_naming_nothing_or_beside_another_line_is_refused_with_a_remedy(
    named: dict[str, Any], says: str
) -> None:
    held = budget_inputs()["s1"]
    arm = next(reading.arm for aspect in held.aspects for reading in aspect.readings)
    with pytest.raises(RefusedInput, match=says) as refused:
        settle_cases(held, [arm], **named)
    assert refused.value.remedy


def test_two_arms_settled_apart_refuse_a_margin() -> None:
    held = budget_inputs()["s1"]
    first, second = (arm.id for arm in held.arms[:2])
    with pytest.raises(RefusedInput, match="not against a line"):
        settle_cases(held, [first, second], measure=next(iter(held.measures)), against="margin:billing")


# --- an assumed count -----------------------------------------------------------------------------------------------


@cache
def _uncounted() -> ReportInput:
    """r2 with its gates read in each task: each task's pass rate states a mean, and no count."""
    return ReportInput.model_validate(gates_in_each_task())


def _per_task_rows(rows: list[SettleRow]) -> list[SettleRow]:
    return [row for row in rows if row.where.startswith("task.")]


def test_an_uncounted_tasks_rows_cannot_be_settled_without_an_assumption() -> None:
    held = _uncounted()
    (pick,) = dominant_options(fixture_inputs()["r2"])
    with pytest.raises(RefusedInput, match="no count to read a rate's spread from"):
        settle_cases(held, [pick], limit=WHOLE)


@pytest.mark.parametrize("assumed", [300, "share"])
def test_an_assumed_count_reads_each_uncounted_task_as_holding_it_keyed_as_an_assumption(assumed: int | str) -> None:
    held = _uncounted()
    (pick,) = dominant_options(fixture_inputs()["r2"])
    result = settle_cases(held, [pick], assumed=assumed, limit=WHOLE)  # type: ignore[arg-type]
    assert result.assumed and f"assumed.{assumed}." in result.assumed
    rows = _per_task_rows(result.rows)
    assert rows
    cases = held.scoreboard[pick].cases
    assert cases is not None
    for row in rows:
        task = next(task for task in held.tasks if f"task.{task.id}" == row.where)
        assert task.share is not None
        count = assumed if isinstance(assumed, int) else round(cases * float(task.share))
        stat = task.per_arm[pick]["pass_rate"]
        assert stat.mean is not None
        criterion = next(c for c in held.criteria if f"criterion {c.id}" == row.against)
        if criterion.statistic != "mean":
            assert row.cases_needed is None, "a p95 states no spread, assumed count or not"
            continue
        assert [figure.value for figure in row.counts] == [count]
        assert all(figure.key.startswith(f"assumed.{assumed}.") for figure in [*row.counts, *row.basis])
        share = float(stat.mean) / 100
        per_case = math.sqrt(share * (1 - share)) * 100
        distance = float(stat.mean) - float(criterion.threshold)
        if distance == 0:
            assert row.cases_needed is None and "exactly on its line" in row.unsettled
            continue
        assert row.cases_needed is not None and row.cases_needed.key.startswith(f"assumed.{assumed}.settle.")
        assert row.cases_needed.value == _least(per_case, distance, _NORMAL.inv_cdf(0.95))
        assert row.wrong_side is not None
        assert row.wrong_side.value == pytest.approx(_NORMAL.cdf(-abs(distance) / (per_case / math.sqrt(count))))


def test_arms_compare_reads_an_uncounted_task_under_an_assumption_and_a_counted_place_as_it_states() -> None:
    held = _uncounted()
    first, second = (arm.id for arm in held.arms[:2])
    plain = arms_compare(held, [first, second], "pass_rate", limit=WHOLE)
    assert all(row.interval is None for row in plain.rows if row.where.startswith("task."))
    assumed = arms_compare(held, [first, second], "pass_rate", assumed=400, limit=WHOLE)
    for row in assumed.rows:
        if row.where.startswith("task."):
            assert row.interval is not None and row.interval.key.startswith("assumed.400.compare.")
            basis = row.basis.value if row.basis is not None else ""
            assert isinstance(basis, str) and "assumed" in basis
        elif row.interval is not None:
            assert not row.interval.key.startswith("assumed.")


def test_an_assumption_that_fills_nothing_is_refused_naming_why() -> None:
    held = _tasks_counted()
    (pick,) = dominant_options(fixture_inputs()["r2"])
    with pytest.raises(RefusedInput, match="nothing to assume") as refused:
        settle_cases(held, [pick], assumed=300)
    assert refused.value.remedy
    first, second = (arm.id for arm in held.arms[:2])
    with pytest.raises(RefusedInput, match="nothing to assume"):
        arms_compare(held, [first, second], "pass_rate", assumed=300)


@pytest.mark.parametrize(
    ("key", "refused"),
    [
        ("assumed.300.spread.{arm}.pass_rate.overall.se", UnresolvableReference),
        ("assumed.2.5.spread.{arm}.pass_rate.task.{task}.se", UnresolvableReference),
        ("assumed.0.spread.{arm}.pass_rate.task.{task}.se", UnresolvableReference),
    ],
)
def test_an_assumed_key_overall_or_on_no_whole_count_is_refused(key: str, refused: type[Exception]) -> None:
    held = _uncounted()
    arm, task = held.arms[0].id, held.tasks[0].id
    with pytest.raises(refused):
        resolve(key.format(arm=arm, task=task), held)


def test_an_assumed_key_where_the_slice_states_its_count_is_refused() -> None:
    held = _tasks_counted()
    arm, task = held.arms[0].id, held.tasks[0].id
    with pytest.raises(UnresolvableReference, match="states its own"):
        resolve(f"assumed.300.spread.{arm}.pass_rate.task.{task}.se", held)
