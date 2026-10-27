"""The analyst's analyses: dominance, decision sensitivity and the measure breakdown, held to the decision code makes.

Contracts held here (``threetears.evals.analysis.analyst.analyses``, ``ops.analyst_analyses``,
``actions.analyst_analyses``, and the helpers they read in ``analysis.report_decision``):

- **dominance** names exactly the decision's survivors (``surviving_options``) and dominant options
  (``dominant_options``), each survivor's gate status and asked standing as the gate check and ``asked_standing``
  read them, and code's call as ``decision_at`` reads it, labelled as code's reading; with one option dominant, every
  other survivor names it among those dominating it; extra axes are descriptive and only narrow who dominates;
  where cost is the only axis the snapshot declares (``weighed_axes``), no option is dominant or dominated, the call
  is a tradeoff and says why, and the extra axes named are weighed beside cost (the undominated set the test
  computes from the figures);
- **sensitivity**: every flip is the figure its key states (a reading's ``flip_margin``, a floor read's interval end,
  a criterion's figure for the arm) inside the range; every segment's call equals ``decision_at`` at its midpoint,
  and its picks each routable task's ``routing_picks`` there; the call at each flip is the segment it falls in by
  ``boundary_belongs_to``; ``holds`` is whether every segment's call is the declared one, and agrees with the
  late-margin check, and ``picks_hold`` whether every segment's picks are the declared ones; a gate that moves a
  routed plan without moving the one-option call says so; a **hypothetical line** sweeps a gate the snapshot does not
  declare, its flips the arms' own figures and its call ``decision_at`` with that gate added;
- **breakdown**: each candidate's ``vs_control`` is its mean minus the control's in that slice, its loss shares over
  one stratification sum to 1 where every stratum states one, and ``concentrated_in`` is the largest; with no control
  it is refused, pointing to ``measure_read`` by stratum; tasks and cells slice where the input holds them;
- **every fixture snapshot** (r1 to r5, the misfit inputs, Kairo round 2): each analysis, on every parameter the input
  holds, answers or refuses, never raises, and every figure it returns resolves to its value;
- **report_decision's helpers**: ``decision_at`` with no overrides is the declared decision, and refuses an override
  naming nothing the input holds.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from functools import cache
from typing import Any

import pytest

from packages.evals.tests.analyst_support import fixture_inputs, fixture_snapshots
from packages.evals.tests.fixtures.report_plans.misfits import reference_data
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, OpsFixture, ops_fixture
from threetears.evals.actions import ActionOutcome, MountedTool, analyst_tools, eval_catalogue
from threetears.evals.analysis.analyst.analyses import GateRead, breakdown, dominance, sensitivity
from threetears.evals.analysis.analyst.snapshot import frozen_input, report_snapshot
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.figure_keys import resolve
from threetears.evals.analysis.report_decision import (
    DecisionAt,
    allowed_modes,
    asked_standing,
    caveat_materiality,
    control_arm,
    decision_at,
    dominant_options,
    margin_sensitivity,
    surviving_options,
    weighed_axes,
)
from threetears.evals.analysis.report_gates import gate_check, gate_findings, routing_picks
from threetears.evals.analysis.report_input import Criterion, ReportInput
from threetears.evals.analysis.report_margins import criterion_figures, flip_margin
from threetears.evals.kernel import EvalReportSnapshot
from threetears.evals.kernel.errors import RefusedInput

#: Every page a test reads whole asks for this many rows.
WHOLE = 200

#: The fields every cited figure carries, by which one is found in a result's JSON.
FIGURE_FIELDS = frozenset({"key", "reads", "value", "kind", "scale", "unit"})


@cache
def _all_inputs() -> dict[str, ReportInput]:
    """Every fixture snapshot's input, by name: r1 to r5, the misfits, and Kairo round 2."""
    return {name: frozen_input(snapshot) for name, snapshot in fixture_snapshots("s").items()}


def _parameters(held: ReportInput) -> list[str]:
    """Every criterion the input can be swept on: each aspect's margin, each floor, each criterion's threshold."""
    floors = dict.fromkeys(f"floor:{read.aspect}/{read.floor}" for read in held.floors)
    return [
        *(f"margin:{aspect.id}" for aspect in held.aspects),
        *floors,
        *(f"threshold:{criterion.id}" for criterion in held.criteria),
    ]


def _arm_id(named: str) -> str:
    assert named.startswith("arm:"), named
    return named.removeprefix("arm:")


def _number(value: object) -> float:
    assert isinstance(value, int | float), value
    return float(value)


def _call(decision: DecisionAt) -> tuple[list[str], list[str]]:
    return sorted(decision.modes), [f"arm:{arm}" for arm in decision.dominant]


def _figures(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        if FIGURE_FIELDS <= node.keys():
            yield node
            return
        for item in node.values():
            yield from _figures(item)
    elif isinstance(node, list):
        for item in node:
            yield from _figures(item)


def _each_figure_resolves(structured: dict[str, Any], held: ReportInput) -> None:
    for figure in _figures(structured):
        value = resolve(figure["key"], held).value
        assert figure["value"] == (list(value) if isinstance(value, tuple) else value), figure["key"]


# =============================================================================
# report_decision's helpers
# =============================================================================


@pytest.mark.parametrize("name", sorted(_all_inputs()))
def test_the_decision_with_no_override_is_the_declared_decision(name: str) -> None:
    held = _all_inputs()[name]
    declared = decision_at(held)
    assert declared.modes == allowed_modes(held)
    dominant = dominant_options(held)
    assert declared.dominant == (tuple(dominant) if len(dominant) == 1 else ())


def test_an_override_naming_nothing_the_input_holds_is_refused() -> None:
    held = fixture_inputs()["r5"]
    with pytest.raises(ValueError, match="holds no aspect 'no such aspect'"):
        decision_at(held, margins={"no such aspect": 3.0})
    with pytest.raises(ValueError, match="reads no floor 'no/such/floor'"):
        decision_at(held, floors={"no/such/floor": 3.0})


def test_the_asked_standing_of_a_change_and_a_choice() -> None:
    change = fixture_inputs()["r5"]
    standing = asked_standing(change)
    control = control_arm(change)
    assert control is not None and standing[control].standing == "control" and standing[control].level == 1
    assert {held.standing for held in standing.values()} <= {
        "control",
        "shown_better",
        "shown_no_worse",
        "not_shown_no_worse",
    }
    choice = fixture_inputs()["r1"]
    assert all(
        held.level is None and held.standing in ("behind", "not_behind") for held in asked_standing(choice).values()
    )


# =============================================================================
# Dominance
# =============================================================================


def _declared_axes(held: ReportInput) -> list[str]:
    """The axes a snapshot declares, read off its own fields rather than through ``weighed_axes``: a gate or guardrail
    criterion or a guardrail read, a control-role arm, an asked aspect, a contrast family's merit priority, and an arm
    whose ``cost_per_1k`` key reads a number."""
    costed = []
    for arm in held.arms:
        try:
            costed.append(isinstance(resolve(f"cost_per_1k.{arm.id}", held).value, int | float))
        except UnresolvableReference:
            costed.append(False)
    declared = {
        "gates": any(criterion.kind in ("gate", "guardrail") for criterion in held.criteria) or bool(held.guardrails),
        "control": any(arm.role == "control" for arm in held.arms),
        "asked": any(aspect.asked for aspect in held.aspects),
        "ranking": any(family.merit_axes for family in held.contrast_families),
        "cost": any(costed),
    }
    return [axis for axis, holds in declared.items() if holds]


def _cost_alone(held: ReportInput) -> bool:
    """Whether cost is the only axis the snapshot declares, with more than one option to rank by it."""
    return set(_declared_axes(held)) <= {"cost"} and len(surviving_options(held)) > 1


@pytest.mark.parametrize("name", sorted(_all_inputs()))
def test_dominance_names_the_decisions_survivors_dominant_options_and_call(name: str) -> None:
    held = _all_inputs()[name]
    result = dominance(held).model_dump(mode="json")
    survivors = surviving_options(held)
    assert [_arm_id(row["arm"]) for row in result["survivors"]] == survivors
    assert [_arm_id(row["arm"]) for row in result["out"]] == [arm.id for arm in held.arms if arm.id not in survivors]
    alone = _cost_alone(held)
    dominant = [] if alone else dominant_options(held)
    assert [_arm_id(arm) for arm in result["dominant"]] == dominant
    assert result["weighs"] == list(weighed_axes(held)) == _declared_axes(held)
    assert result["dominant_with_axes"] == result["dominant"] and result["axes"] == []
    standing = asked_standing(held)
    for row in result["survivors"]:
        arm = _arm_id(row["arm"])
        check = gate_check(held, arm)
        assert row["gate_status"] == check.status
        assert len(row["borders"]) == len(check.borders), "a borderline figure per reason the gate check gives"
        assert row["standing"] == standing[arm].standing
        assert [_arm_id(other) for other in row["behind"]] == list(standing[arm].behind)
        if arm in dominant:
            assert row["dominated_by"] == [], f"{name}: dominant {arm} is dominated by {row['dominated_by']}"
        elif len(dominant) == 1:
            assert f"arm:{dominant[0]}" in row["dominated_by"], f"{name}: {arm} is not dominated by the one dominant"
    for row in result["out"]:
        assert row["gates"] or row["breached"], f"{name}: {row['arm']} was removed with no reason"
    code = result["code_call"]
    if alone:
        assert (code["modes"], code["dominant"]) == (["tradeoff"], [])
        assert "cost is the only axis" in code["withheld"]
        assert all(row["dominated_by"] == [] for row in result["survivors"])
    else:
        assert (code["modes"], code["dominant"]) == _call(decision_at(held))
        assert code["withheld"] == ""
    assert "never a constraint" in code["reads_as"]
    _each_figure_resolves(result, held)


def test_the_fixtures_hold_inputs_declaring_nothing_but_cost_and_the_rule_alone_would_name_the_cheapest() -> None:
    """The control for the cost-alone path: misfits declare nothing but cost, and there the planner's rule names one
    dominant option, the cheapest, which the analysis withholds."""
    alone = [name for name, held in _all_inputs().items() if _cost_alone(held)]
    assert "r2 without its gates" in alone and "r1 reading only bars" in alone
    declared = {axis for held in _all_inputs().values() for axis in _declared_axes(held)}
    # No fixture declares a merit ranking, so that axis is held only by its own field's reading above.
    assert declared >= {"gates", "control", "asked", "cost"}, "some fixture declares each axis"
    for name in alone:
        held = _all_inputs()[name]
        (cheapest,) = dominant_options(held)
        costs = {arm: resolve(f"cost_per_1k.{arm}", held).value for arm in surviving_options(held)}
        assert costs[cheapest] == min(costs.values()), name
        assert dominance(held).dominant == [], name


@pytest.mark.parametrize("name", ["r2 without its gates", "r1 reading only bars"])
def test_with_nothing_but_cost_declared_the_axes_named_are_weighed_beside_cost(name: str) -> None:
    held = _all_inputs()[name]
    axes = [key for key in [*held.judged, *held.measures] if key not in ("cost_per_1k",)]
    result = dominance(held, axes, limit=WHOLE)
    survivors = surviving_options(held)
    costs = {arm: _number(resolve(f"cost_per_1k.{arm}", held).value) for arm in survivors}
    compared: list[tuple[dict[str, float], bool]] = []
    for row in result.axes:
        if row.direction == "unstated":
            continue
        values = {_arm_id(item.arm): _number(item.figure.value) for item in row.arms if item.figure is not None}
        compared.append((values, row.direction == "higher_is_better"))
    assert compared, "the control: an axis with a direction is compared"

    def at_least(first: str, second: str) -> bool:
        if costs[first] > costs[second]:
            return False
        for values, higher in compared:
            if first in values and second in values:
                if (values[first] < values[second]) if higher else (values[first] > values[second]):
                    return False
            elif first not in values and second in values:
                return False
            elif first in values and second not in values:
                return False
        return True

    expected = [arm for arm in survivors if all(at_least(arm, other) for other in survivors if other != arm)]
    assert [_arm_id(arm) for arm in result.dominant_with_axes] == expected
    for row in result.survivors:
        arm = _arm_id(row.arm)
        ahead = [other for other in survivors if other != arm and at_least(other, arm) and not at_least(arm, other)]
        assert [_arm_id(other) for other in row.dominated_with_axes] == ahead, arm
        assert row.dominated_by == []
    assert any(row.dominated_with_axes for row in result.survivors), "the control: the axes rank some options"


def _gate_figures_as_found(held: ReportInput, arm: str, gates: Sequence[GateRead], status: str) -> None:
    """Each gate read is one gate-check finding of that status, in order, citing the criterion's threshold and, where
    the input states the arm's figure, that figure and its gap."""
    findings = [finding for finding in gate_findings(held, arm) if finding.status == status]
    assert [(gate.criterion, gate.status) for gate in gates] == [
        (finding.criterion, status if finding.figure is not None else "no_figure") for finding in findings
    ]
    for gate, finding in zip(gates, findings, strict=True):
        criterion = next(held_criterion for held_criterion in held.criteria if held_criterion.id == gate.criterion)
        cited = {figure.key.rsplit(".", 1)[1]: figure.value for figure in gate.figures}
        assert cited["threshold"] == float(criterion.threshold)
        if finding.figure is not None:
            assert cited["value"] == pytest.approx(finding.figure.value) and "gap" in cited
            if finding.figure.ci is not None:
                assert (cited["low"], cited["high"]) == pytest.approx(finding.figure.ci)
        else:
            assert set(cited) == {"threshold"}, "no figure there: the threshold alone"


@pytest.mark.parametrize("name", sorted(_all_inputs()))
def test_a_removed_option_carries_each_figure_it_fails_on_and_a_survivor_each_it_holds_borderline(name: str) -> None:
    held = _all_inputs()[name]
    result = dominance(held)
    for row in result.out:
        _gate_figures_as_found(held, _arm_id(row.arm), row.gates, "fail")
    for survivor in result.survivors:
        _gate_figures_as_found(held, _arm_id(survivor.arm), survivor.borders, "borderline")


def test_the_gate_figures_walk_has_a_removed_option_failing_on_a_figure_and_a_survivor_borderline() -> None:
    """The control for the walk above: the fixtures hold an option removed on a stated figure, and one borderline."""
    results = [dominance(held) for held in _all_inputs().values()]
    assert any(gate.status == "fail" for result in results for row in result.out for gate in row.gates)
    assert any(row.borders for result in results for row in result.survivors)


@pytest.mark.parametrize("name", ["r1", "r2", "r4", "r5", "kairo round 2"])
def test_extra_axes_are_descriptive_and_only_narrow_who_dominates(name: str) -> None:
    held = _all_inputs()[name]
    axes = [*held.measures, *held.judged]
    result = dominance(held, axes).model_dump(mode="json")
    assert [row["axis"] for row in result["axes"]] == axes
    assert set(result["dominant_with_axes"]) <= set(result["dominant"])
    for row in result["axes"]:
        assert row["descriptive"] is True
        values = {item["arm"]: item["figure"]["value"] for item in row["arms"] if item["figure"] is not None}
        if row["direction"] == "unstated" or not values:
            assert row["best"] == []
            continue
        top = (max if row["direction"] == "higher_is_better" else min)(values.values())
        assert row["best"] == [arm for arm, value in values.items() if value == top]
    _each_figure_resolves(result, held)


# =============================================================================
# Sensitivity
# =============================================================================


def _expected_flips(held: ReportInput, parameter: str) -> dict[tuple[str, str | None], float]:
    """Each flip a sweep must list, by arm and which flip, as the input's own figures give it: a margin's and a
    floor's from zero up, a threshold's anywhere."""
    kind, _colon, name = parameter.partition(":")
    found: dict[tuple[str, str | None], float] = {}
    if kind == "margin":
        aspect = next(aspect for aspect in held.aspects if aspect.id == name)
        for reading in aspect.readings:
            flip = flip_margin(reading)
            if flip is not None:
                found.setdefault((reading.arm, None), flip)
    elif kind == "floor":
        for read in held.floors:
            if f"{read.aspect}/{read.floor}" == name and read.interval is not None:
                found[(read.arm, "held_above")] = -float(read.interval[0])
                found[(read.arm, "broken_at")] = -float(read.interval[1])
    else:
        criterion = next(criterion for criterion in held.criteria if criterion.id == name)
        for arm in held.arms:
            figures = criterion_figures(held, criterion, arm.id)
            for figure in figures:
                if not math.isfinite(figure.value):
                    continue
                cell = None if figure.cell is None else held.cells[figure.cell]
                place = figure.task if cell is None else f"{cell.row}.{cell.column}"
                found[(arm.id, place if len(figures) > 1 else None)] = figure.value
        return found
    return {flip: value for flip, value in found.items() if value >= 0}


def _moved_input(held: ReportInput, parameter: str, value: float) -> ReportInput:
    """The input with a swept threshold at ``value``; a margin or floor sweep moves no criterion of the input."""
    kind, _colon, name = parameter.partition(":")
    if kind != "threshold":
        return held
    criteria = [
        criterion.model_copy(update={"threshold": value}) if criterion.id == name else criterion
        for criterion in held.criteria
    ]
    return held.model_copy(update={"criteria": criteria})


def _decision_at(held: ReportInput, parameter: str, value: float) -> tuple[list[str], list[str]]:
    """Code's call with the swept criterion at ``value``, read independently of the sweep."""
    kind, _colon, name = parameter.partition(":")
    if kind == "margin":
        return _call(decision_at(held, margins={name: value}))
    if kind == "floor":
        floor = next(read.floor for read in held.floors if f"{read.aspect}/{read.floor}" == name)
        return _call(decision_at(held, floors={floor: value}))
    return _call(decision_at(_moved_input(held, parameter, value)))


def _picks(held: ReportInput) -> list[dict[str, Any]]:
    """Each routable task's pick on an input, as a segment lists it: the gate matrix's routing, read independently."""
    picks = routing_picks(held, [arm.id for arm in held.arms])
    return [
        {"task": task.id, "arm": f"arm:{picks[task.id]}" if task.id in picks else None}
        for task in held.tasks
        if task.routable
    ]


def _end(segment: dict[str, Any], side: str) -> float | None:
    end = segment[side]
    return None if end is None else float(end["at"]["value"])


def _probes(flips: Sequence[float], low: float | None) -> list[float]:
    """Values to read a sweep's call at, chosen from the input's own flips and not the sweep's rule: just below and
    just above every flip, each end of the range, and far past the last flip and below the first; a flip the sweep
    missed sits between two of them, on either side of it."""
    values = sorted(set(flips))
    gaps = [high - low_end for low_end, high in zip(values, values[1:], strict=False) if high > low_end]
    epsilon = min([1e-6, *(gap / 4 for gap in gaps)])
    reach = max([1.0, *(abs(value) for value in values)])
    points = {value + side * epsilon * max(1.0, abs(value)) for value in values for side in (-1, 1)}
    points |= {reach * 10, -reach * 10, *(value for value in values)}
    if low is not None:
        points = {point for point in points if point >= low} | {low}
    return sorted(points)


def _holding(segments: list[dict[str, Any]], value: float, boundary: str) -> list[dict[str, Any]]:
    """The segments a value falls in, a value at a flip falling in the piece ``boundary`` names."""
    held = []
    for segment in segments:
        low, high = _end(segment, "low"), _end(segment, "high")
        point = low is not None and low == high == value
        if boundary == "below":
            inside = (low is None or low < value or point) and (high is None or value <= high)
        else:
            inside = (low is None or low <= value) and (high is None or value < high or point)
        if inside:
            held.append(segment)
    return held


def _sweeps() -> list[tuple[str, str]]:
    return [(name, parameter) for name, held in _all_inputs().items() for parameter in _parameters(held)]


@pytest.mark.parametrize(("name", "parameter"), _sweeps(), ids=lambda value: str(value))
def test_each_flip_is_its_figure_and_the_call_at_every_probe_is_its_segments(name: str, parameter: str) -> None:
    held = _all_inputs()[name]
    result = sensitivity(held, parameter, limit=WHOLE).model_dump(mode="json")
    _each_figure_resolves(result, held)

    expected = _expected_flips(held, parameter)
    listed = {(_arm_id(flip["arm"]), flip["where"]): flip["at"]["value"] for flip in result["flips"]}
    assert listed == expected, name
    assert result["total"] == len(expected) and result["next_offset"] is None

    low = 0.0 if not parameter.startswith("threshold:") else None
    segments = result["segments"]
    assert segments, "a sweep always has a segment"
    for point in _probes(list(expected.values()), low):
        (segment,) = _holding(segments, point, result["boundary_belongs_to"])
        assert (segment["modes"], segment["dominant"]) == _decision_at(held, parameter, point), (name, point)
        assert segment["picks"] == _picks(_moved_input(held, parameter, point)), (name, point)
    for left, right in zip(segments, segments[1:], strict=False):
        assert (left["modes"], left["dominant"], left["picks"]) != (
            right["modes"],
            right["dominant"],
            right["picks"],
        ), "adjacent segments differ in the call or a pick"
        assert left["high"] == right["low"], "segments meet at a flip"
    for value in {flip["at"]["value"] for flip in result["flips"]}:
        (segment,) = _holding(segments, value, result["boundary_belongs_to"])
        assert (segment["modes"], segment["dominant"]) == _decision_at(held, parameter, value), (name, value)
    declared = _call(decision_at(held))
    assert (result["declared_call"]["modes"], result["declared_call"]["dominant"]) == declared
    assert result["holds"] == all((segment["modes"], segment["dominant"]) == declared for segment in segments)
    assert result["declared_picks"] == _picks(held)
    assert result["picks_hold"] == all(segment["picks"] == _picks(held) for segment in segments)
    assert result["hypothetical"] is False


def test_a_gate_sweep_says_where_a_routed_plan_moves_though_the_one_option_call_does_not() -> None:
    """The W2 r1 finding: a cross-cutting gate's sweep read only the one-option call. Where the snapshot holds routable
    tasks, some flip moves a task's pick without moving the call, and the sweep says so."""
    held = _all_inputs()["r1"]
    assert any(task.routable for task in held.tasks) and any(task.cross_cutting for task in held.tasks)
    gates = [criterion.id for criterion in held.criteria if criterion.kind == "gate"]
    flips = [flip for gate in gates for flip in sensitivity(held, f"threshold:{gate}", limit=WHOLE).flips]
    assert any(flip.picks_change and not flip.call_changes for flip in flips)
    assert not all(sensitivity(held, f"threshold:{gate}", limit=WHOLE).picks_hold for gate in gates)
    no_routing = _all_inputs()["r5"]
    assert not any(task.routable for task in no_routing.tasks)
    result = sensitivity(no_routing, f"margin:{no_routing.aspects[0].id}")
    assert result.declared_picks == [] and result.picks_hold and all(segment.picks == [] for segment in result.segments)


def _line_scope(held: ReportInput, key: str, statistic: str) -> str:
    """Where a hypothetical line reads, as the contract says: overall where an arm states the statistic overall, else in
    each task, else in each cell."""
    table = held.measures.get(key) or held.judged[key]
    if any(getattr(stat, statistic) is not None for stat in table.per_arm.values()):
        return "overall"
    if any(key in figures for task in held.tasks for figures in task.per_arm.values()):
        return "each_task"
    return "each_cell"


def _with_gate(held: ReportInput, key: str, statistic: str, value: float) -> ReportInput:
    higher = True if key in held.judged else held.measures[key].higher_is_better
    gate = Criterion.model_validate(
        {
            "id": "line under test",
            "kind": "gate",
            "measure": key,
            "statistic": statistic,
            "direction": "higher_is_better" if higher else "lower_is_better",
            "threshold": value,
            "scope": _line_scope(held, key, statistic),
        }
    )
    return held.model_copy(update={"criteria": [*held.criteria, gate]})


@pytest.mark.parametrize(
    ("name", "parameter"),
    [
        ("r2 without its gates", "hypothetical:pass_rate"),
        ("r2 without its gates", "hypothetical:latency_ms/p95"),
        ("r1 reading only bars", "hypothetical:overall"),
        ("r3", "hypothetical:accuracy"),
        ("kairo round 2", "hypothetical:persona.holds_character"),
    ],
)
def test_a_hypothetical_line_sweeps_a_gate_the_snapshot_does_not_declare(name: str, parameter: str) -> None:
    held = _all_inputs()[name]
    spec = parameter.removeprefix("hypothetical:")
    key, _slash, statistic = spec.partition("/")
    statistic = statistic or "mean"
    result = sensitivity(held, parameter, limit=WHOLE)
    assert result.hypothetical and result.declared is None
    assert "does not declare" in result.swept and any("hypothetical line" in note for note in result.notes)
    _each_figure_resolves(result.model_dump(mode="json"), held)
    gate = _with_gate(held, key, statistic, 0.0).criteria[-1]
    expected = sorted(
        figure.value
        for arm in held.arms
        for figure in criterion_figures(held, gate, arm.id)
        if math.isfinite(figure.value)
    )
    assert sorted(_number(flip.at.value) for flip in result.flips) == pytest.approx(expected)
    assert all(not flip.at.key.startswith("settle.") for flip in result.flips), "a flip is the arm's own figure"
    segments = [segment.model_dump(mode="json") for segment in result.segments]
    for point in _probes(expected, None):
        (segment,) = _holding(segments, point, result.boundary_belongs_to)
        assert (segment["modes"], segment["dominant"]) == _call(decision_at(_with_gate(held, key, statistic, point)))
    assert any(segment.dominant for segment in result.segments), "the control: some line names one option"


@pytest.mark.parametrize("name", ["r5", "kairo round 2"])
def test_a_margin_sweep_holds_where_the_late_margin_check_finds_the_margin_immaterial(name: str) -> None:
    held = _all_inputs()[name]
    (aspect,) = [aspect for aspect in held.aspects if aspect.margin is not None]
    result = sensitivity(held, f"margin:{aspect.id}")
    assert result.holds == margin_sensitivity(held).holds
    assert result.holds == caveat_materiality(held)["late:margin"].immaterial
    assert result.declared is not None and result.declared.key == f"aspects.{aspect.id}.margin"


def test_a_sweep_over_a_range_cuts_only_inside_it_and_pages_its_flips() -> None:
    held = fixture_inputs()["r2"]
    whole = sensitivity(held, "threshold:pass_rate", limit=WHOLE)
    values = sorted(float(flip.at.value) for flip in whole.flips if isinstance(flip.at.value, float))
    low, high = (values[2] + values[3]) / 2, (values[-3] + values[-2]) / 2
    ranged = sensitivity(held, "threshold:pass_rate", range_low=low, range_high=high, limit=WHOLE)
    assert {flip.at.key for flip in ranged.flips} == {
        flip.at.key for flip in whole.flips if low <= flip.at.value <= high
    }
    assert ranged.segments[0].low is None and ranged.segments[-1].high is None
    assert "flips outside the range swept are not listed" in ranged.notes
    segments = [segment.model_dump(mode="json") for segment in ranged.segments]
    inside = [float(flip.at.value) for flip in ranged.flips if isinstance(flip.at.value, float)]
    for point in [low, high, *(point for point in _probes(inside, low) if point <= high)]:
        (segment,) = _holding(segments, point, ranged.boundary_belongs_to)
        assert (segment["modes"], segment["dominant"]) == _decision_at(held, "threshold:pass_rate", point), point

    seen: list[str] = []
    offset: int | None = 0
    while offset is not None:
        page = sensitivity(held, "threshold:pass_rate", limit=4, offset=offset)
        seen += [flip.at.key for flip in page.flips]
        offset = page.next_offset
    assert seen == [flip.at.key for flip in whole.flips]


def test_a_criterion_that_never_gates_holds_at_any_threshold() -> None:
    data = reference_data("r4")
    data["criteria"] = [{**criterion, "kind": "bar"} for criterion in data["criteria"]]
    held = ReportInput.model_validate(data)
    result = sensitivity(held, f"threshold:{held.criteria[0].id}")
    assert result.holds and len(result.segments) == 1
    assert any("never gates" in note for note in result.notes)


# =============================================================================
# Breakdown
# =============================================================================


def _with_a_control(name: str) -> ReportInput:
    """A choice made a change: its first arm read as the control, so its tasks or cells slice against it."""
    data = reference_data(name)
    data["arms"][0]["role"] = "control"
    return ReportInput.model_validate(data)


@pytest.mark.parametrize(
    ("held", "measure", "slice_by"),
    [
        (fixture_inputs()["r4"], "persona", "stratum"),
        (fixture_inputs()["r5"], "persona.holds_character", "stratum"),
        (_all_inputs()["kairo round 2"], "persona.holds_character", "stratum"),
        (_with_a_control("r1"), "overall", "task"),
        (_with_a_control("r3"), "accuracy", "cell"),
    ],
    ids=["r4 strata", "r5 strata", "kairo strata", "r1 tasks", "r3 cells"],
)
def test_each_slice_reads_each_candidate_against_the_control_and_its_loss_shares_sum_to_one(
    held: ReportInput, measure: str, slice_by: Any
) -> None:
    result = breakdown(held, measure, slice_by=slice_by, limit=WHOLE).model_dump(mode="json")
    _each_figure_resolves(result, held)
    control = control_arm(held)
    assert result["control"] == f"arm:{control}" and result["exploratory"] is True
    assert result["slices"] and result["total"] == len(result["slices"])
    groups: dict[tuple[str, str | None, str | None], list[float | None]] = {}
    for row in result["slices"]:
        for shown in row["arms"]:
            if shown["vs_control"] is not None:
                assert row["control"] is not None and shown["mean"] is not None
                assert shown["vs_control"]["value"] == pytest.approx(shown["mean"]["value"] - row["control"]["value"])
            if slice_by != "stratum":
                assert shown["loss_share"] is None
                continue
            stratum = next(one for one in held.strata if one.id == row["id"] and one.aspect == row["aspect"])
            share = None if shown["loss_share"] is None else shown["loss_share"]["value"]
            groups.setdefault((shown["arm"], stratum.aspect, stratum.dimension), []).append(share)
    summed = 0
    for (arm, aspect, dimension), shares in groups.items():
        if all(share is not None for share in shares):
            assert math.fsum(share for share in shares if share is not None) == pytest.approx(1.0), arm
            summed += 1
            (concentrated,) = [
                row
                for row in result["concentrated_in"]
                if (row["arm"], row["aspect"], row["dimension"]) == (arm, aspect, dimension)
            ]
            assert concentrated["loss_share"]["value"] == max(share for share in shares if share is not None)
    assert summed or slice_by != "stratum", "the control: some stratification states every loss share"


def test_a_breakdown_pages_its_slices_and_names_where_the_loss_sits_across_every_page() -> None:
    held = _all_inputs()["kairo round 2"]
    whole = breakdown(held, "persona.holds_character", limit=WHOLE)
    seen: list[str] = []
    offset: int | None = 0
    while offset is not None:
        page = breakdown(held, "persona.holds_character", limit=5, offset=offset)
        assert page.concentrated_in == whole.concentrated_in
        seen += [row.id for row in page.slices]
        offset = page.next_offset
    assert seen == [row.id for row in whole.slices]
    one = breakdown(held, "persona.holds_character", arm=held.arms[1].id)
    assert one.arms == [f"arm:{held.arms[1].id}"]


@pytest.mark.parametrize("name", ["r1", "r2", "r3"])
def test_a_breakdown_with_no_control_is_refused_pointing_to_the_read_by_stratum(name: str) -> None:
    held = fixture_inputs()[name]
    assert control_arm(held) is None, "the control: the input has no control"
    with pytest.raises(RefusedInput) as refused:
        breakdown(held, next(iter(held.measures)))
    assert "no control" in refused.value.fault and "measure_read (by=stratum)" in refused.value.remedy


# =============================================================================
# Through the analyst's tool, on every fixture snapshot
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


async def _call_tool(tool: MountedTool, fixture: OpsFixture, arguments: dict[str, Any]) -> ActionOutcome:
    return await tool.call(arguments, host=fixture.host, caller=CALLER)


def _every_call(held: ReportInput) -> Iterator[dict[str, Any]]:
    keys = [*held.measures, *held.judged]
    yield {"action": "options_dominance"}
    yield {"action": "options_dominance", "axes": keys}
    for parameter in _parameters(held):
        yield {"action": "decision_sensitivity", "parameter": parameter}
    for slice_by in ("stratum", "task", "cell"):
        for key in keys:
            yield {"action": "measure_breakdown", "measure_key": key, "slice_by": slice_by}


async def test_every_analysis_on_every_fixture_snapshot_answers_or_refuses_and_never_raises() -> None:
    fixture, snapshots = off_loop(_seeded)
    tool = _analyst_tool()
    answered: set[str] = set()
    for name, snapshot in snapshots.items():
        held = frozen_input(snapshot)
        for arguments in _every_call(held):
            outcome = await _call_tool(tool, fixture, {**arguments, "snapshot_id": snapshot.id})
            where = f"{arguments} on {name}"
            assert outcome.ending in ("ok", "refused"), f"{where}: {outcome.text}"
            if outcome.ending == "refused":
                assert outcome.refusal is not None and outcome.refusal.kind == "validation_failed", where
                assert "remedy:" in outcome.text, where
                continue
            assert outcome.structured is not None
            _each_figure_resolves(outcome.structured, held)
            answered.add(arguments["action"])
    assert answered == {"options_dominance", "decision_sensitivity", "measure_breakdown"}


@pytest.mark.parametrize(
    ("arguments", "says"),
    [
        ({"action": "decision_sensitivity", "parameter": "weighting:tone"}, "is none of margin:<aspect>"),
        ({"action": "decision_sensitivity", "parameter": "margin:no such aspect"}, "names no aspect"),
        ({"action": "decision_sensitivity", "parameter": "floor:no/such"}, "names no floor"),
        (
            {"action": "decision_sensitivity", "parameter": "margin:character adherence (v2)", "range_low": -1.0},
            "swept from zero up",
        ),
        (
            {
                "action": "decision_sensitivity",
                "parameter": "margin:character adherence (v2)",
                "range_low": 9.0,
                "range_high": 3.0,
            },
            "range_low is above range_high",
        ),
        ({"action": "options_dominance", "axes": ["no_such_measure"]}, "no measure or judged dimension"),
        ({"action": "measure_breakdown", "measure_key": "no_such_measure"}, "names no measure"),
        (
            {"action": "measure_breakdown", "measure_key": "persona.holds_character", "arm_filter": "September"},
            "is the control every slice is read against",
        ),
        (
            {"action": "measure_breakdown", "measure_key": "persona.holds_character", "slice_by": "task"},
            "holds no task",
        ),
        (
            {
                "action": "measure_breakdown",
                "measure_key": "persona.holds_character",
                "slice_by": "cell",
                "aspect_filter": "character adherence (v2)",
            },
            "a cell is not held on an aspect",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else str(value.get("parameter") or value),
)
async def test_an_analysis_refuses_arguments_naming_nothing_the_snapshot_holds_with_a_remedy(
    arguments: dict[str, Any], says: str
) -> None:
    fixture, snapshots = off_loop(_seeded)
    outcome = await _call_tool(_analyst_tool(), fixture, {**arguments, "snapshot_id": snapshots["kairo round 2"].id})
    assert outcome.ending == "refused" and outcome.refusal is not None, outcome.text
    assert outcome.refusal.kind == "validation_failed"
    assert says in outcome.text and "remedy:" in outcome.text


async def test_a_breakdown_on_a_snapshot_with_no_control_is_refused_as_the_call() -> None:
    fixture, snapshots = off_loop(_seeded)
    outcome = await _call_tool(
        _analyst_tool(),
        fixture,
        {
            "action": "measure_breakdown",
            "snapshot_id": snapshots["r1"].id,
            "measure_key": "overall",
            "slice_by": "task",
        },
    )
    assert outcome.refusal is not None and outcome.refusal.kind == "validation_failed"
    assert "measure_read (by=stratum)" in outcome.text


def test_a_snapshot_with_a_control_on_tasks_is_a_snapshot_the_breakdown_reads() -> None:
    held = _with_a_control("r1")
    snapshot = report_snapshot(held, scope_id="s", campaign_id=None, source="report_input", created_by="me")
    assert breakdown(frozen_input(snapshot), "overall", slice_by="task").total == len(held.tasks)
