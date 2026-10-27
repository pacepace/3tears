"""How far a sensitivity sweep reaches, and what its per-task picks hold back: said, never left to the reader.

Contracts held here (``threetears.evals.analysis.analyst.analyses.sensitivity``):

- **every flip by default**: with no range, a sweep lists every flip its parameter has and names no unswept side;
- **a range that stops short says so**: where the caller's range leaves out a flip at which the call or a routable
  task's pick changes, ``unswept`` names that side, the nearest such flip, each such flip by its key, the call past it
  (code's call read independently with the line moved there), and whether that side loosens the line; a side with no
  such flip is never named. Held on every fixture and parameter, and on the W2 regression prototype's r2 miss (the
  pass-rate line swept from its declared value up, which hid the cheaper picks below it);
- **a range cut exactly at a flip**: a flip on the range's own end is past it where the end's value falls in the piece
  inside the range (the range's high end on a line whose boundary value belongs below, its low end on one whose
  boundary value belongs above), so the change just beyond it is named; and every flip's ``call_changes`` reads the
  two sides of its value whatever the range, so a flip on the end never reads as moving nothing. Held with ranges cut
  at every flip value of every sweep, on both boundary kinds;
- **a margin over an aspect with no reading of a candidate** says code's rule never reads that candidate against a
  margin (the r4 margin sweep that reported no flip and no reason);
- **the per-task picks' cross-cutting gates**: each arm a cross-cutting task's gate keeps out of every pick is listed
  with the gates it fails there, by their keys, and the routable tasks whose own gates it passes (the r1 finding).
"""

from __future__ import annotations

from functools import cache

import pytest

from packages.evals.tests.analyst_support import KAIRO, fixture_inputs, fixture_snapshots
from threetears.evals.analysis.analyst.analyses import Sensitivity, sensitivity
from threetears.evals.analysis.analyst.snapshot import frozen_input
from threetears.evals.analysis.report.figure_keys import resolve
from threetears.evals.analysis.report_decision import decision_at
from threetears.evals.analysis.report_gates import gate_check, gate_findings, routing_picks
from threetears.evals.analysis.report_input import ReportInput

#: Every flip a test reads asks for this many.
WHOLE = 500


@cache
def _inputs() -> dict[str, ReportInput]:
    inputs = dict(fixture_inputs())
    inputs[KAIRO] = frozen_input(fixture_snapshots("s")[KAIRO])
    return inputs


def _parameters(held: ReportInput) -> list[str]:
    floors = dict.fromkeys(f"floor:{read.aspect}/{read.floor}" for read in held.floors)
    return [
        *(f"margin:{aspect.id}" for aspect in held.aspects),
        *floors,
        *(f"threshold:{criterion.id}" for criterion in held.criteria),
    ]


def _sweeps() -> list[tuple[str, str]]:
    return [(name, parameter) for name, held in _inputs().items() for parameter in _parameters(held)]


def _values(result: Sensitivity) -> list[float]:
    return sorted({float(flip.at.value) for flip in result.flips if isinstance(flip.at.value, int | float)})


@pytest.mark.parametrize(("name", "parameter"), _sweeps(), ids=lambda value: str(value))
def test_a_sweep_with_no_range_walks_every_flip_and_names_no_unswept_side(name: str, parameter: str) -> None:
    held = _inputs()[name]
    result = sensitivity(held, parameter, limit=WHOLE)
    assert result.unswept == []
    assert result.total == len(result.flips) and result.next_offset is None


def _cut_sweeps() -> list[tuple[str, str]]:
    """The sweeps with two flip values or more, between which a range can be cut."""
    return [
        (name, parameter)
        for name, parameter in _sweeps()
        if len(_values(sensitivity(_inputs()[name], parameter, limit=WHOLE))) >= 2
    ]


def test_the_fixtures_hold_ranges_cut_short_of_a_change_on_either_side() -> None:
    """The control for the test below: some cut leaves a change below its range and some above."""
    sides = set()
    for name, parameter in _cut_sweeps():
        held = _inputs()[name]
        values = _values(sensitivity(held, parameter, limit=WHOLE))
        middle = (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2
        for given in ({"range_low": middle}, {"range_high": middle}):
            sides |= {unswept.side for unswept in sensitivity(held, parameter, limit=WHOLE, **given).unswept}  # type: ignore[arg-type]
    assert sides == {"below_range_low", "above_range_high"}


@pytest.mark.parametrize(("name", "parameter"), _cut_sweeps(), ids=lambda value: str(value))
def test_a_range_stopping_short_of_a_change_names_that_side_and_its_nearest_flip(name: str, parameter: str) -> None:
    held = _inputs()[name]
    whole = sensitivity(held, parameter, limit=WHOLE)
    values = _values(whole)
    middle = (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2
    moving = [flip for flip in whole.flips if flip.call_changes or flip.picks_change]
    for given, side in (({"range_low": middle}, "below_range_low"), ({"range_high": middle}, "above_range_high")):
        if parameter.partition(":")[0] in ("margin", "floor") and middle < 0:
            continue
        ranged = sensitivity(held, parameter, limit=WHOLE, **given)  # type: ignore[arg-type]
        past = [
            flip
            for flip in moving
            if (float(flip.at.value) < middle if side == "below_range_low" else float(flip.at.value) > middle)  # type: ignore[arg-type]
        ]
        named = [unswept for unswept in ranged.unswept if unswept.side == side]
        assert [unswept.side for unswept in ranged.unswept] == ([side] if past else []), (name, parameter, given)
        assert any("unswept" in note for note in ranged.notes) == bool(past)
        if not past:
            continue
        (unswept,) = named
        nearest = (
            max(past, key=lambda flip: float(flip.at.value))
            if side == "below_range_low"
            else min(  # type: ignore[arg-type]
                past,
                key=lambda flip: float(flip.at.value),  # type: ignore[arg-type]
            )
        )
        assert unswept.nearest.at.value == nearest.at.value
        assert sorted(figure.key for figure in unswept.past) == sorted(flip.at.key for flip in past)
        assert not unswept.more_past and unswept.past[0].value == unswept.nearest.at.value
        assert resolve(unswept.nearest.at.key, held).value == unswept.nearest.at.value


def _past_end(value: float, end: float, side: str, boundary: str) -> bool:
    """Whether a change at a flip lies past the range's end: beyond it, or on it where the end's value belongs to the
    piece inside the range, read from the sweep's own boundary."""
    if side == "below_range_low":
        return value < end or (value == end and boundary == "above")
    return value > end or (value == end and boundary == "below")


def _at_flip_cuts() -> list[tuple[str, str, float, str]]:
    """Every sweep with a flip, and each value it flips at, cut there at either end."""
    cuts = []
    for name, parameter in _sweeps():
        for value in _values(sensitivity(_inputs()[name], parameter, limit=WHOLE)):
            cuts += [(name, parameter, value, "range_low"), (name, parameter, value, "range_high")]
    return cuts


def test_the_fixtures_hold_ranges_cut_at_a_flip_whose_change_lies_just_past_on_both_boundary_kinds() -> None:
    """The control for the test below: a range cut exactly at a flip leaves the change just past it unread, at the
    high end on a line whose boundary value belongs below and at the low end on one whose boundary value belongs
    above. Both occur, so the strict reading of an end (the W2c miss) fails the test below."""
    seen = set()
    for name, parameter, value, end in _at_flip_cuts():
        held = _inputs()[name]
        whole = sensitivity(held, parameter, limit=WHOLE)
        side = "below_range_low" if end == "range_low" else "above_range_high"
        moving = any(
            (flip.call_changes or flip.picks_change) and float(flip.at.value) == value  # type: ignore[arg-type]
            for flip in whole.flips
        )
        if moving and _past_end(value, value, side, whole.boundary_belongs_to):
            seen.add((side, whole.boundary_belongs_to))
    assert seen == {("above_range_high", "below"), ("below_range_low", "above")}


@pytest.mark.parametrize(("name", "parameter", "value", "end"), _at_flip_cuts(), ids=lambda value: str(value))
def test_a_range_cut_exactly_at_a_flip_names_the_change_just_past_its_end(
    name: str, parameter: str, value: float, end: str
) -> None:
    held = _inputs()[name]
    whole = sensitivity(held, parameter, limit=WHOLE)
    if parameter.partition(":")[0] in ("margin", "floor") and value < 0:
        return
    ranged = sensitivity(held, parameter, limit=WHOLE, **{end: value})  # type: ignore[arg-type]
    moving = [flip for flip in whole.flips if flip.call_changes or flip.picks_change]
    side = "below_range_low" if end == "range_low" else "above_range_high"
    past = [flip for flip in moving if _past_end(float(flip.at.value), value, side, whole.boundary_belongs_to)]  # type: ignore[arg-type]
    assert [unswept.side for unswept in ranged.unswept] == ([side] if past else []), (name, parameter, end, value)
    if past:
        (unswept,) = ranged.unswept
        values = [float(flip.at.value) for flip in past]  # type: ignore[arg-type]
        assert float(unswept.nearest.at.value) == (max(values) if side == "below_range_low" else min(values))  # type: ignore[arg-type]
        assert sorted(figure.key for figure in unswept.past) == sorted(flip.at.key for flip in past)
        assert unswept.nearest.call_changes or unswept.nearest.picks_change
    read = {(flip.at.key, flip.arm): (flip.call_changes, flip.picks_change) for flip in whole.flips}
    for flip in ranged.flips:
        assert (flip.call_changes, flip.picks_change) == read[(flip.at.key, flip.arm)], (flip.at.key, end, value)


def _moved(held: ReportInput, criterion_id: str, threshold: float) -> ReportInput:
    criteria = [
        criterion.model_copy(update={"threshold": threshold}) if criterion.id == criterion_id else criterion
        for criterion in held.criteria
    ]
    return held.model_copy(update={"criteria": criteria})


def test_r2s_pass_rate_line_swept_up_from_its_declared_value_says_the_call_changes_below_it() -> None:
    """The W2 regression prototype's r2 miss, from the input: the analyst swept the pass-rate gate from its declared
    value up; below it, cheaper configurations take the call. The sweep names that side, its nearest flip, and the call
    there, which code's decision with the line moved just past it reads independently."""
    held = fixture_inputs()["r2"]
    gate = next(criterion for criterion in held.criteria if criterion.measure == "pass_rate")
    assert gate.direction == "higher_is_better"
    result = sensitivity(held, f"threshold:{gate.id}", range_low=float(gate.threshold), limit=WHOLE)
    (unswept,) = result.unswept
    assert unswept.side == "below_range_low" and unswept.loosens
    nearest = float(unswept.nearest.at.value)  # type: ignore[arg-type]
    assert nearest < float(gate.threshold)
    flips = sorted(
        {float(flip.at.value) for flip in sensitivity(held, f"threshold:{gate.id}", limit=WHOLE).flips}  # type: ignore[arg-type]
    )
    below = max([value for value in flips if value < nearest], default=nearest - 1.0)
    past = decision_at(_moved(held, gate.id, (below + nearest) / 2))
    assert (unswept.call_past.modes, unswept.call_past.dominant) == (
        sorted(past.modes),
        [f"arm:{arm}" for arm in past.dominant],
    )
    assert (unswept.call_past.modes, unswept.call_past.dominant) != (
        result.declared_call.modes,
        result.declared_call.dominant,
    )


def test_a_margin_sweep_says_which_candidates_its_aspect_holds_no_reading_of() -> None:
    """The r4 finding: a margin sweep over an aspect with no reading reported a tradeoff at every margin and no reason.
    It names the candidates code's rule never reads against the margin; where every candidate has a reading, it says
    nothing of the kind."""
    for name, held in _inputs().items():
        control = next((arm.id for arm in held.arms if arm.role == "control"), None)
        for aspect in held.aspects:
            result = sensitivity(held, f"margin:{aspect.id}", limit=WHOLE)
            read = {reading.arm for reading in aspect.readings}
            unread = [arm.id for arm in held.arms if control is not None and arm.id != control and arm.id not in read]
            said = [note for note in result.notes if "holds no reading of" in note]
            assert bool(said) == bool(unread), (name, aspect.id)
            if unread:
                assert all(f"arm:{arm}" in said[0] for arm in unread)
    r4 = fixture_inputs()["r4"]
    asked = next(aspect for aspect in r4.aspects if aspect.asked)
    assert not asked.readings and any(contrast.measure == asked.id for contrast in r4.contrasts)


@pytest.mark.parametrize("name", sorted(name for name, held in _inputs().items() if _parameters(held)))
def test_each_arm_a_cross_cutting_gate_keeps_out_of_every_pick_is_listed_with_its_gates(name: str) -> None:
    held = _inputs()[name]
    result = sensitivity(held, _parameters(held)[0], limit=WHOLE)
    routable = [task.id for task in held.tasks if task.routable]
    cross = [task.id for task in held.tasks if task.cross_cutting]
    expected: dict[str, tuple[str, list[str], list[str]]] = {}
    if routable and cross:
        for arm in held.arms:
            failing = [task for task in cross if gate_check(held, arm.id, task=task).status == "fail"]
            if failing:
                gates = [f.criterion for f in gate_findings(held, arm.id, task=failing[0]) if f.status == "fail"]
                passes = [task for task in routable if gate_check(held, arm.id, task=task).status != "fail"]
                expected[f"arm:{arm.id}"] = (failing[0], gates, passes)
    listed = {row.arm: (row.task, [gate.criterion for gate in row.gates], row.passes_tasks) for row in result.barred}
    assert listed == expected
    picks = routing_picks(held, [arm.id for arm in held.arms])
    assert not {f"arm:{arm}" for arm in picks.values()} & set(listed), "a barred arm is never a pick"
    for row in result.barred:
        for gate in row.gates:
            assert gate.status == "fail" and gate.place == row.task
            for figure in gate.figures:
                assert resolve(figure.key, held).value == figure.value
    assert any("hold back" in note for note in result.notes) == bool(result.barred)


def test_r1s_per_task_picks_say_a_cross_cutting_gate_holds_back_an_arm_passing_a_task() -> None:
    """The r1 finding: the picks dropped an arm that passes a task's own gates, held back by the cross-cutting gate,
    without saying why."""
    held = fixture_inputs()["r1"]
    result = sensitivity(held, _parameters(held)[0], limit=WHOLE)
    assert any(row.passes_tasks for row in result.barred)
