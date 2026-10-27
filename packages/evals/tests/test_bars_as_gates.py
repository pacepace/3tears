"""The caller chooses how the declared bars are read: as the warnings the snapshot declares, or as gates.

Contracts held here (``bars_as`` on the analyst's ``dominance``, ``sensitivity`` and ``settle_cases``, and the
actions that take it), with expectations read off the input's own fields:

- **by default the bars warn**: dominance's survivors are the decision's, and the result says it read the bars as
  warnings;
- **read as gates**, an option missing a bar on its own aspect is removed, its failing figure among its gates, and the
  result says the caller chose it; every figure cited still resolves to its value on the snapshot as declared;
- **a bar's threshold sweeps** either way: read as warnings, the call holds and each flip is an arm's figure on the
  bar; read as gates, the call at every flip changes exactly where the decision on the input with that bar read as a
  gate, its threshold either side of the flip, changes;
- **settle** says of each criterion row whether its criterion gates as the caller reads the bars;
- **the tool** takes ``bars_as``, warnings by default, and refuses any other reading.
"""

from __future__ import annotations

from typing import Any

import pytest

from packages.evals.tests.analyst_support import budget_inputs, fixture_inputs
from threetears.evals.actions.analyst_analyses import DominanceParams, SensitivityParams, SettleParams
from threetears.evals.analysis.analyst.analyses import dominance, sensitivity
from threetears.evals.analysis.analyst.settle import settle_cases
from threetears.evals.analysis.report.figure_keys import resolve
from threetears.evals.analysis.report_decision import decision_at, surviving_options
from threetears.evals.analysis.report_input import BarRead, Criterion, ReportInput

#: The inputs whose criteria declare bars.
BARRED = ("s2", "s3")


def _held(name: str) -> ReportInput:
    return budget_inputs()[name]


def _bar_value(held: ReportInput, criterion: Criterion, arm: str) -> float | None:
    """The value the pipeline read an arm against the bar a criterion declares at, on the bar's aspect."""
    read: BarRead | None = next(
        (
            read
            for read in held.bars
            if (read.aspect, read.measure, read.arm) == (criterion.aspect, criterion.measure, arm)
        ),
        None,
    )
    return None if read is None or read.value is None else float(read.value)


def _clears(criterion: Criterion, value: float, threshold: float | None = None) -> bool:
    line = float(criterion.threshold) if threshold is None else threshold
    return value >= line if criterion.direction == "higher_is_better" else value <= line


def _keys_resolve(node: Any, held: ReportInput) -> None:
    if isinstance(node, dict):
        if {"key", "value", "kind"} <= node.keys():
            value = resolve(node["key"], held).value
            assert node["value"] == (list(value) if isinstance(value, tuple) else value), node["key"]
            return
        for item in node.values():
            _keys_resolve(item, held)
    elif isinstance(node, list):
        for item in node:
            _keys_resolve(item, held)


@pytest.mark.parametrize("name", BARRED)
def test_by_default_the_bars_warn_and_the_survivors_are_the_decisions(name: str) -> None:
    held = _held(name)
    result = dominance(held, limit=200)
    assert result.bars_as == "warnings" and not result.notes
    assert [row.arm.removeprefix("arm:") for row in result.survivors] == surviving_options(held)


@pytest.mark.parametrize("name", BARRED)
def test_read_as_gates_an_option_missing_a_bar_on_its_own_aspect_is_removed(name: str) -> None:
    held = _held(name)
    bars = [criterion for criterion in held.criteria if criterion.kind == "bar"]
    assert bars
    result = dominance(held, limit=200, bars_as="gates")
    assert result.bars_as == "gates" and any("bars_as=gates" in note for note in result.notes)
    expected_out: dict[str, set[str]] = {}
    for arm in held.arms:
        missed = {
            bar.id for bar in bars if (value := _bar_value(held, bar, arm.id)) is not None and not _clears(bar, value)
        }
        if missed:
            expected_out[arm.id] = missed
    out = {row.arm.removeprefix("arm:"): {gate.criterion for gate in row.gates} for row in result.out}
    assert out == expected_out
    assert all(row.arm.removeprefix("arm:") not in expected_out for row in result.survivors)
    _keys_resolve(result.model_dump(mode="json"), held)


def _as_gate(held: ReportInput, criterion_id: str, threshold: float) -> ReportInput:
    """The input with every bar read as a gate and one criterion's threshold moved, built here, not by the sweep."""
    criteria = [
        criterion.model_copy(
            update={
                "kind": "gate" if criterion.kind == "bar" else criterion.kind,
                "threshold": threshold if criterion.id == criterion_id else criterion.threshold,
            }
        )
        for criterion in held.criteria
    ]
    return held.model_copy(update={"criteria": criteria})


def test_a_bars_threshold_sweeps_the_call_only_when_the_bars_are_read_as_gates() -> None:
    held = _held("s2")
    bar = next(criterion for criterion in held.criteria if criterion.kind == "bar")
    figures = sorted(value for arm in held.arms if (value := _bar_value(held, bar, arm.id)) is not None)
    warned = sensitivity(held, f"threshold:{bar.id}")
    assert warned.bars_as == "warnings" and warned.holds
    assert sorted(float(flip.at.value) for flip in warned.flips) == pytest.approx(figures)
    assert not any(flip.call_changes for flip in warned.flips)
    assert any("bars_as=gates" in note for note in warned.notes)
    gated = sensitivity(held, f"threshold:{bar.id}", bars_as="gates")
    assert gated.bars_as == "gates"
    assert sorted(float(flip.at.value) for flip in gated.flips) == pytest.approx(figures)
    for flip in gated.flips:
        value = float(flip.at.value)
        step = 1e-6 * max(1.0, abs(value))
        below = decision_at(_as_gate(held, bar.id, value - step))
        above = decision_at(_as_gate(held, bar.id, value + step))
        assert flip.call_changes == ((below.modes, below.dominant) != (above.modes, above.dominant)), value
    assert any(flip.call_changes for flip in gated.flips), "the control: read as a gate, the bar moves the call"
    _keys_resolve(gated.model_dump(mode="json"), held)


def test_settle_says_whether_each_criterion_gates_as_the_caller_reads_the_bars() -> None:
    held = _held("s3")
    arm = held.arms[0].id
    warned = settle_cases(held, [arm])
    assert warned.bars_as == "warnings" and warned.rows
    assert all(row.gates is False for row in warned.rows), "a bar read as a warning never gates"
    gated = settle_cases(held, [arm], bars_as="gates")
    assert all(row.gates is True for row in gated.rows)
    assert [row.model_dump(exclude={"gates"}) for row in gated.rows] == [
        row.model_dump(exclude={"gates"}) for row in warned.rows
    ], "the sizes are the same either way"
    gates = fixture_inputs()["r2 with one gate"]
    assert all(row.gates is True for row in settle_cases(gates, [gates.arms[0].id]).rows)


@pytest.mark.parametrize("params", [DominanceParams, SensitivityParams, SettleParams])
def test_the_tool_takes_bars_as_warnings_by_default_and_refuses_any_other_reading(params: type[Any]) -> None:
    fields = params.model_fields
    assert fields["bars_as"].default == "warnings"
    base: dict[str, Any] = {"snapshot_id": "report-snapshot:0"}
    if params is SensitivityParams:
        base["parameter"] = "margin:x"
    if params is SettleParams:
        base["arm_ids"] = ["a"]
    assert params.model_validate({**base, "bars_as": "gates"}).bars_as == "gates"
    with pytest.raises(ValueError, match="bars_as"):
        params.model_validate({**base, "bars_as": "floors"})
