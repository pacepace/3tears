"""Two arms settled apart on an aspect's pass rate: the reading-level figure an asked aspect is decided on.

Contracts held here (``threetears.evals.analysis.analyst.settle`` and the ``settle.passed.`` keys of
``analysis.report.figure_keys``):

- **it reaches the pass rate**: two arms with ``measure_key`` ``passed`` (the name a snapshot's contrasts give a
  trial's pass) are settled apart on each aspect whose scoreboard states both arms' trials passed over usable, one
  row per such aspect, where the W2 regression prototype's Kairo analyst was refused;
- **paired where a reading pairs them**: a candidate against the control is read on the aspect's reading (its own
  change, interval and level, over the cases each arm ran); the sizes and the wrong-side share are the arithmetic on
  those numbers, computed here from the input;
- **independent otherwise**: two candidates are read on each arm's binomial spread from its own trials, on its
  distinct cases where its scoreboard states them;
- **the count basis is stated**: each row cites each arm's usable trials and cases from the aspect's scoreboard and
  a basis naming what the spread rests on; the trials a size takes are read only where both arms ran each case alike;
- **teaching**: a two-arm call naming a measure the arms do not hold lists ``passed`` where an aspect states both
  pass rates, and never where none does.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import pytest

from packages.evals.tests.analyst_support import KAIRO, fixture_inputs, fixture_snapshots
from threetears.evals.analysis.analyst.settle import SettleRow, settle_cases
from threetears.evals.analysis.analyst.snapshot import frozen_input
from threetears.evals.analysis.report.figure_keys import passed_aspects, resolve
from threetears.evals.analysis.report_input import ReportInput, ScoreboardRow
from threetears.evals.analysis.separation import level_share
from threetears.evals.kernel.errors import RefusedInput

_NORMAL = NormalDist()

#: The pass rate's name, as a snapshot's contrasts spell a trial's pass.
PASSED = "passed"


def _kairo() -> ReportInput:
    return frozen_input(fixture_snapshots("s")[KAIRO])


def _board(held: ReportInput, aspect: str) -> dict[str, ScoreboardRow]:
    (named,) = [candidate for candidate in held.aspects if candidate.id == aspect]
    lead = next((candidate.id for candidate in held.aspects if candidate.asked), held.aspects[0].id)
    board = named.scoreboard if named.scoreboard is not None else held.scoreboard if aspect == lead else {}
    return dict(board)


def _z(level: float) -> float:
    return _NORMAL.inv_cdf(0.5 + level / 2)


def test_kairo_holds_a_pass_rate_contrast_and_no_measure_of_that_name() -> None:
    """The control for the tests below: the snapshot spells the asked aspect's pass rate `passed` in its contrasts and
    holds no measure or judged dimension of that key, so the refusal the analyst met was the only answer before."""
    held = _kairo()
    assert any(contrast.measure == PASSED for contrast in held.contrasts)
    assert PASSED not in held.measures and PASSED not in held.judged
    assert passed_aspects(held, [arm.id for arm in held.arms])


@pytest.mark.parametrize("order", ["candidate first", "control first"])
def test_a_candidate_against_the_control_is_settled_on_the_aspects_own_paired_reading(order: str) -> None:
    held = _kairo()
    control = next(arm.id for arm in held.arms if arm.role == "control")
    for candidate in (arm.id for arm in held.arms if arm.role != "control"):
        arms = [candidate, control] if order == "candidate first" else [control, candidate]
        result = settle_cases(held, arms, measure=PASSED, limit=50)
        assert "two-sided" in result.sided
        aspects = passed_aspects(held, arms)
        assert [row.where for row in result.rows] == [f"aspects.{aspect}" for aspect in aspects]
        for row, aspect in zip(result.rows, aspects, strict=True):
            (reading,) = [
                r for aspect_ in held.aspects if aspect_.id == aspect for r in aspect_.readings if r.arm == candidate
            ]
            assert reading.gain_interval is not None
            level = level_share(reading.gain_confidence)
            assert level is not None
            sign = 1.0 if arms[0] == candidate else -1.0
            low, high = reading.gain_interval
            se = (high - low) / 2 / _z(level)
            harmonic = len(reading.cases) / sum(1 / count for count in reading.cases)
            per_case = se * math.sqrt(harmonic)
            difference = sign * float(reading.change)
            assert row.distance is not None and row.distance.value == pytest.approx(difference)
            assert row.cases_needed is not None
            assert row.cases_needed.value == math.ceil((_z(level) * per_case / abs(difference)) ** 2)
            assert row.wrong_side is not None
            assert row.wrong_side.value == pytest.approx(_NORMAL.cdf(-abs(difference) / se))
            assert row.level is not None and row.level.value == f"{level * 100:g}% two-sided"
            _holds_its_count_basis(held, row, aspect, arms, basis="paired")


def test_two_candidates_are_settled_on_each_arms_binomial_spread_read_on_its_cases() -> None:
    held = _kairo()
    first, second = (arm.id for arm in held.arms if arm.role != "control")
    result = settle_cases(held, [first, second], measure=PASSED, limit=50)
    assert result.rows
    for row in result.rows:
        aspect = row.where.removeprefix("aspects.")
        board = _board(held, aspect)
        per_case, se = [], []
        for arm in (first, second):
            stated = board[arm]
            assert stated.passed is not None and stated.usable is not None and stated.cases is not None
            share = stated.passed / stated.usable
            per_case.append(math.sqrt(share * (1 - share)) * 100)
            se.append(per_case[-1] / math.sqrt(stated.cases))
        rates = [board[arm].passed / board[arm].usable for arm in (first, second)]  # type: ignore[operator]
        difference = 100 * (rates[0] - rates[1])
        assert [figure.value for figure in row.estimate] == pytest.approx(rates)
        assert row.distance is not None and row.distance.value == pytest.approx(difference)
        assert row.cases_needed is not None
        assert row.cases_needed.value == math.ceil((_z(0.95) * math.hypot(*per_case) / abs(difference)) ** 2)
        assert row.wrong_side is not None
        assert row.wrong_side.value == pytest.approx(_NORMAL.cdf(-abs(difference) / math.hypot(*se)))
        _holds_its_count_basis(held, row, aspect, [first, second], basis="independent")


def _holds_its_count_basis(held: ReportInput, row: SettleRow, aspect: str, arms: list[str], *, basis: str) -> None:
    """A row cites each arm's usable trials and cases on the aspect, its basis, and trials only where both arms ran
    each case alike."""
    board = _board(held, aspect)
    assert [figure.key for figure in row.counts] == [f"aspects.{aspect}.scoreboard.{arm}.usable" for arm in arms]
    assert [figure.value for figure in row.counts] == [board[arm].usable for arm in arms]
    assert [figure.key for figure in row.cases] == [
        f"aspects.{aspect}.scoreboard.{arm}.cases" for arm in arms if board[arm].cases is not None
    ]
    (stated,) = row.basis
    assert isinstance(stated.value, str) and stated.value.startswith(f"{basis}:")
    repeats = {board[arm].usable / board[arm].cases for arm in arms if board[arm].cases}  # type: ignore[operator]
    if len(repeats) == 1 and all(board[arm].cases for arm in arms):
        (each,) = repeats
        assert row.trials_needed is not None and row.cases_needed is not None
        assert row.trials_needed.value == math.ceil(float(row.cases_needed.value) * each - 1e-9)
    else:
        assert row.trials_needed is None and row.repeats == []
    for figure in [*row.estimate, *row.counts, *row.cases, *row.basis]:
        value = resolve(figure.key, held).value
        assert figure.value == (list(value) if isinstance(value, tuple) else value)


def test_a_two_arm_call_naming_no_held_measure_teaches_the_pass_rate_where_an_aspect_states_it() -> None:
    held = _kairo()
    first, second = (arm.id for arm in held.arms[:2])
    with pytest.raises(RefusedInput) as refused:
        settle_cases(held, [first, second], measure="no such measure")
    assert PASSED in refused.value.valid_values
    without = fixture_inputs()["r4"]
    assert not passed_aspects(without, [arm.id for arm in without.arms])
    pair = [arm.id for arm in without.arms[:2]]
    with pytest.raises(RefusedInput) as unheld:
        settle_cases(without, pair, measure=PASSED)
    assert PASSED not in unheld.value.valid_values
