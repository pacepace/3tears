"""A gate with no figure for an option at a place it reads is unknown there, never a fail.

Contracts held here (``report_margins``' unknown places and undecided options, ``report_gates``' findings, the count
keys, and the analyst's dominance, sensitivity, settle and weakest-margin refusal), on every fixture input (r1 to r5,
the misfits, Kairo round 2, s1 to s3), with expectations read off the input's own tables:

- **a finding with no figure says unknown**, and the gate check still does not pass the option there (a figure the
  input does not hold is not shown passed), so the page and the planner's survivors read as before;
- **dominance** lists as undecided exactly the options no gate figure fails, no breached guardrail removes, and some
  gate has no figure for; it lists a removed option's failing figures as its gates and its unknown places apart; an
  undecided option's figure over every case that misses a line read in each task is handed over as an implied fact,
  labelled inferred and never a verdict;
- **the counts** tell a miss from an unknown: ``count.failing`` counts figures that miss, ``count.unknown`` places with
  none, ``count.failed`` gates missed on a figure, ``count.undecided`` the undecided options;
- **a sweep and a refusal** say so: a threshold sweep names the options with no figure where it reads, settle refuses a
  criterion with no figure for the arm as unknown there, and the weakest-margin refusal names the undecided options.

The control is the r2 variant whose gates are read in each task while latency is recorded only over every task: every
p95 place is unknown, so the options clearing the pass rate in every task are undecided rather than failed.
"""

from __future__ import annotations

import math

import pytest

from packages.evals.tests.analyst_support import KAIRO, budget_inputs, fixture_inputs, fixture_snapshots
from threetears.evals.analysis.analyst.analyses import dominance, sensitivity
from threetears.evals.analysis.analyst.settle import settle_cases
from threetears.evals.analysis.analyst.snapshot import frozen_input
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.figure_keys import resolve
from threetears.evals.analysis.report_decision import surviving_options
from threetears.evals.analysis.report_gates import gate_check, gate_findings
from threetears.evals.analysis.report_input import Criterion, ReportInput
from threetears.evals.kernel.errors import RefusedInput

#: The reference input whose gates are read in each task, latency recorded only over every task.
EACH_TASK = "r2 with its gates read in each task"

#: The criterion kinds that remove an option that misses them.
GATING = ("gate", "guardrail")


def _inputs() -> dict[str, ReportInput]:
    held = dict(fixture_inputs())
    held[KAIRO] = frozen_input(fixture_snapshots("s")[KAIRO])
    held |= {name: budget_inputs()[name] for name in ("s1", "s2", "s3")}
    return held


INPUTS = _inputs()


def _values(held: ReportInput, criterion: Criterion, arm: str) -> list[float | None]:
    """The arm's figure at each place a gate reads it, off the input's tables (None: no finite figure there). Every
    gate in the fixtures reads the lead aspect, whose figures the by-arm maps hold."""
    if criterion.scope == "each_cell":
        stats = [cell.per_arm.get(arm, {}).get(criterion.measure) for cell in held.cells.values()]
    elif criterion.scope == "each_task" or criterion.task is not None:
        stats = [
            task.per_arm.get(arm, {}).get(criterion.measure) for task in held.tasks if criterion.task in (None, task.id)
        ]
    else:
        table = held.measures.get(criterion.measure) or held.judged.get(criterion.measure)
        stats = [table.per_arm.get(arm) if table is not None else None]
    values = [getattr(stat, criterion.statistic) if stat is not None else None for stat in stats]
    return [float(value) if value is not None and math.isfinite(value) else None for value in values]


def _misses(criterion: Criterion, value: float) -> bool:
    threshold = float(criterion.threshold)
    return value < threshold if criterion.direction == "higher_is_better" else value > threshold


def _undecided(held: ReportInput) -> list[str]:
    breached = {str(read.get("arm")) for read in held.guardrails if read.get("decision") == "breached"}
    gates = [criterion for criterion in held.criteria if criterion.kind in GATING]
    found = []
    for arm in held.arms:
        values = [(gate, value) for gate in gates for value in _values(held, gate, arm.id)]
        missed = any(value is not None and _misses(gate, value) for gate, value in values)
        unknown = any(value is None for _gate, value in values) or any(
            not _values(held, gate, arm.id) for gate in gates
        )
        if arm.id not in breached and not missed and unknown:
            found.append(arm.id)
    return found


def test_the_control_holds_options_undecided_on_a_gate_with_no_figure_in_any_task() -> None:
    held = INPUTS[EACH_TASK]
    undecided = _undecided(held)
    assert undecided and not surviving_options(held)
    latency = next(criterion for criterion in held.criteria if criterion.statistic == "p95")
    assert all(value is None for arm in held.arms for value in _values(held, latency, arm.id))


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_a_finding_with_no_figure_is_unknown_and_the_check_still_does_not_pass_it(name: str) -> None:
    held = INPUTS[name]
    for arm in held.arms:
        findings = gate_findings(held, arm.id)
        assert all((finding.figure is None) == (finding.status == "unknown") for finding in findings), arm.id
        if any(finding.status == "unknown" for finding in findings):
            assert gate_check(held, arm.id).status == "fail", "an option not shown to pass a gate does not pass it"


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_dominance_lists_the_undecided_apart_from_the_removed(name: str) -> None:
    held = INPUTS[name]
    result = dominance(held, limit=200)
    undecided = _undecided(held)
    assert [row.arm.removeprefix("arm:") for row in result.undecided] == undecided
    survivors = surviving_options(held)
    removed = [arm.id for arm in held.arms if arm.id not in survivors and arm.id not in undecided]
    assert [row.arm.removeprefix("arm:") for row in result.out] == removed
    for row in result.out:
        assert all(gate.status == "fail" for gate in row.gates)
        assert all(gate.status == "unknown" for gate in row.unknown)
    for row in result.undecided:
        assert row.unknown and all(gate.status == "unknown" for gate in row.unknown)
        assert all({figure.key.rsplit(".", 1)[1] for figure in gate.figures} == {"threshold"} for gate in row.unknown)
    assert any("undecided" in note for note in result.notes) == bool(undecided)


def test_an_undecided_options_figure_over_every_case_that_misses_a_line_read_in_each_task_is_an_implied_fact() -> None:
    held = INPUTS[EACH_TASK]
    result = dominance(held, limit=200)
    for row in result.undecided:
        arm = row.arm.removeprefix("arm:")
        expected = []
        for criterion in held.criteria:
            if (
                criterion.kind not in GATING
                or criterion.scope != "each_task"
                or None not in _values(held, criterion, arm)
            ):
                continue
            table = held.measures.get(criterion.measure) or held.judged[criterion.measure]
            overall = getattr(table.per_arm[arm], criterion.statistic)
            if overall is not None and _misses(criterion, float(overall)):
                expected.append((criterion.id, float(overall)))
        assert [(fact.criterion, fact.figures[-1].value) for fact in row.implied] == expected, arm
        for fact in row.implied:
            assert fact.reads_as == row.implied[0].reads_as, "every implied fact reads as the one same note"
            assert "inferred" in fact.reads_as and "never a verdict" in fact.reads_as
            assert fact.figures[0].value == float(next(c for c in held.criteria if c.id == fact.criterion).threshold)
            assert all(resolve(figure.key, held).value == figure.value for figure in fact.figures)
    assert any(row.implied for row in result.undecided), "the control: a pooled p95 over the budget implies a miss"


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_the_counts_tell_a_miss_from_an_unknown(name: str) -> None:
    held = INPUTS[name]
    gates = [criterion for criterion in held.criteria if criterion.kind in GATING]
    for criterion in held.criteria:
        for arm in held.arms:
            values = _values(held, criterion, arm.id)
            missed = sum(value is not None and _misses(criterion, value) for value in values)
            unknown = sum(value is None for value in values)
            if criterion.aspect is not None:
                continue  # read on its own aspect, held by test_criterion_aspects
            assert resolve(f"count.failing.{criterion.id}.{arm.id}", held).value == missed
            assert resolve(f"count.unknown.{criterion.id}.{arm.id}", held).value == unknown
    for arm in held.arms:
        failed = sum(
            any(value is not None and _misses(gate, value) for value in _values(held, gate, arm.id)) for gate in gates
        )
        assert resolve(f"count.failed.{arm.id}", held).value == failed, arm.id
    assert resolve("count.undecided", held).value == len(_undecided(held))


def test_a_sweep_names_the_options_with_no_figure_where_the_criterion_reads() -> None:
    held = INPUTS[EACH_TASK]
    latency = next(criterion for criterion in held.criteria if criterion.statistic == "p95")
    result = sensitivity(held, f"threshold:{latency.id}")
    assert any("unknown" in note for note in result.notes)
    assert any("undecided" in note for note in result.notes)
    assert result.flips == [], "no figure to flip at"


def test_settle_refuses_a_criterion_with_no_figure_for_the_arm_as_unknown_there() -> None:
    held = INPUTS[EACH_TASK]
    latency = next(criterion for criterion in held.criteria if criterion.statistic == "p95")
    with pytest.raises(RefusedInput) as refused:
        settle_cases(held, [held.arms[0].id], criterion=latency.id)
    assert "unknown there, not a miss" in str(refused.value)


def test_the_weakest_margin_refusal_names_the_undecided_options() -> None:
    held = INPUTS[EACH_TASK]
    with pytest.raises(UnresolvableReference) as refused:
        resolve("weakest.value", held)
    assert "undecided" in str(refused.value)
    assert all(f"`{arm}`" in str(refused.value) for arm in _undecided(held)[:5])
