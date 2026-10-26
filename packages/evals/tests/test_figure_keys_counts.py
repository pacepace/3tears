"""Count keys: what a sentence counts ("four of five tasks", "the only config of nine") is a figure key's.

Contracts held here (``threetears.evals.analysis.report.figure_keys``' count families and its ``words`` filter, and
``report_margins``' ``criterion_misses``, ``criterion_passes`` and ``surviving_arms``):

- **each count** reads, on every fixture input (r1 to r5, the misfits, Kairo round 2, s1 to s3), what is counted
  here off the input's own fields: its arms, aspects, tasks, strata, cells, configs, gates and criteria; the options
  surviving every gate (the decision's own survivors), the options clearing each gate (the gate check's own findings),
  the places a criterion reads and the places an arm misses there (each figure against its threshold, a place with no
  figure missed), the gates an arm misses, a factor's levels;
- **the words filter** spells a count from zero to twenty as its word and a larger one in grouped digits, and is
  refused on a figure that is no count;
- **a count is no typed number**: prose writing one through a key holds no digit the D3 check finds, and the grammar
  teaches the family with its examples rendered.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from packages.evals.tests.analyst_support import KAIRO, budget_inputs, fixture_inputs, fixture_snapshots
from threetears.evals.analysis.analyst.snapshot import frozen_input
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.figure_keys import (
    COUNT_WORDS,
    figure_key_grammar,
    format_figure,
    literal_numbers,
    render,
    resolve,
)
from threetears.evals.analysis.report_decision import surviving_options
from threetears.evals.analysis.report_gates import gate_findings
from threetears.evals.analysis.report_input import Criterion, ReportInput, Stat
from threetears.evals.analysis.report_margins import surviving_arms


def _inputs() -> dict[str, ReportInput]:
    held = dict(fixture_inputs())
    held[KAIRO] = frozen_input(fixture_snapshots("s")[KAIRO])
    held |= {name: budget_inputs()[name] for name in ("s1", "s2", "s3")}
    return held


#: Every input the counts are held to, by name.
INPUTS = _inputs()

#: The criterion kinds a gate count counts.
GATING = ("gate", "guardrail")


def _count(key: str, held: ReportInput) -> float:
    figure = resolve(key, held)
    assert figure.kind == "number" and figure.scale == "count", key
    assert isinstance(figure.value, float)
    return figure.value


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_the_campaigns_counts_read_what_it_holds(name: str) -> None:
    held = INPUTS[name]
    expected = {
        "arms": len(held.arms),
        "aspects": len(held.aspects),
        "tasks": len(held.tasks),
        "strata": len(held.strata),
        "cells": len(held.cells),
        "configs": len(held.configs),
        "gates": sum(criterion.kind in GATING for criterion in held.criteria),
        "criteria": len(held.criteria),
        "surviving": len(surviving_options(held)),
    }
    assert {what: _count(f"count.{what}", held) for what in expected} == expected
    assert surviving_arms(held) == surviving_options(held), "the margins' survivors are the decision's"
    for factor in held.factors:
        assert _count(f"count.levels.{factor.id}", held) == len(factor.levels)


def _stat(held: ReportInput, criterion: Criterion, arm: str, task: str | None, cell: str | None) -> Stat | None:
    """The arm's figure a criterion reads at one place, off the input's own tables."""
    if cell is not None:
        return held.cells[cell].per_arm.get(arm, {}).get(criterion.measure)
    if task is not None:
        return next(t for t in held.tasks if t.id == task).per_arm.get(arm, {}).get(criterion.measure)
    table = held.measures.get(criterion.measure) or held.judged.get(criterion.measure)
    return table.per_arm.get(arm) if table is not None else None


def _places(held: ReportInput, criterion: Criterion) -> list[tuple[str | None, str | None]]:
    """Where a criterion reads, off its declared scope: every cell, its one task or every task, else overall."""
    if criterion.scope == "each_cell":
        return [(None, cell) for cell in held.cells]
    if criterion.scope == "each_task" or criterion.task is not None:
        return [(task.id, None) for task in held.tasks if criterion.task in (None, task.id)]
    return [(None, None)]


def _misses(held: ReportInput, criterion: Criterion, arm: str) -> int:
    """The places an arm misses a criterion at: no finite figure there, or one on the wrong side of the threshold."""
    missed = 0
    for task, cell in _places(held, criterion):
        stat = _stat(held, criterion, arm, task, cell)
        value = getattr(stat, criterion.statistic) if stat is not None else None
        if value is None or value != value or abs(value) == float("inf"):
            missed += 1
        elif criterion.direction == "higher_is_better":
            missed += value < criterion.threshold
        else:
            missed += value > criterion.threshold
    return missed


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_each_criterions_places_and_each_arms_misses_read_its_figures_against_the_threshold(name: str) -> None:
    held = INPUTS[name]
    for criterion in held.criteria:
        assert _count(f"count.places.{criterion.id}", held) == len(_places(held, criterion)), criterion.id
        for arm in held.arms:
            key = f"count.failing.{criterion.id}.{arm.id}"
            assert _count(key, held) == _misses(held, criterion, arm.id), key


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_the_options_clearing_each_gate_and_the_gates_each_arm_misses_agree_with_the_gate_check(name: str) -> None:
    held = INPUTS[name]
    gates = [criterion for criterion in held.criteria if criterion.kind in GATING]
    failed: dict[str, set[str]] = {
        arm.id: {finding.criterion for finding in gate_findings(held, arm.id) if finding.status == "fail"}
        for arm in held.arms
    }
    for gate in gates:
        clearing = sum(gate.id not in failed[arm.id] for arm in held.arms)
        assert _count(f"count.passing.{gate.id}", held) == clearing, gate.id
    for arm in held.arms:
        assert _count(f"count.failed.{arm.id}", held) == len(failed[arm.id]), arm.id
    assert resolve(f"count.passing.{gates[0].id}", held).derived if gates else True


def test_some_fixture_has_an_arm_missing_a_gate_in_some_but_not_all_of_its_tasks() -> None:
    """The control for the counts above: the case an editor flagged ("misses it in four of five tasks") exists."""
    found = [
        (name, criterion.id, arm.id)
        for name, held in INPUTS.items()
        for criterion in held.criteria
        if len(_places(held, criterion)) > 1
        for arm in held.arms
        if 0 < _misses(held, criterion, arm.id) < len(_places(held, criterion))
    ]
    assert found


@pytest.mark.parametrize("count", [0, 1, 4, 9, 20, 21, 1234])
def test_the_words_filter_spells_a_count_as_its_word_and_a_large_one_in_digits(count: int) -> None:
    held = INPUTS["r1"]
    figure = resolve("count.arms", held)
    spelled = format_figure(replace(figure, value=float(count)), ("words",))
    if count < len(COUNT_WORDS):
        assert spelled == COUNT_WORDS[count]
    else:
        assert spelled == f"{count:,}"


def test_the_words_filter_is_refused_on_a_figure_that_is_no_count() -> None:
    held = INPUTS["r4"]
    arm = held.arms[0].id
    with pytest.raises(UnresolvableReference, match="cannot read as a word"):
        render(f"{{cost_per_1k.{arm}|words}}", held, {})


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_a_count_written_through_a_key_types_no_digit(name: str) -> None:
    held = INPUTS[name]
    sentence = "{count.surviving|words} of {count.arms|words} options survive; {count.gates} gates."
    rendered = render(sentence, held, {})
    assert str(len(surviving_options(held))) in rendered or COUNT_WORDS[len(surviving_options(held))] in rendered
    assert literal_numbers(sentence, held) == []


def test_the_grammar_teaches_the_counts_with_their_examples_rendered() -> None:
    grammar = figure_key_grammar()
    assert "- `count.<what>`" in grammar and "- `count.failing.<criterion>.<arm>`" in grammar
    assert '`{count.arms|words}` reads "two"' in grammar
    assert "- `words`:" in grammar
