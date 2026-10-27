"""Keys for a confusion cell, two arms' costs apart, and the pairs of arms a sentence counts.

Contracts held here (``threetears.evals.analysis.report.figure_keys``):

- **a confusion cell** (``confusion.<axis>.<arm>.<expected>.<predicted>``) reads the input's own matrix, a percent of
  its row where the axis states row percents, a count where it counts observations;
- **two arms' costs apart** (``cost.gap.<first>.<second>.<field>``) read the page's own cost per 1,000 (``cost_per_1k``):
  the first's minus the second's, the same per month at the declared volume (refused as unstated where none is
  declared), and the first's as a multiple of the second's; every one derived;
- **pairs of arms** (``count.pairs``, ``count.arm_pairs``) count the pairs the input's pair readings read apart, and
  the pairs of arms there are;
- the grammar teaches each family, and a sentence citing them types no digit.

Every expectation is read off the input, on every fixture (r1 to r5, the misfits, s1 to s3).
"""

from __future__ import annotations

import math
from itertools import permutations

import pytest

from packages.evals.tests.analyst_support import budget_inputs, fixture_inputs
from threetears.evals.analysis.errors import UnresolvableReference, UnstatedFigure
from threetears.evals.analysis.report.figure_keys import figure_key_grammar, literal_numbers, render, resolve
from threetears.evals.analysis.report_input import ReportInput


def _inputs() -> dict[str, ReportInput]:
    return dict(fixture_inputs()) | {name: budget_inputs()[name] for name in ("s1", "s2", "s3")}


#: Every input the keys are held to, by name.
INPUTS = _inputs()


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_every_confusion_cell_reads_the_inputs_own_matrix(name: str) -> None:
    held = INPUTS[name]
    for axis, table in held.confusion.items():
        for arm, matrix in table.per_arm.items():
            for row, expected in enumerate(table.labels):
                for column, predicted in enumerate(table.labels):
                    figure = resolve(f"confusion.{axis}.{arm}.{expected}.{predicted}", held)
                    assert figure.value == float(matrix[row][column])
                    assert figure.scale == ("percent" if table.unit == "row_percent" else "count")


def test_some_fixture_holds_a_confusion_matrix() -> None:
    assert any(held.confusion for held in INPUTS.values())


def _costs(held: ReportInput) -> dict[str, float]:
    """Each arm's cost per 1,000, as its own key reads it."""
    costs: dict[str, float] = {}
    for arm in held.arms:
        try:
            figure = resolve(f"cost_per_1k.{arm.id}", held)
        except UnresolvableReference:
            # NOSILENT: an arm the cost rule states no cost for has no cost key; that is the input's shape.
            continue
        assert isinstance(figure.value, float)
        costs[arm.id] = figure.value
    return costs


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_two_arms_costs_apart_read_the_pages_own_costs(name: str) -> None:
    held = INPUTS[name]
    costs = _costs(held)
    volume = held.meta.volume_per_month
    for first, second in list(permutations(costs, 2))[:12]:
        key = f"cost.gap.{first}.{second}"
        gap = resolve(f"{key}.per_1k", held)
        assert gap.value == pytest.approx(costs[first] - costs[second])
        assert gap.derived and gap.scale == "usd_per_1k"
        if volume is None:
            with pytest.raises(UnstatedFigure):
                resolve(f"{key}.monthly", held)
        else:
            monthly = resolve(f"{key}.monthly", held)
            assert monthly.value == pytest.approx((costs[first] - costs[second]) * float(volume) / 1000)
        if costs[second]:
            assert resolve(f"{key}.multiple", held).value == pytest.approx(costs[first] / costs[second])
        else:
            with pytest.raises(UnstatedFigure):
                resolve(f"{key}.multiple", held)


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_the_pairs_of_arms_are_counted(name: str) -> None:
    held = INPUTS[name]
    read = {frozenset(pair.arms) for aspect in held.aspects for pair in aspect.pairs}
    assert resolve("count.pairs", held).value == len(read)
    assert resolve("count.arm_pairs", held).value == math.comb(len(held.arms), 2)


def test_the_grammar_teaches_each_family() -> None:
    grammar = figure_key_grammar()
    for form in (
        "confusion.<axis>.<arm>.<expected>.<predicted>",
        "cost.gap.<arm>.<arm>.<field>",
        "count.examples.<aspect>.<arm>",
        "count.failures.<aspect>.<arm>",
        "overlap.<aspect>.<aspect>.<field>",
    ):
        assert f"`{form}`" in grammar, form


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_a_sentence_citing_the_new_keys_types_no_digit(name: str) -> None:
    held = INPUTS[name]
    keys = ["{count.pairs|words}", "{count.arm_pairs|words}", "{count.examples|words}"]
    costs = list(_costs(held))
    if len(costs) > 1:
        keys.append(f"{{cost.gap.{costs[0]}.{costs[1]}.per_1k|usd_per_1k}}")
    for axis, table in held.confusion.items():
        arm = next(iter(table.per_arm))
        keys.append(f"{{confusion.{axis}.{arm}.{table.labels[0]}.{table.labels[1]}|pct}}")
    text = "The run read " + ", then ".join(keys) + "."
    assert literal_numbers(text, held) == []
    assert "{" not in render(text, held, {})
