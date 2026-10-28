"""Keys for a confusion cell, two arms' costs apart, and the pairs of arms a sentence counts.

Contracts held here (``threetears.evals.analysis.report.figure_keys``):

- **a confusion cell** (``confusion.<axis>.<arm>.<expected>.<predicted>``) reads the input's own matrix, a percent of
  its row where the axis states row percents, a count where it counts observations;
- **two arms' costs apart** (``cost.gap.<first>.<second>.<field>``) read the page's own cost per 1,000 (``cost_per_1k``):
  the first's minus the second's, the same per month at the declared volume (refused as unstated where none is
  declared), and the first's as a multiple of the second's; every one derived;
- **pairs of arms** (``count.pairs``, ``count.arm_pairs``) count the pairs the input's pair readings read apart, and
  the pairs of arms there are;
- **a cost some trials did not report** (``cost.unreported.<arm>.<field>``) counts them, and reads the page's cost with
  each counted free and at the arm's own 95th percentile, refused where the cost is not read off one measure's
  observations, one per trial;
- a cost multiple keeps its unit under every filter, and a price keeps the figures a reader's multiplication needs;
- the grammar teaches each family, and a sentence citing them types no digit.

Every expectation is read off the input, on every fixture (r1 to r5, the misfits, s1 to s3, and Kairo round 2 for a
cost some trials did not report).
"""

from __future__ import annotations

import math
from itertools import permutations

import pytest

from packages.evals.tests.analyst_support import KAIRO, budget_inputs, fixture_inputs, fixture_snapshots
from threetears.evals.analysis.analyst.snapshot import frozen_input
from threetears.evals.analysis.report.cost_unit import per_count
from threetears.evals.analysis.errors import UnresolvableReference, UnstatedFigure
from threetears.evals.kernel.numbers import spell_dollars, spell_multiple
from threetears.evals.analysis.report.figure_keys import (
    MULTIPLE_UNIT,
    cost_basis,
    figure_filters,
    figure_key_grammar,
    format_figure,
    literal_numbers,
    render,
    resolve,
)
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
def test_a_multiple_keeps_its_unit_under_every_filter_and_rounds_to_what_a_ratio_supports(name: str) -> None:
    """r3's report printed ``{cost.gap.<a>.<b>.multiple|int}`` as "21": the filter dropped the "times" that makes it
    a multiple. Every filter a multiple takes keeps it, and its own spelling never claims a decimal past ten."""
    held = INPUTS[name]
    costs = _costs(held)
    pairs = [(first, second) for first, second in permutations(costs, 2) if costs[second]][:12]
    for first, second in pairs:
        key = f"cost.gap.{first}.{second}.multiple"
        figure = resolve(key, held)
        ratio = costs[first] / costs[second]
        taken = figure_filters(figure)
        assert "int" in taken, key
        assert format_figure(figure) == f"{spell_multiple(ratio)} {MULTIPLE_UNIT}"
        assert format_figure(figure, ("int",)) == f"{round(ratio):,} {MULTIPLE_UNIT}"
        for one in taken:
            assert format_figure(figure, (one,)).endswith(f" {MULTIPLE_UNIT}"), (key, one)
        digits = format_figure(figure).removesuffix(f" {MULTIPLE_UNIT}")
        assert "." not in digits if ratio >= 10 else len(digits.split(".")[1]) <= 2, (key, digits)
    assert pairs or len(costs) < 2, "the control: two arms with a cost"


def test_a_multiple_reads_two_figures_whole_from_ten() -> None:
    assert [spell_multiple(ratio) for ratio in (21.05, 9.96, 2.43, 1.04, 0.995, 0.48, 0.0483, 1240.4)] == [
        "21",
        "10",
        "2.4",
        "1.0",
        "1.0",
        "0.48",
        "0.048",
        "1,240",
    ]


def _price(text: str) -> float:
    """The dollars a spelled price reads, as a reader would take them off the page: ``$8/1000`` is $0.008 (#993)."""
    amount = text.split(" ", 1)[0].replace("$", "").replace(",", "")
    digits, mark, _ = amount.partition("/1000")
    return float(digits) / (1000 if mark else 1)


#: The inputs declaring a monthly volume, which a monthly cost is read at.
VOLUMED = [name for name in sorted(INPUTS) if INPUTS[name].meta.volume_per_month is not None]


@pytest.mark.parametrize("name", VOLUMED)
def test_a_price_per_1000_times_the_volume_reaches_the_monthly_figure_beside_it(name: str) -> None:
    """r2's page printed "$0.30 per 1,000 calls" beside "$1,208 a month" at 4,000,000 calls: $0.30 x 4,000 is
    $1,200. A price keeps three significant figures, so the product is within half a percent of the month."""
    held = INPUTS[name]
    volume = held.meta.volume_per_month
    assert volume is not None
    costs = _costs(held)
    checked = 0
    for arm in costs:
        price = _price(render(f"{{cost_per_1k.{arm}|usd_per_1k}}", held, {}))
        month = resolve(f"cost_per_month.{arm}", held).value
        assert isinstance(month, float)
        assert price * float(volume) / 1000 == pytest.approx(month, rel=0.005), arm
        checked += 1
    assert checked, "the control: a declared volume and a cost"


#: Every input a cost some trials did not report is read on: the fixtures and Kairo round 2, whose arms each leave
#: some trials' cost unreported.
UNREPORTED = INPUTS | {KAIRO: frozen_input(fixture_snapshots("s")[KAIRO])}

#: What a dollar measure's mean is multiplied by to read per 1,000 of what it counts.
_PER_THOUSAND = {"usd": 1000.0, "usd_per_1k": 1.0, "usd_per_1m": 0.001}


@pytest.mark.parametrize("name", sorted(UNREPORTED))
def test_a_cost_some_trials_did_not_report_reads_free_and_at_its_p95_or_says_why_not(name: str) -> None:
    """Kairo's analyst needed to show the trials with no cost could not reverse the arms' order on cost: their cost
    counted free (the least the mean can be) and at each arm's own 95th percentile."""
    held = UNREPORTED[name]
    measure = held.measures.get(cost_basis(held).source)
    for arm in _costs(held):
        key = f"cost.unreported.{arm}"
        stat = measure.per_arm.get(arm) if measure is not None else None
        row = held.scoreboard.get(arm)
        ran = row.ran if row is not None else None
        if measure is None or stat is None or stat.mean is None or stat.n is None or ran is None or stat.n > ran:
            for field in ("missing", "free", "at_p95"):
                with pytest.raises(UnstatedFigure):
                    resolve(f"{key}.{field}", held)
            continue
        factor = _PER_THOUSAND[measure.unit or ""]
        missing = resolve(f"{key}.missing", held)
        free = resolve(f"{key}.free", held)
        assert missing.value == ran - stat.n and missing.scale == "count"
        assert free.value == pytest.approx(stat.mean * stat.n / ran * factor)
        assert free.derived and free.scale == "usd_per_1k"
        cost = resolve(f"cost_per_1k.{arm}", held).value
        assert isinstance(free.value, float) and isinstance(cost, float) and free.value <= cost
        if stat.p95 is None:
            with pytest.raises(UnstatedFigure):
                resolve(f"{key}.at_p95", held)
        else:
            filled = resolve(f"{key}.at_p95", held).value
            assert filled == pytest.approx((stat.mean * stat.n + stat.p95 * (ran - stat.n)) / ran * factor)
        assert render(f"{{{key}.free|usd_per_1k}}", held, {}).endswith(per_count(cost_basis(held).noun, 1_000))


def test_some_arm_leaves_trials_without_a_cost() -> None:
    """The control for the walk above: Kairo's arms each ran trials that report no cost."""
    held = UNREPORTED[KAIRO]
    assert all(resolve(f"cost.unreported.{arm}.missing", held).value > 0 for arm in _costs(held))


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
        "cost.unreported.<arm>.<field>",
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
