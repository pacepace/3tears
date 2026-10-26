"""What a cost is per: every phrase a page prints a cost's count with is read from the input, never assumed.

Contracts held here (``analysis.report.cost_unit``, ``figure_keys.cost_basis``, and every place a page prints a cost):

- a money measure stated per 1,000 or per 1,000,000 of something is per what its label counts, and a page's cost
  (``cost_per_1k.<arm>``) is per what its source counts, on r1 to r5, the misfits and Kairo round 2: the figure keys,
  the gate matrix, the lattice and a plain table all print that count, and never "replies" where the label counts
  something else;
- money per one thing, and the scoreboard's reply cost, is per reply;
- where the input does not say what its cost counts, the phrase is neutral and the basis says it is unstated;
- the digit rule lets prose write the input's own count phrase whole, and still refuses a count it does not state.
"""

from __future__ import annotations

import functools
from typing import Any

import pytest

from packages.evals.tests.analyst_support import KAIRO, fixture_inputs, fixture_snapshots
from threetears.evals.analysis.analyst.blocks import block_render
from threetears.evals.analysis.analyst.snapshot import frozen_input
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.cost_unit import REPLIES, UNSTATED, per_count
from threetears.evals.analysis.report.figure_keys import cost_basis, literal_numbers, render, resolve
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.kernel.report_plan import ChartQuery

#: Every query field, empty: no filter, no limit.
_EMPTY: dict[str, Any] = {
    "aspect": "",
    "measures": [],
    "arms": [],
    "strata": [],
    "limit": 0,
    "transform": "none",
    "small_multiples_by": "none",
}

#: The money units stated per a count of something, and the count.
_COUNTED = {"usd_per_1k": 1_000, "usd_per_1m": 1_000_000}


@functools.cache
def _inputs() -> dict[str, ReportInput]:
    inputs = dict(fixture_inputs())
    inputs[KAIRO] = frozen_input(fixture_snapshots("s")[KAIRO])
    return inputs


def _costed() -> list[str]:
    """The inputs that state a cost for some arm."""
    return [name for name, held in _inputs().items() if cost_basis(held).source]


def _label_noun(held: ReportInput, key: str) -> str:
    """What a counted money measure's label says it counts: its last word, as every fixture label writes it."""
    label = held.measures[key].label
    assert label is not None
    return label.split()[-1]


@pytest.mark.parametrize("name", _costed())
def test_a_pages_cost_reads_per_what_its_source_counts(name: str) -> None:
    held = _inputs()[name]
    basis = cost_basis(held)
    measure = held.measures.get(basis.source)
    if measure is not None and measure.unit in _COUNTED:
        assert basis.noun == _label_noun(held, basis.source)
    else:
        assert basis.noun == REPLIES, "money per one thing, and the scoreboard's reply cost, is per reply"
    arm = next(arm.id for arm in held.arms if _resolves(f"cost_per_1k.{arm.id}", held))
    assert render(f"{{cost_per_1k.{arm}|usd_per_1k}}", held, {}).endswith(f" per 1,000 {basis.noun}")


@pytest.mark.parametrize("name", _costed())
def test_a_counted_measures_own_figure_reads_per_what_its_label_counts(name: str) -> None:
    held = _inputs()[name]
    for key, measure in held.measures.items():
        if measure.unit not in _COUNTED:
            continue
        arm = next((arm for arm, stat in measure.per_arm.items() if stat.mean is not None), None)
        if arm is None:
            continue  # its figures sit in slices alone, which the gate matrix and lattice tests read
        said = render(f"{{measure.{key}.{arm}.mean|usd_per_1k}}", held, {})
        assert said.endswith(per_count(_label_noun(held, key)))


def _resolves(key: str, held: ReportInput) -> bool:
    try:
        resolve(key, held)
    except UnresolvableReference:
        return False
    return True


def test_no_fixture_page_says_replies_where_its_cost_counts_something_else() -> None:
    """The prototype's bug: a cost per 1,000 conversations or calls printed per 1,000 replies, in a block's cells."""
    seen = 0
    for held in _inputs().values():
        basis = cost_basis(held)
        if not basis.source or basis.noun == REPLIES:
            continue
        for kind, source in (("gate_matrix", "tasks"), ("lattice", "configs")):
            try:
                drawn = block_render(
                    held, block_kind=kind, query=ChartQuery.model_validate({**_EMPTY, "source": source})
                )
            except RefusedInput:
                continue
            cells = " ".join(" ".join(row.cells) for row in drawn.rows)
            assert "replies" not in cells, (kind, cells[:300])
            if "$" in cells:
                assert per_count(basis.noun) in cells, (kind, cells[:300])
                seen += 1
    assert seen, "the control: some fixture draws a gate matrix or a lattice over costs counted in other than replies"


def test_a_plain_tables_cost_column_is_named_by_its_measure_and_its_cells_say_no_other_count() -> None:
    held = next(
        held
        for held in _inputs().values()
        if cost_basis(held).noun not in ("", REPLIES) and held.measures[cost_basis(held).source].per_arm
    )
    key = cost_basis(held).source
    drawn = block_render(
        held, block_kind="table", query=ChartQuery.model_validate({**_EMPTY, "source": "measures", "measures": [key]})
    )
    assert all("replies" not in cell for row in drawn.rows for cell in row.cells)


def _unstated(held: ReportInput) -> ReportInput:
    """An input whose counted cost measure's label names no count."""
    document = held.model_dump(mode="json", by_alias=True)
    document["measures"][cost_basis(held).source]["label"] = "Cost"
    return ReportInput.model_validate(document)


def test_an_unstated_count_reads_neutral_and_the_basis_says_so() -> None:
    held = next(held for held in _inputs().values() if cost_basis(held).noun not in ("", REPLIES))
    bare = _unstated(held)
    basis = cost_basis(bare)
    assert basis.source == cost_basis(held).source and not basis.stated and basis.noun == ""
    arm = next(arm.id for arm in bare.arms if _resolves(f"cost_per_1k.{arm.id}", bare))
    said = render(f"{{cost_per_1k.{arm}|usd_per_1k}}", bare, {})
    assert said.endswith(f"per 1,000 {UNSTATED}") and "replies" not in said


def test_the_digit_rule_lets_prose_write_the_inputs_own_count_and_no_other() -> None:
    held = next(held for held in _inputs().values() if cost_basis(held).noun not in ("", REPLIES))
    phrase = per_count(cost_basis(held).noun)
    assert literal_numbers(f"a cost {phrase} is up", held) == []
    assert literal_numbers("a cost per 1,000 widgets is up", held) != []
