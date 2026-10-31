"""A criterion declared on one aspect reads that aspect's figures, in every reader of criteria.

Contracts held here (``report_input.Criterion.aspect``, the builder that sets it, and ``report_margins``' one figure
lookup every criterion reader goes through: the gate check, the margins, the criteria and count keys, the analyst's
dominance, sensitivity and settle, and a block's criterion line):

- **the builder** names on each bar criterion the aspect its bar is declared on, the aspect the pipeline read the bar
  on (``bars``);
- **a criterion on an aspect reads that aspect's figure**, the lead aspect's from the by-arm maps and any other's from
  its ``by_aspect``, so on every shape campaign each bar criterion's figure for an arm is the value the pipeline read
  the bar at; its ``criteria.`` and ``count.failing.`` keys, the analyst's dominance warnings and settle rows read the
  same figure;
- **a reference input tagged on another aspect**: a criterion copied onto a non-lead aspect, whose figures there are
  the lead's mirrored across the threshold, reads the mirrored figures in the gate check, the keys and dominance, while
  the untagged one reads the lead's as before;
- **a chart narrowed to one aspect** draws that aspect's criterion line, where criteria on two aspects read the same
  measure;
- **an input whose criteria name no aspect** dumps as before: the field is left out of the document, so a stored
  input's digest does not move.
"""

from __future__ import annotations

import pytest

from packages.evals.tests.analyst_support import budget_inputs, criterion_on_another_aspect, fixture_inputs
from threetears.evals.analysis.analyst.analyses import dominance
from threetears.evals.analysis.analyst.blocks import block_render
from threetears.evals.analysis.analyst.settle import settle_cases
from threetears.evals.analysis.report.figure_keys import resolve
from threetears.evals.analysis.report_gates import gate_check
from threetears.evals.analysis.report_input import BarRead, Criterion, ReportInput
from threetears.evals.analysis.report_margins import aspect_board, aspect_of, criterion_figures
from threetears.evals.kernel.report_plan import ChartQuery

#: The shape campaigns built from their bundles, the ones whose criteria the builder declares.
SHAPES = ("s1", "s2", "s3")

#: Every query field, empty: no filter, no limit.
_EMPTY = {
    "aspect": "",
    "measures": [],
    "arms": [],
    "strata": [],
    "limit": 0,
    "transform": "none",
    "small_multiples_by": "none",
}


def _bar_reads(held: ReportInput, criterion: Criterion) -> list[BarRead]:
    """The pipeline's own reads of the bar a criterion declares: on its aspect and measure, one per arm."""
    return [read for read in held.bars if (read.aspect, read.measure) == (criterion.aspect, criterion.measure)]


def _misses(value: float, threshold: float, direction: str) -> bool:
    return value < threshold if direction == "higher_is_better" else value > threshold


def _bars(name: str) -> list[Criterion]:
    return [criterion for criterion in budget_inputs()[name].criteria if criterion.kind == "bar"]


def test_the_shapes_hold_bars_on_more_than_one_aspect_reading_one_measure() -> None:
    """The control: s3 declares a bar on each of two aspects over one measure, the case every reader got wrong."""
    held = budget_inputs()["s3"]
    measures = [criterion.measure for criterion in held.criteria]
    assert len(set(measures)) < len(measures) and len({criterion.aspect for criterion in held.criteria}) > 1


@pytest.mark.parametrize("name", SHAPES)
def test_each_bar_criterion_names_the_aspect_its_bar_is_declared_on(name: str) -> None:
    held = budget_inputs()[name]
    declared = {(criterion.aspect, criterion.measure) for criterion in held.criteria if criterion.kind == "bar"}
    assert declared == {(read.aspect, read.measure) for read in held.bars}
    assert all(criterion.aspect in {aspect.id for aspect in held.aspects} for criterion in held.criteria)


@pytest.mark.parametrize("name", SHAPES)
def test_a_bar_criterion_reads_each_arm_at_the_value_the_pipeline_read_its_bar_at(name: str) -> None:
    held = budget_inputs()[name]
    for criterion in _bars(name):
        reads = _bar_reads(held, criterion)
        assert reads, criterion.id
        for read in reads:
            if read.value is None:
                continue
            (figure,) = criterion_figures(held, criterion, read.arm)
            assert figure.value == pytest.approx(read.value), (criterion.id, read.arm)
            assert resolve(f"criteria.{criterion.id}.{read.arm}.value", held).value == pytest.approx(read.value)
            missed = _misses(read.value, read.bar, read.direction)
            assert resolve(f"count.failing.{criterion.id}.{read.arm}", held).value == float(missed)


@pytest.mark.parametrize("name", SHAPES)
def test_dominance_warns_of_each_bar_a_survivor_misses_on_the_bars_own_aspect(name: str) -> None:
    held = budget_inputs()[name]
    result = dominance(held, limit=200)
    for row in result.survivors:
        arm = row.arm.removeprefix("arm:")
        expected = {
            criterion.id
            for criterion in _bars(name)
            for read in _bar_reads(held, criterion)
            if read.arm == arm and read.value is not None and _misses(read.value, read.bar, read.direction)
        }
        assert {warning.criterion for warning in row.warnings if warning.status == "fail"} == expected, arm
    assert any(row.warnings for row in result.survivors) or not any(read.status == "missed" for read in held.bars), (
        "a bar the pipeline read missed is warned of"
    )


def test_settle_reads_a_bar_on_another_aspect_at_that_aspects_figure_and_count() -> None:
    held = budget_inputs()["s3"]
    asked = [aspect.id for aspect in held.aspects if aspect.asked]
    for criterion in _bars("s3"):
        for read in _bar_reads(held, criterion):
            assert read.value is not None
            result = settle_cases(held, [read.arm], criterion=criterion.id)
            (row,) = result.rows
            assert [figure.value for figure in row.estimate] == [pytest.approx(read.value)], criterion.id
            assert row.settles == ("misses" if _misses(read.value, read.bar, read.direction) else "clears")
            table = held.judged.get(criterion.measure) or held.measures[criterion.measure]
            stat = table.per_arm[read.arm] if criterion.aspect == asked[0] else table.by_aspect[read.aspect][read.arm]
            assert [figure.value for figure in row.counts] == [float(stat.n or 0)], criterion.id
            for figure in [*row.counts, *row.count_of, *row.basis]:
                assert resolve(figure.key, held).value == figure.value, figure.key


def test_a_chart_narrowed_to_an_aspect_draws_that_aspects_bar_line() -> None:
    held = budget_inputs()["s3"]
    for criterion in _bars("s3"):
        source = "judged" if criterion.measure in held.judged else "measures"
        query = ChartQuery.model_validate(
            {**_EMPTY, "source": source, "aspect": criterion.aspect, "measures": [criterion.measure]}
        )
        line = "floor" if criterion.direction == "higher_is_better" else "budget"
        block = block_render(held, block_kind="bars", query=query, reference=line)
        assert block.reference is not None and block.reference.key == f"criteria.{criterion.id}.threshold"


# --- a reference input with a criterion on another aspect ------------------------------------------------------------


def test_a_criterion_copied_onto_another_aspect_reads_that_aspects_figures_and_the_original_the_leads() -> None:
    held = fixture_inputs()["r4"]
    tagged, copied, aspect = criterion_on_another_aspect(held)
    copy = next(criterion for criterion in tagged.criteria if criterion.id == copied)
    original = next(criterion for criterion in tagged.criteria if criterion.id == copied.rsplit("@", 1)[0])
    assert copy.aspect == aspect and original.aspect is None
    table = tagged.judged.get(copy.measure) or tagged.measures[copy.measure]
    flipped = 0
    for arm, stat in table.by_aspect[aspect].items():
        assert stat.mean is not None
        (figure,) = criterion_figures(tagged, copy, arm)
        assert figure.value == pytest.approx(stat.mean)
        (lead,) = criterion_figures(tagged, original, arm)
        assert lead.value == pytest.approx(table.per_arm[arm].mean)
        assert resolve(f"criteria.{copy.id}.{arm}.value", tagged).value == pytest.approx(stat.mean)
        missed = _misses(stat.mean, float(copy.threshold), copy.direction)
        assert resolve(f"count.failing.{copy.id}.{arm}", tagged).value == float(missed)
        assert (gate_check(tagged, arm, criterion=copy.id).status == "fail") is missed, arm
        flipped += missed != _misses(lead.value, float(copy.threshold), copy.direction)
    assert flipped, "the control: the copy's figures sit on the other side of the line from the lead's"
    removed = {row.arm.removeprefix("arm:") for row in dominance(tagged, limit=200).out}
    for arm, stat in table.by_aspect[aspect].items():
        if stat.mean is not None and _misses(stat.mean, float(copy.threshold), copy.direction):
            assert arm in removed, arm


@pytest.mark.parametrize("name", sorted(fixture_inputs()))
def test_an_input_whose_criteria_name_no_aspect_dumps_as_it_did(name: str) -> None:
    held = fixture_inputs()[name]
    document = held.model_dump(mode="json", by_alias=True)
    assert all("aspect" not in criterion for criterion in document["criteria"])


def test_an_aspect_id_the_input_does_not_hold_is_refused_naming_it_and_the_ids_it_holds() -> None:
    held = next(inputs for inputs in fixture_inputs().values() if len(inputs.aspects) > 1)
    ids = ", ".join(repr(aspect.id) for aspect in held.aspects)

    for read in (aspect_of, aspect_board):
        with pytest.raises(KeyError) as refused:
            read(held, "no-such-aspect")
        assert refused.value.args == (f"the report input holds no aspect 'no-such-aspect'; its aspects are {ids}",)
    assert aspect_of(held, held.aspects[1].id) is held.aspects[1]
