"""A key reads the table's text: every family whose key prints a measure's figure reads it as the page's table does.

Report prose cites a figure by its key, and the Full results print the same figure in a table, so one figure reads
one way on the page. For each family that prints a measure's figure, a key here reads the text of the table cell
beside it, or the precision of that measure's set where no cell holds the figure, and the text
:class:`~threetears.evals.analysis.report.forms.PageForms` decides for that measure and role. A criterion's figures,
the weakest margin's and a settle distance name their measure only through the criterion, so they carry it
(:attr:`~threetears.evals.analysis.report.figure_keys.Figure.measure`) and read at its sets: a level at its levels'
set, a gap, a margin and a distance at its changes' set. A family no fixture states on a measure's set (a pooled
interval on a score or a measure, an interval derived from the cells on a measure) is read on a copy of a fixture
that states one, each test saying how it was made.
"""

from __future__ import annotations

import pytest

from threetears.evals.analysis.report.block_spec import block_look
from threetears.evals.analysis.report.blocks import TableBlock
from threetears.evals.analysis.report.figure_keys import Figure, format_figure, resolve
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.parts.full_results import every_table
from threetears.evals.analysis.report.fact_arms import fact_arms
from threetears.evals.analysis.report_input import Measure, ReportInput

from packages.evals.tests.report_doc_support import snapshot_of

#: The turn-time criterion on s2, and beta's figures under it.
_CRITERION = "criteria.answers/bar/total_ms.beta"


def _held(name: str) -> ReportInput:
    return ReportInput.model_validate(snapshot_of(name).report_input)


def _table(held: ReportInput, name: str) -> TableBlock:
    tables = every_table(held, block_look(held, fact_arms(held)), forms=PageForms.of(held))
    (table,) = [table for table in tables if table.name == name]
    return table


def _cell(table: TableBlock, column: str, **where: str) -> str:
    """The text of one cell, in the one row whose named columns read as given."""
    (row,) = [row for row in table.rows if all(row[name].text == text for name, text in where.items())]
    return row[column].text


def _level(figure: Figure) -> float:
    assert isinstance(figure.value, float), figure
    return figure.value


def _said(key: str, held: ReportInput) -> str:
    return format_figure(resolve(key, held), forms=PageForms.of(held))


@pytest.fixture(scope="module")
def kairo() -> ReportInput:
    return _held("prototype5/kairo")


@pytest.fixture(scope="module")
def s2() -> ReportInput:
    return _held("prototype2/s2")


@pytest.fixture(scope="module")
def measured_s2(s2: ReportInput, kairo: ReportInput) -> ReportInput:
    """s2 with beta's turn time at 701.37 ms and one comparison on the turn time of -3.7 ms (-6.0 to -1.4), whose
    half-width of 2.3 ms puts every figure of the measure at one decimal (number policy, rule 2): ``701.4 ms``, and a
    gap of 98.63 ms ``98.6``."""
    turn = s2.measures["total_ms"]
    beta = turn.per_arm["beta"].model_copy(update={"mean": 701.37})
    turn = turn.model_copy(update={"per_arm": {**turn.per_arm, "beta": beta}})
    comparison = kairo.contrasts[0].model_copy(
        update={"arm": "beta", "measure": "total_ms", "kind": "measure", "delta": -3.7, "interval": (-6.0, -1.4)}
    )
    return s2.model_copy(update={"measures": {**s2.measures, "total_ms": turn}, "contrasts": [comparison]})


def test_a_contrast_key_reads_its_change_and_interval_as_the_comparisons_table(kairo: ReportInput) -> None:
    table = _table(kairo, "contrasts")
    forms = PageForms.of(kairo)
    delta = resolve("contrast.qwen3.8-flash.__outcome__.delta", kairo)
    interval = resolve("contrast.qwen3.8-flash.__outcome__.interval", kairo)
    assert isinstance(interval.value, tuple)
    tabled = {"arm": "qwen3.8-flash", "measure": "Outcome (1–5)"}
    assert format_figure(delta, forms=forms) == _cell(table, "change", **tabled) == "-0.21"
    assert format_figure(interval, forms=forms) == _cell(table, "interval", **tabled) == "-0.31 to -0.12"
    assert forms.change("__outcome__", _level(delta), kind="judged").text == format_figure(delta, forms=forms)
    assert forms.interval("__outcome__", *interval.value, kind="judged", change=True).text == format_figure(
        interval, forms=forms
    )


def test_a_measure_key_reads_its_mean_as_the_arms_by_measure_table(kairo: ReportInput) -> None:
    figure = resolve("measure.failed_reply_notices.September.mean", kairo)
    tabled = _cell(_table(kairo, "arms_by_measure"), "measure:failed_reply_notices", arm="September")
    assert format_figure(figure, forms=PageForms.of(kairo)) == tabled == "0.0022 notices"
    assert PageForms.of(kairo).level("failed_reply_notices", _level(figure)).text == tabled


def test_a_measure_s_interval_and_a_bar_s_read_their_ends_at_the_measure_s_levels_set(s2: ReportInput) -> None:
    """The arms-by-measure table prints the turn time whole, and a measure's ``ci`` and a bar's ``interval`` on it read
    both ends whole too."""
    assert _cell(_table(s2, "arms_by_measure"), "measure:total_ms", arm="beta") == "701 ms"
    # The unit written once, after the high end (number policy, rule 3).
    assert _said("measure.total_ms.beta.ci", s2) == "673 to 729 ms"
    assert _said("aspects.answers.bar.total_ms.beta.interval", s2) == "673 to 729 ms"


def test_a_judged_key_reads_its_score_as_the_arms_by_measure_table(kairo: ReportInput) -> None:
    figure = resolve("judged.__outcome__.September", kairo)
    tabled = _cell(_table(kairo, "arms_by_measure"), "judged:__outcome__", arm="September")
    assert format_figure(figure, forms=PageForms.of(kairo)) == tabled == "4.69"
    assert PageForms.of(kairo).level("__outcome__", _level(figure), kind="judged").text == tabled


def test_a_stratum_key_reads_its_mean_as_the_strata_table(kairo: ReportInput) -> None:
    figure = resolve("strata.off_lane.September.__outcome__.mean", kairo)
    tabled = _cell(_table(kairo, "strata"), "arm:September", strata="off lane", measure="Outcome (1–5)")
    assert format_figure(figure, forms=PageForms.of(kairo)) == tabled == "4.75"
    assert PageForms.of(kairo).level("__outcome__", _level(figure), kind="judged").text == tabled


def test_a_pooled_key_reads_its_mean_at_its_measure_s_levels_set(kairo: ReportInput) -> None:
    """A score pooled over the strata reads at a score's two decimals, as the strata table prints each stratum's."""
    figure = resolve("pooled.__outcome__.September.mean", kairo)
    assert format_figure(figure, forms=PageForms.of(kairo)) == "4.67"
    assert PageForms.of(kairo).level("__outcome__", _level(figure), kind="judged").text == "4.67"


def test_a_task_key_reads_its_mean_as_the_tasks_table() -> None:
    held = _held("prototype/r1")
    figure = resolve("task.billing.atlas-l.overall.mean", held)
    tabled = _cell(_table(held, "tasks"), "arm:atlas-l", tasks="Billing & account Q&A", measure="Overall quality")
    assert format_figure(figure, forms=PageForms.of(held)) == tabled == "94"
    assert PageForms.of(held).level("overall", _level(figure), kind="judged").text == tabled


def test_a_pooled_interval_on_a_score_off_the_rubric_reads_by_the_number_rule_as_its_table_does() -> None:
    """A judged score on 0–100 has no set (it is no 1–5 rubric score), so the tasks table prints it by the number rule
    (``94``), and its pooled interval's ends read by that rule too."""
    assert _said("pooled.overall.atlas-l.interval", _held("prototype/r1")) == "92.1 to 93.5"


def test_a_derived_interval_on_a_rate_reads_its_ends_as_the_cells_table_reads_a_rate() -> None:
    """The interval derived from a target matrix's cells reads its ends in percent at the rate's one precision, as the
    cells table prints each cell's rate (``94%``)."""
    held = _held("prototype4/r3-without-its-configs")
    assert _cell(_table(held, "cells"), "arm:atlas-m.zero", cells="Simple × Science") == "94%"
    assert _said("overall_interval.balanced_accuracy.atlas-m.zero.interval", held) == "89 to 90%"


def test_a_pooled_interval_on_a_score_reads_its_ends_at_a_score_s_two_decimals(kairo: ReportInput) -> None:
    """Kairo's strata with a narrow half-width (0.08) on the outcome score: the pooled interval's ends read at a
    score's two decimals, as the strata table prints each stratum's (``4.66 to 4.69``), never at two significant figures
    of the interval's width (``4.656 to 4.687``)."""
    strata = [
        stratum.model_copy(
            update={
                "per_arm": {
                    arm: {
                        key: stat.model_copy(update={"half_width": 0.08}) if key == "__outcome__" else stat
                        for key, stat in figures.items()
                    }
                    for arm, figures in stratum.per_arm.items()
                }
            }
        )
        for stratum in kairo.strata
    ]
    held = kairo.model_copy(update={"strata": strata})
    forms = PageForms.of(held)
    figure = resolve("pooled.__outcome__.September.interval", held)
    assert isinstance(figure.value, tuple)
    low, high = (forms.level("__outcome__", end, kind="judged").text for end in figure.value)
    assert format_figure(figure, forms=forms) == f"{low} to {high}" == "4.66 to 4.69"


def test_a_pooled_interval_on_a_measure_reads_its_ends_at_the_measure_s_levels_set() -> None:
    """r1's overall quality held as a measure (its figures the judged dimension's): its tasks' means set the measure's
    levels whole, as the tasks table prints them (``94``), and the pooled interval's ends read whole too (``92 to 93``),
    never at two significant figures of its width (``92.1 to 93.5``)."""
    r1 = _held("prototype/r1")
    overall = r1.judged["overall"]
    measure = Measure(label=overall.label, unit="", per_arm=overall.per_arm, by_aspect=overall.by_aspect)
    held = r1.model_copy(
        update={
            "measures": {**r1.measures, "overall": measure},
            "judged": {key: judged for key, judged in r1.judged.items() if key != "overall"},
            "contrasts": [contrast for contrast in r1.contrasts if contrast.measure != "overall"],
        }
    )
    forms = PageForms.of(held)
    tabled = _cell(_table(held, "tasks"), "arm:atlas-l", tasks="Billing & account Q&A", measure="Overall quality")
    assert tabled == forms.level("overall", _level(resolve("task.billing.atlas-l.overall.mean", held))).text == "94"
    figure = resolve("pooled.overall.atlas-l.interval", held)
    assert isinstance(figure.value, tuple)
    low, high = (forms.level("overall", end).text for end in figure.value)
    assert format_figure(figure, forms=forms) == f"{low} to {high}" == "92 to 93"


def test_a_derived_interval_on_a_measure_reads_its_ends_at_the_measure_s_levels_set() -> None:
    """r3's accuracies in a measured unit (``ms``) in place of a percent: the interval derived from the cells reads its
    ends at the overall measure's levels' set, whole as the arms-by-measure table prints each arm's (``90 ms``), never
    at two significant figures of its width (``89.49 ms to 90.11 ms``)."""
    r3 = _held("prototype4/r3-without-its-configs")
    measured = {
        key: measure.model_copy(update={"unit": "ms", "grade": None})
        if key in ("accuracy", "balanced_accuracy")
        else measure
        for key, measure in r3.measures.items()
    }
    held = r3.model_copy(update={"measures": measured})
    forms = PageForms.of(held)
    tabled = _cell(_table(held, "arms_by_measure"), "measure:balanced_accuracy", arm="Atlas-M · Zero-shot")
    assert (
        tabled
        == forms.level("balanced_accuracy", _level(resolve("measure.balanced_accuracy.atlas-m.zero.mean", held))).text
        == "90 ms"
    )
    figure = resolve("overall_interval.balanced_accuracy.atlas-m.zero.interval", held)
    assert isinstance(figure.value, tuple)
    low, high = (forms.level("balanced_accuracy", end).text for end in figure.value)
    # Each end as the table prints a level, the unit written once, after the high end (number policy, rule 3).
    assert format_figure(figure, forms=forms) == f"{low.removesuffix(' ms')} to {high}" == "89 to 90 ms"


def test_a_criterion_s_figure_and_its_interval_s_ends_read_at_its_measure_s_levels_set(
    measured_s2: ReportInput,
) -> None:
    """The criterion's figure is the turn time's level, which the arms-by-measure table prints at the measure's one
    precision (its comparison's half-width of 2.3 ms reads it to a tenth: ``701.4 ms``), and its interval's ends read
    at it too."""
    forms = PageForms.of(measured_s2)
    value, low, high = (resolve(f"{_CRITERION}.{field}", measured_s2) for field in ("value", "low", "high"))
    tabled = _cell(_table(measured_s2, "arms_by_measure"), "measure:total_ms", arm="beta")
    assert format_figure(value, forms=forms) == tabled == "701.4 ms"
    assert (format_figure(low, forms=forms), format_figure(high, forms=forms)) == ("673.3 ms", "728.7 ms")
    for figure in (value, low, high):
        assert forms.level("total_ms", _level(figure)).text == format_figure(figure, forms=forms), figure.path


def test_a_criterion_s_gap_reads_at_its_measure_s_changes_set_and_unsigned(measured_s2: ReportInput) -> None:
    """The gap is a difference on the turn time, so it reads at the changes' one decimal, as the comparison on it
    does, and unsigned toward passing."""
    assert _said("contrast.beta.total_ms.delta", measured_s2) == "-3.7 ms"
    assert _said(f"{_CRITERION}.gap", measured_s2) == "98.6 ms"


def test_the_weakest_margin_reads_its_figure_and_its_margin_as_its_criterion_s_keys(measured_s2: ReportInput) -> None:
    """The weakest margin is beta's turn-time criterion: its figure reads as the criterion's ``value`` and the table's
    cell, its margin as the criterion's ``gap``."""
    tabled = _cell(_table(measured_s2, "arms_by_measure"), "measure:total_ms", arm="beta")
    assert _said("weakest.value", measured_s2) == _said(f"{_CRITERION}.value", measured_s2) == tabled == "701.4 ms"
    assert _said("weakest.margin", measured_s2) == _said(f"{_CRITERION}.gap", measured_s2) == "98.6 ms"


def test_a_settle_distance_reads_at_its_measure_s_changes_set_signed(measured_s2: ReportInput) -> None:
    """Settling a criterion reads its distance toward passing as a change on the turn time: signed, at the changes'
    one decimal, the criterion's gap with its sign."""
    assert _said(f"settle.{_CRITERION}.overall.distance", measured_s2) == "+98.6 ms"
