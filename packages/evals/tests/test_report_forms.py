"""The decided-figure model: one form per measure per page, reproducing what the page prints today.

``analysis/report/forms.py`` is where every figure's form is to be decided once per page, by its measure. Until the
policy flip it must hand out exactly the text each surface prints now, so routing a surface through it moves no golden.
Held here, on the pages the plan names:

- **Kairo's written page and rung 5's eleven writer cases**: every comparison's change and interval reads as the Full
  results' comparisons table prints it; every measured and judged-score level as the arms-by-measure table does; a
  rate's levels, through the table's column rule, as its rate columns do; and money at the scale its column draws.
- **The scaled campaign's code page**: the pass rate's change, its interval and the level it lands at read as the
  scorecard prints them; **a measurement's scorecard** (the s3 shape), its rates through the page's column rule and its
  money exactly.
- **The ledger**: every text handed out is recorded by measure, kind and role, a figure that is no measurement by its
  fixed form, so a page-level test can hold one figure to one text across surfaces.
- **The forms carry what they were built from**: a written page's its report input, which prose and charts read
  from them, so no figure reads one input in another's forms; an aspect page's its bundle; a reader handed both
  refuses a pair that does not match.

The column rule the tables apply after a cell is built (a money column's own form) is the page's forms' too
(:meth:`PageForms.money_columns`), recording what every money cell prints; a rate's or money's form here is what the
cell is built from.
"""

from __future__ import annotations

import functools
import math
import re
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from unittest import mock

import pytest

from threetears.evals.analysis import campaign_report, service
from threetears.evals.analysis.report.campaign.page import build_campaign_report_in_aspect_forms
from threetears.evals.analysis.analyst.snapshot import frozen_input
from threetears.evals.analysis.report.input_build import report_input
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.campaign_readings import AggregateFigure, CampaignReadings
from threetears.evals.analysis.report.block_spec import block_look, complete_chart
from threetears.evals.analysis.report.blocks import ChartBlock, TableBlock, TableCell, TableColumn, bare_columns
from threetears.evals.analysis.report.charts import BarRow, BarsChart, ChartReference
from threetears.evals.analysis.analyst.cited import cite
from threetears.evals.analysis.report.build import (
    build_code_only_report,
    build_code_only_report_in_forms,
    contrast_evidence,
)
from threetears.evals.analysis.report.figure_keys import (
    Figure,
    figure_key_grammar,
    format_figure,
    key_form,
    render,
    resolve,
)
from threetears.evals.analysis.report.keys.reads import cost_measure, pass_rate_measure
from threetears.evals.analysis.report.forms import ColumnMoney, MeasureForm, PageForms, Printed
from threetears.evals.analysis.report.model import CampaignReport, Report, ReportBlock
from threetears.evals.analysis.report.serialize_md import campaign_report_markdown
from threetears.evals.analysis.report.parts.full_results import every_table
from threetears.evals.analysis.report.parts.tables import arms_key, arms_table, pass_rate_cell
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.fact_arms import fact_arms
from threetears.evals.analysis.bundle.schema import FLOOR_COST_MEASURE, TRIAL_PASS_NAME
from threetears.evals.analysis.bundle.surface import bundle_decision_surface
from threetears.evals.analysis.report.figures import report_rulers
from threetears.evals.analysis.report.words import judged_heading, reading_heading
from threetears.evals.analysis.report_input import Contrast, Measure, ReportInput, Unanswered
from threetears.evals.analysis.report_margins import aspect_board
from threetears.evals.kernel.chart_query import PlannedChart
from threetears.evals.harness import read_writer_cases
from threetears.evals.kernel.numbers import (
    ABSENT,
    MoneyForm,
    format_number,
    format_signed,
    spell_count,
    spell_dollars,
    spell_multiple,
)

from packages.evals.tests.analyst_support import KAIRO, fixture_snapshots
from packages.evals.tests.example_loader import EXAMPLES
from packages.evals.tests.hot_path_support import HOT_PATH_SCOPE, hot_path_corpus
from packages.evals.tests.report_doc_support import snapshot_of
from packages.evals.tests.campaign_support import ASSEMBLED_AT
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_bundle
from packages.evals.tests.fixtures.report_inputs.dataset import kairo_bundle
from packages.evals.tests.campaign_page_support import kairo_page
from packages.evals.tests.report_shapes_support import s1_change, s2_choice, s3_measurement, s4_change


@functools.cache
def _written_inputs() -> dict[str, ReportInput]:
    """Kairo round 2's input, frozen from its bundle, and each of rung 5's writer cases', by name."""
    inputs = {KAIRO: frozen_input(fixture_snapshots("s")[KAIRO])}
    for case in read_writer_cases(EXAMPLES / "5-writer" / "cases"):
        inputs[f"rung 5 {case.name}"] = frozen_input(case.snapshot)
    return inputs


_WRITTEN = sorted(_written_inputs())


def _tables(held: ReportInput) -> dict[str, TableBlock]:
    return {
        table.name: table for table in every_table(held, block_look(held, fact_arms(held)), forms=PageForms.of(held))
    }


@dataclass(frozen=True)
class _Level:
    """One arm's level of one measure, as the arms-by-measure table prints it."""

    key: str
    kind: str
    mean: float
    cell: TableCell


def _levels(held: ReportInput, table: TableBlock) -> Iterator[tuple[int, str, _Level]]:
    """Every measure's level in the arms-by-measure table, with its row and column, each arm by its overall figure."""
    look = block_look(held, fact_arms(held))
    by_name = {look.arm(arm.id).name if arm.id in look.arms else arm.id: arm.id for arm in held.arms}
    for index, row in enumerate(table.rows):
        arm = by_name[row["arm"].text]
        for column, cell in row.items():
            prefix, _, key = column.partition(":")
            if prefix not in ("measure", "judged"):
                continue
            kind = "judged" if prefix == "judged" else "measure"
            stat = (held.judged if kind == "judged" else held.measures)[key].per_arm.get(arm)
            if stat is not None and stat.mean is not None:
                yield index, column, _Level(key, kind, float(stat.mean), cell)


def _arms_by_measure(held: ReportInput) -> TableBlock | None:
    """The arms-by-measure table where it reads each arm's overall figure (no aspect column)."""
    table = _tables(held).get("arms_by_measure")
    if table is None or any(column.key == "aspect" for column in table.columns):
        return None
    return table


# --- the written page ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", _WRITTEN)
def test_every_measure_the_input_holds_has_one_form(name: str) -> None:
    held = _written_inputs()[name]
    forms = PageForms.of(held)
    expected = {(key, "measure") for key in held.measures} | {(key, "judged") for key in held.judged}
    expected |= {(contrast.measure, contrast.kind) for contrast in held.contrasts}
    # A scoreboard that counts passes states the pass rate, whose form is there whether or not a comparison reads it.
    boards = [aspect_board(held, None), *(aspect_board(held, aspect.id) for aspect in held.aspects)]
    if any(row.passed is not None and row.usable for board in boards for row in board.values()):
        expected.add((TRIAL_PASS_NAME, "trial"))
    assert {(form.key, form.kind) for form in forms.measures} == expected


@pytest.mark.parametrize("name", _WRITTEN)
def test_every_comparison_reads_as_the_full_results_print_it(name: str) -> None:
    """A change and its interval: a rate's in points at one decimal, a score's at two, any other at its set's
    decimals, money in its changes' form, each end kept within how far the measure moves."""
    held = _written_inputs()[name]
    table = _tables(held).get("contrasts")
    if table is None:
        pytest.skip(f"{name} states no comparison")
    forms = PageForms.of(held)
    for contrast, row in zip(held.contrasts, table.rows, strict=True):
        if contrast.delta is not None:
            assert forms.change(contrast.measure, contrast.delta, kind=contrast.kind).text == row["change"].text
        if contrast.interval is not None:
            low, high = contrast.interval
            printed = forms.interval(contrast.measure, low, high, kind=contrast.kind, change=True)
            assert printed.text == row["interval"].text
            assert printed.text == f"{printed.low_text} to {printed.high_text}"
            # The ends' digits never carry the unit, so a cell under a head naming it prints them alone.
            assert printed.digits == f"{printed.low_digits} to {printed.high_digits}"
            if printed.unit:
                assert printed.unit not in printed.digits and "$" not in printed.digits, printed


@pytest.mark.parametrize("name", _WRITTEN)
def test_every_measured_or_scored_level_reads_as_the_full_results_print_it(name: str) -> None:
    """A count per trial, a time, a judged score: at the decimals of the set every mean the input holds of it makes."""
    held = _written_inputs()[name]
    table = _arms_by_measure(held)
    if table is None:
        pytest.skip(f"{name} reads no arm by measure")
    forms = PageForms.of(held)
    checked = 0
    for _index, _column, level in _levels(held, table):
        if forms.measure(level.key, kind=level.kind).reads_as in ("measured", "score"):
            assert forms.level(level.key, level.mean, kind=level.kind).text == level.cell.text, level.key
            checked += 1
    if name == KAIRO:
        assert checked, "Kairo prints measured and scored levels by arm, so this must check some"
    elif not checked:
        pytest.skip(f"{name} prints no measured or scored level by arm")


@pytest.mark.parametrize("name", _WRITTEN)
def test_a_rate_column_reads_as_its_levels(name: str) -> None:
    """A rate's level is the figure its cell is built from, and no column rule respells it: each cell prints its
    measure's level as the page's forms decide it (number rule 5)."""
    held = _written_inputs()[name]
    table = _arms_by_measure(held)
    if table is None:
        pytest.skip(f"{name} reads no arm by measure")
    forms = PageForms.of(held)
    rebuilt = [dict(row) for row in table.rows]
    rates: set[str] = set()
    for index, column, level in _levels(held, table):
        if forms.measure(level.key, kind=level.kind).reads_as == "rate":
            text = forms.level(level.key, level.mean, kind=level.kind).text
            rebuilt[index][column] = level.cell.model_copy(update={"value": text})
            rates.add(column)
    if not rates:
        pytest.skip(f"{name} prints no rate by arm")
    for printed, again in zip(table.rows, rebuilt, strict=True):
        for column in rates:
            assert again[column].text == printed[column].text, column


#: An amount of money as a cell prints it, its decimals captured.
_AMOUNT = re.compile(r"\$([\d,]+(?:\.(\d+))?)")


@pytest.mark.parametrize("name", _WRITTEN)
def test_money_reads_at_the_scale_its_column_draws_it(name: str) -> None:
    """A money level is drawn at its measure's one scale (per 1,000 replies, per 1,000,000), which its column then
    writes in the column's own form: the drawn amount is the printed one, to the printed digits."""
    held = _written_inputs()[name]
    table = _arms_by_measure(held)
    if table is None:
        pytest.skip(f"{name} reads no arm by measure")
    forms = PageForms.of(held)
    checked = 0
    for _index, _column, level in _levels(held, table):
        form = forms.measure(level.key, kind=level.kind)
        if form.reads_as != "money":
            continue
        found = _AMOUNT.search(level.cell.text)
        assert found is not None, level.cell.text
        printed, decimals = float(found[1].replace(",", "")), len(found[2] or "")
        assert math.isclose(printed, level.mean * form.factor, abs_tol=0.5 * 10**-decimals + 1e-12), level.key
        checked += 1
    if not checked:
        pytest.skip(f"{name} prints no money by arm")


# --- the code page ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _CodePage:
    page: CampaignReport
    readings: CampaignReadings
    reports: Mapping[str, Report]


@functools.cache
def _scaled() -> _CodePage:
    """The scaled campaign's code page, and the readings and aspects' pages it was laid out over."""
    corpus = hot_path_corpus(scale=1)
    with mock.patch.object(
        service, "build_campaign_report_in_aspect_forms", wraps=service.build_campaign_report_in_aspect_forms
    ) as built:
        page = campaign_report(corpus.host, corpus.campaign.id, HOT_PATH_SCOPE)
    (call,) = built.call_args_list
    _campaign, readings, reports = call.args
    return _CodePage(page, readings, reports)


def _named(blocks: list[ReportBlock], name: str) -> TableBlock:
    (table,) = [block for block in blocks if isinstance(block, TableBlock) and block.name == name]
    return table


def test_the_scaled_campaign_s_pass_rate_reads_as_its_scorecard_prints_it() -> None:
    """Each candidate's change in points and the level it lands at in percent, at the one precision the pass rate's
    narrowest interval sets across the campaign (whole points here), and its interval in the one form, bare under its
    ``(pts)`` head: ``23% · +2 pts`` and ``-3 to +8``."""
    scaled = _scaled()
    forms = PageForms.of_campaign(scaled.readings, scaled.reports)
    matrix = _named(scaled.page.blocks, "scorecard")
    aspects = _named(scaled.page.blocks, "aspects")
    (row,) = matrix.rows
    changes = [cell for key, cell in row.items() if key.startswith("arm:") and cell.level is not None]
    assert len(changes) == len(scaled.readings.readings) == len(aspects.rows)
    for reading, cell, line in zip(scaled.readings.readings, changes, aspects.rows, strict=True):
        assert cell.low is not None and cell.high is not None and cell.level is not None
        change = forms.change("passed", reading.change, kind="trial")
        interval = forms.interval("passed", cell.low, cell.high, kind="trial", change=True)
        level = forms.level("passed", cell.level / 100, kind="trial")
        assert cell.figure == f"{level.text} · {change.text}"
        assert (change.digits, change.unit) == (change.text.removesuffix(" pts"), "pts")
        assert interval.digits == f"{interval.low_digits} to {interval.high_digits}"
        assert interval.text == f"{interval.digits} pts"
        assert (line["change"].text, line["interval"].text) == (change.text, interval.digits)
    assert (changes[0].figure, aspects.rows[0]["interval"].text) == ("23% · +2 pts", "-3 to +8")


def test_a_measurement_s_scorecard_reads_as_its_forms() -> None:
    """A rate's level and its interval, and money on its ruler in its one form, each as the page's forms state them,
    the interval in its one form."""
    shape = s3_measurement()
    page = shape.page()
    forms = PageForms.of_campaign(shape.readings, shape.reports)
    table = _named(page.blocks, "scorecard")
    readings = list(shape.readings.measures)
    assert len(readings) == len(table.rows)
    kinds: set[str] = set()
    for index, reading in enumerate(readings):
        assert reading.value is not None and reading.interval is not None
        form = forms.measure(reading.measure)
        kinds.add(form.reads_as)
        level, interval = forms.bracketed(reading.measure, reading.value, *reading.interval)
        assert table.rows[index]["value"].figure == level.text
        assert table.rows[index]["interval"].text == interval.text
        assert " to " in interval.text and "[" not in interval.text
        if form.reads_as == "money":
            # Money's parts lay out apart: its "$" is the unit's, never the digits'.
            assert (level.unit, level.text) == ("usd", f"${level.digits}") and "$" not in level.digits
            assert interval.digits == f"{interval.low_digits} to {interval.high_digits}" and "$" not in interval.digits
    assert kinds == {"rate", "money"}


# The number policy on the code and aspect pages, row by row against design.md's "What the policy changed" table.


@functools.cache
def _scaled_markdown() -> str:
    return campaign_report_markdown(_scaled().page)


def test_the_p95_time_reads_at_the_places_its_interval_sets_with_each_interval_in_one_form() -> None:
    """A 2.4 s p95 with a ±0.07 s interval reads to the hundredth, its level, its change and both intervals alike, each
    interval ``low to high``, the unit on the figure it follows (``2.42 s (2.35 to 2.49)``): never
    ``2.4 [2.4, 2.5] s`` beside ``+0.028 s [-0.131, 0.187] s``."""
    assert (
        "| 95th-percentile extraction time | candidate-a | 2.39 s | 2.42 s (2.35 to 2.49) | +0.03 s | -0.13 to +0.19 s |"
        in _scaled_markdown()
    )


def test_fields_stripped_reads_at_the_places_its_interval_sets() -> None:
    """A count per trial reads at the hundredth its interval sets, never ``1.5 [1.5, 1.6]`` beside ``+0.095``. The
    change is +0.0947 (0.095 as printed before was itself rounded), so at the hundredth it reads ``+0.09``, not section
    2.5's ``+0.10``."""
    assert (
        "| Fields stripped | candidate-b | 1.43 | 1.53 (1.45 to 1.61) | +0.09 | -0.09 to +0.28 |" in _scaled_markdown()
    )


def test_a_pass_rate_change_reads_whole_with_an_end_off_zero_taking_the_digits_that_keep_it_off() -> None:
    """The pass rate's intervals set whole points across the campaign, so a change reads ``-7 pts``; the interval's high
    end, -0.03, would read as zero, so it alone takes the digits that keep it off (owner ruling, 2026-11-09), and the
    unit is written once: never ``-7.4 pts [-15.09, -0.03] pts``."""
    assert "| Pass rate | candidate-b | 21% | 13% | -7 pts | -15 to -0.03 pts |" in _scaled_markdown()


def test_the_statistics_print_each_effect_and_p_in_its_fixed_form() -> None:
    """An aspect page's Hedges' g, pairing r and adjusted p are no measurement: each cell prints the fixed form its
    page's forms decide (an effect at two decimals, p as APA writes it), never the number rule's ``0.08475``."""
    (fold,) = _scaled().page.aspects
    (table,) = [
        block for block in fold.report.blocks if isinstance(block, TableBlock) and block.name == "comparison_statistics"
    ]
    fixed = PageForms.off_page()
    assert table.rows
    for row in table.rows:
        for key, form in (("hedges_g", fixed.effect), ("pairing_r", fixed.effect), ("p_adjusted", fixed.p)):
            cell = row[key]
            value = float(cell.value) if isinstance(cell.value, int | float) else None
            assert cell.figure == form(value).text


def _narrowed(aspect: AnalysisContextBundle) -> AnalysisContextBundle:
    """``aspect`` with each pass rate contrast's interval cut to ±0.4 points about its change: the campaign's narrowest
    pass rate interval, which sets a tenth of a point where every other aspect's sets whole points."""
    families = [
        family.model_copy(
            update={
                "comparisons": [
                    comparison.model_copy(update={"interval": (comparison.delta - 0.004, comparison.delta + 0.004)})
                    if comparison.reading == "trial" and comparison.delta is not None
                    else comparison
                    for comparison in family.comparisons
                ]
            }
        )
        for family in aspect.multiple_comparisons.families
    ]
    return aspect.model_copy(
        update={"multiple_comparisons": aspect.multiple_comparisons.model_copy(update={"families": families})}
    )


def _pass_rate_changes(*, together: bool) -> tuple[list[str], list[str]]:
    """The two-aspect scaled campaign laid out as the engine lays it out, its first aspect's pass rate contrasts
    narrowed (:func:`_narrowed`): the pass rate's changes its second aspect's page prints in its contrasts table, and
    those the code page prints in its scorecard. ``together`` builds each aspect page's forms over the whole
    campaign, as the engine does (:meth:`PageForms.of_aspects`); else each aspect's alone (:meth:`PageForms.of_bundle`),
    the control."""
    corpus = hot_path_corpus(scale=1, two_aspects=True)
    bundle = service.assemble_campaign_bundle(corpus.host, corpus.campaign.id, HOT_PATH_SCOPE)
    first, second = bundle.aspects
    aspects = [_narrowed(first), second]
    forms = (
        PageForms.of_aspects(aspects, bundle.readings)
        if together
        else {aspect.aspect: PageForms.of_bundle(aspect) for aspect in aspects}
    )
    reports = {
        aspect.aspect: build_code_only_report_in_forms(aspect, assembled_at=ASSEMBLED_AT, forms=forms[aspect.aspect])
        for aspect in aspects
    }
    page = build_campaign_report_in_aspect_forms(corpus.campaign, bundle.readings, reports, aspect_forms=forms)
    (contrasts,) = [
        block
        for block in reports[second.aspect].blocks
        if isinstance(block, TableBlock) and block.name == "comparisons"
    ]
    on_aspect = [row["change"].text for row in contrasts.rows if row["reading"].text == "Pass rate"]
    matrix = _named(page.blocks, "scorecard")
    on_code = [
        cell.text.partition(" · ")[2].partition(" (")[0]
        for row in matrix.rows
        for key, cell in row.items()
        if key.startswith("arm:") and cell.level is not None
    ]
    assert on_aspect and on_code
    return on_aspect, on_code


def test_one_measure_reads_at_one_precision_on_every_page_of_its_campaign() -> None:
    """Set scope is the whole campaign (owner ruling, 2026-11-09): where one aspect's pass rate interval is ten times
    narrower than any other's, every page of the campaign prints the pass rate at the tenth that interval sets, the
    other aspect's page and the code page alike, never at the whole points its own intervals would set alone."""
    tenth = re.compile(r"[+-]\d+\.\d pts")
    on_aspect, on_code = _pass_rate_changes(together=True)
    assert all(tenth.fullmatch(text) for text in on_aspect), on_aspect
    assert all(tenth.fullmatch(text) for text in on_code), on_code
    # The control: built alone, the second aspect's page reads its own whole points.
    alone, _ = _pass_rate_changes(together=False)
    assert all(re.fullmatch(r"[+-]?\d+ pts", text) for text in alone), alone


#: A rate's figure as a page prints it: a percent or a change in points.
_RATE_FIGURE = re.compile(r"([+-]?\d+(?:\.(\d+))?)(?:%| pts)")


def _rate_places(forms: PageForms) -> dict[tuple[str, str], set[int]]:
    """The decimals each rate measure's figures print at in one page's ledger, by measure and kind, its levels,
    changes, interval ends, half-widths and pass^k alike, as its forms decided them (a column's record of a cell
    restates one, and a bar's threshold is declared, not measured); a figure the boundary exception keeps off zero or a
    rate's floor or ceiling (within one point of it) is left out, since it alone takes the digits that keep it off."""
    found: dict[tuple[str, str], set[int]] = {}
    for entry in forms.ledger:
        if not entry.measure or entry.role not in ("level", "change", "interval", "half_width", "pass_k"):
            continue
        for match in _RATE_FIGURE.finditer(entry.text):
            value = abs(float(match[1]))
            if value < 1 or value > 99:
                continue
            found.setdefault((entry.measure, str(entry.kind)), set()).add(len(match[2] or ""))
    return found


# s2's aspect page decides no rate's figure between 1% and 99% (its levels and changes sit at a boundary or are counts),
# so it holds none to check.
@pytest.mark.parametrize("source", ["scaled", "s1_change", "s3_measurement", "s4_change", "toyhost"])
def test_no_rate_prints_at_two_precisions_on_one_page(source: str) -> None:
    """Rule 1: a measure reads at one precision on the whole page. Every rate's levels, changes and interval ends on a
    code page, and on each aspect page, print at one count of decimals, but for an end kept off a boundary."""
    pages = (
        [_page_and_forms(_scaled_page)[1]]
        if source == "scaled"
        else [forms for _, forms in _aspect_pages(source).values()]
    )
    assert any(_rate_places(forms) for forms in pages), "the control: a page states some rate"
    for forms in pages:
        mixed = {key: places for key, places in _rate_places(forms).items() if len(places) > 1}
        assert not mixed, mixed


def _narrowed_readings(readings: CampaignReadings) -> CampaignReadings:
    """``readings`` with each row's pass rate intervals cut to ±0.4 points about its change: the code page's own
    intervals become the campaign's narrowest, which sets a tenth of a point where every aspect's sets whole points."""
    rows = [
        row.model_copy(
            update={
                name: (row.change - 0.4, row.change + 0.4)
                for name in ("gain_interval", "drop_interval", "within_interval")
                if getattr(row, name) is not None
            }
        )
        for row in readings.readings
    ]
    return readings.model_copy(update={"readings": rows})


def test_the_code_page_s_readings_set_the_precision_of_every_aspect_page_the_engine_lays_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine's own route (``campaign_report``): where the code page's readings hold the campaign's narrowest pass
    rate interval, each aspect's page, built by the service in the campaign's forms, prints the pass rate at the
    tenth that interval sets, as the code page does. An aspect page built over its own bundle alone, or in forms that
    leave the readings out, would read its own whole points."""
    real = service.campaign_readings
    monkeypatch.setattr(service, "campaign_readings", lambda *args, **kwargs: _narrowed_readings(real(*args, **kwargs)))
    page = _scaled_page(two_aspects=True)
    tenth = re.compile(r"[+-]\d+\.\d pts")
    on_aspects = [
        row["change"].text
        for fold in page.aspects
        for block in fold.report.blocks
        if isinstance(block, TableBlock) and block.name == "comparisons"
        for row in block.rows
        if row["reading"].text == "Pass rate"
    ]
    on_code = [
        cell.text.partition(" · ")[2].partition(" (")[0]
        for row in _named(page.blocks, "scorecard").rows
        for key, cell in row.items()
        if key.startswith("arm:") and cell.level is not None
    ]
    assert on_aspects and on_code
    assert all(tenth.fullmatch(text) for text in on_aspects), on_aspects
    assert all(tenth.fullmatch(text) for text in on_code), on_code


def _pooled_narrowed(readings: CampaignReadings) -> CampaignReadings:
    """``readings`` with the campaign-wide pooled figures' every interval cut to ±0.1 points about its figure: narrower
    than any interval a reader compares."""
    assert readings.aggregate is not None

    def cut(figure: AggregateFigure | None) -> AggregateFigure | None:
        if figure is None or figure.value is None or figure.interval is None:
            return figure
        return figure.model_copy(update={"interval": (figure.value - 0.1, figure.value + 0.1)})

    arms = [
        arm.model_copy(update={"pass_rate": cut(arm.pass_rate), "change": cut(arm.change)})
        for arm in readings.aggregate.arms
    ]
    return readings.model_copy(update={"aggregate": readings.aggregate.model_copy(update={"arms": arms})})


def _kairo_pass_rate_places(readings: CampaignReadings) -> int | None:
    """The decimals Kairo's aspect page reads the pass rate's changes at, its forms built over the campaign."""
    bundle = kairo_bundle()
    (aspect,) = PageForms.of_aspects(bundle.aspects, readings).values()
    places = aspect.contrast_places(TRIAL_PASS_NAME, "trial")
    assert places is not None
    return places.change


def _kairo_pass_rate_half_width(readings: CampaignReadings) -> float | None:
    """The half-width Kairo's code page reads the pass rate's decimals from (:attr:`MeasureForm.half_width`), its forms
    built over ``readings`` and every aspect page's intervals, as the aspect pages' are."""
    reports = {fold.aspect: fold.report for fold in kairo_page().aspects}
    return PageForms.of_campaign(readings, reports).measure(TRIAL_PASS_NAME, kind="trial").half_width


def test_a_pooled_figure_never_sets_a_measure_s_digits() -> None:
    """Owner ruling, set digits: a measure's digits come from the narrowest interval on a figure a reader compares (an
    arm's level, a comparison's change). The code page's campaign-wide pooled pass rate and its change, cut narrower
    than every compared interval, leave the interval Kairo's pass rate reads its digits from where it was; a compared
    reading cut as narrow becomes that interval (the control). Under the precision rule (owner ruling, 2026-11-14) the
    pass rate reads a tenth from its compared intervals already, a rate's most, so the digits alone no longer tell the
    three apart: the half-width they are read from does."""
    readings = kairo_bundle().readings
    compared = _kairo_pass_rate_half_width(readings)
    assert compared is not None and compared > 1
    assert _kairo_pass_rate_places(readings) == 1
    assert _kairo_pass_rate_half_width(_pooled_narrowed(readings)) == compared
    assert _kairo_pass_rate_places(_pooled_narrowed(readings)) == 1
    assert _kairo_pass_rate_half_width(_narrowed_readings(readings)) == pytest.approx(0.4)


def test_a_written_page_reads_each_measure_at_the_precision_its_campaign_s_other_pages_do() -> None:
    """One form per measure across the campaign: Kairo's written page reads the pass rate at the places its aspect's
    page does, both from the narrowest interval a reader compares, the change 1017 reads at ±3.5 points, never the
    code page's pooled change (±2.86 points). Under the precision rule (owner ruling, 2026-11-14) both read a tenth,
    two significant figures of a 7-point width."""
    held = report_input(kairo_bundle())
    written = PageForms.of(held).measure(TRIAL_PASS_NAME, kind="trial")
    assert written.places == _kairo_pass_rate_places(kairo_bundle().readings) == 1
    assert written.half_width == _kairo_pass_rate_half_width(kairo_bundle().readings) == pytest.approx(3.5, abs=0.05)


def test_a_judged_bar_reads_alike_on_the_code_page_and_its_aspect_page() -> None:
    """A bar on a judged dimension reads on the dimension's own ruler on the code page, as its aspect's page reads it:
    its threshold, its value and its interval in one text each (``at least 90%``, ``95%``, ``75 to 99%``), never the raw
    scale's ``0.9`` and ``0.95``."""
    page = s3_measurement().page()
    code = {
        (row["aspect"].text, row["arm"].text): (row["bar"].text, row["value"].text, row["interval"].text)
        for row in _named(page.blocks, "bars").rows
    }
    checked = 0
    for fold in page.aspects:
        for block in fold.report.blocks:
            if not (isinstance(block, TableBlock) and block.name == "bars"):
                continue
            for row in block.rows:
                if "(judged)" not in row["bar"].text:
                    continue
                bar, value, interval = code[fold.aspect, row["arm"].text]
                assert bar.partition(" ")[2].partition(" ")[2] == row["threshold"].text, (bar, row["threshold"].text)
                assert (value, interval) == (row["value"].text, row["interval"].text)
                checked += 1
    assert checked, "the control: s3 reads a judged bar"


def _a_rate(places: int) -> PageForms:
    """A code page's forms holding one rate, read at ``places``."""
    form = MeasureForm(
        key="accuracy",
        kind="measure",
        reads_as="rate",
        unit="%",
        change_unit="pts",
        factor=100.0,
        change_factor=100.0,
        span=100.0,
        places=places,
        half_width=9.0,
    )
    return PageForms("code", {("accuracy", "measure"): form})


def test_a_rate_s_value_and_its_interval_read_at_one_precision() -> None:
    """A scorecard's rate and its interval share its measure's one precision (#1127 item 10): ``84%`` beside
    ``73 to 92%``, never ``84%`` beside ``[73.2%, 91.5%]``."""
    level, interval = _a_rate(0).bracketed("accuracy", 0.8444, 0.732, 0.915)
    assert (level.text, interval.text) == ("84%", "73 to 92%")


def test_a_rate_of_none_reads_with_its_percent() -> None:
    """A lost-call rate of none reads ``0%`` beside its neighbours' units, never a bare ``0`` (#1127 item 12)."""
    assert _a_rate(1).level("accuracy", 0.0).text == "0%"


def test_a_rate_near_its_ceiling_keeps_off_it() -> None:
    """A rate read in whole percents that would round onto its ceiling takes the digit that keeps it off: ``99.7%``,
    never ``100%`` (owner ruling, 2026-11-09)."""
    forms = _a_rate(0)
    assert forms.level("accuracy", 0.997).text == "99.7%"
    assert forms.level("accuracy", 1.0).text == "100%"


def test_the_code_page_builds_its_forms_once_and_issues_every_change_its_scorecard_prints() -> None:
    """The page's one forms object decides its scorecard: each candidate's change cell holds the number the pass
    rate's form issued, and its text is one the ledger holds."""
    page = _scaled().page
    forms, _ = _ledger_and_page(_scaled_page)
    issued = {entry.text for entry in forms.issued("passed", kind="trial", role="change")}
    (row,) = _named(page.blocks, "scorecard").rows
    changes = [cell for key, cell in row.items() if key.startswith("arm:") and cell.level is not None]
    assert changes
    for cell in changes:
        assert cell.figure.partition(" · ")[2] in issued


def _code_forms() -> PageForms:
    scaled = _scaled()
    return PageForms.of_campaign(scaled.readings, scaled.reports)


def test_a_change_its_half_width_and_its_sized_ends_read_at_the_pass_rate_s_places() -> None:
    """On the scaled campaign the pass rate's narrowest interval sets whole points: a change, how far an interval
    reaches and an interval's ends sized in a sentence all read whole, half to even, but for an end a whole point
    would print as zero, which takes just the digits that keep it off zero."""
    forms = _code_forms()
    change = forms.change("passed", 2.46, kind="trial")
    assert (change.held, change.text) == (2.0, "+2 pts")
    width = forms.half_width("passed", 3.04, kind="trial")
    assert (width.held, width.text, width.unit) == (3.0, "3 pts", "pts")
    assert [entry.role for entry in forms.ledger] == ["change", "half_width"]
    sized = forms.sizes("passed", -3.46, 7.9, kind="trial")
    assert (sized.low_text, sized.high_text) == ("3", "8")
    finer = forms.sizes("passed", -0.03, 4.2, kind="trial")
    assert (finer.low_text, finer.high_text) == ("0.03", "4")


def test_a_level_drawn_in_percent_reads_half_to_even_in_a_cell_and_in_words_alike() -> None:
    """A baseline in percent, ``6.25``, on a pass rate whose interval sets one decimal, is held and stated ``6.2%``
    (half to even, the one rounding) in its cell and in words alike: the words' own half-up ``6.3%`` is gone."""
    form = MeasureForm(
        key="passed",
        kind="trial",
        reads_as="rate",
        unit="%",
        change_unit="pts",
        factor=100.0,
        change_factor=1.0,
        span=100.0,
        places=1,
        half_width=1.5,
    )
    forms = PageForms("code", {("passed", "trial"): form})
    level = forms.drawn_level("passed", 6.25, kind="trial")
    assert (level.held, level.text, level.unit) == (6.2, "6.2%", "%")
    assert forms.drawn_level("passed", None, kind="trial").held is None


@pytest.mark.parametrize(
    ("level", "text", "held"), [(0.95, "95%", 0.95), (0.99688, "99.69%", 0.9969), (0.025, "2.5%", 0.025)]
)
def test_an_interval_s_level_held_as_a_share_reads_to_the_hundredth_that_tells_two_apart(
    level: float, text: str, held: float
) -> None:
    printed = PageForms("code", {}).confidence_share(level)
    assert (printed.text, printed.held, printed.unit) == (text, held, "%")


def test_a_pass_k_and_a_share_are_held_as_a_table_s_cell_holds_them() -> None:
    forms = PageForms("code", {})
    pass_k = forms.pass_k("passed", 0.4567, kind="trial")
    assert (pass_k.held, pass_k.text) == (45.7, "45.7%")
    assert forms.pass_k("passed", None, kind="trial").held is None
    # Recorded under its measure, so the policy flip can re-decide it there.
    assert [entry.text for entry in forms.issued("passed", kind="trial", role="pass_k")] == ["45.7%", ABSENT]
    share = forms.share(0.374)
    assert (share.held, share.text) == (37, "37%") and isinstance(share.held, int)


def test_a_money_level_bracketed_by_its_interval_reads_both_in_one_form() -> None:
    """Money drawn with its interval beside it takes the decimals its level and both ends need, as one cell."""
    shape = s3_measurement()
    forms = PageForms.of_campaign(shape.readings, shape.reports)
    reading = next(reading for reading in shape.readings.measures if forms.measure(reading.measure).reads_as == "money")
    assert reading.value is not None and reading.interval is not None
    level, interval = forms.bracketed(reading.measure, reading.value, *reading.interval)
    form = forms.measure(reading.measure)
    assert form.ruler is not None
    # One form for all three: the level's decimals write both ends.
    found = re.search(r"\.(\d+)$", level.text)
    money = MoneyForm(decimals=len(found.group(1)) if found is not None else 0)
    assert level.text == money.spell(level.held) and level.held == form.ruler.drawn(reading.value)
    low, high = (form.ruler.drawn(end) for end in reading.interval)
    assert interval.text == f"{money.spell(low)} to {money.spell(high)}"
    assert "$" not in interval.digits


@pytest.mark.parametrize(
    "figure",
    [
        lambda forms, key: forms.drawn_level(key, 1.0),
        lambda forms, key: forms.half_width(key, 1.0),
        lambda forms, key: forms.sizes(key, -1.0, 1.0),
        lambda forms, key: forms.bracketed(key, 1.0, 0.0, 2.0),
    ],
)
def test_a_written_page_refuses_a_code_page_s_own_forms(figure: Callable[[PageForms, str], object]) -> None:
    held = _written_inputs()[KAIRO]
    forms = PageForms.of(held)
    with pytest.raises(ValueError, match="only on a code page"):
        figure(forms, next(iter(held.measures)))


@pytest.mark.parametrize(
    ("figures", "decimals"),
    [([2.737, 0.0], 1), ([3.0, 2.0], 1), ([99.2345, 100.0], 0), ([0.0, 0.0], 0), ([None], 0)],
)
def test_a_column_of_measured_figures_reads_at_its_set_s_decimals(
    figures: list[float | None], decimals: int | None
) -> None:
    """A measure that is neither a rate nor money, with no interval anywhere, reads at two significant figures of its
    least figure but zero, over each arm's one figure: ``2.737`` beside ``0`` at one, ``99.2345`` beside ``100`` at
    none; a column of zeros or of no figure at all reads whole, as an exact zero does."""
    shape = s3_measurement()
    base = shape.readings.measures[0]
    made = [
        base.model_copy(
            update={"measure": "made_up", "arm": f"arm {index}", "value": figure, "interval": None, "passed": None}
            | {"trials": None}
        )
        for index, figure in enumerate(figures)
    ]
    readings = shape.readings.model_copy(update={"measures": [*shape.readings.measures, *made]})
    form = PageForms.of_campaign(readings, shape.reports).measure("made_up")
    assert (form.reads_as, form.places) == ("measured", decimals)


#: The fixed-form figures a code page prints only under a head naming its unit, ``(%)`` or ``(pts)``, as bare figures
#: (number rule 3: no unit under a head that names it): a half-width (``Half-width (pts)``), a pass^k (``Pass^k there
#: (%)``) and the between-cases share (``Between cases (%)``). Their ledger texts carry the unit the page leaves to the
#: head; what the column prints is recorded beside them (``column``).
_RESPELLED_ROLES = frozenset({"half_width", "pass_k", "share"})


def _respelled(forms: PageForms, entry) -> bool:
    """Whether a code page prints an issued figure (an entry of ``forms.ledger``) bare, under a head naming its unit,
    by its measure and role:

    - :data:`_RESPELLED_ROLES`, each under a head naming its unit;
    - a rate's interval: a pass rate's under ``Interval (pts)`` or a level's under its head's ``(%)``;
    - a measurement's rate level (kind ``measure``): bare under ``Accuracy (%)`` in its matrix.

    Never excluded: a change's pass rate level and change (kind ``trial``), whose cells carry their decimals, which
    the rule keeps; and an interval's confidence level, which only sentences state and the rule never reads."""
    if entry.role in _RESPELLED_ROLES:
        return True
    if not entry.measure or forms.measure(entry.measure, kind=entry.kind).reads_as != "rate":
        return False
    return entry.role == "interval" or (entry.role == "level" and entry.kind == "measure")


def _ledger_and_page(page_of: Callable[[], CampaignReport]) -> tuple[PageForms, str]:
    """The one forms a code page was built with, and its markdown, with each pass rate a cost and pass rate chart
    states bare under its ``(%)`` head written with its unit after it, as its forms issued it."""
    built: list[PageForms] = []
    real = PageForms.of_campaign

    def capture(readings: CampaignReadings, reports: Mapping[str, Report]) -> PageForms:
        forms = real(readings, reports)
        built.append(forms)
        return forms

    with mock.patch.object(PageForms, "of_campaign", side_effect=capture):
        page = page_of()
    (forms,) = built
    charted = [
        f"{point.pass_rate_text}%"
        for block in page.blocks
        if isinstance(block, ChartBlock) and block.chart == "cost_quality"
        for point in block.points
        if point.pass_rate_text is not None
    ]
    return forms, "\n".join([campaign_report_markdown(page), *charted])


@pytest.mark.parametrize(
    "page_of",
    [
        pytest.param(lambda: _scaled_page(), id="scaled"),
        pytest.param(lambda: _scaled_page(two_aspects=True), id="scaled_two_aspects"),
        pytest.param(lambda: s3_measurement().page(), id="measurement"),
    ],
)
def test_the_ledger_holds_only_text_the_code_page_prints(page_of: Callable[[], CampaignReport]) -> None:
    """Every text a code page's forms issued is on the page: a decision that compares figures prints nothing and
    issues nothing. Excluded, by measure and role: a figure the page prints bare under a head naming its unit
    (:func:`_respelled`), whose ledger text carries the unit; the column's own record of it is held to the page like
    every other entry."""
    forms, body = _ledger_and_page(page_of)
    assert forms.ledger
    missing = [entry for entry in forms.ledger if entry.text not in body and not _respelled(forms, entry)]
    assert not missing, missing


def test_a_decision_that_prints_nothing_issues_nothing() -> None:
    """Comparing two levels at the precision they read at prints neither, so the ledger is left as it was."""
    forms = _code_forms()
    forms.count(3)
    before = forms.ledger
    assert forms.told_apart(0.99688) == 0.9969
    assert forms.drawn_held("passed", 86.65, kind="trial") == 87.0
    assert forms.ledger == before


@pytest.mark.parametrize("two_aspects", [False, True], ids=["scaled", "scaled_two_aspects"])
def test_the_control_cell_issues_only_the_level_it_prints(two_aspects: bool) -> None:
    """On a change's matrix each pass rate level issued is one a cell prints: the control's (its pooled rate and its
    candidates' baselines compared, not issued), each candidate's level beside its change, and each baseline a cell
    names in words; on the row across the aspects, the control's level across them in place of its own figure; and
    each arm's pass rate the cost and pass rate chart states."""
    forms, _ = _ledger_and_page(lambda: _scaled_page(two_aspects=two_aspects))
    rows = _named(_scaled_page(two_aspects=two_aspects).blocks, "scorecard").rows
    assert len(rows) == (3 if two_aspects else 1), "two aspects and the row across them, or one aspect"
    printed = [
        cell
        for row in rows
        for key, cell in row.items()
        if key.startswith("arm:") and cell.is_number and (cell.unit == "%" or cell.level is not None)
    ]
    named = sum((cell.flag or "").count("'s ") for cell in printed if cell.level is not None)
    # The cost and pass rate chart states each arm's pass rate across the aspects too, in its values table.
    charted = sum(
        len(block.points)
        for block in _scaled_page(two_aspects=two_aspects).blocks
        if isinstance(block, ChartBlock) and block.chart == "cost_quality"
    )
    levels = forms.issued("passed", kind="trial", role="level")
    assert len(levels) == len(printed) + named + charted
    controls = [cell for cell in printed if cell.level is None]
    assert len(controls) == len(rows) and {cell.figure for cell in controls} <= {entry.text for entry in levels}


def _scaled_page(*, two_aspects: bool = False) -> CampaignReport:
    corpus = hot_path_corpus(scale=1, two_aspects=two_aspects)
    return campaign_report(corpus.host, corpus.campaign.id, HOT_PATH_SCOPE)


# --- the ledger holds what a rate or money column prints, and what a prose key prints ------------------------------

#: A money figure a cell states as its words (an arm's cost, spelled on its column's ruler): ``$0.49``, ``-$1.20``.
_MONEY_WORDS = re.compile(r"[-+]?\$[\d,]+(?:\.\d+)?")


def _printing(cell: TableCell, bare: bool) -> str | None:
    """How a cell prints a rate's or money's figure, or None for a cell that prints neither: ``number`` (in percent, in
    points or in money, or bare under a head naming its unit), ``rate words`` (a percent or points stated in its words,
    ``[1.0%, 9.0%]``, or any figure under such a head) or ``money words`` (an amount as its words, ``$0.49``)."""
    if cell.is_number:
        return "number" if cell.unit in ("%", "pts", "usd") or (bare and not cell.unit) else None
    if not isinstance(cell.value, str):
        return None
    if _MONEY_WORDS.fullmatch(cell.value):
        return "money words"
    if re.search(r"\d", cell.value) and (bare or "%" in cell.value or "pts" in cell.value):
        return "rate words"
    return None


#: Each table row's measure and reading, by the table's name and the row's place in it, or one cell's, by its column
#: too (a table of an arm per row and a measure per column): what each rate or money cell states a figure of.
_RowMeasures = Mapping[tuple[str, int] | tuple[str, int, str], tuple[str, str]]


def _unheld(
    tables: list[TableBlock], forms: PageForms, *, by_measure: bool = False, measures: _RowMeasures | None = None
) -> tuple[int, list[str]]:
    """How many rate and money cells the tables print, and each one the page's ledger holds no entry of its own for.

    Each cell takes one entry, by role and figure, so a cell is never passed by another cell's entry of the same text: a
    number the column rule's (``column``, its value and its text); words its measure decided, by the role it issued
    them in (``level``, ``change`` or ``interval``), or a column rule read the same, by their text; an interval's level,
    the level's (``confidence``); money as its words, the amount's as its
    builder issued it (``money`` on a ruler, or its measure's ``change`` or ``level``). ``by_measure``, only an entry
    held under a measure counts, but for an interval's level, which is how a family's intervals were read and belongs to
    no measure; and where ``measures`` names a row's measure, or a cell's, only an entry held under exactly that measure
    and reading counts for that cell."""
    entries = Counter((entry.role, entry.value, entry.text, entry.measure, entry.kind) for entry in forms.ledger)
    own: tuple[str, str] | None = None

    def take(role: str, text: str, value: object = None, *, any_value: bool = False) -> bool:
        found = next(
            (
                key
                for key, count in entries.items()
                if count > 0
                and key[0] == role
                and key[2] == text
                and (any_value or key[1] == value)
                and (
                    role == "confidence"
                    or ((key[3], key[4]) == own if own is not None else bool(key[3]) or not by_measure)
                )
            ),
            None,
        )
        if found is None:
            return False
        entries[found] -= 1
        return True

    checked, missing = 0, []
    for table in tables:
        bare = bare_columns(table.columns)
        for at, row in enumerate(table.rows):
            for key, cell in row.items():
                printing = _printing(cell, key in bare)
                if printing is None:
                    continue
                own = (measures or {}).get((table.name, at, key)) or (measures or {}).get((table.name, at))
                checked += 1
                text = cell.figure
                if printing == "number":
                    held = take("column", text, float(cell.value))  # type: ignore[arg-type]
                elif printing == "rate words":
                    # A rate's words are its measure's decided text, issued in its role; a column rule's, where a table
                    # still respells them.
                    held = (
                        take("column", text)
                        or any(take(role, text, any_value=True) for role in ("level", "change", "interval"))
                        or take("confidence", text, any_value=True)
                    )
                else:
                    held = any(take(role, text, any_value=True) for role in ("money", "change", "level"))
                if not held:
                    under = "" if own is None else f" under {own}"
                    missing.append(f"{table.name}.{key} ({printing}): {text!r}{under}")
    return checked, missing


def _page_and_forms(page_of: Callable[[], CampaignReport]) -> tuple[CampaignReport, PageForms]:
    """A code page and the one forms it was built with."""
    built: list[PageForms] = []
    real = PageForms.of_campaign

    def capture(readings: CampaignReadings, reports: Mapping[str, Report]) -> PageForms:
        forms = real(readings, reports)
        built.append(forms)
        return forms

    with mock.patch.object(PageForms, "of_campaign", side_effect=capture):
        page = page_of()
    (forms,) = built
    return page, forms


@pytest.mark.parametrize(
    "page_of",
    [
        pytest.param(lambda: _scaled_page(), id="scaled"),
        pytest.param(lambda: _scaled_page(two_aspects=True), id="scaled_two_aspects"),
        pytest.param(lambda: s3_measurement().page(), id="measurement"),
    ],
)
def test_every_rate_and_money_cell_a_code_page_prints_is_in_its_ledger(page_of: Callable[[], CampaignReport]) -> None:
    """The page's own tables print each rate and money figure as its forms recorded it: the column rules record the
    text a cell prints, not only what the cell was built from, so a page-level test of one figure to one text sees
    every rate and money column. Each aspect's fold is that aspect's own page, with forms of its own."""
    page, forms = _page_and_forms(page_of)
    tables = [block for block in page.blocks if isinstance(block, TableBlock)]
    checked, missing = _unheld(tables, forms)
    assert checked, "a code page prints rates and money, so this must check some"
    assert not missing, missing


#: The code page's cells that print a rate's figure of no measure: an interval's share of its variance between cases,
#: and a test setup's disclosure stated in words.
_NO_MEASURE_CODE_CELLS = frozenset({("precision", "between"), ("test_setup", "detail")})


def _pass_rate_rows(_page: CampaignReport, tables: list[TableBlock]) -> dict[tuple[str, int], tuple[str, str]]:
    """Each row's measure on a code page of pass-rate readings: every one is the pass rate's, how each trial ended."""
    return {(table.name, at): (TRIAL_PASS_NAME, "trial") for table in tables for at in range(len(table.rows))}


def _measurement_rows(_page: CampaignReport, tables: list[TableBlock]) -> dict[tuple[str, int], tuple[str, str]]:
    """Each row's measure on a code page of measurements, read off the readings its campaign's page is built from, in
    the order its tables lay their rows out: the measures table's a reading per row, the scorecard's an aspect's
    measure per row."""
    read = s3_measurement().readings.measures
    by_table = {
        "measures": [reading.measure for reading in read],
        "scorecard": [measure for _aspect, measure in dict.fromkeys((one.aspect, one.measure) for one in read)],
    }
    expected: dict[tuple[str, int], tuple[str, str]] = {}
    for table in tables:
        if table.name in by_table:
            assert len(by_table[table.name]) == len(table.rows), table.name
            expected |= {(table.name, at): (measure, "measure") for at, measure in enumerate(by_table[table.name])}
    assert expected, "a measurement's code page prints its measures, so this must name some"
    return expected


@pytest.mark.parametrize(
    ("page_of", "measures_of"),
    [
        pytest.param(lambda: _scaled_page(), _pass_rate_rows, id="scaled"),
        pytest.param(lambda: _scaled_page(two_aspects=True), _pass_rate_rows, id="scaled_two_aspects"),
        pytest.param(lambda: s3_measurement().page(), _measurement_rows, id="measurement"),
    ],
)
def test_every_rate_and_money_cell_a_code_page_prints_is_held_under_its_measure(
    page_of: Callable[[], CampaignReport],
    measures_of: Callable[[CampaignReport, list[TableBlock]], dict[tuple[str, int], tuple[str, str]]],
) -> None:
    """The code page's tables hold each rate and money cell under the measure its row states a figure of (the pass
    rate's levels, changes, pass^k and half-widths; each measure's value and interval), read off the page's readings,
    so a page-level test of one measure to one precision sees the code page's columns too, and a cell recorded under
    another row's measure fails; a figure of no measure (:data:`_NO_MEASURE_CODE_CELLS`) is left out."""
    page, forms = _page_and_forms(page_of)
    tables = [
        block.model_copy(
            update={
                "rows": [
                    {key: cell for key, cell in row.items() if (block.name, key) not in _NO_MEASURE_CODE_CELLS}
                    for row in block.rows
                ]
            }
        )
        for block in page.blocks
        if isinstance(block, TableBlock)
    ]
    checked, unmeasured = _unheld(tables, forms, by_measure=True, measures=measures_of(page, tables))
    assert checked, "a code page prints rates and money, so this must check some"
    assert not unmeasured, unmeasured


@pytest.mark.parametrize("name", _WRITTEN)
def test_every_rate_and_money_cell_a_written_page_s_tables_print_is_in_its_ledger(name: str) -> None:
    """The Full results and the options table print each rate and money figure as the page's forms recorded it."""
    held = _written_inputs()[name]
    forms = PageForms.of(held)
    look = block_look(held, fact_arms(held))
    tables = [*every_table(held, look, forms=forms), arms_table(held, look, (), forms=forms)]
    checked, missing = _unheld(tables, forms)
    assert checked, f"{name} prints a pass rate or a cost, so this must check some"
    assert not missing, missing


#: The written page's tables whose rate cells are not yet held under their measure: a gate's line is a figure key's
#: text, laid in the cell its key printed, which no builder names a measure for until the key's own text is the cell's.
_UNMEASURED_WRITTEN_TABLES = frozenset({"gate_matrix_full"})


@pytest.mark.parametrize("name", _WRITTEN)
def test_every_rate_and_money_cell_a_written_page_s_tables_print_is_held_under_its_measure(name: str) -> None:
    """The Full results and the options table hold each rate and money cell under the measure it states a figure of
    (a level, a change and its interval, a cost, a pass rate), read off the input for each row or column
    (:func:`_written_cell_measures`), so a page-level test of one measure to one precision sees the written page's rate
    and money columns too, and a cell recorded under another's measure fails; an interval's level is no measure's."""
    held = _written_inputs()[name]
    forms = PageForms.of(held)
    look = block_look(held, fact_arms(held))
    tables = [
        table
        for table in (*every_table(held, look, forms=forms), arms_table(held, look, (), forms=forms))
        if table.name not in _UNMEASURED_WRITTEN_TABLES
    ]
    _, unmeasured = _unheld(tables, forms, by_measure=True, measures=_written_cell_measures(held, tables))
    assert not unmeasured, unmeasured


def test_a_cell_recorded_under_another_measure_fails_the_written_and_code_pages_checks() -> None:
    """The written and code pages' checks read each cell's own measure, never only whether some measure holds it: on
    Kairo's written page a comparison's row, the options table's pass rate and the arms table's cost each read as
    unheld once said to state another measure's figures, and so does the measurement code page's first scorecard row."""
    held = _written_inputs()[KAIRO]
    forms = PageForms.of(held)
    look = block_look(held, fact_arms(held))
    tables = [*every_table(held, look, forms=forms), arms_table(held, look, (), forms=forms)]
    measures = _written_cell_measures(held, tables)
    for cell, printed in (
        (("contrasts", 0), "contrasts.change"),
        (("options", 0, "pass_rate"), "options.pass_rate"),
        (("arms_by_measure", 0, "measure:cost_usd"), "arms_by_measure.measure:cost_usd"),
    ):
        moved = {**measures, cell: ("another_measure", "measure")}
        _, misheld = _unheld(tables, forms, by_measure=True, measures=moved)
        assert misheld and all(line.startswith(printed) and "another_measure" in line for line in misheld), misheld

    page, forms = _page_and_forms(lambda: s3_measurement().page())
    tables = [block for block in page.blocks if isinstance(block, TableBlock) and block.name == "scorecard"]
    moved = {**_measurement_rows(page, tables), ("scorecard", 0): ("another_measure", "measure")}
    _, misheld = _unheld(tables, forms, by_measure=True, measures=moved)
    assert misheld and all(line.startswith("scorecard.") and "another_measure" in line for line in misheld), misheld


def _slice_rows(slices: list[Mapping[str, Mapping[str, object]]], arms: list[str]) -> list[str]:
    """A slice table's measure per row, as it lays them out: each slice in the input's order, each measure the slices
    hold in the order they first hold it, a row wherever an arm the table shows holds the measure there."""
    measures = list(dict.fromkeys(key for per_arm in slices for figures in per_arm.values() for key in figures))
    return [key for per_arm in slices for key in measures if any(key in per_arm.get(arm, {}) for arm in arms)]


def _written_cell_measures(
    held: ReportInput, tables: list[TableBlock]
) -> dict[tuple[str, int] | tuple[str, int, str], tuple[str, str]]:
    """Each rate and money cell's measure on a written page, read off its input in the order its tables lay their rows
    out: an arm's column by the measure it names (``measure:<key>``, ``judged:<key>``), its pass rate and its cost by
    the measures the page reads them from (:func:`~threetears.evals.analysis.report.keys.reads.pass_rate_measure`,
    :func:`~threetears.evals.analysis.report.keys.reads.cost_measure`); a comparison's row by its contrast, in the
    pipeline's order; a stratum's, a task's or a cell's row by the measure it states (a judged dimension no measure
    shares as a judged one)."""
    cost = cost_measure(held)
    named = {"pass_rate": pass_rate_measure(held), **({"cost_per_1k": cost} if cost is not None else {})}
    slices = {
        "strata": [stratum.per_arm for stratum in held.strata],
        "tasks": [task.per_arm for task in held.tasks],
        "cells": [cell.per_arm for cell in held.cells.values()],
    }
    expected: dict[tuple[str, int] | tuple[str, int, str], tuple[str, str]] = {}
    for table in tables:
        if table.name in ("arms_by_measure", "options"):
            for at, row in enumerate(table.rows):
                for column in row:
                    prefix, _, key = column.partition(":")
                    measure = (key, prefix) if prefix in ("measure", "judged") else named.get(column)
                    if measure is not None:
                        expected[(table.name, at, column)] = measure
        elif table.name == "contrasts":
            assert len(held.contrasts) == len(table.rows), table.name
            expected |= {
                (table.name, at): (one.measure, one.kind or "measure") for at, one in enumerate(held.contrasts)
            }
        elif table.name in slices:
            arms = [column.key.removeprefix("arm:") for column in table.columns if column.key.startswith("arm:")]
            rows = _slice_rows(slices[table.name], arms)
            assert len(rows) == len(table.rows), table.name
            expected |= {
                (table.name, at): (key, "judged" if key in held.judged and key not in held.measures else "measure")
                for at, key in enumerate(rows)
            }
    return expected


_ASPECT_SOURCES = ("s1_change", "s2_choice", "s3_measurement", "s4_change", "toyhost")


@functools.cache
def _aspect_pages(source: str) -> dict[str, tuple[Report, PageForms]]:
    """Each aspect's own page (code's page over its bundle), with the one forms it was built in, for one of the four
    campaign shapes or the toy host. Built inside the test, never at import: a shape's bundles stamp the clock."""
    if source == "toyhost":
        bundles = {"toyhost": toyhost_bundle()}
    else:
        shape = {
            "s1_change": s1_change,
            "s2_choice": s2_choice,
            "s3_measurement": s3_measurement,
            "s4_change": s4_change,
        }[source]()
        bundles = {f"{shape.name}/{label}": bundle for label, bundle in shape.bundles.items()}
    pages = {}
    for name, bundle in bundles.items():
        forms = PageForms.of_bundle(bundle)
        pages[name] = (build_code_only_report_in_forms(bundle, assembled_at=ASSEMBLED_AT, forms=forms), forms)
    return pages


def _aspect_row_measures(page: Report, tables: list[TableBlock]) -> dict[tuple[str, int], tuple[str, str]]:
    """Each row's measure on an aspect's page, read off its evidence in the order its tables lay their rows out: a
    contrast's row by the contrast (one under the case floor states none and has no row), a guardrail's by its check, a
    bar's by its verdict, a reading's by the heading it is printed under, and every scoreboard row's cost by the cost
    the floors read."""
    bundle = page.bundle
    surface = bundle_decision_surface(bundle)
    rulers = report_rulers(surface, bundle.measure_catalog)
    contrasts = [
        (comparison.name, comparison.reading)
        for family in bundle.multiple_comparisons.families
        for comparison in family.comparisons
        if PageForms.small_n(comparison) is None
    ]
    guardrails = bundle.guardrails
    checks = [] if guardrails is None or guardrails.withheld is not None else guardrails.checks
    by_table: dict[str, list[tuple[str, str]]] = {
        "comparisons": contrasts,
        "comparison_statistics": contrasts,
        "guardrails": [(check.name, check.reading) for check in checks],
        "bars": [(verdict.name, verdict.reading) for verdict in page.verdicts if verdict.kind == "bar"],
    }
    headings = {
        rulers.of(name, "measure").scaled(reading_heading(surface, name, declared=bundle.check_name(name))): (
            name,
            "measure",
        )
        for name in bundle.measure_catalog
    } | {judged_heading(cell.dimension): (cell.dimension, "judged") for one in surface.cells for cell in one.judged}
    expected: dict[tuple[str, int], tuple[str, str]] = {}
    for table in tables:
        if table.name in by_table:
            assert len(by_table[table.name]) == len(table.rows), table.name
            expected |= {(table.name, at): measure for at, measure in enumerate(by_table[table.name])}
        elif table.name == "scoreboard":
            expected |= {(table.name, at): (FLOOR_COST_MEASURE, "measure") for at in range(len(table.rows))}
        elif table.name == "readings":
            expected |= {(table.name, at): headings[str(row["reading"].value)] for at, row in enumerate(table.rows)}
    return expected


@pytest.mark.parametrize("source", _ASPECT_SOURCES)
def test_every_rate_and_money_cell_an_aspect_page_prints_is_in_its_ledger_under_its_measure(source: str) -> None:
    """An aspect's own page prints each rate and money figure as its forms recorded it, under the measure the cell
    states a figure of: a column rule records a cell under the measure its builder named (:meth:`PageForms.of_measure`),
    so a page-level test of one measure to one precision sees the aspect page's rate and money columns too. Each row's
    measure is read off the page's evidence, so a cell recorded under another row's measure fails."""
    pages = _aspect_pages(source)
    assert pages, f"{source} builds no aspect page"
    for name, (page, forms) in sorted(pages.items()):
        tables = [block for block in page.blocks if isinstance(block, TableBlock)]
        checked, missing = _unheld(tables, forms)
        assert checked, f"{name}: an aspect page prints rates or money, so this must check some"
        assert not missing, (name, missing)
        measures = _aspect_row_measures(page, tables)
        drawn = {table.name for table in tables} & {"comparisons", "bars", "scoreboard"}
        assert {table for table, _ in measures} >= drawn, name
        _, misheld = _unheld(tables, forms, by_measure=True, measures=measures)
        assert not misheld, (name, misheld)


def test_a_cell_recorded_under_another_measure_fails_the_aspect_page_s_check() -> None:
    """The aspect page's check reads each row's own measure, never only whether some measure holds a cell: a cell
    recorded under another measure than its row's is unheld, on a table of one cell and on a real page's first
    contrast, whose cells read as unheld once the row is said to state another measure's figures."""
    forms = PageForms.off_page()
    (row,) = forms.money_columns([{"rate": ColumnMoney(value=0.11, measure="latency", kind="measure")}])
    table = TableBlock(
        section="contrasts",
        name="t",
        title="T",
        columns=[TableColumn(key="rate", header="Rate")],
        rows=[row],
        order="as given",
        total_rows=1,
    )
    assert _unheld([table], forms, by_measure=True, measures={("t", 0): ("latency", "measure")}) == (1, [])
    _, misheld = _unheld([table], forms, by_measure=True, measures={("t", 0): ("accuracy", "measure")})
    assert misheld == ["t.rate (number): '$0.11' under ('accuracy', 'measure')"]

    page, forms = _aspect_pages("s1_change")["s1 change/billing"]
    tables = [block for block in page.blocks if isinstance(block, TableBlock)]
    measures = _aspect_row_measures(page, tables)
    moved = {**measures, ("comparisons", 0): ("another_measure", "measure")}
    _, misheld = _unheld(tables, forms, by_measure=True, measures=moved)
    assert misheld and all(line.startswith("comparisons.") and "another_measure" in line for line in misheld), misheld


def test_a_column_rule_records_a_cell_under_the_measure_its_builder_named_and_the_cell_carries_the_text() -> None:
    """A cell its builder named a figure of a measure is recorded by the money columns under that measure, with what
    the rule prints (``$0.0080`` of ``0.008``), and carries that text as its decided text; a cell named of none is
    recorded under none, as before, and carries its text too."""
    forms = PageForms.off_page()
    named = ColumnMoney(value=0.008, measure="cost_usd", kind="measure")
    laid = forms.money_columns([{"cost": named}, {"cost": ColumnMoney(value=0.11)}])
    assert [row["cost"].value_text for row in laid] == ["$0.0080", "$0.11"]
    assert [(entry.measure, entry.kind, entry.role, entry.text) for entry in forms.ledger] == [
        ("cost_usd", "measure", "column", "$0.0080"),
        ("", "", "column", "$0.11"),
    ]


def test_a_chart_of_one_measure_records_each_value_it_draws_under_that_measure() -> None:
    """A chart drawing one measure's figures records each value's text under it, a value as its level, an
    interval's ends as its interval, as the chart draws them; its reference line, a declared value, is no figure of
    it, and a chart that names no measure records nothing."""
    data = BarsChart(
        rows=[BarRow(label="a", value=62.9, lo=50.0, hi=71.4)],
        domain=(0, 100),
        fmt="pct",
        level="95%",
        reference=ChartReference(value=80.0, kind="floor"),
    )
    forms = PageForms.off_page()
    drawn = forms.chart_texts(data, measure="accuracy", kind="measure")
    (row,) = drawn.rows
    assert [(entry.measure, entry.kind, entry.role, entry.value, entry.text) for entry in forms.ledger] == [
        ("accuracy", "measure", "level", 62.9, row.value_text),
        ("accuracy", "measure", "interval", 50.0, row.lo_text),
        ("accuracy", "measure", "interval", 71.4, row.hi_text),
    ]
    unnamed = PageForms.off_page()
    assert unnamed.chart_texts(data) == drawn
    assert unnamed.ledger == ()


def test_the_money_columns_record_each_amount_as_its_column_s_form_prints_it() -> None:
    """A column holding $0.008 and $0.11 reads ``$0.0080`` and ``$0.11``; the ledger holds exactly those, each by its
    amount, and nothing for the cell of words beside them."""
    forms = PageForms.off_page()
    laid = forms.money_columns(
        [
            {"arm": TableCell(value="A"), "cost": ColumnMoney(value=0.008)},
            {"arm": TableCell(value="B"), "cost": ColumnMoney(value=0.11)},
        ]
    )
    assert [row["cost"].figure for row in laid] == ["$0.0080", "$0.11"]
    assert [(entry.measure, entry.kind, entry.role, entry.value, entry.text) for entry in forms.ledger] == [
        ("", "", "column", 0.008, "$0.0080"),
        ("", "", "column", 0.11, "$0.11"),
    ]


#: The prose keys read on the fixtures each states: a measure's level and its interval at the levels' set, a judged
#: score's change and its interval at a score's two decimals, a count per trial at its set.
_KEYS = {
    "prototype2/s2": ("measure.total_ms.beta.mean", "measure.total_ms.beta.ci"),
    "prototype5/kairo": (
        "contrast.qwen3.8-flash.__outcome__.delta",
        "contrast.qwen3.8-flash.__outcome__.interval",
        "measure.failed_reply_notices.September.mean",
    ),
}


@pytest.mark.parametrize(("name", "key"), [(name, key) for name, keys in _KEYS.items() for key in keys])
def test_a_prose_key_records_each_figure_it_prints_under_its_measure(name: str, key: str) -> None:
    """A key that prints a measure's figure at its set records each figure it prints, under that measure: the ledger
    holds the text a key prints, as it holds a table's, so a page-level test can hold the two to one text."""
    held = ReportInput.model_validate(snapshot_of(name).report_input)
    forms = PageForms.of(held)
    figure = resolve(key, held)
    assert figure.set_of is not None
    written = format_figure(figure, forms=forms)
    entries = forms.ledger
    assert entries, f"{key} printed {written!r} and recorded nothing"
    assert all(entry.text in written for entry in entries), (written, entries)
    measure, kind = figure.set_of
    assert {(entry.measure, entry.kind) for entry in entries} == {(measure, "measure" if kind is None else kind)}
    assert {entry.role for entry in entries} == (
        {"interval"} if figure.kind == "interval" else {"change"} if figure.change else {"level"}
    )


def test_a_key_s_count_is_recorded_with_its_thousands_grouped_as_it_prints() -> None:
    """A count a sentence states groups its thousands, and the ledger holds it as printed."""
    held = _written_inputs()[KAIRO]
    forms = PageForms.of(held)
    count = Figure(path="made.count", kind="number", value=4_000_000.0, scale="count")
    assert format_figure(count, forms=forms) == "4,000,000"
    assert [entry.text for entry in forms.ledger] == ["4,000,000"]


def _kairo_rate() -> str:
    """Kairo's rate measure: its send-message check, a share of trials."""
    return next(key for key in _written_inputs()[KAIRO].measures if key.startswith("goal_state:"))


#: Kairo's rate, money and pass-rate keys, each with the measure and kind its figures are held under, the role, and
#: the texts the key prints of it: a rate's level and interval in percent, money per 1,000 replies, the pass rate's
#: change in points.
_MEASURED_KEYS = [
    ("measure.{rate}.September.mean", ("{rate}", "measure"), "level", "99.7%", ["99.7"]),
    ("measure.{rate}.September.ci", ("{rate}", "measure"), "interval", "98.6 to 99.9%", ["98.6", "99.9"]),
    (
        "measure.production_replicating_cost.September.mean",
        ("production_replicating_cost", "measure"),
        "money",
        "$1.89 per 1,000 replies",
        ["$1.89"],
    ),
    (
        "measure.production_replicating_cost.September.ci",
        ("production_replicating_cost", "measure"),
        "interval",
        "$1.76 to $2.03 per 1,000 replies",
        ["$1.76", "$2.03"],
    ),
    ("contrast.qwen3.8-flash.passed.delta", (TRIAL_PASS_NAME, "trial"), "change", "-7.6", ["-7.6"]),
]


@pytest.mark.parametrize(("key", "held_under", "role", "written", "texts"), _MEASURED_KEYS)
def test_a_rate_or_money_key_records_what_it_prints_under_its_measure(
    key: str, held_under: tuple[str, str], role: str, written: str, texts: list[str]
) -> None:
    """A key on a rate, on money or on the pass rate records each figure it prints under the measure it is of, as a
    plain figure's key does, so a page-level test can hold its text to its table's."""
    rate = _kairo_rate()
    held = _written_inputs()[KAIRO]
    forms = PageForms.of(held)
    assert format_figure(resolve(key.format(rate=rate), held), forms=forms) == written
    measure, kind = held_under
    assert [(entry.measure, entry.kind, entry.role, entry.text) for entry in forms.ledger] == [
        (measure.format(rate=rate), kind, role, text) for text in texts
    ]


#: The written fixtures every key family is read on: Kairo round 2 and the five misfit inputs, then the inputs that
#: state what none of them does: r5 with one arm's unanswered calls counted, a campaign of several aspects with pairs
#: and bars (prototype 2's s2), and r2 with its gates read in each task (rung 5's).
_WRITTEN_FIXTURES = (
    "kairo round 2",
    "r1",
    "r2",
    "r3",
    "r4",
    "r5",
    "r5 with lost calls",
    "s2",
    "r2 with its gates read in each task",
)

#: The roles a table holds a figure of a measure by, as a key may print the same figure: its level, its money, its
#: change.
_FIGURE_ROLES = frozenset({"level", "money", "change"})


def _with_lost_calls(held: ReportInput) -> ReportInput:
    """The input with its second arm's candidate calls counted unanswered: one retried and three lost of 898."""
    arm = held.arms[1].id
    counted = {"candidate": Unanswered(retried=1, lost=3, calls=894, share=4 / 898)}
    rows = {**held.scoreboard, arm: held.scoreboard[arm].model_copy(update={"unanswered": counted})}
    return held.model_copy(update={"scoreboard": rows})


@functools.cache
def _fixture_inputs() -> dict[str, ReportInput]:
    snapshots = fixture_snapshots("s")
    inputs = {name: frozen_input(snapshots[name]) for name in _WRITTEN_FIXTURES[:6]}
    return inputs | {
        "r5 with lost calls": _with_lost_calls(inputs["r5"]),
        "s2": ReportInput.model_validate(snapshot_of("prototype2/s2").report_input),
        "r2 with its gates read in each task": _written_inputs()["rung 5 r2 with its gates read in each task"],
    }


@functools.cache
def _family_forms() -> tuple[str, ...]:
    """Every family's form, as the grammar lists its paths: the resolver's own families, written by the code that
    reads them (:func:`~threetears.evals.analysis.report.figure_keys.figure_key_grammar`), so a family added is walked
    with no list here to extend."""
    grammar = figure_key_grammar(forms_of=PageForms.of)
    paths = grammar.split("\nPaths:\n", 1)[1].split("\nFilters", 1)[0]
    return tuple(re.findall(r"^- `([^`]+)`:", paths, re.MULTILINE))


#: What a refusal names after the start of a path it stopped at: each id, statistic or word that may come next.
_NAMED_NEXT = re.compile(r"`([^`]*)`")

#: How a refusal names a slot a number fills (a hypothetical line, a count of cases), and the number it is read with.
_A_NUMBER = " (a number)"
_SOME_NUMBER = "1"


def _next_in_path(start: str, held: ReportInput) -> list[str]:
    """What may follow the start of a path (ending in a dot) on an input, as the resolver's refusal names it: nothing
    where no family reads the start. A refusal names a few dozen at most, so a slot offering more is walked through
    the ones it names."""
    try:
        resolve(start, held)
    except UnresolvableReference as refusal:
        stopped = f"after `{start}` comes "
        said = str(refusal)
        return _NAMED_NEXT.findall(said.split(stopped, 1)[1]) if stopped in said else []
    raise AssertionError(f"{start!r} ends in a dot, which no key does, and resolved")


@functools.cache
def _every_key(name: str, forms: tuple[str, ...]) -> tuple[str, ...]:
    """Every key the input offers in the families given, walked from each family's leading words through each of its
    slots as the resolver names what may come next there: every id the input holds, each statistic, a number where
    the slot takes one; a slot the input fills with nothing (``none in this input``) ends the walk there."""
    held = _fixture_inputs()[name]
    starts = {form.split("<", 1)[0] for form in forms}
    keys = {start for start in starts if not start.endswith(".")}
    pending = sorted(start for start in starts if start.endswith("."))
    walked: set[str] = set()
    while pending:
        start = pending.pop()
        if start in walked:
            continue
        walked.add(start)
        for named in _next_in_path(start, held):
            if named.endswith(_A_NUMBER):
                named = _SOME_NUMBER
            elif named.startswith("<"):
                continue
            keys.add(start + named)
            pending.append(f"{start}{named}.")
    return tuple(sorted(keys))


def test_the_key_walk_reaches_every_family() -> None:
    """The walk below reads a key of every family the resolver knows on some fixture, so no family's figure escapes it
    for want of a list naming it: each fixture is walked through the families no earlier one reached."""
    forms = set(_family_forms())
    unreached = set(forms)
    for name in _WRITTEN_FIXTURES:
        held = _fixture_inputs()[name]
        for key in _every_key(name, _family_forms()):
            if not unreached:
                break
            try:
                resolve(key, held)
            except UnresolvableReference:
                # NOSILENT: a key the input states no figure for reaches no family; the walk reads every combination of
                # the input's ids, and skips these.
                continue
            unreached.discard(key_form(key, held).form)
    assert len(forms) > 100, "the grammar lists every family, so this must read them"
    assert not unreached, sorted(unreached)


def _tabled(held: ReportInput) -> dict[str, set[tuple[str, str, str]]]:
    """Every figure of a measure the page's tables print, by its value (to ten significant figures, which no
    rounding a table applies touches), with the measure, kind and role each is held under."""
    forms = PageForms.of(held)
    look = block_look(held, fact_arms(held))
    every_table(held, look, forms=forms)
    arms_table(held, look, (), forms=forms)
    arms_key(held, look, forms=forms)
    held_figures: dict[str, set[tuple[str, str, str]]] = {}
    for entry in forms.ledger:
        if entry.measure and entry.role in _FIGURE_ROLES and isinstance(entry.value, float):
            held_figures.setdefault(f"{entry.value:.10g}", set()).add((entry.measure, entry.kind, entry.role))
    return held_figures


def _held_as(figure: Figure, tabled: Mapping[str, set[tuple[str, str, str]]], held: ReportInput) -> tuple[str, ...]:
    """The table cell a key's figure is the figure of, by its value, never by how either prints it: its measure, kind
    and role, or nothing where no one cell's figure is the key's.

    A figure is matched by its number, on its own scale or a percent's (``85.3`` and points read against a share's
    ``0.853``), money on the table's (a cost per 1,000 of ``1.892`` against a cost per reply of ``0.001892``), to a cell
    of the measure its path names (by its id, or a criterion's ``measure``). A key whose path names none is of a figure
    the page holds without naming it: the pass rate or the cost. (So a floor's change on its own measure, which
    ``settle.floor`` reads with no measure in its path, would be passed over; no fixture states one.) A figure in points
    is a
    difference or a spread of a rate, never its level, so it is matched to changes, as a change is; any other to
    levels and money. A zero and a count are passed over (a zero half-width reads like a measure's zero level, a count of
    cases needed like a level in percent, each the one cell of its value); so is any other whole figure but a level
    (:data:`_WHOLE_LEVELS`) or one whose key says where the tables hold it (``ledger_as``), since a declared line, a
    threshold, an adjusted p of 1 or a count of repeats reads like a whole level too. Two cells of one value under two
    measures or roles name neither."""
    value = figure.value
    if not isinstance(value, float) or value == 0 or figure.scale == "count":
        return ()
    if value.is_integer() and figure.ledger_as is None and figure.statistic not in _WHOLE_LEVELS:
        return ()
    named = figure.measure or (figure.set_of[0] if figure.set_of is not None else "")
    unnamed = {pass_rate_measure(held)[0], *(cost_measure(held) or ())[:1]}
    as_change = figure.change or figure.scale == "points"
    if figure.scale in ("points", "percent"):
        values: tuple[float, ...] = (value, value / 100)
    else:
        values = (value / _PER_COUNT.get(figure.scale, 1),)
    found = {
        cell
        for number in values
        for cell in tabled.get(f"{number:.10g}", ())
        if (cell[2] == "change") == as_change and (cell[0] == named if named else cell[0] in unnamed)
    }
    return found.pop() if len(found) == 1 else ()


#: The statistics a key reads a measure's level by, which a whole figure is still matched under (a bar's ``701`` ms,
#: an arm's ``$4.00``): its mean, a bar's or a criterion's value, and an interval's ends; and none, the statistic of a
#: key that reads a level whole by its path alone (``judged.<dim>.<arm>``).
_WHOLE_LEVELS = frozenset({"mean", "value", "low", "high", None})

#: How many of what it counts a money figure's scale reads per (a cost per 1,000 replies), which a table's cost per one
#: reply is that figure over.
_PER_COUNT = {"usd_per_1k": 1_000, "usd_per_1m": 1_000_000}

#: The families a table's cells are read from by key whose paths name no measure, each with the measure and role the
#: table holds its figure under: wherever a fixture's table holds that cell, the walk must match its key, so a family
#: whose figure no longer matches (on a scale read wrong) fails here rather than passing unchecked.
_READ_BY_KEY: dict[str, tuple[Callable[[ReportInput], tuple[str, str] | None], str]] = {
    "pass_rate": (pass_rate_measure, "level"),
    "cost_per_1k": (cost_measure, "money"),
}


@pytest.mark.parametrize("name", _WRITTEN_FIXTURES)
def test_a_key_records_a_figure_a_table_prints_under_the_measure_and_role_the_table_holds_it_by(name: str) -> None:
    """Every key of every family whose figure a table prints records it under the measure and role the table holds it
    by, never as a lone figure: an arm's pass rate is the pass rate's level wherever it is read, its cost per reply the
    page's cost's money, a reading's change the pass rate's change, a percentile its measure's level, whatever the
    key's path names. Each key is matched to the cell its figure is (:func:`_held_as`), and its text may still differ
    from the cell's until the policy flip; where it is held, which no digit reads, may not."""
    held = _fixture_inputs()[name]
    tabled = _tabled(held)
    forms = PageForms.of(held)
    checked: set[str] = set()
    misheld: list[tuple[str, str, tuple[str, ...], set[tuple[str, str, str]]]] = []
    for key in _every_key(name, _family_forms()):
        try:
            figure = resolve(key, held)
        except UnresolvableReference:
            # NOSILENT: a key the input states no figure for (an arm a measure does not read, a slice it does not
            # cut) is no figure a table prints; the walk reads every combination of the input's ids and skips these.
            continue
        if figure.kind != "number" or not (cell := _held_as(figure, tabled, held)):
            continue
        start = len(forms.ledger)
        written = format_figure(figure, forms=forms)
        recorded = {
            (entry.measure, entry.kind, entry.role) for entry in forms.ledger[start:] if entry.role != "interval"
        }
        if recorded != {cell}:
            misheld.append((key, written, cell, recorded))
        checked.add(key.split(".")[0])
    assert checked, f"{name}: its tables print figures keys read, so this must check some"
    assert not misheld, misheld
    held_cells = {cell for cells in tabled.values() for cell in cells}
    for family, (measure_of, role) in _READ_BY_KEY.items():
        measure = measure_of(held)
        if measure is not None and (*measure, role) in held_cells:
            assert family in checked, f"{name}: the table holds {family}'s {role}, and no {family} key matched it"


def test_a_bar_s_figure_records_under_its_measure_in_the_role_the_table_holds_it() -> None:
    """On s2 each arm's figure against a bar records under the bar's measure, as its level (``701 ms``) or, on a money
    measure, as money (``$4.00 per 1,000 replies``), never as a lone figure; where the page's tables print that
    measure (the turn time; s2 prints its cost under the page's cost, not the measuring spend the bar reads), the key's
    entry is the cell's."""
    held = _fixture_inputs()["s2"]
    tabled = _tabled(held)
    forms = PageForms.of(held)
    roles: set[str] = set()
    for read in held.bars:
        figure = resolve(f"aspects.{read.aspect}.bar.{read.measure}.{read.arm}.value", held)
        expected = (
            read.measure,
            "measure",
            "money" if figure.scale in ("usd", "usd_per_1k", "usd_per_1m") else "level",
        )
        start = len(forms.ledger)
        format_figure(figure, forms=forms)
        recorded = [(entry.measure, entry.kind, entry.role) for entry in forms.ledger[start:]]
        assert recorded == [expected], (read, recorded)
        if (cell := _held_as(figure, tabled, held))[:1] == (read.measure,):
            assert cell == expected, (read, cell)
        roles.add(expected[2])
    assert roles == {"level", "money"}, roles
    assert any(cell[0] == "total_ms" for cells in tabled.values() for cell in cells), "s2's tables print the turn time"


# --- the ledger and the fixed forms -------------------------------------------------------------------------------


def test_the_ledger_holds_every_text_by_measure_kind_and_role() -> None:
    held = _written_inputs()[KAIRO]
    forms = PageForms.of(held)
    contrast = next(contrast for contrast in held.contrasts if contrast.interval is not None and contrast.delta)
    assert contrast.interval is not None and contrast.delta is not None
    change = forms.change(contrast.measure, contrast.delta, kind=contrast.kind)
    interval = forms.interval(contrast.measure, *contrast.interval, kind=contrast.kind, change=True)
    count = forms.count(765)
    entries = [(entry.measure, entry.kind, entry.role, entry.value, entry.text) for entry in forms.ledger]
    assert entries == [
        (contrast.measure, contrast.kind, "change", contrast.delta, change.text),
        (contrast.measure, contrast.kind, "interval", contrast.interval, interval.text),
        ("", "", "count", 765.0, "765"),
    ]
    assert [entry.text for entry in forms.issued(contrast.measure, kind=contrast.kind, role="change")] == [change.text]
    assert len(forms.issued(contrast.measure, kind=contrast.kind)) == 2
    assert count.text == count.digits == "765"


def test_an_absent_figure_reads_as_absent_and_is_recorded() -> None:
    held = _written_inputs()[KAIRO]
    forms = PageForms.of(held)
    key = next(iter(held.measures))
    assert forms.level(key, None).text == ABSENT
    assert forms.change(key, math.nan).text == ABSENT
    assert [entry.text for entry in forms.ledger] == [ABSENT, ABSENT]


def _with_contrast(held: ReportInput, index: int, contrast: Contrast, **measures: Measure) -> ReportInput:
    return held.model_copy(
        update={
            "measures": {**held.measures, **measures},
            "contrasts": [*held.contrasts[:index], contrast, *held.contrasts[index + 1 :]],
        }
    )


def _judged_contrast(held: ReportInput) -> tuple[int, Contrast]:
    return next((i, c) for i, c in enumerate(held.contrasts) if c.kind == "judged" and c.measure not in held.measures)


def _as_a_score_measure(held: ReportInput, key: str) -> Measure:
    """A measure declared in unit ``score`` on a key that is also a 1–5 judged dimension, holding its figures."""
    return next(iter(held.measures.values())).model_copy(
        update={"unit": "score", "label": None, "per_arm": held.judged[key].per_arm}
    )


def test_a_comparison_on_a_score_that_names_no_kind_is_kept_within_the_scale_and_named_by_it() -> None:
    """A comparison on a 1–5 score stored with no kind takes its measure's one form, read as the Full results read it:
    its interval kept within the scale's reach and its name carrying the scale, in the forms and the table alike."""
    held = _written_inputs()[KAIRO]
    index, contrast = _judged_contrast(held)
    altered = _with_contrast(held, index, contrast.model_copy(update={"kind": None, "interval": (-9.0, 9.0)}))
    forms = PageForms.of(altered)
    row = _tables(altered)["contrasts"].rows[index]
    assert row["interval"].text == forms.interval(contrast.measure, -9.0, 9.0, kind=None, change=True).text
    assert row["interval"].text == "-4.00 to +4.00"
    assert row["measure"].text.endswith("(1–5)")


def test_a_comparison_that_names_its_kind_measure_is_never_read_as_a_score() -> None:
    """On a key both a measure in unit ``score`` and a 1–5 judged dimension, a comparison that names ``measure`` is
    neither kept within the scale nor named by it, while the measure's own column carries the scale."""
    held = _written_inputs()[KAIRO]
    index, contrast = _judged_contrast(held)
    scored = _as_a_score_measure(held, contrast.measure)
    named = contrast.model_copy(update={"kind": "measure", "interval": (-9.0, 9.0)})
    altered = _with_contrast(held, index, named, **{contrast.measure: scored})
    tables = _tables(altered)
    row = tables["contrasts"].rows[index]
    # Its measure's narrowest half-width is the 9 points given here, so its ends read whole (number policy, rule 2).
    assert row["interval"].text == "-9 to +9"
    assert not row["measure"].text.endswith("(1–5)")
    heads = [column.header for column in tables["arms_by_measure"].columns]
    assert any(head.endswith("(1–5)") for head in heads), heads


def test_a_measure_in_unit_score_on_a_judged_key_is_named_by_its_scale() -> None:
    """A measure declared in unit ``score`` whose key is also a 1–5 judged dimension heads its column with the scale."""
    held = _written_inputs()[KAIRO]
    key = _judged_contrast(held)[1].measure
    altered = held.model_copy(update={"measures": {**held.measures, key: _as_a_score_measure(held, key)}})
    table = _tables(altered)["arms_by_measure"]
    header = next(column.header for column in table.columns if column.key == f"measure:{key}")
    assert header.endswith("(1–5)"), header


def test_a_written_figure_that_is_no_number_reads_as_the_dash_in_its_unit() -> None:
    """A mean or a change that is not a number reads as the number rule spells it on a written page, the dash in the
    figure's unit (``— items``, ``— pts``), in the table and the forms alike."""
    held = _written_inputs()[KAIRO]
    forms = PageForms.of(held)
    key = next(k for k in held.measures if forms.measure(k).reads_as == "measured" and forms.measure(k).unit)
    unit = forms.measure(key).unit
    arm = next(iter(held.measures[key].per_arm))
    stat = held.measures[key].per_arm[arm].model_copy(update={"mean": math.nan})
    measure = held.measures[key].model_copy(update={"per_arm": {**held.measures[key].per_arm, arm: stat}})
    altered = held.model_copy(update={"measures": {**held.measures, key: measure}})
    texts = {row[f"measure:{key}"].text for row in _tables(altered)["arms_by_measure"].rows}
    assert f"— {unit}" in texts
    assert PageForms.of(altered).level(key, math.nan).text == f"— {unit}"
    index, rate = next((i, c) for i, c in enumerate(held.contrasts) if c.kind == "trial")
    changed = _with_contrast(held, index, rate.model_copy(update={"delta": math.nan}))
    assert _tables(changed)["contrasts"].rows[index]["change"].text == "— pts"


def test_a_contrast_that_names_no_kind_reads_as_the_measure_s_own_form() -> None:
    """A comparison stored with no kind is a measure's: one measure never takes two forms."""
    held = _written_inputs()[KAIRO]
    index, contrast = next((i, c) for i, c in enumerate(held.contrasts) if c.kind == "measure")
    unnamed = contrast.model_copy(update={"kind": None})
    altered = held.model_copy(update={"contrasts": [*held.contrasts[:index], unnamed, *held.contrasts[index + 1 :]]})
    forms = PageForms.of(altered)
    assert forms.measure(contrast.measure, kind=None) is forms.measure(contrast.measure, kind="measure")
    assert all(form.kind is not None for form in forms.measures)
    assert contrast.delta is not None
    forms.change(contrast.measure, contrast.delta, kind=None)
    assert [entry.kind for entry in forms.issued(contrast.measure, kind=None)] == ["measure"]


def test_a_measure_the_page_states_no_figure_of_has_no_form() -> None:
    forms = PageForms.of(_written_inputs()[KAIRO])
    with pytest.raises(KeyError):
        forms.measure("no such measure")


def test_a_printed_figure_lays_out_its_digits_and_unit_apart() -> None:
    held = _written_inputs()[KAIRO]
    forms = PageForms.of(held)
    contrast = next(contrast for contrast in held.contrasts if contrast.kind == "trial" and contrast.delta)
    assert contrast.delta is not None
    change = forms.change(contrast.measure, contrast.delta, kind=contrast.kind)
    assert change.unit == "pts"
    assert change.text == f"{change.digits} {change.unit}"


@pytest.mark.parametrize(
    ("printed", "expected"),
    [
        (lambda forms: forms.count(12000), spell_count(12000)),
        (lambda forms: forms.p(0.00008342), "< 0.001"),
        (lambda forms: forms.p(0.4996), "0.50"),
        (lambda forms: forms.effect(0.08475), "0.08"),
        (lambda forms: forms.declared(-15.0, "pts"), "-15 pts"),
        (lambda forms: forms.share(0.374), "37%"),
        (lambda forms: forms.multiple(21.05), spell_multiple(21.05)),
        (lambda forms: forms.money_total(1208.0), spell_dollars(1208.0)),
        (lambda forms: forms.money_total(-1.2), spell_dollars(-1.2)),
        (lambda forms: forms.confidence("99.69%"), "99.69%"),
    ],
)
def test_a_figure_that_is_no_measurement_takes_its_fixed_form(
    printed: Callable[[PageForms], Printed], expected: str
) -> None:
    forms = PageForms("written", {})
    issued = printed(forms)
    assert issued.text == expected
    assert forms.ledger[-1].text == expected and forms.ledger[-1].measure == ""
    if issued.unit == "usd":
        assert "$" not in issued.digits and issued.text.replace("$", "", 1) == issued.digits


@pytest.mark.parametrize(
    ("printed", "expected"),
    [
        (lambda forms: forms.lone(2.737), format_number(2.737)),
        (lambda forms: forms.lone(-0.3, signed=True), format_signed(-0.3)),
        (lambda forms: forms.lone(0.0016, money=True), format_number(0.0016, prefix="$")),
        (lambda forms: forms.reading(0.7349), "0.73"),
        (lambda forms: forms.echoed(1e-07), "1e-07"),
        (lambda forms: forms.hits(0.8333, 90), "75"),
    ],
)
def test_an_off_page_figure_keeps_today_s_form_and_is_held_under_no_measure_unless_named(
    printed: Callable[[PageForms], Printed], expected: str
) -> None:
    forms = PageForms.off_page()
    assert printed(forms).text == expected
    assert (forms.ledger[-1].measure, forms.ledger[-1].kind) == ("", "")


def test_an_off_page_figure_of_a_measure_is_held_under_that_measure_and_kind() -> None:
    forms = PageForms.off_page()
    forms.lone(4.2, measure="clarity", kind="judged")
    forms.hits(0.5, 10, measure="exact_match")
    assert [entry.role for entry in forms.issued("clarity", kind="judged")] == ["lone"]
    # A measure named with no kind is a measure's own, as every page's form reads it.
    assert [entry.text for entry in forms.issued("exact_match")] == ["5"]


def test_the_lone_figure_rule_spells_no_signed_money() -> None:
    with pytest.raises(ValueError, match="no signed money"):
        PageForms.off_page().lone(-1.2, signed=True, money=True)


def test_a_written_page_s_forms_carry_the_input_they_were_built_from() -> None:
    held = _written_inputs()[KAIRO]
    other = held.model_copy()
    forms = PageForms.of(held)
    assert forms.report_input is held
    assert forms.checked(held) is forms
    with pytest.raises(ValueError, match="not built from"):
        forms.checked(other)
    with pytest.raises(ValueError, match="hold no report input"):
        _ = PageForms.off_page().report_input


def test_prose_reads_the_input_its_forms_carry() -> None:
    held = _written_inputs()[KAIRO]
    fewer = held.model_copy(update={"arms": held.arms[:-1]})
    assert render("{count.arms}", {}, forms=PageForms.of(held)) == str(len(held.arms))
    assert render("{count.arms}", {}, forms=PageForms.of(fewer)) == str(len(held.arms) - 1)


def test_a_reader_handed_an_input_and_another_input_s_forms_refuses_them() -> None:
    held = _written_inputs()[KAIRO]
    with pytest.raises(ValueError, match="not built from"):
        cite("count.arms", held, forms=PageForms.of(held.model_copy()))


def test_a_view_of_the_page_s_input_prints_into_the_page_s_ledger() -> None:
    held = _written_inputs()[KAIRO]
    forms = PageForms.of(held)
    view = forms.over(held.model_copy())
    assert view.report_input is not held
    view.count(3)
    assert [entry.role for entry in forms.ledger] == ["count"]


def test_a_view_that_is_no_view_of_the_page_s_input_is_refused() -> None:
    """A configs chart's grid is a view of the page's input; an input that only reads alike (validated afresh) or one
    with another field changed is another input, and the page's forms refuse to print over it."""
    held = _written_inputs()[KAIRO]
    forms = PageForms.of(held)
    grid = held.model_copy(update={"factors": [], "configs": []})
    assert forms.over(grid).report_input is grid
    with pytest.raises(ValueError, match="a view of a report input they were not built from"):
        forms.over(ReportInput.model_validate(held.model_dump(by_alias=True)))
    with pytest.raises(ValueError, match="a view of a report input they were not built from"):
        forms.over(held.model_copy(update={"arms": held.arms[:-1]}))


#: Each table or chart a written page builds from an input and its forms, given forms built from another input.
_HANDED_APART: dict[str, Callable[[ReportInput, PageForms], object]] = {
    "arms_table": lambda held, forms: arms_table(held, block_look(held, fact_arms(held)), (), forms=forms),
    "pass_rate_cell": lambda held, forms: pass_rate_cell(held, held.arms[0].id, forms=forms),
    "arms_key": lambda held, forms: arms_key(held, block_look(held, fact_arms(held)), forms=forms),
    "every_table": lambda held, forms: every_table(held, block_look(held, fact_arms(held)), forms=forms),
    "complete_chart": lambda held, forms: complete_chart(
        PlannedChart.model_validate(
            {
                "kind": "bars",
                "title": "",
                "query": {
                    "source": "measures",
                    "aspect": "",
                    "measures": ["production_replicating_cost"],
                    "arms": [],
                    "strata": [],
                    "limit": 0,
                    "transform": "none",
                    "small_multiples_by": "none",
                },
                "reference": [],
                "emphasis": "",
            }
        ),
        held,
        "block",
        forms=forms,
    ),
}


@pytest.mark.parametrize("build", sorted(_HANDED_APART))
def test_a_table_or_chart_handed_another_input_s_forms_refuses_them(build: str) -> None:
    """Each prints the input's figures in the page's forms, so forms built from another input are refused, never used:
    one input's keys would read in another's sets."""
    held = _written_inputs()[KAIRO]
    _HANDED_APART[build](held, PageForms.of(held))
    with pytest.raises(ValueError, match="not built from"):
        _HANDED_APART[build](held, PageForms.of(held.model_copy()))


def test_an_aspect_page_s_forms_carry_their_bundle() -> None:
    shape = s3_measurement()
    first, second = list(shape.bundles.values())[:2]
    forms = PageForms.of_bundle(first)
    assert forms.bundle is first
    with pytest.raises(ValueError, match="not built from"):
        contrast_evidence(second, forms=forms)
    with pytest.raises(ValueError, match="hold no aspect's bundle"):
        _ = PageForms.of(_written_inputs()[KAIRO]).bundle


def test_an_aspect_page_built_in_handed_forms_reads_as_one_built_in_its_own() -> None:
    """A caller that reads an aspect page's verdict sentences apart (a campaign's page) builds the page's forms once
    and hands them in: the same bundle's forms lay out the page its own would, and another bundle's are refused, so
    no page prints in forms decided over other evidence."""
    shape = s3_measurement()
    first, second = list(shape.bundles.values())[:2]
    own = build_code_only_report(first, assembled_at="2026-11-09T00:00:00Z")
    handed = build_code_only_report_in_forms(
        first, assembled_at="2026-11-09T00:00:00Z", forms=PageForms.of_bundle(first)
    )
    assert handed == own
    with pytest.raises(ValueError, match="not built from"):
        build_code_only_report_in_forms(first, assembled_at="2026-11-09T00:00:00Z", forms=PageForms.of_bundle(second))
