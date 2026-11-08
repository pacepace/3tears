"""The decided-figure model (number forms C2): one form per measure per page, reproducing what the page prints today.

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

The two column rules the tables apply after a cell is built (whole percents per column, a money column's own form)
stay the tables' until they are relocated; a rate's or money's form here is what the cell is built from.
"""

from __future__ import annotations

import functools
import math
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from unittest import mock

import pytest

from threetears.evals.analysis import campaign_report, service
from threetears.evals.analysis.analyst.snapshot import frozen_input
from threetears.evals.analysis.campaign_readings import CampaignReadings
from threetears.evals.analysis.report.block_spec import block_look
from threetears.evals.analysis.report.blocks import TableBlock, TableCell, bare_columns, percent_columns
from threetears.evals.analysis.report.forms import PageForms, Printed, measured_decimals
from threetears.evals.analysis.report.model import CampaignReport, Report, ReportBlock
from threetears.evals.analysis.report.parts.charting import interval_level
from threetears.evals.analysis.report.parts.full_results import every_table
from threetears.evals.analysis.report.parts.look import figure_text
from threetears.evals.analysis.report_arms import fact_arms
from threetears.evals.analysis.bundle.schema import TRIAL_PASS_NAME
from threetears.evals.analysis.report_input import Contrast, Measure, ReportInput
from threetears.evals.analysis.report_margins import aspect_board
from threetears.evals.harness import read_writer_cases
from threetears.evals.kernel.numbers import (
    ABSENT,
    format_number,
    format_signed,
    spell_count,
    spell_dollars,
    spell_multiple,
)

from packages.evals.tests.analyst_support import KAIRO, fixture_snapshots
from packages.evals.tests.example_loader import EXAMPLES
from packages.evals.tests.hot_path_support import HOT_PATH_SCOPE, hot_path_corpus
from packages.evals.tests.report_shapes_support import s3_measurement


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
def test_a_rate_column_reads_as_its_levels_through_the_table_s_column_rule(name: str) -> None:
    """A rate's level is the figure its cell is built from; the table's column rule (whole percents unless two would
    read alike) then spells the column, so the column rebuilt from the forms reads as printed."""
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
    redone = percent_columns(rebuilt, bare=bare_columns(list(table.columns)))
    for printed, again in zip(table.rows, redone, strict=True):
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


def test_a_level_alone_reads_by_the_lone_figure_rule_as_a_chart_s_note_does() -> None:
    """Today a level printed by itself (a chart's note) keeps the lone figure rule, not its set's decimals."""
    held = _written_inputs()[KAIRO]
    forms = PageForms.of(held)
    differing = 0
    for form in forms.measures:
        if form.reads_as != "measured" or form.level_decimals is None or form.levels is None:
            continue
        for mean in held.measures[form.key].per_arm.values():
            if mean.mean is None:
                continue
            alone = forms.level(form.key, float(mean.mean), alone=True).text
            assert alone == figure_text(float(mean.mean) * form.factor, form.levels)
            differing += alone != forms.level(form.key, float(mean.mean)).text
    assert differing, "Kairo must hold a level whose set's decimals differ from the lone rule for this to test anything"


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
    with mock.patch.object(service, "build_campaign_report", wraps=service.build_campaign_report) as built:
        page = campaign_report(corpus.host, corpus.campaign.id, HOT_PATH_SCOPE)
    (call,) = built.call_args_list
    _campaign, readings, reports = call.args
    return _CodePage(page, readings, reports)


def _named(blocks: list[ReportBlock], name: str) -> TableBlock:
    (table,) = [block for block in blocks if isinstance(block, TableBlock) and block.name == name]
    return table


def test_the_scaled_campaign_s_pass_rate_reads_as_its_scorecard_prints_it() -> None:
    """Each candidate's change in points at a point's one decimal, its interval in brackets, and the level it lands
    at in percent to a decimal: ``23.0% · +2.5 pts`` and ``[-3.5, 7.9]``."""
    scaled = _scaled()
    forms = PageForms.of_campaign(scaled.readings, scaled.reports)
    matrix = _named(scaled.page.blocks, "scorecard")
    aspects = _named(scaled.page.blocks, "aspects")
    (row,) = matrix.rows
    changes = [cell for key, cell in row.items() if key.startswith("arm:") and cell.level is not None]
    assert len(changes) == len(scaled.readings.readings) == len(aspects.rows)
    rebuilt = []
    for reading, cell, line in zip(scaled.readings.readings, changes, aspects.rows, strict=True):
        assert cell.low is not None and cell.high is not None and cell.level is not None
        change = forms.change("passed", reading.change, kind="trial")
        interval = forms.interval("passed", cell.low, cell.high, kind="trial", change=True)
        level = forms.level("passed", cell.level / 100, kind="trial")
        assert cell.figure == f"{level.text} · {change.text}"
        assert (change.digits, change.unit) == (change.text.removesuffix(" pts"), "pts")
        assert interval.digits == interval.text == f"[{interval.low_digits}, {interval.high_digits}]"
        rebuilt.append(
            {
                **line,
                "change": line["change"].model_copy(
                    update={"value": change.text, "unit": "", "decimals": None, "signed": False}
                ),
                "interval": line["interval"].model_copy(update={"value": interval.text}),
            }
        )
    # The interval column's head names its unit, (pts), so the page's column rule writes its ends at a point's decimal.
    redone = percent_columns(rebuilt, bare=bare_columns(list(aspects.columns)))
    for printed, again in zip(aspects.rows, redone, strict=True):
        assert (again["change"].text, again["interval"].text) == (printed["change"].text, printed["interval"].text)


def test_a_measurement_s_scorecard_reads_as_its_forms() -> None:
    """A rate's levels and intervals through the page's column rule, and money on its ruler in its one form."""
    shape = s3_measurement()
    page = shape.page()
    forms = PageForms.of_campaign(shape.readings, shape.reports)
    table = _named(page.blocks, "scorecard")
    rebuilt = [dict(row) for row in table.rows]
    readings = list(shape.readings.measures)
    assert len(readings) == len(table.rows)
    kinds: set[str] = set()
    for index, reading in enumerate(readings):
        assert reading.value is not None and reading.interval is not None
        form = forms.measure(reading.measure)
        kinds.add(form.reads_as)
        level = forms.level(reading.measure, reading.value).text
        interval = forms.interval(reading.measure, *reading.interval).text
        if form.reads_as == "money":
            assert table.rows[index]["value"].text == level
            assert table.rows[index]["interval"].text == interval
            printed = forms.level(reading.measure, reading.value)
            ends = forms.interval(reading.measure, *reading.interval)
            # Money's parts lay out apart: its "$" is the unit's, never the digits'.
            assert (printed.unit, printed.text) == ("usd", f"${printed.digits}") and "$" not in printed.digits
            assert ends.digits == f"[{ends.low_digits}, {ends.high_digits}]" and "$" not in ends.digits
        else:
            value = table.rows[index]["value"]
            rebuilt[index]["value"] = value.model_copy(update={"value": level, "unit": "", "decimals": None})
            rebuilt[index]["interval"] = table.rows[index]["interval"].model_copy(update={"value": interval})
    assert kinds == {"rate", "money"}
    redone = percent_columns(rebuilt, bare=bare_columns(list(table.columns)))
    for printed, again in zip(table.rows, redone, strict=True):
        assert again["value"].text == printed["value"].text
        assert again["interval"].text == printed["interval"].text


@pytest.mark.parametrize(
    ("figures", "decimals"), [([2.737, 0.0], 1), ([3.0, 2.0], 1), ([99.2345, 100.0], 0), ([0.0, 0.0], None), ([], None)]
)
def test_a_column_of_measured_figures_reads_at_its_set_s_decimals(figures: list[float], decimals: int | None) -> None:
    assert measured_decimals(figures) == decimals


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
    assert row["interval"].text == "-9.00 to +9.00"
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
        (lambda forms: forms.p(0.00008342), format_number(0.00008342)),
        (lambda forms: forms.effect(0.08475), format_number(0.08475)),
        (lambda forms: forms.declared(-15.0, "pts"), "-15 pts"),
        (lambda forms: forms.share(0.374), "37%"),
        (lambda forms: forms.multiple(21.05), spell_multiple(21.05)),
        (lambda forms: forms.money_total(1208.0), spell_dollars(1208.0)),
        (lambda forms: forms.money_total(-1.2), spell_dollars(-1.2)),
        (lambda forms: forms.confidence("99.69%"), interval_level("99.69%", None)),
    ],
)
def test_a_figure_that_is_no_measurement_keeps_today_s_fixed_form(
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
