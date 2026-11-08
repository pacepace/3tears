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
from threetears.evals.analysis.report.forms import Issued, PageForms, Printed
from threetears.evals.analysis.report.model import CampaignReport, Report, ReportBlock
from threetears.evals.analysis.report.serialize_md import campaign_report_markdown
from threetears.evals.analysis.report.parts.charting import interval_level
from threetears.evals.analysis.report.parts.full_results import every_table
from threetears.evals.analysis.report.parts.look import figure_text
from threetears.evals.analysis.report_arms import fact_arms
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.harness import read_writer_cases
from threetears.evals.kernel.numbers import ABSENT, format_number, spell_count, spell_dollars, spell_multiple

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
    return {table.name: table for table in every_table(held, block_look(held, fact_arms(held)))}


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


def test_a_level_drawn_in_percent_is_held_as_computed_and_alone_keeps_today_s_spelling() -> None:
    """A baseline in percent is rounded as computed (``6.25`` held at ``6.2``, half to even), and stated alone in words
    it is spelled half away from zero (``6.3%``): today's two spellings, relocated, until the policy flip."""
    forms = _code_forms()
    held = forms.drawn_level("passed", 6.25, kind="trial")
    alone = forms.drawn_level("passed", 6.25, kind="trial", alone=True)
    assert (held.held, held.decimals, held.text, held.unit) == (6.2, 1, "6.2%", "%")
    assert (alone.text, alone.held) == ("6.3%", 6.2)
    assert forms.drawn_level("passed", None, kind="trial").held is None


def test_a_code_page_s_change_and_its_half_width_are_held_at_a_point_s_decimal() -> None:
    forms = _code_forms()
    change = forms.change("passed", 2.46, kind="trial")
    assert (change.held, change.decimals, change.text) == (2.5, 1, "+2.5 pts")
    width = forms.half_width("passed", 3.04, kind="trial")
    assert (width.held, width.text, width.unit) == (3.0, "3.0 pts", "pts")
    assert [entry.role for entry in forms.ledger] == ["change", "half_width"]


def test_an_interval_sized_in_a_sentence_reads_each_end_unsigned_at_one_precision() -> None:
    forms = _code_forms()
    sized = forms.sizes("passed", -3.46, 7.9, kind="trial")
    assert (sized.low_text, sized.high_text) == ("3.5", "7.9")
    # An end a point's decimal would print as zero takes the decimals that show it, and the other end takes them too.
    finer = forms.sizes("passed", -0.03, 4.2, kind="trial")
    assert (finer.low_text, finer.high_text) == ("0.03", "4.20")


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
    cell = TableCell(value=level.held, unit=level.unit, decimals=level.decimals)
    assert level.text == cell.figure and level.held == form.ruler.drawn(reading.value)
    assert interval.text == f"[{cell.amount(form.ruler.drawn(reading.interval[0]))}, " + (
        f"{cell.amount(form.ruler.drawn(reading.interval[1]))}]"
    )
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
    [([2.737, 0.0], 1), ([3.0, 2.0], 1), ([99.2345, 100.0], 0), ([0.0, 0.0], None), ([None], None)],
)
def test_a_column_of_measured_figures_reads_at_its_set_s_decimals(
    figures: list[float | None], decimals: int | None
) -> None:
    """A measure that is neither a rate nor money reads at its column's set's decimals, over each arm's one figure:
    ``2.737`` beside ``0`` at one, ``99.2345`` beside ``100`` at none; a column of zeros or of no figure at all is
    left to the number rule."""
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
    assert (form.reads_as, form.level_decimals) == ("measured", decimals)


#: The fixed-form figures a code page prints only under a head naming its unit, ``(%)`` or ``(pts)``, whose bare
#: figures ``blocks.percent_columns`` respells by the unit's rule after the cell is built (``bare_places``): a
#: half-width (``Half-width (pts)``), a pass^k (``Pass^k there (%)``) and the between-cases share (``Between cases
#: (%)``). Until C3e relocates the rule, their ledger texts may read otherwise on the page.
_RESPELLED_ROLES = frozenset({"half_width", "pass_k", "share"})


def _respelled(forms: PageForms, entry: Issued) -> bool:
    """Whether ``blocks.percent_columns`` respells an issued figure on a code page after its cell is built, by its
    measure and role, until C3e:

    - :data:`_RESPELLED_ROLES`, each under a head naming its unit;
    - a rate's interval: a pass rate's under ``Interval (pts)``, respelled at a point's decimal, and a measure's
      ``[76.4%, 99.1%]`` in an interval column, whose percents the rule sets at one precision (``percent_places``);
    - a measurement's rate level (kind ``measure``): bare under ``Accuracy (%)`` in its matrix, or a percent in a value
      column, respelled alike.

    Never excluded: a change's pass rate level and change (kind ``trial``), whose cells carry their decimals, which
    the rule keeps; and an interval's confidence level, which only sentences state and the rule never reads."""
    if entry.role in _RESPELLED_ROLES:
        return True
    if not entry.measure or forms.measure(entry.measure, kind=entry.kind).reads_as != "rate":
        return False
    return entry.role == "interval" or (entry.role == "level" and entry.kind == "measure")


def _ledger_and_page(page_of: Callable[[], CampaignReport]) -> tuple[PageForms, str]:
    """The one forms a code page was built with, and its markdown."""
    built: list[PageForms] = []
    real = PageForms.of_campaign

    def capture(readings: CampaignReadings, reports: Mapping[str, Report]) -> PageForms:
        forms = real(readings, reports)
        built.append(forms)
        return forms

    with mock.patch.object(PageForms, "of_campaign", side_effect=capture):
        page = page_of()
    (forms,) = built
    return forms, campaign_report_markdown(page)


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
    issues nothing. Excluded until C3e, by measure and role: a figure the column rule still respells after its cell is
    built (:func:`_respelled`), whose digits the rule may change."""
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
    assert forms.drawn_held("passed", 86.65, kind="trial") == 86.7
    assert forms.ledger == before


@pytest.mark.parametrize("two_aspects", [False, True], ids=["scaled", "scaled_two_aspects"])
def test_the_control_cell_issues_only_the_level_it_prints(two_aspects: bool) -> None:
    """On a change's matrix each pass rate level issued is one a cell prints: the control's (its pooled rate and its
    candidates' baselines compared, not issued), each candidate's level beside its change, and each baseline a cell
    names in words; on the row across the aspects, the control's level across them in place of its own figure."""
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
    levels = forms.issued("passed", kind="trial", role="level")
    assert len(levels) == len(printed) + named
    controls = [cell for cell in printed if cell.level is None]
    assert len(controls) == len(rows) and {cell.figure for cell in controls} <= {entry.text for entry in levels}


def _scaled_page(*, two_aspects: bool = False) -> CampaignReport:
    corpus = hot_path_corpus(scale=1, two_aspects=two_aspects)
    return campaign_report(corpus.host, corpus.campaign.id, HOT_PATH_SCOPE)


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
