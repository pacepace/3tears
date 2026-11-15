"""A table cell holding a number carries the text its page decided, and nothing else spells it.

One module decides every figure's form (number rule 5), so a cell never spells its own number: a cell built with a
number and no ``value_text``, or with an interval and no texts for its ends, is refused, naming the cell. A glyph's
title and the line under its table print those texts. Money writes ``$`` on each end of an interval, the stated
exception to writing a unit once. An outcome bar's note prints the share that passed in the pass rate's one form, so
the note and the scorecard read one rate alike.
"""

from __future__ import annotations

import re

import pytest

from packages.evals.tests.campaign_page_support import kairo_page
from packages.evals.tests.campaign_support import ASSEMBLED_AT
from packages.evals.tests.g2_support import CASES, g2_bundle, queue_case
from packages.evals.tests.off_loop import off_loop
from threetears.evals import Eval
from threetears.evals.analysis.bundle.schema import TRIAL_PASS_NAME
from threetears.evals.analysis.report.build import build_code_only_report_in_forms
from threetears.evals.analysis.report.figures import ContrastPlaces, MeasureRuler
from threetears.evals.analysis.quantities import MONEY, PERCENT, POINTS
from threetears.evals.analysis.report.blocks import ChartBlock, TableBlock, TableCell
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.kernel.numbers import MoneyForm


def test_a_number_without_its_decided_text_is_refused_naming_the_cell() -> None:
    with pytest.raises(ValueError, match=r"the cell holding 0\.853 in '%' holds a number with no decided text"):
        TableCell(value=0.853, unit=PERCENT)


def test_an_interval_without_its_ends_texts_is_refused_and_ends_texts_without_one_too() -> None:
    with pytest.raises(ValueError, match="has an interval with no decided text for its ends"):
        TableCell(value=85.0, value_text="85%", unit=PERCENT, low=80.0, high=89.0)
    with pytest.raises(ValueError, match="carries an interval's text with no interval"):
        TableCell(value=85.0, value_text="85%", unit=PERCENT, low_text="80", high_text="89")


def test_words_a_boolean_and_nothing_print_as_they_are() -> None:
    assert [TableCell(value=value).text for value in ("v2", True, None)] == ["v2", "yes", "—"]


def test_a_cell_has_no_field_a_serializer_could_spell_a_number_from() -> None:
    assert {"decimals", "signed", "per_thousand"}.isdisjoint(TableCell.model_fields)


def test_a_copy_that_changes_the_number_carries_its_new_text_or_is_refused() -> None:
    cell = TableCell(value=85.0, unit=PERCENT, value_text="85%")
    assert cell.model_copy(update={"value": 86.0, "value_text": "86%"}).text == "86%"
    with pytest.raises(ValueError, match="is copied with no decided text"):
        cell.model_copy(update={"value": 86.0})
    with pytest.raises(ValueError, match="is copied with an interval but no decided text for both its ends"):
        cell.model_copy(update={"low": 80.0, "high": 89.0})
    # A copy that changes nothing the number is spelled from keeps its text.
    assert cell.model_copy(update={"flag": "765 of 897"}).text == "85% (765 of 897)"


def test_a_cell_is_frozen_so_its_text_never_outlives_its_value() -> None:
    cell = TableCell(value=85.0, unit=PERCENT, value_text="85%")
    with pytest.raises(ValueError, match="frozen"):
        cell.value = 86.0  # type: ignore[misc]


def test_a_copy_that_moves_one_end_carries_both_ends_texts() -> None:
    cell = TableCell(value=85.0, unit=PERCENT, value_text="85%", low=80.0, high=89.0, low_text="80", high_text="89")
    with pytest.raises(ValueError, match="no decided text for both its ends"):
        cell.model_copy(update={"low": 81.0, "low_text": "81"})
    moved = cell.model_copy(update={"low": 81.0, "low_text": "81", "high_text": "89"})
    assert (moved.low_text, moved.high_text) == ("81", "89")


def test_a_level_carries_its_text_and_the_cell_opens_with_it() -> None:
    with pytest.raises(ValueError, match="sets its level and the level's text apart"):
        TableCell(value=-8.0, unit=POINTS, level=78.0, value_text="78% · -8 pts")
    with pytest.raises(ValueError, match="does not open with the level it lands at"):
        TableCell(value=-8.0, unit=POINTS, level=78.0, level_text="77%", value_text="78% · -8 pts")


def test_money_writes_its_sign_on_every_end_and_any_other_unit_once() -> None:
    forms = PageForms.off_page()
    money = forms.money_cell(MoneyForm(decimals=0), 20.0, (16.0, 25.0))
    assert (money.text, forms.interval_alone(money).text) == ("$20", "$16 to $25")
    # A count reads whole and grouped, as every count does.
    assert forms.count_cell(1234).text == "1,234" and forms.count_cell(2.5).text == "2.5"
    rate = TableCell(value=85.0, unit=PERCENT, value_text="85%", low=80.0, high=89.0, low_text="80", high_text="89")
    change = TableCell(
        value=-8.0, unit=POINTS, value_text="-8 pts", low=-13.0, high=-2.0, low_text="-13", high_text="-2"
    )
    assert forms.interval_alone(rate).text == "80 to 89%"
    assert forms.interval_alone(change).text == "-13 to -2 pts"


def test_a_glyph_states_its_own_cell_s_texts_and_a_change_without_the_level_it_lands_at() -> None:
    forms = PageForms.off_page()
    cell = TableCell(
        value=-8.0,
        unit=POINTS,
        level=78.0,
        level_text="78%",
        value_text="78% · -8 pts",
        low=-11.0,
        high=-3.0,
        low_text="-11",
        high_text="-3",
        margin=15.0,
        zero_line=True,
    )
    assert forms.glyph_said(cell, POINTS).text == "-8 pts, interval -11 to -3, margin ±15 pts"
    bare = TableCell(
        value=-1.0, value_text="-1.0", low=-3.6, high=1.6, low_text="-3.6", high_text="+1.6", zero_line=True
    )
    assert forms.glyph_said(bare, POINTS).text == "-1.0 pts, interval -3.6 to +1.6"


def test_a_scale_line_states_its_cells_ends_with_the_unit_once() -> None:
    forms = PageForms.off_page()
    cells = [
        TableCell(value=38.0, unit=PERCENT, value_text="38%", low=23.8, high=51.2, low_text="24", high_text="51"),
        TableCell(value=80.0, unit=PERCENT, value_text="80%", low=76.0, high=84.0, low_text="76", high_text="84"),
    ]
    assert forms.scale_span(cells, PERCENT, 23.8, 84.0).text == "24 to 84%"
    money = [TableCell(value=2.0, unit=MONEY, value_text="$2", low=1.0, high=3.0, low_text="$1", high_text="$3")]
    assert forms.scale_span(money, MONEY, 1.0, 3.0).text == "$1 to $3"


#: An outcome bar's note: the share that passed, then of how many.
_NOTE = re.compile(r"^(?P<share>\S+%) passed · (?P<passed>\d+) of (?P<trials>\d+)$")


def test_the_outcome_note_and_the_scorecard_print_one_rate_alike() -> None:
    """Kairo's control passed 765 of 897 trials: its fold's outcome note and the scorecard's cell both read ``85.3%``,
    the pass rate's one precision on the campaign (two significant figures of its narrowest interval's width, 7.0
    points), as the service builds the folds in the campaign's forms."""
    page = kairo_page()
    scorecard = [block for block in page.blocks if isinstance(block, TableBlock) and block.section == "scorecard"]
    stated = {
        (match["passed"], match["trials"]): cell.figure
        for block in scorecard
        for row in block.rows
        for cell in row.values()
        if cell.flag is not None and (match := re.match(r"(?P<passed>\d+) of (?P<trials>\d+) trials", cell.flag))
    }
    notes = []
    for fold in page.aspects:
        for block in fold.report.blocks:
            if isinstance(block, ChartBlock) and block.chart == "outcome_bar":
                drawn = block.kit_chart()
                assert drawn is not None
                rows = drawn[1]["rows"]
                assert isinstance(rows, list)
                notes += [str(row["note"]) for row in rows if isinstance(row, dict)]
    said = [match for note in notes if (match := _NOTE.match(note)) is not None]
    alike = [
        (match["share"], stated[match["passed"], match["trials"]])
        for match in said
        if (match["passed"], match["trials"]) in stated
    ]
    assert alike, f"no outcome note shares a count with a scorecard cell: {notes} against {stated}"
    assert all(share == figure for share, figure in alike), alike
    assert ("85.3%", "85.3%") in alike


def test_a_copy_keeps_its_text_opening_with_its_level() -> None:
    cell = TableCell(value=-8.0, unit=POINTS, level=78.0, level_text="78%", value_text="78% · -8 pts")
    with pytest.raises(ValueError, match="does not open with the level it lands at"):
        cell.model_copy(update={"level_text": "77%"})
    with pytest.raises(ValueError, match="does not open with the level it lands at"):
        cell.model_copy(update={"value_text": "-8 pts"})
    moved = cell.model_copy(update={"level": 77.0, "level_text": "77%", "value_text": "77% · -8 pts"})
    assert moved.text == "77% · -8 pts"


def test_a_money_change_s_interval_writes_its_sign_on_each_end_and_no_unit_after() -> None:
    """Money's ``$`` is a prefix, so a change's interval no money form decides writes it on each end, never a trailing
    ``usd`` (number rule 3's one exception)."""
    forms = PageForms.off_page()
    money = MeasureRuler(1.0, MONEY)
    assert forms.ruled_change_interval(money, -0.01, 0.02).text == "-$0.01 to +$0.02"
    places = ContrastPlaces(level=None, change=2)
    assert forms.ruled_change_interval(money, -0.01, 0.02, places=places).text == "-$0.01 to +$0.02"


async def test_an_outcome_note_counting_lost_trials_is_a_share_not_the_pass_rate() -> None:
    """An arm that lost trials passed a share of every trial its bar draws, which is not its pass rate (passes of its
    usable trials): the note prints it at the pass rate's digits but records it as a share, so the ledger's pass rate
    reads one rate alike."""
    ev = Eval(CASES, {"v4.1-flash": queue_case, "v4-flash": queue_case}, ["label"], k=3, name="queue")
    bundle = off_loop(g2_bundle, await ev.run(store=":memory:", scope="quick"))
    forms = PageForms.of_bundle(bundle)
    report = build_code_only_report_in_forms(bundle, assembled_at=ASSEMBLED_AT, forms=forms)
    (bars,) = [block for block in report.blocks if isinstance(block, ChartBlock) and block.chart == "outcome_bar"]
    lost = [bar for bar in bars.bars if any(segment.state == "lost" for segment in bar.segments)]
    assert lost, "the fixture loses trials"
    shares = {entry.text for entry in forms.issued(TRIAL_PASS_NAME, kind="trial", role="share")}
    assert {bar.passed_text for bar in lost} <= shares
