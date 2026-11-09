"""Number forms: one value reads one way, and a set of figures reads at one precision (visual pass, chunk V5).

The visual pass found a page printing ``-6 pts`` beside ``3.6 pts``, an interval ``[-6.9, 11] pts``, judged scores at
four significant figures with no scale (``0.0686``, ``[-0.06693, 0.2041]``), a count's interval running below zero, an
interval ``[-2.573, 2.573]`` on two arms that every case scored 5, and a level of ``99.69%``. The owner's rulings of
2026-11-08 (docs/design.md, Presentation rulings) set the forms held here:

- a judged score reads at two decimals, its scale named beside it (``(1–5)``);
- any other measured figure (a count per trial, a time) at two significant figures of its set's least figure but zero;
- a change at its unit's precision, signed, a zero too (``0.0 pts``), never ``+0.0``;
- a set (an interval's two ends, a column, a chart's labels) keeps one precision: ``-29.0 to +57.6``, ``[-6.9, 11.0]``,
  and an end that is not zero never reads as one (``-0.03``, never ``-0.0``); a lone figure keeps the number rule;
- an interval never runs past what its measure can take, and two arms that did not vary draw none;
- a family-corrected level reads ``99.7% (95% across 16 comparisons)``, never ``99.69%``.
"""

from __future__ import annotations

import asyncio
import html
import re
from collections.abc import Iterator

import pytest

from threetears.evals import Answer, Eval
from threetears.evals.analysis import CampaignReport, campaign_report
from threetears.evals.analysis.report.blocks import ChartBlock, TableBlock, TableCell
from threetears.evals.analysis.report.charts import DotIntervalChart, DotIntervalRow
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.bundle.surface import bundle_decision_surface
from threetears.evals.analysis.report.build import (
    build_code_only_report,
    contrast_evidence,
    contrast_rows,
    contrast_verdicts,
)
from threetears.evals.analysis.report.figure_keys import Figure, format_figure, resolve
from threetears.evals.analysis.report.html_parts import grid_html
from threetears.evals.analysis.report.parts.full_results import every_table
from threetears.evals.analysis.report.parts.look import PageLook
from threetears.evals.analysis.report_arms import fact_arms
from threetears.evals.analysis.report.block_spec import block_look
from threetears.evals.analysis.campaign_bundle import build_campaign_bundle
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.kernel.numbers import ends_places, set_places, spell_at

from packages.evals.tests.fixtures.report_inputs.dataset import kairo_bundle
from packages.evals.tests.hot_path_support import HOT_PATH_SCOPE, hot_path_corpus
from packages.evals.tests.report_doc_support import snapshot_of
from packages.evals.tests.report_shapes_support import ArmTrials, AspectTrials, shape_of


# --- the rule ----------------------------------------------------------------------------------------------------------


def test_a_set_reads_at_the_decimals_its_least_figure_needs_and_keeps_them_on_a_whole_member() -> None:
    """Two significant figures of the least figure but zero: ``1.434`` beside ``1.486`` is one decimal, ``99.2345``
    beside ``100`` none, a change of ``0.05132`` three. A member keeps the set's decimals where the lone rule would drop
    them (``-6.0``, ``0.0``), and a figure that rounds to zero carries no sign."""
    assert set_places([1.434, 1.486]) == 1
    assert set_places([99.2345, 100.0]) == 0
    assert set_places([0.05132, 0.09474, None, 0.0]) == 3
    assert set_places([0.0996]) == 2, "placed by what it rounds to: 0.10"
    assert [spell_at(-6.0, 1), spell_at(0.0, 1), spell_at(0.02, 1, signed=True), spell_at(3.6, 1, signed=True)] == [
        "-6.0",
        "0.0",
        "0.0",
        "+3.6",
    ]


def test_float_residue_says_nothing_of_a_set_s_precision_and_no_set_reads_past_a_millionth() -> None:
    """A change of float residue (``5.5e-17``) spelled every member of its set to eighteen decimals; a figure under a
    millionth reads as its bound, so it asks for no decimals, and no set reads past six."""
    assert set_places([1.2, 5.5e-17]) == 1
    assert set_places([5.5e-17]) == 0
    assert set_places([0.0000012345]) == 6


def test_an_interval_s_ends_take_more_decimals_only_where_an_end_would_read_as_a_zero_or_two_ends_alike() -> None:
    """``[-15.1, -0.03046]`` at one decimal would print ``-0.0``, an interval clear of zero that reads as touching it;
    both ends take the decimals the near end needs. Ends at the set's decimals stay there."""
    assert ends_places(-15.09, -0.03046, 1) == 2
    assert ends_places(-6.9, 11.0, 1) == 1
    assert ends_places(-0.06, 0.2, 1) == 1, "-0.06 reads -0.1 at one decimal: not a zero"
    assert ends_places(2.351, 2.354, 2) == 3, "two ends that differ never read alike"


# --- the code page's contrasts -----------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scaled() -> CampaignReport:
    """The hot-path oracle's scaled campaign's page: three candidates against a control on judged scores, rates, a count
    per trial, a time and money."""
    corpus = hot_path_corpus(scale=1)
    return campaign_report(corpus.host, corpus.campaign.id, HOT_PATH_SCOPE)


def _comparisons(page: CampaignReport) -> TableBlock:
    found = [
        block
        for aspect in page.aspects
        for block in aspect.report.blocks
        if isinstance(block, TableBlock) and block.name == "comparisons"
    ]
    found += [block for block in page.blocks if isinstance(block, TableBlock) and block.name == "comparisons"]
    return found[0]


def _rows(table: TableBlock, reading: str) -> list[dict[str, str]]:
    return [{key: cell.text for key, cell in row.items()} for row in table.rows if row["reading"].text == reading]


def test_a_judged_score_reads_at_two_decimals_on_its_named_scale(scaled: CampaignReport) -> None:
    """The scaled campaign's judged ``outcome`` printed ``2.832`` beside ``2.897 [2.839, 2.956]``, a change of ``0.0686``
    and ``[-0.06693, 0.2041]``, with no scale. Every figure of the row reads at the two decimals its narrowest interval
    sets, the change signed, each interval in its one form, an end kept off zero on its own (``-0.30 to -0.004``), and
    its name carries the scale."""
    rows = _rows(_comparisons(scaled), "outcome (judged, 1–5)")
    assert rows, [row["reading"].text for row in _comparisons(scaled).rows]
    for row in rows:
        figures = re.findall(r"[+-]?\d+(?:\.\d+)?", row["control"] + " " + row["arm"] + " " + row["change"])
        assert figures and all(re.fullmatch(r"[+-]?\d+\.\d{2}", figure) for figure in figures), row
        assert re.fullmatch(r"[+-]\d+\.\d{2}", row["change"]), row
        low, high = re.fullmatch(r"(\S+) to (\S+)", row["interval"]).groups()  # type: ignore[union-attr]
        assert len(low.partition(".")[2]) >= 2 and len(high.partition(".")[2]) >= 2, row
        assert low[0] in "+-" and high[0] in "+-", row


def test_a_change_in_points_reads_signed_at_its_rate_s_places_and_its_interval_s_ends_alike(
    scaled: CampaignReport,
) -> None:
    """Field accuracy's candidate-b change of -6.0 points printed ``-6 pts`` beside ``+3.6``'s ``3.6 pts``, and Refund
    amount check's interval ``[-6.9, 11] pts``. Every change reads signed at its rate's places, which its narrowest
    interval sets (Field accuracy's tenth, Refund amount check's whole points), both ends of its interval alike and
    signed, the unit once; the pass rate's ``-0.03046`` end reads ``-0.03``, never ``-0`` or ``0``."""
    table = _comparisons(scaled)
    field = {row["contrast"]: row for row in _rows(table, "Field accuracy")}
    assert field["candidate-b"]["change"] == "-6.0 pts"
    assert field["candidate-a"]["change"] == "+3.6 pts"
    refund = {row["contrast"]: row for row in _rows(table, "Refund amount check")}
    assert refund["candidate-a"]["interval"] == "-7 to +11 pts"
    passes = {row["contrast"]: row for row in _rows(table, "Pass rate")}
    assert passes["candidate-b"]["interval"] == "-15 to -0.03 pts"
    for row in table.rows:
        text = row["interval"].text
        if text.endswith(" pts"):
            low, high = re.fullmatch(r"(\S+) to (\S+) pts", text).groups()  # type: ignore[union-attr]
            assert low[0] in "+-" and high[0] in "+-", text
            assert not any(re.fullmatch(r"[+-]?0(?:\.0*)?", end) for end in (low, high)), text


def test_a_count_per_trial_reads_at_the_places_its_interval_sets(scaled: CampaignReport) -> None:
    """``Fields stripped`` printed ``1.434`` beside ``1.486 [1.406, 1.565]`` and a change of ``0.05132``: its levels,
    its changes and their intervals read at the hundredth its narrowest interval sets, ``1.49 (1.41 to 1.56)``."""
    rows = _rows(_comparisons(scaled), "Fields stripped")
    assert rows
    for row in rows:
        assert re.fullmatch(r"\d\.\d{2}", row["control"]), row
        assert re.fullmatch(r"\d\.\d{2} \(\d\.\d{2} to \d\.\d{2}\)", row["arm"]), row
        assert re.fullmatch(r"[+-]\d\.\d{2}", row["change"]), row


# --- no variation ------------------------------------------------------------------------------------------------------


def _answering(misses: int) -> object:
    async def answer(case: dict[str, int]) -> Answer:
        return Answer("right" if case["n"] >= misses else "wrong", model="m-1", cost_usd=2e-5)

    return answer


def _right(case: dict[str, int], answer: str) -> bool:
    return answer == "right"


def test_two_arms_that_never_varied_draw_no_interval_on_their_change() -> None:
    """Every trial on both arms cost the same, so each arm's own interval is a point and the change is exactly zero; the
    page printed an interval on it as if it were noise. The row reads ``no variation (all $0.020)``, draws no glyph, and
    prints each arm's figure alone."""
    cases = [{"n": index} for index in range(12)]
    result = asyncio.run(
        Eval(cases, {"production": _answering(4), "zulu": _answering(2)}, [_right], k=1).run(store=":memory:")  # type: ignore[dict-item]
    )
    table = _comparisons(campaign_report(result.host, result.campaign_id, result.scope_id))
    (cost,) = [row for row in table.rows if row["reading"].text.startswith("Production cost")]
    assert cost["interval"].text == "no variation (all $0.020)", cost["interval"].text
    assert cost["change"].low is None, "no glyph on a change no interval reads"
    assert "[" not in cost["control"].text and "[" not in cost["arm"].text


# --- the written page's Full results -----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kairo_tables() -> Iterator[dict[str, TableBlock]]:
    held = ReportInput.model_validate(snapshot_of("prototype5/kairo").report_input)
    look: PageLook = block_look(held, fact_arms(held))
    yield {table.name: table for table in every_table(held, look, forms=PageForms.of(held))}


def test_the_corrected_comparisons_read_each_figure_at_its_measure_s_one_precision(
    kairo_tables: dict[str, TableBlock],
) -> None:
    """Kairo's Full results printed ``-6 pts`` beside ``-7.6 pts``, ``-0.2129 | -0.3065 to -0.1194`` for a judged score
    with no scale, ``-0.00223 notices | -0.006919 notices to +0.00246 notices`` and a level of ``99.69%``."""
    rows = [{key: cell.text for key, cell in row.items()} for row in kairo_tables["contrasts"].rows]
    said = {(row["arm"], row["measure"]): row for row in rows}
    held = said["qwen3.8-flash", "Persona holds character"]
    assert (held["change"], held["interval"]) == ("-6 pts", "-12 to -2 pts")
    outcome = said["qwen3.8-flash", "Outcome (1–5)"]
    assert (outcome["change"], outcome["interval"]) == ("-0.21", "-0.31 to -0.12")
    notices = said["1017", "Failed-reply notices"]
    assert (notices["change"], notices["interval"]) == ("-0.002 notices", "-0.007 to +0.002 notices")
    assert {row["level"] for row in rows} == {"95% (adjusted for 16 comparisons)"}


def test_a_change_of_float_residue_leaves_its_measure_s_other_changes_at_their_precision() -> None:
    """A change of ``5.5e-17`` notices beside Kairo's ``-0.0022`` spelled the column ``-0.002229654403567447 notices``
    and itself ``+0.000000000000000055 notices``. The residue reads as the zero it is, whole (``0 notices``), and the
    rest at their measure's precision."""
    held = ReportInput.model_validate(snapshot_of("prototype5/kairo").report_input)
    residue = [
        contrast.model_copy(update={"delta": 5.5e-17})
        if (contrast.arm, contrast.measure) == ("qwen3.8-flash", _NOTICES)
        else contrast
        for contrast in held.contrasts
    ]
    held = held.model_copy(update={"contrasts": residue})
    look: PageLook = block_look(held, fact_arms(held))
    (contrasts,) = [table for table in every_table(held, look, forms=PageForms.of(held)) if table.name == "contrasts"]
    said = {
        row["arm"].text: (row["change"].text, row["interval"].text)
        for row in contrasts.rows
        if row["measure"].text == "Failed-reply notices"
    }
    assert said == {
        "1017": ("-0.002 notices", "-0.007 to +0.002 notices"),
        "qwen3.8-flash": ("0 notices", "-0.007 to +0.002 notices"),
    }, said


def test_a_judged_score_reads_at_two_decimals_and_an_exact_zero_whole_in_every_full_results_table(
    kairo_tables: dict[str, TableBlock],
) -> None:
    """The arms table printed ``Outcome`` at ``4.695`` and ``2.81`` beside it; the notices measure ``0.00223`` beside
    ``0 notices``. A score reads at two decimals under its scale's name, a count at one precision, a zero whole."""
    every = [cell.text for table in kairo_tables.values() for row in table.rows for cell in row.values()]
    assert "Outcome (1–5)" in every and "Outcome" not in every
    arms = kairo_tables["arms_by_measure"]
    (key,) = [column.key for column in arms.columns if column.header == "Outcome (1–5)"]
    outcome = [row[key].text for row in arms.rows if key in row]
    assert outcome and all(re.fullmatch(r"\d\.\d{2}", text) for text in outcome), outcome
    notices = [cell.text for row in arms.rows for key, cell in row.items() if "notices" in cell.text]
    assert "0.002 notices" in notices and "0 notices" in notices, notices


def test_a_measured_figure_reads_on_the_scorecard_as_the_full_results_print_it() -> None:
    """The scorecard placed its measured columns by a rule of its own, a decimal more where two figures would print
    alike: mean turn times of 102.2 and 102.4 ms read ``102.2`` and ``102.4`` there and ``102 ms`` in the written page's
    Full results. A set reads at one precision wherever it prints."""
    shape = shape_of(
        "measured",
        [
            AspectTrials(aspect, [ArmTrials("solo", passing=15, latency_ms=first)], margin=None)
            for aspect, first in (("billing", 4.2), ("tone", 4.4))
        ],
        control=False,
        question="How long does solo take?",
    )
    (scorecard,) = [
        block for block in shape.page().blocks if isinstance(block, TableBlock) and block.name == "scorecard"
    ]
    carded = {row["aspect"].text: row["value"] for row in scorecard.rows if row["measure"].text == "Turn time"}
    # The control: two figures 0.2 ms apart, which the measure's intervals read whole, so both read 102.
    assert [cell.value for cell in carded.values()] == [102.0, 102.0]
    bundle = build_campaign_bundle(
        shape.campaign, list(shape.bundles.values()), arms=shape.arms, readings=shape.readings
    )
    held = report_input(bundle)
    look: PageLook = block_look(held, fact_arms(held))
    (arms,) = [table for table in every_table(held, look, forms=PageForms.of(held)) if table.name == "arms_by_measure"]
    written = {row["aspect"].text: row["measure:total_ms"].text for row in arms.rows}
    assert (
        {aspect: f"{cell.text} ms" for aspect, cell in carded.items()}
        == written
        == {
            "billing": "102 ms",
            "tone": "102 ms",
        }
    )


def test_a_choice_s_pairs_table_signs_every_change_as_the_rest_of_the_page_does() -> None:
    """A pair's change read ``15.0`` under ``Change (pts)`` where a negative one carried its ``-``: a pair is read
    first less second, so an unsigned figure reads as a magnitude. Each change carries its sign, a zero none, at the
    pass rate's places (whole points here)."""
    shape = shape_of(
        "choice",
        [
            AspectTrials(aspect, [ArmTrials("solo", passing=first), ArmTrials("duo", passing=second)], margin=None)
            for aspect, first, second in (("billing", 15, 12), ("tone", 12, 15), ("refunds", 14, 14))
        ],
        control=False,
        question="Which arm passes more?",
    )
    (pairs,) = [block for block in shape.page().blocks if isinstance(block, TableBlock) and block.name == "pairs"]
    said = {row["aspect"].text: row["change"].text for row in pairs.rows}
    assert said == {"billing": "+15", "tone": "-15", "refunds": "0"}, said


# --- figure keys, glyphs, cells and charts -----------------------------------------------------------------------------


def test_an_interval_key_reads_both_ends_at_one_precision_and_a_lone_change_keeps_the_number_rule() -> None:
    """The owner's own example of a set at one precision: a change from -29% to +57.6% read ``-29 to +57.6``. A lone figure, written
    whole, keeps the number rule (the analyst's served grammar reads ``-15``)."""
    interval = Figure(path="x", kind="interval", value=(-0.29, 0.576), scale="share", change=True)
    off = PageForms.off_page()
    assert format_figure(interval, forms=off) == "-29.0 to +57.6"
    assert format_figure(interval, ["abs"], forms=off) == "29.0 to 57.6"
    near = Figure(path="x", kind="interval", value=(-0.0216, -0.00037), scale="share", change=True)
    assert format_figure(near, forms=off) == "-2.16 to -0.04", "an end that is not zero never reads as one"
    assert format_figure(Figure(path="x", kind="number", value=-0.15, scale="share", change=True), forms=off) == "-15"
    # A plain interval no table sets reads at two significant figures of its width, never padded to its finer end.
    plain = Figure(path="x", kind="interval", value=(2.839, 3.0), scale="plain")
    assert format_figure(plain, forms=off) == "2.84–3.00"
    assert (
        format_figure(Figure(path="x", kind="interval", value=(0.0009342, 0.2041), scale="plain"), forms=off)
        == "0.0009–0.2041"
    )


def test_a_glyph_s_title_and_its_scale_line_read_a_whole_end_at_the_column_s_precision() -> None:
    """Host pass 6's O-4: a change cell of -5.0 points on ``[-12.8, 10]`` titled its glyph ``-5 pts, interval [-12.8,
    10] pts`` and the line under the table ``drawn from -12.8 to 10 pts``."""
    block = TableBlock(
        section="scorecard",
        name="scorecard",
        title="By aspect",
        shared_scale=True,
        columns=[{"key": "aspect", "header": "Aspect"}, {"key": "arm:v2", "header": "v2 (pts)"}],  # type: ignore[list-item]
        order="aspect by aspect",
        rows=[
            {
                "aspect": TableCell(value="classify"),
                "arm:v2": TableCell(value=-5.0, low=-12.8, high=10.0, zero_line=True),
            }
        ],
        total_rows=1,
    )
    drawn = html.unescape(grid_html(block))
    # The interval in its one form, a change's ends signed.
    assert "<title>-5.0 pts, interval -12.8 to +10.0 pts</title>" in drawn, drawn
    assert '<p class="axis">Each column drawn from -12.8 to 10.0 pts</p>' in drawn


def test_a_change_of_zero_reads_at_its_column_s_precision_and_carries_no_sign() -> None:
    """V2's review: an exact zero change read ``0 pts`` beside ``-5.0 pts``; a tiny positive one read ``+0.0``."""
    assert TableCell(value=0.0, unit="pts", decimals=1, signed=True).text == "0.0 pts"
    assert TableCell(value=0.0009, unit="pts", decimals=1, signed=True).text == "0.0 pts"
    assert TableCell(value=-5.0, unit="pts", decimals=1, signed=True).text == "-5.0 pts"
    assert TableCell(value=0.0686, decimals=2, signed=True).text == "+0.07"


def test_a_chart_s_values_table_reads_its_values_at_the_digits_its_format_names() -> None:
    """V6's review: the preference figure's values table read ``50%`` where the line under it read ``49.998%``. A
    format naming its digits (``pct:3``) is the set's precision, in the table as in the kit's tooltip."""
    chart = DotIntervalChart(
        rows=[DotIntervalRow(label="lead-first", value=49.998, lo=40.0, hi=60.0)],
        domain=(0.0, 100.0),
        fmt="pct:3",
        level="95%",
    )
    _, rows = ChartBlock(section="scorecard", chart="dot_interval", title="t", data=chart).values_table()
    assert rows == [["lead-first", "49.998%", "40.000 to 60.000%"]], rows


# --- an interval never runs past what its measure can take (plan 4.5 row 2) -------------------------------------------


#: Kairo round 2's failed-reply notices: a count per trial its host declares neither non-negative nor ranged.
_NOTICES = "failed_reply_notices"


def _kairo_contrasts(
    declared: dict[str, object] | None = None,
) -> tuple[AnalysisContextBundle, dict[str, dict[str, str]]]:
    """Kairo's code page's contrast rows on the notices measure, by candidate, the measure declared as its host declared
    it, or with ``declared`` laid over its descriptor (a floor, a range), the stored intervals left as the run read
    them."""
    bundle = kairo_bundle()
    (aspect,) = bundle.aspects
    if declared is not None:
        catalog = dict(aspect.measure_catalog)
        catalog[_NOTICES] = catalog[_NOTICES].model_copy(update=declared)
        aspect = aspect.model_copy(update={"measure_catalog": catalog})
    report = build_code_only_report(aspect, assembled_at="2026-10-31T00:00:00+00:00")
    table = _comparisons_of(report.blocks)
    rows = {
        row["contrast"].text: {key: cell.text for key, cell in row.items()}
        for row in table.rows
        if row["reading"].text == "Failed-reply notices"
    }
    return aspect, rows


def _comparisons_of(blocks: object) -> TableBlock:
    return next(block for block in blocks if isinstance(block, TableBlock) and block.name == "comparisons")  # type: ignore[attr-defined]


def _notices_sentence(aspect: AnalysisContextBundle) -> str:
    surface = bundle_decision_surface(aspect)
    forms = PageForms.of_bundle(aspect)
    figures = contrast_evidence(aspect, forms=forms)
    (verdict,) = [
        verdict
        for verdict in contrast_verdicts(aspect, surface, forms=forms)
        if verdict.name == _NOTICES and verdict.arm.startswith("1017")
    ]
    said = figures(verdict)
    assert said is not None
    return said


def test_an_undeclared_floor_leaves_a_count_s_interval_as_the_run_read_it() -> None:
    """Kairo's notices printed ``0.00223 [-0.0008678, 0.005327] notices``. Its host declares the measure neither
    non-negative nor ranged, so nothing here may say it cannot fall below zero: the interval keeps its negative end,
    at the measure's precision, and the change's interval stays as read. The arm that raised none reads ``0 notices``, whole and with its unit."""
    descriptor = kairo_bundle().aspects[0].measure_catalog[_NOTICES]
    assert not descriptor.nonnegative and descriptor.value_range is None, "the control: no floor declared"
    _, rows = _kairo_contrasts()
    row = rows["1017"]
    assert row["control"] == "0.002 notices (-0.001 to 0.005)", row
    assert row["arm"] == "0 notices", row
    assert row["interval"] == "-0.007 to +0.002 notices", row


def test_a_declared_floor_holds_a_level_s_interval_at_zero() -> None:
    """The same measure declared non-negative: the level's interval stops at the floor, which reads ``0``."""
    _, rows = _kairo_contrasts({"nonnegative": True})
    assert rows["1017"]["control"] == "0.002 notices (0 to 0.005)", rows["1017"]


def test_a_declared_range_holds_a_level_s_interval_and_a_change_s_inside_it_in_the_table_and_the_sentence() -> None:
    """Declared on 0 to 0.004 notices: the level's interval stops at the range's top, and a change can move no further
    than the range is wide, in the table's interval cell and in the verdict line's sentence alike."""
    aspect, rows = _kairo_contrasts({"nonnegative": True, "value_range": (0.0, 0.004)})
    row = rows["1017"]
    assert row["control"] == "0.002 notices (0 to 0.004)", row
    assert row["interval"] == "-0.004 to +0.002 notices", row
    sentence = _notices_sentence(aspect)
    assert "0.004 notices better" in sentence and "0.007" not in sentence, sentence


def test_a_written_page_s_change_never_reaches_past_its_measure_s_scale() -> None:
    """The written page's corrected comparisons: a judged score's change can move at most the scale's width (4 on
    1–5), so an interval read past it is held there; the notices measure, which declares no range, keeps its own."""
    held = ReportInput.model_validate(snapshot_of("prototype5/kairo").report_input)
    widened = [
        contrast.model_copy(update={"interval": (-4.6, 0.1)})
        if (contrast.arm, contrast.measure) == ("qwen3.8-flash", "__outcome__")
        else contrast
        for contrast in held.contrasts
    ]
    held = held.model_copy(update={"contrasts": widened})
    look: PageLook = block_look(held, fact_arms(held))
    tables = {table.name: table for table in every_table(held, look, forms=PageForms.of(held))}
    said = {(row["arm"].text, row["measure"].text): row["interval"].text for row in tables["contrasts"].rows}
    assert said["qwen3.8-flash", "Outcome (1–5)"] == "-4.00 to +0.10"
    assert said["1017", "Failed-reply notices"] == "-0.007 to +0.002 notices"


# --- the review's observations ------------------------------------------------------------------------------------------


def test_a_key_reads_a_score_and_a_measure_s_interval_at_the_table_s_precision() -> None:
    """A lone judged-score key read ``-0.2129`` beside the Full results' ``-0.21``; a measure's interval key padded to
    its finer end. Each reads as the Full results print it."""
    held = ReportInput.model_validate(snapshot_of("prototype5/kairo").report_input)
    assert format_figure(resolve("contrast.qwen3.8-flash.__outcome__.delta", held), forms=PageForms.of(held)) == "-0.21"
    assert (
        format_figure(resolve("contrast.qwen3.8-flash.__outcome__.interval", held), forms=PageForms.of(held))
        == "-0.31 to -0.12"
    )
    notices = format_figure(resolve("contrast.1017.failed_reply_notices.interval", held), forms=PageForms.of(held))
    assert notices == "-0.007 to +0.002 notices", notices
    assert (
        format_figure(resolve("contrast.1017.failed_reply_notices.delta", held), forms=PageForms.of(held))
        == "-0.002 notices"
    )


def test_a_key_and_the_full_results_read_a_measure_s_change_by_one_owner_of_its_set() -> None:
    """The keys placed a measure's changes over every comparison on it but a judged one, the Full results over those of
    the row's own kind: a notices change of -0.5 stated with no kind read ``-0.50 notices`` in the table and
    ``-0.5000 notices`` in the key citing it. One owner places the set, for both."""
    held = ReportInput.model_validate(snapshot_of("prototype5/kairo").report_input)
    unkinded = [
        contrast.model_copy(update={"kind": None, "delta": -0.5})
        if (contrast.arm, contrast.measure) == ("qwen3.8-flash", _NOTICES)
        else contrast
        for contrast in held.contrasts
    ]
    held = held.model_copy(update={"contrasts": unkinded})
    look: PageLook = block_look(held, fact_arms(held))
    (contrasts,) = [table for table in every_table(held, look, forms=PageForms.of(held)) if table.name == "contrasts"]
    tabled = {
        row["arm"].text: row["change"].text for row in contrasts.rows if row["measure"].text == "Failed-reply notices"
    }
    keyed = {
        arm: format_figure(resolve(f"contrast.{arm}.{_NOTICES}.delta", held), forms=PageForms.of(held))
        for arm in tabled
    }
    assert tabled == keyed == {"1017": "-0.002 notices", "qwen3.8-flash": "-0.500 notices"}, (tabled, keyed)


def test_a_measure_s_places_are_read_from_its_narrowest_half_width_as_printed() -> None:
    """A measure's decimals come from its uncertainty (number policy, rule 2): the narrowest half-width among its
    headline intervals, as printed, to its first significant figure. Kairo's failed-reply notices: the control's
    interval is 0.0031 either side, so every notice figure reads to three decimals; its send-message check, a share
    printed in percent, is 0.64 points either side at its narrowest, so it reads to a tenth."""
    held = ReportInput.model_validate(snapshot_of("prototype5/kairo").report_input)
    forms = PageForms.of(held)
    notices = forms.measure(_NOTICES)
    assert notices.places == 3
    assert notices.half_width is not None and round(notices.half_width, 4) == 0.0031
    check = next(form for form in forms.measures if form.key.startswith("goal_state:"))
    assert check.places == 1
    assert check.half_width is not None and round(check.half_width, 2) == 0.64


def test_a_figure_under_a_millionth_reads_as_its_bound_and_asks_for_no_decimals() -> None:
    """Float noise of 1e-9 on an interval's end spelled the interval to nine places."""
    assert ends_places(1e-9, 0.5, 1) == 1
    assert spell_at(1e-9, 1) == "< 0.000001" and spell_at(-1e-9, 1) == "> -0.000001"
    assert ends_places(0.0, 1e-12, 1) == 1


def test_a_family_level_under_100_never_reads_100() -> None:
    """A corrected level of 99.95% read ``100%``, which no interval holds at. Its heading now states the guarantee the
    family holds at, and the level itself, where it is printed, reads as stored, never rounded."""
    held = ReportInput.model_validate(snapshot_of("prototype5/kairo").report_input)
    held = held.model_copy(
        update={
            "contrasts": [contrast.model_copy(update={"level": "99.95%"}) for contrast in held.contrasts],
            "contrast_families": [family.model_copy(update={"level": "99.95%"}) for family in held.contrast_families],
        }
    )
    look: PageLook = block_look(held, fact_arms(held))
    (contrasts,) = [table for table in every_table(held, look, forms=PageForms.of(held)) if table.name == "contrasts"]
    assert {row["level"].text for row in contrasts.rows} == {"95% (adjusted for 16 comparisons)"}
    assert PageForms.of(held).confidence("99.95%").text == "99.95%"


def test_a_change_reads_at_its_table_s_precision_however_its_family_spells_it() -> None:
    """Only a contrast's ``delta`` took the table's precision, by name: ``compare.<a>.<b>.<score>.overall.difference``
    read ``-0.2129`` beside its own interval ``-0.31 to -0.12``, and a stratum's ``vs_control`` ``+0.09524`` beside its
    mean ``4.76``. A change takes it by what it is, in every family; a half-width and an adjusted p keep their own."""
    held = ReportInput.model_validate(snapshot_of("prototype5/kairo").report_input)

    def said(key: str) -> str:
        return format_figure(resolve(key, held), forms=PageForms.of(held))

    compare = "compare.qwen3.8-flash.September.__outcome__.overall"
    assert [said(f"{compare}.{field}") for field in ("difference", "interval", "low", "high")] == [
        "-0.21",
        "-0.31 to -0.12",
        "-0.31",
        "-0.12",
    ]
    notices = "compare.1017.September.failed_reply_notices.overall"
    assert (said(f"{notices}.difference"), said(f"{notices}.interval")) == (
        "-0.002 notices",
        "-0.007 to +0.002 notices",
    )
    stratum = "strata.taste.1017.__outcome__"
    assert (said(f"{stratum}.mean"), said(f"{stratum}.vs_control")) == ("4.76", "+0.10")
    # An exact zero reads whole: it states no precision (number policy, rule 4).
    assert said("strata.music_tech.qwen3.8-flash.__outcome__.vs_control") == "0"
    assert said(f"{compare}.half_width") == "0.09358"
    assert said(f"{compare}.level") == "99.69%"


def test_a_program_s_row_says_no_variation_where_the_page_does() -> None:
    """A program's contrast row printed an interval where the page's row said ``no variation``."""
    cases = [{"n": index} for index in range(12)]
    result = asyncio.run(
        Eval(cases, {"production": _answering(4), "zulu": _answering(2)}, [_right], k=1).run(store=":memory:")  # type: ignore[dict-item]
    )
    page = campaign_report(result.host, result.campaign_id, result.scope_id)
    (aspect,) = page.aspects
    (cost,) = [row for row in contrast_rows(aspect.report.bundle) if str(row["reading"]).startswith("Production cost")]
    assert cost["interval"] == "no variation (all $0.020)", cost


def test_only_the_statistics_a_table_prints_take_its_precision() -> None:
    """V5's verify: the score rule reached every plain figure of a judged reading, so an adjusted p of 0.0031 read
    ``0.00``. A judged contrast's and spread's mean, delta, interval and its ends read at a score's two decimals; its
    adjusted p, pairing r, case count, level, standard error, per-case spread, repeats and count keep their own form."""
    held = ReportInput.model_validate(snapshot_of("prototype5/kairo").report_input)
    held = held.model_copy(
        update={
            "contrasts": [
                contrast.model_copy(update={"p_adjusted": 0.0031, "pairing_r": 0.4321})
                if (contrast.arm, contrast.measure) == ("qwen3.8-flash", "__outcome__")
                else contrast
                for contrast in held.contrasts
            ]
        }
    )

    def said(key: str) -> str:
        return format_figure(resolve(key, held), forms=PageForms.of(held))

    contrast = "contrast.qwen3.8-flash.__outcome__"
    assert [said(f"{contrast}.{stat}") for stat in ("delta", "interval", "low", "high")] == [
        "-0.21",
        "-0.31 to -0.12",
        "-0.31",
        "-0.12",
    ]
    assert said(f"{contrast}.p_adjusted") == "0.0031"
    assert said(f"{contrast}.pairing_r") == "0.4321"
    assert said(f"{contrast}.n_cases") == "299"
    assert said(f"{contrast}.level") == "99.69%"
    spread = "spread.qwen3.8-flash.__outcome__.overall"
    assert said(f"{spread}.mean") == "4.48" and said(f"{spread}.interval") == "4.43–4.54"
    assert [said(f"{spread}.{stat}") for stat in ("se", "per_case", "repeats", "n")] == [
        "0.02848",
        "0.4924",
        "2.993",
        "895",
    ]
    judged = "judged.__outcome__.qwen3.8-flash"
    assert said(f"{judged}.mean") == "4.48" and said(judged) == "4.48", "a key naming no statistic reads the mean"
    assert (said(f"{judged}.sem"), said(f"{judged}.n")) == ("0.02848", "895")
