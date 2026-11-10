"""A change on the scorecard never stands without its baseline (visual pass, chunk V2).

The owner, 2026-11-08: "earlier versions had things like charts showing deltas for treatments as -8 or +6 or whatever,
but control was not obvious." The scorecard was that archetype: a candidate's column printed a bare delta (``-7.4``,
Kairo's ``-8 (no worse than the margin)``) under ``candidate-b (pts)``, with no "vs control", no sign, no level and
no measure named, and the control's rate beside it on another basis than the delta's. Held here, on the ladder's
replayed rungs, the Kairo code page, the scaled hot-path campaign and the shared shapes:

- **L1, baseline.** Each candidate's column names the measure and the control its change is against, and the line
  under the matrix says the tick is that control and which way is better;
- **L2, sign and unit.** Each change reads signed and in points: ``+2.5 pts``, ``-7.4 pts``;
- **L3, arithmetic.** A candidate's printed level less the control's printed level is its printed change, to
  within rounding: the control's level is the change's own baseline (the mean of the cases the change reads), which
  says so where it is not the control's trials passed (``mean of 380 cases``);
- **L7, one form.** A change reads at one decimal, its unit's precision, on the matrix and in its fold alike, so
  Kairo's -7.6 never prints as -8;
- a one-arm scorecard is turned: a row per measure, no glyph;
- what limits each interval (Precision) is folded under Methods, never on the scorecard;
- code's own verdict never refuses a page for a host's name holding a stray backtick, and the undecided reading is
  linked once on the verdict line, at its first match outside a code span.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from threetears.evals.analysis import (
    build_campaign_report,
    campaign_report,
    campaign_report_html,
    campaign_report_markdown,
)
from threetears.evals.analysis.campaign_readings import campaign_readings
from threetears.evals.analysis.report.model import CampaignReport, DisclosureBlock, TableBlock, TableCell
from threetears.evals.analysis.style import CANNOT_TELL_APART
from threetears.evals.kernel.numbers import round_half_even
from threetears.evals.quick.files import load_evals

from packages.evals.tests.campaign_page_support import kairo_page
from packages.evals.tests.hot_path_support import HOT_PATH_SCOPE, hot_path_corpus
from packages.evals.tests.ladder_support import load, offline_copy
from packages.evals.tests.report_shapes_support import (
    ArmTrials,
    AspectTrials,
    s1_change,
    s3_measurement,
    shape_of,
)

#: The rungs held here: their directory, the file declaring their eval, and the attribute holding it (None for TOML).
_RUNGS: dict[str, tuple[str, str, str | None]] = {
    "rung1": ("1-triage", "triage.py", "triage_eval"),
    "rung2": ("2-replies", "replies.py", "replies_eval"),
    "rung3": ("3-invoices", "extract.py", "invoices"),
    "rung4": ("4-booking", "agent.py", "booking"),
}

#: A candidate's change cell as every serializer prints it: its level, then its change signed in points at one
#: decimal (an exact zero reads ``0``, which states no precision), then its marks.
#: A zero change reads at the column's one decimal, unsigned (``0.0 pts``, beside ``-5.0 pts``).
_CHANGE = re.compile(r"^(?P<level>\d+(?:\.\d+)?)% · (?P<change>[+-]\d+(?:\.\d+)?|0) pts(?: \((?P<marks>.+)\))?$")

#: The baseline a candidate's cell states, where its aspect's candidates read different cases.
_STATED = re.compile(r"from .+'s (?P<level>\d+(?:\.\d+)?)%")

#: The control's cell: its level at the pass rate's places (an exact zero ``0%``), then what it is over.
_CONTROL = re.compile(r"^(?P<level>\d+(?:\.\d+)?)% \((?P<over>.+)\)$")

#: The line under a table of glyphs.
_AXIS = re.compile(r'<p class="axis">(.*?)</p>')


@pytest.fixture(scope="module")
def pages(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, CampaignReport]]:
    """Every page held here: each rung replayed once from its recordings with no key, the Kairo code page, the scaled
    hot-path campaign read as a host reads it, and the shared change shape."""
    built: dict[str, CampaignReport] = {}
    for name, (directory, file, attribute) in _RUNGS.items():
        with pytest.MonkeyPatch.context() as monkeypatch:
            copy = offline_copy(directory, tmp_path_factory.mktemp(name), monkeypatch)
            if attribute is None:
                (runnable,) = load_evals(copy / file)
            else:
                runnable = getattr(load(copy / file, Path(file).stem, monkeypatch), attribute)
            built[name] = asyncio.run(runnable.run(store=":memory:")).page
    built["kairo"] = kairo_page()
    corpus = hot_path_corpus(scale=1)
    built["scaled"] = campaign_report(corpus.host, corpus.campaign.id, HOT_PATH_SCOPE)
    split = hot_path_corpus(scale=1, two_aspects=True)
    built["scaled_two_aspects"] = campaign_report(split.host, split.campaign.id, HOT_PATH_SCOPE)
    built["s1"] = s1_change().page()
    built["uneven"] = _uneven_baselines()
    yield built


def _uneven_baselines() -> CampaignReport:
    """Two candidates on two aspects. On billing their changes read different cases: v2 is paired with production on
    cases 1 to 15, where production passes every one, and v3 on cases 6 to 20, where it passes 13 of 15; production
    passed 18 of its 20 cases in all. On tone every arm reads the same 20 cases. The readings are read again over the
    cut per-case values, so each change is its own, and the row across both aspects is laid out."""
    arms = [ArmTrials("production", passing=18), ArmTrials("v2", passing=14), ArmTrials("v3", passing=12)]
    tone = [ArmTrials("production", passing=10), ArmTrials("v2", passing=14), ArmTrials("v3", passing=8)]
    shape = shape_of(
        "uneven",
        [AspectTrials("billing", arms, margin=None), AspectTrials("tone", tone, margin=None)],
        control=True,
    )
    kept = {shape.key("v2"): slice(0, 15), shape.key("v3"): slice(5, 20)}
    bundle = shape.bundles["billing"]
    contrasts = [
        contrast.model_copy(
            update={
                "case_values": tuple(values[kept[contrast.contrast.variant_key]] for values in contrast.case_values),
                "case_trials": None,
                "case_passed": None,
            }
        )
        if contrast.reading == "trial" and contrast.case_values is not None
        else contrast
        for contrast in bundle.floor_contrasts
    ]
    bundles = {**shape.bundles, "billing": bundle.model_copy(update={"floor_contrasts": contrasts})}
    readings = campaign_readings(shape.campaign, bundles, arms=shape.arms)
    reports = {label: shape.reports[label].model_copy(update={"bundle": held}) for label, held in bundles.items()}
    return build_campaign_report(shape.campaign, readings, reports)


#: Every page held here, and those reading a change against a control (rungs 1, 2 and 4 measure or choose).
_PAGES = (*_RUNGS, "kairo", "scaled", "scaled_two_aspects", "s1", "uneven")
_CHANGE_PAGES = ("rung3", "kairo", "scaled", "scaled_two_aspects", "s1", "uneven")


def _matrix(page: CampaignReport) -> TableBlock:
    (matrix,) = [block for block in page.blocks if isinstance(block, TableBlock) and block.name == "scorecard"]
    return matrix


def _changes(page: CampaignReport) -> TableBlock:
    """The page's change matrix: it reads a change against a control."""
    assert page.readings.kind == "change" and page.readings.control is not None and page.readings.readings
    return _matrix(page)


def _figure(text: str) -> float:
    return 0.0 if text == "0" else float(text)


def _last_place(*texts: str) -> float:
    """The size of one unit in the last place the finest of ``texts`` prints: ``1`` for whole figures, ``0.1`` for one
    decimal, so a sum of figures each rounded there closes within a multiple of it."""
    decimals = max(len(text.partition(".")[2]) for text in texts)
    return 10.0**-decimals


# --- L1: each change names the measure and the control it is against -------------------------------------------------


@pytest.mark.parametrize("name", _CHANGE_PAGES)
def test_each_candidate_s_column_names_the_measure_and_the_control_its_change_is_against(
    pages: dict[str, CampaignReport], name: str
) -> None:
    """Trips on a candidate's head that names its arm and a unit alone (``candidate-b (pts)``), or a control's that
    names no measure (``▸ control (%)``)."""
    page = pages[name]
    matrix = _changes(page)
    control = page.readings.control_label
    heads = [column.header for column in matrix.columns if column.key.startswith("arm:")]
    assert heads[0] == f"▸ {control} · pass rate", heads
    labels = {row.arm_label for row in page.readings.readings}
    assert sorted(heads[1:]) == sorted(f"{label} · pass rate vs {control}" for label in labels), heads


@pytest.mark.parametrize("name", _CHANGE_PAGES)
def test_the_line_under_the_matrix_names_the_tick_as_the_control_and_says_which_way_is_better(
    pages: dict[str, CampaignReport], name: str
) -> None:
    """Trips on the line that states the scale's ends alone (``Each column drawn from -12.8 to 10 pts``)."""
    page = pages[name]
    matrix = _changes(page)
    assert any(cell.zero_line for row in matrix.rows for cell in row.values()), "the matrix draws a glyph"
    drawn = campaign_report_html(page)
    start = drawn.index('data-table="scorecard"')
    line = re.sub(r"<[^>]+>", "", _AXIS.search(drawn, start).group(1))  # type: ignore[union-attr]
    control = page.readings.control_label
    assert f"change from {control} (the tick)" in line, line
    assert "right is better" in line, line
    # A scale of changes, its ends signed (number rule 3).
    assert re.search(r"Drawn from [+-]?[\d.]+ to [+-]?[\d.]+ pts\.", line), line


# --- L2 and L3: a change is signed, in points, beside its level, and the levels close on it -----------------------


@pytest.mark.parametrize("name", _CHANGE_PAGES)
def test_each_change_reads_its_level_then_its_change_signed_in_points_and_the_arithmetic_closes(
    pages: dict[str, CampaignReport], name: str
) -> None:
    """Trips on a bare delta (``-7.4 (shown worse)``, ``2.5 (cannot tell apart)``), on a level beside a delta on
    another basis than the delta's (scaled's control read 20 of 828 trials, its changes from the mean of 380 cases,
    20.5%), and on a change missing its candidate's level."""
    page = pages[name]
    matrix = _changes(page)
    control_key = f"arm:{page.readings.control}"
    checked = 0
    for row in matrix.rows:
        if row["aspect"].text.startswith("All aspects"):
            continue
        control = _CONTROL.fullmatch(row[control_key].text)
        assert control is not None, row[control_key].text
        for key, cell in row.items():
            if not key.startswith("arm:") or key == control_key or cell.value is None:
                continue
            said = _CHANGE.fullmatch(cell.text)
            assert said is not None, cell.text
            change = _figure(said["change"])
            if change > 0:
                assert said["change"].startswith("+"), cell.text
            # The candidate's level less its baseline is the change, to within each figure's rounding: the control's
            # level, or the baseline the cell states where the aspect's candidates read different cases.
            stated = _STATED.search(said["marks"] or "")
            baseline = float(stated["level"]) if stated is not None else float(control["level"])
            closes = float(said["level"]) - baseline - change
            # Each of the three is rounded on its own, half a unit in its last place, at the pass rate's places.
            slack = 1.5 * _last_place(said["level"], said["change"], stated["level"] if stated else control["level"])
            assert abs(closes) <= slack + 1e-9, (cell.text, row[control_key].text)
            checked += 1
    assert checked, "the matrix holds a change"


def test_a_control_level_that_is_not_its_trials_passed_says_what_it_is_a_mean_of(
    pages: dict[str, CampaignReport],
) -> None:
    """Scaled's control passed 167 of 828 trials (20.2%) over 418 cases, while every change reads the 380 cases both
    arms ran, on which the control's mean is 20.5%: its cell prints the change's own baseline, at the pass rate's whole
    percents, and says so. Trips on ``20 (167 of 828 trials, 418 cases)``."""
    page = pages["scaled"]
    (row,) = _matrix(page).rows
    assert row[f"arm:{page.readings.control}"].text == "21% (mean of 380 cases)"
    # The trials passed stay one step down, in the fold.
    folded = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "aspects")
    assert {cells["control"].text for cells in folded.rows} == {"167 of 828 trials"}


def test_a_control_whose_trials_passed_are_its_baseline_states_its_count() -> None:
    """S1's control ran each case once, so its trials passed are its baseline: ``90% (18 of 20 cases)``, and a
    control that passes nothing still says it cannot show a drop."""
    page = s1_change().page()
    matrix = _matrix(page)
    control = f"arm:{page.readings.control}"
    said = {row["aspect"].text: row[control].text for row in matrix.rows}
    assert said["billing"] == "90% (18 of 20 cases)"
    assert said["refunds"] == "0% (0 of 20 cases; cannot show a drop)"


def test_where_candidates_read_different_cases_each_names_its_own_baseline(
    pages: dict[str, CampaignReport],
) -> None:
    """V2 reads cases 1 to 15, where production passed all, and v3 cases 6 to 20, where it passed 13: no one level is
    every change's baseline. The control's cell is its trials passed, and each candidate states the level its own
    change is read from, which its level less closes on. Trips on a candidate cell naming no baseline (its change
    then read against 90%, which neither is), or on a stated baseline its level and change do not close on."""
    page = pages["uneven"]
    (row, *_) = _matrix(page).rows
    assert row["aspect"].text == "billing", "the control: billing is the split aspect"
    assert row[f"arm:{page.readings.control}"].text == "90% (18 of 20 cases)"
    labels = {reading.arm: reading.arm_label for reading in page.readings.readings}
    stated = {}
    for arm, label in labels.items():
        said = _CHANGE.fullmatch(row[f"arm:{arm}"].text)
        assert said is not None and said["marks"] is not None, row[f"arm:{arm}"].text
        baseline = _STATED.search(said["marks"])
        assert baseline is not None and said["marks"].endswith(f"from production's {baseline['level']}%"), said["marks"]
        slack = 1.5 * _last_place(said["level"], said["change"], baseline["level"])
        assert abs(float(said["level"]) - float(baseline["level"]) - _figure(said["change"])) <= slack + 1e-9, (
            said.string
        )
        stated[label] = baseline["level"]
    assert stated == {"v2": "100", "v3": "87"}


# --- L7: one precision, the pass rate's, on the matrix and in its fold --------------------------------------------

#: The places each page's pass rate reads at, set by its narrowest interval across its campaign: Kairo's a tenth,
#: every other page's whole points.
_CHANGE_PLACES = {"rung3": 0, "kairo": 1, "scaled": 0, "scaled_two_aspects": 0, "s1": 0, "uneven": 0}


@pytest.mark.parametrize("name", _CHANGE_PAGES)
def test_a_change_reads_at_the_pass_rate_s_places_on_the_matrix_and_in_its_fold_alike(
    pages: dict[str, CampaignReport], name: str
) -> None:
    """Kairo's qwen3.8-flash change is -7.6: its fold printed ``-7.6`` while its matrix printed ``-8``, the column's
    whole percents. A change now reads at the places the pass rate's narrowest interval sets across the campaign
    (one decimal at most), rounded half to even, on the matrix and in its fold alike. Trips on a fold and a matrix that
    print one change two ways, and on a change at other places than its cell's."""
    page = pages[name]
    matrix = _changes(page)
    labels = {column.key: column.header.split(" · ")[0] for column in matrix.columns}
    on_matrix = {
        (row["aspect"].text, labels[key]): _CHANGE.fullmatch(cell.text)["change"]  # type: ignore[index]
        for row in matrix.rows
        if not row["aspect"].text.startswith("All aspects")
        for key, cell in row.items()
        if key.startswith("arm:") and key != f"arm:{page.readings.control}" and cell.value is not None
    }
    folded = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "aspects")
    in_fold = {(row["aspect"].text, row["arm"].text): row["change"].text.removesuffix(" pts") for row in folded.rows}
    assert on_matrix == in_fold
    at = _CHANGE_PLACES[name]
    for arm in {reading.arm_label for reading in page.readings.readings}:
        expected = sorted(
            float(round_half_even(reading.change, at)) for reading in page.readings.readings if reading.arm_label == arm
        )
        shown = sorted((text for (_, named), text in on_matrix.items() if named == arm), key=_figure)
        assert [_figure(text) for text in shown] == expected, (arm, expected, shown)
        assert all(text == "0" or len(text.partition(".")[2]) == at for text in shown), shown
    if name == "kairo":
        assert "-7.6" in on_matrix.values()


@pytest.mark.parametrize("name", ["s1", "scaled_two_aspects", "uneven"])
def test_the_row_across_every_aspect_reads_the_mean_of_the_levels_above_and_closes_on_its_change(
    pages: dict[str, CampaignReport], name: str
) -> None:
    """The row's change is the equal-weight mean of the aspects' changes, so its levels are on that basis: the
    control's is the mean of the control's cells above (each its changes' baseline), and says so once; each
    candidate's is the mean of its own cells above, then its change signed, as every candidate cell reads. Its level
    less the control's is then its change. Scaled across two aspects printed the aggregate's pooled 21.1% beside
    candidate-a's ``23.5% · +1.9 pts``, which do not close; the control's cells above read 22% and 21%. Where an
    aspect's candidates read different cases (the uneven page's billing), the control's cells are not every change's
    baseline, so each candidate names the mean of its own baselines above (each its cell's stated one, else the
    control's level on that aspect), and closes on that. Trips on a pooled level, a level that is not the mean of the
    cells above, a bare ``+1.7 pts``, and a split row's candidate naming no baseline, or one not its own mean."""
    page = pages[name]
    matrix = _changes(page)
    *aspects, across = matrix.rows
    assert across["aspect"].text.startswith("All aspects"), "the control: the matrix closes on the row"
    control = f"arm:{page.readings.control}"
    above = [float(_CONTROL.fullmatch(row[control].text)["level"]) for row in aspects]  # type: ignore[index]
    said = re.fullmatch(
        r"(?P<level>\d+(?:\.\d+)?)% \(mean of the aspects above, each weighted equally\)", across[control].text
    )
    assert said is not None, across[control].text
    base = float(said["level"])
    half = 0.5 * _last_place(said["level"]) + 1e-3
    assert abs(base - sum(above) / len(above)) <= half, (base, above)
    if name == "scaled_two_aspects":
        assert (above, base) == ([22.0, 21.0], 22.0)
    split = any(_STATED.search(cell.text) for row in aspects for cell in row.values())
    assert split == (name == "uneven"), "the control: only the uneven page reads an aspect's candidates apart"
    checked = 0
    for key, cell in across.items():
        if not key.startswith("arm:") or key == control:
            continue
        figure = _CHANGE.fullmatch(cell.text)
        assert figure is not None, cell.text
        cells = [_CHANGE.fullmatch(row[key].text) for row in aspects]
        levels = [float(said["level"]) for said in cells]  # type: ignore[index]
        assert abs(float(figure["level"]) - sum(levels) / len(levels)) <= half, (cell.text, levels)
        # Each baseline above: the one its cell states, else the control's level on that aspect.
        baselines = [
            float(stated["level"]) if (stated := _STATED.search(said["marks"] or "")) else level  # type: ignore[index]
            for said, level in zip(cells, above, strict=True)
        ]
        stated = _STATED.search(figure["marks"] or "")
        if split:
            assert stated is not None, cell.text
            against = float(stated["level"])
            assert abs(against - sum(baselines) / len(baselines)) <= half, (cell.text, baselines)
        else:
            assert stated is None, cell.text
            against = base
        # Each level is a mean of cells already rounded at the pass rate's places, rounded again, beside a change and a
        # baseline rounded on their own: the row closes within that rounding (float slack aside).
        slack = 2 * _last_place(figure["level"], figure["change"])
        assert abs(float(figure["level"]) - against - _figure(figure["change"])) <= slack + 1e-9, (cell.text, against)
        checked += 1
    assert checked == len({reading.arm for reading in page.readings.readings})


# --- a one-arm scorecard is turned -----------------------------------------------------------------------------------


def test_a_one_arm_scorecard_is_turned_a_row_per_measure_with_no_glyph() -> None:
    """Discodon's SearXNG quality page drew fourteen glyph columns for its one arm, each on its own scale, wider than
    the frame. A measurement of one arm lays a row per measure, its value, interval and cases, and draws no glyph.
    Trips on a column per measure."""
    page = s3_measurement().page()
    matrix = _matrix(page)
    assert [column.header for column in matrix.columns] == ["Aspect", "Measure", "Value", "Interval", "Cases"]
    assert not [cell for row in matrix.rows for cell in row.values() if cell.low is not None]
    drawn = campaign_report_html(page)
    start = drawn.index('data-table="scorecard"')
    assert "viz-glyph" not in drawn[start : drawn.index("</table>", start)]
    rates = [row for row in matrix.rows if row["measure"].text == "Pass rate"]
    assert [row["aspect"].text for row in rates] == ["accuracy", "clarity"]
    # A rate states its percent and its count on the figure, its interval in the figure's form.
    assert re.fullmatch(r"\d+% \(\d+ of \d+ trials, 20 cases\)", rates[0]["value"].text), rates[0]["value"].text
    assert re.fullmatch(r"\d+ to \d+%", rates[0]["interval"].text), rates[0]["interval"].text
    assert {row["cases"].text for row in matrix.rows} == {"20"}


# --- Precision is a method -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", _PAGES)
def test_what_limits_each_interval_is_folded_under_methods_never_on_the_scorecard(
    pages: dict[str, CampaignReport], name: str
) -> None:
    """Rung 3's Precision table (half-widths, each arm's own, the share between cases) sat straight under its
    scorecard. Trips on a Precision table in the scorecard, or one shown open."""
    page = pages[name]
    precision = [block for block in page.blocks if isinstance(block, TableBlock) and block.name == "precision"]
    if name in ("rung3", "scaled"):
        assert precision, "the page reads what limits its intervals"
    assert all(block.section == "methods" and block.details for block in precision), [
        (block.section, block.details) for block in precision
    ]


# --- code's own verdict, a host name with a backtick, and the undecided reading linked once -----------------------


def test_a_host_name_with_a_stray_backtick_never_refuses_code_s_own_verdict() -> None:
    """An arm named ``v`2`` puts one backtick in the verdict line's decision, whose bold would start outside a code
    span and end inside one. A caller's emphasis is refused for that; code's own decisions read at one weight
    instead, on the campaign's page and on its aspect's. Trips on ``build_campaign_report`` raising."""
    shape = shape_of(
        "stray",
        [AspectTrials("billing", [ArmTrials("production", passing=10), ArmTrials("v`2", passing=18)], margin=None)],
        control=True,
    )
    page = shape.page()
    (answer,) = [block for block in page.blocks if isinstance(block, DisclosureBlock) and block.source == "verdict"]
    assert "v`2" in answer.text and answer.emphasis == []
    (verdict,) = [
        block
        for block in page.aspects[0].report.blocks
        if isinstance(block, DisclosureBlock) and block.source == "verdict"
    ]
    assert verdict.emphasis == []
    assert "v`2" in campaign_report_markdown(page)
    # A caller's emphasis is still held to whole code spans.
    with pytest.raises(ValueError, match="cuts a code span"):
        DisclosureBlock(section="answer", source="verdict", text=answer.text, emphasis=["v`2"])


def test_a_verdict_links_the_undecided_reading_at_its_first_match_outside_a_code_span() -> None:
    """The verdict line links the undecided reading where its sentence stands whole. Its first match in a piece may
    cross a code span (an arm named in backticks holds the words) while a whole match follows it: that one is
    linked, once, and the term is not linked again after the line. Trips on a search that reads only the first match
    and so links the term after the line instead."""
    page = s1_change().page()
    text = (
        "Pass rate: the evidence cannot tell `v2 apart` and production apart. "
        "Tone: the evidence cannot tell v3 and production apart."
    )
    line = DisclosureBlock(
        section="answer", source="verdict", text=text, tone="open", term=CANNOT_TELL_APART, emphasis=[]
    )
    blocks = [
        line if isinstance(block, DisclosureBlock) and block.source == "verdict" else block for block in page.blocks
    ]
    drawn = campaign_report_html(page.model_copy(update={"blocks": blocks}))
    start = drawn.index("Pass rate: the evidence")
    said = drawn[drawn.rindex("<p", 0, start) : drawn.index("</p>", start)]
    links = [found for found in re.findall(r"<a [^>]*>(.*?)</a>", said) if "cannot tell" in found]
    assert links == ["cannot tell v3 and production apart"], said


def test_code_emphasis_keeps_every_decision_unless_one_cuts_a_code_span() -> None:
    """The rule itself: whole decisions stand; one starting or ending inside a code span takes the bold off all."""
    assert DisclosureBlock.code_emphasis("`a` shown better. b too.", ["`a` shown better"]) == ["`a` shown better"]
    assert DisclosureBlock.code_emphasis("a`b shown better.", ["a`b shown better"]) == []


def test_a_change_cell_refuses_a_level_or_a_sign_on_words() -> None:
    """A level and a sign qualify a change, so a cell of words carrying either is refused."""
    with pytest.raises(ValueError, match="qualify a change"):
        TableCell(value="unclear", level=20.0)
    with pytest.raises(ValueError, match="qualify a change"):
        TableCell(value="unclear", signed=True)
    assert TableCell(value=2.5, unit="pts", decimals=1, signed=True, level=23.0).text == "23.0% · +2.5 pts"
    assert TableCell(value=-7.6, unit="pts", decimals=1, signed=True).text == "-7.6 pts"
    assert TableCell(value=10.0, unit="pts", decimals=1, signed=True).text == "+10.0 pts"
