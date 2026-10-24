"""What code read on each aspect of a campaign's page, in its Scorecard, by its kind: one matrix of aspects, a
change's against its candidates, a choice's against its arms and a measurement's against its measures; the long form
folded under it; the lines under a change's, and one table of what would settle each unclear reading; a choice's
frontier."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

import pytest

from threetears.evals.analysis import (
    AnalysisContextBundle,
    CampaignReadings,
    build_code_only_report,
    build_campaign_report,
    campaign_readings,
    campaign_report_markdown,
)
from threetears.evals.analysis.bundle.schema import FamilyComparison
from threetears.evals.analysis.lenses.frontier import FrontierCostDecision, FrontierCostTie, FrontierVerdict
from threetears.evals.analysis.quantities import PERCENT, POINTS
from threetears.evals.analysis.report.campaign import aggregate
from threetears.evals.analysis.report.campaign_page import CHOICE_NOTE, AspectNames, campaign_verdict
from threetears.evals.analysis.report.model import (
    CampaignReport,
    ChartBlock,
    DisclosureBlock,
    Report,
    TableBlock,
    TableCell,
)

from packages.evals.tests.campaign_support import (
    ASSEMBLED_AT,
    CANDIDATE_LABEL,
    CONTROL_LABEL,
    MeasuredKind,
)

from packages.evals.tests.campaign_page_support import (
    SUMMARY,
    campaign_fixture,
    choice_memo,
    first_screen_markdown,
    kind_fixture,
    kind_page_of,
    page_of,
)
from packages.evals.tests.report_shapes_support import SHAPES, Shape, s1_change, s2_choice, s3_measurement, s4_change
from packages.evals.tests.style_support import lint_page


def _scorecard(page: CampaignReport) -> str:
    """The page's Scorecard section in Markdown: its heading up to the next one."""
    markdown = campaign_report_markdown(page)
    start = markdown.index("\n## Scorecard\n")
    end = markdown.find("\n## ", start + 1)
    return markdown[start:] if end == -1 else markdown[start:end]


def _table(page: CampaignReport, name: str) -> TableBlock:
    return next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == name)


def _by_arm(table: TableBlock, row: Mapping[str, TableCell]) -> dict[str, TableCell]:
    """A matrix row's cells by the arm heading each column (its unit and the control's mark left off)."""
    return {
        column.header.removeprefix("▸ ").rsplit(" (", 1)[0]: row[column.key]
        for column in table.columns
        if column.key not in ("aspect", "cases")
    }


async def test_an_aspect_with_no_margin_is_said_never_shown_worse_and_has_no_row_to_settle_against_one() -> None:
    page = await page_of("billing", "tone", margins={"billing": None})
    said = _scorecard(page)
    assert "No margin was declared on billing, so no arm is ever shown worse there." in said
    (billing,) = page.readings.on_aspect("billing")
    assert billing.reading == "unclear" and "no margin was declared" in billing.reason
    # Its row would be a row of dashes under "To settle": it has no margin to show it within, and its floor is broken.
    assert not [block for block in page.blocks if isinstance(block, TableBlock) and block.name == "to_settle"]
    assert "To settle" not in said


async def test_an_unclear_row_with_a_margin_is_still_to_settle_beside_one_without() -> None:
    def tone_unclear(readings: CampaignReadings) -> CampaignReadings:
        rows = [
            row.model_copy(update={"reading": "unclear"}) if row.aspect == "tone" else row for row in readings.readings
        ]
        return readings.model_copy(update={"readings": rows})

    page = await page_of("billing", "tone", margins={"billing": 70.0, "tone": None}, readings=tone_unclear)
    assert {row.aspect: (row.reading, row.margin) for row in page.readings.readings} == {
        "billing": ("unclear", 70.0),
        "tone": ("unclear", None),
    }
    settle = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "to_settle")
    assert [row["aspect"].text for row in settle.rows] == ["billing"]


async def test_a_floor_broken_beside_a_row_not_shown_worse_is_marked_on_its_cell_and_no_sentence_recites_it() -> None:
    """A16: a sentence saying the floor reads "its pass rate" while the row reads "only the pass rate" read as a
    contradiction. The cell says the floor's status beside the row's reading instead."""
    # A 70-point margin on billing: its 60-point drop is not past it, while the 10-point floor is broken.
    page = await page_of("billing", "tone", margins={"billing": 70.0})
    (row,) = page.readings.on_aspect("billing")
    assert row.reading == "unclear"
    matrix = _table(page, "scorecard")
    billing = next(cells for cells in matrix.rows if cells["aspect"].text == "billing")
    assert _by_arm(matrix, billing)[CANDIDATE_LABEL].text == "-60 (unclear; floor broken)"
    assert "The floor broken on" not in campaign_report_markdown(page)


async def test_an_unclear_row_s_cases_show_it_no_worse_than_the_margin_and_not_better() -> None:
    page = await page_of("billing", "tone", margins={"billing": 70.0})
    settle = _table(page, "to_settle")
    assert settle.title == "To settle"
    assert [(cells["shows"].text, cells["decided_by"].text) for cells in settle.rows] == [
        ("no worse than the 70-point margin", "pass rate")
    ]
    note = page.blocks[page.blocks.index(settle) + 1]
    assert isinstance(note, DisclosureBlock)
    assert "it would not show an arm better" in note.text


async def test_a_row_no_worse_than_the_margin_reads_so_beside_its_held_floor_and_has_nothing_to_settle() -> None:
    page = await page_of("steady", "tone")
    (row,) = page.readings.on_aspect("steady")
    (status,) = page.readings.floors
    assert row.reading == "within_margin" and status.status == "held"
    table = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "aspects")
    readings = {cells["aspect"].text: cells["reading"].text for cells in table.rows}
    assert readings == {"steady": "no worse than the margin", "tone": "shown better"}
    # The headline row and the floor read one interval, so neither contradicts the other, and nothing is to settle.
    assert not any(isinstance(block, TableBlock) and block.name == "to_settle" for block in page.blocks)
    said = _scorecard(page)
    assert "A row no worse than the margin reads a plain 95% interval, as a floor does." in said
    assert (
        "An arm is no worse only where it holds on every asked aspect at once, so more aspects never make that "
        "easier, and it needs no correction." in said
    )
    # The one corrected row is tone's, read in v2's family of two at 97.5%, so it is not called a floor's level; the
    # error-rate line is said whatever the rows read.
    assert (
        "Intervals here are wider than a plain 95% interval, because each arm's 2 readings are corrected together."
        in said
    )
    assert "Each is read at 97.5%, and the reading beside it always agrees with it." in said
    assert (
        "For any one arm, the chance of being wrongly shown better on some aspect is at most 2.5%, and likewise for "
        "wrongly shown worse." in said
    )
    assert "Where such an interval sits above zero" not in said
    # One arm: nothing is said across arms.
    assert "not across the" not in said
    markdown = campaign_report_markdown(page)
    steady = markdown[markdown.index("## steady") : markdown.index("## tone")]
    assert "v2 no worse than the margin." in steady
    assert (
        campaign_verdict(page.readings, ["steady", "tone"], AspectNames.of(fold.report.bundle for fold in page.aspects))
        == "steady: v2 no worse than the margin; tone: v2 shown better"
    )


async def test_a_page_whose_every_row_is_no_worse_still_states_the_error_rate_and_that_nothing_corrects_across_arms() -> (
    None
):
    def all_within(readings: CampaignReadings) -> CampaignReadings:
        rows = [
            row.model_copy(update={"reading": "within_margin", "within_interval": (1.3, 62.7), "projection": None})
            for row in readings.readings
        ]
        return readings.model_copy(update={"readings": rows + [rows[0].model_copy(update={"arm": "v3"})]})

    page = await page_of("steady", "tone", readings=all_within)
    said = _scorecard(page)
    assert "The interval is read at" not in said and "Intervals here are wider" not in said
    assert (
        "For any one arm, the chance of being wrongly shown better on some aspect is at most 2.5%, and likewise for "
        "wrongly shown worse." in said
    )
    # Two arms: the bound across them is the page's count of arms times each one's.
    assert (
        "That correction runs within each arm, not across the 2 arms, so the chance that some arm is wrongly shown "
        "better is up to 5%, and likewise for worse." in said
    )
    assert "Correcting across arms would lower it, at the price of wider intervals." in said
    # The clause that the recommendation does not rest on it is wrong: a recommendation can rest on a corrected gain.
    assert "does not rest on it" not in said and "do not rest on it" not in said
    assert "Where such an interval sits above zero, as for v2 on steady" in said
    assert "that is not a gain shown: a gain is read on the corrected interval, which reaches zero." in said


async def test_a_floor_held_beside_a_row_left_unclear_is_marked_on_its_cell_and_only_the_row_is_to_settle() -> None:
    # The same interval against a 10-point margin is unclear while the 20-point floor holds.
    page = await page_of("steady", margins={"steady": 10.0})
    (row,) = page.readings.on_aspect("steady")
    (status,) = page.readings.floors
    assert row.reading == "unclear" and status.status == "held"
    matrix = _table(page, "scorecard")
    (steady,) = matrix.rows
    assert _by_arm(matrix, steady)[CANDIDATE_LABEL].flag == "unclear; floor held"
    assert "The floor held on" not in campaign_report_markdown(page)
    # The held floor has nothing left to show: only the row is to settle.
    settle = _table(page, "to_settle")
    assert [(cells["aspect"].text, cells["shows"].text) for cells in settle.rows] == [
        ("steady", "no worse than the 10-point margin")
    ]
    note = page.blocks[page.blocks.index(settle) + 1]
    assert isinstance(note, DisclosureBlock)
    # A count is the cases the interval needs in all, this run's repeated to that many: not cases to add to it.
    assert note.text.startswith(
        "Each count is the cases a 95% interval needs in all, not cases to add, to rule out a drop past the margin"
    )
    # A18: the sentence promising a floor's count and a row's agree held only for a floor reading one rate.
    assert "same count" not in note.text


async def test_with_no_question_every_reading_is_a_lead() -> None:
    page = await page_of("billing", "tone", question="")
    # Asking nothing is the whole report's caveat: said once, in the question's place on the first screen.
    lead = "No question was declared, so every reading is a lead."
    assert lead in first_screen_markdown(page)
    assert lead not in _scorecard(page) and campaign_report_markdown(page).count(lead) == 1


async def test_a_choice_shows_an_arm_ahead_on_one_aspect_and_no_separation_on_another() -> None:
    page = await kind_page_of("tone", "billing", memo=choice_memo())
    assert page.readings.kind == "choice"
    table = _table(page, "pairs")
    assert table.details
    assert [(row["aspect"].text, row["ahead"].text) for row in table.rows] == [
        ("tone", "alpha"),
        ("billing", "no separation"),
    ]
    # The matrix marks the arm ahead in its own cell, and no other.
    matrix = _table(page, "scorecard")
    marks = {row["aspect"].text: {arm: cell.flag for arm, cell in _by_arm(matrix, row).items()} for row in matrix.rows}
    assert marks == {"tone": {"alpha": "ahead of beta", "beta": None}, "billing": {"alpha": None, "beta": None}}
    said = _scorecard(page)
    assert CHOICE_NOTE in said
    # The writer's summary and recommendation, then the scorecard; no control-based table.
    markdown = campaign_report_markdown(page)
    assert markdown.index(SUMMARY) < markdown.index("**Recommended:** beta (high confidence)")
    assert markdown.index("**Recommended:** beta (high confidence)") < markdown.index("## Scorecard")
    assert "<summary>By aspect, pair by pair</summary>" in said and "▸" not in said
    assert "alpha ahead of beta." in markdown and "alpha and beta not separated." in markdown


async def test_a_choice_with_nothing_separated_says_so_on_every_pair() -> None:
    page = await kind_page_of("billing", "refunds")
    table = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "pairs")
    assert {row["ahead"].text for row in table.rows} == {"no separation"}
    assert all(pair.ahead is None for pair in page.readings.pairs)


_UNREAD = (
    "The campaign names no control, so a margin of 5 points and floor tone/rate_drop/* declared on tone are read "
    "against nothing and not read; its arms are held to its bars."
)


def _unread(bundle: AnalysisContextBundle) -> AnalysisContextBundle:
    assert bundle.declared is not None
    return bundle.model_copy(
        update={"declared": bundle.declared.model_copy(update={"unread_without_control": _UNREAD})}
    )


async def test_a_margin_and_floors_stored_without_a_control_are_said_unread_on_both_pages() -> None:
    page = await kind_page_of("tone", bundle=_unread)
    assert _UNREAD in _scorecard(page)
    (fold,) = page.aspects
    decision = [block for block in fold.report.blocks if block.section == "decision"]
    assert [block.text for block in decision if isinstance(block, DisclosureBlock)] == [_UNREAD]


def _frontier(
    tone: MeasuredKind, decision: FrontierCostDecision, tied: bool
) -> Callable[[AnalysisContextBundle], AnalysisContextBundle]:
    """The tone bundle with its frontier reading alpha as clearing the bar, by ``decision``, beta beside it when tied."""

    def change(bundle: AnalysisContextBundle) -> AnalysisContextBundle:
        ties = (
            [
                FrontierCostTie(
                    variant_key=tone.keys["beta"],
                    model="beta",
                    variant_identity_version=1,
                    production_replicating_cost=0.0005,
                )
            ]
            if tied
            else []
        )
        verdict = FrontierVerdict(
            variant_key=tone.keys["alpha"],
            model="alpha",
            pass_hat_k=0.95,
            variant_identity_version=1,
            cost_decision=decision,
            tied_with=ties,
        )
        (subject,) = bundle.frontier.subjects
        frontier = bundle.frontier.model_copy(update={"subjects": [subject.model_copy(update={"verdict": verdict})]})
        return bundle.model_copy(update={"frontier": frontier})

    return change


@pytest.mark.parametrize(
    ("decision", "tied", "said"),
    [
        (
            "not_separated",
            True,
            "Arms shown to clear the bar on tone: alpha and beta; none is shown cheaper than the rest.",
        ),
        (
            "untested",
            True,
            "Arms shown to clear the bar on tone: alpha and beta; too few priced cases to show any of them cheaper.",
        ),
        (
            "shown_cheapest",
            False,
            "Cheapest arm shown to clear the bar on tone: alpha, shown cheaper than every other that cleared.",
        ),
        ("only_cleared", False, "Only arm shown to clear the bar on tone at a known cost: alpha."),
    ],
)
async def test_the_frontier_line_says_what_the_frontier_read_and_names_every_tied_arm(
    decision: FrontierCostDecision, tied: bool, said: str
) -> None:
    tone = await kind_fixture("tone")
    page = await kind_page_of("tone", bundle=_frontier(tone, decision, tied))
    lines = [
        block.text for block in page.blocks if isinstance(block, DisclosureBlock) and "clear the bar" in block.text
    ]
    assert lines == [said]


async def test_a_change_whose_control_alone_ran_says_its_margin_and_floors_wait_for_a_candidate() -> None:
    campaign, (billing,) = await campaign_fixture("billing")
    control = campaign.control
    assert control is not None
    read = campaign_readings(campaign, {"billing": billing.bundle}, arms=[control])
    assert read.kind == "measurement"
    report = build_code_only_report(billing.bundle, assembled_at=ASSEMBLED_AT, campaign_name="support")
    page = build_campaign_report(campaign, read, {"billing": report}, None)
    said = f"Only the control, {CONTROL_LABEL}, has run, so the margin on billing and 1 floor are read once a candidate runs."
    assert said in _scorecard(page)


async def test_a_measurement_states_its_pass_rate_as_a_count_and_each_rig_apart() -> None:
    def two_rigs(bundle: AnalysisContextBundle) -> AnalysisContextBundle:
        moved = [cell.model_copy(update={"apparatus_class_id": "f" * 64}) for cell in bundle.cell_measures]
        return bundle.model_copy(update={"cell_measures": [*bundle.cell_measures, *moved]})

    one = await kind_page_of("solo")
    table = _table(one, "measures")
    assert "setup" not in {column.key for column in table.columns}
    # A17: the count is of trials, said so where they are not the cases beside it.
    assert (table.rows[0]["value"].text, table.rows[0]["cases"].text) == ("45 of 60 trials", "20")
    assert table.rows[0]["interval"].text.endswith("%]")
    page = await kind_page_of("solo", bundle=two_rigs)
    table = _table(page, "measures")
    setups = [row["setup"].text for row in table.rows]
    assert setups[0] == "every setup" and len(set(setups[1:])) == 2
    # The matrix holds one figure a cell: a measure read under two setups is marked, its figures in the fold.
    (row,) = _table(page, "scorecard").rows
    assert {cell.text for key, cell in row.items() if key != "aspect" and cell.value is None} == {"— (split by setup)"}


# =============================================================================
# The matrix by kind, over the shared shapes
# =============================================================================


@pytest.mark.parametrize(
    ("build", "rows", "columns"),
    [
        # A change: aspects × candidates, the control's rate and the cases beside each aspect.
        (
            s1_change,
            ["billing", "tone", "refunds"],
            ["Aspect", "Cases", "▸ production (%)", "v2 (pts)", "v3 (pts)", "v4 (pts)"],
        ),
        # A choice: aspects × arms.
        (s2_choice, ["answers"], ["Aspect", "Cases", "alpha (%)", "beta (%)", "gamma (%)"]),
        # A measurement: aspects × measures, its one arm named in the title.
        (s3_measurement, ["accuracy", "clarity"], None),
        # One candidate on one aspect: a one-row matrix.
        (s4_change, ["billing"], ["Aspect", "Cases", "▸ production (%)", "v2 (pts)"]),
    ],
    ids=["s1", "s2", "s3", "s4"],
)
def test_the_scorecard_is_one_matrix_of_aspects_by_the_kind_s_columns_with_its_rows_folded_under_it(
    build: Callable[[], Shape], rows: list[str], columns: list[str] | None
) -> None:
    shape = build()
    page = shape.page()
    matrix = _table(page, "scorecard")
    assert matrix.section == "scorecard" and not matrix.details
    assert [row["aspect"].text for row in matrix.rows] == rows
    if columns is not None:
        assert [column.header for column in matrix.columns] == columns
    else:
        assert matrix.title == "solo by aspect"
        assert [column.header for column in matrix.columns][:2] == ["Aspect", "pass rate (%)"]
    # The long form is folded, under the matrix, in the same section.
    folded = [
        block
        for block in page.blocks
        if isinstance(block, TableBlock) and block.details and block.section == "scorecard"
    ]
    assert [block.name for block in folded] == [
        {"change": "aspects", "choice": "pairs", "measurement": "measures"}[shape.readings.kind]
    ]
    assert page.blocks.index(matrix) < page.blocks.index(folded[0])
    # None of the scorecard's tables sits on the first screen's scoreboard any more, and no chart is drawn for it.
    owned = {"scorecard", "aspects", "pairs", "measures", "to_settle", "bars"}
    assert not [
        block
        for block in page.blocks
        if isinstance(block, TableBlock) and block.name in owned and block.section != "scorecard"
    ]
    assert not [block for block in page.blocks if isinstance(block, ChartBlock) and block.section == "scorecard"]


def test_a_change_s_cell_is_its_change_its_interval_against_the_margin_and_its_reading() -> None:
    shape = s1_change()
    matrix = _table(shape.page(), "scorecard")
    billing = _by_arm(matrix, matrix.rows[0])
    (v2,) = [row for row in shape.readings.on_aspect("billing") if row.arm_label == "v2"]
    assert billing["v2"].text == "-60 (shown worse)"
    assert (billing["v2"].low, billing["v2"].high) == v2.drop_interval
    assert billing["v2"].margin == 10 and billing["v2"].zero_line
    assert billing["production"].text == "90"
    assert matrix.rows[0]["cases"].text == "20"


def test_a_cell_is_marked_with_its_floor_s_status_only_where_the_floor_reads_otherwise() -> None:
    """A16 on S1: tone's floor reads `reply.cites_policy` alone, so v4 breaks it while its pass rate is unclear, and v2's
    gain is shown while that check leaves the floor unclear. Billing's floor reads as its rows do, so it marks none."""
    matrix = _table(s1_change().page(), "scorecard")
    flags = {row["aspect"].text: {arm: cell.flag for arm, cell in _by_arm(matrix, row).items()} for row in matrix.rows}
    assert flags["tone"] == {
        "production": None,
        "v2": "shown better; floor unclear",
        "v3": "unclear",
        "v4": "unclear; floor broken",
    }
    assert flags["billing"] == {"production": None, "v2": "shown worse", "v3": "unclear", "v4": "unclear"}


def test_a_control_that_cannot_move_is_marked_and_one_that_can_is_not() -> None:
    """R8 on S1: production passes no refunds case, so with a 10-point margin no drop past it can show; at 90% and 50%
    on billing and tone it can move either way."""
    matrix = _table(s1_change().page(), "scorecard")
    controls = {row["aspect"].text: _by_arm(matrix, row)["production"].text for row in matrix.rows}
    assert controls == {"billing": "90", "tone": "50", "refunds": "0 (cannot show a drop)"}


async def test_a_control_passing_every_trial_cannot_show_a_gain() -> None:
    page = await page_of("steady")
    matrix = _table(page, "scorecard")
    (row,) = matrix.rows
    # At 100% against a 20-point margin a drop can still show, so only the gain is marked.
    assert _by_arm(matrix, row)[CONTROL_LABEL].text == "100 (cannot show a gain)"


def _uneven_control(shares: list[float], passed: int, trials: int) -> Shape:
    """S4 with a 6-point margin and its control's pass rate read over uneven trials per case: ``shares`` its per-case
    shares on ten cases, ``passed`` of ``trials`` its trial count; v2 fails every case."""
    shape = s4_change()
    bundle = shape.bundles["billing"]
    contrasts = [
        contrast.model_copy(
            update={"case_values": (shares, [0.0] * len(shares)), "case_trials": None, "case_passed": None}
        )
        if contrast.reading == "trial"
        else contrast
        for contrast in bundle.floor_contrasts
    ]
    changed = bundle.model_copy(update={"floor_contrasts": contrasts})
    (row,) = shape.readings.readings
    read = row.model_copy(
        update={"margin": 6.0, "control_passed": passed, "control_trials": trials, "control_cases": 10}
    )
    return replace(
        shape,
        readings=shape.readings.model_copy(update={"readings": [read]}),
        reports={"billing": shape.reports["billing"].model_copy(update={"bundle": changed})},
    )


def test_cannot_show_a_drop_is_decided_on_the_per_case_shares_a_drop_is_read_on() -> None:
    """The review's case: ten control cases on a 6-point margin, one with a single trial that passed and nine with two
    failing trials each. Its trial rate is 1 of 19, about 5.3%, at or below the margin, but its per-case shares
    average 10%, so a candidate failing every case drops 10 points: a drop past the margin can show, and the cell is
    not marked. With every share nought, none can."""
    uneven = _uneven_control([1.0] + [0.0] * 9, passed=1, trials=19)
    (row,) = _table(uneven.page(), "scorecard").rows
    assert _by_arm(_table(uneven.page(), "scorecard"), row)["production"].text == "5.3"
    nothing = _uneven_control([0.0] * 10, passed=0, trials=19)
    matrix = _table(nothing.page(), "scorecard")
    assert _by_arm(matrix, matrix.rows[0])["production"].text == "0 (cannot show a drop)"


def test_a_floor_whose_unclear_rate_has_nothing_to_project_names_no_rate_and_no_count() -> None:
    """S1's billing floor reads every rate. With v4's `reply.cites_policy` contrast carrying no per-case values, that
    rate cannot be projected, so the floor has no count, and the table names no deciding rate beside blank counts."""
    shape = s1_change()
    key = shape.key("v4")
    bundle = shape.bundles["billing"]
    families = [
        family.model_copy(
            update={
                "comparisons": [
                    comparison.model_copy(update={"case_values": None, "bounded_interval": None})
                    if comparison.name.endswith("reply.cites_policy") and comparison.contrast.variant_key == key
                    else comparison
                    for comparison in family.comparisons
                ]
            }
        )
        for family in bundle.multiple_comparisons.families
    ]
    changed = bundle.model_copy(
        update={"multiple_comparisons": bundle.multiple_comparisons.model_copy(update={"families": families})}
    )
    reports = {**shape.reports, "billing": shape.reports["billing"].model_copy(update={"bundle": changed})}
    page = build_campaign_report(shape.campaign, shape.readings, reports, None)
    rows = {(row["aspect"].text, row["arm"].text, row["shows"].text): row for row in _table(page, "to_settle").rows}
    floor = rows[("billing", "v4", "the 10-point floor holds")]
    assert (floor["decided_by"].value, floor["none"].value, floor["observed"].value) == (None, None, None)
    assert floor["why"].value  # the floor's own reason, so the row still says why it is open


async def test_the_level_line_counts_only_the_readings_corrected_together() -> None:
    """With no margin on billing, v2's drops are read on tone alone while its gains are read on both: the line says
    each family's own size."""
    said = _scorecard(await page_of("billing", "tone", margins={"billing": None}))
    assert (
        "Intervals here are wider than a plain 95% interval, because each arm's gains on 2 aspects, and its drops on 1, "
        "are each corrected together." in said
    )


def test_cases_and_trials_are_stated_apart_where_an_arm_ran_a_case_more_than_once() -> None:
    """A17: v3 ran tone twice per case, so its passes are of 40 trials over the row's 20 cases; every other arm ran
    once, and its count reads as before. A choice's pairs say it the same way."""
    rows = {(row["aspect"].text, row["arm"].text): row for row in _table(s1_change().page(), "aspects").rows}
    assert (rows[("tone", "v3")]["cases"].text, rows[("tone", "v3")]["passed"].text) == ("20", "20 of 40 trials")
    assert rows[("tone", "v3")]["control"].text == "10 of 20"
    assert rows[("billing", "v2")]["passed"].text == "6 of 20"
    (first, *_) = _table(s2_choice().page(), "pairs").rows
    assert (first["cases"].text, first["first_passed"].text) == ("20", "53 of 60 trials")


def test_one_sizing_table_holds_the_row_s_count_and_the_floor_s_with_the_rate_that_decides_it() -> None:
    """A18 on S1: billing's floor reads every rate, so for v4 it is decided by `reply.cites_policy` and needs more cases
    as observed than v4's pass rate alone; for v3 the pass rate decides both, and the two counts agree. The floor's
    counts are the aspect fold's own."""
    page = s1_change().page()
    settle = _table(page, "to_settle")
    rows = {(row["aspect"].text, row["arm"].text, row["shows"].text): row for row in settle.rows}
    row, floor = (
        rows[("billing", "v4", "no worse than the 10-point margin")],
        rows[("billing", "v4", "the 10-point floor holds")],
    )
    assert (row["decided_by"].text, row["observed"].text) == ("pass rate", "31 cases, $0.22")
    assert (floor["decided_by"].text, floor["observed"].text) == ("`reply.cites_policy`", "41 cases, $0.29")
    v3 = [rows[("billing", "v3", shows)] for shows in ("no worse than the 10-point margin", "the 10-point floor holds")]
    assert {(cells["none"].text, cells["observed"].text) for cells in v3} == {("75 cases, $0.53", "233 cases, $1.6")}
    (fold,) = [fold for fold in page.aspects if fold.aspect == "billing"]
    projected = next(
        block for block in fold.report.blocks if isinstance(block, TableBlock) and block.name == "projection"
    )
    folded = {row["arm"].text: (row["none"].text, row["observed"].text) for row in projected.rows}
    assert folded["v4"] == (floor["none"].text, floor["observed"].text)
    assert "same count" not in campaign_report_markdown(page)
    # A row already settled is never to settle, and a held floor has nothing left to show.
    assert ("billing", "v2", "no worse than the 10-point margin") not in rows
    assert not [key for key in rows if key[:2] == ("refunds", "v2")]


@pytest.mark.parametrize("build", [s1_change, s4_change], ids=["s1", "s4"])
def test_the_level_lines_are_counted_off_the_page_s_own_arms_and_aspects(build: Callable[[], Shape]) -> None:
    shape = build()
    said = _scorecard(shape.page())
    arms = len({row.arm for row in shape.readings.readings})
    if arms == 1:
        # S4: one candidate on one aspect is a family of one, read as a floor reads.
        assert "The interval is read at 95%, as a floor's is." in said
        assert "For any one arm, the chance of being wrongly shown better is at most 2.5%" in said
        assert "not across the" not in said
    else:
        assert "because each arm's 3 readings are corrected together" in said
        assert "Each is read at 98.33%, and the reading beside it always agrees with it." in said
        assert f"not across the {arms} arms, so the chance that some arm is wrongly shown better is up to 7.5%" in said
    assert "does not rest on it" not in said


def _with_trial_contrast(
    shape: Shape, aspect: str, arm: str, change: Callable[[FamilyComparison], FamilyComparison]
) -> dict[str, Report]:
    """The shape's aspect pages, the one on ``aspect`` with its pass-rate contrast of ``arm`` changed."""
    key = shape.key(arm)
    bundle = shape.bundles[aspect]
    families = [
        family.model_copy(
            update={
                "comparisons": [
                    change(comparison)
                    if comparison.reading == "trial" and comparison.contrast.variant_key == key
                    else comparison
                    for comparison in family.comparisons
                ]
            }
        )
        for family in bundle.multiple_comparisons.families
    ]
    changed = bundle.model_copy(
        update={"multiple_comparisons": bundle.multiple_comparisons.model_copy(update={"families": families})}
    )
    return {**shape.reports, aspect: shape.reports[aspect].model_copy(update={"bundle": changed})}


def _lines(page: CampaignReport) -> list[str]:
    return [
        block.text
        for block in page.blocks
        if isinstance(block, DisclosureBlock) and "same per-case values" in block.text
    ]


def test_a_contrast_reading_a_row_s_own_values_to_another_verdict_is_reconciled_in_one_line() -> None:
    """A19: tone's v2 row is shown better; were its fold's pass-rate contrast, over the very same per-case values, not
    separated, the scorecard says both and how each interval was read. S1 as built has no such disagreement."""
    shape = s1_change()
    assert _lines(shape.page()) == []
    reports = _with_trial_contrast(shape, "tone", "v2", lambda c: c.model_copy(update={"verdict": "not_separated"}))
    page = build_campaign_report(shape.campaign, shape.readings, reports, None)
    (line,) = _lines(page)
    assert line.startswith(
        "On tone, v2's row reads shown better and its contrast on the pass rate reads not separated, over the same "
        "per-case values. The row reads a bounded interval at 98.33%, corrected across v2's aspects; the contrast a t "
        "interval at "
    )
    assert line.endswith(", corrected within the aspect's family.")
    assert line in _scorecard(page)


def test_a_contrast_showing_a_drop_the_row_does_not_say_the_row_counts_one_only_past_its_margin() -> None:
    shape = s1_change()
    reports = _with_trial_contrast(shape, "refunds", "v2", lambda c: c.model_copy(update={"verdict": "regressed"}))
    (line,) = _lines(build_campaign_report(shape.campaign, shape.readings, reports, None))
    assert line.startswith(
        "On refunds, v2's row reads no worse than the margin and its contrast on the pass rate reads regressed"
    )
    assert "a bounded interval at 95%, on its own;" in line
    assert line.endswith("The row counts a drop only past the 10-point margin.")


def test_a_contrast_over_other_values_is_not_reconciled_whatever_its_verdict() -> None:
    def elsewhere(comparison: FamilyComparison) -> FamilyComparison:
        assert comparison.case_values is not None
        control, contrast = comparison.case_values
        return comparison.model_copy(
            update={"verdict": "not_separated", "case_values": (control, [1.0 - contrast[0], *contrast[1:]])}
        )

    shape = s1_change()
    reports = _with_trial_contrast(shape, "tone", "v2", elsewhere)
    assert _lines(build_campaign_report(shape.campaign, shape.readings, reports, None)) == []


# =============================================================================
# The seam with the aggregates: the equal-weight row
# =============================================================================


@pytest.mark.parametrize("name", ["s1", "s2"])
def test_the_equal_weight_row_closes_the_matrix_each_arm_s_cell_in_its_own_column(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    shape = SHAPES[name]()
    figures = {arm.label: float(index) for index, arm in enumerate(shape.arms)}
    cells = {shape.key(label): TableCell(value=value, unit=_unit(shape, label)) for label, value in figures.items()}
    monkeypatch.setattr(aggregate, "equal_weight_cells", lambda campaign, readings, reports: cells)
    matrix = _table(shape.page(), "scorecard")
    *aspects, across = matrix.rows
    assert [row["aspect"].text for row in aspects] == [aspect.label for aspect in shape.campaign.aspects]
    assert across["aspect"].text == aggregate.EQUAL_WEIGHT_ROW and across["cases"].value is None
    assert {arm: cell.value for arm, cell in _by_arm(matrix, across).items()} == figures
    # Each figure is placed under the header that names its unit, so it prints as the column's other cells do.
    assert {cell.unit for cell in _by_arm(matrix, across).values()} == {""}
    # The long form holds no such row: the figure across aspects is the matrix's alone.
    folded = _table(shape.page(), "aspects" if name == "s1" else "pairs")
    assert aggregate.EQUAL_WEIGHT_ROW not in {row["aspect"].text for row in folded.rows}


@pytest.mark.parametrize("name", ["s1", "s2"])
def test_an_equal_weight_cell_for_an_arm_the_matrix_has_no_column_for_is_refused_not_dropped(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    shape = SHAPES[name]()
    cells = {shape.arms[0].variant_key: TableCell(value=1.0), "not-an-arm": TableCell(value=2.0)}
    monkeypatch.setattr(aggregate, "equal_weight_cells", lambda campaign, readings, reports: cells)
    with pytest.raises(ValueError, match="names arm\\(s\\) not-an-arm"):
        shape.page()


def _unit(shape: Shape, label: str) -> str:
    """The unit an arm's column holds: a change's control and a choice's arms a pass rate in percent, a change's
    candidates a change in points."""
    control = next((arm.label for arm in shape.arms if arm.role == "control"), None)
    return PERCENT if shape.readings.kind == "choice" or label == control else POINTS


@pytest.mark.parametrize(("name", "label"), [("s1", "production"), ("s1", "v2"), ("s2", "alpha")])
def test_an_equal_weight_cell_in_another_unit_than_its_column_is_refused(
    name: str, label: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pass rate under a change's points column, or a change under a percent column, would print under the wrong
    unit: the matrix refuses it, and refuses a cell stating none or holding words."""
    shape = SHAPES[name]()
    wrong = POINTS if _unit(shape, label) == PERCENT else PERCENT
    for cell in (TableCell(value=1.0, unit=wrong), TableCell(value=1.0), TableCell(value="about 5")):
        monkeypatch.setattr(
            aggregate, "equal_weight_cells", lambda campaign, readings, reports, c=cell: {shape.key(label): c}
        )
        with pytest.raises(ValueError, match=f"cell for arm {shape.key(label)} states"):
            shape.page()
    # A figure withheld carries no value and no unit, and is placed as a dash.
    withheld = TableCell(value=None, flag="withheld")
    monkeypatch.setattr(
        aggregate, "equal_weight_cells", lambda campaign, readings, reports: {shape.key(label): withheld}
    )
    matrix = _table(shape.page(), "scorecard")
    assert _by_arm(matrix, matrix.rows[-1])[label].text == "— (withheld)"


@pytest.mark.parametrize("name", ["s1", "s2", "s3"])
def test_an_equal_weight_cell_where_no_matrix_holds_arms_is_refused_not_dropped(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A measurement's matrix holds measures, and a change or a choice with nothing read lays out none: a figure
    across aspects handed back there is refused."""
    shape = SHAPES[name]()
    if shape.readings.kind != "measurement":
        emptied = shape.readings.model_copy(update={"readings": [], "pairs": [], "floors": []})
        shape = replace(shape, readings=emptied)
    key = shape.arms[0].variant_key
    monkeypatch.setattr(
        aggregate, "equal_weight_cells", lambda campaign, readings, reports: {key: TableCell(value=1.0, unit=PERCENT)}
    )
    with pytest.raises(ValueError, match="lays out no matrix to hold it"):
        shape.page()


def test_with_no_equal_weight_cell_the_matrix_has_only_its_aspects() -> None:
    shape = s1_change()
    assert aggregate.equal_weight_cells(shape.campaign, shape.readings, shape.reports) == {}
    assert aggregate.EQUAL_WEIGHT_ROW not in {row["aspect"].text for row in _table(shape.page(), "scorecard").rows}


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_every_line_the_scorecard_writes_keeps_the_style_rules(name: str) -> None:
    shape = SHAPES[name]()
    labels = [arm.label for arm in shape.arms]
    # An arm's label and a measure's declared name head a column as names, never as a header's words.
    measures = {reading.name for reading in shape.readings.measures}
    names = [*labels, *(f"▸ {label}" for label in labels), *measures]
    assert lint_page(_scorecard(shape.page()), names=names) == []
