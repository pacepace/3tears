"""What code read on each aspect of a campaign's page, by its kind: a change's table, what would settle each unclear
row and the lines under it; a choice's pairs and frontier; a measurement's measures."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from threetears.evals.analysis import (
    AnalysisContextBundle,
    CampaignReadings,
    build_code_only_report,
    build_campaign_report,
    campaign_readings,
    campaign_report_markdown,
)
from threetears.evals.analysis.lenses.frontier import FrontierCostDecision, FrontierCostTie, FrontierVerdict
from threetears.evals.analysis.report.campaign_page import CHOICE_NOTE, campaign_verdict
from threetears.evals.analysis.report.model import (
    DisclosureBlock,
    TableBlock,
)

from packages.evals.tests.campaign_support import (
    ASSEMBLED_AT,
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


async def test_an_aspect_with_no_margin_is_said_never_shown_worse_and_has_no_row_to_settle_against_one() -> None:
    page = await page_of("billing", "tone", margins={"billing": None})
    first = first_screen_markdown(page)
    assert "No margin was declared on billing, so no arm is ever shown worse there." in first
    (billing,) = page.readings.on_aspect("billing")
    assert billing.reading == "unclear" and "no margin was declared" in billing.reason
    # Its row would be a row of dashes under "To show no worse than the margin": it has none to show it within.
    assert not [block for block in page.blocks if isinstance(block, TableBlock) and block.name == "to_settle"]
    assert "To show no worse than the margin" not in first


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


async def test_a_floor_broken_beside_a_row_not_shown_worse_says_the_two_read_different_intervals() -> None:
    # A 70-point margin on billing: its 60-point drop is not past it, while the 10-point floor is broken.
    page = await page_of("billing", "tone", margins={"billing": 70.0})
    (row,) = page.readings.on_aspect("billing")
    assert row.reading == "unclear"
    assert (
        "The floor broken on billing for v2 reads its pass rate at 95% on its own, while its row reads only the pass "
        "rate, corrected across v2's aspects." in first_screen_markdown(page)
    )


async def test_an_unclear_row_s_cases_show_it_no_worse_than_the_margin_and_not_better() -> None:
    page = await page_of("billing", "tone", margins={"billing": 70.0})
    settle = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "to_settle")
    assert settle.title == "To show no worse than the margin"
    note = page.blocks[page.blocks.index(settle) + 1]
    assert isinstance(note, DisclosureBlock)
    assert "would not show the arm better, so its row stays unclear unless its gain clears too" in note.text


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
    first = first_screen_markdown(page)
    assert (
        "A row no worse than the margin reads its interval at 95% on its own, as a floor does: a candidate is no "
        "worse only where it holds on every asked aspect, which needs no correction." in first
    )
    # The one corrected row is tone's, read in v2's family of two at 97.5%, so it is not called a floor's level; the
    # error-rate line is said whatever the rows read.
    assert "The interval is read at 97.5%, corrected across each arm's aspects." in first
    assert "Each arm's gains and its drops are each shown wrongly at most 2.5% of the time." in first
    assert "such an interval clearing zero is not a gain shown" not in first
    markdown = campaign_report_markdown(page)
    steady = markdown[markdown.index("## steady") : markdown.index("## tone")]
    assert "v2 no worse than the margin." in steady
    assert (
        campaign_verdict(page.readings, ["steady", "tone"])
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
    first = first_screen_markdown(page)
    assert "The interval is read at" not in first and "Each interval is read at" not in first
    assert "Each arm's gains and its drops are each shown wrongly at most 2.5% of the time." in first
    assert (
        "Nothing corrects across arms, so with several arms one is shown better or worse by chance more often." in first
    )
    assert (
        "Its gain is still read corrected across the arm's aspects, so such an interval clearing zero is not a gain "
        "shown." in first
    )


async def test_a_floor_held_beside_a_row_left_unclear_says_the_two_read_different_points() -> None:
    # The same interval against a 10-point margin is unclear while the 20-point floor holds.
    page = await page_of("steady", margins={"steady": 10.0})
    (row,) = page.readings.on_aspect("steady")
    (status,) = page.readings.floors
    assert row.reading == "unclear" and status.status == "held"
    assert (
        "The floor held on steady for v2 reads its pass rate against 20 points, while its row reads the pass rate "
        "against the 10-point margin." in first_screen_markdown(page)
    )
    settle = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "to_settle")
    assert [cells["aspect"].text for cells in settle.rows] == ["steady"]
    note = page.blocks[page.blocks.index(settle) + 1]
    assert isinstance(note, DisclosureBlock)
    assert "margin, read at 95%. A floor on the same rate and points gives the same count." in note.text


async def test_with_no_question_every_reading_is_a_lead() -> None:
    first = first_screen_markdown(await page_of("billing", "tone", question=""))
    assert "No question was declared, so every reading is a lead." in first


async def test_a_choice_shows_an_arm_ahead_on_one_aspect_and_no_separation_on_another() -> None:
    page = await kind_page_of("tone", "billing", memo=choice_memo())
    assert page.readings.kind == "choice"
    table = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "pairs")
    assert [(row["aspect"].text, row["ahead"].text) for row in table.rows] == [
        ("tone", "alpha"),
        ("billing", "no separation"),
    ]
    first = first_screen_markdown(page)
    assert CHOICE_NOTE in first
    # The writer's summary and recommendation, then the pairs; no control-based table.
    assert first.index(SUMMARY) < first.index("**Recommended:** beta (high confidence)")
    assert first.index("**Recommended:** beta (high confidence)") < first.index("**By aspect, pair by pair**")
    assert "**By aspect**\n" not in first
    markdown = campaign_report_markdown(page)
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
    assert _UNREAD in first_screen_markdown(page)
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
    assert said in first_screen_markdown(page)


async def test_a_measurement_states_its_pass_rate_as_a_count_and_each_rig_apart() -> None:
    def two_rigs(bundle: AnalysisContextBundle) -> AnalysisContextBundle:
        moved = [cell.model_copy(update={"apparatus_class_id": "f" * 64}) for cell in bundle.cell_measures]
        return bundle.model_copy(update={"cell_measures": [*bundle.cell_measures, *moved]})

    one = await kind_page_of("solo")
    table = next(block for block in one.blocks if isinstance(block, TableBlock) and block.name == "measures")
    assert "setup" not in {column.key for column in table.columns}
    assert table.rows[0]["value"].text == "45 of 60" and table.rows[0]["interval"].text.endswith("%]")
    page = await kind_page_of("solo", bundle=two_rigs)
    table = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "measures")
    setups = [row["setup"].text for row in table.rows]
    assert setups[0] == "every setup" and len(set(setups[1:])) == 2
