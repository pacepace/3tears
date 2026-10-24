"""A campaign's page opens with its loud lines: every bar an arm misses, every floor broken, each followed by the
writer's why where the memo recommends the arm over it."""

from __future__ import annotations

import json


from threetears.evals.analysis import (
    CampaignReadings,
    serialize_report,
)
from threetears.evals.analysis.report.campaign_page import NO_CALL
from threetears.evals.analysis.report.model import (
    DisclosureBlock,
    TableBlock,
    TextBlock,
)
from threetears.evals.kernel.authored import (
    BarMissed,
    FloorOverride,
)

from packages.evals.tests.campaign_support import (
    CANDIDATE_LABEL,
)

from packages.evals.tests.campaign_page_support import (
    choice_memo,
    first_screen_markdown,
    header_html,
    kind_page_of,
    page_of,
    two_aspect_memo,
)


async def test_many_broken_floors_are_warned_of_one_line_per_arm() -> None:
    def four(readings: CampaignReadings) -> CampaignReadings:
        (status,) = readings.floors
        statuses = [
            status.model_copy(update={"aspect": aspect, "floor": f"{aspect}/rate_drop/*"})
            for aspect in ("billing", "tone", "refunds", "returns")
        ]
        return readings.model_copy(update={"floors": statuses})

    page = await page_of("billing", "tone", readings=four)
    alerts = [block for block in page.blocks if isinstance(block, DisclosureBlock) and block.source == "floor"]
    assert [alert.text for alert in alerts] == [
        "Floors broken for v2: its pass rate on billing; its pass rate on tone; its pass rate on refunds; its pass "
        "rate on returns."
    ]


async def test_a_measurement_s_missed_bar_is_first_on_the_page_and_no_arm_is_recommended() -> None:
    page = await kind_page_of("solo", memo=choice_memo("alpha"))
    assert page.readings.kind == "measurement"
    first = first_screen_markdown(page)
    assert first.startswith("**Bar missed on solo: alpha misses the bar on ")
    header = header_html(page)
    assert header.index('role="alert"') < header.index("<h1>")
    assert "Recommended" not in first and page.recommendation is None
    assert json.loads(serialize_report(page, "json"))["recommendation"] is None
    measures = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "measures")
    assert measures.rows[0]["measure"].text == "pass rate"
    bars = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "bars")
    assert [(row["arm"].text, row["status"].text) for row in bars.rows] == [("alpha", "missed")]
    assert NO_CALL not in first


async def test_a_missed_bar_warns_first_on_a_choice_too() -> None:
    first = first_screen_markdown(await kind_page_of("tone"))
    assert first.startswith("**Bar missed on tone: alpha misses the bar on ")


async def test_the_writer_s_why_for_an_overridden_floor_sits_beside_its_warning() -> None:
    memo = two_aspect_memo()
    over = FloorOverride(floor="billing/rate_drop/*", why="Billing refunds are rare this quarter.")
    recommended = memo.recommended()
    assert recommended is not None
    recommendation = recommended.model_copy(update={"arm": CANDIDATE_LABEL, "floors_overridden": [over]})
    page = await page_of("billing", "tone", memo=memo.model_copy(update={"recommendation": [recommendation]}))
    first = first_screen_markdown(page)
    warning = first.index("**Floor broken on billing")
    why = first.index("**The memo recommends v2 over it:** Billing refunds are rare this quarter.")
    assert warning < why < first.index("# support queue")
    header = header_html(page)
    assert header.index('role="alert"') < header.index('data-role="override"') < header.index("<h1>")


async def test_a_why_for_a_missed_bar_sits_beside_its_warning() -> None:
    memo = choice_memo("alpha")
    recommended = memo.recommended()
    assert recommended is not None
    missed = BarMissed(aspect="tone", measure="cost_usd", why="A cent a reply is affordable at this volume.")
    recommendation = recommended.model_copy(update={"bars_missed": [missed]})
    page = await kind_page_of("tone", memo=memo.model_copy(update={"recommendation": [recommendation]}))
    first = first_screen_markdown(page)
    warning = first.index("**Bar missed on tone: alpha")
    why = first.index("**The memo recommends alpha over it:** A cent a reply is affordable at this volume.")
    assert warning < why < first.index("# support queue")


async def test_a_why_for_an_overridden_floor_answers_only_the_arm_the_memo_recommends() -> None:
    def both(readings: CampaignReadings) -> CampaignReadings:
        (status,) = readings.floors
        other = status.model_copy(update={"arm": "c" * 64, "arm_label": "v3"})
        return readings.model_copy(update={"floors": [status, other]})

    memo = two_aspect_memo()
    over = FloorOverride(floor="billing/rate_drop/*", why="Billing refunds are rare this quarter.")
    recommended = memo.recommended()
    assert recommended is not None
    recommendation = recommended.model_copy(update={"arm": CANDIDATE_LABEL, "floors_overridden": [over]})
    page = await page_of(
        "billing", "tone", memo=memo.model_copy(update={"recommendation": [recommendation]}), readings=both
    )
    header = [block for block in page.blocks if block.section == "header"]
    kinds = [
        "why" if isinstance(block, TextBlock) and block.role == "override" else block.text
        for block in header
        if isinstance(block, TextBlock)
        and block.role == "override"
        or isinstance(block, DisclosureBlock)
        and block.source == "floor"
    ]
    # v2's warning, its why, then v3's warning with none: the why answers v2 breaking the floor, not v3.
    assert len(kinds) == 3 and kinds[1] == "why"
    assert kinds[0].startswith(f"Floor broken on billing: {CANDIDATE_LABEL}'s") and kinds[2].startswith(
        "Floor broken on billing: v3's"
    )
