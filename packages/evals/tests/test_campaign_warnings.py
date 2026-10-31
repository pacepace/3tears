"""What warns on a campaign's page, under its question and answer: one plain line per arm for the bars it misses, one
per arm for the floors it breaks, one per arm and decision for the guardrails it is not shown safe on, each saying
what it means for the arm, each followed by the writer's why where the memo recommends the arm over it."""

from __future__ import annotations

import json

import pytest

from threetears.evals.analysis import (
    CampaignReadings,
    serialize_report,
)
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.report.campaign.names import AspectNames
from threetears.evals.analysis.report.campaign.warnings import guardrail_warnings
from threetears.evals.kernel.surface import GuardrailCell, GuardrailCheck, GuardrailReadings
from threetears.evals.analysis.report.campaign_page import NO_CALL, campaign_warnings
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
from packages.evals.tests.report_shapes_support import SHAPES


def _warned(blocks: list[object], source: str) -> list[str]:
    return [block.text for block in blocks if isinstance(block, DisclosureBlock) and block.source == source]


async def test_floors_an_arm_breaks_on_many_aspects_are_one_line_naming_them() -> None:
    def four(readings: CampaignReadings) -> CampaignReadings:
        (status,) = readings.floors
        statuses = [
            status.model_copy(update={"aspect": aspect, "floor": f"{aspect}/rate_drop/*"})
            for aspect in ("billing", "tone", "refunds", "returns")
        ]
        return readings.model_copy(update={"floors": statuses, "aspects": [status.aspect for status in statuses]})

    page = await page_of("billing", "tone", readings=four)
    assert _warned(page.blocks, "floor") == ["Adopting v2 overrides the 10-point floors it breaks on every aspect."]


def test_each_arm_breaking_a_floor_has_its_own_line_and_floors_of_two_margins_are_not_sized() -> None:
    page = SHAPES["s1"]().page()
    assert _warned(page.blocks, "floor") == [
        "Adopting v2 overrides the 10-point floor it breaks on billing.",
        "Adopting v4 overrides the 15-point floor it breaks on tone.",
    ]
    readings = page.readings
    v2 = next(status for status in readings.broken_floors() if status.arm_label == "v2")
    moved = v2.model_copy(update={"aspect": "tone", "floor": "tone/rate_drop/x", "margin": 15.0})
    both = readings.model_copy(update={"floors": [*readings.floors, moved]})
    names = AspectNames.of(fold.report.bundle for fold in page.aspects)
    assert campaign_warnings(both, names)[0] == (
        "Adopting v2 overrides the floors it breaks on 2 of 3 aspects (billing and tone)."
    )


@pytest.mark.parametrize(("name", "line"), [("s2", "Adopting alpha overrides the 2 bars it misses on answers.")])
def test_an_arm_missing_several_bars_has_one_line_counting_them(name: str, line: str) -> None:
    page = SHAPES[name]().page()
    assert _warned(page.blocks, "bar") == [line]
    # The listing line says what the page says, word for word.
    assert campaign_warnings(page.readings, AspectNames.of(fold.report.bundle for fold in page.aspects)) == [line]


def test_a_measurement_s_missed_bar_says_it_falls_short_never_that_adopting_it_overrides_it() -> None:
    page = SHAPES["s3"]().page()
    assert _warned(page.blocks, "bar") == ["solo falls short of the bar on clarity."]


async def test_a_measurement_s_missed_bar_warns_under_its_title_and_no_arm_is_recommended() -> None:
    page = await kind_page_of("solo", memo=choice_memo("alpha"))
    assert page.readings.kind == "measurement"
    first = first_screen_markdown(page)
    assert first.startswith("# ") and "**alpha falls short of the bar on solo.**" in first
    header = header_html(page)
    assert header.index("<h1>") < header.index('<p class="flag warning">') and 'role="alert"' not in header
    assert "Recommended" not in first and page.recommendation is None
    assert json.loads(serialize_report(page, "json"))["recommendation"] is None
    measures = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "measures")
    assert measures.rows[0]["measure"].text == "pass rate"
    bars = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "bars")
    assert [(row["arm"].text, row["status"].text) for row in bars.rows] == [("alpha", "missed")]
    assert NO_CALL not in first


async def test_a_missed_bar_warns_on_a_choice_too() -> None:
    first = first_screen_markdown(await kind_page_of("tone"))
    assert "**Adopting alpha overrides the bar it misses on tone.**" in first


async def test_the_writer_s_why_for_an_overridden_floor_sits_beside_its_warning() -> None:
    memo = two_aspect_memo()
    over = FloorOverride(floor="billing/rate_drop/*", why="Billing refunds are rare this quarter.")
    recommended = memo.recommended()
    assert recommended is not None
    recommendation = recommended.model_copy(update={"arm": CANDIDATE_LABEL, "floors_overridden": [over]})
    page = await page_of("billing", "tone", memo=memo.model_copy(update={"recommendation": [recommendation]}))
    first = first_screen_markdown(page)
    warning = first.index("**Adopting v2 overrides the 10-point floor it breaks on billing.**")
    why = first.index("**The memo recommends v2 over it:** Billing refunds are rare this quarter.")
    assert first.index(f"# {page.title}") < warning < why
    header = header_html(page)
    assert header.index("<h1>") < header.index('<p class="flag warning">') < header.index('data-role="override"')


async def test_a_why_for_a_missed_bar_sits_beside_its_warning() -> None:
    memo = choice_memo("alpha")
    recommended = memo.recommended()
    assert recommended is not None
    missed = BarMissed(aspect="tone", measure="cost_usd", why="A cent a reply is affordable at this volume.")
    recommendation = recommended.model_copy(update={"bars_missed": [missed]})
    page = await kind_page_of("tone", memo=memo.model_copy(update={"recommendation": [recommendation]}))
    first = first_screen_markdown(page)
    warning = first.index("**Adopting alpha overrides the bar it misses on tone.**")
    why = first.index("**The memo recommends alpha over it:** A cent a reply is affordable at this volume.")
    assert first.index(f"# {page.title}") < warning < why


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
    warned = [block for block in page.blocks if block.section == "warnings"]
    kinds = [
        "why" if isinstance(block, TextBlock) else block.text
        for block in warned
        if isinstance(block, TextBlock)
        and block.role == "override"
        or isinstance(block, DisclosureBlock)
        and block.source == "floor"
    ]
    # v2's warning, its why, then v3's warning with none: the why answers v2 breaking the floor, not v3.
    assert kinds == [
        f"Adopting {CANDIDATE_LABEL} overrides the 10-point floor it breaks on billing.",
        "why",
        "Adopting v3 overrides the 10-point floor it breaks on billing.",
    ]


def _guardrail(bundle: AnalysisContextBundle, name: str, arm: str) -> GuardrailCheck:
    """One judged guardrail ``name`` on ``bundle``, left undecided for the arm keyed ``arm`` against the control."""
    control = bundle.scoreboard[0]
    return GuardrailCheck(
        reading="judged",
        name=name,
        higher_is_better=True,
        control=GuardrailCell(variant_key=control.variant_key, apparatus_class_id="a", n_cases=4, mean=1.0),
        contrast=GuardrailCell(variant_key=arm, apparatus_class_id="a", n_cases=4, mean=0.9),
        test="paired",
        delta=-0.1,
        interval=(-0.9, 0.4),
        interval_basis="t",
        margin=0.0,
        margin_declared=False,
        decision="undecided",
        undecided_reason="the interval reaches both sides of 0",
    )


def test_each_guardrail_is_named_with_its_own_aspects_never_another_s() -> None:
    """v2 is undecided on one guardrail on billing and another on tone: the line pairs each with its own aspect,
    never both with both."""
    shape = SHAPES["s1"]()
    v2 = shape.key("v2")
    on = {"billing": "reply.polite", "tone": "reply.on_brand"}
    reports = {
        label: report.model_copy(
            update={
                "bundle": report.bundle.model_copy(
                    update={
                        "guardrails": GuardrailReadings(
                            dimensions=[on[label]], checks=[_guardrail(report.bundle, on[label], v2)]
                        )
                    }
                )
            }
        )
        if label in on
        else report
        for label, report in shape.reports.items()
    }
    names = AspectNames.of(report.bundle for report in reports.values())
    (line,) = guardrail_warnings(shape.campaign, shape.readings, reports, names=names)
    assert isinstance(line, DisclosureBlock) and line.section == "warnings"
    assert line.text == (
        "v2 is not yet shown safe on the reply polite guardrail on billing and the reply on brand guardrail on tone."
    )
