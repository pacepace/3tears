"""What warns on a campaign's page, under its question and answer: one plain line per arm for the bars it misses, one
per arm for the floors it breaks, one per arm and decision for the guardrails it is not shown safe on, each saying
what it means for the arm."""

from __future__ import annotations

import json
from typing import Any

import pytest

from threetears.evals import Eval, grader
from threetears.evals.analysis import (
    CampaignReadings,
    get_campaign,
    CampaignReport,
    serialize_report,
)
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.report.campaign.names import AspectNames

from threetears.evals.analysis.report.campaign.scorecard import READING_WORDS
from threetears.evals.analysis.report.campaign.warnings import guardrail_warnings
from threetears.evals.kernel.surface import GuardrailCell, GuardrailCheck, GuardrailReadings
from threetears.evals.analysis.report.campaign_page import NO_CALL, build_campaign_report, campaign_warnings
from threetears.evals.analysis.report.model import (
    DisclosureBlock,
    TableBlock,
)

from packages.evals.tests.campaign_support import (
    CANDIDATE_LABEL,
)

from packages.evals.tests.campaign_page_support import (
    first_screen_markdown,
    header_html,
    kind_page_of,
    page_of,
)
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.report_shapes_support import SHAPES, ArmTrials, AspectTrials, shape_of


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
    page = await kind_page_of("solo")
    assert page.readings.kind == "measurement"
    first = first_screen_markdown(page)
    assert first.startswith("# ") and "**alpha falls short of the bar on solo.**" in first
    header = header_html(page)
    assert header.index("<h1>") < header.index('<p class="flag warning">') and 'role="alert"' not in header
    assert "Recommended" not in first
    assert "recommendation" not in json.loads(serialize_report(page, "json"))
    measures = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "measures")
    assert measures.rows[0]["measure"].text == "pass rate"
    bars = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "bars")
    assert [(row["arm"].text, row["status"].text) for row in bars.rows] == [("alpha", "missed")]
    assert NO_CALL not in first


async def test_a_missed_bar_warns_on_a_choice_too() -> None:
    first = first_screen_markdown(await kind_page_of("tone"))
    assert "**Adopting alpha overrides the bar it misses on tone.**" in first


async def test_two_arms_breaking_one_floor_each_have_their_own_line() -> None:
    def both(readings: CampaignReadings) -> CampaignReadings:
        (status,) = readings.floors
        other = status.model_copy(update={"arm": "c" * 64, "arm_label": "v3"})
        return readings.model_copy(update={"floors": [status, other]})

    page = await page_of("billing", "tone", readings=both)
    warned = [block for block in page.blocks if block.section == "warnings"]
    assert [block.text for block in warned if isinstance(block, DisclosureBlock) and block.source == "floor"] == [
        f"Adopting {CANDIDATE_LABEL} overrides the 10-point floor it breaks on billing.",
        "Adopting v3 overrides the 10-point floor it breaks on billing.",
    ]
    assert {block.kind for block in warned} == {"disclosure"}, "code's lines alone, no writer's words beside them"


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


#: Thirty cases, as a first comparison holds: at thirty, a pass/fail guardrail's bounded interval is wider than five
#: points even where both arms pass it on every case, so one guarded at five is undecided for the sample's size.
_THIRTY = [{"n": index} for index in range(30)]


def _right(case: dict[str, int], answer: str) -> bool:
    """Whether the answer is right."""
    return answer.startswith("right")


def _no_leak(case: dict[str, int], answer: str) -> bool:
    """Whether the answer keeps the customer's secret."""
    return "SECRET" not in answer


def _answering(*, misses: int, leaks: int) -> Any:
    async def answer(case: dict[str, int]) -> str:
        return ("wrong" if case["n"] < misses else "right") + (" SECRET" if case["n"] < leaks else "")

    return answer


async def _guarded(**arms: Any) -> CampaignReport:
    """The campaign page of ``arms`` (the first the control) over thirty cases, graded by ``_right`` and the
    ``_no_leak`` guardrail at five points."""
    result = await Eval(_THIRTY, arms, [_right, grader(_no_leak, guard=0.05)], k=1).run(store=":memory:")
    return result.page


def _bold(page: CampaignReport) -> list[str]:
    return [line for line in first_screen_markdown(page).splitlines() if line.startswith("**")]


async def test_an_undecided_guardrail_no_arm_failed_says_the_cases_are_too_few_to_decide_it() -> None:
    """Both arms pass the guardrail on every case, and thirty cases cannot show the candidate within five points: the
    line says so, so the sample's size never reads as a defect of the arm."""
    page = await _guarded(current=_answering(misses=10, leaks=0), same=_answering(misses=10, leaks=0))
    (line,) = _warned(page.blocks, "guardrails")
    aspect = page.aspects[0].aspect
    assert line == (
        f"same is not yet shown safe on the no leak guardrail on {aspect}: neither arm failed it on any of 30 cases, "
        "too few cases to show same within its 5-point margin or beyond it."
    )


async def test_an_undecided_guardrail_counts_the_cases_each_arm_failed_it_on() -> None:
    """The counts are read off the trials: the candidate leaked on two cases, once each, and the control on none."""
    page = await _guarded(current=_answering(misses=10, leaks=0), slip=_answering(misses=10, leaks=2))
    (line,) = _warned(page.blocks, "guardrails")
    assert line.endswith(
        ": slip failed it on 2 of 30 cases and current on 0, a case counted once however many of its trials failed "
        "it, too few cases to show slip within its 5-point margin or beyond it."
    )


async def test_with_no_question_and_no_classifier_the_lead_is_the_first_grader_and_the_guardrail_warns_beside_it() -> (
    None
):
    """The one rule with no question asked and a function grader first: the eval files its graders' order on the
    aspect's design, so the first bold line reads that grader, in the aspect page's words, naming both arms and marked
    a lead; the guardrail's warning is the next, never in its place."""
    result = await Eval(
        _THIRTY,
        {"current": _answering(misses=10, leaks=0), "same": _answering(misses=4, leaks=0)},
        [_right, grader(_no_leak, guard=0.05)],
        k=1,
    ).run(store=":memory:")
    page = result.page
    (right,) = [
        verdict
        for verdict in page.aspects[0].report.verdicts
        if verdict.kind == "contrast" and verdict.name == "_right"
    ]
    lead, warning = _bold(page)
    assert lead == f"**{right.heading}: {right.words}, a lead.**"
    assert "same" in right.words and "current" in right.words
    assert warning.startswith("**same is not yet shown safe on the no leak guardrail")
    html = header_html(page)
    assert html.index('<p class="verdict" data-tone="open">') < html.index('<p class="flag warning">')


async def test_a_campaign_stating_no_grader_order_and_asking_no_question_leads_on_the_pass_rate() -> None:
    """A campaign stored with no graders' order, as every campaign filed before the order was, asking no question and
    reading no classifier, leads as it did: on the pass rate, every grader's reading together, marked a lead. The same
    campaign stating its order leads on its first grader."""
    result = await Eval(
        _THIRTY,
        {"current": _answering(misses=10, leaks=0), "same": _answering(misses=4, leaks=0)},
        [_right, grader(_no_leak, guard=0.05)],
        k=1,
    ).run(store=":memory:")
    stated = off_loop(get_campaign, result.host.storage, result.campaign_id, result.scope_id)
    unstated = stated.model_copy(
        update={
            "aspects": [
                aspect.model_copy(update={"design": aspect.design.model_copy(update={"measure_order": []})})
                if aspect.design is not None
                else aspect
                for aspect in stated.aspects
            ]
        }
    )
    reports = {fold.aspect: fold.report for fold in result.page.aspects}
    (row,) = result.page.readings.readings
    lead, _ = _bold(build_campaign_report(unstated, result.page.readings, reports))
    assert lead == f"**Pass rate: same against current {READING_WORDS[row.reading]}, a lead.**"
    led, _ = _bold(build_campaign_report(stated, result.page.readings, reports))
    assert led.startswith("**Right score: ") and led.endswith(", a lead.**")


async def test_with_no_question_the_lead_is_the_classifier_s_accuracy_in_the_aspect_page_s_words() -> None:
    """A field grader listed first is the classifier: the lead reads its accuracy, as the aspect's contrasts table
    words it, both arms named, never the pass rate of every grader together."""

    def labelled(miss: int) -> Any:
        async def answer(case: dict[str, int]) -> dict[str, str]:
            return {"label": "a" if case["n"] >= miss else "b", "note": "SECRET" if case["n"] < 0 else ""}

        return answer

    def no_secret(case: dict[str, Any], answer: dict[str, str]) -> bool:
        """Whether the answer keeps the customer's secret."""
        return "SECRET" not in answer["note"]

    cases = [{"n": index, "label": "a"} for index in range(30)]
    result = await Eval(
        cases, {"current": labelled(10), "same": labelled(4)}, ["label", grader(no_secret, guard=0.05)], k=1
    ).run(store=":memory:")
    (accuracy,) = [
        verdict
        for verdict in result.page.aspects[0].report.verdicts
        if verdict.kind == "contrast" and verdict.name == "accuracy"
    ]
    lead, warning = _bold(result.page)
    assert lead == f"**{accuracy.heading}: {accuracy.words}, a lead.**" and accuracy.heading == "Accuracy"
    assert "same" in accuracy.words and "current" in accuracy.words
    assert warning.startswith("**same is not yet shown safe on the no secret guardrail")


def test_with_a_declared_question_the_lead_is_its_measure_decided_ahead_of_the_guardrail() -> None:
    """A campaign asking its aspect a question leads on the question's measure, the pass rate, unmarked as a lead and
    set in the decided tone, with the guardrail the candidate is undecided on warned beside it."""
    shape = shape_of(
        "asked",
        [AspectTrials("billing", [ArmTrials("production", passing=6), ArmTrials("v2", passing=19)])],
        control=True,
    )
    v2 = shape.key("v2")
    reports = {
        label: report.model_copy(
            update={
                "bundle": report.bundle.model_copy(
                    update={
                        "guardrails": GuardrailReadings(
                            dimensions=["reply.polite"], checks=[_guardrail(report.bundle, "reply.polite", v2)]
                        )
                    }
                )
            }
        )
        for label, report in shape.reports.items()
    }
    assert all(report.bundle.question_declared for report in reports.values())
    page = build_campaign_report(shape.campaign, shape.readings, reports)
    (row,) = shape.readings.readings
    assert row.confirmatory
    lead, warning = _bold(page)
    assert lead == f"**Pass rate: v2 against production {READING_WORDS[row.reading]}.**"
    assert warning == "**v2 is not yet shown safe on the reply polite guardrail on billing.**"
    html = header_html(page)
    assert html.index('<p class="verdict" data-tone="decided">') < html.index('<p class="flag warning">')


async def test_a_page_no_guardrail_warns_on_leads_on_its_scorecard_with_no_lead_line() -> None:
    """With nothing to take the lead's place the first screen is as it was, the scorecard its first reading."""
    result = await Eval(
        _THIRTY, {"current": _answering(misses=10, leaks=0), "same": _answering(misses=4, leaks=0)}, [_right], k=1
    ).run(store=":memory:")
    assert _warned(result.page.blocks, "verdict") == [] and _bold(result.page) == []
