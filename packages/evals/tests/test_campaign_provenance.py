"""A campaign's page names its question and each criterion set after the readings it judges were visible, once,
with its date or that none is recorded; the readings read as any others."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import pytest
from pydantic import ValidationError

from threetears.evals.analysis import (
    AnalysisContextBundle,
    CampaignReport,
    build_code_only_report,
    campaign_report_html,
)
from threetears.evals.analysis.bundle.schema import CriterionKind, LateCriterion
from threetears.evals.analysis.report.model import (
    CAMPAIGN_JSON_SHAPE,
    DisclosureBlock,
    TableBlock,
)

from packages.evals.tests.campaign_support import (
    ASSEMBLED_AT,
    QUESTION,
)

from packages.evals.tests.campaign_page_support import (
    aspect_fixture,
    first_screen_markdown,
    page_of,
)


#: When a criterion is flagged as set below: after its aspect's readings were visible, which these tests take as given.
_SET = "2099-01-01T00:00:00+00:00"


def _set_late(
    *late: tuple[CriterionKind, str | None, str | None], on: Sequence[str] = ("billing",)
) -> Callable[[AnalysisContextBundle], AnalysisContextBundle]:
    """The bundles of the aspects ``on``, each with ``late`` (kind, name, time) as its criteria set after its
    readings were visible."""

    def change(bundle: AnalysisContextBundle) -> AnalysisContextBundle:
        if bundle.aspect not in on:
            return bundle
        criteria = [LateCriterion(criterion=kind, name=name, set_at=at) for kind, name, at in late]
        scope = bundle.reading_scope.model_copy(update={"late_criteria": criteria})
        return bundle.model_copy(update={"reading_scope": scope})

    return change


def _timing(page: CampaignReport) -> list[str]:
    """The first screen's lines naming criteria set after their readings, in order."""
    return [
        block.text
        for block in page.blocks
        if isinstance(block, DisclosureBlock) and block.section == "header" and block.source == "timing"
    ]


async def test_a_margin_late_on_some_aspects_is_one_line_naming_them() -> None:
    page = await page_of("billing", "tone", bundle=_set_late(("margin", None, _SET), on=("tone",)))
    assert _timing(page) == ["The margin on tone was set on 1 Jan 2099, after its readings were visible."]
    # Billing's margin came before its first result: nothing is said of it.
    assert "billing was set" not in first_screen_markdown(page)


async def test_an_aspect_set_after_its_readings_reads_as_asked_and_each_criterion_is_said_once() -> None:
    """Owner rulings (2026-10-21, 2026-10-23): disclosed, not enforced. Its row reads as tone's, set in time, does: no
    lead. Each criterion set late is named once, with its time, under the question, and no line names the question or
    when the runs began."""
    page = await page_of("billing", "late")
    first = first_screen_markdown(page)
    said = _timing(page)
    assert said == [
        "The aspect late was declared on 1 Jan 2099, after its readings were visible.",
        "The control was named on 1 Jan 2099, after the readings on late were visible.",
        "The margin on late was set on 1 Jan 2099, after its readings were visible.",
    ], "billing was set in time, so each line names the aspect it holds on"
    assert all(first.index(QUESTION) < first.index(line) for line in said)
    assert "a lead" not in first
    assert "The question" not in " ".join(said) and "began" not in first and "after the runs" not in first
    (late,) = page.readings.on_aspect("late")
    assert late.confirmatory and late.reading == "better"
    (tone,) = (await page_of("billing", "tone")).readings.on_aspect("tone")
    assert (late.reading, late.change, late.gain_interval) == (tone.reading, tone.change, tone.gain_interval)
    assert _timing(await page_of("billing", "tone")) == [], "set in time: nothing said"


async def test_a_criterion_with_no_recorded_time_says_so_and_never_guesses() -> None:
    page = await page_of(
        "billing", "tone", bundle=_set_late(("aspect", None, None), ("control", None, None), ("margin", None, None))
    )
    assert _timing(page) == [
        "When the aspect billing was declared is not recorded.",
        "When the control was named is not recorded.",
        "When the margin on billing was set is not recorded.",
    ]


async def test_a_bar_and_a_floor_set_after_their_readings_are_each_named_by_what_they_hold() -> None:
    (floor,) = (await aspect_fixture("billing")).campaign.floors
    page = await page_of("billing", "tone", bundle=_set_late(("bar", "cost_usd", _SET), ("floor", floor.key, _SET)))
    bar, floored = _timing(page)
    assert bar.startswith("The ") and bar.endswith(
        " bar on billing was set on 1 Jan 2099, after its readings were visible."
    )
    assert floored == (
        "The floor that no pass rate drops by more than 10 points on billing was set on 1 Jan 2099, after its "
        "readings were visible."
    )


async def test_a_criterion_late_on_every_aspect_is_one_line_never_one_per_aspect() -> None:
    page = await page_of(
        "billing", "tone", bundle=_set_late(("aspect", None, _SET), ("margin", None, _SET), on=("billing", "tone"))
    )
    assert _timing(page) == [
        "Every aspect was declared on 1 Jan 2099, after its readings were visible.",
        "The margins on every aspect were set on 1 Jan 2099, after their readings were visible.",
    ]
    control = await page_of("billing", "tone", bundle=_set_late(("control", None, _SET), on=("billing", "tone")))
    assert _timing(control) == ["The control was named on 1 Jan 2099, after every aspect's readings were visible."]


async def test_the_one_aspect_page_says_its_late_criteria_and_the_campaign_s_fold_does_not_repeat_them() -> None:
    page = await page_of("billing", "late")
    own = build_code_only_report((await aspect_fixture("late")).bundle, assembled_at=ASSEMBLED_AT)
    own_timing = [
        block.text
        for block in own.blocks
        if isinstance(block, DisclosureBlock) and block.section == "header" and block.source == "timing"
    ]
    assert own_timing == _timing(page)
    fold = next(fold for fold in page.aspects if fold.aspect == "late")
    assert not [block for block in fold.head if isinstance(block, DisclosureBlock) and block.source == "timing"]


def test_a_late_criterion_is_dated_in_its_own_offset_as_the_rest_of_the_page_dates_it() -> None:
    """The Methods row dates the design's freeze in the stamp's own offset; a late criterion's line dates its instant
    the same way, so one page never prints one instant as two days."""
    assert LateCriterion(criterion="margin", set_at="2026-10-16T23:30:00-05:00").page_date() == "16 Oct 2026"
    assert LateCriterion(criterion="margin", set_at="2026-10-14T23:30:00-05:00").page_date() == "14 Oct 2026"
    assert LateCriterion(criterion="margin", set_at=None).page_date() is None


@pytest.mark.parametrize("stamp", ["sometime in October", "2026-10-14T09:30:00"])
def test_a_late_criterion_whose_time_cannot_be_ordered_is_refused_not_printed_raw(stamp: str) -> None:
    with pytest.raises(ValidationError, match=rf"a late criterion's set_at is stamped {re.escape(repr(stamp))}"):
        LateCriterion(criterion="margin", set_at=stamp)
    assert LateCriterion(criterion="margin", set_at="2026-10-20T15:00:00+00:00").page_date() == "20 Oct 2026"


@pytest.mark.parametrize("late", [False, True])
async def test_the_methods_name_the_design_without_its_timing(late: bool) -> None:
    bundle = (await aspect_fixture("billing")).bundle
    if late:
        bundle = _set_late(("aspect", None, _SET))(bundle)
    report = build_code_only_report(bundle, assembled_at=ASSEMBLED_AT)
    (methods,) = [block for block in report.blocks if isinstance(block, TableBlock) and block.name == "methods"]
    (design,) = [str(row["value"].value) for row in methods.rows if row["item"].value == "Design"]
    assert design.startswith("question declared, frozen ")


async def test_the_page_json_carries_each_aspect_s_late_criteria_beside_its_readings() -> None:
    page = await page_of("billing", "tone", bundle=_set_late(("margin", None, _SET)))
    aspects = page.json_document()["aspects"]
    assert isinstance(aspects, list)
    assert CAMPAIGN_JSON_SHAPE == 4
    assert [aspect["late_criteria"] for aspect in aspects] == [
        [{"criterion": "margin", "name": None, "set_at": _SET}],
        [],
    ]


async def test_the_lines_naming_late_criteria_show_no_field_name() -> None:
    page = await page_of("billing", "late")
    said = _timing(page)
    assert said and not any("_" in line for line in said)
    assert "late_criteria" not in campaign_report_html(page)
