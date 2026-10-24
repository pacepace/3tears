"""A campaign's page quotes its question, says once that criteria were set after the readings they judge were visible,
and dates each in Design & provenance, or says none is recorded; the readings read as any others. Every report-wide
disclosure is one line on the first screen and its rows there, never one per aspect."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import pytest
from pydantic import ValidationError

from threetears.evals.analysis import (
    AnalysisContextBundle,
    CampaignReport,
    build_campaign_report,
    build_code_only_report,
    campaign_report_html,
    campaign_report_markdown,
)
from threetears.evals.analysis.bundle.schema import CriterionKind, LateCriterion
from threetears.evals.analysis.style import lint_prose
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
from packages.evals.tests.report_shapes_support import SHAPES


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
    """The first screen's caveat lines on when criteria were set, in order."""
    return [
        block.text
        for block in page.blocks
        if isinstance(block, DisclosureBlock) and block.section == "warnings" and block.source == "timing"
    ]


def _criteria(page: CampaignReport) -> list[tuple[str, str, str]]:
    """Design & provenance's rows of criteria set late: criterion, when, and the aspects whose results existed."""
    tables = [block for block in page.blocks if isinstance(block, TableBlock) and block.name == "criteria_set"]
    if not tables:
        return []
    (table,) = tables
    assert table.section == "provenance"
    return [(row["criterion"].text, row["set_on"].text, row["aspects"].text) for row in table.rows]


#: The one line a campaign of two aspects says where criteria on one of them were set late.
_ONE_OF_TWO = "Criteria on 1 of 2 aspects were set after the results there existed, so they were not fixed in advance."


async def test_a_margin_late_on_some_aspects_is_one_caveat_line_and_one_row_naming_them() -> None:
    page = await page_of("billing", "tone", bundle=_set_late(("margin", None, _SET), on=("tone",)))
    assert _timing(page) == [_ONE_OF_TWO]
    assert _criteria(page) == [("margin", "1 Jan 2099", "tone")]
    # Billing's margin came before its first result: nothing is said of it.
    assert "billing" not in [aspects for *_, aspects in _criteria(page)]


async def test_an_aspect_set_after_its_readings_reads_as_asked_and_its_timing_is_said_once() -> None:
    """Owner rulings (2026-10-21, 2026-10-23, 2026-10-24): disclosed, not enforced, and said once. Its row reads as
    tone's, set in time, does: no lead. The first screen says one line for the whole report, under the answer, and
    Design & provenance holds each criterion set late, with its time; no line names the question or when the runs
    began."""
    page = await page_of("billing", "late")
    first = first_screen_markdown(page)
    said = _timing(page)
    assert said == [_ONE_OF_TWO]
    assert _criteria(page) == [
        ("aspect", "1 Jan 2099", "late"),
        ("control", "1 Jan 2099", "late"),
        ("margin", "1 Jan 2099", "late"),
    ], "billing was set in time, so each row names the aspect it holds on"
    assert first.index(QUESTION) < first.index(_ONE_OF_TWO)
    assert "a lead" not in first
    assert "The question" not in " ".join(said) and "began" not in first and "after the runs" not in first
    (late,) = page.readings.on_aspect("late")
    assert late.confirmatory and late.reading == "better"
    (tone,) = (await page_of("billing", "tone")).readings.on_aspect("tone")
    assert (late.reading, late.change, late.gain_interval) == (tone.reading, tone.change, tone.gain_interval)
    timely = await page_of("billing", "tone")
    assert _timing(timely) == [] and _criteria(timely) == [], "set in time: nothing said"


async def test_a_criterion_with_no_recorded_time_says_so_and_never_guesses() -> None:
    page = await page_of(
        "billing", "tone", bundle=_set_late(("aspect", None, None), ("control", None, None), ("margin", None, None))
    )
    assert _timing(page) == [
        "Criteria on 1 of 2 aspects were set after the results there existed or have no recorded time, so they are "
        "not shown fixed in advance."
    ]
    assert [violation for line in _timing(page) for violation in lint_prose(line)] == []
    assert _criteria(page) == [
        ("aspect", "not recorded", "billing"),
        ("control", "not recorded", "billing"),
        ("margin", "not recorded", "billing"),
    ]


async def test_a_bar_and_a_floor_set_after_their_readings_are_each_named_by_what_they_hold() -> None:
    (floor,) = (await aspect_fixture("billing")).campaign.floors
    page = await page_of("billing", "tone", bundle=_set_late(("bar", "cost_usd", _SET), ("floor", floor.key, _SET)))
    (bar, at, on), floored = _criteria(page)
    assert bar.startswith("bar on ") and (at, on) == ("1 Jan 2099", "billing")
    assert floored == ("floor that no pass rate drops by more than 10 points", "1 Jan 2099", "billing")
    assert _timing(page) == [_ONE_OF_TWO]


async def test_a_criterion_late_on_every_aspect_is_one_row_never_one_per_aspect() -> None:
    page = await page_of(
        "billing", "tone", bundle=_set_late(("aspect", None, _SET), ("margin", None, _SET), on=("billing", "tone"))
    )
    assert _timing(page) == [
        "Criteria on every aspect were set after the results there existed, so they were not fixed in advance."
    ]
    assert _criteria(page) == [("aspect", "1 Jan 2099", "every aspect"), ("margin", "1 Jan 2099", "every aspect")]


@pytest.mark.parametrize("on", [("billing",), ("billing", "tone")])
async def test_late_criteria_on_k_of_n_aspects_are_one_timing_line_and_the_table_lists_the_k(
    on: tuple[str, ...],
) -> None:
    page = await page_of("billing", "tone", bundle=_set_late(("margin", None, _SET), on=on))
    assert len(_timing(page)) == 1
    listed = {aspects for *_, aspects in _criteria(page)}
    assert listed == ({"billing"} if len(on) == 1 else {"every aspect"})
    markdown = campaign_report_markdown(page)
    assert markdown.count(_timing(page)[0]) == 1, "said once, on the first screen"
    assert markdown.index("## Design & provenance") < markdown.index("**Criteria set late**")
    html = campaign_report_html(page)
    assert html.index('<section id="provenance">') < html.index('data-table="criteria_set"')
    if len(on) == 1:
        assert '<a href="#aspect-billing">billing</a>' in html[html.index('data-table="criteria_set"') :]


async def test_the_one_aspect_page_says_one_timing_line_and_the_campaign_s_fold_none() -> None:
    page = await page_of("billing", "late")
    own = build_code_only_report((await aspect_fixture("late")).bundle, assembled_at=ASSEMBLED_AT)
    own_timing = [
        block.text
        for block in own.blocks
        if isinstance(block, DisclosureBlock) and block.section == "header" and block.source == "timing"
    ]
    assert own_timing == [
        "Criteria on late were set after the results there existed, so they were not fixed in advance."
    ]
    fold = next(fold for fold in page.aspects if fold.aspect == "late")
    assert not [block for block in fold.head if isinstance(block, DisclosureBlock) and block.source == "timing"]
    fold_markdown = campaign_report_markdown(page).split("## late", 1)[1]
    assert "were set after" not in fold_markdown


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
    assert CAMPAIGN_JSON_SHAPE == 5
    assert [aspect["late_criteria"] for aspect in aspects] == [
        [{"criterion": "margin", "name": None, "set_at": _SET}],
        [],
    ]


async def test_the_lines_naming_late_criteria_show_no_field_name() -> None:
    page = await page_of("billing", "late")
    said = _timing(page) + [cell for row in _criteria(page) for cell in row]
    assert said and not any("_" in line for line in said)
    assert "late_criteria" not in campaign_report_html(page)


def _shape_page(name: str, change: Callable[[AnalysisContextBundle], AnalysisContextBundle]) -> CampaignReport:
    """The page of shape ``name`` with each aspect's evidence changed by ``change``, each aspect's own page built over
    the changed evidence as the shape builds it."""
    shape = SHAPES[name]()
    labels = shape.readings.arm_labels()
    reports = {
        label: build_code_only_report(
            change(bundle), assembled_at=ASSEMBLED_AT, campaign_name=shape.name, arm_labels=labels
        )
        for label, bundle in shape.bundles.items()
    }
    return build_campaign_report(shape.campaign, shape.readings, reports, None)


@pytest.mark.parametrize("name", ["s1", "s2", "s4"])
def test_on_any_shape_late_criteria_on_k_aspects_are_one_line_and_k_listed(name: str) -> None:
    first = SHAPES[name]().campaign.aspects[0].label
    page = _shape_page(name, _set_late(("margin", None, _SET), ("aspect", None, _SET), on=(first,)))
    (line,) = _timing(page)
    total = len(page.aspects)
    assert line.startswith(f"Criteria on {first if total == 1 else f'1 of {total} aspects'} were set after")
    assert {aspects for *_, aspects in _criteria(page)} == {first}
    assert campaign_report_markdown(page).count(line) == 1


@pytest.mark.parametrize("name", ["s1", "s3"])
def test_latency_and_settings_are_one_caveat_line_each_and_no_fold_repeats_them(name: str) -> None:
    def contended(bundle: AnalysisContextBundle) -> AnalysisContextBundle:
        return bundle.model_copy(
            update={
                "latency_contended": "Latency is not compared here: every result's latency was read concurrently.",
                "request_settings_disclosure": "Request settings not confirmed as asked: the judge sent other ones.",
            }
        )

    page = _shape_page(name, contended)
    caveats = [
        block.text
        for block in page.blocks
        if isinstance(block, DisclosureBlock) and block.section == "warnings" and block.source == "measurement"
    ]
    assert caveats == [
        "Latency on every aspect was read while other runs executed beside it, so it is not compared; launching with "
        "`measure_latency=True` reads it clean.",
        "Request settings on every aspect are not confirmed as sent, so the judge or simulated user may have run on "
        "other settings than asked.",
    ]
    assert [violation for line in caveats for violation in lint_prose(line)] == []
    setup = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "test_setup")
    assert [(row["setup"].text, row["aspects"].text) for row in setup.rows] == [
        ("latency not compared", "every aspect"),
        ("request settings not confirmed", "every aspect"),
    ]
    # Each aspect's own page still says them (its JSON is the aspect's whole page); its fold on this page does not.
    for fold in page.aspects:
        assert any(isinstance(b, DisclosureBlock) and b.text.startswith("Latency is not") for b in fold.report.blocks)
    for rendered in (campaign_report_markdown(page), campaign_report_html(page)):
        assert "every result&#x27;s latency was read concurrently" not in rendered
        assert "every result's latency was read concurrently" not in rendered
        assert "the judge sent other ones" not in rendered


def test_a_page_with_nothing_report_wide_has_no_caveat_and_no_design_and_provenance() -> None:
    page = SHAPES["s4"]().page()
    assert _timing(page) == [] and not [block for block in page.blocks if block.section == "provenance"]
    assert "## Design & provenance" not in campaign_report_markdown(page)
