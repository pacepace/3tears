"""Every place a campaign's page names a goal check names it by a name, never its expression: the display name its
template declares (``EvalTemplate.goal_check_names``), else ``Goal check``, numbered where an aspect reads several,
through the one door (:func:`~threetears.evals.analysis.report.campaign.names.measure_name_of`,
``AnalysisContextBundle.check_name``, ``AnalysisContextBundle.named_checks``, ``AspectNames.check_names``). The owner
ruled it on 2026-11-07 (Q11), reversing the rule that printed an unnamed check by its whole expression.

The sweep runs over the shared shapes with every check named, the page's folds included: a change whose floor reads
one check and whose call overrides it, a choice, a change of one aspect, and a measurement held to a bar on a check.
An expression stands beside its name in two places: once in the page's glossary, which defines each check's name by
it, on every page, and in a trial's failed-check line, which a page laid out with ``trials="all"`` states for each
trial: the name, then the expression the trial failed. On ``campaign_report``'s default (``trials="table"``) the
glossary is the one place.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from html import escape
from typing import Literal

import pytest

from threetears.evals.analysis import (
    build_campaign_report,
    build_code_only_report,
    campaign_report_html,
    campaign_report_markdown,
)
from threetears.evals.analysis.campaign_readings import campaign_readings
from threetears.evals.analysis.report.model import CampaignReport, DisclosureBlock, TableBlock
from threetears.evals.kernel.declaration import BarOverride, CampaignCall, OverriddenBar
from threetears.evals.kernel.metrics import goal_check_measure

from packages.evals.tests.campaign_support import ASSEMBLED_AT
from packages.evals.tests.report_shapes_support import (
    CHECKS,
    ArmTrials,
    AspectTrials,
    Shape,
    s1_change,
    s2_choice,
    s4_change,
    shape_of,
)

#: When every call here was made.
_CALLED_AT = "2026-10-18T09:00:00+00:00"

#: The display name each check is declared by.
NAMES = {CHECKS[0]: "Total right", CHECKS[1]: "Cites policy"}


def _measurement() -> Shape:
    """One arm on two aspects, graded on both checks and held to a bar on the first."""
    bar = BarOverride(measure_id=goal_check_measure(CHECKS[0]), threshold=0.9, direction="higher_is_better")
    return shape_of(
        "checked measurement",
        [
            AspectTrials("orders", [ArmTrials("solo", passing=19, second=15)], margin=None, bars=(bar,)),
            AspectTrials("returns", [ArmTrials("solo", passing=12, second=18)], margin=None, bars=(bar,)),
        ],
        control=False,
        question="How often does solo get the total right?",
    )


def _named(shape: Shape, names: Mapping[str, str]) -> Shape:
    """``shape`` with every aspect's template naming its checks ``names``: each bundle carries them, and each aspect's
    own page and the readings are built again over the named bundles."""
    return _named_by(shape, {label: names for label in shape.bundles})


def _named_by(
    shape: Shape, by_aspect: Mapping[str, Mapping[str, str]], *, trials: Literal["table", "all"] = "all"
) -> Shape:
    """``shape`` with each aspect's template naming its checks as ``by_aspect`` says for it, each aspect's page laid
    out at ``trials``: every record by default, as a one-aspect builder lays it out, or ``table``, as
    ``campaign_report`` does by default."""
    bundles = {
        label: bundle.model_copy(update={"goal_check_names": dict(by_aspect.get(label, {}))})
        for label, bundle in shape.bundles.items()
    }
    readings = campaign_readings(shape.campaign, bundles, arms=shape.arms)
    reports = {
        label: build_code_only_report(
            bundle, assembled_at=ASSEMBLED_AT, campaign_name=shape.name, arm_labels=readings.arm_labels(), trials=trials
        )
        for label, bundle in bundles.items()
    }
    return replace(shape, bundles=bundles, reports=reports, readings=readings)


def _page(build: str, names: Mapping[str, str]) -> CampaignReport:
    """The shape's page with its checks named ``names``. S1's carries a call adopting v4 over the floor it breaks on a
    check, so the call's line names that floor; the measurement's a call keeping solo that names none of the bars on
    a check it misses, so the call's lines name each of those."""
    shape = _named({"s1": s1_change, "s2": s2_choice, "s4": s4_change, "measurement": _measurement}[build](), names)
    if build == "measurement":
        assert shape.readings.missed_bars(), "solo misses the bar on a check"
        kept = CampaignCall(arm="solo", variant_key=shape.key("solo"), called_at=_CALLED_AT, called_by="op")
        return build_campaign_report(shape.campaign, shape.readings, shape.reports, call=kept)
    if build != "s1":
        return build_campaign_report(shape.campaign, shape.readings, shape.reports)
    broken = [status.floor for status in shape.readings.broken_floors() if status.reading == CHECKS[1]]
    assert broken, "v4 breaks tone's floor on the second check"
    call = CampaignCall(
        arm="v4",
        variant_key=shape.key("v4"),
        called_at=_CALLED_AT,
        called_by="op",
        reason="Cheaper.",
        floors_overridden=broken,
    )
    return build_campaign_report(shape.campaign, shape.readings, shape.reports, call=call)


#: A trial's failed-check line, which states a check's name and then its expression, in Markdown and in HTML; in
#: HTML the name, a glossary term, is linked to its entry where it is the page's first use of it.
_TRIAL_LINE = {
    "markdown": lambda name, check: re.escape(f"- `{name}`: `{check}`"),
    "html": lambda name, check: (
        rf"<li>(?:<a [^>]*>)?{re.escape(escape(name))}(?:</a>)?: <code>{re.escape(escape(check))}</code></li>"
    ),
}


def _definition(check: str, on: str = "") -> str:
    """The glossary's sentence for a check's name, with the aspect it holds on where it says one."""
    return f"The goal check `{check}`{f' on {on}' if on else ''}, which code runs on every trial."


#: The glossary's entry pairing a name with its expression, in Markdown and in HTML.
_GLOSSARY_ROW = {
    "markdown": lambda name, check: (
        f'- <a id="g-check-{re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")}"></a>**{name}**: {_definition(check)}'
    ),
    "html": lambda name, check: (
        f'<dt id="g-check-{re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")}">{escape(name)}</dt>'
        f"<dd>The goal check <code>{escape(check)}</code>, which code runs on every trial.</dd>"
    ),
}


def _without_entries(said: str, form: str, names: Mapping[str, str]) -> str:
    """``said`` without each check's glossary entry, the one place its expression stands beside its name."""
    for check, name in names.items():
        said = said.replace(_GLOSSARY_ROW[form](name, check), "")
    return said


def _rendered(page: CampaignReport, form: str) -> str:
    return campaign_report_markdown(page) if form == "markdown" else campaign_report_html(page)


@pytest.mark.parametrize("form", ["markdown", "html"])
@pytest.mark.parametrize("build", ["s1", "s2", "s4", "measurement"])
def test_a_named_check_is_printed_by_its_name_wherever_the_page_names_it(build: str, form: str) -> None:
    said = _rendered(_page(build, NAMES), form)
    for check, name in NAMES.items():
        assert said.count(_GLOSSARY_ROW[form](name, check)) == 1, "the glossary pairs the name with its check once"
        said = re.sub(_TRIAL_LINE[form](name, check), "", said)
    said = _without_entries(said, form, NAMES)
    assert any(name in said for name in NAMES.values()), "the page names a check"
    left = [check for check in CHECKS if escape(check) in said or check in said]
    assert not left, f"{left} printed by its expression where a name is declared"


#: What the page calls each check its template names nothing: what it tests, read off its expression.
UNNAMED = {CHECKS[1]: "Reply cites policy check", CHECKS[0]: "Reply total check"}


@pytest.mark.parametrize("form", ["markdown", "html"])
@pytest.mark.parametrize("build", ["s1", "s2", "s4", "measurement"])
def test_with_no_name_each_check_is_named_by_what_it_tests_its_expression_in_the_glossary(
    build: str, form: str
) -> None:
    page = _page(build, {})
    said = _rendered(page, form)
    assert not any(name in said for name in NAMES.values())
    for check, name in UNNAMED.items():
        assert said.count(_GLOSSARY_ROW[form](name, check)) == 1, "the glossary pairs the name with its check once"
        said = re.sub(_TRIAL_LINE[form](name, check), "", said)
    said = _without_entries(said, form, UNNAMED)
    assert any(name in said for name in UNNAMED.values()), "the page names a check"
    left = [check for check in CHECKS if escape(check) in said or check in said]
    assert not left, f"{left} printed by its expression, where the page names it by what it tests"
    assert not any(isinstance(block, TableBlock) and block.name == "check_names" for block in page.blocks)


def _default_page(build: str, by_aspect: Mapping[str, Mapping[str, str]] | None = None) -> CampaignReport:
    """The shape's page as ``campaign_report`` lays it out by default (each fold's failed trials as a table, no
    record), its checks named ``NAMES`` on every aspect unless ``by_aspect`` says otherwise."""
    shape = {"s1": s1_change, "s2": s2_choice, "s4": s4_change, "measurement": _measurement}[build]()
    named = _named_by(shape, by_aspect or {label: NAMES for label in shape.bundles}, trials="table")
    return build_campaign_report(named.campaign, named.readings, named.reports)


@pytest.mark.parametrize("form", ["markdown", "html"])
@pytest.mark.parametrize("build", ["s1", "s2", "s4"])
def test_on_the_default_page_the_glossary_is_the_one_place_each_named_check_s_expression_stands(
    build: str, form: str
) -> None:
    page = _default_page(build)
    # The legend the glossary replaced is gone: a check's name is defined once, at the foot.
    assert not any(isinstance(block, TableBlock) and block.name == "check_names" for block in page.blocks)
    named = {name for name in NAMES.values()}
    assert [(term, said) for term, said in page.glossary if term in named] == [
        (name, _definition(check)) for check, name in sorted(NAMES.items())
    ]
    said = _rendered(page, form)
    for check, name in NAMES.items():
        assert said.count(_GLOSSARY_ROW[form](name, check)) == 1
    # Outside its glossary entry, a named check's expression is printed nowhere on the default page.
    rest = _without_entries(said, form, NAMES)
    assert not [check for check in CHECKS if check in rest or escape(check) in rest]


def test_a_check_two_aspects_name_apart_has_a_glossary_entry_per_name_with_the_aspect_declaring_it() -> None:
    page = _default_page("measurement", {"orders": {CHECKS[0]: "Total right"}, "returns": {CHECKS[0]: "Sum right"}})
    glossary = dict(page.glossary)
    assert glossary["Total right"] == _definition(CHECKS[0], "orders")
    assert glossary["Sum right"] == _definition(CHECKS[0], "returns")
    # The other check is named alike on both aspects, so its entry names no aspect.
    assert glossary[UNNAMED[CHECKS[1]]] == _definition(CHECKS[1])


def test_the_call_names_the_check_floor_it_overrides_by_the_check_s_name() -> None:
    first = campaign_report_markdown(_page("s1", NAMES)).split("\n## ", 1)[0]
    assert "no Cites policy rate drops by more than 15 points on tone" in first


def test_naming_a_check_moves_no_reading() -> None:
    shape = s1_change()
    assert _named(shape, NAMES).readings == shape.readings


def test_the_call_names_a_check_bar_by_the_check_s_name_whether_it_overrides_it_or_not() -> None:
    shape = _named(_measurement(), NAMES)
    (missed, *_) = shape.readings.missed_bars()
    call = CampaignCall(
        arm="solo",
        variant_key=shape.key("solo"),
        called_at=_CALLED_AT,
        called_by="op",
        bars_overridden=[OverriddenBar(aspect=missed.aspect, measure=missed.measure)],
    )
    page = build_campaign_report(shape.campaign, shape.readings, shape.reports, call=call)
    calls = [block.text for block in page.blocks if isinstance(block, DisclosureBlock) and block.source == "call"]
    assert (
        f"The call overrides a missed bar on {missed.aspect}: solo misses the bar on {NAMES[CHECKS[0]]}: its whole "
        "interval is on the bad side of the bar."
    ) in calls
    assert not any(CHECKS[0] in line for line in calls)


def _measure_column(page: CampaignReport) -> str:
    (matrix,) = [block for block in page.blocks if isinstance(block, TableBlock) and block.name == "scorecard"]
    (column,) = [column for column in matrix.columns if column.key == f"measure:{goal_check_measure(CHECKS[0])}"]
    return column.header


def _fold_names(page: CampaignReport) -> dict[str, str]:
    (fold,) = [block for block in page.blocks if isinstance(block, TableBlock) and block.name == "measures"]
    measure = goal_check_measure(CHECKS[0])
    return {
        row["aspect"].text: row["measure"].text
        for row, reading in zip(fold.rows, page.readings.measures, strict=True)
        if reading.measure == measure
    }


@pytest.mark.parametrize(
    ("returns", "named"),
    [({CHECKS[0]: "Sum right"}, "Sum right"), ({}, UNNAMED[CHECKS[0]])],
    ids=["named otherwise", "unnamed"],
)
def test_a_column_spanning_aspects_that_name_its_check_apart_is_headed_by_each_name(
    returns: dict[str, str], named: str
) -> None:
    shape = _named_by(_measurement(), {"orders": {CHECKS[0]: "Total right"}, "returns": returns})
    page = build_campaign_report(shape.campaign, shape.readings, shape.reports)
    # A rate's column names its unit after the name (``(%)``), as every rate header does; each name, in the
    # campaign's order, and never the expression, which the glossary defines each name by.
    assert _measure_column(page) == f"Total right / {named} (%)"
    # Each folded row still names the check as its own aspect does.
    assert _fold_names(page) == {"orders": "Total right", "returns": named}


def test_a_column_whose_aspects_name_its_check_alike_is_headed_by_the_name() -> None:
    shape = _named_by(_measurement(), {"orders": {CHECKS[0]: "Total right"}, "returns": {CHECKS[0]: "Total right"}})
    assert _measure_column(build_campaign_report(shape.campaign, shape.readings, shape.reports)) == "Total right (%)"


def test_a_call_over_a_check_bar_no_longer_missed_names_the_check_by_its_name() -> None:
    shape = _named(_measurement(), NAMES)
    measure = goal_check_measure(CHECKS[0])
    missed = {bar.aspect for bar in shape.readings.missed_bars()}
    (held,) = [aspect.label for aspect in shape.campaign.aspects if aspect.label not in missed]
    call = CampaignCall(
        arm="solo",
        variant_key=shape.key("solo"),
        called_at=_CALLED_AT,
        called_by="op",
        bars_overridden=[OverriddenBar(aspect=held, measure=measure)],
    )
    page = build_campaign_report(shape.campaign, shape.readings, shape.reports, call=call)
    calls = [block.text for block in page.blocks if isinstance(block, DisclosureBlock) and block.source == "call"]
    assert f"The call overrides the bar on {NAMES[CHECKS[0]]} on {held} (no longer missed)." in calls
