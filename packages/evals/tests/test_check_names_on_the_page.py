"""Every place a campaign's page names a goal check names it by the display name its template declares
(``EvalTemplate.goal_check_names``), through the one door (:func:`~threetears.evals.analysis.report.campaign.names.
measure_name_of`, ``AnalysisContextBundle.check_name``, ``AspectNames.check_names``), and by its whole expression only
where none is declared.

The sweep runs over the shared shapes with every check named, the page's folds included: a change whose floor reads
one check and whose call overrides it, a choice, a change of one aspect, and a measurement held to a bar on a check.
The one place an expression still stands beside its name is a trial's failed-check line, which states both: the
name, then the expression the trial failed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from html import escape

import pytest

from threetears.evals.analysis import (
    build_campaign_report,
    build_code_only_report,
    campaign_report_html,
    campaign_report_markdown,
)
from threetears.evals.analysis.campaign_readings import campaign_readings
from threetears.evals.analysis.report.model import CampaignReport
from threetears.evals.kernel.declaration import BarOverride, CampaignCall
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
    bundles = {
        label: bundle.model_copy(update={"goal_check_names": dict(names)}) for label, bundle in shape.bundles.items()
    }
    readings = campaign_readings(shape.campaign, bundles, arms=shape.arms)
    reports = {
        label: build_code_only_report(
            bundle, assembled_at=ASSEMBLED_AT, campaign_name=shape.name, arm_labels=readings.arm_labels()
        )
        for label, bundle in bundles.items()
    }
    return replace(shape, bundles=bundles, reports=reports, readings=readings)


def _page(build: str, names: Mapping[str, str]) -> CampaignReport:
    """The shape's page with its checks named ``names``; S1's with a call adopting v4 over the floor it breaks on a
    check, so the call's line names that floor."""
    shape = _named({"s1": s1_change, "s2": s2_choice, "s4": s4_change, "measurement": _measurement}[build](), names)
    if build != "s1":
        return build_campaign_report(shape.campaign, shape.readings, shape.reports, None)
    broken = [status.floor for status in shape.readings.broken_floors() if status.reading == CHECKS[1]]
    assert broken, "v4 breaks tone's floor on the second check"
    call = CampaignCall(
        arm="v4",
        variant_key=shape.key("v4"),
        called_at="2026-10-18T09:00:00+00:00",
        called_by="op",
        reason="Cheaper.",
        floors_overridden=broken,
    )
    return build_campaign_report(shape.campaign, shape.readings, shape.reports, None, call=call)


#: A trial's failed-check line, which states a check's name and then its expression, in Markdown and in HTML.
_TRIAL_LINE = {
    "markdown": lambda name, check: f"- `{name}`: `{check}`",
    "html": lambda name, check: f"<li>{escape(name)}: <code>{escape(check)}</code></li>",
}


def _rendered(page: CampaignReport, form: str) -> str:
    return campaign_report_markdown(page) if form == "markdown" else campaign_report_html(page)


@pytest.mark.parametrize("form", ["markdown", "html"])
@pytest.mark.parametrize("build", ["s1", "s2", "s4", "measurement"])
def test_a_named_check_is_printed_by_its_name_wherever_the_page_names_it(build: str, form: str) -> None:
    said = _rendered(_page(build, NAMES), form)
    for check, name in NAMES.items():
        said = said.replace(_TRIAL_LINE[form](name, check), "")
    assert any(name in said for name in NAMES.values()), "the page names a check"
    left = [check for check in CHECKS if escape(check) in said or check in said]
    assert not left, f"{left} printed by its expression where a name is declared"


@pytest.mark.parametrize("form", ["markdown", "html"])
@pytest.mark.parametrize("build", ["s1", "s2", "s4", "measurement"])
def test_with_no_name_each_check_the_page_names_is_its_whole_expression(build: str, form: str) -> None:
    said = _rendered(_page(build, {}), form)
    assert not any(name in said for name in NAMES.values())
    assert any(re.search(re.escape(escape(check)), said) for check in CHECKS)


def test_the_call_names_the_check_floor_it_overrides_by_the_check_s_name() -> None:
    first = campaign_report_markdown(_page("s1", NAMES)).split("\n## ", 1)[0]
    assert "no Cites policy rate drops by more than 15 points on tone" in first


def test_naming_a_check_moves_no_reading() -> None:
    shape = s1_change()
    assert _named(shape, NAMES).readings == shape.readings
