"""The stored planned analyses a host holds, read and drawn by this tree's page against the inputs they were written on.

One is held (``fixtures/report_plans/stored/01a177f6.plan.json``): a host's planned analysis of Kairo round 2, written
before the page drew a plan's charts, against the Kairo bundle (the dataset's ``kairo_bundle``), whose report input is
built here by :func:`~threetears.evals.analysis.report_input_build.report_input`. The read path does not validate a
stored plan, so these tests do not either:

- it reads as a report plan, and round-trips;
- every chart it stores draws but one: a forest of contrasts that mixes an ordinal dimension's change (a score change)
  with pass-fail dimensions' (in points), two quantities on the forest's one axis, which the one-unit rule refuses at
  both ends (:func:`~threetears.evals.analysis.report.plan_page.charting.contrast_unit`);
- so the page refuses it, naming that chart, and the service maps the refusal to "generate the analysis again".
"""

from __future__ import annotations

import functools
import json
from typing import Any

import pytest

from packages.evals.tests.fixtures.report_plans import dataset
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.plan_page.build import plan_page
from threetears.evals.analysis.report.plan_page.drawable import undrawn_charts
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.analysis.report_plan_facts import plan_facts
from threetears.evals.kernel.report_plan import ReportPlan

STORED = dataset.HERE / "stored" / "01a177f6.plan.json"

#: Why the one chart the page cannot draw is refused: its contrasts' changes are a score and points.
_MIXED_UNITS = "its measures are in more than one unit (['points', 'score'])"


def _stored() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(STORED.read_text(encoding="utf-8"))
    return loaded


@functools.cache
def _kairo() -> ReportInput:
    return report_input(dataset.kairo_bundle())


def test_the_stored_plan_reads_as_a_report_plan_and_round_trips() -> None:
    plan = ReportPlan.model_validate(_stored())
    assert plan.model_dump(mode="json") == _stored()


def test_every_chart_it_stores_draws_but_the_forest_mixing_a_score_change_with_point_changes() -> None:
    plan = ReportPlan.model_validate(_stored())
    chart = plan.sections[1].chart
    assert (chart.kind, chart.query.source) == ("forest", "contrasts")
    report_input = _kairo()
    kinds = {
        contrast.measure: contrast.kind
        for contrast in report_input.contrasts
        if contrast.measure in chart.query.measures
    }
    assert set(kinds.values()) == {"judged"}
    assert {report_input.judged[key].scale for key in kinds} == {"pass_fail", "ordinal"}
    undrawn = undrawn_charts(plan, report_input, plan_facts(report_input))
    assert [path for path, _ in undrawn] == ["sections[1].chart"]
    assert _MIXED_UNITS in undrawn[0][1]


def test_the_page_refuses_it_naming_that_chart() -> None:
    plan = ReportPlan.model_validate(_stored())
    with pytest.raises(UnresolvableReference, match=r"'forest' chart over 'contrasts'.*more than one unit"):
        plan_page(plan, _kairo(), made="check")
