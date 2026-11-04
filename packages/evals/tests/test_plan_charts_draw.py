"""Every chart a reference input's anchor plan names draws on its own input.

Contracts held here (``threetears.evals.analysis.report.plan_page.drawable``): each reference input's anchor plan
draws every chart on its input. The planner that was refused at generation for a chart the page could not draw is
retired, and its refuse-or-render tests with it; the anchors' charts stay pinned until the plan page leaves.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from packages.evals.tests.fixtures.report_plans.misfits import NAMES, reference_data
from threetears.evals.analysis.report.plan_page.drawable import undrawn_charts
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_plan_facts import plan_facts
from threetears.evals.kernel.report_plan import ReportPlan

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "report_plans"


def _input(name: str) -> ReportInput:
    return ReportInput.model_validate(reference_data(name))


@pytest.mark.parametrize("name", NAMES)
def test_each_anchor_draws_every_chart_on_its_input(name: str) -> None:
    report = _input(name)
    anchor = ReportPlan.model_validate_json((FIXTURES / name / "anchor.plan.json").read_text(encoding="utf-8"))
    assert undrawn_charts(anchor, report, plan_facts(report)) == []
