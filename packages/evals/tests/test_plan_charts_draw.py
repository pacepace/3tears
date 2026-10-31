"""Every chart a generated plan names draws on its own input, or the plan is refused when it is generated.

Contracts held here (``threetears.evals.analysis.report.plan_page.drawable``, ``report_plan_facts.complete_plan``):

- refuse or render, never neither: on degenerate inputs (r2 without its gates, with its gates read in each task or
  with one gate, r3 without its configs, r1 without its gates, r1 reading only bars) and on r1 to r5, a plan whose
  first chart copies the spine code gives, or is a bar chart fitted to the data, is either refused at generation at
  that chart, or accepted and its whole page builds on the input;
- a chart the plan checks accept and the page cannot draw (a gate scatter whose axes no gate reads) is refused at
  generation at its own field path, with the page's own reason, for the planner to repair;
- a generation whose first reply names such a chart is repaired, and the plan it returns renders; one refused twice
  returns nothing to store (one repair, no loop);
- each reference input's anchor plan draws every chart on its input.

The model is the one non-production part: replies written by code from the input's own facts.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from packages.evals.tests.fixtures.report_plans.misfits import MISFITS, NAMES, reference_data
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.toyhost_memo import MODEL, FixturedClient, FixturedCompletion
from packages.evals.tests.toyhost_plan import prose_reply
from threetears.evals.analysis.errors import PlanRefusal
from threetears.evals.analysis.plan_assembly import check_structure, planned_report, prose_slots, read_prose
from threetears.evals.analysis.plan_prompt import EVAL_REPORT_PLAN_DEFAULT
from threetears.evals.analysis.planner import generate_report_plan
from threetears.evals.analysis.report.plan_page.build import plan_page
from threetears.evals.analysis.report.plan_page.drawable import undrawn_charts
from threetears.evals.analysis.report_decision import dominant_options
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_plan_facts import complete_plan, plan_facts
from threetears.evals.kernel.report_plan import PlanProse, PlanStructure, ReportPlan

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "report_plans"

#: Inputs whose criteria or configs leave a chart code or a planner might name with nothing to draw, each read fresh.
DEGENERATE: dict[str, Callable[[], dict[str, Any]]] = MISFITS
#: Every input the agreement is held on.
INPUTS: dict[str, Callable[[], dict[str, Any]]] = {
    **DEGENERATE,
    **{name: (lambda n=name: reference_data(n)) for name in NAMES},
}


def _input(name: str) -> ReportInput:
    return ReportInput.model_validate(INPUTS[name]())


def _query(source: str, aspect: str = "", measures: list[str] | None = None) -> dict[str, Any]:
    return {"source": source, "aspect": aspect, "measures": measures or [], "arms": [], "strata": [], "limit": 0,
            "transform": "none", "small_multiples_by": "none"}  # fmt: skip


def _held(report: ReportInput, source: str) -> list[str]:
    """The measures every arm holds a mean for on ``source``: over the measures, every arm; over the configs, every
    config; over the tasks, every arm in every task. In the input's order."""
    if source == "tasks":
        keys = dict.fromkeys(key for task in report.tasks for figures in task.per_arm.values() for key in figures)
        return [
            key
            for key in keys
            if all(
                key in task.per_arm.get(arm.id, {}) and task.per_arm[arm.id][key].mean is not None
                for task in report.tasks
                for arm in report.arms
            )
        ]
    arms = [config.arm for config in report.configs] if source == "configs" else [arm.id for arm in report.arms]
    return [
        key
        for key, measure in report.measures.items()
        if all(arm in measure.per_arm and measure.per_arm[arm].mean is not None for arm in arms)
    ]


def _figure_source(report: ReportInput) -> str:
    """Where the input holds every arm's figures: its configs, else its measures, else its tasks."""
    if report.configs:
        return "configs"
    return "measures" if _held(report, "measures") else "tasks"


#: Where code's spine reads when it names no source: the first of these the input holds rows for.
_SOURCES = ("tasks", "floors", "measures", "configs", "strata")


def _spine(report: ReportInput) -> dict[str, Any]:
    """The first chart as code's spine gives it, every field it fixes copied, a source the input holds where it names
    none."""
    spine = plan_facts(report).spine
    held = {"tasks": report.tasks, "floors": report.floors, "measures": report.measures, "configs": report.configs,
            "strata": report.strata}  # fmt: skip
    source = spine.source or next(name for name in _SOURCES if held[name])
    measures = list(spine.measures or ()) or (_held(report, "measures")[:1] if source == "measures" else [])
    return {"kind": spine.kind, "query": _query(source, spine.aspect, measures),
            "reference": [{"kind": kind} for kind in spine.reference], "emphasis": ""}  # fmt: skip


def _bars(report: ReportInput) -> dict[str, Any]:
    """A chart fitted to the data: the first measure every arm holds, as bars (one measure, so one unit)."""
    source = "measures" if _held(report, "measures") else "tasks"
    return {"kind": "bars", "query": _query(source, measures=_held(report, source)[:1]), "reference": [],
            "emphasis": ""}  # fmt: skip


def _gate_scatter(report: ReportInput) -> dict[str, Any]:
    """A gate scatter over the axes code's spine fixes where it fixes them, else over the first two measures every arm
    holds: its kind draws its source and its query selects rows, and the page draws it only where one gate reads each
    axis over the whole comparison."""
    spine = plan_facts(report).spine
    source = _figure_source(report)
    fixed = spine.kind == "gate_scatter" and spine.measures is not None
    measures = list(spine.measures or ()) if fixed else _held(report, source)[:2]
    assert len(measures) == 2, f"a gate scatter needs two measures every arm holds: {measures}"
    return {"kind": "gate_scatter", "query": _query(source, measures=measures), "reference": [], "emphasis": ""}


def _structure(report: ReportInput, *charts: dict[str, Any]) -> dict[str, Any]:
    """A structure drawing ``charts`` as its sections, otherwise one the checks accept: decide where code allows it
    (the one dominant option recommended), else the first mode it allows with no option set apart."""
    facts = plan_facts(report)
    mode = "decide" if "decide" in facts.allowed_modes else facts.allowed_modes[0]
    control = next((arm.id for arm in report.arms if arm.role == "control"), None)
    dominant = dominant_options(report)
    picked = dominant[0] if mode == "decide" and len(dominant) == 1 else None
    rows = [{"option": picked, "role": "recommended"}] if picked is not None else []
    rows += [
        {"option": arm.id, "role": "control" if arm.id == control else "alternative"}
        for arm in report.arms
        if arm.id != picked
    ]
    return {
        "mode": mode,
        "headline_kind": {"decide": "decision", "tradeoff": "trade", "no_call": "no_call"}[mode],
        "confidence": "low",
        "caveat_landings": [
            {
                "caveat": caveat.id,
                "at": {"report": "confidence", "method": "method.0"}.get(caveat.reach, "decision.watch"),
            }
            for caveat in facts.caveats
            if not caveat.checked
        ],
        "options": {"rows": rows[:5]},
        "summary": [],
        "sections": [{"id": f"evidence{index}", "chart": chart} for index, chart in enumerate(charts)],
        "explore": [],
        "method_items": 3,
        "durable_claims": [],
    }


def _plan(report: ReportInput, structure: dict[str, Any]) -> ReportPlan:
    """The structure, checked, completed with prose for every key it creates into the stored plan."""
    facts = plan_facts(report)
    chosen = PlanStructure.model_validate(structure)
    check_structure(chosen, report, facts, facts.allowed_modes)
    prose = read_prose(PlanProse.model_validate(prose_reply(report, structure)), prose_slots(chosen, facts))
    return complete_plan(planned_report(chosen, prose, facts), report, facts)


# --- refuse or render ------------------------------------------------------------------------------------------------


#: The two outcomes a plan may have: refused at generation at its first chart, or accepted with its whole page built.
_REFUSED, _RENDERED = "refused", "rendered"
_FIRSTS = ("spine", "bars")


@functools.cache
def _refuse_or_render(name: str, first: str) -> str:
    """A plan opening on the spine code gives, or on a bar chart fitted to the data: refused at that chart, or
    accepted with every section of its page built; never neither."""
    report = _input(name)
    chart = _spine(report) if first == "spine" else _bars(report)
    try:
        plan = _plan(report, _structure(report, chart))
    except PlanRefusal as refused:
        paths = [path for path, _ in refused.faults]
        assert paths and all(path.startswith("sections[0]") for path in paths), paths
        return _REFUSED
    page = plan_page(plan, report)
    assert [section.id for section in page.sections] == [section.id for section in plan.sections]
    return _RENDERED


@pytest.mark.parametrize("first", _FIRSTS)
@pytest.mark.parametrize("name", list(INPUTS))
def test_a_plan_is_refused_at_its_chart_or_its_whole_page_builds(name: str, first: str) -> None:
    assert _refuse_or_render(name, first) in (_REFUSED, _RENDERED)


def test_the_inputs_reach_both_outcomes_and_each_reference_inputs_spine_renders() -> None:
    """Refusal is not the whole story: a check that refused every plan would pass the test above. Each reference
    input's anchor draws on it, so a plan opening on its spine renders; the degenerate inputs are what refuse."""
    outcomes = {(name, first): _refuse_or_render(name, first) for name in INPUTS for first in _FIRSTS}
    assert set(outcomes.values()) == {_REFUSED, _RENDERED}, outcomes
    assert {name: outcomes[name, "spine"] for name in NAMES} == dict.fromkeys(NAMES, _RENDERED)


@pytest.mark.parametrize("name", list(DEGENERATE))
def test_a_chart_the_checks_accept_and_the_page_cannot_draw_is_refused_at_its_field_with_the_pages_reason(
    name: str,
) -> None:
    report = _input(name)
    structure = _structure(report, _spine(report), _gate_scatter(report))
    with pytest.raises(PlanRefusal) as refused:
        _plan(report, structure)
    at_it = [why for path, why in refused.value.faults if path.startswith("sections[1]")]
    assert len(at_it) == 1, refused.value.faults
    assert at_it[0].startswith("is refused by the page") and "'gate_scatter' chart" in at_it[0]
    assert ("sections[1].chart", at_it[0]) in refused.value.faults


@pytest.mark.parametrize("name", NAMES)
def test_each_anchor_draws_every_chart_on_its_input(name: str) -> None:
    report = _input(name)
    anchor = ReportPlan.model_validate_json((FIXTURES / name / "anchor.plan.json").read_text(encoding="utf-8"))
    assert undrawn_charts(anchor, report, plan_facts(report)) == []


# --- a generation ----------------------------------------------------------------------------------------------------


class _Replies(FixturedClient):
    """Hands back one prepared reply per call, in order."""

    def __init__(self, *bodies: dict[str, Any]) -> None:
        super().__init__(json.dumps(bodies[0]))
        self.bodies = list(bodies)

    async def generate(self, *, system: str, user: str, response_format: Any = None, tools: Any = None) -> Any:
        self.completion = FixturedCompletion(content=json.dumps(self.bodies[len(self.calls)]))
        return await super().generate(system=system, user=user, response_format=response_format, tools=tools)


async def _generate(report: ReportInput, client: _Replies) -> Any:
    return await generate_report_plan(
        report, prompt=EVAL_REPORT_PLAN_DEFAULT, model=MODEL, client=client, profile=toyhost_profile()
    )


async def test_a_reply_naming_a_chart_the_page_cannot_draw_is_repaired_and_the_plan_returned_renders() -> None:
    report = _input("r1 without its gates")
    drawable = _structure(report, _spine(report))
    _plan(report, drawable)  # the repair is a plan the page draws
    undrawable = deepcopy(drawable)
    undrawable["sections"].append({"id": "scatter", "chart": _gate_scatter(report)})
    client = _Replies(undrawable, drawable, prose_reply(report, drawable))

    generated = await _generate(report, client)

    assert len(client.calls) == 3 and generated.repair_attempts == 1
    assert "`sections[1].chart` is refused by the page" in (generated.structure_refusal or "")
    plan_page(generated.plan, report)


async def test_a_chart_the_page_cannot_draw_refused_twice_returns_no_plan_to_store() -> None:
    report = _input("r2 with one gate")
    undrawable = _structure(report, _gate_scatter(report))
    client = _Replies(undrawable, undrawable)

    with pytest.raises(PlanRefusal) as refused:
        await _generate(report, client)

    assert len(client.calls) == 2, "one repair, then the refusal: no loop"
    assert "structure call was refused twice" in str(refused.value) and "refused by the page" in str(refused.value)
