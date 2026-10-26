"""The GENERATOR §3 gate function: each arm's status against the gates, with a reason per figure.

Contracts held here (``threetears.evals.analysis.report_gates``, GENERATOR.md §3):

- an arm fails a gate or guardrail on its point estimate (at least the threshold where higher is better, at most
  where lower is), and a passing figure whose interval crosses the threshold is borderline; the interval is the
  figure's ``Stat.ci``, else its ``half_width``, and only a mean carries one, so a p95 is never borderline;
- a gate with no figure for the arm, at any place its scope reads (overall, its task, every task, every cell), is a
  failure there (GENERATOR §1.1), so an unnarrowed check agrees with each narrowed one, and the weakest margin reads
  the same survivors; a figure that is not a number (NaN, an infinity) reads as no figure, so it never passes;
- an arm survives exactly when no check narrowed to one of its places (each task, each cell, and the criteria read
  overall) fails it and no guardrail is breached on it: held both ways, over inputs where an arm fails in a task
  alone, in a cell alone and overall alone;
- a check narrowed to one criterion reads only that one, wherever it reads;
- routing (GENERATOR §3) picks, for each routable task, the cheapest arm passing there among those passing every
  cross-cutting task, each cost the task's own figure per 1,000 replies;
- each reason is one code template: the criterion's reader name (or the statistic it reads, when that is not the
  mean), the figure and the line ("p95 3.6 s > 3 s", "Resolves the issue 90% ±4.4"), with where the figure sits when
  the check is not narrowed to a task or a cell;
- ``surviving_options`` reads its survivors through the gate check, borderline counting as a pass, and gives the
  same survivors, modes and shapes as before the gate check existed: the literal expectations below were captured
  from the code before it was rewritten.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from threetears.evals.analysis.report_decision import allowed_modes, surviving_options
from threetears.evals.analysis.report_gates import GateCheck, gate_check, routing_picks, task_costs
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_margins import weakest_margin
from threetears.evals.analysis.report_plan_checks import classify

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "report_plans"


def _data(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((FIXTURES / name / "input.json").read_text(encoding="utf-8"))
    return loaded


def _criterion(data: dict[str, Any], criterion_id: str) -> dict[str, Any]:
    found: dict[str, Any] = next(c for c in data["criteria"] if c["id"] == criterion_id)
    return found


def _r5_breached() -> dict[str, Any]:
    data = _data("r5")
    data["guardrails"] = [{"arm": "1017", "decision": "breached"}]
    return data


def _r2_no_p95() -> dict[str, Any]:
    data = _data("r2")
    data["measures"]["latency_ms"]["per_arm"]["brisk-s.low.r1"]["p95"] = None
    return data


def _r5_tone() -> dict[str, Any]:
    data = _data("r5")
    data["criteria"] = [
        {"id": "tone", "kind": "guardrail", "measure": "persona.holds_character",
         "direction": "higher_is_better", "threshold": 0.95},
    ]  # fmt: skip
    return data


def _r1_resolves_above_atlas_m() -> dict[str, Any]:
    """Atlas-M's 90% on resolving the issue now sits under the floor: from borderline to failing."""
    data = _data("r1")
    _criterion(data, "trouble.resolves")["threshold"] = 90.1
    return data


def _r1_atlas_m_without_billing_quality() -> dict[str, Any]:
    """The overall quality gate reads every task, and Atlas-M has no quality figure in billing alone: it is not shown to
    pass there, so it does not survive (GENERATOR §1.1)."""
    data = _data("r1")
    del data["tasks"][0]["per_arm"]["atlas-m"]["overall"]
    return data


def _r3_cell_floor_95() -> dict[str, Any]:
    data = _data("r3")
    _criterion(data, "cell_floor")["threshold"] = 95
    return data


def _r4_facts_at_prods_figure() -> dict[str, Any]:
    """The music-facts guardrail raised to prod's own 93.0%: met exactly, which passes."""
    data = _data("r4")
    _criterion(data, "facts")["threshold"] = 93.0
    return data


#: Each case's survivors, allowed modes and shape, captured from ``surviving_options``, ``allowed_modes`` and
#: ``classify`` before survivors were read through the gate check. They must never move, save where GENERATOR §1.1
#: rules: a gate read in every task fails an arm lacking its figure in any one, which moved the one synthetic case
#: built for it ("r1 Atlas-M without billing quality", once ``["atlas-m"]`` and ``decide``).
BEFORE: list[tuple[str, Any, list[str], set[str], str]] = [
    ("r1", lambda: _data("r1"), ["atlas-m"], {"decide"}, "A"),
    ("r2", lambda: _data("r2"), ["brisk-s.low.r1"], {"decide"}, "B"),
    ("r3", lambda: _data("r3"), ["brisk-s.few"], {"decide"}, "C"),
    ("r4", lambda: _data("r4"), ["prod", "f64"], {"tradeoff"}, "D"),
    ("r5", lambda: _data("r5"), ["September", "qwen3.8-flash", "1017"], {"decide"}, "D"),
    ("r5 breached guardrail", _r5_breached, ["September", "qwen3.8-flash"], {"decide"}, "D"),
    ("r2 no p95 for the pick", _r2_no_p95, [], {"no_call"}, "B"),
    ("r5 tone guardrail", _r5_tone, [], {"no_call"}, "D"),
    ("r1 resolves above Atlas-M", _r1_resolves_above_atlas_m, [], {"no_call"}, "A"),
    ("r1 Atlas-M without billing quality", _r1_atlas_m_without_billing_quality, [], {"no_call"}, "A"),
    ("r3 cell floor at 95", _r3_cell_floor_95, [], {"no_call"}, "C"),
    ("r4 facts at prod's figure", _r4_facts_at_prods_figure, ["prod"], {"decide"}, "D"),
]


def _input(data: dict[str, Any]) -> ReportInput:
    return ReportInput.model_validate(data)


def _overall_only(report_input: ReportInput) -> ReportInput:
    """The input with only the criteria read over the whole comparison, so an unnarrowed check reads those alone."""
    kept = [
        criterion
        for criterion in report_input.criteria
        if criterion.scope == "overall" and criterion.task is None  # prose-canary: allow — the declared scope enum
    ]
    return report_input.model_copy(update={"criteria": kept})


def _failing_places(report_input: ReportInput, arm: str) -> list[tuple[str, str]]:
    """Every place a check narrowed to it fails the arm: each task, each cell, and the criteria read overall."""
    failing = [
        ("task", task.id) for task in report_input.tasks if gate_check(report_input, arm, task=task.id).status == "fail"
    ]
    failing += [
        ("cell", cell) for cell in report_input.cells if gate_check(report_input, arm, cell=cell).status == "fail"
    ]
    if gate_check(_overall_only(report_input), arm).status == "fail":
        failing.append(("overall", ""))
    return failing


class TestSurvivorsAreUnchanged:
    @pytest.mark.parametrize(("case", "make", "survivors", "modes", "shape"), BEFORE, ids=[row[0] for row in BEFORE])
    def test_survivors_modes_and_shape_are_as_before(
        self, case: str, make: Any, survivors: list[str], modes: set[str], shape: str
    ):
        report_input = _input(make())
        assert (surviving_options(report_input), allowed_modes(report_input), classify(report_input)[1]) == (
            survivors,
            modes,
            shape,
        )

    @pytest.mark.parametrize(("case", "make", "survivors", "modes", "shape"), BEFORE, ids=[row[0] for row in BEFORE])
    def test_an_arm_survives_exactly_when_no_place_fails_it_and_no_guardrail_is_breached(
        self, case: str, make: Any, survivors: list[str], modes: set[str], shape: str
    ):
        """Both ways, from checks narrowed to each place the gates read: a survivor fails nowhere, and an arm that
        fails nowhere (and is not breached) survives."""
        report_input = _input(make())
        breached = {str(read.get("arm")) for read in report_input.guardrails if read.get("decision") == "breached"}
        failing = {arm.id: _failing_places(report_input, arm.id) for arm in report_input.arms}
        kept = set(surviving_options(report_input))
        for arm in report_input.arms:
            fails_nowhere = not failing[arm.id] and arm.id not in breached
            assert (arm.id in kept) == fails_nowhere, (arm.id, failing[arm.id])

    def test_the_narrowed_survival_check_reaches_a_failure_in_a_task_a_cell_and_overall_alone(self):
        """The cases above hold an arm out by one place each, so the both-ways check is not passing on survivors
        alone: a task (r1, Atlas-M without billing quality), a cell (r3, a cell floor at 95), and the criteria read
        overall (r2, no p95 for the pick)."""
        assert {kind for kind, _ in _failing_places(_input(_r1_atlas_m_without_billing_quality()), "atlas-m")} == {
            "task"
        }
        assert "cell" in {kind for kind, _ in _failing_places(_input(_r3_cell_floor_95()), "brisk-s.few")}
        assert {kind for kind, _ in _failing_places(_input(_r2_no_p95()), "brisk-s.low.r1")} == {"overall"}

    @pytest.mark.parametrize("name", ["r1", "r2", "r3", "r4", "r5"])
    def test_no_task_or_cell_fails_a_reference_survivor(self, name: str):
        report_input = _input(_data(name))
        for arm in surviving_options(report_input):
            places = [gate_check(report_input, arm, task=task.id) for task in report_input.tasks]
            places += [gate_check(report_input, arm, cell=cell) for cell in report_input.cells]
            assert [check.status for check in places if check.status == "fail"] == []

    def test_a_borderline_arm_survives(self):
        """r1: Atlas-M resolves the issue in 90% of troubleshooting, ±4.4, against a floor of 90."""
        report_input = _input(_data("r1"))
        assert gate_check(report_input, "atlas-m").status == "borderline"
        assert surviving_options(report_input) == ["atlas-m"]


class TestTheCheck:
    def test_a_borderline_figure_gives_its_half_width(self):
        assert gate_check(_input(_data("r1")), "atlas-m", task="trouble") == GateCheck(
            arm="atlas-m", status="borderline", fails=(), borders=("Resolves the issue 90% ±4.4",)
        )

    def test_an_unnarrowed_check_says_where_each_figure_sits(self):
        assert gate_check(_input(_data("r1")), "atlas-m").borders == (
            "Resolves the issue 90% ±4.4 in Troubleshooting (multi-turn)",
        )

    def test_a_p95_over_its_budget_fails_and_names_its_statistic(self):
        check = gate_check(_input(_data("r1")), "atlas-l", task="billing")
        assert (check.status, check.fails, check.borders) == ("fail", ("p95 3.6 s > 3 s",), ())

    def test_every_failing_figure_is_one_reason(self):
        check = gate_check(_input(_data("r1")), "brisk-s", task="trouble")
        assert check.fails == (
            "Overall quality floor 79 < 85",
            "Resolves the issue 74% < 90%",
            "Confirms before account changes 88% < 95%",
            "Escalates when stuck 85% < 90%",
        )
        assert check.borders == ("Gives clear steps 90% ±4.4",)

    def test_a_task_check_reads_only_the_gates_in_that_task(self):
        """The speculation gate is read in its own task, so a billing check never fails Brisk-S on it."""
        check = gate_check(_input(_data("r1")), "brisk-s", task="billing")
        assert check == GateCheck(arm="brisk-s", status="pass", fails=(), borders=())

    def test_a_p95_carries_no_interval_so_is_never_borderline(self):
        """Atlas-M's 2.84 s p95 in research sits 0.16 s under the budget, and the input states no interval on it."""
        check = gate_check(_input(_data("r1")), "atlas-m", task="research")
        assert check.status == "pass"

    def test_an_interval_stated_as_its_ends_is_read_on_them(self):
        data = _data("r5")
        data["criteria"] = [
            {"id": "spend", "kind": "gate", "measure": "cost_usd", "direction": "lower_is_better",
             "threshold": 0.0055, "label": "Cost per reply"},
        ]  # fmt: skip
        check = gate_check(_input(data), "September")
        assert (check.status, check.borders) == ("borderline", ("Cost per reply $0.00541, interval $0.00522–$0.00559",))

    def test_a_cell_check_reads_the_cell_floor_in_that_cell(self):
        data = _r3_cell_floor_95()
        report_input = _input(data)
        assert gate_check(report_input, "brisk-s.few", cell="simple|science").borders == ("Cell floor 95.1% ±0.8",)
        assert gate_check(report_input, "brisk-s.few", cell="complex|news").fails == ("Cell floor 82.9% < 95%",)

    def test_an_unnarrowed_cell_figure_names_its_cell(self):
        fails = gate_check(_input(_r3_cell_floor_95()), "brisk-s.few").fails
        assert "Cell floor 82.9% < 95% in Complex × News" in fails

    def test_a_guardrail_is_a_gate(self):
        """r4: f16's music facts sit under the standing guardrail."""
        assert gate_check(_input(_data("r4")), "f16").fails == ("Music facts correct 89.4% < 90%",)

    def test_a_figure_at_the_line_passes(self):
        assert gate_check(_input(_r4_facts_at_prods_figure()), "prod").status == "pass"

    def test_a_gate_with_no_figure_for_the_arm_fails(self):
        check = gate_check(_input(_r2_no_p95()), "brisk-s.low.r1")
        assert (check.status, check.fails) == ("fail", ("Latency budget (p95): no figure",))

    def test_a_task_lacking_the_figure_fails_there(self):
        check = gate_check(_input(_r1_atlas_m_without_billing_quality()), "atlas-m", task="billing")
        assert check.fails == ("Overall quality floor: no figure",)

    def test_an_unnarrowed_check_fails_a_task_lacking_the_figure_and_names_it(self):
        check = gate_check(_input(_r1_atlas_m_without_billing_quality()), "atlas-m")
        assert (check.status, check.fails) == ("fail", ("Overall quality floor: no figure in Billing & account Q&A",))

    def test_an_unnarrowed_check_fails_a_cell_lacking_the_figure_and_names_it(self):
        data = _data("r3")
        del data["cells"]["complex|news"]["per_arm"]["brisk-s.few"]
        report_input = _input(data)
        assert gate_check(report_input, "brisk-s.few").fails == ("Cell floor: no figure in Complex × News",)
        assert gate_check(report_input, "brisk-s.few", cell="complex|news").fails == ("Cell floor: no figure",)
        assert "brisk-s.few" not in surviving_options(report_input)

    def test_the_weakest_margin_reads_the_same_survivors(self):
        """With Atlas-M out for a missing figure, no arm survives and none is one figure from returning."""
        assert weakest_margin(_input(_r1_atlas_m_without_billing_quality())) is None

    def test_an_input_with_no_gate_passes_every_arm(self):
        assert gate_check(_input(_data("r5")), "1017") == GateCheck(arm="1017", status="pass", fails=(), borders=())

    def test_a_check_narrows_to_one_place_at_most(self):
        with pytest.raises(ValueError, match="one place"):
            gate_check(_input(_data("r3")), "brisk-s.few", task="billing", cell="simple|science")

    @pytest.mark.parametrize(("place", "value"), [("task", "nowhere"), ("cell", "nowhere|nothing")])
    def test_a_place_the_input_does_not_hold_is_refused(self, place: str, value: str):
        with pytest.raises(ValueError, match="nowhere"):
            gate_check(_input(_data("r1")), "atlas-m", **{place: value})


class TestAFigureThatIsNotANumber:
    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_a_gate_figure_that_is_not_finite_fails_as_no_figure(self, value: float):
        """r1 Atlas-M resolves the issue in troubleshooting at 90%: read as NaN or infinite, it does not pass."""
        data = _data("r1")
        trouble = next(task for task in data["tasks"] if task["id"] == "trouble")
        trouble["per_arm"]["atlas-m"]["resolves"]["mean"] = value
        report_input = _input(data)
        check = gate_check(report_input, "atlas-m", task="trouble")
        name = next(criterion["label"] for criterion in data["criteria"] if criterion["id"] == "trouble.resolves")
        assert check.status == "fail" and f"{name}: no figure" in check.fails
        assert "atlas-m" not in surviving_options(report_input)

    def test_an_interval_end_that_is_not_a_number_does_not_pass_clear(self):
        """r4 prod's music facts sit clear of the guardrail; an interval end that is NaN cannot show them clear."""
        data = _data("r4")
        facts = data["judged"]["facts"]["per_arm"]["prod"]
        assert gate_check(_input(data), "prod").status == "pass"
        facts["ci"] = [math.nan, facts["mean"]]
        assert gate_check(_input(data), "prod").status == "borderline"


class TestOneCriterion:
    @pytest.mark.parametrize("name", ["r1", "r2", "r3", "r4"])
    def test_an_arm_fails_the_whole_check_exactly_when_it_fails_one_criterion_checked_alone(self, name: str):
        report_input = _input(_data(name))
        gating = [criterion.id for criterion in report_input.criteria if criterion.kind in ("gate", "guardrail")]
        assert gating
        for arm in report_input.arms:
            alone = [gate_check(report_input, arm.id, criterion=criterion) for criterion in gating]
            whole = gate_check(report_input, arm.id)
            assert sorted(reason for check in alone for reason in check.fails) == sorted(whole.fails)
            assert sorted(reason for check in alone for reason in check.borders) == sorted(whole.borders)

    def test_a_criterion_checked_alone_reads_only_that_criterion(self):
        report_input = _input(_data("r1"))
        check = gate_check(report_input, "brisk-s", task="trouble", criterion="trouble.resolves")
        label = next(criterion.label for criterion in report_input.criteria if criterion.id == "trouble.resolves")
        assert check.status == "fail" and len(check.fails) == 1 and check.fails[0].startswith(f"{label} ")
        assert check.borders == ()

    def test_a_criterion_the_input_does_not_gate_on_is_refused(self):
        with pytest.raises(ValueError, match="nowhere"):
            gate_check(_input(_data("r1")), "atlas-m", criterion="nowhere")


class TestRouting:
    def test_each_routable_task_picks_the_cheapest_arm_passing_there_and_in_every_cross_cutting_task(self):
        report_input = _input(_data("r1"))
        arms = [arm.id for arm in report_input.arms]
        cross_cutting = [task.id for task in report_input.tasks if task.cross_cutting]
        assert cross_cutting, "the input holds a cross-cutting task, so the rule has something to hold"

        def passes(arm: str, task: str) -> bool:
            return gate_check(report_input, arm, task=task).status != "fail"

        picks = routing_picks(report_input, arms)
        for task in report_input.tasks:
            if not task.routable:
                assert task.id not in picks
                continue
            costs = task_costs(report_input, task.id)
            eligible = [arm for arm in arms if passes(arm, task.id) and all(passes(arm, x) for x in cross_cutting)]
            eligible = [arm for arm in eligible if arm in costs]
            if not eligible:
                assert task.id not in picks
                continue
            assert picks[task.id] in eligible
            assert costs[picks[task.id]] == min(costs[arm] for arm in eligible)

    def test_a_cross_cutting_failure_keeps_a_cheaper_task_passer_from_being_picked(self):
        """The guard the rule exists for: in some routable task the cheapest arm passing there fails a cross-cutting
        task, and is not the pick."""
        report_input = _input(_data("r1"))
        arms = [arm.id for arm in report_input.arms]
        cross_cutting = [task.id for task in report_input.tasks if task.cross_cutting]
        picks = routing_picks(report_input, arms)
        kept_out = []
        for task in report_input.tasks:
            if not task.routable:
                continue
            costs = task_costs(report_input, task.id)
            passing = [a for a in arms if a in costs and gate_check(report_input, a, task=task.id).status != "fail"]
            cheapest = min(passing, key=lambda arm: costs[arm])
            if any(gate_check(report_input, cheapest, task=x).status == "fail" for x in cross_cutting):
                kept_out.append(task.id)
                assert picks.get(task.id) != cheapest
        assert kept_out

    def test_a_task_cost_is_the_tasks_own_figure_per_1000_replies(self):
        report_input = _input(_data("r1"))
        task = report_input.tasks[0]
        key = next(key for key, measure in report_input.measures.items() if measure.unit == "usd_per_1k")
        expected = {arm: float(figures[key].mean or 0) for arm, figures in task.per_arm.items()}
        assert task_costs(report_input, task.id) == expected

    def test_a_task_the_input_does_not_hold_is_refused(self):
        with pytest.raises(ValueError, match="nowhere"):
            task_costs(_input(_data("r1")), "nowhere")
