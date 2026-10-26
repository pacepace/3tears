"""The GENERATOR §3 gate function: each arm's status against the gates, with a reason per figure.

Contracts held here (``threetears.evals.analysis.report_gates``, GENERATOR.md §3):

- an arm fails a gate or guardrail on its point estimate (at least the threshold where higher is better, at most
  where lower is), and a passing figure whose interval crosses the threshold is borderline; the interval is the
  figure's ``Stat.ci``, else its ``half_width``, and only a mean carries one, so a p95 is never borderline;
- a gate with no figure for the arm, where the check reads it, is a failure;
- each reason is one code template: the criterion's reader name (or the statistic it reads, when that is not the
  mean), the figure and the line ("p95 3.6 s > 3 s", "Resolves the issue 90% ±4.4"), with where the figure sits when
  the check is not narrowed to a task or a cell;
- ``surviving_options`` reads its survivors through the gate check, borderline counting as a pass, and gives the
  same survivors, modes and shapes as before the gate check existed: the literal expectations below were captured
  from the code before it was rewritten.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from threetears.evals.analysis.report_decision import allowed_modes, surviving_options
from threetears.evals.analysis.report_gates import GateCheck, gate_check
from threetears.evals.analysis.report_input import ReportInput
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
    """The overall quality gate reads every task, and Atlas-M has no quality figure in billing alone."""
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
#: ``classify`` before survivors were read through the gate check. They must never move.
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
    ("r1 Atlas-M without billing quality", _r1_atlas_m_without_billing_quality, ["atlas-m"], {"decide"}, "A"),
    ("r3 cell floor at 95", _r3_cell_floor_95, [], {"no_call"}, "C"),
    ("r4 facts at prod's figure", _r4_facts_at_prods_figure, ["prod"], {"decide"}, "D"),
]


def _input(data: dict[str, Any]) -> ReportInput:
    return ReportInput.model_validate(data)


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
    def test_an_arm_survives_exactly_when_the_gate_check_passes_it_and_no_guardrail_is_breached(
        self, case: str, make: Any, survivors: list[str], modes: set[str], shape: str
    ):
        report_input = _input(make())
        breached = {str(read.get("arm")) for read in report_input.guardrails if read.get("decision") == "breached"}
        passed = [
            arm.id
            for arm in report_input.arms
            if arm.id not in breached and gate_check(report_input, arm.id).status != "fail"
        ]
        assert passed == survivors

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

    def test_an_input_with_no_gate_passes_every_arm(self):
        assert gate_check(_input(_data("r5")), "1017") == GateCheck(arm="1017", status="pass", fails=(), borders=())

    def test_a_check_narrows_to_one_place_at_most(self):
        with pytest.raises(ValueError, match="one place"):
            gate_check(_input(_data("r3")), "brisk-s.few", task="billing", cell="simple|science")

    @pytest.mark.parametrize(("place", "value"), [("task", "nowhere"), ("cell", "nowhere|nothing")])
    def test_a_place_the_input_does_not_hold_is_refused(self, place: str, value: str):
        with pytest.raises(ValueError, match="nowhere"):
            gate_check(_input(_data("r1")), "atlas-m", **{place: value})
