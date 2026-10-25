"""The report plan's checks: code classifies the eval, and refuses a plan that disagrees with itself or its input.

Contracts held here (``threetears.evals.analysis.report_plan_checks``, GENERATOR.md §1–§3, §6.2):

- the kit's Kairo round 2 plan passes against r5's input: the base case every refusal below breaks one thing of;
- :func:`classify` looks r1–r5 up as A, B, C, D, D, and an ordered-versions input as E; the criteria signal wins a
  control under hard gates;
- each check refuses at the field path it names: code's facts, mode (one code allows), slots, roles, ids, the
  recommendation (the memo's refusals ported, and the one option that dominates), the warning answers
  (``floor_unclear`` among them), the confidence's form and clauses, the flag, caveat coverage, the charts, the figure
  keys and D3, over every prose field the plan holds.

Modes other than r5's own need inputs that allow them (:mod:`threetears.evals.analysis.report_decision`): a tradeoff
variant where 1017 is the cheapest arm and nothing shows it no worse, a no-call variant where a standing guardrail
removes every arm, and a flash variant where qwen3.8-flash is the cheapest arm and so dominates.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from packages.evals.tests.report_kit_support import KIT
from threetears.evals.analysis.errors import PlanRefusal, SoundnessRefusal
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_plan_checks import (
    classify,
    implied_disclosures,
    raised_warnings,
    validate_plan,
)
from threetears.evals.kernel import report_plan
from threetears.evals.kernel.report_plan import ReportPlan
from threetears.evals.schema.prose import field_is_prose

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "report_plans"
R5_PLAN = KIT / "references" / "r5-kairo-round2.plan.json"
ASPECT = "character adherence (v2)"
FLOOR = "character adherence (v2)/rate_drop/*"


def _input_dict(name: str = "r5") -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((FIXTURES / name / "input.json").read_text(encoding="utf-8"))
    return loaded


def _input(name: str = "r5", data: dict[str, Any] | None = None) -> ReportInput:
    return ReportInput.model_validate(data if data is not None else _input_dict(name))


def _plan() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(R5_PLAN.read_text(encoding="utf-8"))
    return loaded


def _faults(plan: dict[str, Any], data: dict[str, Any] | None = None) -> list[tuple[str, str]]:
    """The faults validate_plan finds; the plan must be refused."""
    with pytest.raises(PlanRefusal) as refused:
        validate_plan(ReportPlan.model_validate(plan), _input(data=data))
    return list(refused.value.faults)


def _paths(plan: dict[str, Any], data: dict[str, Any] | None = None) -> list[str]:
    return [path for path, _ in _faults(plan, data)]


def _why(plan: dict[str, Any], path: str, data: dict[str, Any] | None = None) -> str:
    found = [why for at, why in _faults(plan, data) if at == path]
    assert found, f"no fault at {path}"
    return " ".join(found)


def _passes(plan: dict[str, Any], data: dict[str, Any] | None = None) -> None:
    validate_plan(ReportPlan.model_validate(plan), _input(data=data))


_STEP = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def _at(node: Any, path: str) -> tuple[Any, str | int]:
    """The container holding ``path``'s last step, and that step."""
    steps: list[str | int] = [name if name else int(index) for name, index in _STEP.findall(path)]
    for step in steps[:-1]:
        node = node[step]
    return node, steps[-1]


def _get(plan: dict[str, Any], path: str) -> Any:
    holder, step = _at(plan, path)
    return holder[step]


def _set(plan: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    holder, step = _at(plan, path)
    holder[step] = value
    return plan


def _recommending_flash(plan: dict[str, Any]) -> dict[str, Any]:
    """The base plan switched to recommending qwen3.8-flash, the summary charts following the options rows."""
    plan["options"]["rows"] = [
        {"option": "qwen3.8-flash", "role": "recommended"},
        {"option": "September", "role": "control"},
        {"option": "1017", "role": "alternative"},
    ]
    plan["summary"][0]["query"]["arms"] = ["qwen3.8-flash", "1017"]
    plan["summary"][1]["query"]["arms"] = ["qwen3.8-flash", "September", "1017"]
    return plan


def _cost(data: dict[str, Any], arm: str, usd: float) -> dict[str, Any]:
    """Set one arm's cost per reply on the cost axis (and so its cost per 1,000 replies)."""
    data["measures"]["production_replicating_cost"]["per_arm"][arm]["mean"] = usd
    return data


def _flash_data() -> dict[str, Any]:
    """r5 with qwen3.8-flash the cheapest arm: shown no worse at the declared margin, it dominates."""
    return _cost(_input_dict(), "qwen3.8-flash", 0.0015)


#: What a plan recommending qwen3.8-flash on the flash input lists: the late margin and floor now decide the call
#: (below its flip margin qwen3.8-flash is no longer shown no worse), so neither is immaterial.
_FLASH_CLAUSES = ("late:margin", f"late:floor:{FLOOR}")


def _tradeoff_data() -> dict[str, Any]:
    """r5 with 1017 the cheapest arm and its floor unread, so nothing shows it no worse: September and 1017 survive and
    neither dominates."""
    data = _cost(_input_dict(), "1017", 0.0015)
    data["floors"] = [floor for floor in data["floors"] if floor["arm"] != "1017"]
    return data


def _no_call_data() -> dict[str, Any]:
    """r5 under a standing guardrail every arm fails: no option survives."""
    data = _input_dict()
    data["criteria"] = [
        {"id": "tone", "kind": "guardrail", "measure": "persona.holds_character", "direction": "higher_is_better",
         "threshold": 0.95, "standing": True},
    ]  # fmt: skip
    return data


def _clause(caveat: str) -> dict[str, str]:
    return {"id": caveat, "material_if": "Material only if it changed which option is cheaper."}


def _listing(plan: dict[str, Any], *caveats: str) -> dict[str, Any]:
    """The plan with each caveat added to its confidence's clauses."""
    plan["decision"]["confidence"]["caveats"] += [_clause(caveat) for caveat in caveats]
    return plan


def _floor_status(data: dict[str, Any], arm: str, status: str) -> dict[str, Any]:
    for floor in data["floors"]:
        if floor["arm"] == arm:
            floor["status"] = status
    return data


def _answer(kind: str, aspect: str, name: str) -> dict[str, Any]:
    return {"warning": {"kind": kind, "aspect": aspect, "name": name}, "text": "It stands because the cost is lower."}


# --- the base case -------------------------------------------------------------------------------------------------


class TestTheBaseCase:
    def test_the_kairo_plan_passes_against_its_input(self):
        _passes(_plan())

    def test_a_refusal_is_repairable_and_lists_every_fault_with_its_path(self):
        plan = _set(_set(_plan(), "shape", "A"), "headline_kind", "trade")
        with pytest.raises(SoundnessRefusal) as refused:
            validate_plan(ReportPlan.model_validate(plan), _input())
        assert isinstance(refused.value, PlanRefusal)
        assert {"shape", "headline_kind"} <= {path for path, _ in refused.value.faults}
        assert "`shape`" in str(refused.value) and "`headline_kind`" in str(refused.value)


# --- classification ------------------------------------------------------------------------------------------------


class TestClassify:
    @pytest.mark.parametrize(("name", "shape"), [("r1", "A"), ("r2", "B"), ("r3", "C"), ("r4", "D"), ("r5", "D")])
    def test_each_fixture_looks_up_its_shape(self, name: str, shape: str):
        assert classify(_input(name))[1] == shape

    def test_the_kairo_signals_are_the_plans(self):
        signals, shape = classify(_input())
        plan = ReportPlan.model_validate(_plan())
        assert (signals, shape) == (plan.signals, plan.shape)

    def test_ordered_versions_look_up_shape_e(self):
        data = _input_dict()
        data["meta"]["design_shape"] = "ordered_versions"
        signals, shape = classify(_input(data=data))
        assert (signals.candidate_structure, shape) == ("ordered_versions", "E")

    def test_the_criteria_signal_wins_a_control_under_hard_gates(self):
        data = _input_dict()
        data["criteria"] = [
            {"id": "pass", "kind": "gate", "measure": "passed", "direction": "higher_is_better", "threshold": 0.8}
        ]
        signals, shape = classify(_input(data=data))
        assert (signals.baseline, signals.criteria, shape) == ("control", "hard_gates", "A")

    def test_a_factorial_grid_keeps_its_shape_under_gates(self):
        signals, shape = classify(_input("r2"))
        assert (signals.candidate_structure, signals.criteria, shape) == ("factorial", "hard_gates", "B")

    def test_a_classifier_reads_class_labels_and_a_target_matrix(self):
        signals, _ = classify(_input("r3"))
        assert (signals.score_type, signals.strata) == ("class_label", "target_matrix")

    def test_unordered_candidates_with_no_gate_are_a_selection(self):
        data = _input_dict("r1")
        data["criteria"] = []
        signals, shape = classify(_input(data=data))
        assert (signals.candidate_structure, signals.criteria, shape) == ("unordered", "none_declared", "A")

    def test_the_tasks_and_cross_cutting_aspects_are_read(self):
        signals, _ = classify(_input("r1"))
        assert (signals.strata, signals.cross_cutting, signals.score_type) == ("tasks", ["specul"], "mixed")

    def test_a_guardrail_is_a_margin_rule(self):
        assert classify(_input("r4"))[0].criteria == "margin"

    def test_cost_spanning_more_than_ten_times_is_a_log_axis(self):
        assert classify(_input("r2"))[0].log_cost_axis
        assert not classify(_input("r5"))[0].log_cost_axis


class TestCodesFacts:
    def test_a_shape_off_the_lookup_is_refused(self):
        assert "look up shape 'D'" in _why(_set(_plan(), "shape", "A"), "shape")

    def test_a_signal_off_the_input_is_refused(self):
        assert "'mixed'" in _why(_set(_plan(), "signals.score_type", "graded"), "signals.score_type")


# --- mode, slots and roles -----------------------------------------------------------------------------------------


def _no_call(plan: dict[str, Any]) -> dict[str, Any]:
    plan.update(mode="no_call", headline_kind="no_call")
    plan["options"]["rows"][0]["role"] = "control"
    return plan


#: The flag a tradeoff plan with no declared ranking rule carries.
_NO_RULE_FLAG: dict[str, Any] = {
    "title": "No rule was set in advance to rank cost against character.",
    "text": "The product owner makes the call.",
    "discloses": ["no_rule"],
    "weakens": [],
    "survives": [],
}


def _tradeoff(plan: dict[str, Any]) -> dict[str, Any]:
    plan.update(mode="tradeoff", headline_kind="trade")
    for slot, key in zip(plan["decision"]["slots"], ("the_trade", "where_it_flips", "our_lean", "what_settles_it")):
        slot["key"] = key
    plan["options"]["rows"][0]["role"] = "control"
    plan["flag"] = [dict(_NO_RULE_FLAG)]
    return plan


class TestMode:
    def test_a_headline_kind_off_the_mode_is_refused(self):
        assert "'decision'" in _why(_set(_plan(), "headline_kind", "trade"), "headline_kind")

    def test_slots_out_of_order_are_refused(self):
        plan = _plan()
        slots = plan["decision"]["slots"]
        slots[0], slots[1] = slots[1], slots[0]
        assert "in that order" in _why(plan, "decision.slots")

    def test_tradeoff_slots_in_decide_mode_are_refused(self):
        plan = _tradeoff(_plan())
        plan.update(mode="decide", headline_kind="decision", flag=[])
        plan["options"]["rows"][0]["role"] = "recommended"
        assert "decision.slots" in _paths(plan)

    def test_a_tradeoff_plan_with_its_slots_and_no_recommendation_passes(self):
        _passes(_tradeoff(_plan()), _tradeoff_data())

    def test_a_tradeoff_plan_may_lean_toward_one_option(self):
        _passes(_set(_tradeoff(_plan()), "options.rows[1].role", "lean"), _tradeoff_data())

    def test_a_no_call_takes_decide_modes_slots(self):
        _passes(_no_call(_plan()), _no_call_data())

    def test_a_no_call_with_tradeoff_slots_is_refused(self):
        plan = _tradeoff(_plan())
        plan.update(mode="no_call", headline_kind="no_call", flag=[])
        assert "decision.slots" in _paths(plan, _no_call_data())


class TestOptionRoles:
    def test_decide_needs_its_recommended_option_first(self):
        plan = _plan()
        rows = plan["options"]["rows"]
        rows[0]["role"], rows[1]["role"] = "alternative", "recommended"
        assert "its first row" in _why(plan, "options.rows")

    def test_decide_recommends_exactly_one_option(self):
        plan = _set(_plan(), "options.rows[1].role", "recommended")
        assert "exactly one" in _why(plan, "options.rows")

    def test_decide_takes_no_lean(self):
        assert "options.rows[2].role" in _paths(_set(_plan(), "options.rows[2].role", "lean"))

    def test_a_tradeoff_plan_recommends_nothing(self):
        plan = _set(_tradeoff(_plan()), "options.rows[1].role", "recommended")
        assert "recommends no option" in _why(plan, "options.rows[1].role", _tradeoff_data())

    def test_a_tradeoff_plan_leans_toward_one_option_at_most(self):
        plan = _tradeoff(_plan())
        plan["options"]["rows"][1]["role"] = plan["options"]["rows"][2]["role"] = "lean"
        assert "one option at most" in _why(plan, "options.rows", _tradeoff_data())

    def test_a_no_call_leans_toward_nothing(self):
        plan = _set(_no_call(_plan()), "options.rows[1].role", "lean")
        assert "options.rows[1].role" in _paths(plan, _no_call_data())


class TestIds:
    def test_a_repeated_section_id_is_refused(self):
        assert "'strata'" in _why(_set(_plan(), "sections[0].id", "strata"), "sections")

    def test_a_repeated_option_is_refused(self):
        assert "options.rows" in _paths(_set(_plan(), "options.rows[2].option", "qwen3.8-flash"))

    def test_a_repeated_control_id_is_refused(self):
        assert "explore[0].controls" in _paths(_set(_plan(), "explore[0].controls[1].id", "margin"))


# --- the recommendation --------------------------------------------------------------------------------------------


class TestTheRecommendation:
    def test_a_measurement_recommends_nothing(self):
        data = _input_dict()
        data["arms"] = data["arms"][:1]
        assert "measures one arm" in _why(_plan(), "options.rows[0].role", data)

    def test_a_recommended_option_is_an_arm(self):
        plan = _set(_plan(), "options.rows[0].option", "Sept")
        assert "which is no arm" in _why(plan, "options.rows[0].option")

    def test_a_choice_nothing_separated_is_picked_only_at_low_confidence(self):
        data = _input_dict()
        data["arms"][0]["role"] = "challenger"
        assert "'low'" in _why(_plan(), "decision.confidence.form", data)
        assert "decision.confidence.form" not in _paths(_set(_plan(), "decision.confidence.form", "low"), data)

    def test_a_choice_shown_ahead_may_be_picked_at_any_confidence(self):
        data = _input_dict()
        data["arms"][0]["role"] = "challenger"
        data["aspects"][0]["pairs"] = [{"arms": ["September", "1017"], "change": 3.0, "ahead": "September"}]
        assert "decision.confidence.form" not in _paths(_plan(), data)

    def test_switching_to_a_candidate_every_floor_holds_for_passes(self):
        _passes(_listing(_recommending_flash(_plan()), *_FLASH_CLAUSES), _flash_data())

    def test_an_exploratory_change_switches_to_nothing(self):
        data = _input_dict()
        data["aspects"][0]["asked"] = False
        assert "exploratory" in _why(_recommending_flash(_plan()), "options.rows[0].option", data)

    def test_a_breached_guardrail_refuses_the_switch(self):
        data = _input_dict()
        data["guardrails"] = [
            {"aspect": ASPECT, "arm": "qwen3.8-flash", "name": "tone", "decision": "breached"},
        ]
        assert "breached the guardrail" in _why(_recommending_flash(_plan()), "options.rows[0].option", data)

    def test_a_floor_not_held_refuses_an_unseparated_switch(self):
        data = _floor_status(_input_dict(), "qwen3.8-flash", "unclear")
        plan = _recommending_flash(_plan())
        plan["answers"] = [_answer("floor_unclear", ASPECT, FLOOR)]
        assert f"floor {FLOOR} is unclear" in _why(plan, "options.rows[0].option", data)

    def test_an_asked_aspect_with_no_floor_refuses_an_unseparated_switch(self):
        data = _input_dict()
        data["floors"] = []
        assert "declares no floor" in _why(_recommending_flash(_plan()), "options.rows[0].option", data)

    def test_a_confirmatory_better_reading_licenses_the_switch(self):
        """Shown better, qwen3.8-flash dominates at any margin or floor, so the late margin and floor stay checked."""
        data = _floor_status(_flash_data(), "qwen3.8-flash", "broken")
        data["aspects"][0]["readings"][0]["reading"] = "better"
        plan = _recommending_flash(_plan())
        plan["answers"] = [_answer("floor_broken", ASPECT, FLOOR)]
        _passes(plan, data)

    def test_an_improved_contrast_on_an_asked_aspect_licenses_the_switch(self):
        data = _floor_status(_flash_data(), "qwen3.8-flash", "broken")
        contrast = data["contrasts"][3]
        contrast.update(verdict="improved", aspect=ASPECT)
        plan = _recommending_flash(_plan())
        plan["answers"] = [_answer("floor_broken", ASPECT, FLOOR)]
        _passes(plan, data)


# --- the warnings and their answers --------------------------------------------------------------------------------


class TestRaisedWarnings:
    @pytest.mark.parametrize(("status", "kind"), [("broken", "floor_broken"), ("unclear", "floor_unclear")])
    def test_a_floor_not_held_warns(self, status: str, kind: str):
        warnings = raised_warnings(_input(data=_floor_status(_input_dict(), "qwen3.8-flash", status)), "qwen3.8-flash")
        assert [(w.kind, w.aspect, w.name) for w in warnings] == [(kind, ASPECT, FLOOR)]

    def test_a_held_floor_warns_nothing(self):
        assert raised_warnings(_input(), "qwen3.8-flash") == []

    def test_another_arm_ahead_on_a_pair_warns_behind_on(self):
        data = _input_dict()
        data["aspects"][0]["pairs"] = [{"arms": ["September", "1017"], "change": -3.0, "ahead": "1017"}]
        warnings = raised_warnings(_input(data=data), "September")
        assert [(w.kind, w.aspect, w.name) for w in warnings] == [("behind_on", ASPECT, "1017")]
        assert raised_warnings(_input(data=data), "1017") == []

    def test_a_missed_bar_warns(self):
        data = _input_dict()
        data["bars"] = [
            {"aspect": ASPECT, "arm": "September", "measure": "passed", "bar": 0.9,
             "direction": "higher_is_better", "status": "missed"},
            {"aspect": ASPECT, "arm": "1017", "measure": "passed", "bar": 0.9,
             "direction": "higher_is_better", "status": "met"},
        ]  # fmt: skip
        warnings = raised_warnings(_input(data=data), "September")
        assert [(w.kind, w.aspect, w.name) for w in warnings] == [("bar_missed", ASPECT, "passed")]


class TestAnswers:
    def test_a_broken_floor_on_the_recommended_arm_needs_its_answer(self):
        data = _floor_status(_input_dict(), "qwen3.8-flash", "broken")
        data["aspects"][0]["readings"][0]["reading"] = "better"
        assert "floor_broken" in _why(_recommending_flash(_plan()), "answers", data)

    def test_an_unclear_floor_needs_an_answer_of_its_own_kind(self):
        data = _floor_status(_flash_data(), "qwen3.8-flash", "unclear")
        data["aspects"][0]["readings"][0]["reading"] = "better"
        plan = _recommending_flash(_plan())
        assert "what sample would settle it" in _why(plan, "answers", data)
        plan["answers"] = [_answer("floor_broken", ASPECT, FLOOR)]
        paths = _paths(plan, data)
        assert {"answers", "answers[0].warning"} <= set(paths)
        plan["answers"] = [_answer("floor_unclear", ASPECT, FLOOR)]
        _passes(plan, data)

    def test_a_warning_is_answered_once(self):
        data = _floor_status(_input_dict(), "qwen3.8-flash", "broken")
        data["aspects"][0]["readings"][0]["reading"] = "better"
        plan = _recommending_flash(_plan())
        plan["answers"] = [_answer("floor_broken", ASPECT, FLOOR)] * 2
        assert "exactly one answer" in _why(plan, "answers[1].warning", data)

    def test_an_answer_to_a_warning_never_raised_is_refused(self):
        plan = _plan()
        plan["answers"] = [_answer("floor_broken", ASPECT, FLOOR)]
        assert "did not raise on 'September'" in _why(plan, "answers[0].warning")

    def test_a_missed_bar_on_the_recommended_arm_needs_its_answer(self):
        data = _input_dict()
        data["bars"] = [
            {"aspect": ASPECT, "arm": "September", "measure": "passed", "bar": 0.9,
             "direction": "higher_is_better", "status": "missed"},
        ]  # fmt: skip
        assert "bar_missed 'passed'" in _why(_plan(), "answers", data)
        plan = _plan()
        plan["answers"] = [_answer("bar_missed", ASPECT, "passed")]
        _passes(plan, data)


# --- the flag ------------------------------------------------------------------------------------------------------


#: Kairo's open report-wide caveats, in the order code lists them.
KAIRO_OPEN = ["late:aspect", "late:control", "judge_uncalibrated"]


def _low(plan: dict[str, Any], flagged: list[str]) -> dict[str, Any]:
    """The plan at a low confidence, with ``flagged`` moved from its clauses to a flag."""
    confidence = plan["decision"]["confidence"]
    confidence["form"] = "low"
    confidence["caveats"] = [clause for clause in confidence["caveats"] if clause["id"] not in flagged]
    plan["flag"] = [{**_NO_RULE_FLAG, "title": "The judge is uncalibrated.", "discloses": flagged}]
    return plan


class TestImpliedDisclosures:
    def test_kairo_implies_its_open_late_criteria_and_the_judge(self):
        """The late margin and floor were proved immaterial, so they are not disclosures; the aspect, the control and
        the judge are."""
        assert implied_disclosures(_input(), "decide") == KAIRO_OPEN

    def test_a_calibrated_judge_is_not_disclosed(self):
        data = _input_dict()
        data["caveats"]["judge_tiers"] = {}
        for judged in data["judged"].values():
            judged.update(tier="calibrated", calibration="calibrated")
        assert implied_disclosures(_input(data=data), "decide") == ["late:aspect", "late:control"]

    def test_one_calibrated_dimension_leaves_the_judge_undisclosed(self):
        data = _input_dict()
        data["judged"]["__outcome__"].update(calibration="calibrated")
        assert "judge_uncalibrated" not in implied_disclosures(_input(data=data), "decide")

    def test_a_confound_on_every_candidates_contrasts_is_report_wide(self):
        data = _input_dict()
        for contrast in data["contrasts"]:
            if contrast["arm"] == "qwen3.8-flash":
                contrast["confounds"] = ["observed:reasoning_ratio"]
        assert "confound:observed:reasoning_ratio" in implied_disclosures(_input(data=data), "decide")

    def test_a_confound_on_one_candidate_is_scoped(self):
        assert not [item for item in implied_disclosures(_input(), "decide") if item.startswith("confound:")]

    def test_a_short_run_of_the_control_is_report_wide_and_of_one_candidate_scoped(self):
        data = _input_dict()
        assert not [item for item in implied_disclosures(_input(data=data), "decide") if item.startswith("short_run:")]
        data["caveats"]["short_runs"][0]["arm"] = "September"
        run = data["caveats"]["short_runs"][0]["run"]
        assert f"short_run:{run}" in implied_disclosures(_input(data=data), "decide")

    def test_a_change_with_one_candidate_makes_its_short_run_report_wide(self):
        data = _input_dict()
        data["arms"] = data["arms"][:2]
        runs = [f"short_run:{run['run']}" for run in data["caveats"]["short_runs"]]
        assert set(runs) <= set(implied_disclosures(_input(data=data), "decide"))

    def test_an_apparatus_confound_on_every_comparison_is_report_wide(self):
        data = _input_dict()
        for contrast in data["contrasts"]:
            contrast["confounds"] = ["simulator_sent_settings"]
        assert "apparatus:simulator_sent_settings" in implied_disclosures(_input(data=data), "decide")


class TestNoRule:
    def test_a_tradeoff_with_no_declared_rule_implies_no_rule(self):
        assert implied_disclosures(_input("r4"), "tradeoff") == ["no_rule"]

    @pytest.mark.parametrize("mode", ["decide", "no_call"])
    def test_only_a_tradeoff_implies_no_rule(self, mode: str):
        assert implied_disclosures(_input("r4"), mode) == []

    def test_a_declared_merit_priority_is_a_rule(self):
        data = _input_dict("r4")
        data["contrast_families"][0]["merit_axes"] = ["quality", "cost"]
        assert implied_disclosures(_input(data=data), "tradeoff") == []

    def test_a_tradeoff_flag_disclosing_no_rule_passes(self):
        _passes(_tradeoff(_plan()), _tradeoff_data())

    def test_a_tradeoff_plan_with_no_flag_is_refused(self):
        plan = _set(_tradeoff(_plan()), "flag", [])
        assert "no_rule" in _why(plan, "flag", _tradeoff_data())

    def test_a_tradeoff_flag_leaving_out_no_rule_is_refused(self):
        plan = _low(_tradeoff(_plan()), ["judge_uncalibrated"])
        assert "no_rule" in _why(plan, "flag[0].discloses", _tradeoff_data())

    def test_a_decide_plan_disclosing_no_rule_is_refused(self):
        plan = _low(_plan(), ["no_rule", "judge_uncalibrated"])
        assert "a decide plan makes the call" in _why(plan, "flag[0].discloses")

    def test_a_tradeoff_with_a_declared_rule_discloses_no_no_rule(self):
        data = _tradeoff_data()
        data["contrast_families"][0]["merit_axes"] = ["quality", "cost"]
        plan = _low(_tradeoff(_plan()), ["no_rule", "judge_uncalibrated"])
        assert "merit priority" in _why(plan, "flag[0].discloses", data)


class TestTheFlag:
    def test_kairos_open_caveats_sit_in_its_confidence_so_it_has_no_flag(self):
        assert _plan()["flag"] == []
        _passes(_plan())

    def test_under_a_low_confidence_a_report_wide_caveat_may_be_flagged_instead(self):
        _passes(_low(_plan(), ["judge_uncalibrated"]))

    def test_a_scoped_confound_in_the_flag_is_refused(self):
        plan = _low(_plan(), ["judge_uncalibrated", "confound:observed:reasoning_ratio"])
        assert "Watch" in _why(plan, "flag[0].discloses")

    def test_a_scoped_short_run_in_the_flag_is_refused(self):
        run = _input_dict()["caveats"]["short_runs"][0]["run"]
        plan = _low(_plan(), ["judge_uncalibrated", f"short_run:{run}"])
        assert "flag[0].discloses" in _paths(plan)

    def test_a_disclosure_named_twice_is_refused(self):
        plan = _low(_plan(), ["judge_uncalibrated", "judge_uncalibrated"])
        assert "flag[0].discloses" in _paths(plan)

    def test_a_flag_with_nothing_report_wide_to_disclose_is_refused(self):
        data = _input_dict()
        data["caveats"]["late_criteria"] = []
        data["caveats"]["judge_tiers"] = {}
        for judged in data["judged"].values():
            judged.update(tier="calibrated", calibration="calibrated")
        plan = _set(_plan(), "decision.confidence", {"form": "low", "caveats": []})
        plan["flag"] = [{**_NO_RULE_FLAG, "discloses": ["confound:model"]}]
        assert "drop the flag" in _why(plan, "flag[0]", data)


# --- the confidence ------------------------------------------------------------------------------------------------


class TestTheConfidence:
    def test_plain_high_while_caveats_stay_open_is_refused_naming_them(self):
        plan = _set(_plan(), "decision.confidence", {"form": "high", "caveats": []})
        why = _why(plan, "decision.confidence.form")
        assert all(repr(caveat) in why for caveat in KAIRO_OPEN)

    def test_plain_high_with_nothing_open_passes(self):
        data = _input_dict()
        data["caveats"]["late_criteria"] = [late for late in data["caveats"]["late_criteria"] if late["name"]]
        data["caveats"]["late_criteria"].append({"criterion": "margin", "name": None, "set_at": "25 Oct 2026"})
        data["caveats"]["judge_tiers"] = {}
        for judged in data["judged"].values():
            judged.update(tier="calibrated", calibration="calibrated")
        _passes(_set(_plan(), "decision.confidence", {"form": "high", "caveats": []}), data)

    def test_plain_high_listing_a_caveat_is_refused(self):
        data = _input_dict()
        data["caveats"]["late_criteria"] = []
        data["caveats"]["judge_tiers"] = {}
        for judged in data["judged"].values():
            judged.update(tier="calibrated", calibration="calibrated")
        plan = _set(_plan(), "decision.confidence", {"form": "high", "caveats": [_clause("confound:model")]})
        plan["decision"]["slots"][2]["caveats"].remove("confound:model")
        assert "conditional on none" in _why(plan, "decision.confidence.caveats", data)

    def test_an_open_caveat_left_out_of_a_conditional_confidence_is_refused(self):
        plan = _plan()
        plan["decision"]["confidence"]["caveats"].pop(1)
        assert "'late:aspect'" in _why(plan, "decision.confidence.caveats")

    def test_an_open_caveat_moved_to_the_flag_under_a_conditional_confidence_is_refused(self):
        plan = _low(_plan(), ["judge_uncalibrated"])
        plan["decision"]["confidence"]["form"] = "high_if_immaterial"
        assert "'judge_uncalibrated'" in _why(plan, "decision.confidence.caveats")

    @pytest.mark.parametrize("text", ["", "   "])
    def test_an_open_caveat_without_its_clause_is_refused(self, text: str):
        plan = _set(_plan(), "decision.confidence.caveats[0].material_if", text)
        assert "judge_uncalibrated" in _why(plan, "decision.confidence.caveats[0].material_if")

    def test_conditional_confidence_with_nothing_open_is_refused(self):
        data = _input_dict()
        data["caveats"]["late_criteria"] = []
        data["caveats"]["judge_tiers"] = {}
        for judged in data["judged"].values():
            judged.update(tier="calibrated", calibration="calibrated")
        plan = _set(_plan(), "decision.confidence.caveats", [])
        assert "'high'" in _why(plan, "decision.confidence.form", data)

    def test_low_is_allowed_whatever_the_caveats(self):
        _passes(_set(_plan(), "decision.confidence.form", "low"))

    def test_a_caveat_code_proved_immaterial_is_refused_in_the_confidence(self):
        plan = _listing(_plan(), "late:margin")
        why = _why(plan, "decision.confidence.caveats[3].id")
        assert "immaterial" in why and "the call holds at every margin" in why

    def test_the_flash_recommendation_must_list_the_late_margin_it_now_rests_on(self):
        plan = _recommending_flash(_plan())
        assert "'late:margin'" in _why(plan, "decision.confidence.caveats", _flash_data())


# --- coverage ------------------------------------------------------------------------------------------------------


class TestCoverage:
    def test_a_dropped_scoped_caveat_is_refused_naming_it(self):
        plan = _plan()
        plan["decision"]["slots"][2]["caveats"].remove("confound:model")
        assert "'confound:model'" in _why(plan, "decision.slots")

    def test_a_dropped_method_caveat_is_refused_naming_it(self):
        plan = _plan()
        plan["method"][5]["caveats"] = []
        assert "'latency'" in _why(plan, "method")

    def test_a_caveat_stated_twice_is_refused(self):
        plan = _set(_plan(), "sections[0].caveats", ["confound:model"])
        assert "exactly once" in _why(plan, "sections[0].caveats[0]")

    def test_a_report_wide_caveat_also_in_the_flag_is_refused(self):
        plan = _plan()
        plan["flag"] = [{**_NO_RULE_FLAG, "discloses": ["judge_uncalibrated"]}]
        assert "exactly once" in _why(plan, "flag[0].discloses[0]")

    @pytest.mark.parametrize("caveat", ["latency", "unreported:input_tokens", "apparatus:simulator_sent_settings"])
    def test_a_method_level_caveat_may_sit_beside_a_claim_instead(self, caveat: str):
        plan = _plan()
        plan["method"] = [{**item, "caveats": [c for c in item["caveats"] if c != caveat]} for item in plan["method"]]
        plan["decision"]["slots"][2]["caveats"].append(caveat)
        _passes(plan)

    def test_method_takes_a_scoped_caveat_never(self):
        plan = _plan()
        plan["decision"]["slots"][2]["caveats"].remove("confound:model")
        plan["method"][0]["caveats"] = ["confound:model"]
        assert "method-level" in _why(plan, "method[0].caveats[0]")

    def test_method_takes_a_report_wide_caveat_never(self):
        plan = _plan()
        plan["decision"]["confidence"]["caveats"].pop(1)
        plan["method"][0]["caveats"] = ["late:aspect"]
        assert "report-wide" in _why(plan, "method[0].caveats[0]")

    def test_a_report_wide_caveat_beside_one_claim_is_refused(self):
        plan = _plan()
        plan["decision"]["confidence"]["caveats"].pop(0)
        plan["decision"]["slots"][2]["caveats"].append("judge_uncalibrated")
        assert "report-wide" in _why(plan, "decision.slots[2].caveats[4]")

    def test_an_immaterial_caveat_landed_anywhere_is_refused(self):
        plan = _plan()
        plan["method"][0]["caveats"] = [f"late:floor:{FLOOR}"]
        assert "immaterial" in _why(plan, "method[0].caveats[0]")

    def test_an_unknown_caveat_is_refused(self):
        plan = _set(_plan(), "sections[1].caveats", ["confound:temperature"])
        assert "no caveat of this input" in _why(plan, "sections[1].caveats[0]")


# --- charts --------------------------------------------------------------------------------------------------------


class TestCharts:
    def test_summary_arms_out_of_the_options_order_are_refused(self):
        plan = _set(_plan(), "summary[1].query.arms", ["September", "1017", "qwen3.8-flash"])
        assert "in its order" in _why(plan, "summary[1].query.arms")

    def test_a_change_from_control_summary_chart_leaves_the_control_out(self):
        plan = _set(_plan(), "summary[0].query.arms", ["September", "qwen3.8-flash", "1017"])
        assert "leaves out the control" in _why(plan, "summary[0].query.arms")

    def test_a_kind_that_cannot_draw_its_source_is_refused(self):
        assert "cannot draw 'measures'" in _why(_set(_plan(), "summary[1].kind", "forest"), "summary[1].kind")
        assert "sections[1].chart.kind" in _paths(_set(_plan(), "sections[1].chart.kind", "lattice"))

    def test_a_forest_over_rows_without_intervals_is_refused(self):
        data = _input_dict()
        for contrast in data["contrasts"]:
            if contrast["measure"] == "passed":
                contrast.update(interval=None, verdict="untested")
        assert "carries no interval" in _why(_plan(), "summary[0].kind", data)

    def test_an_absent_measure_is_refused(self):
        plan = _plan()
        plan["summary"][0]["query"]["measures"].append("persona.booth_stays_offstage")
        assert "persona.booth_stays_offstage" in _why(plan, "summary[0].query.measures")

    def test_a_measure_on_a_source_that_names_none_is_refused(self):
        plan = _set(_plan(), "sections[0].chart.query.measures", ["passed"])
        assert "names none" in _why(plan, "sections[0].chart.query.measures")

    def test_an_absent_arm_is_refused(self):
        plan = _set(_plan(), "sections[2].chart.query.arms", ["qwen3.8-flash", "1018"])
        assert "1018" in _why(plan, "sections[2].chart.query.arms")

    def test_an_absent_aspect_is_refused(self):
        plan = _set(_plan(), "sections[0].chart.query.aspect", "character adherence")
        assert "sections[0].chart.query.aspect" in _paths(plan)

    def test_an_absent_stratum_is_refused(self):
        plan = _set(_plan(), "sections[2].chart.query.strata", ["off_lane", "offlane"])
        assert "offlane" in _why(plan, "sections[2].chart.query.strata")

    def test_a_query_selecting_nothing_is_refused(self):
        plan = _set(_plan(), "sections[2].chart.query.arms", ["September"])
        assert "selects nothing" in _why(plan, "sections[2].chart.query")

    def test_an_unresolvable_reference_key_is_refused(self):
        plan = _set(_plan(), "sections[0].chart.reference[0].key", "aspect.floor")
        assert "sections[0].chart.reference[0].key" in _paths(plan)

    def test_an_emphasis_on_nothing_present_is_refused(self):
        assert "sections[0].chart.emphasis" in _paths(_set(_plan(), "sections[0].chart.emphasis", "qwen"))


# --- figure keys and D3 --------------------------------------------------------------------------------------------


class TestFigureKeys:
    def test_an_unknown_key_in_prose_is_refused_at_its_field(self):
        plan = _set(_plan(), "headline", "Keep {control.nickname}.")
        assert "control.nickname" in _why(plan, "headline")

    def test_a_filter_the_figure_does_not_suit_is_refused(self):
        plan = _set(_plan(), "method[3].text", "{judge.model}; evidence tier {judge.tier|pct}.")
        assert "method[3].text" in _paths(plan)

    def test_a_double_brace_is_refused(self):
        assert "decision.slots[2].text" in _paths(_set(_plan(), "decision.slots[2].text", "See {{cell|x}}."))

    def test_a_durable_claims_key_must_resolve(self):
        plan = _set(_plan(), "durable_claims[0].keys[0]", "cost.uplift.qwen3.9-flash")
        assert "durable_claims[0].keys[0]" in _paths(plan)

    def test_an_option_label_holds_no_key(self):
        plan = _set(_plan(), "explore[0].controls[1].options[0]", "Pass rate ({reading.1017.level})")
        assert "no figure key" in _why(plan, "explore[0].controls[1].options[0]")


#: Every prose field the Kairo plan holds: D3 and the figure keys cover each.
KAIRO_PROSE = [
    "headline",
    "lede",
    *(f"decision.slots[{index}].text" for index in range(4)),
    *(f"decision.confidence.caveats[{index}].material_if" for index in range(3)),
    "summary[0].title",
    "summary[1].title",
    *(
        f"sections[{index}].{field}"
        for index in range(3)
        for field in ("eyebrow", "headline", "takeaway", "caption", "chart.title")
    ),
    "explore[0].title",
    "explore[0].recomputes",
    "explore[0].controls[0].title",
    "explore[0].controls[1].title",
    *(f"method[{index}].{field}" for index in range(9) for field in ("term", "text")),
    "durable_claims[0].text",
    "durable_claims[1].text",
]

#: The plan's prose fields, by model: the D3 field set this module documents.
PROSE_FIELDS = {
    ("PlannedReport", "headline"),
    ("PlannedReport", "lede"),
    ("Flag", "title"),
    ("Flag", "text"),
    ("DecisionSlot", "text"),
    ("CaveatClause", "material_if"),
    ("WarningAnswer", "text"),
    ("DurableClaim", "text"),
    ("PlannedSection", "eyebrow"),
    ("PlannedSection", "headline"),
    ("PlannedSection", "takeaway"),
    ("PlannedSection", "caption"),
    ("PlannedChart", "title"),
    ("Explore", "title"),
    ("Explore", "recomputes"),
    ("ExploreControl", "title"),
    ("MethodItem", "term"),
    ("MethodItem", "text"),
    # The planner's own models of the same prose: the flag's words, an answer by its warning's position, and a prose
    # call's text, which assembles into one of the fields above.
    ("PlannedFlag", "title"),
    ("PlannedFlag", "text"),
    ("PlannedAnswer", "text"),
    ("ProseItem", "text"),
}


class TestD3:
    def test_the_prose_fields_are_the_documented_set(self):
        """D3 walks every field typed prose; this holds that set to the one the module documents."""
        models = [
            getattr(report_plan, name)
            for name in report_plan.__all__
            if isinstance(getattr(report_plan, name), type) and name not in ("ReportPlan", "ChartSpec", "Section")
        ]
        prose = {
            (model.__name__, name)
            for model in models
            for name, field in model.model_fields.items()
            if field_is_prose(field)
        }
        assert prose == PROSE_FIELDS

    def test_the_kairo_plan_holds_every_prose_field_listed(self):
        plan = _plan()
        for path in KAIRO_PROSE:
            assert isinstance(_get(plan, path), str), path

    @pytest.mark.parametrize("path", KAIRO_PROSE)
    def test_a_typed_digit_in_any_prose_field_is_refused_there(self, path: str):
        plan = _plan()
        _set(plan, path, f"{_get(plan, path)} 7")
        assert "['7']" in _why(plan, path)

    @pytest.mark.parametrize("path", ["flag[0].title", "flag[0].text"])
    def test_a_typed_digit_in_the_flag_is_refused(self, path: str):
        plan = _tradeoff(_plan())
        _set(plan, path, f"{_get(plan, path)} 7")
        assert "['7']" in _why(plan, path, _tradeoff_data())

    def test_a_typed_digit_in_an_answer_is_refused(self):
        data = _input_dict()
        data["bars"] = [
            {"aspect": ASPECT, "arm": "September", "measure": "passed", "bar": 0.9,
             "direction": "higher_is_better", "status": "missed"},
        ]  # fmt: skip
        plan = _plan()
        plan["answers"] = [{**_answer("bar_missed", ASPECT, "passed"), "text": "It misses by 2 points."}]
        assert "['2']" in _why(plan, "answers[0].text", data)

    def test_a_typed_digit_in_an_option_label_is_refused(self):
        plan = _set(_plan(), "explore[0].controls[1].options[0]", "Pass rate (95%)")
        assert "['95']" in _why(plan, "explore[0].controls[1].options[0]")

    def test_a_controls_scale_and_the_code_filled_masthead_are_not_prose(self):
        plan = _set(_plan(), "explore[0].controls[0].max", 25)
        plan["masthead"]["system"] = "Kairo persona model, round 3"
        _passes(plan)

    def test_names_the_input_holds_keep_their_digits(self):
        plan = _set(_plan(), "sections[2].takeaway", "Failed trials of qwen3.8-flash on deepseek/deepseek-v4.1-flash.")
        _passes(plan)
