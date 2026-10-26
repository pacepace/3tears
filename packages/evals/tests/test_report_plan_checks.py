"""The report plan's checks: code classifies the eval, and refuses a plan that disagrees with itself or its input.

Contracts held here (``threetears.evals.analysis.report_plan_checks``, GENERATOR.md §1–§3, §6.2):

- the kit's Kairo round 2 plan passes against r5's input: the base case every refusal below breaks one thing of;
- :func:`classify` looks r1–r5 up as A, B, C, D, D, and an ordered-versions input as E; the criteria signal wins a
  control under hard gates;
- :func:`spine` reads each shape's spine chart and the query fields the input settles, the rubric's spine table, and
  a plan whose first evidence section is not it (its kind, a query field or reference line code fixed) is refused;
- each check refuses at the field path it names: code's facts, mode (one code allows), slots, roles, ids, the
  recommendation (the memo's refusals ported, and the one option that dominates; a choice's pick no pair reading
  shows ahead is stated at low confidence unless declared gates leave it the one dominant survivor), the warning answers
  (``floor_unclear`` among them), the confidence's form and clauses, the flag, caveat coverage, the charts (each kind
  over the sources it draws, small multiples only of a kind the kit panels), the options columns (one of
  ``OPTION_COLUMN_FAMILIES``), the figure keys, D3, and no unit written right after a key that prints it nor money per
  reply under one cent on a page stating costs per 1,000 replies, over every prose field the plan holds;
- completion fills each reference line's key: a margin at the chart's aspect's margin, a floor or a budget at the
  threshold of the one criterion reading the chart's one measure (direction, statistic and place matched), and refuses
  a line with no such criterion or several, naming why.

Modes other than r5's own need inputs that allow them (:mod:`threetears.evals.analysis.report_decision`): a tradeoff
variant where 1017 is the cheapest arm and nothing shows it no worse, a no-call variant where a standing guardrail
removes every arm, and a flash variant where qwen3.8-flash is the cheapest arm and so dominates.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import BaseModel

from packages.evals.tests.report_kit_support import KIT
from packages.evals.tests.report_plan_support import planned_reply, structure_reply
from threetears.evals.analysis.plan_assembly import check_structure
from threetears.evals.analysis.report_plan_facts import complete_plan, plan_facts
from threetears.evals.analysis.errors import PlanRefusal, SoundnessRefusal
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_decision import surviving_options
from threetears.evals.analysis.report_plan_checks import (
    OPTION_COLUMN_FAMILIES,
    classify,
    implied_disclosures,
    raised_warnings,
    spine,
    validate_plan,
)
from threetears.evals.kernel import report_plan
from threetears.evals.kernel.report_plan import PlannedReport, PlanStructure, ReportPlan
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


def _ordered_versions() -> ReportInput:
    """Kairo read as ordered versions: shape E."""
    data = _input_dict()
    data["meta"]["design_shape"] = "ordered_versions"
    return _input(data=data)


#: Each shape's spine chart kind, read through :func:`spine` off one input per shape: r1 to r4, then Kairo as ordered
#: versions.
_SPINES: dict[str, str] = {
    classify(given)[1]: spine(given).kind
    for given in (*(_input(name) for name in ("r1", "r2", "r3", "r4")), _ordered_versions())
}


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
        signals, shape = classify(_ordered_versions())
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


# --- the spine -----------------------------------------------------------------------------------------------------


#: A corrected contrast showing qwen3.8-flash's p95 improved on an asked latency aspect.
_P95_IMPROVED: dict[str, Any] = {
    "arm": "qwen3.8-flash",
    "control": "September",
    "kind": "measure",
    "measure": "research_p95_s",
    "aspect": "latency",
    "verdict": "improved",
    "delta": -0.8,
    "interval": [-1.2, -0.4],
    "p_adjusted": 0.001,
}


def _latency_question() -> dict[str, Any]:
    """r5 as a measured question: its pass-rate readings unclear with no margin and no floor read, and an asked latency
    aspect on which a corrected contrast separates qwen3.8-flash."""
    data = _input_dict()
    data["aspects"][0]["margin"] = None
    for reading in data["aspects"][0]["readings"]:
        reading["within_interval"] = [-6.0, 2.0]
        reading["margin"] = None
    data["floors"] = []
    data["aspects"].append({"id": "latency", "name": "latency", "asked": True, "readings": []})
    data["measures"]["research_p95_s"] = {"unit": "s", "higher_is_better": False, "per_arm": {}}
    data["contrasts"].append(_P95_IMPROVED)
    return data


def _first_chart_faults(report: ReportInput, aspect: str, measures: list[str]) -> list[tuple[str, str]]:
    """The faults the structure check finds at the first section's chart when it is a forest of contrasts on
    ``aspect`` over ``measures``: code's facts complete it there, its one unit and one level among them."""
    structure = structure_reply(_plan())
    structure["sections"][0]["chart"] = {
        "kind": "forest",
        "query": {"source": "contrasts", "aspect": aspect, "measures": measures, "arms": [], "strata": [],
                  "limit": 0, "transform": "none", "small_multiples_by": "none"},
        "reference": [],
        "emphasis": "",
    }  # fmt: skip
    facts = plan_facts(report)
    try:
        check_structure(PlanStructure.model_validate(structure), report, facts, facts.allowed_modes)
    except PlanRefusal as refused:
        return [(path, why) for path, why in refused.faults if path.startswith("sections[0].chart")]
    return []


class TestTheSpine:
    @pytest.mark.parametrize(
        ("name", "fact"),
        [
            ("r1", {"kind": "gate_matrix", "query": {"source": "tasks", "aspect": ""}, "reference": []}),
            (
                "r2",
                {
                    "kind": "gate_scatter",
                    "query": {"source": "configs", "aspect": "", "measures": ["pass_rate", "latency_ms"]},
                    "reference": [],
                },
            ),
            (
                "r3",
                {
                    "kind": "lattice",
                    "query": {"source": "configs", "aspect": "", "measures": ["balanced_accuracy"]},
                    "reference": [],
                },
            ),
            ("r4", {"kind": "forest", "query": {"source": "contrasts", "aspect": ""}, "reference": []}),
            (
                "r5",
                {
                    "kind": "forest",
                    "query": {"source": "readings", "aspect": ASPECT, "measures": []},
                    "reference": [{"kind": "margin"}],
                },
            ),
        ],
    )
    def test_each_fixture_reads_its_shapes_spine(self, name: str, fact: dict[str, Any]):
        assert spine(_input(name)).as_fact() == fact

    def test_a_measured_question_opens_on_the_contrasts_that_decided_it(self):
        """No candidate separated by an asked reading (unclear, no margin), one separated by a corrected p95 contrast on
        the asked latency aspect: the forest reads that contrast, the evidence the call rests on."""
        data = _latency_question()
        assert spine(_input(data=data)).as_fact() == {
            "kind": "forest",
            "query": {"source": "contrasts", "aspect": "latency", "measures": ["research_p95_s"]},
            "reference": [],
        }

    def test_a_measured_spine_keeps_one_aspect_unit_and_level_so_its_own_query_passes_the_unit_check(self):
        """Two asked aspects decide the call in two units (a p95 in seconds, a judged rate in points): the spine reads
        the first deciding contrast's aspect and the measures there in its unit, so the chart it fixes carries one."""
        data = _latency_question()
        data["aspects"].append({"id": "tone", "name": "tone", "asked": True, "readings": []})
        data["contrasts"].append(
            {**_P95_IMPROVED, "kind": "judged", "measure": "persona.holds_character", "aspect": "tone"}
        )
        report = _input(data=data)
        found = spine(report)
        assert (found.source, found.aspect, found.measures) == ("contrasts", "latency", ("research_p95_s",))
        assert _first_chart_faults(report, found.aspect, list(found.measures)) == []
        # The check bites: both deciding measures, across both aspects, mix seconds and points on one chart.
        mixed = _first_chart_faults(report, "", ["research_p95_s", "persona.holds_character"])
        assert any(path == "sections[0].chart.query" and "one unit" in why for path, why in mixed), mixed

    def test_the_measured_spine_reads_the_survivors_as_the_modes_do(self):
        """A candidate a gate removes does not hold the spine on readings: qwen3.8-flash's reading separates it, and
        it fails a cost guardrail (a guardrail keeps shape D, a gate would make it A); 1017 survives, separated only by
        its p95 contrast."""
        data = _latency_question()
        data["aspects"][0]["readings"][0]["within_interval"] = [-10.79, -3.381]
        data["contrasts"][-1] = {**_P95_IMPROVED, "arm": "1017"}
        data["criteria"] = [
            {"id": "cost", "kind": "guardrail", "measure": "production_replicating_cost",
             "direction": "lower_is_better", "threshold": 0.00241}
        ]  # fmt: skip
        report = _input(data=data)
        assert surviving_options(report) == ["September", "1017"]
        found = spine(report)
        assert (found.source, found.aspect, found.measures) == ("contrasts", "latency", ("research_p95_s",))

    def test_a_contrasts_forest_draws_no_margin_line_though_the_aspect_declares_one(self):
        data = _input_dict()
        for reading in data["aspects"][0]["readings"]:
            reading["within_interval"] = None
        assert data["aspects"][0]["margin"] is not None
        found = spine(_input(data=data))
        assert (found.source, found.measures, found.reference) == ("contrasts", None, ())

    def test_readings_that_separate_keep_the_readings_spine_beside_a_separating_contrast(self):
        data = _input_dict()
        data["aspects"].append({"id": "latency", "name": "latency", "asked": True, "readings": []})
        data["measures"]["research_p95_s"] = {"unit": "s", "higher_is_better": False, "per_arm": {}}
        data["contrasts"].append(_P95_IMPROVED)
        found = spine(_input(data=data))
        assert (found.source, found.aspect, found.reference) == ("readings", ASPECT, ("margin",))

    def test_a_spine_the_input_settles_no_source_for_leaves_it_to_the_planner(self):
        data = _input_dict("r1")
        data["tasks"] = []
        found = spine(_input(data=data))
        assert (found.kind, found.source, found.as_fact()["query"]) == ("gate_matrix", "", {})

    def test_a_gate_scatter_names_its_axes_only_where_the_gates_read_two_measures(self):
        data = _input_dict("r2")
        data["criteria"] = data["criteria"][:1]
        assert spine(_input(data=data)).measures is None

    def test_the_first_section_off_the_spine_is_refused_naming_it(self):
        plan = _plan()
        plan["sections"][0], plan["sections"][1] = plan["sections"][1], plan["sections"][0]
        assert _paths(plan) == ["sections[0].chart.kind"]
        why = _why(plan, "sections[0].chart.kind")
        assert "is 'diverging_bars'" in why and "shape D's spine, code's fact, is a 'forest' of 'readings'" in why

    @pytest.mark.parametrize(
        ("path", "value"),
        [
            ("sections[0].chart.query.source", "contrasts"),
            ("sections[0].chart.query.aspect", ""),
            ("sections[0].chart.reference", []),
        ],
    )
    def test_a_query_field_or_reference_code_fixed_is_refused_off_it(self, path: str, value: Any):
        assert path in _paths(_set(_plan(), path, value))

    def test_a_gate_scatter_off_codes_axes_is_refused(self):
        plan: dict[str, Any] = json.loads((FIXTURES / "r2" / "anchor.plan.json").read_text(encoding="utf-8"))
        plan["sections"][0]["chart"]["query"]["measures"] = ["pass_rate", "cost_per_1k"]
        faults = _faults(plan, _input_dict("r2"))
        assert [path for path, _ in faults] == ["sections[0].chart.query.measures"]

    def test_a_plan_with_no_evidence_section_is_refused(self):
        plan = _set(_plan(), "sections", [])
        assert "hold no evidence section" in _why(plan, "sections")

    def test_what_code_does_not_fix_is_the_planners(self):
        plan = _set(_set(_plan(), "sections[0].chart.query.arms", ["1017"]), "sections[0].chart.emphasis", "1017")
        _passes(plan)

    def test_every_shapes_spine_is_the_rubrics(self):
        """Every shape's spine kind, read off an input of that shape, is the rubric's."""
        rubric: dict[str, Any] = json.loads((FIXTURES / "rubric" / "v2.json").read_text(encoding="utf-8"))
        assert _SPINES == {shape: kind for shape, [kind] in rubric["spine"].items()}


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


class TestAllowedModes:
    @pytest.mark.parametrize("make", [_tradeoff, _no_call])
    def test_a_mode_code_does_not_allow_is_refused(self, make: Any):
        why = _why(make(_plan()), "mode")
        assert "allows only ['decide']" in why and "'September' dominates" in why

    def test_a_decision_on_a_tradeoff_input_is_refused(self):
        assert "allows only ['tradeoff']" in _why(_plan(), "mode", _tradeoff_data())

    def test_a_decision_must_recommend_the_option_that_dominates(self):
        assert "recommend 'September'" in _why(_recommending_flash(_plan()), "options.rows[0].option")

    def test_the_option_that_dominates_may_be_a_candidate(self):
        assert "mode" not in _paths(_recommending_flash(_plan()), _flash_data())


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


def _r4_plan() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((FIXTURES / "r4" / "anchor.plan.json").read_text(encoding="utf-8"))
    return loaded


def _cost_gate(threshold: float, direction: str = "lower_is_better") -> dict[str, Any]:
    return {
        "id": "cost",
        "kind": "gate",
        "measure": "production_replicating_cost",
        "direction": direction,
        "threshold": threshold,
    }


class TestOnlySurvivorsArePicked:
    """A recommended or lean row naming an arm names a survivor of the gates and guardrails, in every mode."""

    def test_a_decision_by_a_ranking_rule_recommends_a_survivor(self):
        data = _input_dict("r4")
        data["contrast_families"][0]["merit_axes"] = ["quality", "cost"]
        plan = _r4_plan()
        plan["options"]["rows"] = [
            {"option": "f16", "role": "recommended"},
            {"option": "prod", "role": "control"},
            {"option": "f64", "role": "alternative"},
        ]
        plan.update(mode="decide", headline_kind="decision")
        why = _why(plan, "options.rows[0].option", data)
        assert "'f16', marked 'recommended', and it does not survive" in why and "['prod', 'f64']" in why
        plan["options"]["rows"] = [
            {"option": "prod", "role": "recommended"},
            {"option": "f64", "role": "alternative"},
            {"option": "f16", "role": "alternative"},
        ]
        assert "options.rows[0].option" not in _paths(plan, data), "the control survives, and the rule may pick it"

    def test_a_tradeoffs_lean_is_a_survivor(self):
        plan = _r4_plan()
        _passes(plan, _input_dict("r4"))
        rows = plan["options"]["rows"]
        rows[1]["role"], rows[2]["role"] = "cheapest", "lean"
        assert "'f16', marked 'lean', and it does not survive" in _why(
            plan, "options.rows[2].option", _input_dict("r4")
        )

    def test_a_changes_candidate_shown_better_that_failed_a_gate_is_refused(self):
        data = _input_dict()
        data["criteria"] = [_cost_gate(0.0024)]
        data["contrasts"].append(
            {"arm": "qwen3.8-flash", "control": "September", "kind": "judged", "measure": "persona.holds_character",
             "aspect": ASPECT, "verdict": "improved", "delta": 3.0, "interval": [1.0, 5.0], "p_adjusted": 0.01}
        )  # fmt: skip
        why = _why(_recommending_flash(_plan()), "options.rows[0].option", data)
        assert "'qwen3.8-flash', marked 'recommended', and it does not survive" in why

    def test_a_choices_low_confidence_pick_a_gate_removed_is_refused(self):
        data = _input_dict()
        data["arms"][0]["role"] = "challenger"
        data["criteria"] = [_cost_gate(0.002, "higher_is_better")]
        plan = _set(_plan(), "decision.confidence.form", "low")
        assert "'September', marked 'recommended', and it does not survive" in _why(
            plan, "options.rows[0].option", data
        )


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

    def test_an_ungated_choice_whose_pairs_do_not_separate_its_pick_is_still_low(self):
        data = _input_dict()
        data["arms"][0]["role"] = "challenger"
        data["aspects"][0]["pairs"] = [{"arms": ["September", "1017"], "change": 0.4, "ahead": None}]
        assert "no declared gate" in _why(_plan(), "decision.confidence.form", data)

    @pytest.mark.parametrize("name", ["r1", "r2", "r3"])
    def test_a_choice_gates_leave_one_dominant_survivor_may_state_high(self, name: str):
        plan: dict[str, Any] = json.loads((FIXTURES / name / "anchor.plan.json").read_text(encoding="utf-8"))
        assert plan["decision"]["confidence"]["form"] == "high"
        _passes(plan, _input_dict(name))

    def test_a_gated_choice_is_settled_only_for_the_one_dominant_survivor(self):
        data = _input_dict()
        data["arms"][0]["role"] = "challenger"
        cost_gate = {
            "id": "cost",
            "kind": "gate",
            "measure": "production_replicating_cost",
            "direction": "lower_is_better",
            "threshold": 0.01,
        }
        data["criteria"] = [cost_gate]
        assert "decision.confidence.form" not in _paths(_plan(), data)
        data["criteria"] = [{**cost_gate, "direction": "higher_is_better", "threshold": 0.002}]
        assert "no declared gate" in _why(_plan(), "decision.confidence.form", data)

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


# --- an aspect unsettled past its margin --------------------------------------------------------------------------


def _nothing_open(data: dict[str, Any]) -> dict[str, Any]:
    """The input with no late criterion and a calibrated judge: no report-wide caveat of its own stays open."""
    data["caveats"]["late_criteria"] = []
    data["caveats"]["judge_tiers"] = {}
    for judged in data["judged"].values():
        judged.update(tier="calibrated", calibration="calibrated")
    return data


def _voice(data: dict[str, Any], arm: str | None = None) -> dict[str, Any]:
    """The input with an unasked ``voice`` aspect read like the asked one, against the same declared margin, and
    ``arm``'s reading there moved so its interval, its width kept, straddles the margin's line (``-margin``)."""
    voice = json.loads(json.dumps(data["aspects"][0]))
    voice.update(id="voice", name="voice", asked=False)
    for reading in voice["readings"]:
        if reading["arm"] == arm:
            low, high = reading["within_interval"]
            width, line = high - low, -reading["margin"]
            reading["within_interval"] = [line - width / 2, line + width / 2]
    data["aspects"].append(voice)
    return data


_UNSETTLED = {"id": "unsettled:voice", "material_if": "Voice is unsettled. Material only if a loss that large would "
              "change the call."}  # fmt: skip


def _confident(plan: dict[str, Any], form: str, *clauses: dict[str, str]) -> dict[str, Any]:
    return _set(plan, "decision.confidence", {"form": form, "caveats": list(clauses)})


def _ranked_flash() -> dict[str, Any]:
    """Nothing open; qwen3.8-flash shown better on an asked latency aspect and dearer than September, so nothing
    dominates and the declared priority (latency, then cost) decides: the pick is the plan's, not code's."""
    data = _nothing_open(_input_dict())
    data["aspects"].append({"id": "latency", "name": "latency", "asked": True, "readings": []})
    data["measures"]["research_p95_s"] = {"unit": "s", "higher_is_better": False, "per_arm": {}}
    data["contrasts"].append(_P95_IMPROVED)
    data["contrast_families"][0]["merit_axes"] = ["latency", "cost"]
    return data


def _confidence_and_coverage(plan: dict[str, Any], data: dict[str, Any]) -> list[tuple[str, str]]:
    """The faults the confidence and coverage checks find, apart from any the plan's r5 prose keys meet elsewhere."""
    try:
        validate_plan(ReportPlan.model_validate(plan), _input(data=data))
    except PlanRefusal as refused:
        return [(at, why) for at, why in refused.faults if at.startswith("decision.confidence") or "unsettled:" in why]
    return []


class TestUnsettledPastTheMargin:
    """GENERATOR §2.5 on the issue's state: the asked aspect decides the call for qwen3.8-flash, and its unasked voice
    reading allows a loss past the declared margin."""

    def test_inside_its_margin_plain_high_passes(self):
        data = _voice(_nothing_open(_flash_data()))
        assert "unsettled:voice" not in implied_disclosures(_input(data=data), "decide")
        _passes(_confident(_recommending_flash(_plan()), "high"), data)

    def test_plain_high_is_refused_naming_it(self):
        data = _voice(_nothing_open(_flash_data()), "qwen3.8-flash")
        assert implied_disclosures(_input(data=data), "decide") == ["unsettled:voice"]
        why = _why(_confident(_recommending_flash(_plan()), "high"), "decision.confidence.form", data)
        assert "open caveats remain: ['unsettled:voice']" in why

    def test_high_if_immaterial_carrying_it_passes(self):
        data = _voice(_nothing_open(_flash_data()), "qwen3.8-flash")
        _passes(_confident(_recommending_flash(_plan()), "high_if_immaterial", _UNSETTLED), data)

    def test_low_still_passes_listing_it_or_flagging_it(self):
        data = _voice(_nothing_open(_flash_data()), "qwen3.8-flash")
        _passes(_confident(_recommending_flash(_plan()), "low", _UNSETTLED), data)
        plan = _confident(_recommending_flash(_plan()), "low")
        plan["flag"] = [{**_NO_RULE_FLAG, "title": "Voice is unsettled.", "discloses": ["unsettled:voice"]}]
        _passes(plan, data)

    def test_on_a_candidate_not_picked_it_sits_beside_a_claim_under_plain_high(self):
        data = _voice(_nothing_open(_flash_data()), "1017")
        plan = _confident(_recommending_flash(_plan()), "high")
        plan["decision"]["slots"][2]["caveats"].append("unsettled:voice")
        _passes(plan, data)

    def test_a_pick_by_the_ranking_rule_is_conditional_on_it_though_code_scopes_it(self):
        data = _voice(_ranked_flash(), "qwen3.8-flash")
        assert "unsettled:voice" not in implied_disclosures(_input(data=data), "decide")
        high = dict(_confidence_and_coverage(_confident(_recommending_flash(_plan()), "high"), data))
        assert "open caveats remain: ['unsettled:voice']" in high["decision.confidence.form"]
        carried = _confident(_recommending_flash(_plan()), "high_if_immaterial", _UNSETTLED)
        assert _confidence_and_coverage(carried, data) == []

    def test_a_flag_disclosing_it_weakens_the_measures_its_aspects_contrasts_read(self):
        """A low plan leaves the open unsettled aspect to the flag, which weakens the measures read on that aspect and
        leaves the others standing."""
        data = _voice(_nothing_open(_flash_data()), "qwen3.8-flash")
        on_voice = {**data["contrasts"][1], "arm": "1017", "aspect": "voice"}
        data["contrasts"].append(on_voice)
        plan = _confident(_recommending_flash(_plan()), "low")
        plan["flag"] = [{**_NO_RULE_FLAG, "title": "Voice is unsettled.", "discloses": ["unsettled:voice"]}]
        completed, faults = _completed("r5", plan, data)
        assert faults == [] and completed is not None
        (flag,) = completed.flag
        assert flag.discloses == ["unsettled:voice"]
        assert flag.weakens == [f"judged:{on_voice['measure']}"]
        assert flag.weakens[0] not in flag.survives and flag.survives


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

    @pytest.mark.parametrize(
        ("name", "path", "source", "kind"),
        [
            ("r1", "sections[2].chart", "tasks", "range_bars"),
            ("r5", "sections[1].chart", "strata", "dot_interval"),
            ("r2", "sections[1].chart", "configs", "diverging_bars"),
        ],
    )
    def test_a_kind_draws_each_source_its_resolver_reads(self, name: str, path: str, source: str, kind: str):
        plan = _anchor(name)
        assert _get(plan, f"{path}.query.source") == source
        assert f"{path}.kind" not in _anchor_paths(_set(plan, f"{path}.kind", kind), name)

    def test_a_gate_matrix_draws_readings(self):
        plan = _set(_plan(), "summary[0].kind", "gate_matrix")
        plan["summary"][0]["query"] |= {"source": "readings", "aspect": ASPECT, "measures": []}
        assert "summary[0].kind" not in [path for path, _ in _faults_or_none(plan)]

    def test_range_bars_still_draw_no_strata(self):
        plan = _set(_plan(), "sections[1].chart.kind", "range_bars")
        assert "cannot draw 'strata'" in _why(plan, "sections[1].chart.kind")


def _anchor(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((FIXTURES / name / "anchor.plan.json").read_text(encoding="utf-8"))
    return loaded


def _anchor_faults(plan: dict[str, Any], name: str, data: dict[str, Any] | None = None) -> list[tuple[str, str]]:
    """The faults validate_plan finds on an edited anchor against its input; empty where it passes."""
    try:
        validate_plan(ReportPlan.model_validate(plan), _input(name, data))
    except PlanRefusal as refused:
        return list(refused.faults)
    return []


def _anchor_paths(plan: dict[str, Any], name: str, data: dict[str, Any] | None = None) -> list[str]:
    return [path for path, _ in _anchor_faults(plan, name, data)]


#: The kinds whose kit chart model carries ``panels`` (``kit/API.md``: bars, and diverging bars as bars, the two
#: scatters, dot intervals, the heatmap and confusion matrices).
PANELLED = {"bars", "diverging_bars", "scatter_frontier", "gate_scatter", "dot_interval", "heatmap", "confusion"}


class TestSmallMultiples:
    def test_small_multiples_of_a_forest_are_refused(self):
        plan = _set(_plan(), "summary[0].query.small_multiples_by", "arm")
        why = _why(plan, "summary[0].query.small_multiples_by")
        assert "no small multiples of a 'forest'" in why and "'none'" in why

    @pytest.mark.parametrize("kind", sorted(set(_SPINES.values()) | PANELLED | {"range_bars", "examples", "table"}))
    def test_only_a_kind_the_kit_panels_repeats_its_panel(self, kind: str):
        plan = _set(_plan(), "summary[1].kind", kind)
        _set(plan, "summary[1].query.small_multiples_by", "arm")
        refused = "summary[1].query.small_multiples_by" in [path for path, _ in _faults_or_none(plan)]
        assert refused == (kind not in PANELLED), kind

    def test_no_small_multiples_is_never_refused(self):
        plan = _set(_plan(), "summary[0].query.small_multiples_by", "none")
        assert "summary[0].query.small_multiples_by" not in [path for path, _ in _faults_or_none(plan)]


def _faults_or_none(plan: dict[str, Any], data: dict[str, Any] | None = None) -> list[tuple[str, str]]:
    try:
        validate_plan(ReportPlan.model_validate(plan), _input(data=data))
    except PlanRefusal as refused:
        return list(refused.faults)
    return []


class TestOptionColumns:
    def test_a_column_off_the_families_is_refused(self):
        plan = _plan()
        plan["options"]["columns"].append("monthly_cost")
        assert "is no column family" in _why(plan, "options.columns[5]")

    @pytest.mark.parametrize("column", ["judged:persona.booth_stays_offstage", "measure:production_replicating_cost"])
    def test_a_keyed_column_on_a_held_key_passes(self, column: str):
        plan = _plan()
        plan["options"]["columns"].insert(4, column)
        _passes(plan)

    @pytest.mark.parametrize("column", ["judged:production_replicating_cost", "measure:persona.nope", "judged:"])
    def test_a_keyed_column_on_a_key_the_input_does_not_hold_is_refused(self, column: str):
        plan = _plan()
        plan["options"]["columns"].append(column)
        assert "which the input does not hold" in _why(plan, "options.columns[5]")

    @pytest.mark.parametrize("name", ["r1", "r2", "r3", "r4", "r5"])
    def test_every_column_code_emits_is_one_of_the_families(self, name: str):
        for column in plan_facts(_input(name)).columns:
            family, colon, _ = column.partition(":")
            assert (column if not colon else f"{family}:" + ("<dim>" if family == "judged" else "<key>")) in (
                OPTION_COLUMN_FAMILIES
            ), column


#: r5 with a measure stated in seconds, read by one arm.
def _seconds_data() -> dict[str, Any]:
    data = _input_dict()
    data["measures"]["reply_s"] = {"unit": "s", "higher_is_better": False, "per_arm": {"September": {"mean": 3.2}}}
    return data


_SECONDS = "{measure.reply_s.September.mean}"
_R2_P95 = "{measure.latency_ms.brisk-s.low.r1.p95|int} ms"


class TestUnits:
    @pytest.mark.parametrize("after", [" ms", " milliseconds", "ms", " MS"])
    def test_a_unit_after_a_key_that_prints_it_is_refused(self, after: str):
        plan = _replace_in(_anchor("r2"), "lede", _R2_P95, "{measure.latency_ms.brisk-s.low.r1.p95}" + after)
        why = " ".join(why for path, why in _anchor_faults(plan, "r2") if path == "lede")
        assert "right after {measure.latency_ms.brisk-s.low.r1.p95}" in why and "'513 ms'" in why

    def test_a_unit_after_a_key_spelled_bare_passes(self):
        plan = _anchor("r2")
        assert _R2_P95 in plan["lede"]
        assert "lede" not in _anchor_paths(plan, "r2")

    @pytest.mark.parametrize("after", [" seconds", " second", " s.", " secs"])
    def test_seconds_after_a_seconds_key_are_refused(self, after: str):
        plan = _set(_plan(), "sections[0].caption", f"The reply takes {_SECONDS}{after}")
        assert "right after" in _why(plan, "sections[0].caption", _seconds_data())

    @pytest.mark.parametrize("after", [" so it is slow.", " slower", "."])
    def test_a_word_that_only_starts_like_a_unit_passes(self, after: str):
        plan = _set(_plan(), "sections[0].caption", f"The reply takes {_SECONDS}{after}")
        assert "sections[0].caption" not in [path for path, _ in _faults_or_none(plan, _seconds_data())]

    @pytest.mark.parametrize("after", ["%", " %", " percent", " per cent"])
    def test_a_percent_after_a_percent_key_is_refused(self, after: str):
        plan = _set(_plan(), "sections[0].caption", "{pass_rate.September|pct}" + after + " pass.")
        assert f"writes {after.strip()!r} right after {{pass_rate.September|pct}}" in _why(plan, "sections[0].caption")

    def test_points_after_a_key_spelled_in_points_pass(self):
        plan = _plan()
        assert "|pts} points" in _get(plan, "decision.slots[1].text")
        _passes(plan)

    def test_a_units_own_words_after_a_key_that_prints_them_are_refused(self):
        plan = _replace_in(
            _plan(),
            "decision.slots[0].text",
            "{cost_per_1k.September|usd_per_1k})",
            "{cost_per_1k.September|usd_per_1k} per 1,000 replies)",
        )
        assert "'per 1,000 replies'" in _why(plan, "decision.slots[0].text")

    def test_the_words_after_a_cost_spelled_without_them_pass(self):
        plan = _replace_in(
            _plan(),
            "decision.slots[0].text",
            "{cost_per_1k.September|usd_per_1k})",
            "{cost_per_1k.September} per 1,000 replies)",
        )
        _passes(plan)

    @pytest.mark.parametrize("after", [" dollars", " USD", " $"])
    def test_money_after_a_key_that_prints_its_sign_is_refused(self, after: str):
        plan = _replace_in(
            _plan(),
            "decision.slots[0].text",
            "{cost_per_1k.September|usd_per_1k})",
            "{cost_per_1k.September}" + after + ")",
        )
        assert "right after {cost_per_1k.September}" in _why(plan, "decision.slots[0].text")

    def test_a_unit_outside_the_vocabulary_is_refused_where_it_repeats_the_printed_one(self):
        plan = _anchor("r2")
        plan["lede"] += " It makes {measure.calls_per_task.atlas-m.off.r0.mean} calls a task."
        why = " ".join(why for path, why in _anchor_faults(plan, "r2") if path == "lede")
        assert "writes 'calls'" in why

    def test_the_per_million_words_after_a_key_that_prints_them_are_refused(self):
        plan = _anchor("r3")
        plan["lede"] += " Zero-shot costs {measure.cost_per_1m.atlas-m.zero.mean} per 1,000,000 replies."
        why = " ".join(why for path, why in _anchor_faults(plan, "r3") if path == "lede")
        assert "writes 'per 1,000,000 replies'" in why

    def test_the_per_million_words_after_a_whole_number_pass(self):
        plan = _anchor("r3")
        plan["lede"] += " Zero-shot costs {measure.cost_per_1m.atlas-m.zero.mean|int} dollars per 1,000,000 replies."
        assert "lede" not in _anchor_paths(plan, "r3")

    def test_money_per_reply_under_a_cent_is_refused_where_costs_read_per_thousand(self):
        plan = _replace_in(
            _plan(),
            "decision.slots[0].text",
            "({cost_per_1k.September|usd_per_1k})",
            "({scoreboard.September.reply_cost_usd} a reply)",
        )
        why = _why(plan, "decision.slots[0].text")
        assert "under one cent" in why and "{scoreboard.September.reply_cost_usd|usd_per_1k}" in why

    def test_money_per_reply_under_a_cent_keyed_per_thousand_passes(self):
        plan = _replace_in(
            _plan(),
            "decision.slots[0].text",
            "{cost_per_1k.September|usd_per_1k}",
            "{scoreboard.September.reply_cost_usd|usd_per_1k}",
        )
        _passes(plan)

    def test_a_change_in_money_under_a_cent_keeps_its_other_filters_in_the_key_it_asks_for(self):
        plan = _set(
            _plan(), "sections[0].caption", "It costs {contrast.1017.production_replicating_cost.delta|abs|usd} more."
        )
        assert "{contrast.1017.production_replicating_cost.delta|abs|usd_per_1k}" in _why(plan, "sections[0].caption")

    def test_money_per_reply_of_a_cent_or_more_passes(self):
        data = _input_dict()
        data["scoreboard"]["September"]["reply_cost_usd"] = 0.02
        plan = _replace_in(
            _plan(),
            "decision.slots[0].text",
            "{cost_per_1k.September|usd_per_1k}",
            "{scoreboard.September.reply_cost_usd}",
        )
        assert "decision.slots[0].text" not in [path for path, _ in _faults_or_none(plan, data)]

    def test_money_under_a_cent_passes_where_the_page_states_no_cost_per_thousand(self):
        data = _input_dict()
        for row in data["scoreboard"].values():
            row["reply_cost_usd"] = None
        for measure in data["measures"].values():
            measure.pop("merit_axis", None)
        data["measures"]["cost_usd"]["per_arm"]["September"]["mean"] = 0.004
        plan = _set(_plan(), "sections[0].caption", "A reply costs {measure.cost_usd.September.mean}.")
        assert "sections[0].caption" not in [path for path, _ in _faults_or_none(plan, data)]


def _replace_in(plan: dict[str, Any], path: str, old: str, new: str) -> dict[str, Any]:
    text = _get(plan, path)
    assert old in text, f"{path} no longer holds {old!r}"
    return _set(plan, path, text.replace(old, new))


# --- reference lines ------------------------------------------------------------------------------------------------


def _completed(
    name: str, plan: dict[str, Any], data: dict[str, Any] | None = None
) -> tuple[ReportPlan | None, list[tuple[str, str]]]:
    """An anchor's planned part completed against its input: the stored plan, or the faults completion found."""
    report = _input(name, data)
    row = next((row["option"] for row in plan["options"]["rows"] if row["role"] == "recommended"), None)
    warnings = [warning.model_dump() for warning in raised_warnings(report, row)] if row is not None else []
    try:
        return complete_plan(PlannedReport.model_validate(planned_reply(plan, warnings)), report), []
    except PlanRefusal as refused:
        return None, list(refused.faults)


def _line(name: str, path: str, kind: str, **query: Any) -> dict[str, Any]:
    plan = _anchor(name)
    _set(plan, f"{path}.reference", [{"kind": kind, "key": "placeholder"}])
    _get(plan, f"{path}.query").update(query)
    return plan


def _key_at(plan: ReportPlan, path: str) -> str:
    chart = _get(plan.model_dump(mode="json", by_alias=True), path)
    (reference,) = chart["reference"]
    return str(reference["key"])


def _line_fault(name: str, plan: dict[str, Any], path: str, data: dict[str, Any] | None = None) -> str:
    completed, faults = _completed(name, plan, data)
    assert completed is None
    found = [why for at, why in faults if at == f"{path}.reference[0].kind"]
    assert found, faults
    return " ".join(found)


class TestReferenceLines:
    @pytest.mark.parametrize("name", ["r1", "r2", "r3", "r4", "r5"])
    def test_each_anchor_completes_with_the_reference_lines_it_stores(self, name: str):
        completed, faults = _completed(name, _anchor(name))
        assert completed is not None, faults
        stored = ReportPlan.model_validate(_anchor(name))
        charts = [*stored.summary, *(section.chart for section in stored.sections)]
        again = [*completed.summary, *(section.chart for section in completed.sections)]
        assert [chart.reference for chart in again] == [chart.reference for chart in charts]

    def test_a_margin_line_sits_at_the_asked_aspects_margin(self):
        completed, faults = _completed("r5", _anchor("r5"))
        assert completed is not None, faults
        assert _key_at(completed, "sections[0].chart") == "aspect.margin"

    def test_a_floor_sits_at_the_one_criterion_reading_the_charts_measure(self):
        completed, faults = _completed("r2", _line("r2", "summary[0]", "floor"))
        assert completed is not None, faults
        assert _key_at(completed, "summary[0]") == "criteria.pass_rate.threshold"

    def test_a_floor_on_a_task_reads_the_task_criterion_where_the_chart_keeps_that_task(self):
        completed, faults = _completed("r1", _line("r1", "sections[1].chart", "floor"))
        assert completed is not None, faults
        assert _get(_anchor("r1"), "sections[1].chart.query.strata") == ["specul"]
        assert _key_at(completed, "sections[1].chart") == "criteria.specul.declines.threshold"

    def test_a_task_criterion_reads_nothing_on_a_chart_keeping_every_task(self):
        plan = _line("r1", "sections[1].chart", "floor", strata=[])
        assert "no criterion reads 'declines'" in _line_fault("r1", plan, "sections[1].chart")

    def test_a_budget_on_range_bars_sits_at_the_p95_criterion(self):
        plan = _line("r1", "sections[2].chart", "budget")
        _set(plan, "sections[2].chart.kind", "range_bars")
        _set(plan, "sections[2].chart.query.small_multiples_by", "none")
        completed, faults = _completed("r1", plan)
        assert completed is not None, faults
        assert _key_at(completed, "sections[2].chart") == "criteria.p95.threshold"

    def test_a_budget_on_bars_of_the_mean_reads_no_p95_criterion(self):
        why = _line_fault("r1", _line("r1", "sections[2].chart", "budget"), "sections[2].chart")
        assert "no criterion reads 'latency_s'" in why and "['mean']" in why

    def test_a_floor_never_sits_at_a_budget(self):
        plan = _line("r1", "sections[2].chart", "floor")
        _set(plan, "sections[2].chart.kind", "range_bars")
        _set(plan, "sections[2].chart.query.small_multiples_by", "none")
        assert "higher is better" in _line_fault("r1", plan, "sections[2].chart")

    def test_a_floor_on_a_chart_of_cells_sits_at_the_each_cell_criterion(self):
        completed, faults = _completed("r3", _line("r3", "sections[1].chart", "floor"))
        assert completed is not None, faults
        assert _key_at(completed, "sections[1].chart") == "criteria.cell_floor.threshold"

    def test_two_criteria_reading_the_charts_measure_are_refused_naming_both(self):
        data = _input_dict("r2")
        data["criteria"].append({**data["criteria"][0], "id": "pass_rate.again", "threshold": 95.0})
        why = _line_fault("r2", _line("r2", "summary[0]", "floor"), "summary[0]", data)
        assert "['pass_rate', 'pass_rate.again']" in why

    def test_a_chart_reading_several_measures_takes_no_floor(self):
        plan = _line("r2", "summary[0]", "floor", measures=["pass_rate", "outcome"])
        assert "the measures ['pass_rate', 'outcome']" in _line_fault("r2", plan, "summary[0]")

    def test_a_chart_of_readings_reads_no_measure_for_a_floor(self):
        plan = _anchor("r5")
        _set(plan, "sections[0].chart.reference", [{"kind": "floor", "key": "placeholder"}])
        assert "reads no measure" in _line_fault("r5", plan, "sections[0].chart")

    def test_a_no_preference_line_is_refused(self):
        plan = _line("r2", "summary[0]", "no_preference")
        assert "a margin, floor or budget line is what it can draw" in _line_fault("r2", plan, "summary[0]")


def _level_at(name: str, plan: dict[str, Any], path: str, data: dict[str, Any] | None = None) -> str:
    completed, faults = _completed(name, plan, data)
    assert completed is not None, faults
    return str(_get(completed.model_dump(mode="json", by_alias=True), f"{path}.level"))


class TestChartLevel:
    def test_diverging_strata_with_standard_errors_and_no_interval_are_drawn_at_the_descriptive_level(self):
        plan = _anchor("r5")
        assert (_get(plan, "sections[1].chart.kind"), _get(plan, "sections[1].chart.query.source")) == (
            "diverging_bars",
            "strata",
        )
        assert _level_at("r5", plan, "sections[1].chart") == "95% uncorrected"

    def test_bars_over_the_same_strata_draw_no_level(self):
        plan = _set(_anchor("r5"), "sections[1].chart.kind", "bars")
        assert _level_at("r5", plan, "sections[1].chart") == ""

    def test_strata_figures_carrying_an_interval_are_not_drawn_at_the_descriptive_level(self):
        data = _input_dict()
        for stratum in data["strata"]:
            for figures in stratum["per_arm"].values():
                for stat in figures.values():
                    stat["ci"] = [0.0, 1.0]
        assert _level_at("r5", _anchor("r5"), "sections[1].chart", data) == ""


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
    *(f"method[{index}].{field}" for index in range(8) for field in ("term", "text")),
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
    ("_PlannedSection", "eyebrow"),
    ("_PlannedSection", "headline"),
    ("_PlannedSection", "takeaway"),
    ("_PlannedSection", "caption"),
    ("PlannedChart", "title"),
    ("Explore", "title"),
    ("Explore", "recomputes"),
    ("ExploreControl", "title"),
    ("MethodItem", "term"),
    ("MethodItem", "text"),
    # The planner's own models of the same prose, which the module keeps private and the walk reaches through the
    # models holding them: the flag's words, an answer by its warning's position, and a prose call's text, which
    # assembles into one of the fields above.
    ("_PlannedFlag", "title"),
    ("_PlannedFlag", "text"),
    ("_PlannedAnswer", "text"),
    ("_ProseItem", "text"),
}


class TestD3:
    def test_the_prose_fields_are_the_documented_set(self):
        """D3 walks every field typed prose; this holds that set to the one the module documents. The models walked
        are every one the module exports and every one their fields hold, the stored plan's completions of the
        planner's parts aside (their prose is the parts')."""
        stored = (report_plan.ReportPlan, report_plan.ChartSpec, report_plan.Section)
        models: list[type[BaseModel]] = []
        unread = [
            getattr(report_plan, name) for name in report_plan.__all__ if isinstance(getattr(report_plan, name), type)
        ]
        while unread:
            model = unread.pop()
            if model in models or model in stored:
                continue
            models.append(model)
            for field in model.model_fields.values():
                unread += [
                    arg
                    for arg in (field.annotation, *get_args(field.annotation))
                    if isinstance(arg, type) and issubclass(arg, BaseModel)
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
