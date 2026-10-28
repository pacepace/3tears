"""The weakest margin and reader names: two facts code computes for the planner, and the plan checks that hold to them.

Contracts held here (``threetears.evals.analysis.report_margins``, ``report.figure_keys``, ``report_plan_checks``):

- **The weakest margin** is the figure nearest the line it is read against, relative to the line: on r1 to r5 it is the
  margin a reviewer names (the pick at its troubleshooting floor; the pick a fraction over the pass-rate gate; the pick's
  balanced accuracy over the overall gate, not a rejected arm's narrower pass of a gate it does not need; a rejected
  arm one guardrail short; the candidate whose worst drop sits nearest the margin). An option that does not survive
  counts only by the one gate figure keeping it out; a reading wins a tie with the floor that copies it. A figure or
  interval end that is not a finite number is no figure: it gives no margin, and an option lacking a gate's figure
  does not survive, as the gate check reads it.
- ``weakest.*`` figure keys render it, in its line's units, and refuse by name where the input reads no figure against
  a line. A decide plan's Watch cites one; a tradeoff plan's slots are not asked to.
- **Reader names**: each id's one name (declared, else the package's reader name for it), which the ``.name`` and
  ``.label`` keys print and the planner's facts carry. Above Full results, prose that types an id-shaped id differing
  from its reader name is refused, matched as D3 matches a name; a plain-word id, a percentile's name, a key and Method
  are not. Quoting ``{meta.question}`` above Full results is refused; Method may.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from packages.evals.tests.fixtures.report_plans import dataset
from threetears.evals.analysis.errors import PlanRefusal, UnresolvableReference
from threetears.evals.analysis.planner import first_plan_request
from threetears.evals.analysis.report.figure_keys import reader_name, reader_names, render, resolve
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_decision import surviving_options
from threetears.evals.analysis.report_margins import criterion_figures, margins, weakest_margin
from threetears.evals.analysis.report_plan_checks import validate_plan
from threetears.evals.analysis.report_plan_facts import plan_facts
from threetears.evals.kernel.host.measures import MeasureRegistry
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.kernel.host.sweepables import SHARED_CORE
from threetears.evals.kernel.report_plan import ReportPlan

HERE = dataset.HERE


def _input(name: str) -> ReportInput:
    return dataset.report_input_of(name)


def _anchor(name: str = "r5") -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((HERE / name / "anchor.plan.json").read_text(encoding="utf-8"))
    return loaded


def _faults(plan: dict[str, Any], name: str = "r5") -> list[tuple[str, str]]:
    try:
        validate_plan(ReportPlan.model_validate(plan), _input(name))
    except PlanRefusal as refused:
        return list(refused.faults)
    return []


def _slot(plan: dict[str, Any], key: str) -> dict[str, Any]:
    found: dict[str, Any] = next(slot for slot in plan["decision"]["slots"] if slot["key"] == key)
    return found


# --- the weakest margin --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "arm", "source", "where", "gap"),
    [
        ("r1", "atlas-m", "trouble.resolves", "trouble", 0.0),
        ("r2", "brisk-s.low.r1", "pass_rate", None, 0.14),
        ("r3", "brisk-s.few", "overall", None, 0.7),
        ("r4", "f16", "facts", None, -0.6),
        ("r5", "qwen3.8-flash", "character adherence (v2)", None, 4.21),
    ],
)
def test_the_weakest_margin_on_each_fixture(name: str, arm: str, source: str, where: str | None, gap: float) -> None:
    found = weakest_margin(_input(name))
    assert found is not None
    assert (found.arm, found.source, found.task) == (arm, source, where)
    assert found.gap == pytest.approx(gap)


def test_a_rejected_option_counts_only_by_the_one_gate_figure_keeping_it_out() -> None:
    held = margins(_input("r3"))
    nano = [margin for margin in held if margin.arm == "nano-xs.few"]
    assert [(margin.source, margin.cell, margin.gap < 0) for margin in nano] == [("cell_floor", "complex|news", True)]


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_a_gate_figure_that_is_not_finite_is_no_figure_and_gives_no_margin(value: float) -> None:
    """r2's pick at a pass rate that is not a number: no figure for the gate, so it neither survives nor holds a
    margin, as the gate check and the survivors read it."""
    report_input = _input("r2")
    weakest = weakest_margin(report_input)
    assert weakest is not None
    data = report_input.model_dump(by_alias=True, exclude_none=True)
    key = next(c["measure"] for c in data["criteria"] if c["id"] == weakest.source)
    data["measures"][key]["per_arm"][weakest.arm]["mean"] = value
    broken = ReportInput.model_validate(data)
    criterion = next(c for c in broken.criteria if c.id == weakest.source)
    assert criterion_figures(broken, criterion, weakest.arm) == []
    assert weakest.arm not in surviving_options(broken)
    held = margins(broken)
    assert weakest.arm not in {margin.arm for margin in held}
    assert all(math.isfinite(margin.gap) for margin in held)


def test_a_reading_interval_end_that_is_not_finite_gives_no_margin() -> None:
    """r5's weakest margin is a reading's worst end; read as NaN, it gives no margin, and the weakest is finite."""
    report_input = _input("r5")
    weakest = weakest_margin(report_input)
    assert weakest is not None and weakest.kind == "margin"
    data = report_input.model_dump(by_alias=True, exclude_none=True)
    aspect = next(a for a in data["aspects"] if a["id"] == weakest.source)
    reading = next(r for r in aspect["readings"] if r["arm"] == weakest.arm)
    reading["within_interval"] = [math.nan, reading["within_interval"][1]]
    for floor in data["floors"]:
        if floor["arm"] == weakest.arm:
            floor["interval"] = [math.nan, floor["interval"][1]]
    held = margins(ReportInput.model_validate(data))
    assert not [m for m in held if m.arm == weakest.arm and m.kind in ("margin", "floor")]
    assert all(math.isfinite(margin.gap) for margin in held)


def test_a_reading_wins_the_tie_with_the_floor_that_copies_it() -> None:
    found = weakest_margin(_input("r5"))
    assert found is not None and found.kind == "margin"
    assert any(margin.kind == "floor" and margin.relative == found.relative for margin in margins(_input("r5")))


def test_the_weakest_keys_render_in_their_lines_units() -> None:
    said = "{weakest.arm}: {weakest.value} against {weakest.criterion} at {weakest.threshold}, gap {weakest.margin}"
    assert render(said, _input("r5"), {}) == (
        "qwen3.8-flash (A): -10.8 against the margin on character adherence (v2) at -15, gap 4.2"
    )
    assert render(said, _input("r1"), {}) == (
        "Atlas-M (B): 90% against Resolves the issue in Troubleshooting (multi-turn) at 90%, gap 0"
    )
    assert render("{weakest.margin|pts}", _input("r4"), {}) == "-0.6"


def test_a_weakest_key_is_refused_by_name_where_nothing_is_read_against_a_line() -> None:
    bare = _input("r4").model_copy(update={"criteria": []})
    assert weakest_margin(bare) is None
    with pytest.raises(UnresolvableReference, match="reads no figure against a criterion, margin or floor"):
        resolve("weakest.margin", bare)


def test_a_decide_watch_citing_no_weakest_key_is_refused_and_a_tradeoff_plan_is_not_asked() -> None:
    plan = _anchor("r5")
    _slot(plan, "watch")["text"] = "{arm:1017} spent {reasoning_ratio.1017|pct} of its output reasoning."
    faults = _faults(plan)
    assert [path for path, _ in faults] == ["decision.slots[2].text"]
    assert "qwen3.8-flash (A), -10.8 against the margin on character adherence (v2) at -15" in faults[0][1]
    assert _faults(_anchor("r4"), "r4") == []


# --- reader names --------------------------------------------------------------------------------------------------


def test_each_id_has_one_reader_name_its_keys_print() -> None:
    r5 = _input("r5")
    names = reader_names(r5)
    assert names["stratum"]["off_lane"] == "off lane"
    assert names["judged"]["persona.holds_character"] == "persona holds character"
    assert names["aspect"] == {"character adherence (v2)": "character adherence (v2)"}
    assert resolve("judged.persona.holds_character.label", r5).value == "persona holds character"
    r1 = _input("r1")
    assert reader_name("criterion", "trouble.resolves", r1) == "Resolves the issue"
    assert reader_name("task", "trouble", r1) == "Troubleshooting (multi-turn)"
    assert reader_name("measure", "cost_per_1k", r1) == "Cost per 1k conversations"


def test_the_facts_and_the_planners_message_carry_the_weakest_margin_and_the_reader_names() -> None:
    facts = plan_facts(_input("r5"))
    assert facts.weakest is not None
    assert facts.weakest.said.startswith("qwen3.8-flash (A): -10.8 against the margin on character adherence (v2)")
    assert "weakest.margin" in facts.weakest.keys
    assert facts.names["stratum"]["off_lane"] == "off lane"
    profile = HostProfile(host_id="names", host_sweepables=SHARED_CORE, measures=MeasureRegistry(()))
    _, user, _ = first_plan_request(_input("r5"), "prompt", profile)
    assert '"weakest_margin":{"said":"qwen3.8-flash (A)' in user
    assert '"off_lane":"off lane"' in user


def test_an_id_shaped_id_above_full_results_is_refused_with_its_reader_name() -> None:
    plan = _anchor("r5")
    _slot(plan, "next")["text"] += " Off_Lane asks matter most."
    plan["sections"][1]["headline"] = "The persona.holds_character loss sits in {strata.off_lane.name}"
    assert _faults(plan) == [
        (
            "decision.slots[3].text",
            "types the stratum id 'off_lane' above Full results: write its reader name, 'off lane'",
        ),
        (
            "sections[1].headline",
            "types the judged id 'persona.holds_character' above Full results: write its reader name, "
            "'persona holds character'",
        ),
    ]


def test_a_shown_name_that_is_a_bounded_prefix_of_an_id_does_not_hide_it() -> None:
    """Names and ids match in one pass, longest first: the shown name ``Tone`` never masks the ``tone`` of the id
    ``tone.kept``, so both ids the text types are found, in the order it types them."""
    data = json.loads((HERE / "r5" / "input.json").read_text(encoding="utf-8"))
    data["arms"][1]["short_name"] = "Tone"
    data["strata"].append({**data["strata"][0], "id": "tone.kept"})
    plan = _anchor("r5")
    plan["sections"][1]["headline"] = "The tone.kept loss and off_lane."
    with pytest.raises(PlanRefusal) as refused:
        validate_plan(ReportPlan.model_validate(plan), ReportInput.model_validate(data))
    assert [why for path, why in refused.value.faults if path == "sections[1].headline"] == [
        "types the stratum id 'tone.kept' above Full results: write its reader name, 'tone kept'",
        "types the stratum id 'off_lane' above Full results: write its reader name, 'off lane'",
    ]


def _r1_with_arm(old: str, new: str, **fields: str) -> tuple[dict[str, Any], ReportInput]:
    """r1's anchor plan and input with the arm ``old`` renamed ``new`` throughout, named by its id as an input built
    from a run names it (its alias is its id), with ``fields`` set on it."""
    plan: dict[str, Any] = json.loads((HERE / "r1" / "anchor.plan.json").read_text(encoding="utf-8").replace(old, new))
    data = json.loads((HERE / "r1" / "input.json").read_text(encoding="utf-8").replace(old, new))
    arm = next(arm for arm in data["arms"] if arm["id"] == new)
    arm.update({"name": new, **fields})
    return plan, ReportInput.model_validate(data)


def _plan_faults(plan: dict[str, Any], report_input: ReportInput) -> list[tuple[str, str]]:
    try:
        validate_plan(ReportPlan.model_validate(plan), report_input)
    except PlanRefusal as refused:
        return list(refused.faults)
    return []


def test_a_lever_arms_id_the_page_prints_by_another_name_is_refused_though_a_hyphen_alone_marks_it() -> None:
    """A lever arm's id is the campaign's label (``fanout-1``): where the page prints the arm as "fanout 1", prose
    typing the id is refused, its one id mark a hyphen; its alias, which an input built from a run sets to the id,
    does not hide it."""
    plan, report_input = _r1_with_arm("brisk-s", "fanout-1", short_name="fanout 1")
    assert _plan_faults(plan, report_input) == []

    _slot(plan, "why_not")["text"] += " Most fanout-1 answers still read well."

    assert _plan_faults(plan, report_input) == [
        (
            "decision.slots[1].text",
            "types the arm id 'fanout-1' above Full results: write the name the page prints for it, 'fanout 1', or its "
            "key {arm:fanout-1}",
        )
    ]


def test_a_model_arms_id_that_is_its_served_model_is_the_readers_word() -> None:
    """A model arm's id is the model's name a reader knows: typed beside a short name the page prints, it stands."""
    model = "deepseek/deepseek-v4.1-flash"
    plan, report_input = _r1_with_arm("atlas-m", model, short_name="DeepSeek Flash", served_model=model)
    assert _plan_faults(plan, report_input) == []

    _slot(plan, "why_not")["text"] += f" The {model} replies were short."

    assert _plan_faults(plan, report_input) == []


def test_an_ordinary_hyphenated_word_and_an_arm_id_its_printed_name_differs_from_only_in_case_stand() -> None:
    """A hyphen marks only a token that is an arm's id the page prints otherwise: "cost-per-reply", "off-lane" and
    "high-effort" are words, and the id ``open-70b``, printed "Open-70B", is its own name in any case."""
    plan, report_input = _r1_with_arm("brisk-s", "fanout-1", short_name="fanout 1")

    _slot(plan, "why_not")["text"] += (
        " The cost-per-reply gap is small, off-lane asks hold, high-effort runs cost more, and open-70b is close."
    )

    assert _plan_faults(plan, report_input) == []


def test_a_plain_word_id_a_percentile_a_key_and_method_are_not_refused() -> None:
    plan = _anchor("r5")
    _slot(plan, "next")["text"] += " The multi and about strata matter less."
    plan["method"][0]["text"] += " Strata: off_lane, small_talk."
    assert _faults(plan) == []
    r1 = _anchor("r1")
    _slot(r1, "why_not")["text"] += " Billing and the p95 budget are fine."
    assert _faults(r1, "r1") == []


def test_the_quoted_question_is_refused_above_full_results_and_allowed_in_method() -> None:
    plan = _anchor("r5")
    plan["sections"][0]["takeaway"] = "The question: {meta.question}"
    assert [path for path, _ in _faults(plan)] == ["sections[0].takeaway"]
    plan = _anchor("r5")
    plan["method"][0]["text"] += " The question: {meta.question}"
    assert _faults(plan) == []


def test_every_anchor_and_current_planner_run_cites_the_weakest_margin_in_a_decide_watch() -> None:
    """The current runs only: a historical run predates the weakest margin, and is kept as it was recorded."""
    plans = [*HERE.glob("r*/anchor.plan.json"), *(dataset.current_run(name) for name in dataset.current_runs())]
    assert plans
    for path in plans:
        plan = dataset.load_plan(Path(path))
        if plan.mode != "decide":
            continue
        watch = next(slot.text for slot in plan.decision.slots if slot.key == "watch")
        assert "{weakest." in watch, path
