"""The one caveat list is raised from what a report input and its bundle hold, and the first screen is read off it.

S5a-2's contracts, each held here through the front door (``report_input`` over a campaign bundle, or a stored input
read back), never on entries built by hand:

- **The list returns exactly what the page's reads returned before it.** For every report input a fixture holds,
  stored or built (``caveat_pins``), the stated caveats (:func:`~threetears.evals.analysis.caveats.from_legacy`), the
  implied ones, which stated caveats code proved immaterial and the options code's read calls are what the tree before
  the list computed (``fixtures/caveat_list/before.json``); and the page's thin reads of the list say the same.
- **Each of the sixteen reasons raises one caveat** with its class, reach, state and reason, from a reading's state
  on a stored input.
- **The bundle's facts raise theirs**: an arm read per answering model is withheld, split on the served model; a run
  that did not finish gates the campaign; request settings not confirmed are class B.
- **The information architecture's three campaigns (section 3.3)** give the slot contents it states, from real
  campaigns' bundles with the scenario's facts set where the assembly records them.
- **Every reason has its remedy**, the act that addresses it, and every caveat a typed step
  (:func:`~threetears.evals.analysis.caveats.next_step`), ``no_action`` with why where nothing closes it: each call it
  makes is one its catalogue action accepts, completed with the catalogue's own example for what it leaves to the
  taker; a step that is more evidence is a declared look and states a planned size read from the figure it would
  settle (a design that clears four times in five at today's estimate, held by a seeded simulation), or why none can
  be read; a step that spends names the action that prices it.
- **Ids are unique by construction**: three coordinates split on one reading raise three caveats, and two raisers
  that raise different caveats under one id are refused.
- **The numbers side imports nothing from the page**: the answer read, a criterion's status, the cost rule and the
  list.
"""

from __future__ import annotations

import ast
import json
import math
import random
from collections.abc import Iterator
from statistics import NormalDist
from pathlib import Path
from typing import Any, get_args

import pytest

from packages.evals.tests.caveat_pins import CASE_DIRS, PINS, read_fixture, built_inputs, caveat_reads, stored_inputs
from threetears.evals.actions.engine import eval_catalogue
from threetears.evals.analysis import answer, caveats, costs, criteria, per_input
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.caveats import (
    CLASS_OF_KIND,
    REMEDY_OF_REASON,
    STEP_OF_REMEDY,
    Plan,
    call_read,
    caveat_list,
    caveat_step,
    from_legacy,
    gate_step,
    implied,
    next_step,
    plan_of,
    raise_caveats,
    remedy,
    screen,
)
from threetears.evals.analysis.checks import Check, Looks
from threetears.evals.analysis.report.call import caveat_materiality, implied_caveats, report_caveats
from threetears.evals.analysis.report_input import (
    CaveatEntry,
    Gate,
    LateCriterion,
    Remedy,
    RemedyStep,
    ReportInput,
    report_input_digest,
)
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.kernel.declaration import CampaignCall
from threetears.evals.schema.reading_states import REASONS_OF, ArmIdentity, StateOf

BEFORE: dict[str, dict[str, Any]] = json.loads(PINS.read_text(encoding="utf-8"))

#: Every input a fixture holds, read once per process.
_INPUTS: dict[str, ReportInput] = {}


def _inputs() -> dict[str, ReportInput]:
    if not _INPUTS:
        _INPUTS.update(stored_inputs())
        _INPUTS.update(built_inputs())
    return _INPUTS


def _case_bundle(name: str) -> CampaignBundle:
    """A real campaign's bundle, frozen in its writer case."""
    return CampaignBundle.from_dict(read_fixture(CASE_DIRS[1] / f"{name}.json.gz")["snapshot"]["bundle"])


# --- the list returns what the reads returned -------------------------------------------------------------------------


def test_every_fixture_input_is_pinned() -> None:
    assert sorted(_inputs()) == sorted(BEFORE), "an input was added or lost: pin it on the tree before the list"
    assert len(BEFORE) >= 40


@pytest.mark.parametrize("name", sorted(BEFORE))
def test_the_list_returns_exactly_what_the_page_s_reads_returned_before_it(name: str) -> None:
    held = _inputs()[name]
    before = BEFORE[name]

    assert [[entry.id, entry.kind, entry.reach] for entry in from_legacy(held)] == before["stated"]
    assert caveat_reads(held) == before, "a thin read of the list moved"
    if held.caveat_list is not None:
        listed = [entry.id for entry in held.caveat_list]
        assert listed[: len(before["stated"])] == [caveat_id for caveat_id, _kind, _reach in before["stated"]]


def test_the_thin_reads_are_the_list_s_stated_and_implied_entries_read_off_the_input_as_it_stands() -> None:
    """A built input carries its list, and the page's reads say the list's stated and implied entries; a copy of the
    input with a part changed reads its own, never the list the original was built with."""
    held = _inputs()["built:toyhost"]
    assert held.caveat_list is not None
    stated = from_legacy(held)
    read = {entry.id for entry in [*stated, *implied(held, stated)]}
    listed = [(entry.id, entry.reach) for entry in held.caveat_list if entry.id in read]

    assert [(caveat.id, caveat.reach) for caveat in [*report_caveats(held), *implied_caveats(held)]] == listed
    unjudged = held.model_copy(update={"judged": {}, "caveats": held.caveats.model_copy(update={"judge_tiers": {}})})
    assert "judge_uncalibrated" in {caveat.id for caveat in report_caveats(held)}
    assert "judge_uncalibrated" not in {caveat.id for caveat in report_caveats(unjudged)}
    assert set(caveat_materiality(held)) == {caveat.id for caveat in report_caveats(held)}


# --- each reason raises one caveat ------------------------------------------------------------------------------------


def _every_state() -> Iterator[tuple[str, str, str | None]]:
    for state, reasons in REASONS_OF.items():
        for reason in sorted(reasons):
            coordinate = "served_model" if reason in ("split", "identity_unrecorded") else None
            yield state, reason, coordinate


@pytest.mark.parametrize(("state", "reason", "coordinate"), list(_every_state()))
def test_a_reading_in_each_state_raises_one_caveat_with_its_class_reach_state_and_reason(
    state: str, reason: str, coordinate: str | None
) -> None:
    stored = _inputs()["stored:report_inputs/r4/input.json"]
    contrast = stored.contrasts[0]
    held = stored.model_copy(
        update={
            "contrasts": [
                contrast.model_copy(update={"state": StateOf(state=state, reason=reason, coordinate=coordinate)}),
                *stored.contrasts[1:],
            ]
        }
    )

    typed = [entry for entry in raise_caveats(held) if entry.state is not None]

    reading = f"{contrast.aspect or stored.aspects[0].id}/{contrast.measure}"
    assert [(entry.id, entry.state, entry.reason, entry.coordinate) for entry in typed] == [
        (f"{state}:{reading}", state, reason, coordinate)
    ]
    (entry,) = typed
    assert entry.cls == CLASS_OF_KIND[state]
    assert entry.reach == ("scoped" if state == "not_available" else "report")
    assert entry.arms == [contrast.arm] and entry.readings == [reading]
    assert caveat_step(entry).remedy == REMEDY_OF_REASON[reason] != "no_action", "every reason has a remedy"


def test_a_check_and_looks_on_the_bundle_raise_their_caveats_through_the_builder() -> None:
    bundle = _messy(unfinished=False)
    built = report_input(bundle)
    by_kind = {entry.kind: entry for entry in built.caveat_list or []}

    check = by_kind["check"]
    assert (check.cls, check.against, check.check, check.reach) == ("C", "model", "met", "report")
    assert by_kind["looks"].cls == "D"
    assert built.judged[_JUDGE].checks and built.aspects[0].looks == Looks(taken=3)


# --- the three campaigns of the information architecture -------------------------------------------------------------

#: The real campaigns the scenarios are read over.
_CLEAN, _KAIRO, _MESSY = "s4", "discodon-kairo-models-pilot-01a18f50", "discodon-planted-effect-01a18b10"
#: The host measure whose definition changed unrecorded, and the messy campaign's judged dimension and split arm.
_HOST_MEASURE, _JUDGE, _SPLIT = "arrival_greeting_never", "persona.holds_character", "sabotaged"


def _key_of(bundle: CampaignBundle, label: str) -> str:
    return next(
        key
        for key in sorted({cell.variant_key for a in bundle.aspects for cell in a.cells})
        if bundle.label_of(key) == label
    )


def test_a_clean_campaign_called_by_a_person_shows_no_gate_no_reason_and_the_call() -> None:
    bundle = _case_bundle(_CLEAN)
    called = "production"
    call = CampaignCall(
        arm=called,
        variant_key=_key_of(bundle, called),
        called_by="owner",
        made_by="person",
        run_ids=bundle.run_ids,
        evidence_fingerprint=bundle.fingerprint(),
    )

    value = report_input(bundle, call=call_read(call, bundle)).first_screen

    assert value is not None
    assert value.gate is None and value.reasons == [] and value.remainder == 0 and value.open_report_caveats == 0
    assert value.call is not None and value.call.arm == called and value.call.made_by == "person"
    assert value.call.runs_joined_since == 0 and value.call.identity_changed is False
    assert value.steps == []


def _with_identity(bundle: CampaignBundle, identity: ArmIdentity) -> CampaignBundle:
    """The bundle with every design arm read as ``identity``: what an assembly records of runs that did not record a
    host measure's definition."""
    aspects = [
        aspect.model_copy(
            update={
                "design": aspect.design.model_copy(
                    update={
                        "contrasts": [arm.model_copy(update={"identity": identity}) for arm in aspect.design.contrasts]
                    }
                )
            }
        )
        for aspect in bundle.aspects
    ]
    return bundle.model_copy(update={"aspects": aspects})


def test_today_s_campaign_opens_on_the_withheld_host_measure_and_says_a_model_called_it_on_moved_evidence() -> None:
    bundle = _with_identity(_case_bundle(_KAIRO), ArmIdentity(measure_definition={_HOST_MEASURE: "unrecorded"}))
    called = "1017"
    call = CampaignCall(
        arm=called,
        variant_key=_key_of(bundle, called),
        called_by="analyst",
        made_by="model",
        model="analyst-1",
        run_ids=bundle.run_ids,
        evidence_fingerprint="v4:" + "0" * 64,
    )

    built = report_input(bundle, call=call_read(call, bundle))
    value = built.first_screen
    by_id = {entry.id: entry for entry in built.caveat_list or []}

    assert value is not None and value.gate is None, "every run finished"
    first = by_id[value.reasons[0]]
    assert (first.cls, first.state, first.reason, first.coordinate) == (
        "A",
        "withheld",
        "identity_unrecorded",
        "measure_definition",
    )
    assert first.readings[0].endswith(f"/{_HOST_MEASURE}")
    assert len(value.reasons) == 3 and value.open_report_caveats > 0, "not high: open report-wide caveats stand"
    step = value.steps[0]
    assert (step.remedy, step.closes, step.action) == ("remeasure_or_attest", first.id, "run_remeasure")
    assert step.calls == [{"store": True}] and step.needs == ["run_id"], "stored, on the runs the taker names"
    assert value.call is not None and value.call.made_by == "model" and value.call.model == "analyst-1"
    assert value.call.identity_changed is True and value.call.runs_joined_since == 0


def _messy(*, unfinished: bool) -> CampaignBundle:
    """The planted-effect campaign as the messy one: an arm split by answering model, its judge checked against a
    model only, three looks none planned, criteria declared before the results, and, while ``unfinished``, one run
    cancelled."""
    bundle = _case_bundle(_MESSY)
    split = _key_of(bundle, _SPLIT)
    (aspect,) = bundle.aspects
    run = aspect.run_summaries[0].run_id
    check = Check(instrument=_JUDGE, criterion="kappa", against="model", against_id="reference-1", result="met")
    held = aspect.model_copy(
        update={
            "cells": [
                cell.model_copy(update={"rig_class_id": f"{cell.apparatus_class_id}:{index}"})
                if cell.variant_key == split
                else cell
                for index, cell in enumerate(aspect.cells)
            ],
            "judged_measures": [
                measure.model_copy(update={"checks": [check]}) if measure.name == _JUDGE else measure
                for measure in aspect.judged_measures
            ],
            "reading_scope": aspect.reading_scope.model_copy(update={"looks": Looks(taken=3), "late_criteria": []}),
            "incomplete_runs": {run: "cancelled"} if unfinished else {},
            "run_summaries": [
                summary.model_copy(update={"status": "cancelled", "ended_because": "stopped by hand"})
                if unfinished and summary.run_id == run
                else summary
                for summary in aspect.run_summaries
            ],
        }
    )
    return bundle.model_copy(update={"aspects": [held]})


def test_the_messy_campaign_s_unfinished_run_gates_it_and_counts_three_waiting() -> None:
    bundle = _messy(unfinished=True)
    value = report_input(bundle).first_screen

    assert value is not None and value.gate is not None
    assert value.gate.rule == "unfinished" and value.gate.waiting == 3
    (run,) = value.gate.runs
    assert (run.status, run.ended_because) == ("cancelled", "stopped by hand")
    assert value.reasons == [] and value.remainder == 0, "an answer that does not exist has no reasons"
    assert [(step.remedy, step.action, step.calls, step.needs) for step in value.steps] == [
        ("resume_or_rerun", "job_resume", [{"job_id": f"run:{run.run}"}], [])
    ]
    assert value.steps[0].priced_by == "jobs_resumable", "a resume is priced by what the run spent against its cap"


def test_the_messy_campaign_resumed_shows_exactly_three_reasons_in_class_order_with_their_steps() -> None:
    built = report_input(_messy(unfinished=False))
    value = built.first_screen
    by_id = {entry.id: entry for entry in built.caveat_list or []}

    assert value is not None and value.gate is None
    assert [by_id[reason].cls for reason in value.reasons] == ["A", "C", "D"]
    split = by_id[value.reasons[0]]
    assert (split.state, split.reason, split.coordinate, split.arms) == ("withheld", "split", "served_model", [_SPLIT])
    assert by_id[value.reasons[1]].against == "model"
    assert value.remainder == 0 and value.open_report_caveats == 3
    assert [step.remedy for step in value.steps] == ["rerun_on_one_cell", "check_against_people", "declare_looks"]
    assert value.steps[0].arms == [_SPLIT] and value.steps[0].calls == [{}]
    assert value.steps[0].needs == ["template_id", "subject_id", "models"]


def test_the_list_and_the_first_screen_are_outside_the_input_s_digest() -> None:
    """Both are derived from the input and its bundle, so how caveats are raised or ranked re-keys no snapshot."""
    held = _inputs()["built:toyhost"]
    bare = held.model_copy(update={"caveat_list": None, "first_screen": None})

    assert held.caveat_list and held.first_screen is not None
    assert report_input_digest(held) == report_input_digest(bare)


def test_the_campaign_bundle_holds_the_list_and_the_first_screen_its_input_computes() -> None:
    """The code page reads the bundle and the written page the input: both hold one value."""
    from packages.evals.tests.test_hot_path_oracle import subject
    from threetears.evals.analysis.service import assemble_campaign_bundle

    read = subject("toyhost")
    bundle = assemble_campaign_bundle(read.host, read.campaign_id, read.scope_id)
    built = report_input(bundle)

    assert bundle.caveat_list == built.caveat_list and bundle.first_screen == built.first_screen
    assert "caveat_list" not in bundle.frame() and "first_screen" not in bundle.frame()


# --- every reason has its remedy --------------------------------------------------------------------------------------

#: The engine's catalogue, by action name: what every step's calls are checked against.
_CATALOGUE = {action.name: action for action in eval_catalogue().actions}


def test_every_reason_has_exactly_one_remedy_and_every_remedy_a_rule() -> None:
    reasons = set().union(*REASONS_OF.values())
    assert len(reasons) == 16
    assert set(REMEDY_OF_REASON) == reasons, "a reason with no remedy, or a remedy for no reason"
    assert set(STEP_OF_REMEDY) == set(get_args(Remedy))
    assert "no_action" not in REMEDY_OF_REASON.values(), "every reason has an act that addresses it"


@pytest.mark.parametrize(
    ("reason", "addressed_by"),
    [
        ("instrument_below_bar", "replace_instrument"),
        ("people_disagree", "sharpen_criterion"),
        ("no_variation", "add_cases"),
    ],
)
def test_a_remedy_is_what_addresses_its_reason_never_a_second_read_of_the_same_evidence(
    reason: str, addressed_by: Remedy
) -> None:
    """A judge below its bar was checked and missed: a second check against people is one more look at the same
    instrument. Raters who disagree say the criterion is unclear, which a new judge reads as unclearly. Every case
    reading alike is no proof none differs, and only more cases narrow that."""
    assert REMEDY_OF_REASON[reason] == addressed_by
    assert REMEDY_OF_REASON["instrument_below_bar"] != "check_against_people"


def _entries_of_every_kind() -> Iterator[CaveatEntry]:
    """One caveat of each kind the class table places, each reason of a state kind standing for that kind."""
    for kind, cls in CLASS_OF_KIND.items():
        if kind in REASONS_OF:
            for reason in sorted(REASONS_OF[kind]):
                coordinate = "rig" if reason in ("split", "identity_unrecorded") else None
                yield CaveatEntry.model_validate(
                    {
                        "id": f"{kind}:{reason}",
                        "kind": kind,
                        "class": cls,
                        "reach": "report",
                        "state": kind,
                        "reason": reason,
                        "coordinate": coordinate,
                    }
                )
        else:
            yield CaveatEntry.model_validate({"id": kind, "kind": kind, "class": cls, "reach": "report"})


def test_every_kind_the_first_screen_can_show_has_an_act_and_every_remedy_closes_something() -> None:
    """The closed set of first-screen reasons is every classed kind: each has an act. A method-only kind and an
    analyst's own kind take ``no_action``, saying which; nothing else does."""
    every = list(_entries_of_every_kind())
    for entry in every:
        if entry.cls is not None:
            assert remedy(entry) != "no_action", entry.id
    unreported = next(entry for entry in every if entry.kind == "unreported")
    own = CaveatEntry.model_validate({"id": "analyst:mine", "kind": "analyst_own", "reach": "report"})
    assert (caveat_step(unreported).remedy, caveat_step(unreported).no_action) == ("no_action", "method_only")
    assert (caveat_step(own).remedy, caveat_step(own).no_action) == ("no_action", "unclassed_kind")
    assert {remedy(entry) for entry in [*every, own]} == set(get_args(Remedy)), "a remedy no caveat reaches"


@pytest.mark.parametrize("name", sorted(BEFORE))
def test_every_reason_a_first_screen_shows_has_its_step_in_its_order(name: str) -> None:
    held = _inputs()[name]
    value = screen(held, caveat_list(held))

    if value.gate is None:
        assert [step.closes for step in value.steps] == value.reasons
    for step in value.steps:
        assert (step.remedy == "no_action") == (step.no_action is not None)


def _steps_of(remedied: Remedy) -> Iterator[RemedyStep]:
    for runs in ((), ("run-1", "run-2")):
        for coordinate in (None, "served_model", "measure_definition"):
            yield next_step(remedied, arms=["a"], runs=runs, coordinate=coordinate, plan=Plan(3))


@pytest.mark.parametrize("remedied", get_args(Remedy))
def test_every_call_a_step_makes_is_one_its_catalogue_action_accepts(remedied: Remedy) -> None:
    """Each call names only the action's own parameters, typed as it takes them; with what the step leaves to the
    taker filled from the catalogue's own example, the action's parameters accept it. A step that spends is priced by
    an action that spends nothing, and one that takes the launch's own arguments accepts the call as well."""
    for step in _steps_of(remedied):
        if step.action is None:
            assert step.calls == [] and step.needs == []
            continue
        action = _CATALOGUE[step.action]
        fields = action.params.model_fields
        required = {name for name, field in fields.items() if field.is_required()}
        assert step.calls and set(step.needs) <= set(fields), sorted(set(step.needs) - set(fields))
        filled = {name: action.example[name] for name in step.needs if name in action.example}
        assert {name for name in step.needs if name not in filled} <= set(fields) - required
        for call in step.calls:
            assert set(call) <= set(fields), sorted(set(call) - set(fields))
            assert not set(call) & set(step.needs)
            assert required <= set(call) | set(step.needs), sorted(required - set(call) - set(step.needs))
            action.params.model_validate({**filled, **call})
        if step.spends:
            assert step.priced_by is not None
            pricer = _CATALOGUE[step.priced_by]
            assert pricer.permission == "read"
            if issubclass(pricer.params, action.params):
                for call in step.calls:
                    pricer.params.model_validate({**filled, **call})


def test_a_resume_is_one_call_per_run_and_a_remeasure_stores_each_run_read_again() -> None:
    resume = next_step("resume_or_rerun", runs=["r1", "r2"])
    assert (resume.action, resume.calls, resume.needs) == (
        "job_resume",
        [{"job_id": "run:r1"}, {"job_id": "run:r2"}],
        [],
    )
    remeasure = next_step("remeasure_or_attest", runs=["r1"], coordinate="measure_definition")
    assert (remeasure.action, remeasure.calls) == ("run_remeasure", [{"run_id": "r1", "store": True}])
    unmeasured = next_step("remeasure_or_attest", coordinate="served_model")
    assert (unmeasured.action, unmeasured.calls) == (None, []), "no action re-measures a served model"


def test_no_step_reads_until_it_separates_and_none_is_taken_by_code() -> None:
    """A step's size is fixed before it runs: it holds no field that waits on a result. And nothing in the package
    takes one: the step's shape is read only by the modules that compute it."""
    assert set(RemedyStep.model_fields) == {
        "remedy",
        "closes",
        "action",
        "calls",
        "needs",
        "arms",
        "runs",
        "planned_cases",
        "unsized",
        "declared_look",
        "results_needed",
        "recording",
        "spends",
        "priced_by",
        "no_action",
    }
    with pytest.raises(ValueError, match="planned size"):
        RemedyStep(remedy="add_cases", declared_look=True)
    with pytest.raises(ValueError, match="planned size"):
        RemedyStep(remedy="add_cases", declared_look=True, planned_cases=3, unsized="no_count")
    with pytest.raises(ValueError, match="prices it"):
        RemedyStep(remedy="add_cases", spends=True)
    with pytest.raises(ValueError, match="says why"):
        RemedyStep(remedy="no_action")
    source = Path(caveats.__file__).resolve().parents[1]
    readers = sorted(
        str(path.relative_to(source))
        for path in source.rglob("*.py")
        if "RemedyStep" in path.read_text(encoding="utf-8")
    )
    assert readers == ["analysis/__init__.py", "analysis/caveats.py", "analysis/report_input.py"]


def test_a_caveat_with_no_reason_takes_its_kind_s_remedy() -> None:
    for kind, remedied in (
        ("looks", "declare_looks"),
        ("judge_uncalibrated", "check_against_people"),
        ("late", "confirm_on_fresh_cases"),
        ("no_control", "name_control"),
    ):
        entry = CaveatEntry.model_validate({"id": kind, "kind": kind, "class": CLASS_OF_KIND[kind], "reach": "report"})
        assert caveat_step(entry).remedy == remedied


# --- a step that is more evidence is planned and a declared look -----------------------------------------------------


@pytest.mark.parametrize("remedied", ["add_cases", "confirm_on_fresh_cases"])
def test_a_step_that_adds_cases_is_a_declared_look_with_its_planned_size_or_why_it_has_none(remedied: Remedy) -> None:
    """IA section 9.2 D's guard, held by name: a step that adds cases states a planned size and counts as a declared
    look, and one that can read no size says why rather than run open-ended."""
    planned = next_step(remedied, arms=["a"], plan=Plan(7))
    assert (planned.declared_look, planned.planned_cases, planned.unsized) == (True, 7, None)
    assert planned.spends and planned.priced_by == "launch_estimate"
    unsized = next_step(remedied, arms=["a"])
    assert (unsized.declared_look, unsized.planned_cases, unsized.unsized) == (True, None, "nothing_to_settle")


def test_too_few_cases_gate_is_lifted_by_a_declared_look_sized_to_the_floor_from_the_fewest_an_arm_holds() -> None:
    step = gate_step(Gate(rule="too_small", fewest_cases=3))
    assert (step.remedy, step.declared_look, step.planned_cases) == ("add_cases", True, 2)
    unknown = gate_step(Gate(rule="too_small"))
    assert (unknown.declared_look, unknown.planned_cases, unknown.unsized) == (True, None, "no_count")


@pytest.mark.parametrize("remedied", get_args(Remedy))
def test_only_a_step_that_adds_cases_is_a_look_and_one_that_spends_is_priced(remedied: Remedy) -> None:
    for step in _steps_of(remedied):
        assert step.declared_look == (remedied in ("add_cases", "confirm_on_fresh_cases"))
        assert step.declared_look == (step.planned_cases is not None)
        assert (step.priced_by is not None) == step.spends


@pytest.mark.parametrize("name", sorted(BEFORE))
def test_every_not_separated_reading_s_step_is_a_declared_look_with_a_size_or_why(name: str) -> None:
    """Every precision caveat (class E) any fixture raises is closed by more evidence: a declared look whose size is
    read from the input or says why none is."""
    held = _inputs()[name]
    for caveat in caveat_list(held):
        if caveat.cls != "E":
            continue
        step = caveat_step(caveat, plan=plan_of(held, caveat))
        assert step.action == "run_launch" and step.declared_look, caveat.id
        assert (step.planned_cases is None) != (step.unsized is None), caveat.id


def test_a_first_screen_showing_a_precision_reason_offers_a_declared_look() -> None:
    """On a stored input's own first screen, read through the list, the class E reason shown is closed by a declared
    look: a gate cleared on a 95th percentile reads no interval, so the step says so instead of a size."""
    stored = _inputs()["stored:report_inputs/r2/input.json"]
    value = screen(stored, caveat_list(stored))
    by_id = {caveat.id: caveat for caveat in caveat_list(stored)}

    shown = [step for step in value.steps if by_id[step.closes or ""].cls == "E"]
    assert shown, "the fixture's first screen shows a precision reason"
    for step in shown:
        assert (step.remedy, step.declared_look) == ("add_cases", True)
        assert (step.planned_cases, step.unsized) == (None, "no_interval")


def test_a_gate_cleared_exactly_on_its_line_reads_no_size() -> None:
    held = _inputs()["stored:report_inputs/r1/input.json"]
    borderline = next(caveat for caveat in caveat_list(held) if caveat.kind == "borderline")
    assert plan_of(held, borderline) == Plan(unsized="on_the_line"), "an arm reads 90 against a gate at 90"


def test_a_figure_with_too_few_cases_is_sized_to_the_floor_and_one_where_every_case_read_alike_is_not() -> None:
    stored = _inputs()["stored:report_inputs/r4/input.json"]
    for reason, cases, plan in (
        ("too_few", 3, Plan(2)),
        ("too_few", None, Plan(unsized="no_count")),
        ("no_variation", 20, Plan(unsized="no_spread")),
    ):
        entry = CaveatEntry.model_validate(
            {
                "id": f"not_available:{reason}",
                "kind": "not_available",
                "reach": "scoped",
                "state": "not_available",
                "reason": reason,
                "cases": cases,
            }
        )
        assert plan_of(stored, entry) == plan, reason


#: The seed, designs and tolerance of the planned size's simulation.
_SEED, _DESIGNS, _TOLERANCE = 20261115, 4000, 0.025


def _unsettled_input(change: float, spread: float, cases: int, margin: float, level: float) -> ReportInput:
    """The reasoning-effort campaign with both candidates' readings on one aspect set to ``change``, its interval at
    ``level`` read from a per-case spread of ``spread`` over ``cases`` paired cases, against ``margin``."""
    held = _inputs()["case:discodon-1017-reasoning-effort-01a18d4d"].model_copy(update={"caveat_list": None})
    half = NormalDist().inv_cdf(0.5 + level / 2) * spread / math.sqrt(cases)
    interval = (change - half, change + half)
    aspects = [
        aspect.model_copy(
            update={
                "margin": margin,
                "readings": [
                    reading.model_copy(
                        update={
                            "change": change,
                            "within_interval": interval,
                            "gain_interval": interval,
                            "gain_confidence": level,
                            "margin": margin,
                            "cases": [cases, cases],
                        }
                    )
                    for reading in aspect.readings
                ],
            }
        )
        if aspect.id == "queue discipline"
        else aspect
        for aspect in held.aspects
    ]
    return held.model_copy(update={"aspects": aspects})


def _clearing_share(
    rng: random.Random, change: float, spread: float, line: float, level: float, **design: int
) -> float:
    """How often a design's one-sided interval at ``level`` sits wholly above ``line`` when the truth is ``change``:
    ``fresh`` cases drawn anew, after ``held`` cases that read exactly ``change`` where given."""
    z = NormalDist().inv_cdf(level)
    held, fresh = design.get("held", 0), design["fresh"]
    cleared = 0
    for _ in range(_DESIGNS):
        drawn = [rng.gauss(change, spread) for _ in range(fresh)]
        total = held + fresh
        drawn_mean = sum(drawn) / fresh
        variance = sum((value - drawn_mean) ** 2 for value in drawn) / (fresh - 1)
        mean = (held * change + drawn_mean * fresh) / total
        se = math.sqrt(variance / total)
        cleared += mean - z * se > line
    return cleared / _DESIGNS


def test_the_planned_size_clears_the_margin_four_times_in_five_at_today_s_estimate() -> None:
    """Seeded simulation of the size an unsettled reading is planned at. Truth equal to today's estimate (a 4-point
    drop, per-case spread 40 points, 80 paired cases, a 10-point margin), 4,000 designs: a fresh design of today's
    cases plus the planned ones clears the margin's line in 80% of designs (held within 2.5 points), adding the planned
    cases to the ones it rests on clears in at least that share, and today's size clears in under half. The size is
    read from the reading's own interval, so a reading resting on more cases plans more, never a fixed few."""
    change, spread, cases, margin, level = -4.0, 40.0, 80, 10.0, 0.95
    held = _unsettled_input(change, spread, cases, margin, level)
    caveat = next(entry for entry in raise_caveats(held) if entry.id == "unsettled:queue discipline")
    plan = plan_of(held, caveat)

    assert plan is not None and plan.cases is not None and plan.unsized is None
    step = screen(held, [caveat]).steps[0]
    assert (step.remedy, step.planned_cases, step.declared_look) == ("add_cases", plan.cases, True)
    rng = random.Random(_SEED)
    total = cases + plan.cases
    assert abs(_clearing_share(rng, change, spread, -margin, level, fresh=total) - 0.80) <= _TOLERANCE
    assert _clearing_share(rng, change, spread, -margin, level, held=cases, fresh=plan.cases) >= 0.80 - _TOLERANCE
    assert _clearing_share(rng, change, spread, -margin, level, fresh=cases) < 0.5
    wider = _unsettled_input(change, spread * math.sqrt(10), cases * 10, margin, level)
    wider_caveat = next(entry for entry in raise_caveats(wider) if entry.id == "unsettled:queue discipline")
    wider_plan = plan_of(wider, wider_caveat)
    assert wider_plan is not None and wider_plan.cases is not None and wider_plan.cases > 5 * plan.cases


def test_a_pick_among_many_is_confirmed_on_fresh_cases_at_a_size_read_from_the_pick() -> None:
    held = _inputs()["built:scaled"]
    caveat = next(entry for entry in caveat_list(held) if entry.kind == "best_of_n")
    plan = plan_of(held, caveat)
    step = caveat_step(caveat, plan=plan)

    assert (step.remedy, step.action, step.declared_look) == ("confirm_on_fresh_cases", "run_launch", True)
    assert plan is not None and step.planned_cases == plan.cases is not None


# --- ids are unique by construction ----------------------------------------------------------------------------------


def test_three_coordinates_split_on_one_reading_raise_three_caveats_under_three_ids() -> None:
    stored = _inputs()["stored:report_inputs/r4/input.json"]
    arm = stored.arms[-1].model_copy(
        update={"identity": ArmIdentity(served_model="split", template="split", rig="split")}
    )
    held = stored.model_copy(update={"arms": [*stored.arms[:-1], arm]})

    split = [entry for entry in raise_caveats(held) if entry.reason == "split"]
    assert sorted(entry.coordinate or "" for entry in split) == ["rig", "served_model", "template"]
    assert len({entry.id for entry in split}) == 3


def test_two_different_caveats_under_one_id_are_refused_never_one_kept() -> None:
    stored = _inputs()["stored:report_inputs/r4/input.json"]
    aspect = stored.aspects[0].id
    floors = [
        LateCriterion(criterion="floor", name="twice", set_at="2026-11-01", aspect=aspect, reading=reading)
        for reading in ("passed", "cost")
    ]
    held = stored.model_copy(
        update={"caveats": stored.caveats.model_copy(update={"late_criteria": floors}), "caveat_list": None}
    )

    with pytest.raises(ValueError, match="late:floor:twice"):
        raise_caveats(held)


# --- the numbers side imports nothing from the page -------------------------------------------------------------------


@pytest.mark.parametrize("module", [answer, criteria, costs, caveats, per_input], ids=lambda module: module.__name__)
def test_the_numbers_side_of_the_call_imports_nothing_from_the_page(module: Any) -> None:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert not [name for name in imported if name.startswith("threetears.evals.analysis.report.")]
    assert not [name for name in imported if name == "threetears.evals.analysis.report"]
