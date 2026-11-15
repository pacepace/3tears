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
  taker; a step that adds cases states a planned size read from the figure it would settle, against the rule and
  interval that lift its caveat (a design that lifts it at least four times in five at today's estimate, held by
  seeded simulations through the code that raises it), and launches exactly that many new cases; one with no size,
  or none that can close it (a percentile's point estimate, a criterion or a pick made on these results), says why
  and makes no call; a step that spends names the action that prices it.
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
import statistics
from collections.abc import Iterator
from pathlib import Path
from typing import Any, get_args

import pytest

from packages.evals.tests.caveat_pins import CASE_DIRS, PINS, read_fixture, built_inputs, caveat_reads, stored_inputs
from threetears.evals.actions.engine import eval_catalogue
from threetears.evals.analysis import answer, caveats, costs, criteria, per_input, stats
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.caveats import (
    CLASS_OF_KIND,
    REMEDY_OF_REASON,
    STEP_OF_REMEDY,
    ArmRuns,
    Plan,
    Setup,
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
from threetears.evals.analysis.checks import JudgeCheck, Looks
from threetears.evals.analysis.report.call import caveat_materiality, implied_caveats, report_caveats
from threetears.evals.analysis.report_input import (
    CaveatEntry,
    Gate,
    LateCriterion,
    Remedy,
    RemedyStep,
    ReportInput,
    ShortRun,
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
    check = JudgeCheck(
        kind="judge", instrument=_JUDGE, criterion="kappa", against="model", against_id="reference-1", result="met"
    )
    held = aspect.model_copy(
        update={
            "cells": [
                # What the pooling sets for an arm read per answering model: the rig split and the coordinate.
                cell.model_copy(
                    update={"rig_class_id": f"{cell.apparatus_class_id}:{index}", "split_on": ["served_model"]}
                )
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
    bundle = _messy(unfinished=False)
    built = report_input(bundle)
    value = built.first_screen
    by_id = {entry.id: entry for entry in built.caveat_list or []}

    assert value is not None and value.gate is None
    assert [by_id[reason].cls for reason in value.reasons] == ["A", "C", "D"]
    split = by_id[value.reasons[0]]
    assert (split.state, split.reason, split.coordinate, split.arms) == ("withheld", "split", "served_model", [_SPLIT])
    assert by_id[value.reasons[1]].against == "model"
    assert value.remainder == 0 and value.open_report_caveats == 3
    assert [step.remedy for step in value.steps] == ["rerun_on_one_cell", "check_against_people", "declare_looks"]
    rerun = value.steps[0]
    assert rerun.arms == [_SPLIT]
    assert (rerun.action, rerun.calls, rerun.then, rerun.no_action) == (None, [], [], "lift_not_shown"), (
        "the provider split the answering model: no launch parameter holds it to one, so no re-run is shown to lift it "
        "and the step makes no call"
    )


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
        if step.remedy == "no_action":
            assert step.no_action is not None
        if step.no_action is not None:
            assert step.action is None and step.calls == [], "a step no action takes makes no call"


#: What a launch can fill from a reading that holds its bundle: the subject, the aspect's template, and two arms'
#: runs, the control's among them, launched at five repeats per case where a launch's default is three.
_HELD = Setup(
    subject_id="subject-1",
    templates={"accuracy": "tmpl-1"},
    arms={
        "accuracy": {
            "a": ArmRuns(("run-a",), "model-a", 5, False),
            "c": ArmRuns(("run-c1", "run-c2"), "model-c", 5, False),
        }
    },
    control="c",
    aspects=("accuracy",),
)

#: Every launch parameter that sets an arm up beyond its template, subject and model: each is filled or named.
_ARM_SETUP = {
    "k_runs",
    "measure_latency",
    "arm_schedule",
    "judge_model",
    "simulator_model",
    "overlays",
    "apparatus_settings",
    "cell_timeout_s",
}


#: The held arms with one more, ``b``, that moved a lever beyond its model: on one model with the control, as a prompt
#: A/B is, so a launch takes it in a call of its own.
_MOVED = _HELD._replace(arms={"accuracy": {**_HELD.arms["accuracy"], "b": ArmRuns(("run-b",), "model-c", 5, True)}})


def _steps_of(remedied: Remedy) -> Iterator[RemedyStep]:
    for setup, arms in ((Setup(), ["a"]), (_HELD, ["a", "c"]), (_HELD, []), (_MOVED, ["b", "c"])):
        for runs in ((), ("run-1", "run-2")):
            for coordinate in (None, "served_model", "measure_definition"):
                for alike in ((), ("reviewer_pool",)):
                    for contrast in (False, True):
                        yield next_step(
                            remedied,
                            arms=arms,
                            runs=runs,
                            readings=["accuracy/passed"],
                            coordinate=coordinate,
                            plan=Plan(3),
                            setup=setup,
                            alike=alike,
                            contrast=contrast,
                        )


def _parts(step: RemedyStep) -> Iterator[RemedyStep]:
    """The step's own call and every one that follows it, in order."""
    yield step
    yield from step.then


@pytest.mark.parametrize("remedied", get_args(Remedy))
def test_every_call_a_step_makes_is_one_its_catalogue_action_accepts(remedied: Remedy) -> None:
    """Each call, the step's own and every one that follows it, names only its action's own parameters, typed as it
    takes them; with what the step leaves to the taker, to the arms' launch or to an earlier call filled from the
    catalogue's own example, the action's parameters accept it. A step that launches runs repeats the arms' set-up,
    filling or naming every setting that defines an arm, and joins its runs to a campaign, since a launched run joins
    none: a re-run then takes out what it replaces, and a confirmation creates its campaign before it launches and
    joins the runs to that one. A step that spends is priced by an action that spends nothing, which accepts each
    launch's calls where it takes the launch's own arguments."""
    for step in _steps_of(remedied):
        if step.action is None:
            assert step.calls == [] and step.needs == [] and step.then == []
            assert not step.alike and step.planned_cases is None and not step.spends
            # The one statement a step with no call keeps: a contrast's arms its launch could not pair.
            assert not step.limits or (step.limits, step.no_action) == (["unpaired_arms"], "lift_not_shown")
            continue
        for part in _parts(step):
            action = _CATALOGUE[part.action or ""]
            fields = action.params.model_fields
            required = {name for name, field in fields.items() if field.is_required()}
            left = [*part.needs, *part.as_launched, *part.after]
            assert part.calls and set(left) <= set(fields), sorted(set(left) - set(fields))
            filled = {name: action.example[name] for name in left if name in action.example}
            assert {name for name in left if name not in filled} <= set(fields) - required
            for call in part.calls:
                assert set(call) <= set(fields), sorted(set(call) - set(fields))
                assert not set(call) & set(left)
                assert required <= set(call) | set(left), sorted(required - set(call) - set(left))
                action.params.model_validate({**filled, **call})
        actions = [part.action for part in _parts(step)]
        rule = STEP_OF_REMEDY[remedied]
        if "run_launch" in actions:
            launch = next(part for part in _parts(step) if part.action == "run_launch")
            for call in launch.calls:
                assert _ARM_SETUP <= set(call) | set(launch.as_launched), "a launch left at a default is another arm"
                if call.get("n_variations"):
                    assert "variation_model" in launch.as_launched, "a generating arm's writer is named"
                if rule.replaces:
                    assert {"case_set_name", "case_set_version"} <= set(launch.as_launched), "a re-run's case set"
            assert not launch.call_arms or len(launch.call_arms) == len(launch.calls)
            if rule.balances:
                assert step.alike == ["reviewer_pool"] and "apparatus_settings" in launch.as_launched
            if rule.adds_evidence:
                assert "axis_may_not_generate" in step.limits, "the reading holds no template's axes"
            after = actions[actions.index("run_launch") + 1 :]
            assert "campaign_runs_add" in after, "a launched run joins no campaign until a call joins it"
            join = next(part for part in step.then if part.action == "campaign_runs_add")
            assert join.after["run_ids"] == "launched_runs"
            if rule.new_campaign:
                assert actions[0] == "campaign_create" and join.after["campaign_id"] == "created_campaign"
                assert "floors" in step.needs, "a floor a late criterion names is declared in the new campaign"
                assert {"this_report_stays_flagged", "control_may_read_late"} <= set(step.limits)
            else:
                assert "campaign_id" in join.needs, "the report holds its campaign by name"
            if rule.replaces:
                leaves = after[after.index("campaign_runs_add") :]
                assert "campaign_runs_remove" in leaves, "a re-run's replaced runs leave the campaign"
        else:
            assert not rule.launches, "a step that launches names its launch"
        if rule.adds_evidence:
            assert step.planned_cases is not None and step.planned_cases > 0
            launches = [call for part in _parts(step) if part.action == "run_launch" for call in part.calls]
            assert launches
            for call in launches:
                assert call["n_variations"] == step.planned_cases, "the planned size is the count of new cases"
        if step.spends:
            assert step.priced_by is not None
            pricer = _CATALOGUE[step.priced_by]
            assert pricer.permission == "read"
            for part in _parts(step):
                action = _CATALOGUE[part.action or ""]
                if issubclass(pricer.params, action.params):
                    left = [*part.needs, *part.as_launched]
                    filled = {name: action.example[name] for name in left if name in action.example}
                    for call in part.calls:
                        pricer.params.model_validate({**filled, **call})


def test_a_launch_repeats_the_arms_set_up_as_their_runs_were_launched() -> None:
    """New cases are more evidence for an arm only when run as that arm. The scaled campaign's arms ran at two repeats
    per case, where a launch left to its default runs three: the confirmation's launch fills the template, subject,
    models and repeats the arms' runs were launched with, read off the runs themselves, and names every other setting
    that defines an arm, to be set as they were. An arm that moved a lever beyond its model takes a launch of its own,
    so no other arm runs at its settings."""
    from packages.evals.tests.test_hot_path_oracle import subject
    from threetears.evals.analysis.service import assemble_campaign_bundle

    read = subject("scaled")
    bundle = assemble_campaign_bundle(read.host, read.campaign_id, read.scope_id)
    built = report_input(bundle)
    assert built.first_screen is not None
    step = next(step for step in built.first_screen.steps if step.closes == "best_of_n")
    launch = next(part for part in step.then if part.action == "run_launch")
    (lead,) = bundle.aspects
    summaries = {summary.run_id: summary for summary in lead.run_summaries}
    design = [arm for arm in (lead.design.control_arm, *lead.design.contrasts) if arm is not None]
    ran = {bundle.label_of(arm.variant_key): [summaries[run] for run in arm.run_ids] for arm in design}
    (call,) = launch.calls
    assert (call["template_id"], call["subject_id"]) == (lead.template_id, bundle.subject_id)
    assert call["models"] == [ran[arm][0].candidate_model for arm in step.arms]
    assert {summary.k_runs for arm in step.arms for summary in ran[arm]} == {call["k_runs"]}
    assert call["k_runs"] != _CATALOGUE["run_launch"].params.model_fields["k_runs"].default, "set up off the default"
    assert launch.needs == []
    rigs = [summary.apparatus for arm in step.arms for summary in ran[arm]]
    for name in ("measure_latency", "arm_schedule"):
        values = {getattr(rig, name) for rig in rigs if rig is not None}
        if None not in rigs and len(values) == 1 and None not in values:
            assert call[name] == values.pop(), f"{name} is filled from what the runs recorded"
        else:
            assert name in launch.as_launched, f"{name} is named where the runs did not record one value"
    order = (
        "measure_latency",
        "arm_schedule",
        "judge_model",
        "simulator_model",
        "overlays",
        "apparatus_settings",
        "cell_timeout_s",
        "variation_model",
    )
    assert launch.as_launched == [name for name in order if name not in call], (
        "every setting of an arm the reading holds no value for is named, and the writer of a generating launch too"
    )
    assert launch.call_arms == [list(step.arms)]


#: Four arms on one aspect: ``a`` and the control ``c`` differ only in their model; ``b`` moved a lever beyond its model;
#: ``d`` ran at other repeats, latency and schedule.
_FOUR = {
    **_HELD.arms["accuracy"],
    "b": ArmRuns(("run-b",), "model-a", 5, True),
    "d": ArmRuns(("run-d",), "model-d", 2, False, measure_latency=True, arm_schedule="sequential"),
}


def test_a_contrast_s_step_launches_only_the_arms_the_control_s_new_cases_pair_with() -> None:
    """One launch call takes one set of overlays for all its models, so an arm that moved a lever beyond its model, or
    ran at other repeats, latency or schedule, would run in a call of its own, whose new cases the control never runs;
    a contrast pairs only cases both sides ran. A step whose caveat reads a contrast launches only the arms in the
    control's call, at the planned size, and names the others (``unpaired_arms``) rather than spend on calls that add
    no pair. A confirmation, whose new campaign reads each arm's own figures too, still launches them and names them."""
    apart = next_step(
        "add_cases",
        arms=["a", "b", "c", "d"],
        readings=["accuracy/passed"],
        plan=Plan(4),
        setup=_HELD._replace(arms={"accuracy": _FOUR}),
        contrast=True,
    )
    assert [(call["models"], call["k_runs"]) for call in apart.calls] == [(["model-a", "model-c"], 5)]
    assert apart.call_arms == [["a", "c"]], "only the arm the control's new cases pair with is launched"
    assert (apart.limits, apart.unpaired_arms) == (["axis_may_not_generate", "unpaired_arms"], ["b", "d"])
    assert (apart.planned_cases, apart.spends, apart.priced_by) == (4, True, "launch_estimate")
    assert [part.action for part in apart.then] == ["campaign_runs_add"]
    confirm = next_step(
        "confirm_on_fresh_cases",
        arms=["a", "b", "c", "d"],
        readings=["accuracy/passed"],
        plan=Plan(4),
        setup=_HELD._replace(arms={"accuracy": _FOUR}),
    )
    launch = next(part for part in confirm.then if part.action == "run_launch")
    assert launch.call_arms == [["a", "c"], ["b"], ["d"]] and confirm.unpaired_arms == ["b", "d"], (
        "a confirmation launches every arm it confirms, and names the ones its contrasts cannot pair"
    )


def test_a_prompt_a_b_s_unsettled_step_makes_no_call_it_says_cannot_lift_it() -> None:
    """The commonest A/B: one model, two prompts. The candidate ``b`` moved its prompt, so a launch would run it in a
    call of its own, on new cases the control ``c`` never runs; the unsettled contrast pairs only the cases both ran, so
    no call lifts it. The step makes none and prices none, states no size, and names ``b`` as the arm it cannot pair
    (``lift_not_shown``), rather than spend on calls its own statement says add no pair."""
    unsettled = CaveatEntry(
        id="unsettled:accuracy", kind="unsettled", reach="report", arms=["b"], readings=["accuracy/passed"]
    )
    step = caveat_step(unsettled, plan=Plan(5, arms=("b", "c")), setup=_MOVED)
    assert (step.remedy, step.action, step.calls, step.then, step.no_action) == (
        "add_cases",
        None,
        [],
        [],
        "lift_not_shown",
    )
    assert (step.limits, step.unpaired_arms, step.arms) == (["unpaired_arms"], ["b"], ["b", "c"])
    assert (step.planned_cases, step.unsized, step.declared_look) == (None, None, False), "no call, no size"
    assert (step.spends, step.priced_by, step.call_arms, step.needs, step.as_launched) == (False, None, [], [], [])


def test_a_gate_s_step_launches_every_arm_and_names_none_unpaired() -> None:
    """A gate reads each arm's own usable cases, so an arm's new cases count toward it whichever call ran them:
    pairing does not apply. The gate's step keeps every call, each at its arms' own settings, and names no arm as
    unpaired."""
    step = gate_step(Gate(rule="too_small", fewest_cases=3), _HELD._replace(arms={"accuracy": _FOUR}))
    assert step.call_arms == [["a", "c"], ["b"], ["d"]]
    assert [(call["models"], call["k_runs"]) for call in step.calls] == [
        (["model-a", "model-c"], 5),
        (["model-a"], 5),
        (["model-d"], 2),
    ]
    assert "measure_latency" in step.as_launched, "two of the arms' runs recorded no rig, so it is the taker's"
    assert (step.limits, step.unpaired_arms) == (["axis_may_not_generate"], [])
    assert step.planned_cases is not None and all(call["n_variations"] == step.planned_cases for call in step.calls)


def test_a_balanced_re_run_runs_every_arm_beside_the_control_on_the_stored_cases() -> None:
    """A re-run that balances a confound runs the confounded arm beside the control, each in the call its settings
    take, on the template's stored cases: every call runs the same cases, so nothing is unpaired."""
    rerun = next_step(
        "rerun_balanced",
        arms=["b"],
        readings=["accuracy/passed"],
        setup=_HELD._replace(arms={"accuracy": _FOUR}),
        alike=["reviewer_pool"],
    )
    assert rerun.call_arms == [["b"], ["c"]] and rerun.limits == [] and rerun.unpaired_arms == [], (
        "a re-run runs the template's stored cases in every call, so its calls pair; the control is run beside it"
    )
    assert rerun.alike == ["reviewer_pool"] and "apparatus_settings" in rerun.as_launched
    every = {"run-b", "run-c1", "run-c2"}
    remove = next(part for part in rerun.then if part.action == "campaign_runs_remove")
    assert set(remove.calls[0]["run_ids"]) == every, "every run the re-run replaces leaves, the control's too"
    unheld = next_step("rerun_balanced", arms=["b"], readings=["accuracy/passed"], setup=_HELD)
    assert (unheld.action, unheld.no_action) == (None, "lift_not_shown"), "no dimension held alike, no call"


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
        "size_bias",
        "declared_look",
        "results_needed",
        "recording",
        "spends",
        "priced_by",
        "no_action",
        "as_launched",
        "after",
        "then",
        "alike",
        "call_arms",
        "limits",
        "unpaired_arms",
    }
    with pytest.raises(ValueError, match="planned size"):
        RemedyStep(remedy="add_cases", declared_look=True)
    with pytest.raises(ValueError, match="planned size"):
        RemedyStep(remedy="add_cases", declared_look=True, planned_cases=3, unsized="no_count")
    with pytest.raises(ValueError, match="prices it"):
        RemedyStep(remedy="add_cases", spends=True)
    with pytest.raises(ValueError, match="says why"):
        RemedyStep(remedy="no_action")
    with pytest.raises(ValueError, match="open-ended"):
        RemedyStep(remedy="add_cases", action="run_launch", calls=[{}], unsized="no_count")
    with pytest.raises(ValueError, match="no reason no action"):
        RemedyStep(remedy="add_cases", action="run_launch", calls=[{}], no_action="no_interval_to_gain")
    joined = RemedyStep(remedy="add_cases", action="campaign_runs_add", calls=[{}], needs=["campaign_id"])
    with pytest.raises(ValueError, match="none follows it"):
        RemedyStep(remedy="add_cases", then=[joined])
    with pytest.raises(ValueError, match="once"):
        RemedyStep(remedy="add_cases", action="run_launch", calls=[{"k_runs": 2}], as_launched=["k_runs"])
    with pytest.raises(ValueError, match="once"):
        RemedyStep(
            remedy="add_cases",
            action="campaign_runs_add",
            calls=[{}],
            needs=["run_ids"],
            after={"run_ids": "launched_runs"},
        )
    with pytest.raises(ValueError, match="no calls of its own"):
        RemedyStep(
            remedy="add_cases", action="run_launch", calls=[{}], then=[joined.model_copy(update={"then": [joined]})]
        )
    with pytest.raises(ValueError, match="cannot pair"):
        RemedyStep(remedy="add_cases", action="run_launch", calls=[{}], limits=["unpaired_arms"])
    with pytest.raises(ValueError, match="cannot pair"):
        RemedyStep(remedy="add_cases", action="run_launch", calls=[{}], unpaired_arms=["b"])
    with pytest.raises(ValueError, match="every call"):
        RemedyStep(remedy="add_cases", action="run_launch", calls=[{}, {}], call_arms=[["a"]])
    with pytest.raises(ValueError, match="apparatus_settings"):
        RemedyStep(remedy="rerun_balanced", action="run_launch", calls=[{}], alike=["reviewer_pool"])
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


# --- a step that is more evidence is planned, and launches its size or says why it cannot ----------------------------


def test_a_step_that_adds_cases_launches_its_planned_size_as_new_cases_or_says_why_it_has_none() -> None:
    """IA section 9.2 D's guard, held by name: a step that adds cases states a planned size, launches exactly that many
    new cases (never the stored ones again) and says it is one more look at its readings; one that can read no size
    says why and makes no call, rather than run open-ended."""
    planned = next_step("add_cases", arms=["a"], plan=Plan(7))
    assert (planned.declared_look, planned.planned_cases, planned.unsized) == (True, 7, None)
    assert planned.calls == [{"n_variations": 7}]
    assert planned.spends and planned.priced_by == "launch_estimate"
    assert [(part.action, part.needs, part.after) for part in planned.then] == [
        ("campaign_runs_add", ["campaign_id"], {"run_ids": "launched_runs"})
    ], "the new runs join the campaign that reads them, and nothing it holds leaves"
    unsized = next_step("add_cases", arms=["a"])
    assert (unsized.declared_look, unsized.planned_cases, unsized.unsized) == (False, None, "nothing_to_settle")
    assert (unsized.action, unsized.calls, unsized.spends) == (None, [], False)


def test_a_confirmation_creates_its_campaign_before_any_result_then_launches_its_size_and_joins_the_runs_to_it() -> (
    None
):
    """Cases a criterion or a pick never saw, in a campaign declared before their results: ``campaign_create`` takes a
    design with no run, the launch generates the planned count of new cases, the runs join the created campaign and
    its control is named from them. ``sweep_launch`` cannot: it declares no criteria, generates no new cases and joins
    each run as it is created, so no declaration precedes its results; its parameters say so."""
    planned = next_step("confirm_on_fresh_cases", arms=["a", "c"], plan=Plan(7), setup=_HELD)
    assert (planned.planned_cases, planned.action, planned.no_action, planned.declared_look) == (
        7,
        "campaign_create",
        None,
        False,
    )
    assert planned.calls == [{"subject_id": "subject-1"}] and planned.needs == [
        "name",
        "behavior",
        "question",
        "aspects",
        "floors",
    ]
    assert planned.limits == ["this_report_stays_flagged", "control_may_read_late", "axis_may_not_generate"], (
        "the caveat is read off this report's results, which nothing lifts; the control is named from a run that "
        "exists only once the launch starts; and the reading holds no template's axes"
    )
    launch, join, control = planned.then
    assert (launch.action, launch.calls) == (
        "run_launch",
        [
            {
                "template_id": "tmpl-1",
                "subject_id": "subject-1",
                "n_variations": 7,
                "models": ["model-a", "model-c"],
                "k_runs": 5,
            }
        ],
    )
    assert (join.action, join.after) == (
        "campaign_runs_add",
        {"campaign_id": "created_campaign", "run_ids": "launched_runs"},
    )
    assert (control.action, control.after) == (
        "campaign_update",
        {"campaign_id": "created_campaign", "control_from_run_id": "launched_control_run"},
    )
    assert planned.spends and planned.priced_by == "launch_estimate"
    sweep = _CATALOGUE["sweep_launch"].params.model_fields
    assert "n_variations" not in sweep and not {"question", "aspects"} & set(sweep)
    unsized = next_step("confirm_on_fresh_cases", arms=["a"])
    assert (unsized.unsized, unsized.no_action, unsized.action, unsized.spends) == (
        "nothing_to_settle",
        None,
        None,
        False,
    )


def test_too_few_cases_gate_is_lifted_by_a_declared_look_sized_to_the_floor_from_the_fewest_an_arm_holds() -> None:
    step = gate_step(Gate(rule="too_small", fewest_cases=3))
    assert (step.remedy, step.declared_look, step.planned_cases, step.calls) == (
        "add_cases",
        True,
        2,
        [{"n_variations": 2}],
    )
    unknown = gate_step(Gate(rule="too_small"))
    assert (unknown.declared_look, unknown.planned_cases, unknown.unsized, unknown.calls) == (
        False,
        None,
        "no_count",
        [],
    )


def test_a_campaign_too_small_to_read_is_gated_with_the_fewest_cases_an_arm_holds_and_planned_to_the_floor() -> None:
    bundle = _messy(unfinished=False)
    (aspect,) = bundle.aspects
    boards = [board.model_copy(update={"usable_cases": 3 + index}) for index, board in enumerate(aspect.scoreboard)]
    small = bundle.model_copy(update={"aspects": [aspect.model_copy(update={"scoreboard": boards})]})

    value = report_input(small).first_screen

    assert value is not None and value.gate is not None
    assert (value.gate.rule, value.gate.fewest_cases) == ("too_small", 3)
    (step,) = value.steps
    assert (step.remedy, step.declared_look, step.planned_cases) == ("add_cases", True, 2)
    assert step.calls and all(
        (call["n_variations"], call["template_id"], call["subject_id"]) == (2, aspect.template_id, small.subject_id)
        for call in step.calls
    ), "every arm the aspect holds runs the floor's new cases on its own template"
    assert [part.action for part in step.then] == ["campaign_runs_add"], "the new runs join it, and none leaves"


@pytest.mark.parametrize("remedied", get_args(Remedy))
def test_only_a_step_that_adds_cases_is_a_look_and_one_that_spends_is_priced(remedied: Remedy) -> None:
    for step in _steps_of(remedied):
        # A contrast's step whose every arm but the control is unpaired makes no call: it reads nothing again, states
        # no size and spends nothing (test_a_prompt_a_b_s_unsettled_step_makes_no_call_it_says_cannot_lift_it).
        stranded = step.no_action == "lift_not_shown" and bool(step.unpaired_arms)
        assert stranded == (remedied == "add_cases" and step.action is None)
        assert step.declared_look == (remedied == "add_cases" and not stranded), "only a step that re-reads looks again"
        assert (step.planned_cases is not None) == (
            remedied in ("add_cases", "confirm_on_fresh_cases") and not stranded
        )
        assert (step.priced_by is not None) == step.spends


@pytest.mark.parametrize("name", sorted(BEFORE))
def test_every_not_separated_reading_s_step_launches_its_size_or_says_why_it_cannot(name: str) -> None:
    """Every precision caveat (class E) any fixture raises takes a step that can close it: a launch of exactly its
    planned size in new cases, a look at the readings it names; or no call, saying why (no size can be read, or no
    added case gives the figure an interval)."""
    held = _inputs()[name]
    for caveat in caveat_list(held):
        if caveat.cls != "E":
            continue
        step = caveat_step(caveat, plan=plan_of(held, caveat))
        if step.planned_cases is not None:
            assert step.action == "run_launch" and step.declared_look, caveat.id
            assert step.calls == [{"n_variations": step.planned_cases}], caveat.id
        else:
            assert (step.action, step.calls, step.declared_look) == (None, [], False), caveat.id
            assert (step.unsized is None) != (step.no_action is None), caveat.id


def test_a_first_screen_showing_a_percentile_s_point_estimate_offers_no_act_that_cannot_work() -> None:
    """On a stored input's own first screen, read through the list, the class E reason shown is a gate cleared on a
    95th percentile, which the engine states no interval for at any size: the step says no added case gives it one,
    and offers no launch."""
    stored = _inputs()["stored:report_inputs/r2/input.json"]
    value = screen(stored, caveat_list(stored))
    by_id = {caveat.id: caveat for caveat in caveat_list(stored)}

    shown = [step for step in value.steps if by_id[step.closes or ""].cls == "E"]
    assert shown, "the fixture's first screen shows a precision reason"
    for step in shown:
        assert (step.remedy, step.no_action, step.action, step.declared_look) == (
            "no_action",
            "no_interval_to_gain",
            None,
            False,
        )


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


#: The seed of every planned size's simulation, and how far a share it states may sit below what it promises.
_SEED, _TOLERANCE = 20261115, 0.025

#: The gate the borderline simulation reads, its arm and the place it reads it (r1: resolves, in the trouble task).
_GATE, _GATED, _GATE_TASK = "trouble.resolves", "atlas-m", "trouble"


def _standardized(rng: random.Random, count: int, mean: float, spread: float) -> list[float]:
    """``count`` values whose mean and sample spread are exactly ``mean`` and ``spread``: today's cases, read as the
    truth a simulation draws new cases from."""
    drawn = [rng.gauss(0.0, 1.0) for _ in range(count)]
    centre = sum(drawn) / count
    scale = math.sqrt(sum((value - centre) ** 2 for value in drawn) / (count - 1))
    return [mean + spread * (value - centre) / scale for value in drawn]


def _gate_input(values: list[float]) -> ReportInput:
    """r1 with the gated arm's figure on the gate read from ``values`` by the pipeline's own mean interval (its
    clustered standard error and :func:`~threetears.evals.analysis.stats.mean_interval`, at the table's 95%), and the
    other survivor's figure cleared, so the borderline caveat is that arm's alone."""
    held = _inputs()["stored:report_inputs/r1/input.json"]
    count = len(values)
    mean = sum(values) / count
    sem = stats.clustered_standard_error(values, list(range(count)))
    assert sem is not None
    interval = stats.mean_interval(mean, sem, count)
    assert interval is not None
    figure = {"mean": mean, "ci": interval, "half_width": None, "sem": sem, "n": count, "n_cases": count}
    tasks = []
    for task in held.tasks:
        if task.id == _GATE_TASK:
            per_arm = {arm: dict(stats_of) for arm, stats_of in task.per_arm.items()}
            per_arm[_GATED]["resolves"] = per_arm[_GATED]["resolves"].model_copy(update=figure)
            per_arm["atlas-l"]["resolves"] = per_arm["atlas-l"]["resolves"].model_copy(update={"half_width": 0.5})
            task = task.model_copy(update={"per_arm": per_arm})
        tasks.append(task)
    return held.model_copy(update={"tasks": tasks, "caveat_list": None})


def _gate_lifted(values: list[float]) -> bool:
    """Whether a design reading ``values`` lifts the borderline caveat by the code that raises it, on its passing side:
    the gate's status there is ``pass``, and the list's implied raiser raises no borderline on it."""
    held = _gate_input(values)
    raised = {entry.id for entry in implied(held)}
    (read,) = criteria.criterion_reads(held, _GATED, criterion=_GATE)
    return read.status == "pass" and f"borderline:{_GATE}" not in raised


def test_a_borderline_gate_s_planned_cases_lift_it_four_times_in_five_at_today_s_estimate() -> None:
    """Seeded simulation through the real path, 2,000 designs. Today: 180 cases with a mean of 93 and a per-case spread
    of 25 against a gate at 90, its 95% interval reaching below the line: borderline. Drawing the planned new cases
    from today's estimate, pooling them with today's and reading the figure with the pipeline's own mean interval, the
    list raises no borderline and the gate passes in 84.7% of designs (held: at least 77.5%); a fresh design of the
    whole planned size lifts it in 80.5% (held within 2.5 points of 80%). Sized one-sided as before (z = 1.645 +
    0.842, 256 cases to add, not 373), the fresh design lifted it in 69.1% and the pooled one in 77.4%, which both
    bounds refuse."""
    rng = random.Random(_SEED)
    today = _standardized(rng, 180, 93.0, 25.0)
    held = _gate_input(today)
    caveat = next(entry for entry in implied(held) if entry.id == f"borderline:{_GATE}")
    assert caveat.arms == [_GATED]
    plan = plan_of(held, caveat)
    assert plan is not None and plan.cases is not None and plan.unsized is None
    step = caveat_step(caveat, plan=plan)
    assert (step.remedy, step.action, step.calls, step.declared_look) == (
        "add_cases",
        "run_launch",
        [{"n_variations": plan.cases}],
        True,
    )

    designs = 2000
    pooled = sum(_gate_lifted(today + [rng.gauss(93.0, 25.0) for _ in range(plan.cases)]) for _ in range(designs))
    fresh = sum(_gate_lifted([rng.gauss(93.0, 25.0) for _ in range(180 + plan.cases)]) for _ in range(designs))
    assert pooled / designs >= 0.80 - _TOLERANCE, pooled / designs
    assert abs(fresh / designs - 0.80) <= _TOLERANCE, fresh / designs


#: The paired outcomes of one case, control then candidate, and how often each comes up: a 4-point drop with a
#: per-case spread of about 40 points (60 both pass, 10 the control only, 6 the candidate only, 24 neither, per 100).
_PAIRS: tuple[tuple[tuple[int, int], int], ...] = (((1, 1), 60), ((1, 0), 10), ((0, 1), 6), ((0, 0), 24))
#: The aspect the unsettled simulation reads, and its margin in points.
_UNSETTLED, _MARGIN = "queue discipline", 10.0


def _paired_input(pairs: list[tuple[int, int]]) -> ReportInput:
    """The reasoning-effort campaign with both candidates' readings on one aspect read from ``pairs`` by the
    pipeline's own nominal interval (:func:`~threetears.evals.analysis.stats.bounded_difference_interval`, paired, at
    :data:`~threetears.evals.analysis.stats.INTERVAL_LEVEL`), against a 10-point margin."""
    held = _inputs()["case:discodon-1017-reasoning-effort-01a18d4d"].model_copy(update={"caveat_list": None})
    control = [float(first) for first, _second in pairs]
    candidate = [float(second) for _first, second in pairs]
    bounded = stats.bounded_difference_interval(
        control, candidate, paired=True, value_range=(0.0, 1.0), confidence=stats.INTERVAL_LEVEL
    )
    assert bounded is not None
    interval = (bounded[0] * 100, bounded[1] * 100)
    change = (sum(candidate) - sum(control)) / len(pairs) * 100
    update = {
        "change": change,
        "within_interval": interval,
        "gain_interval": interval,
        "gain_confidence": stats.INTERVAL_LEVEL,
        "margin": _MARGIN,
        "cases": [len(pairs), len(pairs)],
    }
    aspects = [
        aspect.model_copy(
            update={"margin": _MARGIN, "readings": [reading.model_copy(update=update) for reading in aspect.readings]}
        )
        if aspect.id == _UNSETTLED
        else aspect
        for aspect in held.aspects
    ]
    return held.model_copy(update={"aspects": aspects})


def _draw_pairs(rng: random.Random, count: int) -> list[tuple[int, int]]:
    outcomes = [pair for pair, _weight in _PAIRS]
    return rng.choices(outcomes, weights=[weight for _pair, weight in _PAIRS], k=count)


def test_an_unsettled_reading_s_planned_cases_lift_it_at_least_four_times_in_five() -> None:
    """Seeded simulation through the real path, 800 designs each. Today: 100 paired cases, a 4-point drop against a
    10-point margin, its nominal (bounded) interval reaching past the margin's line: unsettled. Drawing the planned new
    cases from today's estimate, pooling them with today's and reading the interval with the pipeline's own bounded
    function, :func:`~threetears.evals.analysis.answer.unsettled_arms` (what raises the caveat) no longer raises it in
    94.4% of designs, and a fresh design of the whole planned size in 89.9% (each held at least 77.5%). The size reads
    its spread off today's bounded interval, which is wider than its cases' own spread at a small count, so it lifts
    more often than four in five: a size, never a shortfall. The excess is bounded by the method: a spread read ``k``
    times too wide asks for about ``k**2`` times the cases, which a closing interval as many times wider clears in
    about ``Phi(z * k)`` of designs (``z`` the quantile of four in five). Today's interval is ``k = 1.40`` times the
    normal one at the cases' own spread, so the fresh rate is held at most ``Phi(0.84 * 1.40) = 88.1%`` plus 2.5 points,
    an approximate ceiling: the bounded interval's width is not exactly ``k`` times the normal one at the larger count,
    and the part that shrinks with the count puts the true rate a little above ``Phi(z * k)``, which the 2.5 points
    cover. A size a third too large passes, a doubled one (about 99%) does not. (The one-sided size this replaced reached
    80.6% and 82.0% here; the borderline simulation above, on a t interval, is the one that refuses it.) The plan is
    read from the nominal interval at its own level, never the gain interval at a corrected one, so a corrected gain
    level changes nothing; the step names the control, whose new cases pair with the candidates'."""
    today = [pair for pair, weight in _PAIRS for _ in range(weight)]
    held = _paired_input(today)
    caveat = next(entry for entry in raise_caveats(held) if entry.id == f"unsettled:{_UNSETTLED}")
    plan = plan_of(held, caveat)
    assert plan is not None and plan.cases is not None and plan.unsized is None
    step = screen(held, [caveat]).steps[0]
    assert (step.remedy, step.planned_cases, step.declared_look) == ("add_cases", plan.cases, True)
    assert step.calls == [{"n_variations": plan.cases}]
    control = answer.control_arm(held)
    assert control is not None and step.arms == [*caveat.arms, control]
    corrected = held.model_copy(
        update={
            "aspects": [
                aspect.model_copy(
                    update={
                        "readings": [
                            reading.model_copy(update={"gain_interval": (-20.0, 12.0), "gain_confidence": 0.9969})
                            for reading in aspect.readings
                        ]
                    }
                )
                for aspect in held.aspects
            ]
        }
    )
    assert plan_of(corrected, caveat) == plan, "the gain interval at a corrected level is not what lifts it"

    rng = random.Random(_SEED)
    designs = 800

    def lifted(pairs: list[tuple[int, int]]) -> bool:
        return f"unsettled:{_UNSETTLED}" not in answer.unsettled_arms(_paired_input(pairs))

    pooled = sum(lifted(today + _draw_pairs(rng, plan.cases)) for _ in range(designs))
    fresh = sum(lifted(_draw_pairs(rng, len(today) + plan.cases)) for _ in range(designs))
    assert pooled / designs >= 0.80 - _TOLERANCE, pooled / designs
    assert fresh / designs >= 0.80 - _TOLERANCE, fresh / designs
    # The method's upper bound: today's interval against a normal one at the truth's own per-case spread.
    weights = sum(weight for _pair, weight in _PAIRS)
    drift = sum((second - first) * weight for (first, second), weight in _PAIRS) / weights
    spread = math.sqrt(sum(((second - first) - drift) ** 2 * weight for (first, second), weight in _PAIRS) / weights)
    reading = next(reading for aspect in held.aspects if aspect.id == _UNSETTLED for reading in aspect.readings)
    assert reading.within_interval is not None
    width = (reading.within_interval[1] - reading.within_interval[0]) / 2 / 100
    normal = statistics.NormalDist()
    wide = width / (normal.inv_cdf(0.5 + stats.INTERVAL_LEVEL / 2) * spread / math.sqrt(len(today)))
    assert 1.3 < wide < 1.5, wide
    assert fresh / designs <= normal.cdf(normal.inv_cdf(0.80) * wide) + _TOLERANCE, fresh / designs


def test_a_gate_cleared_on_a_point_estimate_is_sized_to_the_floor_or_says_no_case_gives_it_an_interval() -> None:
    """A mean read from fewer cases than the floor where its interval is first stated is sized to that floor and
    launched as that many new cases; a percentile, or a mean read from that many that still states none, gains no
    interval from more cases, so its step takes no action and says so."""
    held = _inputs()["stored:report_inputs/r1/input.json"]
    tasks = []
    for task in held.tasks:
        if task.id == _GATE_TASK:
            per_arm = {arm: dict(stats_of) for arm, stats_of in task.per_arm.items()}
            for arm in ("atlas-l", _GATED):
                per_arm[arm]["resolves"] = per_arm[arm]["resolves"].model_copy(
                    update={"half_width": None, "n": 3, "n_cases": 3}
                )
            task = task.model_copy(update={"per_arm": per_arm})
        tasks.append(task)
    few = held.model_copy(update={"tasks": tasks, "caveat_list": None})
    caveat = next(entry for entry in implied(few) if entry.id == f"point_estimate:{_GATE}")
    plan = plan_of(few, caveat)
    assert plan == Plan(stats.SMALL_N_BAND_FLOOR - 3)
    step = caveat_step(caveat, plan=plan)
    assert (step.remedy, step.action, step.calls) == ("add_cases", "run_launch", [{"n_variations": 2}])

    enough = few.model_copy(
        update={
            "tasks": [
                task.model_copy(
                    update={
                        "per_arm": {
                            arm: {
                                key: stat.model_copy(update={"n": 40, "n_cases": 40}) if key == "resolves" else stat
                                for key, stat in stats_of.items()
                            }
                            for arm, stats_of in task.per_arm.items()
                        }
                    }
                )
                for task in few.tasks
            ]
        }
    )
    percentile = next(entry for entry in caveat_list(held) if entry.kind == "point_estimate")
    for source, entry in ((enough, caveat), (held, percentile)):
        plan = plan_of(source, entry)
        assert plan == Plan(no_action="no_interval_to_gain"), entry.id
        step = caveat_step(entry, plan=plan)
        assert (step.remedy, step.no_action, step.action, step.calls) == ("no_action", "no_interval_to_gain", None, [])


def test_every_case_reading_alike_states_no_size_and_offers_nothing_to_repeat() -> None:
    """At today's estimate no added case is expected to differ, so no size can be promised: the step says why, makes
    no call and is no look."""
    entry = CaveatEntry.model_validate(
        {
            "id": "not_available:no_variation",
            "kind": "not_available",
            "reach": "scoped",
            "state": "not_available",
            "reason": "no_variation",
            "cases": 20,
        }
    )
    step = caveat_step(entry, plan=plan_of(_inputs()["stored:report_inputs/r4/input.json"], entry))
    assert (step.remedy, step.unsized, step.planned_cases) == ("add_cases", "no_spread", None)
    assert (step.action, step.calls, step.declared_look, step.spends) == (None, [], False, False)


def test_a_pick_among_many_is_confirmed_in_a_new_campaign_at_a_size_that_says_it_reads_high() -> None:
    """No run added to this campaign lifts a pick among many (raised from how many candidates it holds, whatever they
    read): the step says a new campaign is needed, makes no call that would re-run the cases the pick was made on, and
    states the confirmation's size, read from the pick's own estimate and marked too small for it."""
    held = _inputs()["built:scaled"]
    caveat = next(entry for entry in caveat_list(held) if entry.kind == "best_of_n")
    plan = plan_of(held, caveat)
    step = caveat_step(caveat, plan=plan)

    assert (step.remedy, step.action, step.no_action) == ("confirm_on_fresh_cases", "campaign_create", None)
    assert plan is not None and plan.cases is not None and step.planned_cases == plan.cases
    launch, join, *_control = step.then
    assert (launch.action, launch.calls) == ("run_launch", [{"n_variations": plan.cases}]), "new cases, never its own"
    assert (join.action, join.after["campaign_id"]) == ("campaign_runs_add", "created_campaign")
    assert step.size_bias == "picked_best_of_n" and not step.declared_look
    assert step.spends and step.priced_by == "launch_estimate"
    control = answer.control_arm(held)
    assert control is None or control in step.arms
    settled = held.model_copy(
        update={
            "caveat_list": None,
            "aspects": [
                aspect.model_copy(
                    update={
                        "readings": [
                            reading.model_copy(update={"change": 30.0, "within_interval": (25.0, 35.0)})
                            for reading in aspect.readings
                        ]
                    }
                )
                for aspect in held.aspects
            ],
        }
    )
    assert "best_of_n" in {entry.id for entry in implied(settled)}, "however its readings settle, the pick stays one"


def test_a_late_criterion_is_confirmed_in_a_new_campaign_never_by_re_running_its_cases() -> None:
    for name, held in _inputs().items():
        for caveat in caveat_list(held):
            if caveat.kind != "late":
                continue
            step = caveat_step(caveat, plan=plan_of(held, caveat))
            assert step.remedy == "confirm_on_fresh_cases" and not step.declared_look, (name, caveat.id)
            if step.planned_cases is None:
                assert (step.action, step.calls) == (None, []), (name, caveat.id)
                continue
            assert step.action == "campaign_create", (name, caveat.id)
            launch, join, *_control = step.then
            assert launch.calls == [{"n_variations": step.planned_cases}], (name, caveat.id)
            assert join.after["campaign_id"] == "created_campaign", (name, caveat.id)


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


def test_one_short_run_stated_on_two_arms_is_one_caveat_holding_both_never_one_dropped() -> None:
    """A short run stated once per arm it ran is one caveat naming both arms and the run, its reach judged on both."""
    stored = _inputs()["stored:report_inputs/r4/input.json"]
    first, second = stored.arms[0].id, stored.arms[1].id
    runs = [ShortRun(run="run-9", arm=arm, note="lost two trials") for arm in (first, second)]
    held = stored.model_copy(
        update={"caveats": stored.caveats.model_copy(update={"short_runs": runs}), "caveat_list": None}
    )

    (short,) = [entry for entry in raise_caveats(held) if entry.kind == "short_run"]
    assert (short.id, short.arms, short.runs) == ("short_run:run-9", [first, second], ["run-9"])
    one = held.model_copy(update={"caveats": held.caveats.model_copy(update={"short_runs": runs[:1]})})
    (alone,) = [entry for entry in raise_caveats(one) if entry.kind == "short_run"]
    assert alone.arms == [first], "the merge holds both parts, not the first one kept"
    reach = "report" if all(comparison & {first, second} for comparison in answer.comparisons(held)) else "scoped"
    assert short.reach == reach


def test_one_id_stated_on_two_aspects_is_one_caveat_holding_both_never_one_dropped() -> None:
    """Two late floors under one key, each on its own aspect, are one caveat (the id a report document names it by)
    that names both readings; and no fixture's list holds an id twice."""
    stored = _inputs()["stored:report_inputs/r4/input.json"]
    aspect = stored.aspects[0].id
    floors = [
        LateCriterion(criterion="floor", name="twice", set_at="2026-11-01", aspect=aspect, reading=reading)
        for reading in ("passed", "cost")
    ]
    held = stored.model_copy(
        update={"caveats": stored.caveats.model_copy(update={"late_criteria": floors}), "caveat_list": None}
    )

    (late,) = [entry for entry in raise_caveats(held) if entry.kind == "late"]
    assert (late.id, late.readings) == ("late:floor:twice", [f"{aspect}/passed", f"{aspect}/cost"])
    for name, value in _inputs().items():
        ids = [entry.id for entry in raise_caveats(value.model_copy(update={"caveat_list": None}))]
        assert len(ids) == len(set(ids)), name


# --- the numbers side imports nothing from the page -------------------------------------------------------------------


@pytest.mark.parametrize("module", [answer, criteria, costs, caveats, per_input], ids=lambda module: module.__name__)
def test_the_numbers_side_of_the_call_imports_nothing_from_the_page(module: Any) -> None:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert not [name for name in imported if name.startswith("threetears.evals.analysis.report.")]
    assert not [name for name in imported if name == "threetears.evals.analysis.report"]
