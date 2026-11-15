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
- **Every reason has its remedy**, a typed step (:func:`~threetears.evals.analysis.caveats.next_step`): each action it
  names is in the engine's catalogue, a step that is more evidence states a planned size and is a declared look, and a
  step that spends names the action that prices it.
- **The numbers side imports nothing from the page**: the answer read, a criterion's status, the cost rule and the
  list.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, get_args

import pytest

from packages.evals.tests.caveat_pins import CASE_DIRS, PINS, read_fixture, built_inputs, caveat_reads, stored_inputs
from threetears.evals.actions.engine import eval_catalogue
from threetears.evals.analysis import answer, caveats, costs, criteria, per_input
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.caveats import (
    CLASS_OF_KIND,
    IMPLIED_KINDS,
    STATED_KINDS,
    PRICED_BY,
    REMEDY_OF_REASON,
    STEP_OF_REMEDY,
    call_read,
    caveat_step,
    from_legacy,
    next_step,
    raise_caveats,
)
from threetears.evals.analysis.checks import Check, Looks
from threetears.evals.analysis.report.call import caveat_materiality, implied_caveats, report_caveats
from threetears.evals.analysis.report_input import CaveatEntry, Remedy, RemedyStep, ReportInput, report_input_digest
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
    listed = [(entry.id, entry.reach) for entry in held.caveat_list if entry.kind in STATED_KINDS | IMPLIED_KINDS]

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
    assert caveat_step(entry) is not None, "every reason has a remedy"


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
    assert step.arguments["coordinate"] == "measure_definition"
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
    assert [(step.remedy, step.action, step.arguments) for step in value.steps] == [
        ("resume_or_rerun", "job_resume", {"runs": [run.run]})
    ]


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
    assert value.steps[0].arguments == {"arms": [_SPLIT], "coordinate": "served_model"}


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


def test_every_reason_has_exactly_one_remedy_and_every_remedy_a_rule() -> None:
    reasons = set().union(*REASONS_OF.values())
    assert len(reasons) == 16
    assert set(REMEDY_OF_REASON) == reasons, "a reason with no remedy, or a remedy for no reason"
    assert set(STEP_OF_REMEDY) == set(get_args(Remedy))
    assert "nothing_to_do" in REMEDY_OF_REASON.values(), "the answer itself is marked, never left without a remedy"


def test_every_action_a_step_names_is_in_the_engine_s_catalogue() -> None:
    catalogue = {action.name for action in eval_catalogue().actions}
    named = {rule.action for rule in STEP_OF_REMEDY.values() if rule.action is not None}
    named |= {next_step("remeasure_or_attest", coordinate="measure_definition").action or ""}
    assert named and named <= catalogue, sorted(named - catalogue)
    assert PRICED_BY in catalogue


@pytest.mark.parametrize("remedied", get_args(Remedy))
def test_a_step_that_is_more_evidence_is_planned_and_a_declared_look_and_one_that_spends_is_priced(
    remedied: Remedy,
) -> None:
    rule = STEP_OF_REMEDY[remedied]
    for cases in (None, 2, 40):
        step = next_step(remedied, arms=["a"], cases=cases)
        assert step.declared_look == rule.adds_evidence
        if rule.adds_evidence:
            assert step.planned_cases is not None and step.planned_cases >= 1
            assert (step.planned_cases + (cases or 0)) >= 5, "planned to reach the floor where an interval is stated"
        else:
            assert step.planned_cases is None
        assert step.spends == rule.spends and step.priced_by == (PRICED_BY if rule.spends else None)


def test_no_step_reads_until_it_separates_and_none_is_taken_by_code() -> None:
    """A step's size is fixed before it runs: it holds no field that waits on a result. And nothing in the package
    takes one: the step's shape is read only by the modules that compute it."""
    assert set(RemedyStep.model_fields) == {
        "remedy",
        "closes",
        "action",
        "arguments",
        "planned_cases",
        "declared_look",
        "results_needed",
        "recording",
        "spends",
        "priced_by",
    }
    with pytest.raises(ValueError, match="planned size"):
        RemedyStep(remedy="add_cases", declared_look=True)
    with pytest.raises(ValueError, match="prices it"):
        RemedyStep(remedy="add_cases", spends=True)
    source = Path(caveats.__file__).resolve().parents[1]
    readers = sorted(
        str(path.relative_to(source))
        for path in source.rglob("*.py")
        if "RemedyStep" in path.read_text(encoding="utf-8")
    )
    assert readers == ["analysis/__init__.py", "analysis/caveats.py", "analysis/report_input.py"]


def test_a_caveat_with_no_reason_takes_its_kind_s_remedy() -> None:
    for kind, remedied in (("looks", "declare_looks"), ("judge_uncalibrated", "check_against_people")):
        entry = CaveatEntry.model_validate({"id": kind, "kind": kind, "class": CLASS_OF_KIND[kind], "reach": "report"})
        step = caveat_step(entry)
        assert step is not None and step.remedy == remedied


# --- the numbers side imports nothing from the page -------------------------------------------------------------------


@pytest.mark.parametrize("module", [answer, criteria, costs, caveats, per_input], ids=lambda module: module.__name__)
def test_the_numbers_side_of_the_call_imports_nothing_from_the_page(module: Any) -> None:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert not [name for name in imported if name.startswith("threetears.evals.analysis.report.")]
    assert not [name for name in imported if name == "threetears.evals.analysis.report"]
