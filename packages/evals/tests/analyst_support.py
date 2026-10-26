"""The fixture snapshots every analyst tool is held to, and the calls each tool is held to them with.

The inputs are the reference inputs r1 to r5 (bare report inputs, as a local analyst server seeds them), the misfit
inputs (:mod:`packages.evals.tests.fixtures.report_plans.misfits`), and the Kairo round 2 campaign frozen from its
bundle with the analyst's example pool, so a snapshot holding a bundle is held to the same contract as a bare one.

:data:`ANALYST_CALLS` says how each analyst action is called on one input. An action it does not name is called with
its own example's parameters, the snapshot id put in, so every action a later chunk registers is held to the contract
with nothing added here; a chunk whose action needs arguments read off the input adds its entry.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import cache
from typing import Any

from packages.evals.tests.fixtures.report_plans.dataset import kairo_bundle
from packages.evals.tests.fixtures.report_plans.misfits import MISFITS, NAMES, reference_data
from threetears.evals.actions import Action, analyst_actions
from threetears.evals.analysis.analyst.snapshot import ANALYST_EXAMPLES, report_snapshot
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.kernel import EvalReportSnapshot

__all__ = [
    "ANALYST_CALLS",
    "KAIRO",
    "TEXT_BUDGET",
    "action_calls",
    "read_actions",
    "fixture_inputs",
    "fixture_snapshots",
]

#: The most characters an analyst action's text may run to at its default limits: what one tool result may take of an
#: analyst's context.
TEXT_BUDGET = 12_000

#: The Kairo round 2 snapshot's name among the fixtures.
KAIRO = "kairo round 2"

#: When the Kairo bundle is recorded as assembled: a fixed instant, so its snapshot is the same on every run.
_KAIRO_ASSEMBLED_AT = "2026-10-20T00:00:00+00:00"

#: Who the fixture snapshots are frozen by.
_FROZEN_BY = "agent:test"


def _dominance_calls(held: ReportInput) -> list[dict[str, Any]]:
    """The decision's own axes, then two extra ones where the input holds them."""
    keys = [*held.measures, *held.judged]
    return [{}, {"axes": keys[:2]}] if keys else [{}]


def _sensitivity_calls(held: ReportInput) -> list[dict[str, Any]]:
    """The first margin, floor and gate the input declares, one sweep each."""
    margins = [f"margin:{aspect.id}" for aspect in held.aspects if aspect.readings]
    floors = [f"floor:{read.aspect}/{read.floor}" for read in held.floors]
    gates = [f"threshold:{criterion.id}" for criterion in held.criteria if criterion.kind in ("gate", "guardrail")]
    return [{"parameter": found[0]} for found in (margins, floors, gates) if found]


def _breakdown_calls(held: ReportInput) -> list[dict[str, Any]]:
    """Where the input has a control: per way of slicing it holds, the first measure the control holds there."""
    control = next((arm.id for arm in held.arms if arm.role == "control"), None)
    if control is None:
        return []
    slices = {
        "stratum": [stratum.per_arm for stratum in held.strata],
        "task": [task.per_arm for task in held.tasks],
        "cell": [cell.per_arm for cell in held.cells.values()],
    }
    calls: list[dict[str, Any]] = []
    for slice_by, held_by in slices.items():
        keys = [key for per_arm in held_by for key in per_arm.get(control, {})]
        if keys:
            calls.append({"measure_key": keys[0], "slice_by": slice_by})
    return calls


#: How each analyst action is called on one input, where its example's parameters will not do: the argument sets,
#: without ``action`` or ``snapshot_id``.
ANALYST_CALLS: dict[str, Callable[[ReportInput], list[dict[str, Any]]]] = {
    "options_dominance": _dominance_calls,
    "decision_sensitivity": _sensitivity_calls,
    "measure_breakdown": _breakdown_calls,
}


@cache
def fixture_inputs() -> Mapping[str, ReportInput]:
    """Every bare fixture input, by name: r1 to r5, then the misfits.

    :return: The inputs.
    :rtype: Mapping[str, ReportInput]
    """
    inputs = {name: ReportInput.model_validate(reference_data(name)) for name in NAMES}
    inputs |= {name: ReportInput.model_validate(build()) for name, build in MISFITS.items()}
    return inputs


@cache
def _kairo_snapshot(scope_id: str) -> EvalReportSnapshot:
    bundle = kairo_bundle()
    return report_snapshot(
        report_input(bundle, examples=ANALYST_EXAMPLES),
        scope_id=scope_id,
        campaign_id=bundle.campaign_id,
        source="campaign",
        created_by=_FROZEN_BY,
        bundle=bundle,
        assembled_at=_KAIRO_ASSEMBLED_AT,
    )


def fixture_snapshots(scope_id: str) -> dict[str, EvalReportSnapshot]:
    """A snapshot of every fixture input in one scope, by name, and of the Kairo round 2 campaign last.

    :param scope_id: The scope they live in.
    :ptype scope_id: str
    :return: The snapshots, unsaved.
    :rtype: dict[str, EvalReportSnapshot]
    """
    snapshots = {
        name: report_snapshot(held, scope_id=scope_id, campaign_id=None, source="report_input", created_by=_FROZEN_BY)
        for name, held in fixture_inputs().items()
    }
    snapshots[KAIRO] = _kairo_snapshot(scope_id)
    return snapshots


def action_calls(action: Action, held: ReportInput, snapshot_id: str) -> list[dict[str, Any]]:
    """The calls one analyst action is held to on one snapshot: its entry in :data:`ANALYST_CALLS`, else its example.

    :param action: The action.
    :ptype action: Action
    :param held: The snapshot's report input.
    :ptype held: ReportInput
    :param snapshot_id: The snapshot.
    :ptype snapshot_id: str
    :return: Each call's arguments, ``action`` and ``snapshot_id`` among them.
    :rtype: list[dict[str, Any]]
    """
    build = ANALYST_CALLS.get(action.name)
    argument_sets = build(held) if build is not None else [dict(action.example)]
    return [{**arguments, "action": action.name, "snapshot_id": snapshot_id} for arguments in argument_sets]


def read_actions() -> list[Action]:
    """Every analyst action that reads a snapshot: the ones held to the contract on every fixture."""
    return [action for action in analyst_actions() if action.permission == "read"]
