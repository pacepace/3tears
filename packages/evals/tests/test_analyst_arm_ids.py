"""Every analyst action names an arm the way any result prints it: one rule, held across the actions.

Contracts held here (``threetears.evals.actions.params.written_arm`` and every analyst action's arm parameters):

- **one rule**: an arm written as its id, as the cited form a result prints (``arm:<id>``) or as its figure key
  (``{arm:<id>}``) is the same arm; anything else passes through to the parameter's own type;
- **every arm parameter takes it**: each analyst action's parameters that name an arm (``arm_filter``, ``arm_ids``,
  ``what_if``, a block's ``emphasis`` and its query's ``arms``) say so in their description, and a call naming its
  arms in the printed forms answers exactly as the call naming them by id, on the fixture snapshots;
- **the W2 regression**: ``settle_cases`` on the arm the overview prints as ``arm:<id>`` answers instead of refusing.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from packages.evals.tests.analyst_support import action_calls, fixture_snapshots, read_actions
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, OpsFixture, ops_fixture
from threetears.evals.actions import ActionOutcome, MountedTool, analyst_tools, eval_catalogue
from threetears.evals.actions.params import ARM_FORMS, written_arm
from threetears.evals.analysis.analyst.snapshot import frozen_input
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel import EvalReportSnapshot

#: The snapshots the actions are called on: a routed choice, a sweep over many arms, and a bundle-built change.
SNAPSHOTS = ("r1", "r2", "kairo round 2")

#: The parameters that name one arm, and the one naming a list of arms.
ONE_ARM = ("arm_filter", "what_if", "emphasis")
ARM_LISTS = ("arm_ids",)


def _seeded() -> tuple[OpsFixture, dict[str, EvalReportSnapshot]]:
    fixture = ops_fixture()
    snapshots = fixture_snapshots(TOYHOST_SCOPE)
    for snapshot in snapshots.values():
        fixture.host.eval_host.storage.save_report_snapshot(snapshot)
    return fixture, snapshots


def _tool() -> MountedTool:
    (spec,) = analyst_tools()
    return eval_catalogue().mount(spec)


async def _call(tool: MountedTool, fixture: OpsFixture, arguments: dict[str, Any]) -> ActionOutcome:
    return await tool.call(arguments, host=fixture.host, caller=CALLER)


@pytest.mark.parametrize(
    ("written", "read"),
    [
        ("fast.v2", "fast.v2"),
        ("arm:fast.v2", "fast.v2"),
        ("{arm:fast.v2}", "fast.v2"),
        (" arm:fast.v2 ", "fast.v2"),
        ("{fast.v2}", "{fast.v2}"),
        ("armada", "armada"),
        (None, None),
        (["arm:a"], ["arm:a"]),
    ],
)
def test_an_arm_written_as_any_result_prints_it_reads_to_its_id_and_anything_else_passes_through(
    written: object, read: object
) -> None:
    assert written_arm(written) == read


def _arm_parameters(action_name: str) -> list[str]:
    """The parameters of one analyst action that name an arm, read off its parameters' schema: each whose name is one
    of the arm parameters, or the block query's description naming arms."""
    action = next(action for action in read_actions() if action.name == action_name)
    fields = action.params.model_fields
    return [name for name in fields if name in (*ONE_ARM, *ARM_LISTS)]


def test_every_arm_parameter_says_it_takes_an_arm_in_every_printed_form() -> None:
    named = {action.name: _arm_parameters(action.name) for action in read_actions()}
    assert {"arms_compare", "settle_cases", "measure_breakdown", "price_tradeoff", "block_render"} <= {
        name for name, held in named.items() if held
    }
    for action in read_actions():
        schema = action.params.model_json_schema()
        for name in named[action.name]:
            assert ARM_FORMS in json.dumps(schema["properties"][name]), (action.name, name)
        if "query" in schema["properties"] and action.name == "block_render":
            assert ARM_FORMS in schema["properties"]["query"]["description"]


def _rewritten(arguments: dict[str, Any], form: Callable[[str], str], held: ReportInput) -> dict[str, Any] | None:
    """The call with every arm it names written in ``form``; None where it names none."""
    arms = {arm.id for arm in held.arms}
    found = False
    rewritten = dict(arguments)
    for name in ONE_ARM:
        if isinstance(arguments.get(name), str) and arguments[name] in arms:
            rewritten[name] = form(arguments[name])
            found = True
    for name in ARM_LISTS:
        if isinstance(arguments.get(name), list):
            rewritten[name] = [form(arm) for arm in arguments[name]]
            found = True
    query = arguments.get("query")
    if isinstance(query, dict) and query.get("arms"):
        rewritten["query"] = {**query, "arms": [form(arm) for arm in query["arms"]]}
        found = True
    return rewritten if found else None


def _calls_naming_arms(held: ReportInput, snapshot_id: str) -> list[dict[str, Any]]:
    """Every contract call naming an arm, plus a block drawing two arms with one stressed and a two-arm settle."""
    calls = [
        arguments
        for action in read_actions()
        for arguments in action_calls(action, held, snapshot_id)
        if _rewritten(arguments, lambda arm: arm, held) is not None
    ]
    first, last = held.arms[0].id, held.arms[-1].id
    key = next(iter([*held.judged, *held.measures]))
    calls.append(
        {
            "action": "block_render",
            "snapshot_id": snapshot_id,
            "block_kind": "bars",
            "query": {"source": "measures" if key in held.measures else "judged", "arms": [first, last]},
            "emphasis": last,
        }
    )
    calls.append({"action": "measure_read", "snapshot_id": snapshot_id, "measure_key": key, "arm_filter": last})
    return calls


@pytest.mark.parametrize("name", SNAPSHOTS)
async def test_a_call_naming_its_arms_as_results_print_them_answers_as_the_call_naming_them_by_id(name: str) -> None:
    fixture, snapshots = off_loop(_seeded)
    tool = _tool()
    snapshot = snapshots[name]
    held = frozen_input(snapshot)
    calls = _calls_naming_arms(held, snapshot.id)
    assert any("arm_ids" in call for call in calls) and any("arm_filter" in call for call in calls)
    for arguments in calls:
        plain = await _call(tool, fixture, arguments)
        for form in (lambda arm: f"arm:{arm}", lambda arm: f"{{arm:{arm}}}"):
            rewritten = _rewritten(arguments, form, held)
            assert rewritten is not None
            printed = await _call(tool, fixture, rewritten)
            where = f"{arguments['action']} on {name}: {rewritten}"
            assert printed.ending == plain.ending, f"{where}: {printed.text}"
            assert (printed.text, printed.structured) == (plain.text, plain.structured), where


async def test_settle_cases_reads_the_arm_the_overview_prints_rather_than_refusing_it() -> None:
    """The W2 r2 finding: the overview prints `arm:<id>` and settle_cases refused that spelling."""
    fixture, snapshots = off_loop(_seeded)
    tool = _tool()
    snapshot = snapshots["r2"]
    held = frozen_input(snapshot)
    overview = await _call(tool, fixture, {"action": "campaign_overview", "snapshot_id": snapshot.id})
    candidate = next(arm.id for arm in held.arms if arm.role != "control")
    assert f"arm:{candidate}" in overview.text
    settled = await _call(
        tool, fixture, {"action": "settle_cases", "snapshot_id": snapshot.id, "arm_ids": [f"arm:{candidate}"]}
    )
    assert settled.ending == "ok", settled.text
    assert settled.structured is not None and settled.structured["arms"] == [f"arm:{candidate}"]
