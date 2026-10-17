"""Rung 4, ``examples/4-booking/``: an agent books on a calendar world, graded by the day it leaves; then served.

The agent runs here as a reader runs it, from a copy of its directory with no key set, so its model can only replay
``recordings/agent.jsonl``. Nothing here assumes what the model did: each cell is read back from the store and held
to the checks. The loop itself is driven by a scripted model below, so a refused call and the agent being told why
are covered whatever the recording holds. The world passes the engine's own conformance kit, and ``serve.py`` hosts
the same runs on the command line and over MCP.
"""

from __future__ import annotations

import asyncio
import json
import runpy
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from threetears.evals.kernel import strict_response_format
from threetears.evals.kernel.host import check_world_conformance
from threetears.evals.quick import run_cli
from threetears.evals.run import get_result_trace, list_results
from threetears.evals.schema import ClientRequestSettings, RecordedCall
from threetears.evals.testing import RecordedCompletions, RecordedPrompt
from packages.evals.tests.ladder_support import (
    EXAMPLES,
    count_calls,
    load,
    non_blank_lines,
    offline_copy,
    recorded_spend,
    secret_markers,
    stated_spend,
)
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.report_support import page_violations
from packages.evals.tests.test_package_matrix import REPO_ROOT, SOURCE_ROOT, public_root_violations

RUNG = EXAMPLES / "4-booking"
AGENT, SERVE = RUNG / "agent.py", RUNG / "serve.py"
RECORDING = RUNG / "recordings" / "agent.jsonl"
CASES = [json.loads(line) for line in (RUNG / "requests.jsonl").read_text(encoding="utf-8").splitlines()]


def _agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    offline_copy("4-booking", tmp_path, monkeypatch)
    return load(Path("agent.py"), "agent", monkeypatch)


def test_the_requests_hold_full_day_hard_negatives() -> None:
    assert len(CASES) == 12
    for case in CASES:
        free = [hour for hour in range(case["start"], case["end"]) if hour not in case["world"]["busy"]]
        # A request is to be booked exactly when its window has a free hour; the "full" ones have none.
        assert (case["expect"] == "book") == bool(free) == (case["stratum"] != "full"), case["id"]
    assert sum(case["stratum"] == "full" for case in CASES) == 4


def test_the_recording_covers_every_first_turn(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _agent(tmp_path, monkeypatch)
    directive = strict_response_format("Turn", module.Turn.model_json_schema())
    asked = [
        RecordedPrompt(module.SYSTEM, f"Request: {case['ask']}", directive)
        for case in CASES
        for _ in range(module.booking.k)
    ]
    RecordedCompletions.check_covers(
        RECORDING, asked, settings=ClientRequestSettings(max_tokens=4096, reasoning_effort="low")
    )


async def test_offline_it_replays_the_loop_and_grades_the_day_each_cell_left(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _agent(tmp_path, monkeypatch)
    served = count_calls(module.model, monkeypatch)

    result = await module.booking.run(store=":memory:")

    assert len(served.calls) == recorded_spend(RECORDING)[0], "every recorded turn is replayed, and no other"
    assert page_violations(result.report) == []
    summary, host = result.summary, result.host
    assert summary.status == "completed" and (summary.n_cases, summary.k_runs, summary.n_excluded) == (12, 3, 0)
    assert summary.errors == []
    # The did-nothing control: an agent that did nothing would pass the first check in the 4 full-day cases only,
    # and the other two in every case, so their pass rates are printed as no measurement at all.
    assert [(goal.proof, goal.did_nothing_passed, goal.did_nothing_cases) for goal in summary.goal_checks] == [
        ("unproven", 4, 12),
        ("unproven", 12, 12),
        ("unproven", 12, 12),
    ]
    lines = [line for line in summary.render().splitlines() if "goal check" in line]
    assert "NOT A MEASUREMENT" not in lines[0] and all("NOT A MEASUREMENT" in line for line in lines[1:])
    # Each cell, read back: the checks graded the day it left and the calls it made.
    cases = {case.id: case for case in off_loop(host.storage.query_test_cases, summary.scope_id)}
    for cell in off_loop(list_results, host.storage, summary.run_id, summary.scope_id):
        trace = off_loop(get_result_trace, host.storage, cell)
        assert trace is not None and trace.call_ledger is not None and trace.end_state is not None
        seed, params = cases[cell.test_case_id].host_payload["seed"], cases[cell.test_case_id].variation_params
        booked = trace.end_state["booked"]
        assert trace.end_state["busy"] == seed["busy"], "no tool frees a taken hour"
        assert booked == [call.params["hour"] for call in trace.call_ledger.calls if call.action == "book"]
        fits, clear, once = (outcome.passed for outcome in cell.goal_state_outcomes)
        assert fits == (bool(booked) == (params["expect"] == "book"))
        assert clear == (not set(booked) & set(seed["busy"]))
        assert once == (len(booked) <= 1)


async def test_the_loop_tells_the_model_a_refused_call_and_acts_on_the_next(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _agent(tmp_path, monkeypatch)
    turns = [
        module.Turn(tool="find_free", hour=0, reply=""),
        module.Turn(tool="book", hour=20, reply=""),  # outside 9 to 16: the world refuses it
        module.Turn(tool="book", hour=11, reply=""),
        module.Turn(tool="done", hour=0, reply="Booked 11:00."),
    ]
    asked: list[str] = []

    async def scripted(user: str, schema: type[Any], *, system: str | None = None) -> Any:
        asked.append(user)
        return turns[(len(asked) - 1) % len(turns)]

    monkeypatch.setattr(module.model, "json", scripted)
    monkeypatch.setattr(module.booking, "cases", [case for case in CASES if case["id"] == "r01"])  # 9 and 11 taken
    monkeypatch.setattr(module.booking, "k", 1)

    result = await module.booking.run(store=":memory:")

    assert asked[0] == f"Request: {module.booking.cases[0]['ask']}"
    assert asked[1].endswith("find_free(hour=0) said: [10, 12, 13, 14, 15, 16]")
    refused = asked[2].rsplit("\n\n", 1)[1]
    assert refused.startswith("book(hour=20) said: refused: ") and "20" in refused
    assert asked[3].endswith("book(hour=11) said: booked 11:00")
    # Only the calls the world made are recorded, and the agent booked a taken hour: the second check catches it.
    summary, storage = result.summary, result.host.storage
    (cell,) = off_loop(list_results, storage, summary.run_id, summary.scope_id)
    trace = off_loop(get_result_trace, storage, cell)
    assert trace is not None and trace.call_ledger is not None and trace.end_state is not None
    assert trace.call_ledger.calls == [
        RecordedCall(tool="day", action="find_free", params={}),
        RecordedCall(tool="day", action="book", params={"hour": 11}),
    ]
    assert trace.end_state["booked"] == [11]
    assert [outcome.passed for outcome in cell.goal_state_outcomes] == [True, False, True]


async def test_the_booking_world_passes_the_engine_s_conformance_kit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    booking = _agent(tmp_path, monkeypatch).booking
    world = booking.world
    assert world is not None and world.registry is not None
    report = await check_world_conformance(world.registry, expressions=booking.plan.goal_checks)
    assert [(result.check, result.dimension, result.detail) for result in report.failures] == []
    assert {result.dimension for result in report.proved if result.dimension} == {"busy", "booked"}


def test_serve_hosts_the_runs_on_the_command_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _agent(tmp_path, monkeypatch)
    result = asyncio.run(module.booking.run())  # the default store, which serve.py's host reads
    capsys.readouterr()

    assert run_cli(["ls", "--host", "serve:host", "--scope", "booking"]) == 0
    out = capsys.readouterr().out
    assert result.campaign_id in out and result.run_id in out


async def test_serve_mounts_the_read_actions_over_mcp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fastmcp = pytest.importorskip("fastmcp")
    module = _agent(tmp_path, monkeypatch)
    result = await module.booking.run()
    servers: list[Any] = []
    monkeypatch.setattr(fastmcp.FastMCP, "run", lambda self, *args, **kwargs: servers.append(self))

    runpy.run_path("serve.py", run_name="__main__")

    (server,) = servers
    async with fastmcp.Client(server) as client:
        assert [tool.name for tool in await client.list_tools()] == ["evals"]
        listed = await client.call_tool("evals", {"action": "campaigns_list"})
    assert result.campaign_id in str(listed.content)


def test_the_docstring_states_what_the_recording_cost() -> None:
    calls, dollars = recorded_spend(RECORDING)
    assert stated_spend(AGENT) == (calls, round(dollars, 4))


def test_the_recording_holds_no_key_material() -> None:
    assert secret_markers(RECORDING) == []


def test_the_agent_and_its_host_stay_within_100_lines() -> None:
    assert non_blank_lines(AGENT) + non_blank_lines(SERVE) <= 100


def test_the_rung_reaches_the_engine_only_through_public_roots() -> None:
    files = [(path.name, path) for path in (AGENT, SERVE)]
    assert public_root_violations(SOURCE_ROOT, files, consumer_root=REPO_ROOT) == []
