"""``examples/world.py`` runs as a newcomer would run it, with no API key, and its tool loop drives the world.

With no key the example replays its recording: each case's room is seeded, Claude's recorded turns switch the light
through the tool, the engine reads each room back, and the goal checks grade it. Nothing here assumes what Claude
did: each cell is read back from the store and held to the checks, and the line the example prints for it. The loop
itself is driven by a scripted client below, so a refused call and the model being told why are covered whatever
the recording holds.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from threetears.evals.analysis import EvalSummary
from threetears.evals.quick import callable_host
from threetears.evals.run import get_result_trace, list_results
from threetears.evals.schema import RecordedCall
from threetears.evals.testing import RecordedPrompt
from packages.evals.tests.example_loader import check_covers, load_example, replaying
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.test_package_matrix import REPO_ROOT, SOURCE_ROOT, public_root_violations

#: The example under test.
WORLD_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "world.py"


def _load() -> ModuleType:
    return load_example(WORLD_EXAMPLE.name)


async def test_with_no_api_key_the_example_replays_its_recording_and_says_so(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load()
    first_turns = [
        RecordedPrompt(module.SYSTEM, f"Sensors: {_room(case)}\n\n{case['ask']}", module.TURN)
        for case in module.CASES
        for _ in range(2)
    ]
    check_covers(module, first_turns, settings=module.SETTINGS)
    replay = replaying(module, monkeypatch)
    host = _host_of(module, monkeypatch)

    summary = await module.main()

    assert 8 <= replay.served() <= 32, "each of 8 cells takes one to four turns"
    assert isinstance(summary, EvalSummary)
    assert (summary.status, summary.candidate_model) == ("completed", module.MODEL)
    assert (summary.n_cases, summary.k_runs, summary.n_results, summary.n_scored) == (4, 2, 8, 8)
    assert summary.errors == []
    assert [(goal.check, goal.n) for goal in summary.goal_checks] == [(check, 8) for check in module.GOAL_CHECKS]
    # Neither check is proven (a quick run names no control), and a candidate that did nothing passes the second in
    # every case: its pass count is printed as no measurement, never as an unqualified pass.
    assert [(goal.proof, goal.did_nothing_cases) for goal in summary.goal_checks] == [("unproven", 4), ("unproven", 4)]
    goal_lines = [line for line in summary.render().splitlines() if "goal check" in line]
    assert all(" — " in line for line in goal_lines), "an unproven check's pass count is never printed bare"
    assert "NOT A MEASUREMENT" in goal_lines[1]

    out = capsys.readouterr().out
    assert out.startswith("ANTHROPIC_API_KEY is not set: replaying recordings/world.jsonl, Claude (")
    assert summary.render() in out
    # Each cell, read back: the checks graded the room it left and the calls it made, and its line says so.
    cases = {case.id: case for case in off_loop(host.storage.query_test_cases, summary.scope_id)}
    for result in off_loop(list_results, host.storage, summary.run_id, summary.scope_id):
        trace = off_loop(get_result_trace, host.storage, result)
        assert trace is not None and trace.call_ledger is not None and trace.end_state is not None
        seed = cases[result.test_case_id].host_payload["seed"]
        lit_iff_dark, never_needless = (outcome.passed for outcome in result.goal_state_outcomes)
        assert lit_iff_dark == ((trace.end_state["light"] == "on") == (seed["daylight"] == "dark"))
        assert never_needless == all(call.params["to"] != seed["light"] for call in trace.call_ledger.calls)
        assert trace.end_state["daylight"] == seed["daylight"], "no tool changes the daylight"
        calls = ", ".join(f"{call.action}(to={call.params['to']})" for call in trace.call_ledger.calls) or "no call"
        assert f"{seed['daylight']}, light {seed['light']}" in out and calls in out


def _room(case: dict[str, str]) -> dict[str, str]:
    """What the room's sensors show at the start of ``case``: its seed, as the example seeds it."""
    return {"light": case["light"], "daylight": case["daylight"]}


async def test_the_loop_tells_the_model_a_refused_call_and_acts_on_the_next(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load()
    turns = [
        {"tool": "switch_light", "arguments": json.dumps({"to": "dim"}), "reply": ""},  # the world refuses it
        {"tool": "switch_light", "arguments": json.dumps({"to": "on"}), "reply": ""},
        {"tool": "none", "arguments": "{}", "reply": "Lamp on for reading."},
    ]
    sent: list[dict[str, Any]] = []

    async def generate(*, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
        sent.append({"system": system, "user": user, "response_format": response_format})
        return SimpleNamespace(content=json.dumps(turns[(len(sent) - 1) % len(turns)]))

    async def aclose() -> None:
        return None

    monkeypatch.setattr(module, "client", lambda model, settings: SimpleNamespace(generate=generate, aclose=aclose))
    monkeypatch.setattr(module, "how_it_runs", lambda what: f"scripted ({what})")
    host = _host_of(module, monkeypatch)
    monkeypatch.setattr(module, "CASES", module.CASES[:1])  # dark, light off, about to read
    summary = await module.main()

    assert len(sent) == 3 * summary.k_runs
    first = sent[0]
    assert first["system"] == module.SYSTEM and first["response_format"] == module.TURN
    assert "switch_light: Turn the room's light on or off." in first["system"]
    assert first["user"] == f"Sensors: {_room(module.CASES[0])}\n\n{module.CASES[0]['ask']}"
    refused = sent[1]["user"].rsplit("\n\n", 1)[1]
    assert refused.startswith('You called switch_light with {"to": "dim"}: refused: ')
    assert "dim" in refused.split(": refused: ", 1)[1], "the model is told why the world refused it"
    accepted = sent[2]["user"].rsplit("\n\n", 1)[1]
    assert accepted == 'You called switch_light with {"to": "on"}: The light is now on.'
    # Only the call the world made is recorded, and the room ends lit: both checks pass.
    for result in off_loop(list_results, host.storage, summary.run_id, summary.scope_id):
        trace = off_loop(get_result_trace, host.storage, result)
        assert trace is not None and trace.call_ledger is not None
        assert trace.call_ledger.calls == [RecordedCall(tool="room", action="switch_light", params={"to": "on"})]
        assert trace.end_state == {"light": "on", "daylight": "dark"}
    assert [(goal.passed, goal.n) for goal in summary.goal_checks] == [(2, 2), (2, 2)]


def _host_of(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The host the example will build and run in, made here so the test can read the cells back from it."""
    host = callable_host(world=module.ROOM)
    monkeypatch.setattr(module, "callable_host", lambda **kwargs: host)
    return host


def test_the_example_reaches_the_engine_only_through_public_roots() -> None:
    assert public_root_violations(SOURCE_ROOT, [("world.py", WORLD_EXAMPLE)], consumer_root=REPO_ROOT) == []
