"""``examples/world.py`` runs as a newcomer would run it, with no API key, and its Claude loop reads real SDK replies.

Offline, the candidate is the example's scripted stand-in, so the run is deterministic: each case's room
is seeded, the stand-in switches the light through the tool, the engine reads each room back and the goal
checks grade it — passing everywhere but the one mistake the stand-in is written to make. The Claude
candidate is exercised against replies built from the SDK's own types, so a field the SDK renames turns
this red rather than the first live run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from threetears.evals.contracts import RecordedCall
from threetears.evals.quick import EvalSummary, callable_host
from threetears.evals.run import get_result_trace, list_results
from packages.evals.tests.test_package_matrix import REPO_ROOT, SOURCE_ROOT, public_root_violations

#: The example under test.
WORLD_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "world.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("world_example", WORLD_EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_with_no_api_key_the_example_runs_offline_and_says_so(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    module = _load()
    summary = await module.main()
    assert isinstance(summary, EvalSummary)
    assert (summary.status, summary.candidate_model) == ("completed", "offline")
    assert (summary.n_cases, summary.k_runs, summary.n_results, summary.n_scored) == (4, 2, 8, 8)
    lit_iff_dark, never_needless = module.LIGHT_ENDS_ON_IFF_DARK, module.NEVER_SWITCHED_NEEDLESSLY
    # The stand-in lights the bright room it is asked to read in: that case's two cells fail the first check.
    assert [(goal.check, goal.passed, goal.n) for goal in summary.goal_checks] == [
        (lit_iff_dark, 6, 8),
        (never_needless, 8, 8),
    ]
    assert summary.errors == []
    out = capsys.readouterr().out
    assert out.startswith("ANTHROPIC_API_KEY is not set: running OFFLINE")
    assert summary.render() in out
    # Then each cell, in case order: where the room started, what was called, where it ended, what failed.
    cells = [line.split() for line in out.split(summary.render())[1].strip().splitlines()]
    assert [" ".join(cell) for cell in cells[::2]] == [
        "dark, light off -> switch_light(to=on) -> light on ok",
        "dark, light on -> no call -> light on ok",
        "bright, light off -> switch_light(to=on) -> light on wrong light",
        "bright, light on -> switch_light(to=off) -> light off ok",
    ]
    assert cells[::2] == cells[1::2]  # the stand-in does the same thing on both repeats


async def test_each_room_is_seeded_from_its_case_and_read_back_as_the_stand_in_left_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    host = _host_of(module, monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    summary = await module.main()
    cases = {case.id: case for case in host.storage.query_test_cases(summary.scope_id)}
    seen = []
    for result in list_results(host.storage, summary.run_id, summary.scope_id):
        trace = get_result_trace(host.storage, result)
        assert trace is not None and trace.call_ledger is not None and trace.end_state is not None
        case = cases[result.test_case_id].host_payload
        seen.append(
            (
                case["seed"],
                trace.end_state,
                trace.call_ledger.calls,
                [outcome.passed for outcome in result.goal_state_outcomes],
            )
        )
    on = [RecordedCall(tool="room", action="switch_light", params={"to": "on"})]
    off = [RecordedCall(tool="room", action="switch_light", params={"to": "off"})]
    dark_off, dark_on = {"light": "off", "daylight": "dark"}, {"light": "on", "daylight": "dark"}
    bright_off, bright_on = {"light": "off", "daylight": "bright"}, {"light": "on", "daylight": "bright"}
    expected = [
        (dark_off, dark_on, on, [True, True]),  # dark, about to read: switched on
        (dark_on, dark_on, [], [True, True]),  # already right: left alone, no call
        (bright_off, bright_on, on, [False, True]),  # the deliberate mistake: lit a bright room for reading
        (bright_on, bright_off, off, [True, True]),  # bright: switched off
    ]
    assert sorted(seen, key=str) == sorted(expected * 2, key=str)


def _host_of(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The host the example will build and run in, made here so the test can read the cells back from it."""
    host = callable_host(world=module.ROOM)
    monkeypatch.setattr(module, "callable_host", lambda **kwargs: host)
    return host


def test_the_example_reaches_the_engine_only_through_public_roots() -> None:
    assert public_root_violations(SOURCE_ROOT, [("world.py", WORLD_EXAMPLE)], consumer_root=REPO_ROOT) == []


async def test_the_claude_candidate_runs_the_tool_loop_against_sdk_replies(monkeypatch: pytest.MonkeyPatch) -> None:
    anthropic = pytest.importorskip("anthropic")
    module = _load()
    # The id is the example's own, read from it, so a model rev in the example is the only edit.
    model = module.MODEL
    usage = {"input_tokens": 300, "output_tokens": 40}

    def reply(stop_reason: str, content: list[dict[str, Any]]) -> Any:
        return anthropic.types.Message.model_validate(
            {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": content,
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": usage,
            }
        )

    replies = [
        # First a call the world refuses, then the right one, then the answer.
        reply("tool_use", [{"type": "tool_use", "id": "t1", "name": "switch_light", "input": {"to": "dim"}}]),
        reply("tool_use", [{"type": "tool_use", "id": "t2", "name": "switch_light", "input": {"to": "on"}}]),
        reply("end_turn", [{"type": "text", "text": "Lamp on for reading."}]),
    ]
    sent: list[dict[str, Any]] = []

    async def create(**request: Any) -> Any:
        sent.append({**request, "messages": list(request["messages"])})
        return replies[(len(sent) - 1) % len(replies)]

    class Client:
        messages = SimpleNamespace(create=create)

        async def __aenter__(self) -> Client:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(anthropic, "AsyncAnthropic", Client)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    host = _host_of(module, monkeypatch)
    monkeypatch.setattr(module, "CASES", module.CASES[:1])  # dark, light off, about to read
    summary = await module.main()

    assert len(sent) == 3 * summary.k_runs
    first = sent[0]
    assert first["model"] == model and first["tools"][0]["name"] == "switch_light"
    assert first["tools"][0]["input_schema"]["properties"] == {"to": {"enum": ["on", "off"]}}
    assert "'daylight': 'dark'" in first["messages"][0]["content"]
    refused = sent[1]["messages"][-1]["content"][0]
    assert refused["tool_use_id"] == "t1" and refused["is_error"] is True and "dim" in refused["content"]
    accepted = sent[2]["messages"][-1]["content"][0]
    assert (accepted["tool_use_id"], accepted["is_error"], accepted["content"]) == ("t2", False, "The light is now on.")
    # Only the call the world made is recorded, and the room ends lit: both checks pass.
    for result in list_results(host.storage, summary.run_id, summary.scope_id):
        trace = get_result_trace(host.storage, result)
        assert trace is not None and trace.call_ledger is not None
        assert trace.call_ledger.calls == [RecordedCall(tool="room", action="switch_light", params={"to": "on"})]
        assert trace.end_state == {"light": "on", "daylight": "dark"}
    assert [(goal.passed, goal.n) for goal in summary.goal_checks] == [(2, 2), (2, 2)]
