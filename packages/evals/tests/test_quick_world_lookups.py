"""A world of lookups: an agent's tools, called live, recorded to a file, and replayed from it.

``World(tools=[...])(agent)`` hands the agent the case and its tools. With no recording every call is live. With
``recording=``, a file that is not there is recorded — every call live, its first answer kept — and a file that is
there is replayed, the tool never called; ``RECORD=1`` records it again. An ask the recording does not hold, or holds
fewer times, excludes the cell as the rig's failure — even when the agent catches the miss itself. Graders read the
agent's answer, and the calls it made, on the :class:`~threetears.evals.quick.world.WorldOutcome`. Every arm of a
comparison is served the same recording.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from threetears.evals import Eval, World
from threetears.evals.quick.tools import ToolRecordingMiss

CASES = [{"n": 1}, {"n": 2}, {"n": 3}]


@pytest.fixture(autouse=True)
def _replay_unless_told(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with RECORD unset: a recording that is there is replayed."""
    monkeypatch.delenv("RECORD", raising=False)


class Dice:
    """A tool whose live answer differs on every call, counting how often it was really called."""

    def __init__(self) -> None:
        self._rolls = itertools.count(100)
        self.calls = 0

    def roll(self, sides: int) -> int:
        """Roll a die with this many sides."""
        self.calls += 1
        return next(self._rolls) % sides + 1


async def adds_a_roll(case: Mapping[str, Any], tools: Mapping[str, Any]) -> int:
    """Add a d20 roll to the case's number."""
    return int(case["n"]) + await tools["roll"](sides=20)


def total(case: Mapping[str, Any], out: Any) -> float:
    """The answer, as a number."""
    return float(out.answer)


async def test_an_agent_is_handed_its_tools_live_with_no_recording_sync_and_async_alike() -> None:
    async def lookup(key: str) -> dict[str, Any]:
        """Look a key up."""
        return {"key": key, "found": True}

    async def both(case: Mapping[str, Any], tools: Mapping[str, Any]) -> int:
        """Look up, then roll."""
        found = await tools["lookup"](key=str(case["n"]))
        return int(found["key"]) * 10 + await tools["roll"](sides=6)

    dice = Dice()
    world = World(tools=[dice.roll, lookup])
    summary = (await Eval(CASES, world(both), [total], k=1).run(store=":memory:")).summary
    assert summary.status == "completed" and summary.n_scored == 3 and dice.calls == 3
    # Keys 1, 2, 3 times ten, plus rolls of 100, 101, 102 on a d6: 5, 6, 1.
    assert summary.measures[0].mean == pytest.approx((15 + 26 + 31) / 3)


async def test_a_grader_reads_the_calls_the_agent_made() -> None:
    def rolled_a_d20(case: Mapping[str, Any], out: Any) -> bool:
        """The agent rolled a d20 once."""
        return [(call.action, call.params) for call in out.calls] == [("roll", {"sides": 20})]

    world = World(tools=[Dice().roll])
    summary = (await Eval(CASES, world(adds_a_roll), [rolled_a_d20], k=1).run(store=":memory:")).summary
    assert summary.measures[0].mean == 1


async def test_the_first_run_records_and_a_replay_serves_the_recording_and_never_calls_the_live_tool(
    tmp_path: Path,
) -> None:
    dice = Dice()
    world = World(tools=[dice.roll], recording=tmp_path / "rolls.jsonl")
    recorded = await Eval(CASES, world(adds_a_roll), [total], k=1).run(store=":memory:")
    assert recorded.summary.status == "completed" and dice.calls == 3
    assert any("recorded to" in line for line in recorded.disclosures)
    captured = recorded.summary.measures[0].mean

    for attempt in (1, 2):
        replay = await Eval(CASES, {f"replay-{attempt}": world(adds_a_roll)}, [total], k=2).run(store=":memory:")
        assert dice.calls == 3, "a replay called the live tool"
        assert (replay.summary.status, replay.summary.n_scored, replay.summary.n_excluded) == ("completed", 6, 0)
        assert replay.summary.measures[0].mean == captured
        assert any("replayed from" in line for line in replay.disclosures)


async def test_record_set_to_one_records_the_file_again_live(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dice = Dice()
    world = World(tools=[dice.roll], recording=tmp_path / "rolls.jsonl")
    first = await Eval(CASES, world(adds_a_roll), [total], k=1).run(store=":memory:")
    monkeypatch.setenv("RECORD", "1")
    again = await Eval(CASES, world(adds_a_roll), [total], k=1).run(store=":memory:")
    assert dice.calls == 6
    assert again.summary.measures[0].mean != first.summary.measures[0].mean
    monkeypatch.delenv("RECORD")
    replay = await Eval(CASES, world(adds_a_roll), [total], k=1).run(store=":memory:")
    assert dice.calls == 6 and replay.summary.measures[0].mean == again.summary.measures[0].mean


async def test_an_ask_the_recording_does_not_hold_excludes_the_cell_as_the_rigs_failure_and_stays_offline(
    tmp_path: Path,
) -> None:
    dice = Dice()
    world = World(tools=[dice.roll], recording=tmp_path / "rolls.jsonl")
    await Eval(CASES, world(adds_a_roll), [total], k=1).run(store=":memory:")

    async def rolls_bigger_on_two(case: Mapping[str, Any], tools: Mapping[str, Any]) -> int:
        """Roll a d100 on two."""
        return int(case["n"]) + await tools["roll"](sides=100 if case["n"] == 2 else 20)

    replay = (await Eval(CASES, world(rolls_bigger_on_two), [total], k=1).run(store=":memory:")).summary
    assert dice.calls == 3
    assert (replay.status, replay.n_scored, replay.n_candidate_failed, replay.n_excluded) == ("completed", 2, 0, 1)
    assert len(replay.errors) == 1 and "ToolRecordingMiss" in replay.errors[0]


async def test_a_miss_the_agent_swallows_still_excludes_the_cell(tmp_path: Path) -> None:
    dice = Dice()
    world = World(tools=[dice.roll], recording=tmp_path / "rolls.jsonl")
    await Eval(CASES, world(adds_a_roll), [total], k=1).run(store=":memory:")

    async def shrugs_off_failures(case: Mapping[str, Any], tools: Mapping[str, Any]) -> int:
        """Roll a d12, and answer 0 on any failure."""
        try:
            return int(case["n"]) + await tools["roll"](sides=12)  # never recorded
        except Exception:  # noqa: BLE001 — the point: an agent that absorbs every tool failure
            return 0

    replay = (await Eval(CASES, world(shrugs_off_failures), [total], k=1).run(store=":memory:")).summary
    # Every cell missed, so the rig measured nothing: the run fails, and no cell is scored as the agent's 0.
    assert dice.calls == 3
    assert (replay.status, replay.n_scored, replay.n_candidate_failed, replay.n_excluded) == ("failed", 0, 0, 3)


async def test_asking_more_often_than_the_recording_did_is_a_miss_not_a_neighbours_answer(tmp_path: Path) -> None:
    dice = Dice()
    world = World(tools=[dice.roll], recording=tmp_path / "rolls.jsonl")
    await Eval(CASES, world(adds_a_roll), [total], k=1).run(store=":memory:")

    async def rolls_twice_on_three(case: Mapping[str, Any], tools: Mapping[str, Any]) -> int:
        """Roll once, and twice on three."""
        answer = await adds_a_roll(case, tools)
        return answer + (await tools["roll"](sides=20) if case["n"] == 3 else 0)

    replay = (await Eval(CASES, world(rolls_twice_on_three), [total], k=1).run(store=":memory:")).summary
    assert dice.calls == 3
    assert (replay.n_scored, replay.n_excluded) == (2, 1)
    assert "ToolRecordingMiss" in replay.errors[0] and "asked 2 time(s)" in replay.errors[0]


async def test_a_recording_with_a_line_it_cannot_read_is_refused_before_anything_runs(tmp_path: Path) -> None:
    (tmp_path / "rolls.jsonl").write_text("not a line a recording writes\n", encoding="utf-8")
    dice = Dice()
    world = World(tools=[dice.roll], recording=tmp_path / "rolls.jsonl")
    with pytest.raises(ToolRecordingMiss, match="line 1 is not a line a tool recording writes"):
        await Eval(CASES, world(adds_a_roll), [total], k=1).run(store=":memory:")
    assert dice.calls == 0


async def test_a_tool_answer_no_recording_can_hold_excludes_the_cell() -> None:
    def odd(case_n: int) -> object:
        """Answer something JSON cannot hold."""
        return object()

    async def asks(case: Mapping[str, Any], tools: Mapping[str, Any]) -> int:
        """Ask the odd tool."""
        await tools["odd"](case_n=case["n"])
        return 1

    summary = (await Eval(CASES[:1], World(tools=[odd])(asks), [total], k=1).run(store=":memory:")).summary
    assert (summary.n_scored, summary.n_candidate_failed, summary.n_excluded) == (0, 0, 1)
    assert "which a recording cannot keep" in summary.errors[0]


async def test_a_tool_that_raises_is_the_agents_to_handle() -> None:
    def broken() -> int:
        """Time out."""
        raise TimeoutError("the service timed out")

    async def asks(case: Mapping[str, Any], tools: Mapping[str, Any]) -> int:
        """Ask the broken tool."""
        return int(await tools["broken"]())

    summary = (await Eval(CASES[:1], World(tools=[broken])(asks), [total], k=1).run(store=":memory:")).summary
    assert (summary.n_scored, summary.n_candidate_failed, summary.n_excluded) == (0, 1, 0)
    assert "TimeoutError: the service timed out" in summary.errors[0]


def _untyped(sides):
    """A tool whose parameter has no type."""
    return sides


@pytest.mark.parametrize(
    ("tools", "said"),
    [
        ([], "a world with no state is its tools"),
        ([3], "write it as a def"),
        ([lambda sides: sides], "write it as a def"),
        ([Dice().roll, Dice().roll], "tools named roll more than once"),
        ([_untyped], "has no type"),
    ],
    ids=["none", "not a function", "a lambda", "two of one name", "an untyped parameter"],
)
def test_tools_no_agent_could_be_handed_are_refused(tools: Any, said: str) -> None:
    with pytest.raises(ValueError, match=said):
        World(tools=tools)


async def test_a_synchronous_agent_handed_a_worlds_tools_is_refused_before_anything_runs() -> None:
    def sync_agent(case: Mapping[str, Any], tools: Mapping[str, Any]) -> int:
        """Cannot await its tools."""
        return 1

    with pytest.raises(ValueError, match="must be one: its tools are async functions"):
        await Eval(CASES, World(tools=[Dice().roll])(sync_agent), [total], k=1).run(store=":memory:")


async def test_every_arm_of_a_comparison_is_served_one_recording(tmp_path: Path) -> None:
    dice = Dice()
    world = World(tools=[dice.roll], recording=tmp_path / "rolls.jsonl")
    await Eval(CASES, world(adds_a_roll), [total], k=1).run(store=":memory:")
    rolled: dict[str, list[int]] = {}

    def arm(name: str, offset: int) -> Any:
        async def agent(case: Mapping[str, Any], tools: Mapping[str, Any]) -> int:
            """Roll a d20 and add an offset."""
            roll = await tools["roll"](sides=20)
            rolled.setdefault(name, []).append(roll)
            return int(case["n"]) + roll + offset

        return agent

    result = await Eval(CASES, {"base": world(arm("base", 0)), "plus": world(arm("plus", 1))}, [total], k=1).run(
        store=":memory:"
    )
    assert dice.calls == 3
    assert sorted(rolled["base"]) == sorted(rolled["plus"])
    means = {name: summary.measures[0].mean for name, summary in result.arms.items()}
    assert means["plus"] == pytest.approx(means["base"] + 1)
