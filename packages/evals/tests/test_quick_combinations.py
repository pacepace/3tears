"""The quick layer's additions, used together: each was built and tested on its own branch.

``Answer`` (a runner's spend), a world of lookups (tools), ``k=v`` arm names (a second lever) and a world with state
each have their own tests. These pin that they compose: a tool-using runner can report its spend, and a comparison
runs world arms and factorial tool-using arms.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, Field

from threetears.evals import Answer, Eval, World
from threetears.evals.analysis import EvalSummary
from threetears.evals.quick.world import WorldOutcome

SCOPE = "quick-combinations-tests"
CASES = [{"n": 1}, {"n": 2}, {"n": 3}]


def double(n: int) -> int:
    """Double a number."""
    return 2 * n


#: The world of lookups every tool-using runner here calls.
DOUBLING = World(tools=[double])


def exact(case: Mapping[str, Any], out: WorldOutcome) -> bool:
    return bool(out.answer == 2 * case["n"])


def mean_exact(summary: EvalSummary) -> float | None:
    (measure,) = [measure for measure in summary.measures if measure.name == "exact"]
    return measure.mean


async def test_a_tool_using_runner_reports_its_spend_with_an_answer() -> None:
    async def doubling(case: Mapping[str, Any], tools: Mapping[str, Any]) -> Answer:
        return Answer(await tools["double"](n=case["n"]), model="m", input_tokens=10, output_tokens=2, cost_usd=0.001)

    summary = (await Eval(CASES, DOUBLING(doubling), [exact], k=1).run(store=":memory:", scope=SCOPE)).summary
    assert mean_exact(summary) == 1
    assert summary.role_spend("candidate").calls == 3
    assert summary.role_spend("candidate").cost_usd == 0.003


async def test_a_comparison_runs_tool_using_arms_on_two_factors() -> None:
    def arm(offset: int) -> Any:
        async def doubling(case: Mapping[str, Any], tools: Mapping[str, Any]) -> int:
            return int(await tools["double"](n=case["n"])) + offset

        return DOUBLING(doubling)

    comparison = await Eval(
        CASES,
        {"model=m, variant=right": arm(0), "model=m, variant=off-by-one": arm(1)},
        [exact],
        k=1,
    ).run(store=":memory:", scope=SCOPE)
    assert mean_exact(comparison.arms["model=m, variant=right"]) == 1
    assert mean_exact(comparison.arms["model=m, variant=off-by-one"]) == 0


class Room(BaseModel):
    """A room with a lamp."""

    lamp: Literal["on", "off"] = Field(description="The lamp.")
    dark: bool = Field(description="Night?")


def switch(room: Room, to: Literal["on", "off"]) -> str:
    """Switch the lamp."""
    room.lamp = to
    return f"lamp {to}"


async def test_a_comparison_runs_world_arms_and_grades_each_ones_end_state() -> None:
    room = World(Room, tools=[switch])
    cases = [
        {"dark": True, "world": {"lamp": "off", "dark": True}},
        {"dark": False, "world": {"lamp": "off", "dark": False}},
    ]

    async def sensible(case: Mapping[str, Any], tools: Mapping[str, Any]) -> str:
        return str(await tools["switch"](to="on" if case["dark"] else "off"))

    async def always_on(case: Mapping[str, Any], tools: Mapping[str, Any]) -> str:
        return str(await tools["switch"](to="on"))

    comparison = await Eval(
        cases,
        {"sensible": room(sensible), "always_on": room(always_on)},
        ['(state.lamp == "on") == state.dark'],
        k=1,
    ).run(store=":memory:", scope=SCOPE)
    assert [(goal.passed, goal.n) for goal in comparison.arms["sensible"].goal_checks] == [(2, 2)]
    assert [(goal.passed, goal.n) for goal in comparison.arms["always_on"].goal_checks] == [(1, 2)]
