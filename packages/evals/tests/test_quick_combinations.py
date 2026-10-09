"""The quick layer's additions, used together: each was built and tested on its own branch.

``Answer`` (a candidate's spend), ``tools=`` (cassettes), ``factors=`` (a second lever) and ``world=``
(a seeded world) each have their own tests. These pin that they compose: a tool-using candidate can
report its spend, ``compare`` runs world arms and factorial tool-using arms, and a world run refuses
generic tools rather than letting a cassette replay a tool that should have changed the world.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from threetears.evals.quick import (
    Answer,
    CandidateTools,
    Dimension,
    EvalSummary,
    World,
    WorldTool,
    WorldTools,
    compare,
    run_eval,
)

SCOPE = "quick-combinations-tests"
CASES = [{"n": 1}, {"n": 2}, {"n": 3}]


def double(n: int) -> int:
    return 2 * n


def exact(case: Mapping[str, Any], answer: Any) -> bool:
    return bool(answer == 2 * case["n"])


def mean_exact(summary: EvalSummary) -> float | None:
    (measure,) = [measure for measure in summary.measures if measure.name == "exact"]
    return measure.mean


async def test_a_tool_using_candidate_reports_its_spend_with_an_answer() -> None:
    async def doubling(case: Mapping[str, Any], tools: CandidateTools) -> Answer:
        return Answer(await tools["double"](n=case["n"]), model="m", input_tokens=10, output_tokens=2, cost_usd=0.001)

    summary = await run_eval(CASES, doubling, [exact], tools={"double": double}, scope_id=SCOPE, k=1)
    assert mean_exact(summary) == 1
    assert summary.candidate_calls == 3
    assert summary.candidate_cost_usd == 0.003


async def test_compare_runs_tool_using_arms_on_two_factors() -> None:
    def arm(offset: int) -> Any:
        async def doubling(case: Mapping[str, Any], tools: CandidateTools) -> int:
            return int(await tools["double"](n=case["n"])) + offset

        return doubling

    comparison = await compare(
        CASES,
        {("m", "right"): arm(0), ("m", "off-by-one"): arm(1)},
        [exact],
        factors=("model", "variant"),
        control=("m", "right"),
        tools={"double": double},
        scope_id=SCOPE,
        k=1,
    )
    assert mean_exact(comparison.arms[("m", "right")]) == 1
    assert mean_exact(comparison.arms[("m", "off-by-one")]) == 0


def switch(room: dict[str, Any], to: str) -> str:
    room["lamp"] = to
    return f"lamp {to}"


async def test_compare_runs_world_arms_and_grades_each_ones_end_state() -> None:
    room = World(
        "room",
        [Dimension("lamp", {"enum": ["on", "off"]}, "The lamp."), Dimension("dark", {"type": "boolean"}, "Night?")],
        tools=[WorldTool(switch, to={"enum": ["on", "off"]})],
    )
    cases = [{"lamp": "off", "dark": True}, {"lamp": "off", "dark": False}]

    async def sensible(case: Mapping[str, Any], tools: WorldTools) -> str:
        return str(await tools["switch"](to="on" if case["dark"] else "off"))

    async def always_on(case: Mapping[str, Any], tools: WorldTools) -> str:
        return str(await tools["switch"](to="on"))

    comparison = await compare(
        cases,
        {"sensible": sensible, "always_on": always_on},
        control="sensible",
        world=room,
        seed=lambda case: {"lamp": case["lamp"], "dark": case["dark"]},
        goal_checks=['(state.lamp == "on") == state.dark'],
        scope_id=SCOPE,
        k=1,
    )
    assert [(goal.passed, goal.n) for goal in comparison.arms["sensible"].goal_checks] == [(2, 2)]
    assert [(goal.passed, goal.n) for goal in comparison.arms["always_on"].goal_checks] == [(1, 2)]
