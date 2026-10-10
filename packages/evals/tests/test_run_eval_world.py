"""``run_eval(world=, seed=, goal_checks=)``: each cell seeds its world, the candidate acts on it, code grades the end.

The world example (``test_world_example.py``) is the happy path read from outside. These pin the
mechanism: the seed lands before the candidate's first turn, every cell starts from its own case's
state, the end state is read back after the last turn, a tool call that succeeds is recorded where
``calls(...)`` reads it and one the world refuses is not, the goal checks grade the result — and a
world-less call is exactly what it was.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from threetears.evals.contracts import RecordedCall
from threetears.evals.quick import Dimension, ToolRefused, World, WorldTool, WorldTools, callable_host, run_eval
from threetears.evals.run import get_result_trace, list_results, list_runs, list_templates

SCOPE = "run-eval-world-tests"


def switch(room: dict[str, Any], to: str) -> str:
    """Turn the lamp on or off."""
    room["lamp"] = to
    return f"lamp {to}"


def broken(room: dict[str, Any]) -> str:
    """A tool whose own code fails."""
    raise RuntimeError("the relay is stuck")


def room() -> World:
    return World(
        "room",
        [
            Dimension("lamp", {"enum": ["on", "off"]}, "What the candidate switches."),
            Dimension("dark", {"type": "boolean"}, "Whether the lamp is needed."),
        ],
        tools=[WorldTool(switch, to={"enum": ["on", "off"]}), WorldTool(broken)],
    )


CASES = [{"lamp": "off", "dark": True}, {"lamp": "on", "dark": False}]
LIT_IFF_DARK = '(state.lamp == "on") == state.dark'


def start(case: Mapping[str, Any]) -> Mapping[str, Any]:
    return {"lamp": case["lamp"], "dark": case["dark"]}


async def sensible(case: Mapping[str, Any], tools: WorldTools) -> str:
    seen = await tools.view()
    wanted = "on" if seen["dark"] else "off"
    return "kept" if wanted == seen["lamp"] else await tools["switch"](to=wanted)


async def stored(host: Any, run_id: str) -> list[Any]:
    """Each result with its trace, in case order."""
    pairs = [(result, get_result_trace(host.storage, result)) for result in list_results(host.storage, run_id, SCOPE)]
    return sorted(pairs, key=lambda pair: pair[0].test_case_id)


async def test_the_seed_lands_before_the_first_turn_and_every_cell_starts_from_its_own_case() -> None:
    first_sight: list[dict[str, Any]] = []

    async def meddling(case: Mapping[str, Any], tools: WorldTools) -> str:
        assert list(tools) == ["switch", "broken"]  # the mapping a tool-using candidate is handed
        first_sight.append(await tools.view())
        await tools.call("switch", to="on")  # every cell leaves the lamp on; the next must not inherit it
        return "done"

    await run_eval(CASES, meddling, world=room(), seed=start, goal_checks=[LIT_IFF_DARK], scope_id=SCOPE, k=2)
    assert sorted(map(str, first_sight)) == sorted(map(str, [start(case) for case in CASES] * 2))


async def test_the_end_state_is_read_back_after_the_last_turn_and_the_goal_check_grades_it() -> None:
    world = room()
    host = callable_host(world=world)
    summary = await run_eval(
        CASES, sensible, host=host, world=world, seed=start, goal_checks=[LIT_IFF_DARK], scope_id=SCOPE, k=1
    )
    assert (summary.status, summary.n_scored) == ("completed", 2)
    assert [(goal.check, goal.passed, goal.n) for goal in summary.goal_checks] == [(LIT_IFF_DARK, 2, 2)]
    # A quick run names no control, so the check is unproven, and its do-nothing baseline is read off each case's seed:
    # both cases start with the lamp wrong, so doing nothing passes neither.
    (goal,) = summary.goal_checks
    assert (goal.proof, goal.did_nothing_passed, goal.did_nothing_cases) == ("unproven", 0, 2)
    assert f"goal check {LIT_IFF_DARK}: passed 2/2 — unproven" in summary.render()
    for (result, trace), case in zip(await stored(host, summary.run_id), CASES, strict=True):
        assert trace is not None
        # Both cases started wrong, so the stored end state is the candidate's, never the seed.
        assert trace.end_state == {"lamp": "on" if case["dark"] else "off", "dark": case["dark"]} != start(case)
        assert [outcome.passed for outcome in result.goal_state_outcomes] == [True]
    (run,) = list_runs(host, SCOPE)
    assert run.world_placements == {"lamp": "representable", "dark": "representable"}
    assert run.goal_check_proofs == {LIT_IFF_DARK: "unproven"}


def test_a_check_unevaluable_against_a_starting_state_loses_only_its_own_baseline() -> None:
    """A check that raises at a starting state gets no do-nothing figure — never one counted as passed or failed —
    and the other checks keep theirs. The arithmetic stands in for any check that raises when graded there."""
    unevaluable = '1 / length(calls("room.switch")) > 0'
    baseline = room().did_nothing_passes([LIT_IFF_DARK, unevaluable], [(start(case), {}) for case in CASES])
    assert baseline == {LIT_IFF_DARK: 0}


async def test_a_check_doing_nothing_passes_in_every_case_never_reads_as_a_measurement() -> None:
    never_needless = 'all(it.to != variation.lamp for it in calls("room.switch"))'
    summary = await run_eval(
        CASES, sensible, world=room(), seed=start, goal_checks=[never_needless], scope_id=SCOPE, k=1
    )
    (goal,) = summary.goal_checks
    assert (goal.passed, goal.n, goal.did_nothing_passed, goal.did_nothing_cases) == (2, 2, 2, 2)
    (line,) = [line for line in summary.render().splitlines() if "goal check" in line]
    assert line.endswith(
        "passed 2/2 — NOT A MEASUREMENT: a candidate that did nothing passes it in 2 of 2 case(s), "
        "so this pass rate does not beat doing nothing"
    )


async def test_a_call_that_succeeds_is_recorded_for_calls_and_one_the_world_refuses_is_not() -> None:
    async def sloppy(case: Mapping[str, Any], tools: WorldTools) -> str:
        with pytest.raises(ToolRefused, match="dim"):
            await tools.call("switch", to="dim")
        with pytest.raises(ToolRefused, match="no tool 'shout'"):
            await tools.call("shout")
        await tools.call("switch", to="on")
        return await tools.call("switch", to="on")

    world = room()
    host = callable_host(world=world)
    once = 'call_count("room.switch") <= 1'
    summary = await run_eval(
        CASES[:1], sloppy, host=host, world=world, seed=start, goal_checks=[once], scope_id=SCOPE, k=1
    )
    ((result, trace),) = await stored(host, summary.run_id)
    assert trace is not None and trace.call_ledger is not None
    assert trace.call_ledger.calls == [RecordedCall(tool="room", action="switch", params={"to": "on"})] * 2
    assert [(outcome.expression, outcome.passed) for outcome in result.goal_state_outcomes] == [(once, False)]


async def test_a_tool_that_raises_excludes_the_cell_as_the_rigs_fault() -> None:
    async def presses(case: Mapping[str, Any], tools: WorldTools) -> str:
        return await tools.call("broken")

    summary = await run_eval(
        CASES[:1], presses, world=room(), seed=start, goal_checks=[LIT_IFF_DARK], scope_id=SCOPE, k=1
    )
    assert (summary.n_scored, summary.n_candidate_failed, summary.n_excluded) == (0, 0, 1)
    assert "the tool broken raised RuntimeError: the relay is stuck" in summary.errors[0]
    assert summary.goal_checks == []


async def test_scorers_grade_a_world_candidates_answer_beside_its_goal_checks() -> None:
    def answered(case: Mapping[str, Any], answer: Any) -> bool:
        return isinstance(answer, str) and bool(answer)

    summary = await run_eval(
        CASES, sensible, [answered], world=room(), seed=start, goal_checks=[LIT_IFF_DARK], scope_id=SCOPE, k=1
    )
    assert [(m.name, m.mean) for m in summary.measures] == [("answered", 1.0)]
    assert summary.goal_checks[0].passed == 2


async def test_a_world_runs_starting_states_and_checks_are_part_of_its_case_set() -> None:
    world = room()
    host = callable_host(world=world)

    def all_off(case: Mapping[str, Any]) -> Mapping[str, Any]:
        return {"lamp": "off", "dark": case["dark"]}

    kwargs: dict[str, Any] = {"host": host, "world": world, "scope_id": SCOPE, "k": 1}
    first = await run_eval(CASES, sensible, seed=start, goal_checks=[LIT_IFF_DARK], **kwargs)
    again = await run_eval(CASES, sensible, seed=start, goal_checks=[LIT_IFF_DARK], **kwargs)
    reseeded = await run_eval(CASES, sensible, seed=all_off, goal_checks=[LIT_IFF_DARK], **kwargs)
    rechecked = await run_eval(CASES, sensible, seed=start, goal_checks=['state.lamp == "on"'], **kwargs)
    assert first.template_id == again.template_id
    assert len({first.template_id, reseeded.template_id, rechecked.template_id}) == 3


async def test_a_world_less_call_is_unchanged_it_opens_no_world_and_keeps_no_ledger() -> None:
    async def double(case: Mapping[str, Any]) -> int:
        return int(case["n"]) * 2

    def even(case: Mapping[str, Any], answer: Any) -> bool:
        return answer % 2 == 0

    host = callable_host([even])
    summary = await run_eval([{"n": 1}], double, [even], host=host, scope_id=SCOPE, k=1)
    assert summary.goal_checks == [] and "goal check" not in summary.render()
    ((result, trace),) = await stored(host, summary.run_id)
    assert trace is not None
    assert (trace.end_state, trace.call_ledger, result.goal_state_outcomes) == (None, None, [])
    assert host.profile.world is None and list_runs(host, SCOPE)[0].world_placements == {}


# --- refusals: each made before anything is stored ------------------------------------------------


#: Stands in for the test's own world in a parametrized case, which cannot build one per test itself.
WORLD = object()


@pytest.mark.parametrize(
    ("kwargs", "said"),
    [
        ({"seed": start, "goal_checks": [LIT_IFF_DARK]}, r"pass the world they read \(world=\)"),
        ({"goal_checks": [LIT_IFF_DARK]}, r"pass the world they read \(world=\)"),
        ({"world": WORLD, "goal_checks": [LIT_IFF_DARK]}, "pass seed="),
        ({"world": WORLD, "seed": lambda case: {"lamp": "off"}, "goal_checks": [LIT_IFF_DARK]}, "gave case 0 no dark"),
        (
            {"world": WORLD, "seed": lambda case: {"lamp": "dim", "dark": True}, "goal_checks": [LIT_IFF_DARK]},
            "case 0 a starting state the world",
        ),
        (
            {"world": WORLD, "seed": lambda case: case["nope"], "goal_checks": [LIT_IFF_DARK]},
            "seed= raised on case 0: KeyError",
        ),
        ({"world": WORLD, "seed": start, "goal_checks": ["state.colour == 1"]}, "reads state.colour"),
        ({"world": WORLD, "seed": start, "goal_checks": ['call_count("room.dim") == 0']}, "names room.dim"),
        ({"world": room(), "seed": start, "goal_checks": [LIT_IFF_DARK]}, "does not declare world 'room'"),
        (
            {"world": WORLD, "seed": start, "goal_checks": [LIT_IFF_DARK], "tools": {"switch": lambda to: to}},
            "pass no tools= beside world=",
        ),
        (
            {"world": WORLD, "seed": start, "goal_checks": [LIT_IFF_DARK], "cassette_mode": "capture"},
            "no cassette_mode",
        ),
    ],
    ids=[
        "seed with no world",
        "goal checks with no world",
        "a world with no seed",
        "a seed leaving a dimension unset",
        "a seed the schema refuses",
        "a seed that raises",
        "a check reading undeclared state",
        "a check naming an unknown tool",
        "a host that does not declare the world",
        "generic tools beside a world",
        "a cassette mode on a world run",
    ],
)
async def test_an_incoherent_world_run_is_refused_before_anything_is_stored(kwargs: dict[str, Any], said: str) -> None:
    world = room()
    host = callable_host(world=world)
    if kwargs.get("world") is WORLD:
        kwargs = kwargs | {"world": world}
    with pytest.raises(ValueError, match=said):
        await run_eval(CASES, sensible, host=host, scope_id=SCOPE, **kwargs)
    assert list_templates(host.storage, SCOPE) == []


async def test_a_well_formed_quick_world_passes_the_engine_s_own_conformance_kit() -> None:
    """The kit reads the subject view one entry per surface; the quick world once returned its state bare.

    Every dimension's ``perception_ab`` then read nothing on the ``view`` surface and failed, so the engine's own
    world could not pass the kit a host is told to run.
    """
    from threetears.evals.contracts.host.world_conformance import check_world_conformance

    world = room()
    report = await check_world_conformance(world.registry, expressions=[LIT_IFF_DARK])

    failed = [
        (result.check, result.dimension, result.detail) for result in report.results if result.outcome == "failed"
    ]
    assert failed == []
    ab = {result.dimension: result.outcome for result in report.results if result.check == "perception_ab"}
    assert ab == {"lamp": "passed", "dark": "passed"}


def note(room: dict[str, Any], text: str) -> str:
    """Leave a note by the lamp."""
    room["note"] = text
    return "noted"


def noting_room() -> World:
    return World(
        "room",
        [
            Dimension("lamp", {"enum": ["on", "off"]}, "What the candidate switches."),
            Dimension("dark", {"type": "boolean"}, "Whether the lamp is needed."),
            Dimension("note", {"type": "string"}, "What the candidate wrote down."),
        ],
        tools=[WorldTool(switch, to={"enum": ["on", "off"]}), WorldTool(note, text={"type": "string"})],
    )


def noting_start(case: Mapping[str, Any]) -> Mapping[str, Any]:
    return {**start(case), "note": ""}


async def test_a_string_match_over_a_free_text_tool_parameter_is_refused_before_anything_runs() -> None:
    """Authoring refuses it, so the quick path does: a free string is what the model wrote, not structure."""
    ran: list[str] = []

    async def candidate(case: Mapping[str, Any], tools: WorldTools) -> str:
        ran.append("ran")
        return "x"

    with pytest.raises(ValueError, match=r"room.note's text is free text"):
        await run_eval(
            CASES,
            candidate,
            world=noting_room(),
            seed=noting_start,
            goal_checks=['any(it.text == "lamp fixed" for it in calls("room.note"))'],
            scope_id=SCOPE,
        )
    assert ran == [], "refused up front, never graded"


async def test_a_comparison_over_an_enum_closed_tool_parameter_is_accepted_and_graded() -> None:
    """The quick world describes its tools' parameters from their schemas, so authoring's closure rule admits it."""
    summary = await run_eval(
        CASES,
        sensible,
        world=noting_room(),
        seed=noting_start,
        goal_checks=['any(it.to == "on" for it in calls("room.switch"))'],
        scope_id=SCOPE,
        k=1,
    )
    assert summary.status == "completed"
    (goal,) = summary.goal_checks
    assert (goal.passed, goal.n) == (1, 2), "only the dark case switches the lamp on"


async def test_a_check_the_grammar_refuses_is_refused_by_the_quick_path() -> None:
    with pytest.raises(ValueError, match=r"intersects\(\) over variation.p"):
        await run_eval(
            [{"p": "on", **case} for case in CASES],
            sensible,
            world=room(),
            seed=start,
            goal_checks=['intersects(["on"], variation.p)'],
            scope_id=SCOPE,
        )


def test_authoring_on_a_quick_world_host_closes_its_tools_parameters_from_their_schemas() -> None:
    """The quick host described no tool parameters, so authoring refused even an enum-closed comparison on it."""
    from threetears.evals.contracts import EvalTemplate, ValidationFailedError
    from threetears.evals.run.authoring import refuse_unsupplied_world

    world = noting_room()
    profile = callable_host(world=world).profile

    def template(check: str) -> EvalTemplate:
        return EvalTemplate(scope_id=SCOPE, name="t", intent="i", candidate_kind="callable", goal_state_checks=[check])

    refuse_unsupplied_world(template('any(it.to == "on" for it in calls("room.switch"))'), profile=profile)
    with pytest.raises(ValidationFailedError, match="room.note's text is free text"):
        refuse_unsupplied_world(template('calls("room.note")[0].text == "x"'), profile=profile)
    with pytest.raises(ValidationFailedError, match="names calls this host does not define"):
        refuse_unsupplied_world(template('calls("room.paint").length > 0'), profile=profile)
