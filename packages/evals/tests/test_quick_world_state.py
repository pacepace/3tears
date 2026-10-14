"""A world with state: each cell seeds its world from its case, the agent acts on it, code grades the end.

The world example (``test_world_example.py``) is the happy path read from outside. These pin the mechanism: the seed
(each case's reserved ``world`` key) lands before the agent's first turn, every cell starts from its own case's state,
the end state is read back after the last turn, a tool call that succeeds is recorded where ``calls(...)`` reads it
and one the world refuses is not, the expression graders grade the result — and a world-less eval is exactly what it
was.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import pytest
from pydantic import BaseModel, Field

from threetears.evals import Eval, World
from threetears.evals.quick import as_host
from threetears.evals.quick.world import ToolRefused
from threetears.evals.schema import RecordedCall
from threetears.evals.run import get_result_trace, list_results, list_runs, list_templates
from packages.evals.tests.off_loop import off_loop

SCOPE = "quick-world-state-tests"


class Room(BaseModel):
    """A room with a lamp."""

    lamp: Literal["on", "off"] = Field(description="What the agent switches.")
    dark: bool = Field(description="Whether the lamp is needed.")


def switch(state: Room, to: Literal["on", "off"]) -> str:
    """Turn the lamp on or off."""
    state.lamp = to
    return f"lamp {to}"


def broken(state: Room) -> str:
    """A tool whose own code fails."""
    raise RuntimeError("the relay is stuck")


def room() -> World:
    return World(Room, tools=[switch, broken])


def start(case: Mapping[str, Any]) -> dict[str, Any]:
    return {"lamp": case["lamp"], "dark": case["dark"]}


#: Each case states its lamp and darkness (read as ``variation.*``), and starts its world from them.
CASES = [
    {"lamp": "off", "dark": True, "world": {"lamp": "off", "dark": True}},
    {"lamp": "on", "dark": False, "world": {"lamp": "on", "dark": False}},
]
LIT_IFF_DARK = '(state.lamp == "on") == state.dark'


async def sensible(case: Mapping[str, Any], tools: Any) -> str:
    """Light the room when it is dark, and only then."""
    seen = await tools.view()
    wanted = "on" if seen["dark"] else "off"
    return "kept" if wanted == seen["lamp"] else await tools["switch"](to=wanted)


async def stored(host: Any, run_id: str) -> list[Any]:
    """Each result with its trace, in case order."""
    pairs = [
        (result, off_loop(get_result_trace, host.storage, result))
        for result in off_loop(list_results, host.storage, run_id, SCOPE)
    ]
    return sorted(pairs, key=lambda pair: pair[0].test_case_id)


async def test_the_seed_lands_before_the_first_turn_and_every_cell_starts_from_its_own_case() -> None:
    first_sight: list[dict[str, Any]] = []

    async def meddling(case: Mapping[str, Any], tools: Any) -> str:
        """Leave the lamp on."""
        assert list(tools) == ["switch", "broken"]  # the mapping a tool-using agent is handed
        assert "world" not in case  # the starting state is the world's, never shown as the case
        first_sight.append(await tools.view())
        await tools.call("switch", to="on")  # every cell leaves the lamp on; the next must not inherit it
        return "done"

    await Eval(CASES, room()(meddling), [LIT_IFF_DARK], k=2).run(store=":memory:")
    assert sorted(map(str, first_sight)) == sorted(map(str, [start(case) for case in CASES] * 2))


async def test_the_end_state_is_read_back_after_the_last_turn_and_the_goal_check_grades_it() -> None:
    ev = Eval(CASES, room()(sensible), [LIT_IFF_DARK], k=1)
    host = as_host([ev], store=":memory:")
    summary = (await ev.run(host=host, scope=SCOPE)).summary
    assert (summary.status, summary.n_scored) == ("completed", 2)
    assert [(goal.check, goal.passed, goal.n) for goal in summary.goal_checks] == [(LIT_IFF_DARK, 2, 2)]
    # An eval names no control, so the check is unproven, and its do-nothing baseline is read off each case's seed:
    # both cases start with the lamp wrong, so doing nothing passes neither.
    (goal,) = summary.goal_checks
    assert (goal.proof, goal.did_nothing_passed, goal.did_nothing_cases) == ("unproven", 0, 2)
    assert f"goal check {LIT_IFF_DARK}: passed 2/2 — unproven" in summary.render()
    for (result, trace), case in zip(await stored(host, summary.run_id), CASES, strict=True):
        assert trace is not None
        # Both cases started wrong, so the stored end state is the agent's, never the seed.
        assert trace.end_state == {"lamp": "on" if case["dark"] else "off", "dark": case["dark"]} != start(case)
        assert [outcome.passed for outcome in result.goal_state_outcomes] == [True]
    (run,) = off_loop(list_runs, host, SCOPE)
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
    summary = (await Eval(CASES, room()(sensible), [never_needless], k=1).run(store=":memory:")).summary
    (goal,) = summary.goal_checks
    assert (goal.passed, goal.n, goal.did_nothing_passed, goal.did_nothing_cases) == (2, 2, 2, 2)
    (line,) = [line for line in summary.render().splitlines() if "goal check" in line]
    assert line.endswith(
        "passed 2/2 — NOT A MEASUREMENT: a candidate that did nothing passes it in 2 of 2 case(s), "
        "so this pass rate does not beat doing nothing"
    )


async def test_a_call_that_succeeds_is_recorded_for_calls_and_one_the_world_refuses_is_not() -> None:
    async def sloppy(case: Mapping[str, Any], tools: Any) -> str:
        """Try a dim lamp, shout, then switch on twice."""
        with pytest.raises(ToolRefused, match="dim"):
            await tools.call("switch", to="dim")
        # A refusal is a ValueError, which an agent written against no type of this package catches.
        with pytest.raises(ValueError, match="no tool 'shout'"):
            await tools.call("shout")
        await tools.call("switch", to="on")
        return await tools.call("switch", to="on")

    once = 'call_count("room.switch") <= 1'
    ev = Eval(CASES[:1], room()(sloppy), [once], k=1)
    host = as_host([ev], store=":memory:")
    summary = (await ev.run(host=host, scope=SCOPE)).summary
    ((result, trace),) = await stored(host, summary.run_id)
    assert trace is not None and trace.call_ledger is not None
    assert trace.call_ledger.calls == [RecordedCall(tool="room", action="switch", params={"to": "on"})] * 2
    assert [(outcome.expression, outcome.passed) for outcome in result.goal_state_outcomes] == [(once, False)]


async def test_a_tool_that_raises_excludes_the_cell_as_the_rigs_fault() -> None:
    async def presses(case: Mapping[str, Any], tools: Any) -> str:
        """Press the broken relay."""
        return await tools.call("broken")

    summary = (await Eval(CASES[:1], room()(presses), [LIT_IFF_DARK], k=1).run(store=":memory:")).summary
    assert (summary.n_scored, summary.n_candidate_failed, summary.n_excluded) == (0, 0, 1)
    assert "the tool broken raised RuntimeError: the relay is stuck" in summary.errors[0]
    assert summary.goal_checks == []


async def test_function_graders_read_the_answer_the_state_and_the_calls_beside_the_goal_checks() -> None:
    def answered(case: Mapping[str, Any], out: Any) -> bool:
        """The agent said something."""
        return isinstance(out.answer, str) and bool(out.answer)

    def lit_iff_dark(case: Mapping[str, Any], out: Any) -> bool:
        """The state the agent left, as the state model."""
        return isinstance(out.state, Room) and (out.state.lamp == "on") == out.state.dark

    def switched_at_most_once(case: Mapping[str, Any], out: Any) -> bool:
        """The calls the agent made."""
        return len([call for call in out.calls if call.action == "switch"]) <= 1

    graders = [answered, lit_iff_dark, switched_at_most_once, LIT_IFF_DARK]
    summary = (await Eval(CASES, room()(sensible), graders, k=1).run(store=":memory:")).summary
    assert [(m.name, m.mean) for m in summary.measures] == [
        ("answered", 1.0),
        ("lit_iff_dark", 1.0),
        ("switched_at_most_once", 1.0),
    ]
    assert summary.goal_checks[0].passed == 2


async def test_a_worlds_starting_states_and_checks_are_part_of_its_case_set() -> None:
    world = room()
    all_off = [{**case, "world": {"lamp": "off", "dark": case["dark"]}} for case in CASES]

    async def template_of(cases: list[dict[str, Any]], check: str) -> str | None:
        return (await Eval(cases, world(sensible), [check], k=1).run(store=":memory:")).template_id

    first = await template_of(CASES, LIT_IFF_DARK)
    again = await template_of(CASES, LIT_IFF_DARK)
    reseeded = await template_of(all_off, LIT_IFF_DARK)
    rechecked = await template_of(CASES, 'state.lamp == "on"')
    assert first == again
    assert len({first, reseeded, rechecked}) == 3


async def test_a_world_less_eval_is_unchanged_it_opens_no_world_and_keeps_no_ledger() -> None:
    async def double(case: Mapping[str, Any]) -> int:
        return int(case["n"]) * 2

    def even(case: Mapping[str, Any], answer: Any) -> bool:
        return answer % 2 == 0

    ev = Eval([{"n": 1}], double, [even], k=1)
    host = as_host([ev], store=":memory:")
    summary = (await ev.run(host=host, scope=SCOPE)).summary
    assert summary.goal_checks == [] and "goal check" not in summary.render()
    ((result, trace),) = await stored(host, summary.run_id)
    assert trace is not None
    assert (trace.end_state, trace.call_ledger, result.goal_state_outcomes) == (None, None, [])
    assert host.profile.world is None and off_loop(list_runs, host, SCOPE)[0].world_placements == {}


# --- refusals: each made before anything is stored ------------------------------------------------


@pytest.mark.parametrize(
    ("cases", "checks", "said"),
    [
        ([{"lamp": "off", "dark": True}], [LIT_IFF_DARK], "case\\(s\\) 0 carry none"),
        ([{"world": {"lamp": "off"}}], [LIT_IFF_DARK], "case 0's world is no Room"),
        ([{"world": {"lamp": "dim", "dark": True}}], [LIT_IFF_DARK], "case 0's world is no Room"),
        ([{"world": "off"}], [LIT_IFF_DARK], "a starting state maps each field to its value"),
        (CASES, ["state.colour == 1"], "reads state.colour"),
        (CASES, ['call_count("room.dim") == 0'], "names room.dim"),
    ],
    ids=[
        "a case with no world",
        "a world leaving a field unset",
        "a world the state refuses",
        "a world that is no mapping",
        "a check reading undeclared state",
        "a check naming an unknown tool",
    ],
)
async def test_an_incoherent_world_eval_is_refused_before_anything_is_stored(
    cases: list[dict[str, Any]], checks: list[str], said: str
) -> None:
    world = room()
    host = as_host([Eval(CASES, world(sensible), [LIT_IFF_DARK])], store=":memory:")
    with pytest.raises(ValueError, match=said):
        await Eval(cases, world(sensible), checks, k=1).run(host=host, scope=SCOPE)
    assert off_loop(list_templates, host.storage, SCOPE) == []


async def test_a_host_that_does_not_declare_the_world_is_refused_before_anything_is_stored() -> None:
    host = as_host([Eval(CASES, room()(sensible), [LIT_IFF_DARK])], store=":memory:")
    with pytest.raises(ValueError, match="does not declare world 'room'"):
        await Eval(CASES, room()(sensible), [LIT_IFF_DARK], k=1).run(host=host, scope=SCOPE)
    assert off_loop(list_templates, host.storage, SCOPE) == []


def test_an_expression_grader_with_no_world_with_state_is_refused_where_the_eval_is_written() -> None:
    async def plain(case: Mapping[str, Any]) -> str:
        return "x"

    with pytest.raises(ValueError, match="read a world's state, and the runner acts on none"):
        Eval(CASES, plain, [LIT_IFF_DARK])


async def test_cases_carrying_a_world_for_a_runner_on_none_are_refused_before_anything_runs() -> None:
    async def plain(case: Mapping[str, Any]) -> str:
        return "x"

    def said(case: Mapping[str, Any], answer: Any) -> bool:
        return bool(answer)

    with pytest.raises(ValueError, match="carry a starting state under 'world'"):
        await Eval(CASES, plain, [said], k=1).run(store=":memory:")


def test_a_state_field_with_no_description_is_refused() -> None:
    class Bare(BaseModel):
        lamp: str

    with pytest.raises(ValueError, match="Bare.lamp has no description"):
        World(Bare, tools=[])


async def test_a_well_formed_quick_world_passes_the_engine_s_own_conformance_kit() -> None:
    """The kit reads the subject view one entry per surface; the quick world once returned its state bare.

    Every dimension's ``perception_ab`` then read nothing on the ``view`` surface and failed, so the engine's own
    world could not pass the kit a host is told to run.
    """
    from threetears.evals.kernel.host.world_conformance import check_world_conformance

    world = room()
    assert world.registry is not None
    report = await check_world_conformance(world.registry, expressions=[LIT_IFF_DARK])

    failed = [
        (result.check, result.dimension, result.detail) for result in report.results if result.outcome == "failed"
    ]
    assert failed == []
    ab = {result.dimension: result.outcome for result in report.results if result.check == "perception_ab"}
    assert ab == {"lamp": "passed", "dark": "passed"}


class NotingRoom(BaseModel):
    """A room with a lamp and a notepad."""

    lamp: Literal["on", "off"] = Field(description="What the agent switches.")
    dark: bool = Field(description="Whether the lamp is needed.")
    note: str = Field(description="What the agent wrote down.")


def note(state: NotingRoom, text: str) -> str:
    """Leave a note by the lamp."""
    state.note = text
    return "noted"


def noting_room() -> World:
    def switch(state: NotingRoom, to: Literal["on", "off"]) -> str:
        """Turn the lamp on or off."""
        state.lamp = to
        return f"lamp {to}"

    return World(NotingRoom, tools=[switch, note])


#: The cases, each starting with an empty note; the world is ``noting_room`` (``calls("noting_room.switch")``).
NOTING_CASES = [{**case, "world": {**case["world"], "note": ""}} for case in CASES]


async def test_a_string_match_over_a_free_text_tool_parameter_is_refused_before_anything_runs() -> None:
    """Authoring refuses it, so the quick path does: a free string is what the model wrote, not structure."""
    ran: list[str] = []

    async def agent(case: Mapping[str, Any], tools: Any) -> str:
        """Run, which it must not."""
        ran.append("ran")
        return "x"

    check = 'any(it.text == "lamp fixed" for it in calls("noting_room.note"))'
    with pytest.raises(ValueError, match=r"noting_room.note's text is free text"):
        await Eval(NOTING_CASES, noting_room()(agent), [check]).run(store=":memory:")
    assert ran == [], "refused up front, never graded"


async def test_a_comparison_over_an_enum_closed_tool_parameter_is_accepted_and_graded() -> None:
    """The quick world describes its tools' parameters from their types, so authoring's closure rule admits it."""
    check = 'any(it.to == "on" for it in calls("noting_room.switch"))'
    summary = (await Eval(NOTING_CASES, noting_room()(sensible), [check], k=1).run(store=":memory:")).summary
    assert summary.status == "completed"
    (goal,) = summary.goal_checks
    assert (goal.passed, goal.n) == (1, 2), "only the dark case switches the lamp on"


async def test_a_check_the_grammar_refuses_is_refused_by_the_quick_path() -> None:
    ev = Eval([{"p": "on", **case} for case in CASES], room()(sensible), ['intersects(["on"], variation.p)'])
    with pytest.raises(ValueError, match=r"intersects\(\) over variation.p"):
        await ev.run(store=":memory:")


def test_authoring_on_a_quick_world_host_closes_its_tools_parameters_from_their_types() -> None:
    """The quick host described no tool parameters, so authoring refused even an enum-closed comparison on it."""
    from threetears.evals.schema import EvalTemplate
    from threetears.evals.kernel import ValidationFailedError
    from threetears.evals.run.authoring import refuse_unsupplied_world

    world = noting_room()
    profile = as_host([Eval(NOTING_CASES, world(sensible), [LIT_IFF_DARK])], store=":memory:").profile

    def template(check: str) -> EvalTemplate:
        return EvalTemplate(scope_id=SCOPE, name="t", intent="i", candidate_kind="callable", goal_state_checks=[check])

    refuse_unsupplied_world(template('any(it.to == "on" for it in calls("noting_room.switch"))'), profile=profile)
    with pytest.raises(ValidationFailedError, match="noting_room.note's text is free text"):
        refuse_unsupplied_world(template('calls("noting_room.note")[0].text == "x"'), profile=profile)
    with pytest.raises(ValidationFailedError, match="names calls this host does not define"):
        refuse_unsupplied_world(template('calls("noting_room.paint").length > 0'), profile=profile)
