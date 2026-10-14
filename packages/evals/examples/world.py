"""Does the model turn the light on when the room is dark, judged by what it does rather than what it says?

Here the model acts: each case seeds a small world (a room's ``light`` and ``daylight``), the model changes
it through the world's own tool, and the engine reads the room back after the model's last turn. Goal-state
checks, code over the end state and the calls made, grade it with no judge. New here: ``world=``, ``seed=``
and ``goal_checks=``; unlike ``cassettes.py``'s ``tools=``, the tools act on the cell's world. The engine's
words for this: ``docs/concepts.md`` (World, Goal-state check).

The model acts through the engine's completion protocol, a prompt in and text out, so the loop is the example's
own: each turn the model answers in a JSON shape naming the tool it calls and its arguments, or ``none`` and its
reply to the person, and the loop makes the call and tells the model what the tool said.

Run it with ``python world.py`` (``README.md`` beside it says how). With ``ANTHROPIC_API_KEY`` set, Claude runs the
loop (about 16 short calls, well under a cent); without it, it replays ``recordings/world.jsonl``, what Claude
answered when the run was recorded, and calls nothing. ``RECORD=1 python world.py``, with the key set, records it
again.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from threetears.evals.analysis import EvalSummary
from threetears.evals.providers import models_client
from threetears.evals.quick import Dimension, World, WorldTool, WorldTools, callable_host, run_eval
from threetears.evals.run import get_result_trace, list_results, run_blocking
from threetears.evals.schema import ClientRequestSettings, EvalResult, EvalTrace
from threetears.evals.testing import RecordedCompletions

MODEL = "claude-haiku-5-5"
#: How every turn is asked for: an output cap and low effort. No temperature: this model refuses one.
SETTINGS = ClientRequestSettings(max_tokens=1024, reasoning_effort="low")
RECORDING = Path(__file__).with_name("recordings") / "world.jsonl"
RERECORD = "RECORD=1 python world.py"

# -----------------------------------------------------------------------------
# 1. The world: the state the model acts on, and the one tool that changes it.
# -----------------------------------------------------------------------------


# The model sees this docstring as the tool's description.
def switch_light(room: dict, to: str) -> str:
    """Turn the room's light on or off."""
    room["light"] = to
    return f"The light is now {to}."


ROOM = World(
    "room",
    [
        Dimension("light", {"enum": ["on", "off"]}, "The lamp the assistant controls."),
        Dimension("daylight", {"enum": ["dark", "bright"]}, "Whether the room needs the lamp at all."),
    ],
    tools=[WorldTool(switch_light, to={"enum": ["on", "off"]})],
)

# -----------------------------------------------------------------------------
# 2. The cases: each one's starting room, and what the person says.
# -----------------------------------------------------------------------------

CASES = [
    {"light": "off", "daylight": "dark", "ask": "I'm about to read."},
    {"light": "on", "daylight": "dark", "ask": "Make the room comfortable."},
    {"light": "off", "daylight": "bright", "ask": "I'm about to read."},
    {"light": "on", "daylight": "bright", "ask": "Make the room comfortable."},
]

# -----------------------------------------------------------------------------
# 3. The goal-state checks: ``state.*`` reads the room the model left, ``calls(...)`` what it did.
# -----------------------------------------------------------------------------

LIGHT_ENDS_ON_IFF_DARK = '(state.light == "on") == (state.daylight == "dark")'
# ``variation.light`` is the case's starting light, so this is "never switched to where it already was".
NEVER_SWITCHED_NEEDLESSLY = 'all(it.to != variation.light for it in calls("room.switch_light"))'
GOAL_CHECKS = [LIGHT_ENDS_ON_IFF_DARK, NEVER_SWITCHED_NEEDLESSLY]
# How main() names a check in the line for a cell that failed it.
FAILED_AS = {LIGHT_ENDS_ON_IFF_DARK: "wrong light", NEVER_SWITCHED_NEEDLESSLY: "needless switch"}

# -----------------------------------------------------------------------------
# 4. The candidate: Claude in a tool loop, handed the case and ``room``, the tools bound to that case's world.
#
# ``room.view()`` is what the room's sensors show; a call through ``room`` changes the world and is recorded.
# -----------------------------------------------------------------------------

SYSTEM = "You control the lights in the user's room. Keep the light on when it is dark and off when it is bright."
SYSTEM += """

You act through tools. Answer every turn with one JSON object: to call a tool, put its name in "tool" and its
arguments, as a JSON object, in "arguments"; when you are done, put "none" in "tool" and what you tell the user in
"reply". You are told what each call did before your next turn. The tools:
"""
SYSTEM += "\n".join(f"- {t.name}: {t.description} Arguments: {json.dumps(t.input_schema)}" for t in ROOM.tools.values())

#: The shape every turn's reply is held to: a strict JSON schema, sent as each call's ``response_format``.
TURN = {
    "type": "json_schema",
    "json_schema": {
        "name": "turn",
        "schema": {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": [*ROOM.tools, "none"]},
                "arguments": {"type": "string", "description": "The tool's arguments, as a JSON object; {} for none."},
                "reply": {"type": "string", "description": "What you tell the user, once you are done."},
            },
            "required": ["tool", "arguments", "reply"],
        },
    },
}


def assistant(llm: Any) -> Any:
    """A candidate that lets ``llm`` act on the room through its tools, until it calls none."""

    async def act(case: dict, room: WorldTools) -> str:
        user = f"Sensors: {await room.view()}\n\n{case['ask']}"
        for _ in range(4):  # a model that keeps calling tools cannot run up a bill
            turn = json.loads((await llm.generate(system=SYSTEM, user=user, response_format=TURN)).content)
            if turn["tool"] == "none":
                return turn["reply"]
            try:
                arguments = json.loads(turn["arguments"])
                if not isinstance(arguments, dict):
                    raise ValueError(f"the arguments {turn['arguments']} are not a JSON object")
                said = await room.call(turn["tool"], **arguments)
            except ValueError as refused:  # a call the world refused changed nothing; the model is told why
                said = f"refused: {refused}"
            user += f"\n\nYou called {turn['tool']} with {turn['arguments']}: {said}"
        return turn["reply"]

    return act


def client(model: str, settings: ClientRequestSettings) -> Any:
    """``model``, asked as ``settings`` say: live with ``ANTHROPIC_API_KEY`` (recorded with ``RECORD=1``), else replayed."""
    key, recording = os.environ.get("ANTHROPIC_API_KEY"), os.environ.get("RECORD") == "1"
    if not key:
        if recording:
            raise SystemExit("RECORD=1 records a live run, which needs ANTHROPIC_API_KEY")
        return RecordedCompletions.replay(RECORDING, settings=settings, model=model, rerecord=RERECORD)
    live = models_client(
        model,
        api_key=key,
        provider="anthropic",  # models' registry does not place this id on its own
        max_tokens=settings.max_tokens,
        reasoning_effort=settings.reasoning_effort,
    )  # and no temperature, so the model's own sampling applies
    return RecordedCompletions.record(live, RECORDING, settings=settings, replace=True) if recording else live


def how_it_runs(what: str) -> str:
    """The first line printed: running live, recording, or replaying a recording made on its date."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        recording = f", recording it into recordings/{RECORDING.name}" if os.environ.get("RECORD") == "1" else ""
        return f"Running live against Claude ({what}){recording}."
    made = json.loads(RECORDING.read_text(encoding="utf-8").splitlines()[0])["recorded_at"][:10]
    return (
        f"ANTHROPIC_API_KEY is not set: replaying recordings/{RECORDING.name}, Claude ({what}) as recorded on {made}."
    )


# -----------------------------------------------------------------------------
# 5. Seed every case's room, let the candidate act on it twice, grade how each room ended, and print each cell.
# -----------------------------------------------------------------------------


async def main() -> EvalSummary:
    llm = client(MODEL, SETTINGS)
    print(how_it_runs(MODEL) + "\n")

    host = callable_host(world=ROOM)  # kept, so each cell's trace can be read back below
    try:
        summary = await run_eval(
            CASES,
            assistant(llm),
            host=host,
            world=ROOM,  # the state each cell seeds and the engine reads back
            seed=lambda case: {"light": case["light"], "daylight": case["daylight"]},  # each case's starting room
            goal_checks=GOAL_CHECKS,  # graded against the room the candidate left
            scope_id="world",
            k=2,
            model=MODEL,
        )
    finally:
        await llm.aclose()
    # Each "goal check" line says in how many cells the room ended as the check requires, and how many cases a
    # model that did nothing would pass: a quick run proves no check, so each is marked unproven. The second check
    # passes for doing nothing in every case, so its 8/8 is marked as no measurement at all.
    print(summary.render())

    # Each cell: the room it started in, the calls the model made, the room it left, and the checks it failed.
    print()

    def read_cells() -> list[tuple[EvalResult, EvalTrace, dict]]:
        """Each cell's result, its trace and its case's seed: store reads, so run off the event loop below."""
        seeds = {case.id: case.host_payload["seed"] for case in host.storage.query_test_cases(summary.scope_id)}
        results = list_results(host.storage, summary.run_id, summary.scope_id)
        return [(result, get_result_trace(host.storage, result), seeds[result.test_case_id]) for result in results]

    cells = await run_blocking(host.blocking_executor, read_cells)
    for result, trace, seed in sorted(cells, key=lambda cell: (cell[0].test_case_id, cell[0].k_iteration)):
        start, end = f"{seed['daylight']}, light {seed['light']}", f"light {trace.end_state['light']}"
        calls = ", ".join(f"{call.action}(to={call.params['to']})" for call in trace.call_ledger.calls) or "no call"
        failed = ", ".join(FAILED_AS[it.expression] for it in result.goal_state_outcomes if not it.passed) or "ok"
        print(f"  {start:<17} -> {calls:<20} -> {end:<9}  {failed}")
    return summary


if __name__ == "__main__":
    asyncio.run(main())
