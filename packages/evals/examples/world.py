"""Does the model turn the light on when the room is dark, judged by what it does rather than what it says?

Here the model acts: each case seeds a small world (a room's ``light`` and ``daylight``), the model changes
it through the world's own tool, and the engine reads the room back after the model's last turn. Goal-state
checks, code over the end state and the calls made, grade it with no judge. New here: ``world=``, ``seed=``
and ``goal_checks=``; unlike ``cassettes.py``'s ``tools=``, the tools act on the cell's world. The engine's
words for this: ``docs/concepts.md`` (World, Goal-state check).

Run it with ``python packages/evals/examples/world.py``. With ``ANTHROPIC_API_KEY`` set, Claude runs a real
tool-use loop (about 16 short calls, well under a cent); without it, a rule-based stand-in runs, with one
deliberate mistake so that a check fails, and its numbers say nothing about Claude.
"""

import asyncio
import os

from threetears.evals.quick import Dimension, EvalSummary, World, WorldTool, WorldTools, run_eval

MODEL = "claude-haiku-5-5"

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

# -----------------------------------------------------------------------------
# 4. The candidates: Claude in a tool-use loop, or the offline stand-in.
#
# Both are handed the case and ``room``, the tools bound to that case's world. ``room.view()`` is what
# the room's sensors show; a call through ``room`` changes the world and is recorded.
# -----------------------------------------------------------------------------

SYSTEM = "You control the lights in the user's room. Keep the light on when it is dark and off when it is bright."


async def claude_assistant(case: dict, room: WorldTools) -> str:
    """Claude, with the room's sensor readings and the switch_light tool, until it stops calling tools."""
    import anthropic  # imported here so the offline path does not need the package

    tools = [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in room.declared]
    messages: list = [{"role": "user", "content": f"Sensors: {await room.view()}\n\n{case['ask']}"}]
    async with anthropic.AsyncAnthropic() as client:  # reads ANTHROPIC_API_KEY
        for _ in range(4):  # a model that keeps calling tools cannot run up a bill
            reply = await client.messages.create(
                model=MODEL, max_tokens=1024, output_config={"effort": "low"},
                system=SYSTEM, tools=tools, messages=messages,
            )  # fmt: skip
            if reply.stop_reason != "tool_use":
                break
            messages.append({"role": "assistant", "content": reply.content})
            results = []
            for block in (b for b in reply.content if b.type == "tool_use"):
                try:
                    said, failed = await room.call(block.name, **block.input), False
                except ValueError as refused:  # a call the world refused changed nothing; the model is told why
                    said, failed = str(refused), True
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": said, "is_error": failed})
            messages.append({"role": "user", "content": results})
    return "".join(block.text for block in reply.content if block.type == "text")


async def offline_assistant(case: dict, room: WorldTools) -> str:
    """Stand-in, not a model: matches the light to the daylight, but always lights the room for reading."""
    seen = await room.view()
    wanted = "on" if seen["daylight"] == "dark" or "read" in case["ask"] else "off"  # the deliberate mistake
    if wanted == seen["light"]:
        return "Nothing to change."
    return await room["switch_light"](to=wanted)


# -----------------------------------------------------------------------------
# 5. Seed every case's room, let the candidate act on it twice, grade how each room ended, and print it.
# -----------------------------------------------------------------------------


async def main() -> EvalSummary:
    online = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if online:
        print(f"Running against Claude ({MODEL}).\n")
    else:
        print("ANTHROPIC_API_KEY is not set: running OFFLINE, with a rule-based stand-in for the model.\n")

    summary = await run_eval(
        CASES,
        claude_assistant if online else offline_assistant,
        world=ROOM,  # the state each cell seeds and the engine reads back
        seed=lambda case: {"light": case["light"], "daylight": case["daylight"]},  # each case's starting room
        goal_checks=GOAL_CHECKS,  # graded against the room the candidate left
        scope_id="world",
        k=2,
        model=MODEL if online else "offline",
    )
    # Each "goal check" line says in how many cells the room ended as the check requires.
    print(summary.render())
    return summary


if __name__ == "__main__":
    asyncio.run(main())
