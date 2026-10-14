"""Rung 4: is your agent doing the right thing to the world, not only saying it?

Each request seeds one day's calendar; the agent books through two tools, and the graders read the day it left.
From this directory, ``python -m threetears.evals agent.py`` replays ``recordings/agent.jsonl`` offline, and
``RECORD=1`` with ``OPENROUTER_API_KEY`` set records it again. As recorded on deepseek/deepseek-v4-flash:
100 calls, $0.0080 (the recording's own cost_usd fields, summed).
"""

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from threetears.evals import Eval, Model, World

model = Model("deepseek/deepseek-v4-flash", max_tokens=4096, effort="low", recording="recordings/agent.jsonl")


class Day(BaseModel):
    busy: list[int] = Field(description="The hours already taken; 9 is 9:00 to 10:00.")
    booked: list[int] = Field([], description="The hours the agent booked.")


def find_free(day: Day) -> list[int]:
    """List the day's free hours, 9 to 16."""
    return [hour for hour in range(9, 17) if hour not in day.busy + day.booked]


def book(day: Day, hour: Annotated[int, Field(ge=9, le=16)]) -> str:
    """Book the one-hour meeting at an hour; the calendar does not stop a double booking."""
    day.booked.append(hour)
    return f"booked {hour}:00"


class Turn(BaseModel):
    tool: Literal["find_free", "book", "done"]
    hour: int
    reply: str


SYSTEM = (
    "You book one-hour meetings on today's calendar. Each turn, call one tool: find_free (hour 0) lists the free "
    "hours; book books the meeting at `hour`. When the meeting is booked, or it cannot be, answer done with your "
    "reply to the person. Never book over a taken hour, and book at most once."
)


async def agent(case: dict, tools: Mapping) -> str:
    """Book the meeting a request asks for, or say why it cannot be booked."""
    said = f"Request: {case['ask']}"
    for _ in range(5):  # a loop that never says done cannot run up a bill
        turn = await model.json(said, Turn, system=SYSTEM)
        if turn.tool == "done":
            return turn.reply
        try:
            answer = await (tools["book"](hour=turn.hour) if turn.tool == "book" else tools["find_free"]())
        except ValueError as refused:  # a call the world refuses changes nothing, and the agent is told why
            answer = f"refused: {refused}"
        said += f"\n\n{turn.tool}(hour={turn.hour}) said: {answer}"
    return "ran out of turns"


def in_window(case: dict, out) -> bool:
    """Every booking falls inside the hours the request asked for."""
    return all(case["start"] <= hour < case["end"] for hour in out.state.booked)


booking = Eval(
    "requests.jsonl",  # each case's "world" is the day it starts with
    runner=World(Day, tools=[find_free, book])(agent),  # the model's turns replay; tools that move state run live
    grader=[
        '(state.booked.length >= 1) == (variation.expect == "book")',  # booked exactly when a free hour fits
        "not intersects(state.booked, state.busy)",  # never over a taken hour: doing nothing passes this too
        'call_count("day.book") <= 1',  # and so does this: only the first check beats the did-nothing control
        in_window,
    ],
    name="booking",
)
