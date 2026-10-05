"""A scripted simulator-role client for the conversation suites: picks and lines from a script.

One class, shared by the scheduler, session-break and conversation suites, because all three drive
the same two kinds of simulator-role call and each would otherwise grow its own copy of the routing.
It routes on the ``response_format`` name the driver sends — ``next_speaker`` for a scheduling pick,
``simulated_user_reply`` for an actor's line — which is the protocol the driver speaks, not a reading
of any prompt's prose. The speaking actor is recovered from the policy line the suites write as
``You speak as <id>.``, a convention of these tests' own actors.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from threetears.evals.contracts.models import ActorPolicy

__all__ = ["DONE", "Raw", "ScriptedResponse", "ScriptedTable", "actor"]

#: A line that says the actor is done with the conversation.
DONE = object()


class Raw(str):
    """A completion text handed back verbatim, for a reply of the test's own (broken) shape."""


@dataclass
class ScriptedResponse:
    """The response shape the driver reads: content plus the usage attributes a client reports."""

    content: str
    model: str = "sim-model"
    cost_usd: float | None = 0.001


def actor(actor_id: str, **overrides: Any) -> ActorPolicy:
    """An actor whose policy names it, which is how :class:`ScriptedTable` knows who is speaking."""
    fields: dict[str, Any] = {"id": actor_id, "policy": f"You speak as {actor_id}.", "intent": "Engage the candidate."}
    fields.update(overrides)
    return ActorPolicy(**fields)


@dataclass
class ScriptedTable:
    """Answers each scheduling call from ``picks`` and each actor's call from ``lines[actor]``, in order.

    A pick or line that is a :class:`Raw` is returned verbatim; :data:`DONE` is the actor leaving.
    Running out of script fails the test loudly: the driver made a call nobody expected.
    """

    picks: list[str] = field(default_factory=list)
    lines: dict[str, list[Any]] = field(default_factory=dict)
    #: Each scheduling call, as ``(user prompt, the enum of legal answers it sent)``.
    schedule_calls: list[tuple[str, list[str]]] = field(default_factory=list)
    #: The actor each utterance call spoke for, in order.
    utterance_calls: list[str] = field(default_factory=list)

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
        assert response_format is not None, "every simulator-role call sends a schema"
        name = response_format["json_schema"]["name"]
        if name == "next_speaker":
            enum = response_format["json_schema"]["schema"]["properties"]["next"]["enum"]
            self.schedule_calls.append((user, list(enum)))
            assert self.picks, "the driver asked the scheduler more often than the script expected"
            pick = self.picks.pop(0)
            return ScriptedResponse(pick if isinstance(pick, Raw) else json.dumps({"next": pick}))
        assert name == "simulated_user_reply", name
        match = re.search(r"You speak as (\S+)\.", system)
        assert match is not None, "these suites' actors name themselves in their policy"
        speaker = match.group(1)
        self.utterance_calls.append(speaker)
        script = self.lines.get(speaker)
        assert script, f"{speaker} was asked to speak more often than the script expected"
        line = script.pop(0)
        if isinstance(line, Raw):
            return ScriptedResponse(line)
        if line is DONE:
            return ScriptedResponse(json.dumps({"utterance": "I'm done here.", "done": True}))
        return ScriptedResponse(json.dumps({"utterance": line, "done": False}))
