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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from threetears.evals.kernel.candidate_kind import CandidateOutput, CellPending, CellReservation
from threetears.evals.schema.models import ActorPolicy

__all__ = ["DONE", "FakeCellSink", "Raw", "ScriptedResponse", "ScriptedTable", "actor"]

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
    #: What every response reports it cost; ``None`` is a client that prices nothing.
    cost_usd: float | None = 0.001
    #: What the client says one call can cost at most, before it is made; ``None`` is a client that cannot say.
    ceiling_usd: float | None = None

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        del system, user, response_format
        return self.ceiling_usd

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
        assert response_format is not None, "every simulator-role call sends a schema"
        name = response_format["json_schema"]["name"]
        if name == "next_speaker":
            enum = response_format["json_schema"]["schema"]["properties"]["next"]["enum"]
            self.schedule_calls.append((user, list(enum)))
            assert self.picks, "the driver asked the scheduler more often than the script expected"
            pick = self.picks.pop(0)
            return ScriptedResponse(
                pick if isinstance(pick, Raw) else json.dumps({"next": pick}), cost_usd=self.cost_usd
            )
        assert name == "simulated_user_reply", name
        match = re.search(r"You speak as (\S+)\.", system)
        assert match is not None, "these suites' actors name themselves in their policy"
        speaker = match.group(1)
        self.utterance_calls.append(speaker)
        script = self.lines.get(speaker)
        assert script, f"{speaker} was asked to speak more often than the script expected"
        line = script.pop(0)
        if isinstance(line, Raw):
            return ScriptedResponse(line, cost_usd=self.cost_usd)
        if line is DONE:
            return ScriptedResponse(json.dumps({"utterance": "I'm done here.", "done": True}), cost_usd=self.cost_usd)
        return ScriptedResponse(json.dumps({"utterance": line, "done": False}), cost_usd=self.cost_usd)


# parity-with: threetears.evals.kernel.candidate_kind.CellSink
@dataclass
class FakeCellSink:
    """One cell alone, keeping its spend as the engine's sink does.

    The engine's rule: a running total per source, only ever added to. It asks no ceiling: the cost cap is read
    between cells, so a reservation is never refused and the cap is never reported reached.
    """

    #: Each source's running total.
    spent: dict[CellPending, float | None] = field(default_factory=dict)
    #: The reservations of the calls in flight.
    held: list[_HeldOnFakeSink] = field(default_factory=list)
    pending: list[CellPending] = field(default_factory=list)
    #: The kind's last reading of its candidate side (``report_progress``), or ``None`` when it registered none.
    progress: Callable[[], CandidateOutput] | None = None

    def waiting_on(self, pending: CellPending) -> None:
        self.pending.append(pending)

    def report_progress(self, read: Callable[[], CandidateOutput]) -> None:
        self.progress = read

    def add_spend(self, source: CellPending, cost_usd: float | None) -> None:
        assert cost_usd is None or cost_usd >= 0, cost_usd
        total = self.spent.get(source, 0.0)
        self.spent[source] = None if total is None or cost_usd is None else total + cost_usd

    def reserve(self, source: CellPending, ceiling_usd: float | None) -> _HeldOnFakeSink:
        hold = _HeldOnFakeSink(sink=self, source=source, ceiling_usd=ceiling_usd or 0.0)
        self.held.append(hold)
        return hold

    def cost_cap_reached(self) -> bool:
        return False


@dataclass(eq=False)
class _HeldOnFakeSink(CellReservation):
    """One held call on :class:`FakeCellSink`, closed once."""

    sink: FakeCellSink
    source: CellPending
    ceiling_usd: float
    closed: bool = False

    def settle(self, cost_usd: float | None) -> None:
        assert not self.closed, "a reservation is settled once"
        self.closed = True
        self.sink.held.remove(self)
        self.sink.add_spend(self.source, cost_usd)

    def release(self) -> None:
        if not self.closed:
            self.closed = True
            self.sink.held.remove(self)
