"""Support ticket status: replays a ticket's history of events to the status it is in now.

A ticket starts ``new``. Events move it between ``new``, ``open``, ``pending``, ``solved`` and ``closed``:

- ``agent_reply``: an agent answered; a ``new`` or ``open`` ticket becomes ``pending`` (waiting on the customer).
- ``customer_reply``: the customer wrote; a ``pending`` ticket becomes ``open``. A customer writing back on a
  ``solved`` ticket reopens it: it becomes ``open`` and its ``reopens`` count goes up by one.
- ``solve``: an agent marks a ``new``, ``open`` or ``pending`` ticket ``solved``.
- ``close``: the nightly job closes tickets that have been ``solved`` for four days.
- ``note``: an internal note, which never moves a ticket.

An event that does not apply to the ticket's status leaves the status as it is (an agent replying on a solved
ticket, for instance). A closed ticket never moves again; a customer writing on one starts a follow-up ticket, which
``follow_up`` records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

STATUSES = ("new", "open", "pending", "solved", "closed")
EVENTS = ("agent_reply", "customer_reply", "solve", "close", "note")

#: (status, event) -> the status the event moves the ticket to. A pair that is not here leaves the status alone.
TRANSITIONS: dict[tuple[str, str], str] = {
    ("new", "agent_reply"): "pending",
    ("open", "agent_reply"): "pending",
    ("pending", "customer_reply"): "open",
    ("new", "solve"): "solved",
    ("open", "solve"): "solved",
    ("pending", "solve"): "solved",
    ("solved", "close"): "closed",
}


@dataclass
class Ticket:
    id: str
    status: str = "new"
    reopens: int = 0
    follow_up: bool = False
    history: list[str] = field(default_factory=list)


def apply(ticket: Ticket, event: str) -> Ticket:
    """Move ``ticket`` by one event, in place, and return it."""
    if event not in EVENTS:
        raise ValueError(f"unknown event {event!r} on ticket {ticket.id}")
    ticket.history.append(event)
    if ticket.status == "closed":
        if event == "customer_reply":
            ticket.follow_up = True
        return ticket
    moved = TRANSITIONS.get((ticket.status, event), ticket.status)
    if ticket.status == "solved" and moved == "open":
        ticket.reopens += 1
    ticket.status = moved
    return ticket


def replay(ticket_id: str, events: list[dict[str, Any]]) -> Ticket:
    """A ticket's status after its events, each ``{"at": "2026-02-03T09:14", "event": "agent_reply"}``, applied in
    time order (the export does not promise its order)."""
    ticket = Ticket(id=ticket_id)
    for event in sorted(events, key=lambda e: e["at"]):
        apply(ticket, event["event"])
    return ticket
