"""World events: what happened to a cell's world while its candidate ran, beyond what it was seeded with.

A seed sets the world before the first turn. Two kinds of thing move it afterwards, and both are
facts about the cell that a reader of its result needs and that its end state alone cannot show:

* **a triggered dimension firing.** Seeding a triggered dimension arms it; its condition fires it.
  The condition may be made to happen by the rig through the host's ``fire`` handle (a turn
  trigger the session advances), or it may happen in the world on its own account (an event the
  candidate's own action brought about; a ruling only a person can make) and be recorded by the
  kind that saw it. Which of the two it was is recorded, never inferred: a rig firing a condition
  and a candidate bringing it about are different findings about the candidate.
* **ambient perturbation.** The host's ``perturb_ambient`` handle moves state no dimension declares,
  at a turn the template states, so a candidate that perceives undeclared state is exposed by a run
  rather than only by the conformance kit.

The goal language reads the fired half (``fired("<dimension>")``), and the result stores both
(:attr:`~threetears.evals.contracts.models.EvalResult.world_events`), so a re-check re-grades a fired
check from what the cell recorded.

Pure data: no I/O, no registry, JSON-serializable by construction. The live handle that records
these is :class:`~threetears.evals.contracts.world_session.WorldSession`.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from threetears.evals.contracts.base import EvalDocumentModel

#: What moved the world: a triggered dimension's condition, by its trigger kind, or ambient perturbation.
#:
#: The first three are the registry's ``TriggerKind`` values (``turn``, ``event``, ``human``), carried
#: so a reader can tell a ruling only a person could make from a clock the rig advanced without
#: re-reading a registration that may since have moved.
WorldEventKind = Literal["turn", "event", "human", "ambient"]

#: Who made it happen.
#:
#: ``rig``    the session made it happen: a fire handle, or the ambient-perturbation handle.
#: ``world``  it happened in the world on its own account and the kind recorded it — an event the
#:            candidate's own action brought about, or a person's act.
WorldEventCause = Literal["rig", "world"]


class WorldEvent(EvalDocumentModel):
    """One thing that moved a cell's world after it was seeded, in the order it happened."""

    kind: WorldEventKind = Field(description="A triggered dimension's trigger kind, or ``ambient``.")
    dimension: str | None = Field(
        default=None,
        min_length=1,
        description="The triggered dimension that fired. None exactly when ``kind`` is ``ambient``: ambient "
        "perturbation moves state no dimension declares.",
    )
    condition: str | None = Field(
        default=None,
        min_length=1,
        description="The trigger's condition, in the host's words, as the registry declared it when it fired. "
        "None exactly when ``kind`` is ``ambient``.",
    )
    caused_by: WorldEventCause = Field(description="Whether the rig made it happen or the world did.")
    armed: bool = Field(
        default=False,
        description="Whether this cell's seed armed the dimension before it fired. Always true for a firing "
        "the rig caused, which is refused on an unarmed dimension; a firing the world caused may be of a "
        "condition the host's world holds of its own. False for ambient perturbation, which arms nothing.",
    )
    turn: int | None = Field(
        default=None,
        ge=1,
        description="The candidate turn it happened before or during, counted from 1, as the kind reported it. "
        "None for a kind that has no turns to count.",
    )
    moved: list[str] | None = Field(
        default=None,
        description="Ambient perturbation only: what the host's handle reported moving. ``[]`` is a handle that "
        "reported moving nothing; None is one that reported nothing at all, which is a different fact.",
    )

    @model_validator(mode="after")
    def _shape_follows_kind(self) -> Self:
        """Refuse a record whose fields contradict its kind.

        Returns:
            The record.

        Raises:
            ValueError: An ambient record names a dimension, a condition or a world cause, or claims to be
                armed; a fired record names no dimension or condition, or carries what ambient moved; a
                ``human`` record claims the rig caused it — no fire handle can be bound to a human trigger.
        """
        if self.kind == "ambient":
            if self.dimension is not None or self.condition is not None:
                raise ValueError("ambient perturbation moves undeclared state, so it names no dimension or condition")
            if self.caused_by != "rig" or self.armed:
                raise ValueError("ambient perturbation is the rig's act and arms nothing")
            return self
        if self.dimension is None or self.condition is None:
            raise ValueError(f"a {self.kind} firing names the dimension that fired and its condition")
        if self.moved is not None:
            raise ValueError("only ambient perturbation reports what it moved")
        if self.kind == "human" and self.caused_by == "rig":
            raise ValueError("a human trigger cannot be fired by the rig; only a person brings it about")
        if self.caused_by == "rig" and not self.armed:
            raise ValueError("the rig fires only what the cell's seed armed")
        return self


def fired_dimensions(events: list[WorldEvent] | tuple[WorldEvent, ...]) -> frozenset[str]:
    """The triggered dimensions that fired in a cell, whoever caused them — what ``fired()`` reads.

    Args:
        events: The cell's world events.

    Returns:
        The names of every dimension that fired at least once.
    """
    return frozenset(event.dimension for event in events if event.dimension is not None)


__all__ = ["WorldEvent", "WorldEventCause", "WorldEventKind", "fired_dimensions"]
