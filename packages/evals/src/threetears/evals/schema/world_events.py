"""World events: what happened to a cell's world while its candidate ran, beyond what it was seeded with.

A seed sets the world before the first turn. Two kinds of thing move it afterwards, and both are
facts about the cell that a reader of its result needs and that its end state alone cannot show:

* **a triggered dimension firing.** Seeding a triggered dimension arms it; its condition fires it.
  The condition may be made to happen by the rig through the host's ``fire`` handle (a turn
  trigger the kind fires as its turns pass — the session advances no clock of its own), or it may happen in the world on its own account (an event the
  candidate's own action brought about; a ruling only a person can make) and be recorded by the
  kind that saw it. Which of the two it was is recorded, never inferred: a rig firing a condition
  and a candidate bringing it about are different findings about the candidate.
* **ambient perturbation.** The host's ``perturb_ambient`` handle moves state no dimension declares,
  at a turn the template states, so a candidate that perceives undeclared state is exposed by a run
  rather than only by the conformance kit.

**Provenance is per event, never per dimension.** A dimension can fire more than once and for more than
one reason — the event a cell's seed armed, and an event the host's world holds of its own on the same
dimension (a module's own weather turning beside the storm the seed scheduled). Every firing names the
host's identity of the event that fired (:attr:`WorldEvent.event`), and ``armed`` says whether that event
is one this cell's seed armed: an identity a seed handle returned when it armed it. Deriving ``armed``
from the dimension instead would record the world's own firing as the seed's whenever the seed armed
anything on that dimension, and the two could not be told apart afterwards.

The goal language reads the fired half — ``fired("<dimension>")`` for any firing, ``fired_armed("<dimension>")``
for a firing of the event the seed armed — through :class:`Firings`, and the result stores both
(:attr:`~threetears.evals.schema.models.EvalResult.world_events`), so a re-check re-grades a fired
check from what the cell recorded.

Pure data: no I/O, no registry, JSON-serializable by construction. The live handle that records
these is :class:`~threetears.evals.kernel.world_session.WorldSession`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self

from pydantic import Field, model_validator

from threetears.evals.schema.base import EvalDocumentModel

if TYPE_CHECKING:
    from threetears.evals.schema.models import ApparatusProvenance

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
    event: str | None = Field(
        default=None,
        min_length=1,
        description="The host's identity of the event that fired: the one its seed handle returned when the "
        "cell's seed armed it, or the one the kind observed firing in the world. Required for a firing, so "
        "provenance is a property of the event and never inferred from the dimension; None exactly when "
        "``kind`` is ``ambient``.",
    )
    armed: bool = Field(
        default=False,
        description="Whether the event that fired is one this cell's seed armed — an identity a seed handle "
        "returned when it armed it. Always true for a firing the rig caused, which is refused on an unarmed "
        "dimension; a firing the world caused is armed only when it is the seed's own event, and not when the "
        "host's world fired an event of its own on the same dimension. False for ambient perturbation, which "
        "arms nothing.",
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
            ValueError: An ambient record names a dimension, a condition, an event or a world cause, or claims
                to be armed; a fired record names no dimension, condition or event, or carries what ambient
                moved; a ``human`` record claims the rig caused it — no fire handle can be bound to a human
                trigger.
        """
        if self.kind == "ambient":
            if self.dimension is not None or self.condition is not None or self.event is not None:
                raise ValueError(
                    "ambient perturbation moves undeclared state, so it names no dimension, condition or event"
                )
            if self.caused_by != "rig" or self.armed:
                raise ValueError("ambient perturbation is the rig's act and arms nothing")
            return self
        if self.dimension is None or self.condition is None or self.event is None:
            raise ValueError(f"a {self.kind} firing names the dimension that fired, its condition and the event")
        if self.moved is not None:
            raise ValueError("only ambient perturbation reports what it moved")
        if self.kind == "human" and self.caused_by == "rig":
            raise ValueError("a human trigger cannot be fired by the rig; only a person brings it about")
        if self.caused_by == "rig" and not self.armed:
            raise ValueError("the rig fires only what the cell's seed armed")
        return self


@dataclass(frozen=True)
class Firings:
    """What fired in a cell, as the goal language reads it: every dimension that fired, and the armed ones.

    Built from a cell's world events by :meth:`of` — a live session's, or a stored result's on a
    re-check — or stated for a control end state. ``fired("<dimension>")`` reads :attr:`dimensions`;
    ``fired_armed("<dimension>")`` reads :attr:`armed`, and is *not established* when
    :attr:`armed_known` is False.

    Attributes:
        dimensions: Every triggered dimension that fired at least once, whoever caused it.
        armed: The dimensions on which the event the cell's seed armed fired — a subset of ``dimensions``.
        armed_known: Whether which firings were armed can be known at all. False for a witnessed cell:
            no seed armed anything there, so the events record ``armed=False`` because nothing could
            mark them armed, not because the cell established that none was — and ``fired_armed()``,
            negated or not, is then not established rather than a verdict.
    """

    dimensions: frozenset[str] = frozenset()
    armed: frozenset[str] = frozenset()
    armed_known: bool = True

    def __post_init__(self) -> None:
        """Refuse an armed firing that is not a firing, and an armed firing where none can be known.

        Raises:
            ValueError: ``armed`` names a dimension ``dimensions`` does not, or names any while
                ``armed_known`` is False.
        """
        if stray := sorted(self.armed - self.dimensions):
            raise ValueError(
                f"armed firing(s) {', '.join(map(repr, stray))} are not among the dimensions that fired; an armed "
                "firing is a firing"
            )
        if self.armed and not self.armed_known:
            raise ValueError(
                f"armed firing(s) {', '.join(map(repr, sorted(self.armed)))} are stated where which firings were "
                "armed cannot be known; a cell no seed armed has no armed firing to name"
            )

    @classmethod
    def of(cls, events: Iterable[WorldEvent], *, provenance: ApparatusProvenance) -> Firings:
        """What a cell's world events say fired, and what its apparatus lets them say about arming.

        The one rule for what a cell's firings can establish, read by the live grading of a cell and by
        the re-check of a stored one alike, so a re-check never establishes what the original grading
        could not. A ``commissioned`` cell's seed armed its events (or armed none, which is itself
        known), so each event's ``armed`` is the record. A ``witnessed`` cell had no seed: its events
        say ``armed=False`` because nothing could mark them otherwise, so which firings were armed is
        unknowable and ``fired_armed()`` is not established.

        Args:
            events: The cell's world events, in any order.
            provenance: The apparatus provenance of the run the cell belongs to
                (:attr:`~threetears.evals.schema.models.EvalRun.apparatus_provenance`).

        Returns:
            Every dimension that fired, and those on which an armed event fired — or, for a witnessed
            cell, no armed set and ``armed_known=False``.
        """
        fired = [event for event in events if event.dimension is not None]
        dimensions = frozenset(event.dimension for event in fired if event.dimension is not None)
        if provenance == "witnessed":
            return cls(dimensions=dimensions, armed_known=False)
        return cls(
            dimensions=dimensions,
            armed=frozenset(event.dimension for event in fired if event.dimension is not None and event.armed),
        )


__all__ = ["Firings", "WorldEvent", "WorldEventCause", "WorldEventKind"]
