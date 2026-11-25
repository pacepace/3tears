"""Whether a reading counts: the one closed set of states, and the one closed set of reasons.

A reading or a contrast is in exactly one of four states. ``counted``: read, shown, and it moves the answer.
``set_aside``: read and shown with its figures, and it moves no standing, for a stated reason about its instrument.
``withheld``: figures exist beneath it, and none is shown at this level because the evidence under it is not one whole
thing. ``not_available``: no figure can be computed. "Undecided" is not among them: it is a result of a counted
reading, never a way of not counting. Below a reading a trial is lost, with its cause, which is no state of these.

Each state but ``counted`` carries one reason from its own set (:data:`REASONS_OF`); two reasons,
``identity_unrecorded`` and ``trials_lost``, are each in two. Every part of the package that
says a reading is not counted says it with these words, checked by :func:`check_state`; a new way of not counting is a
new member here and nowhere else.

An arm's identity has one state per coordinate (:data:`IdentityState`): the arm holds ``one`` cell on it, is ``split``
over several, or the coordinate is ``unrecorded``, which matches nothing. A reason that is about a coordinate names
it.

Nothing here reads an eval's kind, a campaign's shape or a measure's name, and the module imports nothing from the
package but the stored shapes' base. It sits with the stored shapes because every layer holds these values: a blind
read's preference (stored on a durable blind read, so :class:`StateOf` keeps a newer minor's fields as every shape a
stored document embeds does), the bundle and the report input alike.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal, Self, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from threetears.evals.schema.base import CoreShapeModel

__all__ = [
    "ArmIdentity",
    "CaveatClass",
    "CaveatReach",
    "AttestableCoordinate",
    "Coordinate",
    "IdentityState",
    "Maker",
    "ReadingState",
    "SampleKind",
    "Reason",
    "StateOf",
    "check_state",
]

#: The four states a reading or a contrast is in.
ReadingState = Literal["counted", "set_aside", "withheld", "not_available"]

#: Why a reading is set aside: each is a fact about the instrument that read it.
SetAsideReason = Literal[
    "instrument_unchecked",
    "instrument_below_bar",
    "instrument_leans",
    "people_disagree",
    "identical_arms_separated",
]

#: Why a reading is withheld: the evidence under it is not one whole thing. ``trials_lost``: the runs finished and
#: the rig lost too many of their trials for what is left to be read as the arm's.
WithheldReason = Literal["unfinished", "split", "identity_unrecorded", "measured_under_concurrency", "trials_lost"]

#: Why no figure can be computed. Two reasons are here as well as among the withheld reasons, the only two that two
#: states share. ``identity_unrecorded``: a reading over runs with an unrecorded coordinate is withheld (its parts are
#: shown one step down), and a figure that cannot be computed across such runs at all (a check's result carried onto a
#: contrast) is not available; an unrecorded coordinate matches nothing in either. ``trials_lost``: a reading the rig
#: lost too many trials of is withheld, and a figure every one of whose trials was lost is not available.
NotAvailableReason = Literal[
    "too_few",
    "no_variation",
    "not_recorded",
    "replayed",
    "not_drawn_at_random",
    "skipped_past_limit",
    "identity_unrecorded",
    "trials_lost",
]

#: Every reason a reading is not counted.
Reason = SetAsideReason | WithheldReason | NotAvailableReason

#: An arm's identity on one coordinate.
IdentityState = Literal["one", "split", "unrecorded"]

#: The coordinates of a cell an arm can be split on, or leave unrecorded. Extended here and nowhere else.
Coordinate = Literal["served_model", "measure_definition", "template", "rig"]

#: The coordinates a host may state of a stored run after the fact (the ``runs_attest`` action): the two the package
#: itself reads, which keep a run that left them unrecorded in a cell of its own. The others are recorded at launch.
#: Extended here and nowhere else.
AttestableCoordinate = Literal["served_model", "measure_definition"]

#: What a caveat does to the answer, most serious first: ``A`` the answer was changed; ``B`` the evidence is not what
#: it appears; ``C`` an instrument on an asked reading is unchecked, or checked against a model only; ``D`` the test's
#: level is loosened; ``E`` precision is missing on the deciding measure.
CaveatClass = Literal["A", "B", "C", "D", "E"]

#: Where a caveat may land: report-wide, beside the claims it is scoped to, or in the method section.
CaveatReach = Literal["report", "scoped", "method"]

#: Who made a judgment: a rating, a preference, a recommendation or a call.
Maker = Literal["person", "model"]

#: How the items a check was read on were sampled: ``random`` (the first of a drawn order), ``selected`` (any other
#: set), or ``none`` (nothing was read). A property of a check's sample, and no state of a reading.
SampleKind = Literal["random", "selected", "none"]

#: The reasons each state may carry; ``counted`` carries none.
REASONS_OF: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "counted": frozenset(),
        "set_aside": frozenset(get_args(SetAsideReason)),
        "withheld": frozenset(get_args(WithheldReason)),
        "not_available": frozenset(get_args(NotAvailableReason)),
    }
)

#: The reasons that name a coordinate, and must: a split is a split on something, and an unrecorded identity is
#: unrecorded on something, which is what its remedy turns on.
_NAMES_A_COORDINATE: Final = frozenset({"split", "identity_unrecorded"})


def check_state(state: str, reason: str | None, coordinate: str | None = None) -> None:
    """Refuse a state, reason and coordinate that do not belong together.

    :param state: One of :data:`ReadingState`.
    :param reason: The reason, a member of the state's own set (:data:`REASONS_OF`); None for ``counted`` only.
    :param coordinate: The coordinate a ``split`` is on, or the one an unrecorded identity names; None otherwise.
    :raises ValueError: If the state is none of the four, ``counted`` carries a reason, another state carries none or
        one of another state's, a ``split`` or an unrecorded identity names no coordinate, or a reason that names none
        carries one.
    """
    allowed = REASONS_OF.get(state)
    if allowed is None:
        raise ValueError(f"{state!r} is not a reading state: one of {', '.join(REASONS_OF)}")
    if state == "counted":
        if reason is not None or coordinate is not None:
            raise ValueError("a counted reading carries no reason and no coordinate")
        return
    if reason not in allowed:
        raise ValueError(
            f"a {state} reading carries one of its own reasons ({', '.join(sorted(allowed))}), not {reason!r}"
        )
    if reason not in _NAMES_A_COORDINATE and coordinate is not None:
        raise ValueError(f"the reason {reason!r} names no coordinate, and {coordinate!r} was given")
    if reason in _NAMES_A_COORDINATE and coordinate is None:
        raise ValueError(f"the reason {reason!r} names the coordinate it is about")
    if coordinate is not None and coordinate not in get_args(Coordinate):
        raise ValueError(f"{coordinate!r} is not a coordinate: one of {', '.join(get_args(Coordinate))}")


class StateOf(CoreShapeModel):
    """The state of one reading, contrast, interval or pair, where it is not simply counted.

    A holder with none reads ``counted``. The three fields are checked together (:func:`check_state`) when the value is
    built and when a stored one is read, so no document holds a reason outside its state's set.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: ReadingState = Field(description="counted, set_aside, withheld or not_available.")
    reason: Reason | None = Field(default=None, description="Why it is not counted; none for counted.")
    coordinate: Coordinate | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The coordinate a split is on, or the one an unrecorded identity names.",
    )

    @model_validator(mode="after")
    def _belongs_together(self) -> Self:
        check_state(self.state, self.reason, self.coordinate)
        return self


class ArmIdentity(BaseModel):
    """Whether an arm is one thing, a state per coordinate of :data:`Coordinate`; a coordinate left out was not
    read."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    served_model: IdentityState | None = Field(
        default=None, description="Whether one model answered the arm's candidate calls."
    )
    measure_definition: dict[str, IdentityState] = Field(
        default_factory=dict,
        description="Whether each host measure's values were read under one definition, by measure.",
    )
    template: IdentityState | None = Field(
        default=None, description="Whether the arm's runs ran one revision of one template."
    )
    rig: IdentityState | None = Field(
        default=None, description="Whether the arm's runs were measured under one rig: one measurement context."
    )
