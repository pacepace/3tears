"""The first screen, computed once: which caveats a report opens with, in what order, and what stops it being read.

:func:`first_screen` is the single place a report's first-screen qualifications are chosen. It takes the one caveat
list, the gate and the call, and returns the typed value both pages print
(:class:`~threetears.evals.analysis.report_input.FirstScreen`). It is pure: no clock, no store, no text. It reads no
eval kind, no campaign shape, no measure name and no word of prose; every comparison it makes is on a member of the
closed sets of :mod:`~threetears.evals.analysis.reading_states`.

The rules, each held by one test:

- **Eligible.** A caveat is eligible when it reaches the whole report, code has not proved it immaterial, and no other
  caveat silences it.
- **Silenced** (:func:`subsumed`). A withheld reading silences every other caveat scoped only to it. A reading set
  aside for a cause silences a caveat stating that cause on the readings set aside. Identical arms that separated
  silence every instrument caveat on that reading. A check against people silences a check against a model on the same
  criterion.
- **Ranked** (:data:`RANKS`). The order is data: evidence not whole, identical arms separated, identity unrecorded, a
  late criterion (and every other fact that makes the evidence not what it appears), a judge not checked, a judge
  checked against a model only (a model of a candidate's family first), repeated looks, a judge checked against a
  model that is itself checked, then few cases. A caveat no row places ranks last. Within a rank, a caveat on an asked
  reading of an arm code's own read calls comes first, then by id.
- **Capped.** Three reasons are shown; the rest are counted, never dropped.
- **Gated.** A campaign that cannot be read shows no reason: the gate carries how many wait below it.
- **The call** is copied through and is never a reason.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from typing import Final, NamedTuple

from threetears.evals.analysis.reading_states import CaveatClass, ReadingState, Reason
from threetears.evals.analysis.report_input import CallRead, CaveatEntry, FirstScreen, Gate

__all__ = ["RANKS", "REASONS_SHOWN", "Rank", "first_screen", "subsumed"]

#: How many reasons the trust slot shows before it counts the rest.
REASONS_SHOWN: Final = 3


class _Any:
    """Matches whatever a caveat holds."""


_ANY: Final = _Any()


class Rank(NamedTuple):
    """One row of the ranking: the facts of a caveat it places, each a closed-set word or left open.

    :ivar name: What the row is, for a test and a reviewer to name it by.
    :ivar cls: The caveat's class, where the row asks for one.
    :ivar state: Its state.
    :ivar reason: Its reason.
    :ivar against: What its instrument was checked against: ``people``, ``model``, or None for nothing.
    :ivar same_family: Whether the model checked against is of a candidate's family.
    :ivar against_checked: Whether the model checked against is itself checked against people.
    """

    name: str
    cls: CaveatClass | _Any = _ANY
    state: ReadingState | _Any = _ANY
    reason: Reason | _Any = _ANY
    against: str | None | _Any = _ANY
    same_family: bool | _Any = _ANY
    against_checked: bool | _Any = _ANY

    def places(self, caveat: CaveatEntry) -> bool:
        """Whether every fact the row asks for is the caveat's."""
        return all(
            asked is _ANY or asked == held
            for asked, held in (
                (self.cls, caveat.cls),
                (self.state, caveat.state),
                (self.reason, caveat.reason),
                (self.against, caveat.against),
                (self.same_family, caveat.same_family),
                (self.against_checked, caveat.against_checked),
            )
        )


#: The ranking, most serious first; a caveat takes the first row that places it. The statistician's order
#: (2026-11-14): looks stay below both judge rows at every look count, and above a model that is itself checked.
RANKS: Final[tuple[Rank, ...]] = (
    Rank("evidence_not_whole", state="withheld", reason="unfinished"),
    Rank("evidence_not_whole", state="withheld", reason="split"),
    Rank("evidence_not_whole", state="withheld", reason="measured_under_concurrency"),
    Rank("identical_arms_separated", reason="identical_arms_separated"),
    Rank("identity_unrecorded", reason="identity_unrecorded"),
    Rank("late_criterion", cls="B"),
    Rank("judge_unchecked", state="set_aside", reason="people_disagree"),
    Rank("judge_unchecked", state="set_aside", reason="instrument_leans"),
    Rank("judge_unchecked", state="set_aside", reason="instrument_below_bar"),
    Rank("judge_unchecked", state="set_aside", reason="instrument_unchecked"),
    Rank("answer_changed", cls="A"),
    Rank("judge_unchecked", cls="C", against=None),
    Rank("judge_unchecked", cls="C", against="people"),
    Rank("judge_checked_against_a_model_only", cls="C", against="model", same_family=True),
    Rank("judge_checked_against_a_model_only", cls="C", against="model", against_checked=False),
    Rank("repeated_looks", cls="D"),
    Rank("judge_checked_against_a_checked_model", cls="C", against="model"),
    Rank("few_cases", cls="E"),
)


def _rank(caveat: CaveatEntry) -> int:
    return next((index for index, row in enumerate(RANKS) if row.places(caveat)), len(RANKS))


def subsumed(caveats: Iterable[CaveatEntry]) -> list[CaveatEntry]:
    """The caveats, each one another silences carrying that one's id (``subsumed_by``); the rest unchanged, in the
    order given.

    A caveat already silenced keeps what silenced it, and the result does not depend on the order given: where
    several caveats would silence one, the one with the least id is named.

    :param caveats: The caveat list.
    :ptype caveats: Iterable[CaveatEntry]
    :return: The same caveats, the silenced ones marked.
    :rtype: list[CaveatEntry]
    """
    held = list(caveats)
    by_id = sorted(held, key=lambda caveat: caveat.id)
    withheld: dict[str, str] = {}
    set_aside: dict[tuple[str, Reason | None], str] = {}
    separated: dict[str, str] = {}
    people: dict[str, str] = {}
    for caveat in by_id:
        for reading in caveat.readings:
            if caveat.state == "withheld":
                withheld.setdefault(reading, caveat.id)
            if caveat.state == "set_aside":
                set_aside.setdefault((reading, caveat.reason), caveat.id)
                if caveat.reason == "identical_arms_separated":
                    separated.setdefault(reading, caveat.id)
        if caveat.against == "people" and caveat.criterion is not None:
            people.setdefault(caveat.criterion, caveat.id)

    def silenced_by(caveat: CaveatEntry) -> str | None:
        readings = caveat.readings
        if caveat.state != "withheld" and len(readings) == 1 and readings[0] in withheld:
            return withheld[readings[0]]
        if caveat.state != "set_aside" and readings:
            if all((reading, caveat.reason) in set_aside for reading in readings):
                return set_aside[readings[0], caveat.reason]
            if caveat.cls == "C" and all(reading in separated for reading in readings):
                return separated[readings[0]]
        if caveat.against == "model" and caveat.criterion in people:
            return people[caveat.criterion]
        return None

    marked: list[CaveatEntry] = []
    for caveat in held:
        by = caveat.subsumed_by if caveat.subsumed_by is not None else silenced_by(caveat)
        marked.append(caveat if by == caveat.subsumed_by else caveat.model_copy(update={"subsumed_by": by}))
    return marked


def first_screen(
    caveats: Sequence[CaveatEntry],
    *,
    gate: Gate | None,
    called_arms: Collection[str],
    call: CallRead | None = None,
) -> FirstScreen:
    """The first screen's slots for one caveat list.

    :param caveats: The one caveat list, in any order.
    :ptype caveats: Sequence[CaveatEntry]
    :param gate: What stops the campaign being read, or None when it can be.
    :ptype gate: Gate | None
    :param called_arms: The arms code's own read of the options calls, by id.
    :ptype called_arms: Collection[str]
    :param call: The call, where one was made; copied through untouched.
    :ptype call: CallRead | None
    :return: The gate (with the count waiting below it), the reasons shown and the count past them, the count of
        open report-wide caveats, and the call.
    :rtype: FirstScreen
    """
    called = set(called_arms)

    def key(caveat: CaveatEntry) -> tuple[int, int, str]:
        touches = caveat.asked and (not caveat.arms or bool(called.intersection(caveat.arms)))
        return _rank(caveat), 0 if touches else 1, caveat.id

    eligible = sorted(
        (
            caveat
            for caveat in subsumed(caveats)
            if caveat.reach == "report" and caveat.immaterial is not True and caveat.subsumed_by is None
        ),
        key=key,
    )
    shown = [] if gate is not None else [caveat.id for caveat in eligible[:REASONS_SHOWN]]
    return FirstScreen(
        gate=None if gate is None else gate.model_copy(update={"waiting": len(eligible)}),
        reasons=shown,
        remainder=0 if gate is not None else max(len(eligible) - REASONS_SHOWN, 0),
        open_report_caveats=len(eligible),
        called_arms=sorted(called),
        call=call,
    )
