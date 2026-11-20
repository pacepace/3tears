"""The first screen, computed once: which caveats a report opens with, in what order, and what stops it being read.

:func:`first_screen` is the single place a report's first-screen qualifications are chosen. It takes the one caveat
list, the gate and the call, and returns the typed value both pages print
(:class:`~threetears.evals.analysis.report_input.FirstScreen`). It is pure: no clock, no store, no text. It reads no
eval kind, no campaign shape, no measure name and no word of prose; every comparison it makes is on a member of the
closed sets of :mod:`~threetears.evals.schema.reading_states`.

The rules, each held by one test:

- **Eligible.** A caveat is eligible when it reaches the whole report, code has not proved it immaterial, and no other
  caveat silences it. One that says precision is missing (class E) is eligible only where the answer turns on the
  reading it names; anywhere else it is stated in its table cell, one step down.
- **Silenced** (:func:`subsumed`). A withheld reading silences every other caveat scoped only to it: to that reading
  and to arms it is withheld on (a reading withheld on one arm leaves a caveat on the reading's every arm standing). A reading set
  aside for a cause silences a caveat stating that cause on the readings set aside. Identical arms that separated
  silence every instrument caveat on that reading. A check against people whose result decides (met or not met)
  silences a check against a model on the same criterion; one that is undecided or insufficient speaks for nothing.
- **Ranked** (:data:`RANKS`). The order is data, eight ranks: evidence not whole, identical arms separated, identity
  unrecorded, a late criterion (and every other fact that makes the evidence not what it appears), a judge not
  checked, a judge checked against a model only, repeated looks, then few cases. A caveat no row places ranks last.
  Within a rank the order is the rank's own where one was fixed with it (a model of a candidate's family first), then
  a caveat on an asked reading of an arm code's own read calls, then by id.
- **Capped.** Three reasons are shown; the rest are counted, never dropped.
- **Gated.** A campaign that cannot be read shows no reason: the gate carries how many wait below it.
- **The call** is copied through and is never a reason.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from typing import Final, NamedTuple

from threetears.evals.schema.reading_states import CaveatClass, ReadingState, Reason
from threetears.evals.analysis.report_input import CallRead, CaveatEntry, FirstScreen, Gate

__all__ = ["first_screen"]

#: How many reasons the trust slot shows before it counts the rest.
REASONS_SHOWN: Final = 3


class _Any:
    """Matches whatever a caveat holds."""


_ANY: Final = _Any()


class Rank(NamedTuple):
    """One row of the ranking: a rank, and the facts of a caveat it places, each a closed-set word or left open.

    :ivar rank: The rank, 1 the most serious. Several rows may hold one rank.
    :ivar name: What the rank is, for a test and a reviewer to name it by.
    :ivar within: The row's place inside its rank, where the order inside the rank was fixed with it; rows of one rank
        that share a place are told apart by the next keys.
    :ivar cls: The caveat's class, where the row asks for one.
    :ivar state: Its state.
    :ivar reason: Its reason.
    :ivar against: What its instrument was checked against: ``people``, ``model``, or None for nothing.
    :ivar same_family: Whether the model checked against is of a candidate's family.
    """

    rank: int
    name: str
    within: int = 0
    cls: CaveatClass | _Any = _ANY
    state: ReadingState | _Any = _ANY
    reason: Reason | _Any = _ANY
    against: str | None | _Any = _ANY
    same_family: bool | _Any = _ANY

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
            )
        )


#: The ranking: the statistician's eight ranks (2026-11-14), most serious first; a caveat takes the first row that
#: places it, and one no row places ranks after all eight. Rank 1 is every withheld reading, whatever withheld it, and
#: every figure the rig lost the trials of, so
#: identity unrecorded is rank 3 only where it has not already withheld a reading. Looks stay below both judge ranks
#: at every look count.
#:
#: **Ties.** Caveats of one rank are ordered by the row's place inside the rank (rank 5: the reason a reading was set
#: aside, in the order one is stated where two apply, then a counted reading whose judge has no check; rank 6: a
#: model of a candidate's family first), then by whether the caveat is on an asked reading of an arm code's own read
#: calls, then by id.
RANKS: Final[tuple[Rank, ...]] = (
    Rank(1, "evidence_not_whole", state="withheld"),
    Rank(1, "evidence_not_whole", reason="trials_lost"),
    Rank(2, "identical_arms_separated", reason="identical_arms_separated"),
    Rank(3, "identity_unrecorded", reason="identity_unrecorded"),
    Rank(4, "late_criterion", cls="B"),
    Rank(5, "judge_unchecked", 0, state="set_aside", reason="people_disagree"),
    Rank(5, "judge_unchecked", 1, state="set_aside", reason="instrument_leans"),
    Rank(5, "judge_unchecked", 2, state="set_aside", reason="instrument_below_bar"),
    Rank(5, "judge_unchecked", 3, state="set_aside", reason="instrument_unchecked"),
    Rank(5, "judge_unchecked", 4, cls="C", against=None),
    Rank(5, "judge_unchecked", 4, cls="C", against="people"),
    Rank(6, "judge_checked_against_a_model_only", 0, cls="C", against="model", same_family=True),
    Rank(6, "judge_checked_against_a_model_only", 1, cls="C", against="model"),
    Rank(7, "repeated_looks", cls="D"),
    Rank(8, "few_cases", cls="E"),
)

#: The results of a check that decide: only these speak for the instrument they checked.
_DECIDES: Final = frozenset({"met", "not_met"})

#: Where a caveat no row places ranks: after every rank.
_UNRANKED: Final = (max(row.rank for row in RANKS) + 1, 0)


def _rank(caveat: CaveatEntry) -> tuple[int, int]:
    return next(((row.rank, row.within) for row in RANKS if row.places(caveat)), _UNRANKED)


def _eligible(caveat: CaveatEntry) -> bool:
    """Whether a caveat may be a first-screen reason: it reaches the whole report, code has not proved it immaterial,
    nothing silences it, and, where what it says is that precision is missing (class E), the answer turns on the
    reading it names."""
    if caveat.reach != "report" or caveat.immaterial is True or caveat.subsumed_by is not None:
        return False
    return caveat.cls != "E" or caveat.deciding


def subsumed(caveats: Iterable[CaveatEntry]) -> list[CaveatEntry]:
    """The caveats, each one another silences carrying that one's id (``subsumed_by``); the rest unchanged, in the
    order given.

    A caveat already silenced keeps what silenced it, and the result does not depend on the order given: where
    several caveats would silence one, the one with the least id is named.

    :param caveats: The caveat list.
    :return: The same caveats, the silenced ones marked.
    """
    held = list(caveats)
    by_id = sorted(held, key=lambda caveat: caveat.id)
    withheld: dict[str, tuple[str, frozenset[str]]] = {}
    set_aside: dict[tuple[str, Reason | None], str] = {}
    separated: dict[str, str] = {}
    people: dict[str, str] = {}
    for caveat in by_id:
        for reading in caveat.readings:
            if caveat.state == "withheld":
                withheld.setdefault(reading, (caveat.id, frozenset(caveat.arms)))
            if caveat.state == "set_aside":
                set_aside.setdefault((reading, caveat.reason), caveat.id)
                if caveat.reason == "identical_arms_separated":
                    separated.setdefault(reading, caveat.id)
        if caveat.against == "people" and caveat.criterion is not None and caveat.check in _DECIDES:
            people.setdefault(caveat.criterion, caveat.id)

    def silenced_by(caveat: CaveatEntry) -> str | None:
        readings = caveat.readings
        if caveat.state != "withheld" and len(readings) == 1 and readings[0] in withheld:
            by, arms = withheld[readings[0]]
            # Scoped only to it: on every arm it is withheld on, and the reading withheld on every arm or on these.
            if not arms or (caveat.arms and set(caveat.arms) <= arms):
                return by
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
    :param gate: What stops the campaign being read, or None when it can be.
    :param called_arms: The arms code's own read of the options calls, by id.
    :param call: The call, where one was made; copied through untouched.
    :return: The gate (with the count waiting below it), the reasons shown and the count past them, the count of
        open report-wide caveats, and the call.
    """
    called = set(called_arms)

    def key(caveat: CaveatEntry) -> tuple[int, int, int, str]:
        touches = caveat.asked and (not caveat.arms or bool(called.intersection(caveat.arms)))
        return *_rank(caveat), 0 if touches else 1, caveat.id

    eligible = sorted((caveat for caveat in subsumed(caveats) if _eligible(caveat)), key=key)
    shown = [] if gate is not None else [caveat.id for caveat in eligible[:REASONS_SHOWN]]
    return FirstScreen(
        gate=None if gate is None else gate.model_copy(update={"waiting": len(eligible)}),
        reasons=shown,
        remainder=0 if gate is not None else max(len(eligible) - REASONS_SHOWN, 0),
        open_report_caveats=len(eligible),
        called_arms=sorted(called),
        call=call,
    )
