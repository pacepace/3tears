"""The one caveat list's tables: each kind's class, each reason's remedy, and a caveat of the earlier shape as an entry.

A caveat is a :class:`~threetears.evals.analysis.report_input.CaveatEntry`: a qualification of the answer, typed with
its class, reach, state and reason, so the first screen is ranked from values and never from a sentence
(:func:`~threetears.evals.analysis.first_screen.first_screen`). A part of the package that has something to qualify
adds a raiser here and one row of :data:`CLASS_OF_KIND`; it renders no line of its own.

**Classes.** A kind's class says what it does to the answer. The first rows are fixed by the information
architecture; the rows marked default are this module's, for kinds that existed before classes did.

**Remedies.** What would close a caveat is not stored: :func:`remedy` reads it off the caveat's reason
(:data:`REMEDY_OF_REASON`, one act per reason), else off its kind.

Nothing here reads an eval's kind, a campaign's shape or a measure's name.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Final, Literal, Protocol

from threetears.evals.analysis.reading_states import CaveatClass
from threetears.evals.analysis.report_input import CaveatEntry

#: Nothing is declared yet: the tables' readers are the raisers and the page that renders the first screen.
__all__: list[str] = []

#: The class of each caveat kind; None for a kind that only ever lands in the method section. A kind absent here (an
#: analyst's own) has no class and ranks after every class.
CLASS_OF_KIND: Final[Mapping[str, CaveatClass | None]] = MappingProxyType(
    {
        # A: the answer was changed.
        "set_aside": "A",
        "withheld": "A",
        # B: the evidence is not what it appears.
        "late": "B",
        "confound": "B",
        "apparatus": "B",
        "short_run": "B",  # default
        "no_control": "B",  # default
        "nothing_declared": "B",  # default
        # C: an instrument on an asked reading is unchecked, or checked against a model only.
        "judge_uncalibrated": "C",
        "judge_partly_uncalibrated": "C",
        "judge_unstated": "C",
        # D: the test's level is loosened.
        "looks": "D",
        "best_of_n": "D",  # default: a pick among several is the same loosening
        # E: precision is missing on the deciding measure.
        "unsettled": "E",  # default
        "point_estimate": "E",  # default
        "borderline": "E",  # default
        # Method reach, never first-screen.
        "latency": None,
        "unreported": None,
    }
)

#: The one act that would close a caveat, as a closed kind a page words.
Remedy = Literal[
    "resume_or_rerun",
    "rerun_on_one_cell",
    "remeasure_or_attest",
    "rerun_serially",
    "check_against_people",
    "replace_instrument",
    "repair_rig",
    "add_cases",
    "record_and_rerun",
    "run_live",
    "draw_at_random",
    "finish_read",
    "declare_looks",
]

#: What would close a caveat, by its reason: every reason of
#: :data:`~threetears.evals.analysis.reading_states.REASONS_OF` has one.
REMEDY_OF_REASON: Final[Mapping[str, Remedy]] = MappingProxyType(
    {
        "instrument_unchecked": "check_against_people",
        "instrument_below_bar": "check_against_people",
        "instrument_leans": "replace_instrument",
        "people_disagree": "replace_instrument",
        "identical_arms_separated": "repair_rig",
        "unfinished": "resume_or_rerun",
        "split": "rerun_on_one_cell",
        "identity_unrecorded": "remeasure_or_attest",
        "measured_under_concurrency": "rerun_serially",
        "too_few": "add_cases",
        "no_variation": "add_cases",
        "not_recorded": "record_and_rerun",
        "replayed": "run_live",
        "not_drawn_at_random": "draw_at_random",
        "skipped_past_limit": "finish_read",
    }
)


#: What would close a caveat that carries no reason, by its kind.
_REMEDY_OF_KIND: Final[Mapping[str, Remedy]] = MappingProxyType({"looks": "declare_looks"})


def remedy(caveat: CaveatEntry) -> Remedy | None:
    """The one act that would close ``caveat``: its reason's, else its kind's.

    :param caveat: The caveat.
    :ptype caveat: CaveatEntry
    :return: The act, or None where neither its reason nor its kind names one.
    :rtype: Remedy | None
    """
    if caveat.reason is not None:
        return REMEDY_OF_REASON[caveat.reason]
    return _REMEDY_OF_KIND.get(caveat.kind)


class _Legacy(Protocol):
    """A caveat as the page's own read gives it: an id, its kind and where it may land."""

    @property
    def id(self) -> str: ...

    @property
    def kind(self) -> str: ...

    @property
    def reach(self) -> Literal["report", "scoped", "method"]: ...


def from_legacy(caveats: Iterable[_Legacy]) -> list[CaveatEntry]:
    """Caveats of the earlier shape as entries of the one list: the same ids in the same order, each with its kind's
    class and no state or reason, which reads as raised by nothing typed.

    How a report input that carries no list is given one when it is read: nothing is written back to it.

    :param caveats: The caveats, each with an ``id``, a ``kind`` and a ``reach``.
    :ptype caveats: Iterable[_Legacy]
    :return: One entry per caveat.
    :rtype: list[CaveatEntry]
    """
    return [
        CaveatEntry(id=caveat.id, kind=caveat.kind, cls=CLASS_OF_KIND.get(caveat.kind), reach=caveat.reach)
        for caveat in caveats
    ]
