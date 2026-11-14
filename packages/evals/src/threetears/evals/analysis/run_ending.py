"""Whether a reading may open with a verdict: the one decision every surface asks, and its words.

A verdict is read first and acted on. One over a run that did not finish is a figure over whichever cells happened to
be stored when the run stopped, so no surface opens with it: the console print, the aspect's page, the campaign's page,
the written page, a listing's line and the gate each ask :func:`stopped_short` and, where it answers, open on what
stopped the run instead (:meth:`StoppedShort.sentences`), with no verdict.

**The decision is the run's stored status alone.** A run that reached ``completed`` may be opened with a verdict;
every other status may not: ``failed``, ``exhausted``, ``cancelled``, ``budget_stopped``, and a run still ``running``
or ``pending`` when it was read (one interrupted and never taken up again reads so too). A cancel and a budget stop are
designed stops, and they keep what the run delivered, but which cells they kept is decided by when the stop landed,
exactly as a failure's are, so they are read alike. How much of its matrix a ``completed`` run delivered is another
question, which the run's completeness answers and every page already discloses.

Nothing here reads an eval's kind or a campaign's shape: a surface hands in each member run's ending and prints what
comes back.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Final

from threetears.evals.kernel.text_block import one_line

__all__ = ["RunEnding", "StoppedShort", "stopped_short"]

#: The one status a run's evidence is whole under.
_FINISHED: Final = "completed"

#: How each unfinished status reads after the run's name, and how the line about its readings says when they stop.
_HOW: Final[dict[str, tuple[str, str]]] = {
    "failed": ("failed", "recorded before it failed"),
    "exhausted": ("stopped when its provider refused the account or the model", "recorded before it stopped"),
    "cancelled": ("was cancelled", "recorded before it stopped"),
    "budget_stopped": ("was stopped by its budget", "recorded before it stopped"),
    "running": ("has not finished", "recorded so far"),
    "pending": ("has not started", "recorded so far"),
}


@dataclass(frozen=True)
class RunEnding:
    """How one run stood when it was read.

    :ivar run_id: The run's id.
    :ivar status: Its stored status (``EvalRun.status``).
    :ivar reason: Why it ended, in its record's own words, where the record gives one: the cancel's or the budget
        stop's reason, the error a failed run recorded last.
    """

    run_id: str
    status: str
    reason: str | None = None

    @property
    def how(self) -> str:
        """How the run ended, as it reads after the run's name: ``failed``, ``was cancelled``; a status this build
        has no words for by its name (``ended <status>``)."""
        return _HOW[self.status][0] if self.status in _HOW else f"ended {self.status}"

    @property
    def recorded(self) -> str:
        """When the run's readings stop, as the line about them says it: ``recorded before it failed``."""
        return _HOW[self.status][1] if self.status in _HOW else "recorded before it stopped"


@dataclass(frozen=True)
class StoppedShort:
    """The runs of one reading that did not finish, which the reading opens on in place of a verdict.

    :ivar endings: Each unfinished run's ending, in the order they were handed in; never empty.
    """

    endings: tuple[RunEnding, ...]

    def sentences(self, who: Callable[[RunEnding], str]) -> list[str]:
        """What stopped each run, one sentence each, without a closing stop: ``<who> failed, so there is no verdict:
        <reason>``, the reason on one line and left off where the run's record gives none.

        :param who: Names a run as the surface names it (``the run``, an arm's name).
        :ptype who: Callable[[RunEnding], str]
        :return: The sentences, in the endings' order.
        :rtype: list[str]
        """
        return [
            f"{who(ending)} {ending.how}, so there is no verdict"
            + (f": {one_line(ending.reason)}" if ending.reason and ending.reason.strip() else "")
            for ending in self.endings
        ]

    def covers(self, results: int | None = None) -> str:
        """What the readings after the opening cover, in one clause without a closing stop: ``the readings below cover
        only the 4 result(s) recorded before it failed``.

        :param results: How many results the readings cover, where the surface counts them; None leaves the count out.
        :ptype results: int | None
        :return: The clause.
        :rtype: str
        """
        said = {ending.recorded for ending in self.endings}
        recorded = said.pop() if len(said) == 1 else "recorded before it stopped"
        counted = "the results" if results is None else f"the {results} result(s)"
        return f"the readings below cover only {counted} {recorded}"


def stopped_short(endings: Iterable[RunEnding]) -> StoppedShort | None:
    """Whether a reading over these runs may open with a verdict: None when it may, else what stopped it.

    :param endings: Every member run's ending; a run that finished is handed in like any other.
    :ptype endings: Iterable[RunEnding]
    :return: None when every run reached ``completed``; else the unfinished runs, which the reading opens on.
    :rtype: StoppedShort | None
    """
    short = tuple(ending for ending in endings if ending.status != _FINISHED)
    return StoppedShort(short) if short else None
