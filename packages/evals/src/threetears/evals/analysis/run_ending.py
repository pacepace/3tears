"""Whether a reading may open with a verdict: the one decision every surface asks, and its words.

A verdict is read first and acted on. One over a run that did not finish is a figure over whichever cells happened to
be stored when the run stopped, so no surface opens with it: the console print, the aspect's page, the campaign's page,
the written page, a listing's line and the gate (``evals gate``) each ask :func:`stopped_short` and, where it answers, open on what
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
from typing import TYPE_CHECKING, Final

from threetears.evals.kernel.text_block import one_line

if TYPE_CHECKING:
    from threetears.evals.analysis.bundle.schema import AnalysisContextBundle

__all__ = ["RunEnding", "StoppedShort", "bundle_stopped_short", "stopped_short", "unfinished_opening"]

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

    def pooled(self) -> str:
        """What the readings after the opening rest on where the reading pools several runs, and the remedy, in one
        sentence without a closing stop: a campaign's contrasts are computed with an unfinished run's partial results
        counted beside every finished run's, so the sentence says so and how to read a verdict.

        :return: The sentence.
        :rtype: str
        """
        live = all(ending.status in ("running", "pending") for ending in self.endings)
        partial = "recorded so far" if live else "recorded before the stop"
        return (
            f"The readings below include the partial results {partial}, so resume the run or leave it out of the "
            "campaign to read a verdict"
        )


def stopped_short(endings: Iterable[RunEnding]) -> StoppedShort | None:
    """Whether a reading over these runs may open with a verdict: None when it may, else what stopped it.

    :param endings: Every member run's ending; a run that finished is handed in like any other.
    :ptype endings: Iterable[RunEnding]
    :return: None when every run reached ``completed``; else the unfinished runs, which the reading opens on.
    :rtype: StoppedShort | None
    """
    short = tuple(ending for ending in endings if ending.status != _FINISHED)
    return StoppedShort(short) if short else None


def bundle_stopped_short(bundle: AnalysisContextBundle) -> StoppedShort | None:
    """Whether a reading over ``bundle`` may open with a verdict, as every page asks it: None when it may, else the
    member runs that did not finish (:func:`stopped_short`), each with why its
    record says it ended (``RunSummary.ended_because``; none on a bundle assembled before that was carried).

    :param bundle: The campaign's evidence.
    :ptype bundle: AnalysisContextBundle
    :return: None when every member run completed; else the unfinished runs, in the bundle's order.
    :rtype: StoppedShort | None
    """
    reasons = {summary.run_id: summary.ended_because for summary in bundle.run_summaries}
    return stopped_short(
        RunEnding(run_id=run_id, status=status, reason=reasons.get(run_id))
        for run_id, status in bundle.incomplete_runs.items()
    )


def _unfinished_sentences(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> list[str]:
    """What stopped each unfinished run of ``bundle``, a sentence each with its closing stop, as a page opens with
    them in place of a verdict; empty when every run completed (:func:`bundle_stopped_short`).

    A run is named by its arm (``v2's run``), by its id too where the arm has several runs, and as ``The run`` where
    the bundle reads one arm; a run the realized design places in no arm by its id.

    :param bundle: The campaign's evidence.
    :ptype bundle: AnalysisContextBundle
    :param arm: Names an arm by its variant key, as every table of the report does.
    :ptype arm: Callable[[str], str]
    :return: The sentences.
    :rtype: list[str]
    """
    short = bundle_stopped_short(bundle)
    if short is None:
        return []
    design = bundle.design
    placed = [*([design.control_arm] if design.control_arm is not None else []), *design.contrasts]
    arm_of = {run_id: each for each in placed for run_id in each.run_ids}

    def who(ending: RunEnding) -> str:
        of = arm_of.get(ending.run_id)
        if of is None:
            return f"Run {ending.run_id}"
        if len(placed) == 1 and len(of.run_ids) == 1:
            return "The run"
        return f"{arm(of.variant_key)}'s run{'' if len(of.run_ids) == 1 else f' {ending.run_id}'}"

    return [f"{sentence.rstrip('.')}." for sentence in short.sentences(who)]


def unfinished_opening(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> tuple[str, str] | None:
    """What a reading over ``bundle`` opens with in place of a verdict where a run of it did not finish, or None when
    every run completed: what stopped each run, and that followed by what the readings below still count and the remedy
    (:meth:`StoppedShort.pooled`).

    :param bundle: The campaign's evidence.
    :ptype bundle: AnalysisContextBundle
    :param arm: Names an arm by its variant key, as every table of the report does.
    :ptype arm: Callable[[str], str]
    :return: What stopped the runs, then the whole opening; None when every run completed.
    :rtype: tuple[str, str] | None
    """
    short = bundle_stopped_short(bundle)
    if short is None:
        return None
    stopped = " ".join(_unfinished_sentences(bundle, arm))
    return stopped, f"{stopped} {short.pooled()}."
