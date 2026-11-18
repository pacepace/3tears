"""Run completeness: the sentence a short or degraded run must carry wherever its numbers are read.

:func:`completeness_disclosure` says how much of a run's planned work produced a usable measurement, and
:func:`degraded_run_disclosures` collects that sentence for every degraded run a projection pools.
:func:`bundle_stopped_short` says whether a reading over a bundle may open with a verdict, and
:func:`unfinished_opening` what it opens with where it may not, and :func:`gate_rule` which rule of the verdict chain
stops a reading over a bundle being read at all.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Final, Literal

from threetears.evals.analysis.run_ending import RunEnding, StoppedShort, stopped_short
from threetears.evals.analysis.stats import SMALL_N_BAND_FLOOR

if TYPE_CHECKING:
    from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, ArmScoreboard
    from threetears.evals.schema.models import (
        EvalRun,
        RunCompleteness,
    )


# The distinguishing clause of the degraded disclosure, split out for the same
# reason the withheld-partition clauses above are: tests pin the CLAUSE, so the
# sentence around it stays free to be rewritten for a reader.
DEGRADED_RUN_CLAUSE = "produced a usable measurement"

# What the breakdown above cannot claim when the counts were reconstructed rather
# than tallied by the loop. Rendered as its own sentence — the causes it qualifies
# are printed as fact, and a reader who takes "N never ran" literally on a run
# whose process was killed is reading a lower bound as a measurement.
RECONSTRUCTED_COUNTS_CLAUSE = (
    "The process running it died, so these counts were reconstructed from what reached storage: "
    "a cell that ran and failed to write is indistinguishable from one that never ran, and is "
    "counted among the latter."
)


def completeness_disclosure(completeness: RunCompleteness | None) -> str | None:
    """The one sentence a surface must show about a run that came up short.

    Returns ``None`` for a run that delivered its whole matrix, and for one
    carrying no completeness record at all — a record's absence is not evidence
    of a shortfall. A run without one has not reached a terminal state, or had
    every attempt to write the record refused (rare, and logged where it
    happened); rendering asserts nothing about either.

    The shortfall is broken down by cause because the three are not the same
    problem: cells the loop never ran mean the run was launched against less than
    it recorded, lost writes mean the harness dropped data it had, and infra
    exclusions mean the cells ran and told us nothing about the candidate. At
    least one is positive whenever the run is degraded. They account for the
    whole of ``expected - measured`` on the ``produced <= expected`` side, which
    is every run any caller can produce today; on the other side — a loop handed
    more cases than the run recorded — the surplus is not a cause of a shortfall
    and is deliberately not netted against one, so the printed causes are the
    reasons a cell went missing rather than a balancing identity.

    :param completeness: The run's stored record, or ``None``.
    :return: The disclosure, rendered verbatim by every surface, or ``None`` when
        there is nothing a reader needs warning about.
    """
    if completeness is None or not completeness.degraded:
        return None

    causes: list[str] = []
    never_ran = completeness.expected_cells - completeness.produced_cells
    if never_ran > 0:
        causes.append(f"{never_ran} never ran")
    lost = completeness.produced_cells - completeness.persisted_cells
    if lost > 0:
        causes.append(f"{lost} ran but the write was lost, leaving no stored result")
    if completeness.infra_excluded_cells > 0:
        causes.append(
            f"{completeness.infra_excluded_cells} excluded as a harness failure rather than a candidate outcome"
        )

    breakdown = f" ({'; '.join(causes)})" if causes else ""
    return (
        f"DEGRADED: {completeness.measured_cells} of {completeness.expected_cells} cells "
        f"{DEGRADED_RUN_CLAUSE}{breakdown}. Every rate on this run is computed over that shorter "
        "denominator, so reading it beside a complete run compares two different populations."
        + (f" {RECONSTRUCTED_COUNTS_CLAUSE}" if completeness.counted_from == "stored_results" else "")
    )


def degraded_run_disclosures(runs: Iterable[EvalRun]) -> dict[str, str]:
    """The short-matrix runs among these, each with the sentence that qualifies it.

    The seam the *pooling* surfaces read. :func:`completeness_disclosure` answers
    about one run, and the run-scoped surfaces (``get_run``, ``run_summary``,
    ``runs_compare``) each call it for the run they are about. An aggregator has
    no such run: it pools dozens into a rate, and every one of them was silently
    admitted because nothing had asked the question over a *set*. This asks it
    once, so ``frontier``, ``results_pivot`` and ``history`` cannot come to
    disagree about which of their runs were short.

    **Degradation is a property of the data, not of the run's status.** A
    ``completed`` run that lost a cell to a harness exclusion is degraded and
    passes every status filter there is, so a surface that reasoned about
    truncation from ``status`` alone would still pool it — which is exactly how
    a completed-but-short run reached all three aggregators by default.

    :param runs: The runs behind an answer. Order is irrelevant; keying is by id.
    :return: ``run_id -> disclosure`` for each degraded run, omitting every run that
        delivered its whole matrix and every run carrying no completeness record
        (a record's absence is not evidence of a shortfall — see
        :func:`completeness_disclosure`). Empty means nothing needs disclosing,
        so a caller can use emptiness as the predicate rather than re-deriving it.
    """
    return {run.id: sentence for run in runs if (sentence := completeness_disclosure(run.completeness)) is not None}


__all__ = [
    "bundle_stopped_short",
    "unfinished_opening",
    "completeness_disclosure",
    "DEGRADED_RUN_CLAUSE",
    "degraded_run_disclosures",
    "RECONSTRUCTED_COUNTS_CLAUSE",
]


def bundle_stopped_short(bundle: AnalysisContextBundle) -> StoppedShort | None:
    """Whether a reading over ``bundle`` may open with a verdict, as every page asks it: None when it may, else the
    member runs that did not finish (:func:`~threetears.evals.analysis.run_ending.stopped_short`), each with why its
    record says it ended (``RunSummary.ended_because``; none on a bundle assembled before that was carried).

    :param bundle: The campaign's evidence.
    :return: None when every member run completed; else the unfinished runs, in the bundle's order.
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
    :param arm: Names an arm by its variant key, as every table of the report does.
    :return: The sentences.
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
    (:meth:`~threetears.evals.analysis.run_ending.StoppedShort.pooled`).

    :param bundle: The campaign's evidence.
    :param arm: Names an arm by its variant key, as every table of the report does.
    :return: What stopped the runs, then the whole opening; None when every run completed.
    """
    short = bundle_stopped_short(bundle)
    if short is None:
        return None
    stopped = " ".join(_unfinished_sentences(bundle, arm))
    return stopped, f"{stopped} {short.pooled()}."


#: The share of an arm's trials whose loss to the test setup, alone or as the gap between two arms, means the run
#: measured its setup rather than the models. A stand-in until a test across arms decides it.
SETUP_GATE_SHARE: Final = 1 / 3

#: The rules of the verdict chain that stop a reading being read at all, in the order the chain reads them.
GateRule = Literal["setup", "unfinished", "too_small"]


def lost_share(board: ArmScoreboard) -> float:
    """The share of an arm's trials lost to the test setup; 0 for an arm that ran none.

    :param board: The arm's scoreboard.
    :return: The share, 0 to 1.
    """
    return sum(board.lost.values()) / board.ran if board.ran else 0.0


def setup_failed(bundle: AnalysisContextBundle) -> bool:
    """Whether the setup gate holds: an arm lost more than :data:`SETUP_GATE_SHARE` of its trials to the test setup,
    or two arms' shares differ by more than that.

    :param bundle: The aspect's evidence.
    :return: True when the run measured its setup rather than the models.
    """
    shares = [lost_share(board) for board in bundle.scoreboard if board.ran]
    return bool(shares) and (max(shares) > SETUP_GATE_SHARE or max(shares) - min(shares) > SETUP_GATE_SHARE)


def gate_rule(bundle: AnalysisContextBundle) -> GateRule | None:
    """Which rule of the verdict chain stops a reading over ``bundle`` being read, in the chain's order: the setup lost
    too many trials, a member run did not finish, or, with more than one arm, an arm has too few usable cases. None
    when it can be read (one arm, or enough cases), or when no trial ran, which is nothing to read rather than a gate.

    :param bundle: The aspect's evidence.
    :return: The rule, or None.
    """
    boards = bundle.scoreboard
    if boards and setup_failed(bundle):
        return "setup"
    if bundle_stopped_short(bundle) is not None:
        return "unfinished"
    if len(boards) > 1 and any(board.usable_cases < SMALL_N_BAND_FLOOR for board in boards):
        return "too_small"
    return None
