"""Run completeness: the sentence a short or degraded run must carry wherever its numbers are read.

:func:`completeness_disclosure` says how much of a run's planned work produced a usable measurement, and
:func:`degraded_run_disclosures` collects that sentence for every degraded run a projection pools.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

    Args:
        completeness: The run's stored record, or ``None``.

    Returns:
        The disclosure, rendered verbatim by every surface, or ``None`` when
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

    Args:
        runs: The runs behind an answer. Order is irrelevant; keying is by id.

    Returns:
        ``run_id -> disclosure`` for each degraded run, omitting every run that
        delivered its whole matrix and every run carrying no completeness record
        (a record's absence is not evidence of a shortfall — see
        :func:`completeness_disclosure`). Empty means nothing needs disclosing,
        so a caller can use emptiness as the predicate rather than re-deriving it.
    """
    return {run.id: sentence for run in runs if (sentence := completeness_disclosure(run.completeness)) is not None}


__all__ = [
    "completeness_disclosure",
    "DEGRADED_RUN_CLAUSE",
    "degraded_run_disclosures",
    "RECONSTRUCTED_COUNTS_CLAUSE",
]
