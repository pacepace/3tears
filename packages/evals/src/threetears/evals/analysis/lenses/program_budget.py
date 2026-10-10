"""The program-budget view: spend, never excluding what quality excludes.

:func:`compute_program_budget` totals every run's spend, including the runs a quality view drops by default,
so the budget a program reads is the money it spent rather than the money behind the numbers it kept.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.models import NON_TERMINAL_RUN_STATUSES
from threetears.evals.analysis.reporting import _subject_id_of, pooled_cost_compositions

if TYPE_CHECKING:
    from threetears.evals.schema.models import (
        EvalResult,
        EvalRun,
    )


# A run whose status a quality view drops BY DEFAULT. The read-tier quality
# surfaces (pivot / frontier / history) default to `status="completed"`, so any
# other status is spend that is invisible to them until a caller passes
# `status='all'` — at which point the run is admitted carrying its DEGRADED
# completeness disclosure. Named as a constant because the budget view's whole
# point is the *complement* of this set: the runs whose dollars are real even
# though their numbers carry no quality signal under the default cohort.
#
# "By default" is load-bearing and was once missing: this view stated that a
# stopped run "will never enter a quality view", which `status='all'` disproves
# in one call. Nor is the complement of this set the set of runs a quality
# surface can safely pool — a `completed` run that lost a cell to a harness
# exclusion is short too, and no status filter can see it. That is
# `RunCompleteness`'s question, not this constant's.
_QUALITY_INCLUDED_STATUS = "completed"


class BudgetRun(EvalBaseModel):
    """One run's real spend, counted whatever its status.

    Budget is the one lens that never excludes. A cancelled, failed, or
    budget-stopped run spent real dollars — an ``exhausted`` run carries its
    accumulated cost, never zero — and dropping it, as every quality view
    deliberately does by default, would under-report the bill. ``cumulative_cost_usd`` is
    the running total in ``created_at`` order, so the series answers "what had
    this scope cost by run N".

    ``subject_id`` / ``subject_label`` are carried as context, never as a
    partition: dollars are dollars across subjects (unlike composite quality,
    which is never pooled), so the budget totals sum over every subject. A
    blank ``subject_id`` is fine here — a run whose subject was never captured
    still spent money, and excluding it is exactly what a budget view must not
    do.
    """

    run_id: str
    status: str
    subject_id: str = ""
    subject_label: str = ""
    cost_usd: float
    n_results: int
    #: This run's results and re-judges whose spend could not be priced — real spend absent from
    #: ``cost_usd``, which sums the priced ones and is a floor whenever this is non-zero.
    n_unpriced: int
    created_at: str
    cumulative_cost_usd: float
    #: Which roles this run's dollars were summed over — the distinct
    #: ``EvalResult.cost_roles`` its results recorded. Normally one entry: a run resolves
    #: its cost convention once at launch. Empty when the run has no results.
    cost_compositions: list[list[str]] = []


class ProgramBudget(EvalBaseModel):
    """Program-lens spend over a scope, cancelled/failed/exhausted included.

    The program cost lens is *everything* — candidate, inner-agent,
    external, judge, simulator — which is what ``EvalResult.cost_usd`` already
    holds (the authoritative blended total). "Everything" is the lens, not
    necessarily the dollars: search providers report no per-call figure, so
    ``external`` contributes money only for a run whose operator declared the
    account's credit rate, and is counted-but-unpriced otherwise. That is why
    this view totals across a boundary it has to name: ``cost_compositions``
    lists the distinct role sets pooled into ``total_cost_usd``, and more than
    one entry means the number spans a change in what a dollar figure includes.
    Budget sums across **every** run
    regardless of status, which is the axis on which this view differs from the
    quality surfaces: they exclude non-``completed`` runs because an in-flight or
    failed run has no stable quality signal, and budget never does because the
    spend was real either way.

    ``n_incomplete_runs`` / ``incomplete_cost_usd`` make that asymmetry legible
    on the wire rather than only in the totals: they count the spend a quality
    view would have dropped. That bucket is then SPLIT, because its two halves
    mean opposite things to anyone deciding about a budget: an in-flight run's
    dollars have not bought a quality signal *yet* and normally will, while a
    cancelled or failed run's dollars never will. Reported as one number, the
    live half reads as waste. ``unattributed_cost_usd`` is real spend on results
    whose run is not in the listed set — normally zero within one scope, and
    surfaced rather than silently folded away so the never-exclude guarantee is
    structural, not a comment.
    """

    runs: list[BudgetRun] = []
    total_cost_usd: float = 0.0
    n_runs: int = 0
    n_incomplete_runs: int = 0
    incomplete_cost_usd: float = 0.0
    #: The half of the incomplete bucket that is still under way — ``pending`` or
    #: ``running``. Dropped by a quality view because a run that has not finished has
    #: no stable signal yet, not because the money went nowhere.
    n_in_flight_runs: int = 0
    in_flight_cost_usd: float = 0.0
    #: The statuses actually present in the in-flight half, sorted and distinct. Carried
    #: rather than left to a reader's enumeration of the status vocabulary: a surface that
    #: names the bucket from a hardcoded list describes a set it never looked at, and goes
    #: quietly wrong the day a status is added.
    in_flight_statuses: list[str] = []
    #: The half that stopped without completing — cancelled, failed, budget-stopped, and
    #: any terminal status added later. These dollars are final: the run is not coming
    #: back, so whatever it bought is all it will ever buy. That is a statement about the
    #: RUN and deliberately no longer one about the quality views: those default to
    #: ``status="completed"`` and so skip it, but ``status='all'`` admits it, where it
    #: arrives carrying its DEGRADED completeness disclosure. The earlier wording said no
    #: quality view would ever read it, which one call disproves.
    n_terminal_incomplete_runs: int = 0
    terminal_incomplete_cost_usd: float = 0.0
    #: The statuses actually present in the terminal half, sorted and distinct — same
    #: reason as :attr:`in_flight_statuses`.
    terminal_incomplete_statuses: list[str] = []
    unattributed_cost_usd: float = 0.0
    #: Distinct role sets the pooled ``total_cost_usd`` was summed over, sorted. One entry
    #: is a total whose parts mean the same thing; two or more is a total spanning a
    #: composition change, which is a real caveat on any comparison drawn across it.
    cost_compositions: list[list[str]] = []
    #: The part of the total spent re-judging results after their runs finished
    #: (``EvalResult.judge_rescores``). In the total and in each run's row because it is real
    #: program spend on that run's results; kept apart here because it is not in any result's
    #: ``cost_usd``, which measures the cell as it ran.
    rejudge_cost_usd: float = 0.0
    #: Results and re-judges across the scope whose spend could not be priced. Every dollar
    #: figure above sums the priced ones only, so a non-zero count makes each of them a floor that
    #: understates the bill by an amount nobody knows — reported beside them, never blended in.
    n_unpriced: int = 0


def compute_program_budget(runs: list[EvalRun], results: list[EvalResult]) -> ProgramBudget:
    """Sum program-lens spend per run over a scope, excluding nothing.

    Deliberately does **not** route through :func:`project_score_records`: that
    projection drops results whose run has no captured subject, which for spend
    would under-count the bill — the one thing a budget view must never do. It
    aggregates ``EvalResult.cost_usd`` (the authoritative blended total) directly
    instead, so a subject-less or failed run's dollars are still counted, plus each
    result's re-judge spend (``judge_rescores``), which that total leaves out.

    Args:
        runs: Every run in the scope, all statuses. Order is irrelevant; the
            rows are emitted in ``created_at`` order for the cumulative series.
        results: Every result in the scope. A result's ``cost_usd`` and its
            re-judge spend are attributed to its ``eval_run_id``.

    Returns:
        A :class:`ProgramBudget` — one :class:`BudgetRun` per listed run (zero
        cost for a run with no results), the cumulative series, and the
        incomplete-spend / unattributed-spend accounting that keeps the total
        honest about what it includes. The incomplete spend is reported both
        whole and split into its in-flight and terminally-incomplete halves,
        each with the statuses it was actually made of.
    """
    costs_by_run: dict[str, list[float]] = {}
    # Each run's results and re-judges whose spend went unpriced: in no dollar figure, so counted.
    unpriced_by_run: dict[str, int] = {}
    # Grouped alongside the dollars rather than looked up per row later, so a row's
    # disclosure is derived from the same results its total was summed from and the two
    # cannot come to describe different sets.
    results_by_run: dict[str, list[EvalResult]] = {}
    # A re-judge's spend rides with the result it re-scored: real program spend on that run,
    # which the result's own ``cost_usd`` deliberately leaves out.
    rejudge_cost = 0.0
    for result in results:
        spends = [result.cost_usd, *(rescore.cost_usd for rescore in result.judge_rescores)]
        result_rejudge = sum(rescore.cost_usd for rescore in result.judge_rescores if rescore.cost_usd is not None)
        rejudge_cost += result_rejudge
        costs_by_run.setdefault(result.eval_run_id, []).append(sum(spend for spend in spends if spend is not None))
        unpriced_by_run[result.eval_run_id] = unpriced_by_run.get(result.eval_run_id, 0) + spends.count(None)
        results_by_run.setdefault(result.eval_run_id, []).append(result)

    listed_ids = {run.id for run in runs}
    # Real spend on results whose run is not among those listed — never dropped,
    # merely un-rowed. Zero in the normal within-scope call; nonzero only on
    # an integrity gap, and then it belongs in the total, not the floor.
    unattributed = sum(cost for run_id, costs in costs_by_run.items() if run_id not in listed_ids for cost in costs)

    rows: list[BudgetRun] = []
    running = 0.0
    for run in sorted(runs, key=lambda r: (r.created_at, r.id)):
        costs = costs_by_run.get(run.id, [])
        cost = sum(costs)
        running += cost
        rows.append(
            BudgetRun(
                run_id=run.id,
                status=run.status,
                subject_id=_subject_id_of(run),
                subject_label=run.subject_snapshot.subject_label,
                cost_usd=cost,
                n_results=len(costs),
                n_unpriced=unpriced_by_run.get(run.id, 0),
                created_at=run.created_at,
                cumulative_cost_usd=running,
                cost_compositions=pooled_cost_compositions(results_by_run.get(run.id, [])),
            )
        )

    incomplete = [row for row in rows if row.status != _QUALITY_INCLUDED_STATUS]
    # Split by whether the run can still BECOME a quality signal. Classified against
    # NON_TERMINAL_RUN_STATUSES rather than against a list of failure statuses, for the
    # reason that constant documents: a status added to the vocabulary lands on the
    # terminal side by default, so the worst a new status can do here is be described as
    # final one release too early — never be described as failed while it is still running.
    in_flight = [row for row in incomplete if row.status in NON_TERMINAL_RUN_STATUSES]
    terminal_incomplete = [row for row in incomplete if row.status not in NON_TERMINAL_RUN_STATUSES]
    # Over EVERY result the total counted, listed and unattributed alike — the disclosure
    # has to cover the same population as the number it qualifies, or it would vouch for
    # dollars it never looked at.
    compositions = pooled_cost_compositions(results)
    return ProgramBudget(
        runs=rows,
        total_cost_usd=running + unattributed,
        n_runs=len(rows),
        n_incomplete_runs=len(incomplete),
        incomplete_cost_usd=sum(row.cost_usd for row in incomplete),
        n_in_flight_runs=len(in_flight),
        in_flight_cost_usd=sum(row.cost_usd for row in in_flight),
        in_flight_statuses=sorted({row.status for row in in_flight}),
        n_terminal_incomplete_runs=len(terminal_incomplete),
        terminal_incomplete_cost_usd=sum(row.cost_usd for row in terminal_incomplete),
        terminal_incomplete_statuses=sorted({row.status for row in terminal_incomplete}),
        unattributed_cost_usd=unattributed,
        cost_compositions=compositions,
        rejudge_cost_usd=rejudge_cost,
        n_unpriced=sum(unpriced_by_run.values()),
    )


__all__ = [
    "BudgetRun",
    "compute_program_budget",
    "ProgramBudget",
]
