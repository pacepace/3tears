"""Orphaned runs: spend that no campaign claims, and that therefore no analysis reads.

:func:`compute_orphaned_runs` lists every run in a scope no campaign names, with what it cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.analysis.reporting import _subject_id_of

if TYPE_CHECKING:
    from threetears.evals.schema.models import (
        EvalResult,
        EvalRun,
    )


class OrphanedRun(EvalBaseModel):
    """One run belonging to no campaign, with what it cost to produce.

    ``archived`` is carried rather than filtered on, and the distinction is the
    point of the row. An archived orphan was deliberately retired by an operator;
    an un-archived one is data nobody decided anything about, which is the finding
    this view exists to surface. Collapsing the two would report a curated
    exclusion as a leak.
    """

    run_id: str
    status: str
    archived: bool
    subject_id: str = ""
    subject_label: str = ""
    template_id: str | None = None
    #: The run's priced results' spend — a floor when :attr:`n_unpriced` is non-zero.
    cost_usd: float
    n_results: int
    #: The run's results whose spend could not be priced, absent from ``cost_usd``.
    n_unpriced: int
    created_at: str


class OrphanedRunsResult(EvalBaseModel):
    """Every run in a storage scope that no campaign holds, plus what that costs.

    Campaign membership is curated, never queried — a run enters a campaign only
    because an operator attached it — so a scope accumulates runs no analysis
    surface will ever read. Nothing reported the gap: a scope can hold more than twice
    the runs its only campaign holds, and the money the unheld runs spent is invisible to
    every quality and analysis view at once.

    ``n_campaigns_scanned`` is the honest denominator: every campaign in the
    scope, which is every campaign that can hold one of its runs, since a campaign
    holds only runs in its own scope. Zero campaigns makes every run an
    orphan, which is true and reads as alarming — the count is what lets a reader
    tell that state from a real leak.
    """

    orphaned_runs: list[OrphanedRun] = []
    n_orphaned: int = 0
    n_runs_in_scope: int = 0
    orphaned_cost_usd: float = 0.0
    #: Orphans an operator archived. Real spend, and deliberately retired — counted in
    #: the total, named separately so a curated exclusion is not read as overlooked data.
    n_archived_orphans: int = 0
    archived_orphan_cost_usd: float = 0.0
    n_campaigns_scanned: int = 0
    #: Orphans' results whose spend could not be priced — absent from every dollar figure here.
    n_unpriced: int = 0


def compute_orphaned_runs(
    runs: list[EvalRun],
    results: list[EvalResult],
    campaign_run_id_sets: list[list[str]],
) -> OrphanedRunsResult:
    """Report the runs no campaign claims, and the spend they represent.

    Args:
        runs: Every run in the scope, all statuses and including archived
            ones — an archived run's dollars were still spent, and this view
            reports on spend as much as on membership.
        results: Every result in the scope; a result's ``cost_usd`` is
            attributed to its ``eval_run_id``, matching :func:`compute_program_budget`.
        campaign_run_id_sets: One ``run_ids`` list per campaign in the
            scope. Taken as a list of lists rather than a pre-flattened set
            so the campaign COUNT is derived from the same argument as the claimed
            ids — a separately-passed count could disagree with the membership it
            claims to describe.

    Returns:
        An :class:`OrphanedRunsResult`, rows in ``created_at`` order.
    """
    claimed_run_ids = {run_id for run_ids in campaign_run_id_sets for run_id in run_ids}
    costs_by_run: dict[str, list[float | None]] = {}
    for result in results:
        costs_by_run.setdefault(result.eval_run_id, []).append(result.cost_usd)

    rows: list[OrphanedRun] = []
    for run in sorted(runs, key=lambda r: (r.created_at, r.id)):
        if run.id in claimed_run_ids:
            continue
        costs = costs_by_run.get(run.id, [])
        rows.append(
            OrphanedRun(
                run_id=run.id,
                status=run.status,
                archived=run.archived,
                subject_id=_subject_id_of(run),
                subject_label=run.subject_snapshot.subject_label,
                template_id=run.template_id,
                cost_usd=sum(cost for cost in costs if cost is not None),
                n_results=len(costs),
                n_unpriced=costs.count(None),
                created_at=run.created_at,
            )
        )

    archived_rows = [row for row in rows if row.archived]
    return OrphanedRunsResult(
        orphaned_runs=rows,
        n_orphaned=len(rows),
        n_runs_in_scope=len(runs),
        orphaned_cost_usd=sum(row.cost_usd for row in rows),
        n_archived_orphans=len(archived_rows),
        archived_orphan_cost_usd=sum(row.cost_usd for row in archived_rows),
        n_campaigns_scanned=len(campaign_run_id_sets),
        n_unpriced=sum(row.n_unpriced for row in rows),
    )


__all__ = [
    "compute_orphaned_runs",
    "OrphanedRun",
    "OrphanedRunsResult",
]
