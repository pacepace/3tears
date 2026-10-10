"""What the engine spent outside any run in a scope: the out-of-run ledger read, summed and rendered.

:func:`scope_out_of_run_spend` reads the ledger (``EvalStorage.query_out_of_run_spend``) and sums it per
purpose and per launch; :func:`out_of_run_spend_text` is the one rendering every surface shows — an
action, the command line — with what could not be priced counted beside each sum, so a floor never
reads as a total.
"""

from __future__ import annotations

import math

from collections.abc import Sequence


from threetears.evals.schema.base import EvalBaseModel


from threetears.evals.kernel.host import EvalHost


from threetears.evals.schema.out_of_run_spend import OutOfRunPurpose, OutOfRunSpend


class OutOfRunSpendTotals(EvalBaseModel):
    """What a set of out-of-run calls spent, summed — with what could not be summed counted beside it.

    **Missing is not zero.** A call that reported no cost, and every call that raised (which reports
    nothing, and may have been billed), are counted in ``n_unpriced`` and left out of ``priced_usd``,
    so ``priced_usd`` is a floor whenever ``n_unpriced`` is not 0.

    Attributes:
        n_calls: The calls.
        n_raised: Those that raised rather than returning.
        n_unpriced: Those whose cost is unknown — a completed call that reported none, or a raised call.
        priced_usd: The reported cost of the priced calls, summed.
        ceiling_usd: The ceilings the calls were admitted at, summed over those the client could price.
        n_unbounded: Those admitted with no ceiling (the client could not price them and no cap was enforced).
    """

    n_calls: int
    n_raised: int
    n_unpriced: int
    priced_usd: float
    ceiling_usd: float
    n_unbounded: int

    @classmethod
    def of(cls, rows: Sequence[OutOfRunSpend]) -> OutOfRunSpendTotals:
        """Sum ``rows``.

        Args:
            rows: Ledger rows.

        Returns:
            Their totals.
        """
        return cls(
            n_calls=len(rows),
            n_raised=sum(1 for row in rows if row.outcome == "raised"),
            n_unpriced=sum(1 for row in rows if row.cost_usd is None),
            priced_usd=math.fsum(row.cost_usd for row in rows if row.cost_usd is not None),
            ceiling_usd=math.fsum(row.priced_ceiling_usd for row in rows if row.priced_ceiling_usd is not None),
            n_unbounded=sum(1 for row in rows if row.priced_ceiling_usd is None),
        )


class OutOfRunSpendReport(EvalBaseModel):
    """The calls the engine made outside any run in a scope — case generations, rubric proposals and analysis
    generations — and their totals.

    Read off the out-of-run ledger (``EvalStorage.query_out_of_run_spend``), the one record of spend no run's
    cost carries: a run's results sum what its cells spent, never what was spent writing its cases.

    Attributes:
        scope_id: The scope read.
        purpose: The purpose the read was narrowed to, or ``None`` for every purpose.
        launch_group_id: The launch the read was narrowed to, or ``None`` for every launch.
        template_id: The template the read was narrowed to, or ``None`` for every template.
        rows: Every matching call, oldest first.
        totals: Every matching call, summed.
        by_purpose: The totals per purpose that appears, in the order purposes first appear.
        by_launch: The totals per launch group that appears (case generations; a proposal or an analysis belongs to none, and
            is not listed here), in the order launches first appear.
    """

    scope_id: str
    purpose: OutOfRunPurpose | None
    launch_group_id: str | None
    template_id: str | None
    rows: list[OutOfRunSpend]
    totals: OutOfRunSpendTotals
    by_purpose: dict[str, OutOfRunSpendTotals]
    by_launch: dict[str, OutOfRunSpendTotals]


def scope_out_of_run_spend(
    host: EvalHost,
    scope_id: str,
    *,
    purpose: OutOfRunPurpose | None = None,
    launch_group_id: str | None = None,
    template_id: str | None = None,
) -> OutOfRunSpendReport:
    """What the engine spent outside any run in a scope, call by call and summed, optionally narrowed.

    Args:
        host: The host whose store holds the ledger.
        scope_id: The scope to read.
        purpose: Only calls made for this purpose (``variation``, ``proposer`` or ``analysis``).
        launch_group_id: Only the calls one launch's case generation made; its runs carry the same group id.
        template_id: Only calls made for this template.

    Returns:
        The report.
    """
    rows = host.storage.query_out_of_run_spend(
        scope_id, purpose=purpose, launch_group_id=launch_group_id, template_id=template_id
    )
    by_purpose: dict[str, list[OutOfRunSpend]] = {}
    by_launch: dict[str, list[OutOfRunSpend]] = {}
    for row in rows:
        by_purpose.setdefault(row.purpose, []).append(row)
        if row.launch_group_id is not None:
            by_launch.setdefault(row.launch_group_id, []).append(row)
    return OutOfRunSpendReport(
        scope_id=scope_id,
        purpose=purpose,
        launch_group_id=launch_group_id,
        template_id=template_id,
        rows=rows,
        totals=OutOfRunSpendTotals.of(rows),
        by_purpose={name: OutOfRunSpendTotals.of(group) for name, group in by_purpose.items()},
        by_launch={name: OutOfRunSpendTotals.of(group) for name, group in by_launch.items()},
    )


def _totals_line(totals: OutOfRunSpendTotals) -> str:
    """One line of totals, saying when the sum is a floor."""
    floor = f", {totals.n_unpriced} unpriced (so at least)" if totals.n_unpriced else ""
    raised = f", {totals.n_raised} raised" if totals.n_raised else ""
    unbounded = f", {totals.n_unbounded} admitted unbounded" if totals.n_unbounded else ""
    return (
        f"{totals.n_calls} call(s): ${totals.priced_usd:.4f} reported{floor}{raised}; admitted at up to "
        f"${totals.ceiling_usd:.4f}{unbounded}"
    )


def out_of_run_spend_text(report: OutOfRunSpendReport) -> str:
    """The scope's out-of-run spend as an operator reads it: the totals, per purpose and launch, then each call.

    Args:
        report: What :func:`scope_out_of_run_spend` returned.

    Returns:
        The text.
    """
    narrowed = ", ".join(
        f"{name} {value}"
        for name, value in (
            ("purpose", report.purpose),
            ("launch", report.launch_group_id),
            ("template", report.template_id),
        )
        if value is not None
    )
    lines = [f"out-of-run spend in scope {report.scope_id}" + (f" ({narrowed})" if narrowed else "")]
    if not report.rows:
        lines.append("no out-of-run call is ledgered here")
        return "\n".join(lines)
    lines.append(f"total: {_totals_line(report.totals)}")
    lines += [f"purpose {name}: {_totals_line(totals)}" for name, totals in report.by_purpose.items()]
    lines += [f"launch {name}: {_totals_line(totals)}" for name, totals in report.by_launch.items()]
    for row in report.rows:
        cost = f"${row.cost_usd:.4f}" if row.cost_usd is not None else "unpriced"
        ceiling = f"${row.priced_ceiling_usd:.4f}" if row.priced_ceiling_usd is not None else "unbounded"
        failed = f" {row.failure}" if row.failure else ""
        lines.append(
            f"  {row.created_at}  {row.purpose}  {row.model}  {row.outcome}{failed}  {cost} (ceiling {ceiling})"
            f"  template {row.template_id}  launch {row.launch_group_id}"
            + (f"  campaign {row.campaign_id}" if row.campaign_id is not None else "")
        )
    return "\n".join(lines)


__all__ = [
    "OutOfRunSpendReport",
    "OutOfRunSpendTotals",
    "out_of_run_spend_text",
    "scope_out_of_run_spend",
]
