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


from threetears.evals.schema.out_of_run_spend import KnownOutOfRunPurpose, OutOfRunSpend


class OutOfRunSpendTotals(EvalBaseModel):
    """What a set of out-of-run calls spent, summed — with what could not be summed counted beside it.

    **Missing is not zero.** A call that reported no cost, and every call that raised (which reports
    nothing, and may have been billed), are counted in ``n_unpriced`` and left out of ``priced_usd``,
    so ``priced_usd`` is a floor whenever ``n_unpriced`` is not 0. A row that ledgers several calls (a report's
    turns, ``OutOfRunSpend.calls``) counts each of them, its priced share in ``priced_usd`` and its unpriced calls in
    ``n_unpriced``.

    :ivar n_calls: The calls.
    :ivar n_raised: Those that raised rather than returning.
    :ivar n_unpriced: Those whose cost is unknown — a completed call that reported none, or a raised call.
    :ivar priced_usd: The reported cost of the priced calls, summed.
    :ivar ceiling_usd: The ceilings the calls were admitted at, summed over those the client could price.
    :ivar n_unbounded: Those admitted with no ceiling (the client could not price them and no cap was enforced).
    :ivar n_budgeted: Those a report's budget admitted (``OutOfRunSpend.admitted_by``), each turn on its own price
        ceiling, which the ledger does not hold.
    """

    n_calls: int
    n_raised: int
    n_unpriced: int
    priced_usd: float
    ceiling_usd: float
    n_unbounded: int
    n_budgeted: int = 0

    @classmethod
    def of(cls, rows: Sequence[OutOfRunSpend]) -> OutOfRunSpendTotals:
        """Sum ``rows``.

        :param rows: Ledger rows.
        :ptype rows: Sequence[OutOfRunSpend]
        :return: Their totals.
        :rtype: OutOfRunSpendTotals
        """
        return cls(
            n_calls=sum(_calls(row) for row in rows),
            n_raised=sum(_raised(row) for row in rows),
            n_unpriced=sum(_unpriced(row) for row in rows),
            priced_usd=math.fsum(_priced(row) for row in rows),
            ceiling_usd=math.fsum(row.priced_ceiling_usd for row in rows if row.priced_ceiling_usd is not None),
            n_unbounded=sum(_calls(row) for row in rows if row.priced_ceiling_usd is None and row.admitted_by is None),
            n_budgeted=sum(_calls(row) for row in rows if row.admitted_by is not None),
        )


def _calls(row: OutOfRunSpend) -> int:
    """The calls one row ledgers: its ``calls``, or one."""
    return row.calls if row.calls is not None else 1


def _raised(row: OutOfRunSpend) -> int:
    """The raised calls of one row: its ``raised_calls``, or one when its one call raised."""
    if row.raised_calls is not None:
        return row.raised_calls
    return 1 if row.outcome == "raised" else 0


def _unpriced(row: OutOfRunSpend) -> int:
    """The unpriced calls of one row: its ``unpriced_calls``, or one when its one call has no cost."""
    if row.unpriced_calls is not None:
        return row.unpriced_calls
    return 1 if row.cost_usd is None else 0


def _priced(row: OutOfRunSpend) -> float:
    """What the priced calls of one row reported: its ``priced_usd``, or its one call's cost."""
    if row.priced_usd is not None:
        return row.priced_usd
    return row.cost_usd if row.cost_usd is not None else 0.0


class OutOfRunSpendReport(EvalBaseModel):
    """The calls the engine made outside any run in a scope — case generations, rubric proposals and analysis
    generations — and their totals.

    Read off the out-of-run ledger (``EvalStorage.query_out_of_run_spend``), the one record of spend no run's
    cost carries: a run's results sum what its cells spent, never what was spent writing its cases.

    :ivar scope_id: The scope read.
    :ivar purpose: The purpose the read was narrowed to, or ``None`` for every purpose.
    :ivar launch_group_id: The launch the read was narrowed to, or ``None`` for every launch.
    :ivar template_id: The template the read was narrowed to, or ``None`` for every template.
    :ivar rows: Every matching call, oldest first.
    :ivar totals: Every matching call, summed.
    :ivar by_purpose: The totals per purpose that appears, in the order purposes first appear.
    :ivar by_launch: The totals per launch group that appears (case generations; a proposal or an analysis belongs to
        none, and is not listed here), in the order launches first appear.
    """

    scope_id: str
    purpose: KnownOutOfRunPurpose | None
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
    purpose: KnownOutOfRunPurpose | None = None,
    launch_group_id: str | None = None,
    template_id: str | None = None,
) -> OutOfRunSpendReport:
    """What the engine spent outside any run in a scope, call by call and summed, optionally narrowed.

    :param host: The host whose store holds the ledger.
    :ptype host: EvalHost
    :param scope_id: The scope to read.
    :ptype scope_id: str
    :param purpose: Only calls made for this purpose (``variation``, ``proposer`` or ``analysis``).
    :ptype purpose: KnownOutOfRunPurpose | None
    :param launch_group_id: Only the calls one launch's case generation made; its runs carry the same group id.
    :ptype launch_group_id: str | None
    :param template_id: Only calls made for this template.
    :ptype template_id: str | None
    :return: The report.
    :rtype: OutOfRunSpendReport
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
    """One line of totals, saying when the sum is a floor, then how the calls were admitted, each way counted.

    The ceiling sum is said of the calls it was summed over alone (those admitted at a priced ceiling), and left out
    where none was: a call a report budget admitted, each turn on a ceiling the ledger does not hold, and a call
    admitted unbounded, are each counted, never read as a ceiling of $0.
    """
    floor = f", {totals.n_unpriced} unpriced (so at least)" if totals.n_unpriced else ""
    raised = f", {totals.n_raised} raised" if totals.n_raised else ""
    reported = f"{totals.n_calls} call(s): ${totals.priced_usd:.4f} reported{floor}{raised}"
    ceilinged = totals.n_calls - totals.n_unbounded - totals.n_budgeted
    admitted = [
        *([f"{ceilinged} admitted at up to ${totals.ceiling_usd:.4f}"] if ceilinged else []),
        *([f"{totals.n_budgeted} admitted by a report budget"] if totals.n_budgeted else []),
        *([f"{totals.n_unbounded} admitted unbounded"] if totals.n_unbounded else []),
    ]
    return f"{reported}; {', '.join(admitted)}" if admitted else reported


def out_of_run_spend_text(report: OutOfRunSpendReport) -> str:
    """The scope's out-of-run spend as an operator reads it: the totals, per purpose and launch, then each call.

    :param report: What :func:`scope_out_of_run_spend` returned.
    :ptype report: OutOfRunSpendReport
    :return: The text.
    :rtype: str
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
        admitted = f"admitted by {row.admitted_by}" if row.admitted_by is not None else f"ceiling {ceiling}"
        if row.calls is not None:
            # A row of several calls: how many, and the priced share beside the unpriced, never one figure for both.
            unpriced = f", {row.unpriced_calls} unpriced (so at least)" if row.unpriced_calls else ""
            cost = f"{row.calls} calls: ${_priced(row):.4f} reported{unpriced}"
        failed = f" {row.failure}" if row.failure else ""
        lines.append(
            f"  {row.created_at}  {row.purpose}  {row.model}  {row.outcome}{failed}  {cost} ({admitted})"
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
