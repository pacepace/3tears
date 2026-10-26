"""The analyst's fact operations: each reads one snapshot's frozen report input and lists what it holds.

Each operation reads the snapshot's input (:func:`~threetears.evals.ops.analyst.snapshot_input`, refusing one whose
input moved or no longer reads) and runs its computation (:mod:`threetears.evals.analysis.analyst.facts`) on the
host's blocking executor. A paged operation given ``fits`` (a test of whether a page's text is short enough,
supplied by the caller that renders it) returns the largest page within ``limit`` that passes it, so a page of long
rows is shorter rather than overlong; the rows it leaves off are the next page's. A computation's refusal of a filter naming nothing the snapshot holds reaches the caller as
a :class:`~threetears.evals.kernel.errors.ValidationFailedError` carrying its remedy. Every operation is a pure read:
the same snapshot and parameters always return the same result.

The result models are the computation's, re-exported here for the actions layer, which reads no analysis module.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from threetears.evals.analysis.analyst.facts import (
    CONTRASTS_LIMIT,
    EXAMPLES_LIMIT,
    MEASURE_LIMIT,
    READINGS_LIMIT,
    CampaignOverview,
    CaveatListing,
    ContrastListing,
    ExampleListing,
    MeasureBy,
    MeasureFigures,
    MeasureStatistic,
    ReadingsListing,
)
from threetears.evals.analysis.analyst import facts
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.errors import NotFoundError, RefusedInput, ValidationFailedError
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.kernel.report_snapshot import FIND_SNAPSHOT
from threetears.evals.ops.analyst import snapshot_input
from threetears.evals.ops.host import OpsHost

__all__ = [
    "CONTRASTS_LIMIT",
    "EXAMPLES_LIMIT",
    "MEASURE_LIMIT",
    "READINGS_LIMIT",
    "CampaignOverview",
    "CaveatListing",
    "ContrastListing",
    "ExampleListing",
    "MeasureBy",
    "MeasureFigures",
    "MeasureStatistic",
    "ReadingsListing",
    "campaign_overview",
    "caveats_list",
    "contrasts_list",
    "examples_list",
    "measure_read",
    "readings_list",
]


def _refused[T](compute: Callable[[], T]) -> T:
    """Run one computation, its refusal of what the caller asked turned into the service's. Blocking.

    :raises ValidationFailedError: The computation refused a filter or a key, with its remedy.
    """
    try:
        return compute()
    except RefusedInput as refused:
        raise ValidationFailedError(refused.fault, remedy=refused.remedy) from refused


def _sized[T](compute: Callable[[int], T], limit: int, fits: Callable[[T], bool] | None) -> T:
    """The page of at most ``limit`` rows ``compute`` builds, as large as ``fits`` passes. Blocking.

    A page grows with its rows, so the largest that fits is found by halving between one row and ``limit``. A page
    of one row is returned even where it does not fit: one row is what a page always holds.
    """
    page = compute(limit)
    if fits is None or fits(page):
        return page
    low, high = 1, limit - 1
    best = compute(1)
    while low <= high:
        middle = (low + high) // 2
        candidate = compute(middle)
        if fits(candidate):
            best, low = candidate, middle + 1
        else:
            high = middle - 1
    return best


async def _read[T](host: OpsHost, scope_id: str, snapshot_id: str, compute: Callable[[ReportInput], T]) -> T:
    """One snapshot's input, and a computation run on it on the host's blocking executor."""
    held = await snapshot_input(host, scope_id, snapshot_id)
    return await run_blocking(host.eval_host.blocking_executor, _refused, lambda: compute(held))


async def campaign_overview(host: OpsHost, scope_id: str, *, snapshot_id: str) -> CampaignOverview:
    """What a snapshot holds: its campaign, arms, aspects, measures, criteria, caveats and prior insights.

    :param host: The host whose store holds the snapshot.
    :ptype host: OpsHost
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param snapshot_id: The snapshot, as the freeze named it.
    :ptype snapshot_id: str
    :return: The overview.
    :rtype: CampaignOverview
    :raises NotFoundError: No snapshot with that id in the scope.
    :raises ValidationFailedError: The snapshot does not read; the remedy is to freeze again.
    """
    return await _read(host, scope_id, snapshot_id, lambda held: facts.campaign_overview(held, snapshot_id))


async def readings_list(
    host: OpsHost,
    scope_id: str,
    *,
    snapshot_id: str,
    aspect_filter: str | None = None,
    arm_filter: str | None = None,
    limit: int = READINGS_LIMIT,
    offset: int = 0,
    fits: Callable[[ReadingsListing], bool] | None = None,
) -> ReadingsListing:
    """A page of each aspect's readings against the control, pair readings, floors and bars.

    :param host: The host whose store holds the snapshot.
    :ptype host: OpsHost
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param snapshot_id: The snapshot, as the freeze named it.
    :ptype snapshot_id: str
    :param aspect_filter: Only this aspect's rows.
    :ptype aspect_filter: str | None
    :param arm_filter: Only this arm's rows.
    :ptype arm_filter: str | None
    :param limit: The most rows on the page.
    :ptype limit: int
    :param offset: The rows to skip.
    :ptype offset: int
    :param fits: Whether a page's text is short enough; the page is the largest within ``limit`` that passes, and
        ``limit`` rows where none is given.
    :ptype fits: Callable[[ReadingsListing], bool] | None
    :return: The page.
    :rtype: ReadingsListing
    :raises NotFoundError: No snapshot with that id in the scope.
    :raises ValidationFailedError: The snapshot does not read, or a filter names nothing it holds.
    """
    return await _read(
        host,
        scope_id,
        snapshot_id,
        lambda held: _sized(
            lambda rows: facts.readings_list(
                held, aspect_filter=aspect_filter, arm_filter=arm_filter, limit=rows, offset=offset
            ),
            limit,
            fits,
        ),
    )


async def contrasts_list(
    host: OpsHost,
    scope_id: str,
    *,
    snapshot_id: str,
    arm_filter: str | None = None,
    measure_filter: str | None = None,
    aspect_filter: str | None = None,
    limit: int = CONTRASTS_LIMIT,
    offset: int = 0,
    fits: Callable[[ContrastListing], bool] | None = None,
) -> ContrastListing:
    """A page of the corrected contrasts, and the contrast families correcting them.

    :param host: The host whose store holds the snapshot.
    :ptype host: OpsHost
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param snapshot_id: The snapshot, as the freeze named it.
    :ptype snapshot_id: str
    :param arm_filter: Only this arm's contrasts.
    :ptype arm_filter: str | None
    :param measure_filter: Only the contrasts on this measure or judged dimension.
    :ptype measure_filter: str | None
    :param aspect_filter: Only the contrasts on this aspect.
    :ptype aspect_filter: str | None
    :param limit: The most contrasts on the page.
    :ptype limit: int
    :param offset: The contrasts to skip.
    :ptype offset: int
    :param fits: Whether a page's text is short enough; the page is the largest within ``limit`` that passes, and
        ``limit`` contrasts where none is given.
    :ptype fits: Callable[[ContrastListing], bool] | None
    :return: The page.
    :rtype: ContrastListing
    :raises NotFoundError: No snapshot with that id in the scope.
    :raises ValidationFailedError: The snapshot does not read, or a filter names nothing it holds.
    """
    return await _read(
        host,
        scope_id,
        snapshot_id,
        lambda held: _sized(
            lambda rows: facts.contrasts_list(
                held,
                arm_filter=arm_filter,
                measure_filter=measure_filter,
                aspect_filter=aspect_filter,
                limit=rows,
                offset=offset,
            ),
            limit,
            fits,
        ),
    )


async def measure_read(
    host: OpsHost,
    scope_id: str,
    *,
    snapshot_id: str,
    measure_key: str,
    by: MeasureBy = "arm",
    statistics: Sequence[MeasureStatistic] = ("mean", "ci", "n"),
    aspect_filter: str | None = None,
    arm_filter: str | None = None,
    limit: int = MEASURE_LIMIT,
    offset: int = 0,
    fits: Callable[[MeasureFigures], bool] | None = None,
) -> MeasureFigures:
    """A page of one measure's or judged dimension's statistics, by arm or within each stratum, task or cell.

    :param host: The host whose store holds the snapshot.
    :ptype host: OpsHost
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param snapshot_id: The snapshot, as the freeze named it.
    :ptype snapshot_id: str
    :param measure_key: The measure or judged dimension, by key.
    :ptype measure_key: str
    :param by: What to read it by.
    :ptype by: MeasureBy
    :param statistics: The statistics to read.
    :ptype statistics: Sequence[MeasureStatistic]
    :param aspect_filter: Only this aspect.
    :ptype aspect_filter: str | None
    :param arm_filter: Only this arm.
    :ptype arm_filter: str | None
    :param limit: The most rows on the page.
    :ptype limit: int
    :param offset: The rows to skip.
    :ptype offset: int
    :param fits: Whether a page's text is short enough; the page is the largest within ``limit`` that passes, and
        ``limit`` rows where none is given.
    :ptype fits: Callable[[MeasureFigures], bool] | None
    :return: The page.
    :rtype: MeasureFigures
    :raises NotFoundError: No snapshot with that id in the scope.
    :raises ValidationFailedError: The snapshot does not read; the key names no measure; a filter names nothing it
        holds; it holds no slice of that kind; or a statistic or filter does not apply to the read.
    """
    return await _read(
        host,
        scope_id,
        snapshot_id,
        lambda held: _sized(
            lambda rows: facts.measure_read(
                held,
                measure_key,
                by=by,
                statistics=statistics,
                aspect_filter=aspect_filter,
                arm_filter=arm_filter,
                limit=rows,
                offset=offset,
            ),
            limit,
            fits,
        ),
    )


async def examples_list(
    host: OpsHost,
    scope_id: str,
    *,
    snapshot_id: str,
    arm_filter: str | None = None,
    stratum_filter: str | None = None,
    aspect_filter: str | None = None,
    outcome_filter: str | None = None,
    limit: int = EXAMPLES_LIMIT,
    offset: int = 0,
    fits: Callable[[ExampleListing], bool] | None = None,
) -> ExampleListing:
    """A page of the failed trials the snapshot froze, each with its stored result where the bundle named it.

    :param host: The host whose store holds the snapshot.
    :ptype host: OpsHost
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param snapshot_id: The snapshot, as the freeze named it.
    :ptype snapshot_id: str
    :param arm_filter: Only this arm's examples.
    :ptype arm_filter: str | None
    :param stratum_filter: Only the examples in this stratum or task.
    :ptype stratum_filter: str | None
    :param aspect_filter: Only this aspect's examples.
    :ptype aspect_filter: str | None
    :param outcome_filter: Only the examples of this failure class.
    :ptype outcome_filter: str | None
    :param limit: The most examples on the page.
    :ptype limit: int
    :param offset: The examples to skip.
    :ptype offset: int
    :param fits: Whether a page's text is short enough; the page is the largest within ``limit`` that passes, and
        ``limit`` examples where none is given.
    :ptype fits: Callable[[ExampleListing], bool] | None
    :return: The page.
    :rtype: ExampleListing
    :raises NotFoundError: No snapshot with that id in the scope.
    :raises ValidationFailedError: The snapshot does not read, or a filter names nothing it holds.
    """
    held = await snapshot_input(host, scope_id, snapshot_id)
    eval_host = host.eval_host
    snapshot = await run_blocking(
        eval_host.blocking_executor, eval_host.storage.load_report_snapshot, snapshot_id, scope_id
    )
    if snapshot is None:
        raise NotFoundError("report snapshot", snapshot_id, remedy=FIND_SNAPSHOT)
    results = dict(snapshot.example_result_ids)
    return await run_blocking(
        eval_host.blocking_executor,
        _refused,
        lambda: _sized(
            lambda rows: facts.examples_list(
                held,
                results,
                arm_filter=arm_filter,
                stratum_filter=stratum_filter,
                aspect_filter=aspect_filter,
                outcome_filter=outcome_filter,
                limit=rows,
                offset=offset,
            ),
            limit,
            fits,
        ),
    )


async def caveats_list(host: OpsHost, scope_id: str, *, snapshot_id: str) -> CaveatListing:
    """Every caveat the snapshot carries, where it may land, whether it is immaterial, and what it touches.

    :param host: The host whose store holds the snapshot.
    :ptype host: OpsHost
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param snapshot_id: The snapshot, as the freeze named it.
    :ptype snapshot_id: str
    :return: The caveats.
    :rtype: CaveatListing
    :raises NotFoundError: No snapshot with that id in the scope.
    :raises ValidationFailedError: The snapshot does not read; the remedy is to freeze again.
    """
    return await _read(host, scope_id, snapshot_id, facts.caveats_list)
