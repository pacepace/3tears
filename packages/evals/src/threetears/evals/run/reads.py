"""The run package's reads: list a scope's runs, and read a run's results and a result's trace.

These are the reads every read surface stands on — the run listing the comparison lenses narrow,
the results a run produced, and the trace a drill-down opens — written as functions over
:class:`~threetears.evals.kernel.storage.ResultStore` (the listing, over the host) and typed
parameters, so a client of the package reads its runs without a host's service. A host's service may keep its own method names and
delegate here, so every surface it serves reaches these through it.

The scope a run lives in is ``scope_id`` here — the engine's word for a partition it never
interprets. The host chooses what a scope is and passes it through.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from threetears.evals.kernel.errors import NotFoundError, ValidationFailedError
from threetears.evals.kernel.storage import RunsRead
from threetears.evals.kernel.status_filter import StatusFilterError, validate_status_filter
from threetears.evals.schema.logs import get_logger
from threetears.evals.kernel.remedies import FIND_RESULT

if TYPE_CHECKING:
    from threetears.evals.kernel.host.eval_host import EvalHost
    from threetears.evals.schema.models import EvalResult, EvalRun, EvalRunStatus, EvalTrace
    from threetears.evals.kernel.storage import ResultStore

log = get_logger(__name__)


def _listing_status_filter(status: str | None) -> EvalRunStatus | None:
    """Resolve :func:`list_runs`'s status filter, refusing an unknown one as caller input.

    The twin of the comparison lenses' status seam (``_status_filter`` in
    :mod:`threetears.evals.analysis.reads`) for the one surface where an UNSPECIFIED
    filter means "every run" rather than ``completed`` — see
    :func:`~threetears.evals.kernel.status_filter.validate_status_filter` for why those two
    defaults cannot honestly be one function.

    What is duplicated between the two is four lines of ``try``/``except``; what
    is not duplicated is everything that would drift — the accepted vocabulary
    and the refusal message both live once, in the kernel, and both of these
    resolve through it. Collapsing the pair behind a keyword argument was the
    alternative and is worse: it makes the comparison default reachable by
    omission from a listing surface, which is precisely the mirror image of the
    listing regression described below.

    Before this existed, ``list_runs`` handed the raw string to the storage
    predicate: ``status=all`` was matched as a literal status, so a scope of
    9 runs answered HTTP 200 with an empty list — as did ``status=banana``.

    Args:
        status: Raw filter from the caller, straight off the wire.

    Returns:
        ``None`` to list every run, else the status to filter on.

    Raises:
        ValidationFailedError: The value is not a status a run can carry, nor
            ``"all"``.
    """
    try:
        return validate_status_filter(status)
    except StatusFilterError as e:
        raise ValidationFailedError(e.fault, remedy=e.remedy) from e


def list_runs(
    host: EvalHost,
    scope_id: str,
    *,
    status: str | None = None,
    include_archived: bool = False,
    label: str | None = None,
) -> list[EvalRun]:
    """List eval runs in a scope, optionally filtered by status.

    **Archived runs are excluded by default**, and this is the single seam the
    QUALITY surfaces — ``comparison_sets``, ``pivot``, ``frontier``,
    ``history``, export — inherit that exclusion from rather than each
    re-deriving it. Pass ``include_archived=True`` to see them; ``get_run``
    always reads one regardless, so an archived run is never unreachable.

    **The COST surfaces deliberately opt back in** (``program_budget`` and the engine's launch pricer,
    ``history_launch_pricer``, both pass ``include_archived=True``). Archiving is a
    measurement curation, not a financial one: a run retired because its
    observation is junk still spent its dollars. The rule: quality views exclude
    them, budget views never do.
    The split is intentional — do not unify it to make this docstring shorter.

    The archive predicate runs in Python over what the store returns: the store's
    run query filters on scope and status only.

    **A listed run's payload is incomplete by the host's declaration.** Every run in a
    scope is hydrated here, and a host that freezes something heavy into each run's
    payload (the subject's recent turn memory, say, most of every stored run) would
    otherwise pay it once per run per listing — enough to run a production process out of
    memory. ``HostProfile.listing_elisions`` names what the store leaves out,
    each returned run records it, and the host's reader of a left-out value refuses. Read
    one run whole (through the host's single-run read) for the whole payload.

    **An unspecified status lists every run, and ``"all"`` says so out loud.**
    This is the one status-taking surface whose unspecified filter is not
    ``completed``: it is an inventory, not a verdict, and an operator opens it
    precisely to find the run that failed. ``all`` is therefore a synonym for
    omitting the argument here rather than a widening — but it is accepted, and
    must be, because it is the spelling the comparison surfaces teach and a
    caller carrying it over used to get a silent empty list back.
    Resolution and the refusal of anything else live in
    :func:`_listing_status_filter`.

    Args:
        host: The host: its storage, and the listing elisions its profile declares.
        scope_id: Partition key — the scope whose runs to list.
        status: Optional run status filter (e.g. ``"completed"``). Unspecified,
            blank, or ``"all"`` (any case) lists every run.
        include_archived: Include operator-archived runs (default: exclude).
        label: Only runs an operator gave this label (``EvalRun.labels``), compared after stripping it; None lists
            every run. Like the archive predicate, it runs in Python over what the store returns.

    Returns:
        Eval runs in the scope matching the filters.

    Raises:
        ValidationFailedError: ``status`` is not one a run can carry, nor
            ``"all"``.
        UnreadableRunError: A run the listing matched does not validate under this build's model; the inventory
            that leaves it out and names it instead is :func:`list_runs_disclosing`.
    """
    runs = host.storage.query_eval_runs(
        scope_id, status=_listing_status_filter(status), elide_payload=host.profile.listing_elisions
    )
    return _narrowed(runs, include_archived=include_archived, label=label)


def list_runs_disclosing(
    host: EvalHost,
    scope_id: str,
    *,
    status: str | None = None,
    include_archived: bool = False,
    label: str | None = None,
) -> RunsRead:
    """:func:`list_runs` for an inventory: every run that reads, and each matched row that does not, named.

    :func:`list_runs` refuses a scope holding a run this build's model cannot read
    (:class:`~threetears.evals.kernel.UnreadableRunError`), which is right for the quality and cost surfaces that
    stand on it: a comparison or a budget over the runs that happened to read is a wrong answer, not a partial one.
    An inventory is the one reader better served by the rest: the operator opening it is looking for a run, and one
    row nothing can read must not hide every other. So the refused rows come back beside the runs, never dropped,
    for the surface to state (``runs_list`` counts them and names ``runs_unreadable``).

    A refused row is matched by its stored status alone: whether it is archived or carries ``label`` is a field of a
    document that does not read, so the archive and label filters cannot be applied to it, and it is returned
    whatever they ask.

    Args:
        host: As :func:`list_runs` takes it.
        scope_id: As :func:`list_runs` takes it.
        status: As :func:`list_runs` takes it.
        include_archived: As :func:`list_runs` takes it.
        label: As :func:`list_runs` takes it.

    Returns:
        The runs listed, and each matched row the model refused.

    Raises:
        ValidationFailedError: ``status`` is not one a run can carry, nor ``"all"``.
        UnreadableRunError: A run read, but writing it lost was refused by a model.
    """
    read = host.storage.query_eval_runs_isolating(
        scope_id, status=_listing_status_filter(status), elide_payload=host.profile.listing_elisions
    )
    return RunsRead(
        runs=_narrowed(read.runs, include_archived=include_archived, label=label), unreadable=read.unreadable
    )


def _narrowed(runs: list[EvalRun], *, include_archived: bool, label: str | None) -> list[EvalRun]:
    """``runs`` narrowed by the listing's label and archive filters, which run in Python over what the store returned."""
    if label is not None:
        runs = [run for run in runs if label.strip() in run.labels]
    if include_archived:
        return runs
    return [run for run in runs if not run.archived]


def list_results(storage: ResultStore, run_id: str, scope_id: str) -> list[EvalResult]:
    """List every result for one eval run within its scope.

    Args:
        storage: Eval storage backend.
        run_id: The eval run whose results to list.
        scope_id: Partition key — the scope the run lives in.

    Returns:
        Every :class:`~threetears.evals.schema.models.EvalResult` produced by the run.
    """
    return storage.query_eval_results_by_run(run_id, scope_id)


def get_result(storage: ResultStore, result_id: str, scope_id: str) -> EvalResult:
    """Load one eval result by id within its scope.

    Args:
        storage: Eval storage backend.
        result_id: The result's UUID.
        scope_id: Partition key — the scope the result lives in.

    Returns:
        The loaded :class:`~threetears.evals.schema.models.EvalResult`.

    Raises:
        NotFoundError: No result with that id in the scope.
    """
    result = storage.load_eval_result(result_id, scope_id)
    if result is None:
        raise NotFoundError("eval result", result_id, remedy=FIND_RESULT)
    return result


def get_result_trace(storage: ResultStore, result: EvalResult) -> EvalTrace | None:
    """Load one result's trace payload, or ``None`` when it stored none.

    Separate from :func:`get_result` because it is a second point read, and only the
    two drill-downs want it — every list and aggregate surface reads results that never
    carry it. Callers deciding whether detail EXISTS should read ``EvalResult.has_trace``
    rather than calling this and checking for ``None``: that is what the marker is for,
    and it costs no read.

    **Takes the result rather than its ids so the marker can be checked against
    reality in one place.** ``has_trace=True`` with no document is not an ordinary
    absence — the marker is written from the payload write's own outcome, so the two
    disagreeing means a payload was deleted without its result, or a write landed
    half. Both read surfaces would otherwise render that as "this result has no
    trace", silently, which is the one reading that is certainly wrong.

    Args:
        storage: Eval storage backend.
        result: The result whose payload to load.

    Returns:
        The payload, or ``None`` when the result recorded none.
    """
    payload = storage.load_eval_trace(result.id, result.scope_id)
    if payload is None and result.has_trace:
        log.warning(
            "Eval result %s (run %s, scope %s) claims a stored trace but no eval_trace document exists; "
            "rendering it as absent",
            result.id,
            result.eval_run_id,
            result.scope_id,
        )
    return payload


__all__ = [
    "get_result",
    "get_result_trace",
    "list_results",
    "list_runs",
    "list_runs_disclosing",
]
