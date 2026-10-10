"""The run-status filter a read surface takes: its vocabulary, its two defaults and its refusal.

A caller names the runs a read covers by status — ``completed``, ``failed``, ``all`` — and every
surface that takes one resolves it here, so the accepted set, the ``all`` sentinel and the message
refusing anything else exist once. Two packages read it and may not import each other: the run
listing in :mod:`threetears.evals.run.reads` (where an unspecified filter means every run) and the
comparison lenses in :mod:`threetears.evals.analysis.reads` (where it means ``completed``). Contracts
is the package both rows of the dependency matrix admit, and the vocabulary is a rule over
:data:`~threetears.evals.schema.models.RUN_STATUSES`, which already lives here.
"""

from __future__ import annotations

from typing import TypeIs

from threetears.evals.schema.models import RUN_STATUSES, EvalRunStatus

# Default run-status filter for comparability grouping: an in-flight run has no
# stable case set to compare on.
DEFAULT_COMPARISON_STATUS: EvalRunStatus = "completed"

#: The sentinel that lifts the status filter. Not a run status — no run is ever
#: stored carrying it — which is why it is the one accepted value that is
#: case-folded: it is a word the caller types, never a value matched against
#: stored data.
ALL_STATUSES = "all"


class StatusFilterError(ValueError):
    """A run-status filter was supplied that no run can carry.

    Raised rather than passed through, for the reason
    :class:`~threetears.evals.analysis.lenses.history.HistoryError` is
    raised rather than returned as an empty series: a status outside
    :data:`~threetears.evals.schema.models.RUN_STATUSES` matches no run, so the answer is a
    well-formed EMPTY one that reads exactly like the truthful "no runs of that
    kind exist". A typo and a fact rendered identically, and on the surfaces that
    name a verdict.

    Its sibling ``metric`` already refused this way — ``?metric=banana`` is a 422
    naming the three it accepts — while ``?status=banana`` returned 200 with an
    empty body and the whole corpus booked to ``results_outside_queried_runs``.
    One seam, one behaviour.
    """


def _is_run_status(value: str) -> TypeIs[EvalRunStatus]:
    """Whether ``value`` is a status a run can carry."""
    return value in RUN_STATUSES


def validate_status_filter(status: str | None) -> EvalRunStatus | None:
    """Map a caller's status filter to a service argument, where unspecified means UNFILTERED.

    The whole of the status vocabulary — the accepted set, the ``all`` sentinel,
    and the refusal message — lives here, and :func:`normalize_status_filter` is
    this function plus one default. Two entry points rather than one, because the
    two seams that take a ``status`` disagree, correctly, about what an ABSENT
    one means:

    - The five COMPARISON surfaces name a verdict, so an unspecified filter
      narrows to ``completed`` — an in-flight run has no stable case set to
      compare on. That is :func:`normalize_status_filter`.
    - ``list_runs`` is an INVENTORY, so an unspecified filter shows every run.
      Narrowing it to ``completed`` would hide exactly the failed, cancelled and
      budget-stopped runs an operator opens the list to find. That is
      this function.

    Split by name rather than expressed as one function with a
    default-carrying keyword, deliberately: a defaulted parameter puts the
    comparison behaviour one forgotten keyword away from a listing surface, and
    the resulting regression — a listing quietly narrowed to ``completed`` — is
    the silent-empty-answer class this seam exists to remove, not announce.

    **Real statuses are matched exactly; only ``"all"`` folds case.** That split
    is not an oversight: ``"all"`` is a sentinel this function invents, so it
    never has to equal anything stored, while every other accepted value is
    compared by exact equality against ``EvalRun.status`` downstream. Every other
    enumerated argument a read surface takes refuses the same way — an unknown
    ``metric`` and an unknown ``fmt``
    (:func:`~threetears.evals.analysis.lenses.export.serialize_export`) are both exact-match. Folding ``"Completed"``
    instead of refusing it would buy nothing now that a refusal names the
    vocabulary: the operator learns ``completed`` in one round-trip either way.

    Args:
        status: Raw filter from the caller. ``None``, empty, or ``"all"`` (any
            case) all mean every run; anything else must be a status a run can
            actually carry.

    Returns:
        ``None`` to apply no filter at all, else the status to filter on.

    Raises:
        StatusFilterError: The value is neither ``"all"`` nor a member of
            :data:`~threetears.evals.schema.models.RUN_STATUSES`.
    """
    cleaned = (status or "").strip()
    if not cleaned or cleaned.lower() == ALL_STATUSES:
        return None
    if not _is_run_status(cleaned):
        raise StatusFilterError(
            f"unknown run status {cleaned!r} — expected one of {', '.join(sorted(RUN_STATUSES))}, "
            f"or {ALL_STATUSES!r} for every run regardless of status"
        )
    return cleaned


def normalize_status_filter(status: str | None) -> EvalRunStatus | None:
    """Map a caller's status filter for the COMPARISON surfaces, identically everywhere.

    :func:`validate_status_filter` plus the one thing that differs between the
    two status seams: an unspecified filter here means ``completed``, not "every
    run". Read that function for the vocabulary, the ``all`` sentinel, the
    refusal, and why the two defaults are two named functions instead of one
    keyword.

    Lives here rather than at each adapter because the two surfaces already
    disagreed once: an empty ``?status=`` meant "all runs" over REST and
    "completed" over MCP, which is the silent-divergence class the parity gate
    exists to catch. One seam, two callers — and one place the validation can be
    added without five adapters each growing their own copy of it.

    Args:
        status: Raw filter from the caller. ``None`` or empty means unspecified;
            ``"all"`` (any case) lifts the filter entirely; anything else must be
            a status a run can actually carry.

    Returns:
        ``None`` to group every run, else the status to filter on.

    Raises:
        StatusFilterError: The value is neither ``"all"`` nor a member of
            :data:`~threetears.evals.schema.models.RUN_STATUSES`.
    """
    if not (status or "").strip():
        return DEFAULT_COMPARISON_STATUS
    return validate_status_filter(status)


__all__ = [
    "ALL_STATUSES",
    "DEFAULT_COMPARISON_STATUS",
    "StatusFilterError",
    "normalize_status_filter",
    "validate_status_filter",
]
