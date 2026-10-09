"""One run's results, listed light and read whole: which cell is which, and what one of them stored.

``run_get`` answers how a run came out — its counts and each measure's mean — and stops there, so a cell
that came out wrong could be found but not read: what its candidate did, which of its actions the world
refused and what the tool answered, which roles spent what, were stored and reachable through no
surface but the database. :func:`results_list` names the run's results, a light row each, so an operator
can pick one; :func:`result_get` reads that one back as stored, its trace beside it.

**The listing is light and the read is whole, on purpose.** A result row is small; its trace is not —
the candidate's output and spans are most of a cell's bytes, which is why the store keeps them in a
sibling document no list or aggregate reads (:class:`~threetears.evals.contracts.models.EvalTrace`). So the
listing reads results alone and pages them, and only the one read a person asked for pays the second
point read.

**The trace is returned as the kind stored it, and never interpreted here.** The engine does not know
what a kind's output documents hold: for a kind whose candidate acts on tools they carry each action with
whatever the kind recorded about it — whether it succeeded, what the tool said — and for a classifier a
label. Projecting a typed "actions" view out of them would mean the engine guessing a kind's shape, which
is the one thing the trace's contract forbids; returning them verbatim means an action the kind recorded
as failed reads as failed, in the kind's own words. The call ledger beside them is the kind's other
record, and holds only the calls that succeeded (:mod:`threetears.evals.contracts.call_ledger`): the two
are different facts, so both are returned and neither stands in for the other.

**Every read is in the caller's scope.** A run or a result in another scope is not found, as every other
read answers it — never an empty listing, which would read as a run that produced nothing.
"""

from __future__ import annotations

from pydantic import Field

from threetears.evals.contracts import (
    CellTermination,
    EvalResult,
    EvalTrace,
    ResultCondition,
    ResultOutcome,
    classify_result,
    counted_goal_verdicts,
    resolve_result_condition,
)
from threetears.evals.contracts.base import EvalBaseModel
from threetears.evals.contracts.errors import ValidationFailedError
from threetears.evals.contracts.host import EvalHost
from threetears.evals.run import get_result, get_result_trace, get_run, list_results


class ResultLine(EvalBaseModel):
    """One result, as a run's listing shows it: where it sits, the condition it is in, and its headline measures."""

    id: str = Field(description="The result's id, which result_get reads.")
    test_case_id: str = Field(description="The case the cell measured.")
    k_iteration: int = Field(description="Which repeat of the case, from 1.")
    model: str = Field(description="The arm's candidate model.")
    variant_key: str = Field(description="The contestant the cell measured, as the analysis pools it.")
    condition: ResultOutcome = Field(
        description="How the result counts: ok (the candidate delivered; scored), candidate_fail (scored as a hard "
        "fail) or infra_exclude (a harness fault; in no aggregate)."
    )
    termination: CellTermination = Field(description="How the cell ended as work, as the runner recorded it.")
    cost_usd: float | None = Field(description="The cell's blended spend; None when a call in it went unpriced.")
    goal_checks_passed: int | None = Field(
        description="Goal checks counted as passed, as every rate counts them (all failed on a candidate failure); "
        "None for a harness fault, whose checks enter no rate."
    )
    goal_checks: int = Field(description="Goal checks the result carries.")
    judge_scores: dict[str, int] = Field(
        description="Each judged dimension's score as the judge gave it, the two reserved axes among them; the "
        "condition says whether it counts."
    )
    host_measures: dict[str, bool | float | str] = Field(description="The host's own measures, as stored.")
    has_trace: bool = Field(description="Whether a trace was stored, which result_get returns beside the result.")


class ResultListing(EvalBaseModel):
    """One page of a run's results, in a stable order: by case, then repeat, then id."""

    run_id: str
    condition_filter: ResultOutcome | None = Field(description="The condition listed; None lists every result.")
    total: int = Field(description="How many of the run's results match the filter, across every page.")
    offset: int = Field(description="Where this page starts in that order.")
    limit: int | None = Field(description="The most rows a page holds; None for every row from the offset.")
    next_offset: int | None = Field(description="The offset of the next page; None when this page is the last.")
    results: list[ResultLine]


class ResultDetail(EvalBaseModel):
    """One stored result read whole: the record, the condition it resolves to, and its trace.

    ``result`` is the stored record as written, its per-role usage rows (``usage``) and every error field
    among it. ``trace`` is the sibling payload as written — the candidate's output documents, the call
    ledger, what the judge read, the world's end state and the spans — or ``None`` when the cell stored none.
    """

    result: EvalResult
    condition: ResultCondition
    trace: EvalTrace | None = Field(description="The stored trace; None when the cell wrote none.")


def _judge_scores(result: EvalResult) -> dict[str, int]:
    """Each score the judge gave, by dimension: the rubric's, then the transcript and outcome axes when scored."""
    scores = [*result.rubric_scores, result.transcript_score, result.outcome_score]
    return {score.dim: score.score for score in scores if score is not None}


def _line(result: EvalResult) -> ResultLine:
    counted = counted_goal_verdicts(result)
    return ResultLine(
        id=result.id,
        test_case_id=result.test_case_id,
        k_iteration=result.k_iteration,
        model=result.model,
        variant_key=result.variant_key,
        condition=classify_result(result),
        termination=result.termination,
        cost_usd=result.cost_usd,
        goal_checks_passed=None if counted is None else sum(passed for _, passed in counted),
        goal_checks=len(result.goal_state_outcomes),
        judge_scores=_judge_scores(result),
        host_measures=dict(result.host_measures),
        has_trace=result.has_trace,
    )


def results_list(
    host: EvalHost,
    run_id: str,
    scope_id: str,
    *,
    condition: ResultOutcome | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> ResultListing:
    """One run's results, a light row each, paged in a stable order.

    The order is by case, then repeat, then result id, so a page boundary falls in the same place on every
    read of a finished run and the repeats of one case sit together. A run still going is still adding
    results, and a page read while it runs can shift.

    The run is read first, so a run that is not in the scope is refused as not found rather than listed
    as empty — an empty page would say the run produced nothing.

    Args:
        host: The host whose store holds the run.
        run_id: The run.
        scope_id: The scope it lives in.
        condition: List only results in this condition; ``None`` lists every one.
        offset: How many rows of the order to skip.
        limit: The most rows to return; ``None`` returns every row from ``offset``. A surface that hands
            the page to an agent bounds it (the ``results_list`` action does).

    Returns:
        The page.

    Raises:
        NotFoundError: No run with that id in the scope.
        ValidationFailedError: ``offset`` is negative, or ``limit`` is below 1.
    """
    if offset < 0:
        raise ValidationFailedError(f"offset is {offset}; a page starts at 0 or later")
    if limit is not None and limit < 1:
        raise ValidationFailedError(f"limit is {limit}; a page holds at least 1 row, or None for every row")
    run = get_run(host.storage, run_id, scope_id)
    results = sorted(
        list_results(host.storage, run.id, scope_id),
        key=lambda result: (result.test_case_id, result.k_iteration, result.id),
    )
    matching = [result for result in results if condition is None or classify_result(result) is condition]
    end = len(matching) if limit is None else offset + limit
    return ResultListing(
        run_id=run.id,
        condition_filter=condition,
        total=len(matching),
        offset=offset,
        limit=limit,
        next_offset=end if end < len(matching) else None,
        results=[_line(result) for result in matching[offset:end]],
    )


def result_get(host: EvalHost, result_id: str, scope_id: str) -> ResultDetail:
    """One stored result, read whole with its trace — what its candidate did, as the cell recorded it.

    Two point reads on the result's own partition: the result, then its trace when it records one
    (:func:`~threetears.evals.run.get_result_trace`, which warns when the record claims a trace no document
    backs). Nothing is recomputed but the condition, which is resolved by the one function every surface
    asks (:func:`~threetears.evals.contracts.resolve_result_condition`), so its disclosure reads here as it
    reads anywhere else.

    Args:
        host: The host whose store holds the result.
        result_id: The result.
        scope_id: The scope it lives in.

    Returns:
        The result, its condition and its trace.

    Raises:
        NotFoundError: No result with that id in the scope.
    """
    result = get_result(host.storage, result_id, scope_id)
    trace = get_result_trace(host.storage, result) if result.has_trace else None
    return ResultDetail(result=result, condition=resolve_result_condition(result), trace=trace)


__all__ = ["ResultDetail", "ResultLine", "ResultListing", "result_get", "results_list"]
