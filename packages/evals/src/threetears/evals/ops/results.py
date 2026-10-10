"""One run's results, listed light and read one part at a time: which cell is which, and what one of them stored.

``run_get`` answers how a run came out — its counts and each measure's mean — and stops there, so a cell
that came out wrong could be found but not read: what its candidate produced, what its kind recorded about
the calls it made, which roles spent what, were stored and reachable through no surface but the database.
:func:`results_list` names the run's results, a light row each, so an operator can pick one;
:func:`result_get` reads that one back as stored.

**The listing is light, and the read is cut into parts by weight.** A result row is small; its trace is
not — the candidate's output, what the judge read and the spans are most of a cell's bytes, which is why
the store keeps them in a sibling document no list or aggregate reads
(:class:`~threetears.evals.contracts.models.EvalTrace`). So the listing reads results alone and pages them,
and a read returns one :data:`ResultPart` of the trace: the ``record`` by default — the output the kind
stored, its call ledger and the world's end state, beside the result itself — and the judge's evidence and
the spans only when asked for by name. The judge's evidence restates the conversation as the judge read it
and the spans can outweigh everything else, so neither rides on a read that did not ask for it, in the
text or in the structured payload.

**The output is returned as the kind stored it, and never interpreted here.** The engine does not know what
a kind's output documents hold: whatever the kind wrote about each turn — the reply, the tools it called,
whether a call succeeded and what the tool answered — is there exactly as far as the kind wrote it, and
nothing more. A kind that records only what succeeded leaves no trace of a refused call in its output.
Projecting a typed "actions" view out of the documents would mean the engine guessing a kind's shape, which
is the one thing the trace's contract forbids. The call ledger beside them is the kind's other record, and
holds only the calls that succeeded (:mod:`threetears.evals.contracts.call_ledger`): the two are different
facts, so both are returned and neither stands in for the other.

**Every read is in the caller's scope.** A run or a result in another scope is not found, as every other
read answers it — never an empty listing, which would read as a run that produced nothing.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from threetears.evals.analysis.reporting import LatencyPartition, decompose_total_ms
from threetears.evals.contracts import (
    CallLedger,
    CellTermination,
    EvalResult,
    JudgedArtifact,
    JudgeEvidence,
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

#: Which part of a stored result :func:`result_get` returns. ``record`` is the result with what its kind
#: stored about the cell (output, call ledger, end state); ``judge`` is what the judge was sent; ``spans`` is
#: the harvested spans.
ResultPart = Literal["record", "judge", "spans"]

#: Whether a result's trace can be read: ``stored``; ``none`` — the cell wrote none (its record says so);
#: or ``missing`` — the record says one was written and no document backs it, which is a fault, never an
#: ordinary absence (:func:`~threetears.evals.run.get_result_trace` logs it).
TraceState = Literal["stored", "none", "missing"]


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
    total_ms: float | None = Field(
        description="Wall-clock of the candidate's turns, in milliseconds; None when nothing timed them."
    )
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
    has_trace: bool = Field(description="Whether a trace was stored, which result_get reads in parts.")


class ResultListing(EvalBaseModel):
    """One page of a run's results, in a stable order: by case, then repeat, then id."""

    run_id: str
    condition_filter: ResultOutcome | None = Field(description="The condition listed; None lists every result.")
    total: int = Field(description="How many of the run's results match the filter, across every page.")
    offset: int = Field(description="Where this page starts in that order.")
    limit: int | None = Field(description="The most rows a page holds; None for every row from the offset.")
    next_offset: int | None = Field(description="The offset of the next page; None when this page is the last.")
    results: list[ResultLine]


class TraceRecord(EvalBaseModel):
    """The ``record`` part of a stored trace: what the kind stored about the cell, without the two heavy parts."""

    output: list[dict[str, Any]] = Field(
        description="The candidate's output documents exactly as its kind stored them; what each holds is the kind's."
    )
    call_ledger: CallLedger | None = Field(
        description="The calls the kind recorded as succeeded; None when the kind keeps no ledger."
    )
    end_state: dict[str, Any] | None = Field(
        description="The world the cell left behind, by dimension; None when nothing was read."
    )
    judged_artifact: JudgedArtifact | None = Field(
        description="What the judge read, by the kind's declaration; None when nothing was sent to a judge. Read "
        "the evidence itself with part judge."
    )
    span_count: int = Field(description="How many spans were stored; read them with part spans.")


class TraceJudge(EvalBaseModel):
    """The ``judge`` part of a stored trace: what the judge was sent, as the kind rendered it."""

    judged_artifact: JudgedArtifact
    evidence: JudgeEvidence


class ResultDetail(EvalBaseModel):
    """One part of one stored result: the record and its condition always, and the trace part asked for.

    ``result`` is the stored record as written — its per-role usage rows (``usage``) and every error field
    among it — which is small, so every part carries it and names whose part it is. Exactly one of
    ``record``, ``judge`` and ``spans`` can be set, the one ``part`` names, and it is ``None`` when the trace
    holds nothing of that part or ``trace_state`` says there is no trace to read.
    """

    part: ResultPart = Field(description="Which part of the trace this read returns.")
    result: EvalResult
    condition: ResultCondition
    trace_state: TraceState = Field(
        description="stored; none — the cell wrote no trace; or missing — its record says one was written and "
        "no document backs it."
    )
    record: TraceRecord | None = Field(
        default=None, description="With part record: what the kind stored about the cell."
    )
    judge: TraceJudge | None = Field(
        default=None, description="With part judge: what the judge was sent; None when nothing was."
    )
    spans: list[dict[str, Any]] | None = Field(default=None, description="With part spans: the stored spans.")
    latency_partition: LatencyPartition = Field(
        description="The result's total_ms split into llm_ms, tool_ms and the orchestration_ms remainder, or the "
        "sentence saying why the split is withheld — derived on each read by the one function the analysis uses, "
        "never stored."
    )


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
        total_ms=None if result.latency is None else result.latency.total_ms,
        goal_checks_passed=None if counted is None else sum(passed for _, passed in counted),
        goal_checks=len(result.goal_state_outcomes),
        judge_scores={score.dim: score.score for score in result.judge_scores()},
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


def result_get(host: EvalHost, result_id: str, scope_id: str, *, part: ResultPart = "record") -> ResultDetail:
    """One stored result and one part of its trace — what the cell recorded, as its kind stored it.

    Two point reads on the result's own partition: the result, then its trace when its record says one was
    written (:func:`~threetears.evals.run.get_result_trace`, which logs a record whose trace no document
    backs; it reads here as ``missing``, never as none stored). Nothing is recomputed but the condition and
    the latency partition, each by the one function every surface asks
    (:func:`~threetears.evals.contracts.resolve_result_condition`,
    :func:`~threetears.evals.analysis.reporting.decompose_total_ms`), so each reads here as it reads anywhere
    else. The whole trace document is read whichever part is asked for: the bound is on what is
    returned, which is what a reader's context pays for.

    Args:
        host: The host whose store holds the result.
        result_id: The result.
        scope_id: The scope it lives in.
        part: The part of the trace to return (:data:`ResultPart`).

    Returns:
        The result, its condition and the part asked for.

    Raises:
        NotFoundError: No result with that id in the scope — another type's id among them.
    """
    result = get_result(host.storage, result_id, scope_id)
    trace = get_result_trace(host.storage, result) if result.has_trace else None
    state: TraceState = "stored" if trace is not None else "missing" if result.has_trace else "none"
    record: TraceRecord | None = None
    judge: TraceJudge | None = None
    spans: list[dict[str, Any]] | None = None
    if trace is not None and part == "record":
        record = TraceRecord(
            output=trace.trace,
            call_ledger=trace.call_ledger,
            end_state=trace.end_state,
            judged_artifact=trace.judged_artifact,
            span_count=len(trace.otel_trace),
        )
    elif trace is not None and part == "judge" and trace.judge_evidence and trace.judged_artifact is not None:
        judge = TraceJudge(judged_artifact=trace.judged_artifact, evidence=trace.judge_evidence)
    elif trace is not None and part == "spans":
        spans = trace.otel_trace
    return ResultDetail(
        part=part,
        result=result,
        condition=resolve_result_condition(result),
        trace_state=state,
        record=record,
        judge=judge,
        spans=spans,
        latency_partition=decompose_total_ms(result.latency),
    )


__all__ = [
    "ResultDetail",
    "ResultLine",
    "ResultListing",
    "ResultPart",
    "TraceJudge",
    "TraceRecord",
    "TraceState",
    "result_get",
    "results_list",
]
