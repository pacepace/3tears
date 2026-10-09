"""Turning eval results and cells into the numbers a run reports.

**The placement rule, stated once so a later reader can recover it: every pure function
that turns eval results into the numbers a run reports lives here — unless its declared
vocabulary belongs to one host's tool, in which case it lives with the catalogue that
declares that vocabulary.** Purity is the criterion, and the module name is narrower than
its contents on purpose: latency and cost are not scores, and renaming a module that
external requirements already reference costs more than the mismatch does. The exception
half is not a hedge — it is what puts a host tool's own rollup (a conclusion-lifecycle summary, say)
in that host's adapter rather than here, and without it
that absence reads as an oversight.

So: what a single result scored (:func:`result_composite`), what a set scored together
(:func:`compute_pass_hat_k`, :func:`compute_composite_summary`, :func:`compute_dimension_summary`),
whether the loop delivered the matrix its run promised (:func:`summarize_completeness`,
:func:`reconstruct_completeness`), and what a set of results cost and how long it took
(:func:`compute_cost_summary`, :func:`compute_latency_summary`). Every one is a pure function
over eval models — give it the same results and it gives the same answer, with no storage,
no clock and no host in reach.

They live here rather than beside the run loop that produces their inputs, and the reason is
structural rather than tidiness: they are the contract a result is READ through, by the run
package and the analysis package alike, and the matrix lets analysis import contracts and not
run. Nothing in them names a host's subject, surface or model, so every package
that needs a composite reaches them without reaching the code that drives a candidate.

:class:`CellSummary` travels with them for the same reason and one more: it is
what :func:`summarize_completeness` counts, it is bounded by construction
(a run's retained footprint is ``O(cells)``, never ``O(cells × trace volume)``),
and every field on it is an eval model or a scalar.

Not to be confused with two neighbours that answer different questions:
:mod:`threetears.evals.contracts.metrics` describes what a measure *is* — its family, unit and
how far its meaning travels — and computes no values; :mod:`threetears.evals.analysis.stats`
holds dependency-free statistical primitives (Student-t, standard error) that
know no eval model at all.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict

from threetears.evals.contracts.models import (
    CellTermination,
    EvalResult,
    EvalRun,
    LatencyMetrics,
    RubricScore,
    RunCompleteness,
)
from threetears.evals.contracts.result_condition import (
    ResultOutcome,
    candidate_failure_cause,
    classify_result,
    counted_rubric_scores,
    trial_exclusion,
)
from threetears.evals.contracts.usage_capture import production_replicating_cost, resolve_result_usage


@dataclass(frozen=True, slots=True)
class CellSummary:
    """What a run retains per cell after the full result is durably persisted.

    Every field is bounded in size, so a run's retained footprint is
    ``O(cells)`` rather than ``O(cells × trace volume)``. That distinction is
    the point of this type: :class:`~threetears.evals.contracts.models.EvalResult` carries
    the full turn-by-turn ``trace`` and the harvested ``otel_trace``, and both
    grow with how much the candidate actually did. Retaining whole results for
    the length of a run made peak memory scale with the sweep's total trace
    volume, which is a curve that meets any fixed container limit at some
    matrix size.

    This is a *pointer plus the coordinates a caller can act on* — never a
    second copy of the record. To read a cell's transcript, scores or spans,
    load it back through the storage layer by :attr:`result_id`
    (:meth:`~threetears.evals.contracts.storage.EvalStorage.load_eval_result`) — but check
    :attr:`persisted` first, because a pointer is only as good as the write it
    points at, and ``save_eval_result`` reports a failed write by returning
    ``False`` rather than raising.
    """

    #: Id of the persisted :class:`~threetears.evals.contracts.models.EvalResult` — the
    #: handle for reading the full record back out of storage.
    result_id: str
    test_case_id: str
    model: str
    k_iteration: int
    #: How the cell participates in scoring (ok / candidate fail / infra
    #: exclude), classified from the persisted result's structured error
    #: fields rather than re-derived by any caller.
    outcome: ResultOutcome
    #: How the cell's execution ended.
    termination: CellTermination
    #: The result's ``cost_usd`` — ``None`` when its spend could not be priced.
    cost_usd: float | None
    #: Whether the write this summary points at actually landed. ``False`` means
    #: the cell ran and produced a result that no longer exists anywhere: the
    #: storage layer logs and returns ``False`` on a write failure instead of
    #: raising, and the run continues. Recorded rather than inferred, because
    #: this is the one field whose ``False`` a reader cannot recover by loading
    #: the record — there is no record to load.
    persisted: bool

    @classmethod
    def from_result(cls, result: EvalResult, *, persisted: bool) -> CellSummary:
        """Summarize a persisted result down to its bounded coordinates.

        Args:
            result: the result that was just written.
            persisted: what ``storage.save_eval_result`` reported. Required
                rather than defaulted, so a new call site has to state whether
                the write was checked instead of inheriting an optimistic
                ``True``.
        """
        return cls(
            result_id=result.id,
            test_case_id=result.test_case_id,
            model=result.model,
            k_iteration=result.k_iteration,
            outcome=classify_result(result),
            termination=result.termination,
            cost_usd=result.cost_usd,
            persisted=persisted,
        )


def summarize_completeness(run: EvalRun, cells: Sequence[CellSummary]) -> RunCompleteness:
    """Count what a finished run loop delivered against the matrix the run promised.

    The denominator is read from the RUN document rather than from the cells,
    because the question is whether the loop delivered what the run committed to
    — a count derived from the same list the loop walked would be internally
    consistent no matter which cases went missing before it started.

    ``infra_excluded_cells`` is counted over the *persisted* cells only, so
    :attr:`~threetears.evals.contracts.models.RunCompleteness.measured_cells` (persisted minus
    excluded) cannot double-subtract a cell that was both excluded and lost.

    Args:
        run: The authoritative run document — the source of the promised matrix.
        cells: What :func:`~threetears.evals.run.runner.execute_run` returned, one summary per cell it ran.

    Returns:
        The counts, ready to store on the run.
    """
    persisted = [cell for cell in cells if cell.persisted]
    return RunCompleteness(
        expected_cells=run.expected_cells,
        produced_cells=len(cells),
        persisted_cells=len(persisted),
        infra_excluded_cells=sum(1 for cell in persisted if cell.outcome is ResultOutcome.INFRA_EXCLUDE),
        counted_from="run_loop",
    )


def reconstruct_completeness(run: EvalRun, results: Sequence[EvalResult]) -> RunCompleteness:
    """Count what a run delivered from the rows it left behind, its own tally being gone.

    For the one stop :func:`~threetears.evals.contracts.scoring.summarize_completeness` cannot describe: a hard kill
    (OOM, eviction, SIGKILL) takes the process, the loop's frame and its cell
    tally together, so the run document is repaired by a *later* process which
    never saw the loop. Its only evidence is what reached storage.

    That evidence is strictly weaker, in one specific way worth stating rather
    than smoothing over: a cell that ran and whose write was refused left no row,
    and no read of storage can tell it apart from a cell that never ran at all.
    So ``produced_cells`` is set equal to ``persisted_cells`` and the missing
    cells land wholly in ``expected - produced``. The size of the shortfall is
    right; its attribution leans toward "never ran". :attr:`counted_from
    <threetears.evals.contracts.models.RunCompleteness.counted_from>` records that this is a
    reconstruction so no reader has to infer it from the equality.

    Args:
        run: The abandoned run being reclaimed — the source of the promised matrix.
        results: Every result stored against that run.

    Returns:
        The counts, ready to store on the run.
    """
    return RunCompleteness(
        expected_cells=run.expected_cells,
        produced_cells=len(results),
        persisted_cells=len(results),
        infra_excluded_cells=sum(1 for result in results if classify_result(result) is ResultOutcome.INFRA_EXCLUDE),
        counted_from="stored_results",
    )


def capability_scores(result: EvalResult) -> list[RubricScore]:
    """The result's rubric scores on the capability axis — the ones a composite or pass^k may combine.

    A boundary dimension (:data:`~threetears.evals.contracts.models.RubricAxis`) is a guardrail: something the
    candidate must not do. Averaged with capability it lets a gain on one pay for a loss on the other, so
    every whole-trial measure reads this list and the boundary scores are decided on their own
    (:attr:`~threetears.evals.analysis.bundle.AnalysisContextBundle.guardrails`). A score judged before the
    axis was stamped (``axis`` None) is read as capability, which is how it was read then.
    """
    return [score for score in result.rubric_scores if score.axis != "boundary"]


def result_composite(result: EvalResult) -> float | None:
    """Normalize one result's capability rubric dims to a single 0–1 quality score.

    Composite = mean of the result's capability ``rubric_scores`` (:func:`capability_scores`), each put on
    0–1 by :attr:`~threetears.evals.contracts.models.RubricScore.normalized`: ``(score - 1) / 4`` for a 1–5
    dim, the 1 or 0 itself for a pass/fail one. Capability rubric dims only — a boundary dim is a guardrail
    and is never averaged in, and the reserved ``transcript_score`` / ``outcome_score`` axes are excluded,
    mirroring :func:`~threetears.evals.contracts.scoring.compute_dimension_summary`.

    Three-way by error category (candidate vs infra):
      * **infra-excluded** (``infra_error`` / ``judge_error``) → ``None``. This
        result must be DROPPED from the quality mean, not coerced to 0.0 — an
        infra failure isn't zero quality, it's *unmeasured*. Callers exclude it
        by classifying with :func:`classify_result` before calling here.
      * **candidate fail** (any cause
        :func:`~threetears.evals.contracts.result_condition.candidate_failure_cause` names — a failed
        model in ``candidate_error``, or a turn the output cap cut off) → ``0.0``. A broken
        candidate scores zero — it depresses the mean, it doesn't vanish.
      * **ok** → the 0–1 composite, or ``None`` when the result has no rubric
        dims. A rubric-group caller coerces that ``None`` to ``0.0`` (a genuine
        no-quality-signal slot can't hide); a goal-only group leaves it undefined.

    Because ``None`` is returned for BOTH the infra-excluded and the ok-no-rubric
    cases, a caller that needs to tell them apart (drop vs coerce) must call
    :func:`classify_result` itself — :func:`~threetears.evals.contracts.scoring._group_case_composites` does.

    Args:
        result: The eval result to score.

    Returns:
        The 0–1 composite, ``0.0`` for a candidate failure, or ``None`` when
        there is no scoreable quality signal (infra-excluded or no rubric dims).
    """
    outcome = classify_result(result)
    if outcome is ResultOutcome.INFRA_EXCLUDE:
        return None
    if outcome is ResultOutcome.CANDIDATE_FAIL:
        return 0.0
    scores = capability_scores(result)
    if not scores or trial_exclusion(result) is not None:
        return None
    return sum(s.normalized for s in scores) / len(scores)


def _result_passes(result: EvalResult, *, rubric_threshold: int) -> bool:
    """A result passes iff every goal-state and every capability rubric dim cleared the bar.

    A boundary dim is a guardrail, decided on its own, and takes no part (:func:`capability_scores`).

    Only decides pass/fail for a result that is not infra-excluded — callers
    (:func:`compute_pass_hat_k`) drop ``INFRA_EXCLUDE`` before calling this, so a
    judge/timeout/factory infra failure never reaches here to be floored (that
    would score infra as candidate quality).

    Two fail guards remain: a **candidate failure** is a hard fail, whatever its cause — read
    from :func:`~threetears.evals.contracts.result_condition.candidate_failure_cause`, the one place
    that names why a result is one, so a cause added there is a fail here without a second edit.
    Today that is a model the candidate's configuration runs failing (its own turn model or an
    inner agent's model; a refusal of the calling account is infra, not this) — otherwise the judge
    scores a failed transcript as a clean pass — or a turn the output cap cut off, which
    leaves a clean transcript and no ``candidate_error`` at all, so reading that field alone would
    pass it. And a result with zero scoreable outcomes is not a vacuous pass — otherwise an
    empty-score slot would inflate pass^k to 100%.
    """
    if candidate_failure_cause(result) is not None:
        return False
    scores = capability_scores(result)
    if not result.goal_state_outcomes and not scores:
        return False
    if any(not o.passed for o in result.goal_state_outcomes):
        return False
    if any(not s.clears(rubric_threshold) for s in scores):
        return False
    return True


def _already_failed(result: EvalResult, *, rubric_threshold: int) -> bool:
    """Whether what the trial has already shown decides it as a fail, whatever its unscored dims say.

    A failed goal-state check or a scored rubric dim under the bar fails the trial on its own, so
    a judge answering it could not tell on another dim leaves nothing undecided. Without this the
    exclusion meant for an undecidable trial also swallowed decided failures — a hold or act
    template lost exactly the trials it failed, and pass^k rose.
    """
    if any(not o.passed for o in result.goal_state_outcomes):
        return True
    return any(not s.clears(rubric_threshold) for s in capability_scores(result))


class PassHatPoint(TypedDict):
    """One point of the pass^k curve: the estimate at one depth, and how many cases it rests on."""

    #: The depth: the number of attempts that must ALL pass.
    k: int
    #: The mean, over the cases measured at least ``k`` times, of each case's unbiased
    #: ``C(c, k) / C(n, k)``. ``None`` when no case was measured that deep — nothing was
    #: measured at this depth, which is not a rate of zero.
    pass_hat_k: float | None
    #: The cases this point averages over: those with at least ``k`` scored attempts.
    n_cases: int


def _case_pass_hat_k(n: int, c: int, k: int) -> float:
    """The unbiased per-case estimate of pass^k from ``c`` passes in ``n >= k`` scored attempts.

    ``C(c, k) / C(n, k)`` is the share of the ``k``-subsets of the case's attempts in which every
    attempt passed (Yao et al. 2024, τ-bench). Its expectation over the case's attempts is ``p^k``
    for a case that passes each attempt with probability ``p``, whatever ``n`` is — which is what
    lets cases measured to different depths be averaged without the shallow ones flattering the
    mean, and lets extra attempts sharpen the estimate instead of making it harder to clear.
    """
    return math.comb(c, k) / math.comb(n, k)


def _pass_hat_k_curve(attempts_by_case: Iterable[Sequence[bool]]) -> list[PassHatPoint]:
    """The pass^1..pass^K curve over every case's scored attempts, ``K`` the deepest case's count.

    Each point averages only the cases measured at least ``k`` times: the estimator is undefined
    below that depth, and a case cannot stand in for a depth it was not measured at. pass^1 is
    therefore the per-case pass rate averaged over cases.
    """
    cases = [(len(attempts), sum(attempts)) for attempts in attempts_by_case if attempts]
    deepest = max((n for n, _ in cases), default=0)
    curve: list[PassHatPoint] = []
    for k in range(1, deepest + 1):
        estimates = [_case_pass_hat_k(n, c, k) for n, c in cases if n >= k]
        curve.append(
            {
                "k": k,
                "pass_hat_k": math.fsum(estimates) / len(estimates) if estimates else None,
                "n_cases": len(estimates),
            }
        )
    return curve


def pass_hat_k_at(curve: Sequence[PassHatPoint], k: int) -> PassHatPoint:
    """Read one depth off a pass^k curve, as an unmeasured point when the curve never reached it.

    Args:
        curve: A curve from :func:`compute_pass_hat_k` or :func:`pool_pass_hat_k`.
        k: The depth wanted, at least 1.

    Returns:
        The curve's point at ``k``, or ``{"k": k, "pass_hat_k": None, "n_cases": 0}`` when no
        case was measured that deep.

    Raises:
        ValueError: ``k`` is below 1, where pass^k has no meaning.
    """
    if k < 1:
        raise ValueError(f"pass^k needs k >= 1, got {k}")
    for point in curve:
        if point["k"] == k:
            return point
    return {"k": k, "pass_hat_k": None, "n_cases": 0}


def _trial_pass(result: EvalResult, *, rubric_threshold: int) -> tuple[bool | None, bool]:
    """Whether one attempt passed, ``None`` when it is left out — and whether it left as cannot-tell.

    Infra-excluded attempts (judge_error, a cell timeout charged to the rig —
    runner._DEADLINE_CHARGE —, factory/simulator/delivery infra) are left out, so they count
    toward no case's depth; a case whose every attempt is excluded vanishes from the denominator.
    A judge that could not tell on a dim the pass needs leaves the attempt out too, unless a failed
    check or a sub-bar dim has already decided it — then it is the fail it is. Candidate failures
    are attempts that failed.
    """
    exclusion = trial_exclusion(result)
    if exclusion == "judge_cannot_tell" and _already_failed(result, rubric_threshold=rubric_threshold):
        # The unscored dim cannot rescue a trial a failed check or a sub-bar dim has already
        # decided, so it counts as the fail it is rather than vanishing from the denominator.
        exclusion = None
    if exclusion is not None:
        return None, exclusion == "judge_cannot_tell"
    return classify_result(result) is ResultOutcome.OK and _result_passes(
        result, rubric_threshold=rubric_threshold
    ), False


def _pass_hat_k_entry(attempts_by_case: Mapping[Any, list[bool]], *, k: int, n_cannot_tell: int) -> dict[str, Any]:
    """The one row shape both pass^k producers return, so the two cannot describe a pool differently."""
    curve = _pass_hat_k_curve(attempts_by_case.values())
    headline = pass_hat_k_at(curve, k)
    return {
        "pass_hat_k": headline["pass_hat_k"],
        "k": k,
        "n_cases_at_k": headline["n_cases"],
        "pass_hat_k_curve": curve,
        # Counts stay counts: zero scored cases is a true zero, and the rate beside it is None.
        "n_test_cases": sum(1 for attempts in attempts_by_case.values() if attempts),
        # Iterations left out because the judge could not tell on a rubric dim — the one exclusion
        # that is not a fault, so it is counted on its own rather than vanishing.
        "n_cannot_tell_excluded": n_cannot_tell,
    }


def compute_pass_hat_k(
    results: Sequence[EvalResult],
    *,
    rubric_threshold: int = 3,
    k: int | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Estimate pass^k per ``(model, eval_run_id)`` — the chance that k attempts at a case ALL pass.

    A result *passes* when every ``goal_state_outcomes`` entry passed AND every capability
    ``rubric_scores`` entry clears the bar: at or above ``rubric_threshold`` on a 1–5 dimension, a pass
    on a pass/fail one. A boundary dimension is a guardrail and is not part of pass^k
    (:func:`capability_scores`).

    **pass^k, not pass@k.** pass@k (Chen et al. 2021) is the chance that AT LEAST ONE of k attempts
    passes; pass^k (Yao et al. 2024, τ-bench) is the chance that ALL k do — the opposite end, and
    the one a reliability claim needs. The field is ``pass_hat_k`` so it cannot be read as the other.

    **The estimator is unbiased at any depth.** Per case with ``n`` scored attempts and ``c``
    passes, pass^k is ``C(c, k) / C(n, k)``, defined only where ``n >= k``; the reported value is the
    mean over the cases that qualify. A run stopped part-way (cells execute in a per-run shuffled
    order, so it leaves an arbitrary subset of its matrix) therefore leaves its shallow cases out
    of the deep points rather than letting "passed its one attempt" stand in for "passed all
    three". The full curve pass^1..pass^K travels with the headline, each point carrying the cases
    it was computed over; pass^1 is the per-case pass rate averaged over cases.

    One run is one arm (one variant under one measurement context), so grouping by run never
    pools two configurations. Repeats across RUNS of one configuration are pooled by
    :func:`pool_pass_hat_k`, which is what a cross-run surface calls.

    Args:
        results: The results to score.
        rubric_threshold: The 1–5 score a rubric dimension must reach to pass.
        k: The depth of the headline ``pass_hat_k``. ``None`` takes each group's deepest
            ``k_iteration`` attempted — the run's planned k on a run that finished its matrix.

    Returns:
        ``{(model, eval_run_id): {"pass_hat_k": float | None, "k": int, "n_cases_at_k": int,
        "pass_hat_k_curve": [{"k", "pass_hat_k", "n_cases"}, ...], "n_test_cases": int,
        "n_cannot_tell_excluded": int}}``.

        ``pass_hat_k`` is the curve's value at ``k`` and ``n_cases_at_k`` the cases it averages:
        ``None`` and 0 when no case was scored that deep — nothing was measured there, which is
        not a pass rate of zero. The curve runs from 1 to the deepest scored case and is empty
        when nothing was scored. ``n_test_cases`` counts the cases with at least one scored
        attempt. A ``(model, run)`` whose every result was excluded still appears, with zero
        cases, rather than vanishing.

        The default ``k`` counts infra-excluded attempts, so it is the deepest iteration
        *attempted*, and it is per ``(model, run)``: pooling the run's models would let one
        model's deeper cells set the depth for a model that never got past its first.

    Raises:
        ValueError: ``k`` is below 1.
    """
    if k is not None and k < 1:
        raise ValueError(f"pass^k needs k >= 1, got {k}")
    attempts: dict[tuple[str, str], dict[Hashable, list[bool]]] = {}
    cannot_tell: dict[tuple[str, str], int] = {}
    k_observed: dict[tuple[str, str], int] = {}
    for r in results:
        group = (r.model, r.eval_run_id)
        # Registered before the exclusion check, so a group with only infra failures still
        # reports — as nothing measured, not as absent.
        cases = attempts.setdefault(group, {})
        k_observed[group] = max(k_observed.get(group, 0), r.k_iteration)
        passed, was_cannot_tell = _trial_pass(r, rubric_threshold=rubric_threshold)
        if was_cannot_tell:
            cannot_tell[group] = cannot_tell.get(group, 0) + 1
        if passed is None:
            continue
        cases.setdefault(r.test_case_id, []).append(passed)

    return {
        group: _pass_hat_k_entry(
            cases,
            k=k if k is not None else k_observed.get(group, 1),
            n_cannot_tell=cannot_tell.get(group, 0),
        )
        for group, cases in attempts.items()
    }


def pass_hat_k_cell(run: EvalRun) -> tuple[str, ...]:
    """The configuration a run's attempts are repeats of, for pooling pass^k across runs.

    A cell is a variant under one measurement context (:mod:`threetears.evals.contracts.identity`).
    The variant is stamped per result; the context is the run's ``context_key``, the digest of
    every condition a comparison must hold fixed (case basis, judges and their configs, simulator,
    cassette, seeded world, subject state, tool permissions) — and NOT ``k_runs``, which is how
    many repeats were taken under a condition rather than a condition. Two runs sharing it, at one
    identity version and one apparatus provenance, measured one configuration, so their attempts
    at a case are more attempts at that case.

    A run that carries no ``context_key`` (its host assembled it without the launch) is its own
    cell: nothing recorded says what it held fixed, and pooling it would be a merge nothing
    downstream can undo. Refusing costs only depth.

    Args:
        run: The run whose cell is wanted.

    Returns:
        A hashable cell id; equal for two runs exactly when their attempts may pool.
    """
    if run.context_key is None:
        return ("run", run.id)
    return ("context", run.context_key, str(run.identity_version), run.apparatus_provenance)


def pool_pass_hat_k_attempts(
    results: Sequence[EvalResult],
    *,
    cell_of_run: Mapping[str, Hashable],
    rubric_threshold: int = 3,
) -> tuple[dict[tuple[Hashable, str, int, str, str], list[bool]], int]:
    """Each case's scored attempts in one pool, keyed as :func:`pool_pass_hat_k` keys them.

    The grouping behind :func:`pool_pass_hat_k`, public so a surface that needs the cases themselves — an
    interval over them, or a test pairing two contestants on the cases both ran — reads them through the
    one keying that produced the headline rather than re-deriving it.

    Args:
        results: The pool.
        cell_of_run: Run id → its cell, from :func:`pass_hat_k_cell`; a run absent from it is its own cell.
        rubric_threshold: The 1–5 score a rubric dimension must reach to pass.

    Returns:
        ``(attempts, n_cannot_tell)``: each ``(cell, variant_key, identity_version, model, test_case_id)``
        unit's scored attempts in result order, and the attempts left out because a judge could not tell.
        A unit whose every attempt was left out is absent.
    """
    cases: dict[tuple[Hashable, str, int, str, str], list[bool]] = {}
    n_cannot_tell = 0
    for r in results:
        passed, was_cannot_tell = _trial_pass(r, rubric_threshold=rubric_threshold)
        n_cannot_tell += was_cannot_tell
        if passed is None:
            continue
        cell = cell_of_run.get(r.eval_run_id, ("run", r.eval_run_id))
        cases.setdefault((cell, r.variant_key, r.identity_version, r.model, r.test_case_id), []).append(passed)
    return cases, n_cannot_tell


def case_pass_hat_k(attempts: Sequence[bool], k: int) -> float | None:
    """One case's unbiased pass^k estimate from its scored attempts, ``None`` below ``k`` of them.

    ``C(c, k) / C(n, k)`` (see :func:`compute_pass_hat_k`): the per-case value every pass^k headline
    averages, public so an interval or a paired test over cases reads the same number the headline does.

    Args:
        attempts: The case's scored attempts.
        k: The depth, at least 1.

    Returns:
        The estimate, or ``None`` when the case has fewer than ``k`` scored attempts — it cannot stand in
        for a depth it was not measured at.

    Raises:
        ValueError: ``k`` is below 1.
    """
    if k < 1:
        raise ValueError(f"pass^k needs k >= 1, got {k}")
    if len(attempts) < k:
        return None
    return _case_pass_hat_k(len(attempts), sum(attempts), k)


def pool_pass_hat_k(
    results: Sequence[EvalResult],
    *,
    cell_of_run: Mapping[str, Hashable],
    k: int,
    rubric_threshold: int = 3,
) -> dict[str, Any]:
    """Estimate pass^k over one pool of results, pooling a case's attempts across runs of one cell.

    The cross-run sibling of :func:`compute_pass_hat_k`, with the same estimator and the same row.
    A *case* here is a test case within one cell: its attempts from every run of that cell pool
    into one ``(n, c)``, so a repeat run of a configuration adds depth to its cases instead of
    adding a second copy of each. Attempts are keyed by ``(cell, variant_key, identity_version,
    model, test_case_id)``, so a caller handing in a mixed pool still never pools two variants,
    and the same test case measured under two cells stays two cases, averaged side by side.

    Args:
        results: The pool — typically one contestant's results across runs.
        cell_of_run: Run id → its cell, from :func:`pass_hat_k_cell`. A run absent from the map is
            its own cell, which is the direction that cannot produce a wrong merge.
        k: The depth of the headline ``pass_hat_k``. Required: a pool spans runs, whose
            ``k_iteration`` counters restart, so no depth can be read off the attempts.
        rubric_threshold: The 1–5 score a rubric dimension must reach to pass.

    Returns:
        One entry in :func:`compute_pass_hat_k`'s row shape, ``n_test_cases`` counting
        case-within-cell units.

    Raises:
        ValueError: ``k`` is below 1.
    """
    if k < 1:
        raise ValueError(f"pass^k needs k >= 1, got {k}")
    cases, n_cannot_tell = pool_pass_hat_k_attempts(results, cell_of_run=cell_of_run, rubric_threshold=rubric_threshold)
    return _pass_hat_k_entry(cases, k=k, n_cannot_tell=n_cannot_tell)


# =============================================================================
# Latency aggregates — computed at query time, not stored
# =============================================================================


def percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile of an already-sorted, non-empty list.

    **Nearest-rank, not interpolated**: rank = ceil(pct/100 * n), 1-indexed — an observed value. No engine
    surface reads it: at ``pct=50`` it is the lower middle value at an even count, not the median, which the
    run summary reads with :func:`median_unbiased_quantile` at 0.5. **It is not the engine's tail estimator**: at ``pct=95`` it is the
    sample maximum for every ``n <= 19``, which falls below the true 95th percentile most of the time at
    the sizes a run has, so a tail is read with :func:`median_unbiased_quantile` instead.

    **There is a second percentile in this module and it differs on TWO axes.**
    :func:`median_unbiased_quantile` interpolates (Hyndman–Fan type 8) and takes its quantile on the
    **[0, 1] scale** where this one takes **0-100**.

    **The scale difference is the dangerous one and is why this raises.** Handed ``0.95`` by
    someone carrying the bundle's habit across, nearest-rank would compute rank 1 and return the
    *minimum* — a plausible number, wrong by the width of the distribution, with nothing to
    notice it. Anything in ``(0, 1]`` is refused by name rather than served, because on a run's
    handful of cells nobody asks for a first percentile or below.

    **The closed end at 1.0 is the point of the band, not an off-by-one.** The guard first
    refused ``0 < pct < 1`` and left ``1.0`` through, which is the one value where the two
    conventions invert completely: on the neighbour's ``[0, 1]`` scale it means the MAXIMUM,
    here it resolves to rank 1 and returns the MINIMUM. It is also the value a caller reaches
    for most readily after ``0.95``. The cost is real and small — ``p1`` is not obtainable
    through this function — and it is the trade the guard exists to make. Pass ``100`` for the
    maximum; if a genuine ``p1`` is ever wanted, widening the band back is the diff-visible act,
    and the reason belongs in that commit.

    Public rather than private because it has consumers outside this module — a host's
    own rollups — and an underscore on a name another
    module imports says the opposite of what is true.

    Args:
        sorted_values: Ascending-sorted values, at least one element.
        pct: The percentile wanted, on the **0-100** scale.

    Returns:
        The rank-nearest observed value.

    Raises:
        ValueError: If ``pct`` is outside ``[0, 100]``, or in ``(0, 1]`` — see above.
    """
    if not 0 <= pct <= 100:
        raise ValueError(f"pct must be in [0, 100], got {pct}")
    if 0 < pct <= 1:
        raise ValueError(
            f"pct={pct} is on the [0, 1] scale; this function takes 0-100 "
            "(median_unbiased_quantile, which the analysis bundle reads its percentiles with, is the [0, 1] one). "
            "For the maximum, pass 100 — on this scale 1.0 is the FIRST percentile and "
            "resolves to the minimum."
        )
    n = len(sorted_values)
    rank = math.ceil(pct / 100.0 * n)
    idx = min(max(rank, 1), n) - 1
    return sorted_values[idx]


#: Hyndman and Fan's type 8 plotting position, ``h = (n + 1/3) q + 1/3``: the one rule both the run summary
#: and the analysis bundle read a percentile with.
_TYPE_8_OFFSET = 1.0 / 3.0


def median_unbiased_quantile_min_n(q: float) -> int:
    """The fewest observations at which :func:`median_unbiased_quantile` gives the ``q`` quantile at all.

    Type 8's position ``h = (n + 1/3) q + 1/3`` must fall inside the sample, ``1 <= h <= n``: past either
    end the estimate would be the extreme observation, whose median sits short of the quantile it is
    labelled. For the 95th percentile (and the 5th) that is 13 observations: the largest of 13 falls below
    the true 95th percentile ``0.95^13 = 0.51`` of the time, the largest of 5 ``0.95^5 = 0.77`` of it.

    Args:
        q: The quantile, in ``(0, 1)``.

    Returns:
        The smallest ``n`` with ``1 <= h <= n``.

    Raises:
        ValueError: ``q`` is outside ``(0, 1)``.
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"a quantile needs 0 < q < 1, got {q}")
    n = 1
    while not 1.0 <= (n + _TYPE_8_OFFSET) * q + _TYPE_8_OFFSET <= n:
        n += 1
    return n


def median_unbiased_quantile(sorted_values: Sequence[float], q: float) -> float | None:
    """The ``q`` quantile of an ascending sample, read so it is as likely above the truth as below it.

    Hyndman and Fan's (1996) type 8, the estimator they recommend: interpolation between the order
    statistics at position ``h = (n + 1/3) q + 1/3``, which is approximately median-unbiased whatever the
    distribution — the figure falls below the population quantile about half the time. The two rules this
    replaced both fall short of a tail: nearest-rank is the sample maximum for every ``n <= 19`` (below the
    true 95th percentile 0.77 of the time at five observations), and linear interpolation (numpy's
    default, type 7) still sat below it 0.68 of the time at thirty.

    **Below :func:`median_unbiased_quantile_min_n` observations it answers ``None``**, because no order
    statistic, nor any interpolation between two, is median-unbiased there: the position falls outside
    the sample, and the only figure left is the extreme observation, which is a different statistic
    with a different name. A caller reports the maximum (or minimum) under that name instead.

    Args:
        sorted_values: The observations, ascending.
        q: The quantile, in ``(0, 1)`` — on the [0, 1] scale, not :func:`percentile`'s 0–100.

    Returns:
        The estimate, or ``None`` when the sample is too small to give one.

    Raises:
        ValueError: ``q`` is outside ``(0, 1)``.
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"a quantile needs 0 < q < 1, got {q}")
    n = len(sorted_values)
    position = (n + _TYPE_8_OFFSET) * q + _TYPE_8_OFFSET
    if not 1.0 <= position <= n:
        return None
    lower = math.floor(position)
    fraction = position - lower
    if fraction == 0.0:
        return float(sorted_values[lower - 1])
    return float(sorted_values[lower - 1] + fraction * (sorted_values[lower] - sorted_values[lower - 1]))


def compute_latency_summary(
    results: list[EvalResult],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Aggregate per-result :class:`LatencyMetrics` by ``(model, eval_run_id)``.

    Mirrors :func:`compute_pass_hat_k`: latency aggregates are derived at query
    time, never stored on :class:`EvalRun`, and **infra-excluded results are
    dropped** by the same :func:`~threetears.evals.contracts.result_condition.classify_result`
    predicate. Feeds the price/performance frontier and the comparison
    views alongside cost — comparisons between configurations, which
    is why an apparatus fault must not contribute a time. A cell cancelled on its
    deadline times its deadline; a cell cut short by a cassette miss or a broken
    tool times a truncated conversation; neither is how fast this configuration
    runs. A **candidate** failure keeps its timings — the candidate failing is an
    outcome of the configuration, not of the harness.

    Results with ``latency is None`` are skipped — the marker means the cell
    timed nothing at all, so it has no measured time to contribute. A
    ``(model, run_id)`` group with no measured results is omitted entirely
    rather than reported as zero, and a group whose every cell was infra-excluded
    is omitted the same way.

    **Each component aggregates over its own measured subset**, because the
    three are independently nullable: a cell that harvested an ``llm.call``
    span but no ``agent.invoke`` turn-root contributes to ``mean_llm_ms`` and
    to nothing else. Folding its unknown total in as a zero would drag the
    mean toward an instant that nobody observed. A component whose subset is
    empty is **absent from the row** rather than present as a number.

    A group therefore survives with only ``n_results`` when no result
    contributed to any component this function aggregates. That is no longer
    the same as "the carrier measured nothing": ``async_wait_ms`` and
    ``judge_ms`` are timed off a monotonic clock rather than harvested, so a
    spanless cell that waited in the drain loop, or ran a judge, still carries a
    real measurement — neither of which this summary aggregates. A component
    added to :class:`LatencyMetrics` is NOT aggregated here by arriving; it is
    aggregated when a branch below reads it.
    Read a bare ``n_results`` as "nothing HERE was measured", which
    is still a different answer from the omission above, where no result
    reached the harvest at all.

    **The median is the standard one**, read by the same rule as the tail —
    :func:`median_unbiased_quantile` at 0.5, whose position ``(n + 1) / 2`` is the middle value, or the mean of
    the two middle values at an even count — so it agrees with the bundle's ``p50``. A figure computed before
    this rule read nearest-rank, which took the lower of the two middle values at an even count.

    **The tail is the 95th percentile only where one can be estimated.** ``p95_total_ms`` is
    :func:`median_unbiased_quantile` (Hyndman–Fan type 8), present from 13 measured totals, and absent below:
    there the only figure a sample offers for its tail is its slowest observation, which is not a 95th
    percentile — at five totals it falls below the true one 77% of the time — so it is reported as
    ``max_total_ms``, under its own name, at every size. A row stored before this rule read its ``p95_total_ms``
    nearest-rank, which is that same maximum for every ``n <= 19``.

    Returns:
        ``{(model, eval_run_id): {"mean_total_ms", "median_total_ms",
        "p95_total_ms", "max_total_ms", "mean_llm_ms", "mean_tool_ms", "n_total_ms",
        "n_llm_ms", "n_tool_ms", "n_results"}}``, with a component and its
        count both absent when nothing measured it, and ``p95_total_ms`` absent below 13 totals. ``n_results`` counts the
        results carrying a :class:`LatencyMetrics` at all; each ``n_<field>``
        is the denominator its own mean was computed over, and they can
        legitimately disagree.
        ``mean_llm_ms`` is the model-attributable axis (tool time is
        model-independent), so it is the one to compare across models;
        ``mean_tool_ms`` completes the total = llm + tool decomposition
        (total also absorbs framework overhead, so the three need not sum).
    """
    grouped: dict[tuple[str, str], list[LatencyMetrics]] = {}
    for r in results:
        # Dropped before the null check, and on the same predicate compute_pass_hat_k
        # uses: a harness failure's timings are a measurement of the harness.
        if classify_result(r) is ResultOutcome.INFRA_EXCLUDE:
            continue
        if r.latency is None:
            continue
        grouped.setdefault((r.model, r.eval_run_id), []).append(r.latency)

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, metrics in grouped.items():
        totals = sorted(m.total_ms for m in metrics if m.total_ms is not None)
        llms = [m.llm_ms for m in metrics if m.llm_ms is not None]
        tools = [m.tool_ms for m in metrics if m.tool_ms is not None]

        row: dict[str, Any] = {"n_results": len(metrics)}
        # Each mean carries its OWN denominator. Without it a mean over two of
        # thirty observations is printed identically to one over all thirty,
        # which is the same defect as an unmeasured cell rendering as a
        # measured one — moved up a level from the value to its evidence.
        if totals:
            row["mean_total_ms"] = sum(totals) / len(totals)
            # The standard median — the middle value, or the mean of the two middle values at an even count —
            # which is what type 8 gives at 0.5 at every n. Nearest-rank took the lower middle value.
            row["median_total_ms"] = median_unbiased_quantile(totals, 0.5)
            # The tail, median-unbiased, and ABSENT below the 13 observations at which any estimate of a
            # 95th percentile can be: under that the only candidate is the slowest observation, which falls
            # below the true p95 most of the time and is reported under its own name, never this one.
            tail = median_unbiased_quantile(totals, 0.95)
            if tail is not None:
                row["p95_total_ms"] = tail
            row["max_total_ms"] = totals[-1]
            row["n_total_ms"] = len(totals)
        if llms:
            row["mean_llm_ms"] = sum(llms) / len(llms)
            row["n_llm_ms"] = len(llms)
        if tools:
            row["mean_tool_ms"] = sum(tools) / len(tools)
            row["n_tool_ms"] = len(tools)
        out[key] = row
    return out


# =============================================================================
# Cost aggregates — computed at query time, not stored
# =============================================================================


def compute_cost_summary(
    results: list[EvalResult],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Aggregate per-result ``cost_usd`` by ``(model, eval_run_id)``.

    Mirrors :func:`compute_pass_hat_k` and :func:`compute_latency_summary`: cost
    aggregates are derived at query time, never stored on :class:`EvalRun`.
    Feeds the price/performance frontier and the comparison views
    alongside latency.

    Every ``(model, run_id)`` group with at least one result is reported, and
    ``n_results`` counts them all — a group never disappears for want of a
    measurement.

    **This aggregate deliberately does NOT drop infra-excluded results**, and is
    the one place the family diverges: :func:`~threetears.evals.contracts.scoring.compute_pass_hat_k`,
    :func:`compute_latency_summary`, :func:`compute_dimension_summary` and
    :func:`compute_composite_summary` all drop them, because each answers "how
    good / how fast is this configuration" and a harness failure is no evidence
    either way. Cost answers a different question — what the program spent — and
    the tokens burned by a cell that later died in the apparatus were still
    billed. Dropping them would under-report the run's spend and, because
    ``run_summary`` takes its group set from this function's keys, would delete
    an all-excluded model's headline row from the report that exists to disclose
    it. ``n_prod_cost_usd`` carries how well evidenced the prod figure is; the
    exclusion state of a cell is recorded on the cell.

    Two cost axes are aggregated: the program spend (blended ``cost_usd``, incl.
    judge + simulator) and the ``production_replicating_cost`` (candidate +
    inner_agent + external only — what the subject would cost in prod), so a
    reporting view can lead with the prod-replicating number (what a config
    costs to *run*) while keeping program spend for the cost-management lens.

    **Both axes leave a result they cannot price out of their dollars, and count it.**

    ``cost_usd`` is ``None`` for a result whose spend went unpriced — a model call its client
    reported no price for. Such a result is **absent from the program total and mean rather than
    counted as a zero**, ``n_cost_usd`` is the denominator the mean was computed over, and the
    total and mean are absent from the row when no result in the group was priced. A cell the
    deadline stopped holds what it had captured, never the call in flight when it struck
    (``termination`` says which results stop short), so the total is a floor for those.

    ``production_replicating_cost`` returns ``None`` when no production role
    observed a cost (or a delivery was substituted), which is a real unmeasured
    marker — so the prod axis honours it the way :func:`compute_latency_summary`
    honours a null component. Such a result is **absent from the prod mean rather
    than counted as a zero**, ``n_prod_cost_usd`` carries the denominator that mean
    was actually computed over, and both the total and the mean are **absent from
    the row** when nothing in the group measured one. Counting those results as
    zero would put whichever config was least measured at the top of a
    lower-is-better axis.

    ``n_cost_usd`` and ``n_prod_cost_usd`` can therefore be smaller than ``n_results``, and the
    gap is how many results contributed nothing. Read them, not ``n_results``, when judging how
    well evidenced a cost figure is.

    Returns:
        ``{(model, eval_run_id): {"total_cost_usd", "mean_cost_usd", "n_cost_usd",
        "total_prod_cost_usd", "mean_prod_cost_usd", "n_prod_cost_usd",
        "n_results"}}``. The program pair is absent when no result in the group was priced,
        and the three prod keys absent together when none measured a production-replicating
        cost. ``n_cost_usd`` and ``n_results`` are always present.
    """
    grouped: dict[tuple[str, str], list[tuple[float | None, float | None]]] = {}
    for r in results:
        # The canonical resolution, so this aggregate answers "does this result have a
        # prod cost" the same way the single-result surfaces do.
        resolved_r = resolve_result_usage(r)
        prod = production_replicating_cost(resolved_r.usage, substituted_deliveries=resolved_r.substituted_deliveries)
        grouped.setdefault((r.model, r.eval_run_id), []).append((r.cost_usd, prod))

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in grouped.items():
        priced = [program for program, _ in rows if program is not None]
        observed_prod = [prod for _, prod in rows if prod is not None]

        row: dict[str, Any] = {"n_cost_usd": len(priced), "n_results": len(rows)}
        if priced:
            total = sum(priced)
            row["total_cost_usd"] = total
            row["mean_cost_usd"] = total / len(priced)
        if observed_prod:
            prod_total = sum(observed_prod)
            row["total_prod_cost_usd"] = prod_total
            row["mean_prod_cost_usd"] = prod_total / len(observed_prod)
            row["n_prod_cost_usd"] = len(observed_prod)
        out[key] = row
    return out


# =============================================================================
# Per-rubric-dimension aggregates — computed at query time, not stored
# =============================================================================


def compute_dimension_summary(
    results: list[EvalResult],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Aggregate per-result ``rubric_scores`` by ``(model, eval_run_id, dim)``.

    Derived at query time and never stored, as :func:`~threetears.evals.contracts.scoring.compute_pass_hat_k`,
    :func:`compute_latency_summary` and :func:`compute_cost_summary` are. Feeds
    the per-dimension breakdown view — where pass^k answers *whether*
    a model cleared the bar, this answers *which dimension* it cleared or
    missed (e.g. cost-frontier DeepSeek scoring grounding=4 but coverage=2).

    **Infra-excluded results are dropped**, through the same
    :func:`~threetears.evals.contracts.result_condition.classify_result` predicate
    :func:`~threetears.evals.contracts.scoring.compute_pass_hat_k` and :func:`_group_case_composites` use — one notion of
    exclusion, so the three cannot drift apart. Such a result usually *does* carry
    rubric scores: the judge reads whatever transcript survived and the runner
    stores that reading, deliberately, for forensics. Averaging it is the defect.
    A judge reading a broken transcript scores it low, so pooling those scores
    drags a model's dimension mean down for an apparatus fault — on the one table
    an operator consults to decide which dimension the model missed, which is
    exactly the confusion the exclusion machinery exists to prevent. A
    **candidate** failure is kept at each dim's floor: the
    configuration delivered no turn, so the judge's reading of what it left — silence
    scored as restraint — is not what the end user got.

    :func:`compute_cost_summary` is the family's deliberate exception — it counts
    every result, because program spend is accounting rather than a measurement
    of the candidate: money spent on a cell the harness broke was still spent.

    Only the template rubric dimensions are aggregated: ``EvalResult.rubric_scores``
    holds the judge-scored template dims (the run loop assembles it from
    ``template.rubric`` alone), while the reserved dual-score axes
    (``__transcript__`` / ``__outcome__``) live in the separate
    ``transcript_score`` / ``outcome_score`` fields and are not included here.
    A ``(model, run_id)`` slot with no surviving rubric scores — a goal-only
    template, a factory-failure result, or a group whose every cell was
    infra-excluded — contributes **no dimension rows**. That is pass^k's own
    choice ("a case whose every iteration is excluded therefore vanishes from its
    run's denominator") carried onto a table whose key *is* a dimension: a
    dimension exists here only because some judge scored it, so an all-excluded
    group has no key under which to report itself unmeasured. The group does not
    disappear from the run summary — its headline row is keyed off
    :func:`compute_cost_summary`, which reports every group with results, and the
    run's degraded disclosure names the excluded cells.

    Returns:
        ``{(model, eval_run_id, dim): {"scale", "mean_score", "min_score",
        "max_score", "n"}}``. Scores are the stored integers — 1–5, or 1/0 on a pass/fail
        dimension, whose ``mean_score`` is therefore its pass rate; ``mean_score``
        is a float, ``min_score`` / ``max_score`` are ints. ``n`` counts the
        scores that survived the exclusion above, so it is comparable with
        pass^k's ``n_test_cases`` rather than with the group's raw result count.
    """
    grouped: dict[tuple[str, str, str], list[int]] = {}
    scales: dict[tuple[str, str, str], set[str]] = {}
    for r in results:
        # `counted_rubric_scores` drops an infra failure (its judge score is stored for forensics,
        # not for averaging) and counts a candidate failure at each dim's floor — the one rule the
        # score projection reads too, so the two cannot disagree about a dim.
        counted = counted_rubric_scores(r)
        if counted is None:
            continue
        for s, value in counted:
            grouped.setdefault((r.model, r.eval_run_id, s.dim), []).append(value)
            scales.setdefault((r.model, r.eval_run_id, s.dim), set()).add(s.scale)

    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, scores in grouped.items():
        if len(scales[key]) > 1:
            # A mean of 1–5 scores and 1/0 answers is neither a level nor a pass rate.
            raise ValueError(
                f"dimension {key[2]!r} was judged on more than one scale in run {key[1]}: {sorted(scales[key])}"
            )
        n = len(scores)
        out[key] = {
            "scale": next(iter(scales[key])),
            "mean_score": sum(scores) / n,
            "min_score": min(scores),
            "max_score": max(scores),
            "n": n,
        }
    return out


# =============================================================================
# Mean-composite quality score — computed at query time, never stored
#
# The continuous sibling of pass^k: where pass^k answers "how reliably did the
# model clear the bar" (binary per case, gated by ``rubric_threshold``), the
# composite answers "how good was it on average" (continuous, threshold-free).
# A quality regression usually shows in the composite before cases start
# failing pass^k — which is why the compare view shows both, and an
# analytics tier reuses these aggregators.
# =============================================================================


def _group_case_composites(
    results: list[EvalResult],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Group results into per-``(model, run)`` per-test-case composite scores.

    Shared core of :func:`compute_composite_summary` and
    :func:`compute_per_case_composites`. For each ``(model, eval_run_id)`` group
    it records whether the group has any rubric dims (``has_rubric``) and, when
    it does, the per-``test_case_id`` composite — the mean over that case's
    k-iterations.

    Error handling mirrors pass^k: **infra-excluded**
    iterations (``infra_error`` / ``judge_error``, which is where a cell timeout
    charged to the rig lands — ``runner._DEADLINE_CHARGE``) are DROPPED —
    never added to a case's composite list — so an infra failure neither floors
    the mean nor counts as a case; a case whose every iteration is excluded
    vanishes from ``per_case`` entirely. **Candidate-failure** iterations score
    ``0.0`` (depress, don't vanish); genuine no-quality-signal ok iterations are
    coerced to ``0.0`` within a rubric group so they can't hide. Every group is
    registered even if all its results are excluded (empty ``per_case`` then).
    Goal-only groups (no rubric dims anywhere) get an empty ``per_case`` — the
    composite is undefined there, not zero.

    Returns:
        ``{(model, run_id): {"has_rubric": bool, "per_case": {tc_id: float}}}``.
    """
    raw: dict[tuple[str, str], dict[str, list[float | None]]] = {}
    has_rubric: dict[tuple[str, str], bool] = {}
    for r in results:
        key = (r.model, r.eval_run_id)
        has_rubric.setdefault(key, False)  # register the group even if all-excluded
        if trial_exclusion(r) is not None:
            continue  # a fault, or the judge could not tell on a dim the composite needs — unmeasured, not zero
        has_rubric[key] = has_rubric[key] or bool(capability_scores(r))
        raw.setdefault(key, {}).setdefault(r.test_case_id, []).append(result_composite(r))

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rubric in has_rubric.items():
        cases = raw.get(key, {})
        per_case: dict[str, float] = {}
        if rubric:
            for tc_id, comps in cases.items():
                vals = [c if c is not None else 0.0 for c in comps]
                per_case[tc_id] = sum(vals) / len(vals)
        out[key] = {"has_rubric": rubric, "per_case": per_case}
    return out


def compute_composite_summary(
    results: list[EvalResult],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Aggregate the mean-composite quality score by ``(model, eval_run_id)``.

    The continuous, threshold-free quality aggregate — mean over test cases of
    each case's :func:`result_composite` (k-iterations averaged first, so each
    case weighs equally, matching pass^k's per-case denominator). Query-time,
    never stored; reused by an analytics tier.

    Returns:
        ``{(model, run_id): {"mean_composite": float | None, "n_cases": int}}``.
        ``mean_composite`` is ``None`` for goal-only groups (no rubric dims);
        ``n_cases`` is the number of cases contributing to the composite (0 when
        undefined).
    """
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, info in _group_case_composites(results).items():
        per_case: dict[str, float] = info["per_case"]
        if info["has_rubric"] and per_case:
            out[key] = {
                "mean_composite": sum(per_case.values()) / len(per_case),
                "n_cases": len(per_case),
            }
        else:
            out[key] = {"mean_composite": None, "n_cases": 0}
    return out


def compute_per_case_composites(
    results: list[EvalResult],
) -> dict[tuple[str, str, str], float]:
    """Per-``(model, run, test_case)`` composite scores — the pairing atom.

    Flattens :func:`_group_case_composites` to one value per case so the run
    comparison can pair scores by ``test_case_id`` across two runs
    for a paired significance test. Goal-only groups contribute nothing.

    Returns:
        ``{(model, run_id, test_case_id): composite}``.
    """
    out: dict[tuple[str, str, str], float] = {}
    for (model, run_id), info in _group_case_composites(results).items():
        for tc_id, val in info["per_case"].items():
            out[(model, run_id, tc_id)] = val
    return out


__all__ = [
    "CellSummary",
    "capability_scores",
    "compute_composite_summary",
    "compute_cost_summary",
    "compute_dimension_summary",
    "compute_latency_summary",
    "compute_pass_hat_k",
    "compute_per_case_composites",
    "case_pass_hat_k",
    "median_unbiased_quantile",
    "median_unbiased_quantile_min_n",
    "pass_hat_k_at",
    "pass_hat_k_cell",
    "PassHatPoint",
    "percentile",
    "pool_pass_hat_k",
    "pool_pass_hat_k_attempts",
    "reconstruct_completeness",
    "result_composite",
    "summarize_completeness",
]
