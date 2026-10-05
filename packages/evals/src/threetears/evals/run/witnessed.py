"""Recording a cell a host observed rather than ran: the engine's one assembly, called from outside its runner.

A host that witnesses a session — real people, no rig — and wants it read beside the cells its runs
produce records it through :func:`record_witnessed_cell`. That builds the cell's
:class:`~threetears.evals.contracts.models.EvalResult` and
:class:`~threetears.evals.contracts.models.EvalTrace` through
:func:`~threetears.evals.run.runner.assemble_completed_cell`, the same function the runner calls for
every cell that reaches its end, so the two cannot drift apart.

Its own module rather than a function in the runner because it is not part of the runner's dispatch:
the runner reads ``template.candidate_kind`` exactly once and dispatches on it, while this reads the
kind a witnessed run was stamped with and dispatches on nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from threetears.evals.contracts.candidate_kind import CandidateOutput
from threetears.evals.contracts.host.eval_host import EvalHost
from threetears.evals.contracts.host.traces import CellTrace
from threetears.evals.contracts.identity import resolve_variant_identity
from threetears.evals.contracts.models import (
    ConversationStopCause,
    EvalResult,
    EvalRun,
    EvalTestCase,
    EvalTrace,
    JudgedArtifact,
)
from threetears.evals.contracts.spend import ExternalRateTable
from threetears.evals.contracts.world_events import WorldEvent
from threetears.evals.run.runner import assemble_completed_cell, hold_to_declaration

#: The stop causes only the engine's simulator produces, which a witnessed session — no simulator in it —
#: cannot carry (:func:`record_witnessed_cell`).
_SIMULATOR_STOP_CAUSES = frozenset({ConversationStopCause.USER_DONE, ConversationStopCause.SIMULATOR_ERROR})


def record_witnessed_cell(
    host: EvalHost,
    run: EvalRun,
    test_case: EvalTestCase,
    output: CandidateOutput,
    *,
    k_iteration: int,
    result_id: str,
    scored_at: str,
    judged_artifact: JudgedArtifact,
    external_rates: ExternalRateTable | None = None,
    spans: CellTrace | None = None,
    world_events: Sequence[WorldEvent] | None = None,
    end_state: Mapping[str, Any] | None = None,
) -> tuple[EvalResult, EvalTrace]:
    """Record one cell a host observed — a session it witnessed, not one the engine ran — as a result and its trace.

    The host hands over what the candidate produced, in the same
    :class:`~threetears.evals.contracts.candidate_kind.CandidateOutput` a kind's ``invoke`` returns, and
    this builds the two documents through **the same assembly every completed cell of a run takes**:
    the async-delivery spend folded into the usage rows, the candidate/infra error taxonomy, the blended
    cost, the covariates, the latency record and the trace's id. So a witnessed cell and a run's cell
    over the same output are the same record, field for field, except for the ids.

    **Idempotent by construction.** The result's id and timestamp are the caller's, and the trace's id
    is derived from the result's, so recording the same observation twice under the same id produces
    the same pair — and saving it twice overwrites rather than duplicates. It persists nothing: save
    the pair with ``host.storage.save_eval_result(result, trace)``, as the runner does each cell's.

    **Witnessed runs only.** A ``commissioned`` run's cells are what its rig produced, and the runner
    (:func:`~threetears.evals.run.runner.execute_run`) is the only thing that drives that rig; a host recording a cell onto one would
    file an observation it made outside the rig under a run that claims the rig made it. So a run whose
    ``apparatus_provenance`` is not ``witnessed`` is refused. The runner does not come through here —
    it calls the assembly directly with the ids it mints.

    **Unjudged.** Nothing here calls a judge, so the cell carries no scores, and a run naming a judge
    model is refused for the reason :func:`~threetears.evals.run.runner.execute_run` refuses a judged run with no judge service:
    every read surface takes ``judge_model`` to mean the cell was scored by it.

    **What the host writes around it.** A witnessed session has no template, so its run carries
    ``template_id=None`` and so does its case: an :class:`~threetears.evals.contracts.models.EvalTestCase`
    with ``template_id=None``, its stimulus in ``variation_params`` and ``host_payload``, saved with
    ``host.storage.save_test_case`` — re-check and re-judge read it back by id, and no launch will run it,
    because a launch refuses a case whose template is not its own. A conversation its real participants
    ended carries ``stop_cause=participants_ended``; the simulator's two causes (``user_done``,
    ``simulator_error``) are refused here. The run's terminal state is the host's to write too, as the
    job manager writes a commissioned run's: ``status="completed"`` and
    ``completeness=summarize_completeness(run, cells)``, ``cells`` being one
    ``CellSummary.from_result(result, persisted=...)`` per cell the capture recorded, the ``persisted``
    flag what its save returned.

    Args:
        host: The host the run belongs to — its profile resolves the run's contestant identity.
        run: The witnessed run the cell belongs to.
        test_case: The case the observation answers; one of ``run.test_case_ids``.
        output: What the candidate produced.
        k_iteration: The repeat, ``1..run.k_runs``.
        result_id: The result's id — the caller's, so a re-capture of the same observation is the same record.
        scored_at: When the observation was recorded, as an ISO-8601 timestamp.
        judged_artifact: What the kind that produced ``output`` declares a judge reads. Held to the
            output as the runner holds it, and kept on the trace beside any evidence.
        external_rates: The rate table that prices the output's external calls, or ``None`` for none.
        spans: The host's trace record of the session's own work, or ``None`` when it collected none —
            the trace's spans and the result's latency buckets are read off it.
        world_events: What moved the session's world, or ``None`` when it had none.
        end_state: The world the session left, or ``None`` when it was not read.

    Returns:
        The result and its trace, unsaved.

    Raises:
        ValueError: The run is not ``witnessed``; it names a judge model; ``test_case`` is not one of
            its cases, or sits in another scope or under another template; ``k_iteration`` is outside
            ``1..run.k_runs``; or the output's stop cause is one only the engine's simulator produces.
        CandidateKindDefect: ``output`` contradicts ``judged_artifact``.
    """
    if run.apparatus_provenance != "witnessed":
        raise ValueError(
            f"run {run.id} is {run.apparatus_provenance!r}: only a witnessed run's cells are recorded by its host. "
            "A commissioned run's cells are what its rig produced, and execute_run is what drives that rig"
        )
    if run.judge_model is not None:
        raise ValueError(
            f"run {run.id} names judge model {run.judge_model!r}, and a witnessed cell is recorded unjudged; a "
            "run naming a judge claims every cell was scored by it. Record the witnessed run with no judge model"
        )
    if test_case.id not in run.test_case_ids:
        raise ValueError(
            f"test case {test_case.id!r} is not one of run {run.id}'s cases; a run's cases are its denominator, "
            "so a cell outside them is an observation no read of the run would count"
        )
    if (test_case.scope_id, test_case.template_id) != (run.scope_id, run.template_id):
        raise ValueError(
            f"test case {test_case.id!r} is in scope {test_case.scope_id!r} under template {test_case.template_id!r}, "
            f"and run {run.id} is in scope {run.scope_id!r} under template {run.template_id!r}; a witnessed cell's "
            "case belongs where its run does — template None on both for a session no template set"
        )
    if not 1 <= k_iteration <= run.k_runs:
        raise ValueError(f"k_iteration {k_iteration} is outside run {run.id}'s repeats (1..{run.k_runs})")
    if output.stop_cause in _SIMULATOR_STOP_CAUSES:
        raise ValueError(
            f"a witnessed cell's conversation cannot have stopped on {output.stop_cause.value!r}: that cause names the "
            "engine's simulator, and a witnessed session has none. A session its people ended is "
            "'participants_ended'"
        )
    # Read once: the kind the run stamped at its creation, and so the kind the cell is recorded under.
    candidate_kind = run.candidate_kind
    hold_to_declaration(candidate_kind, judged_artifact, output)
    result, trace = assemble_completed_cell(
        scope_id=run.scope_id,
        eval_run_id=run.id,
        test_case_id=test_case.id,
        model=run.candidate_model,
        candidate_kind=candidate_kind,
        k_iteration=k_iteration,
        subject_id=run.subject_snapshot.subject_id,
        # Read off the run exactly as execute_run reads it, so the two key a cell the same way.
        variant=resolve_variant_identity(run=run, profile=host.profile),
        judge_model=None,
        output=output,
        judged_artifact=judged_artifact,
        rate_table=external_rates,
        result_id=result_id,
        scored_at=scored_at,
        judged=None,
        judge_ms=None,
        spans=spans,
        # Nothing here samples the host's job count: a witnessed session ran under no eval job.
        concurrent_eval_jobs=None,
        # No run ledger metered a witnessed session; the kind's own count, if any, is still folded.
        metered_cell=None,
        world_events=list(world_events) if world_events is not None else None,
        end_state=dict(end_state) if end_state is not None else None,
    )
    return result, trace


__all__ = ["record_witnessed_cell"]
