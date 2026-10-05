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

**A witnessed cell can be judged, as it is recorded.** A witnessed session has no template that SET it,
but a judge reads a template's intent and rubric, so a run that wants its cells judged names the
template it is judged against and carries the judge apparatus a launch would stamp — through
:func:`stamp_witnessed_judge`, before its identity is stamped, since the judge is part of the
conditions the run was measured under. :func:`record_witnessed_cell` then scores each cell through the
runner's own judge phase, from the apparatus the run recorded, so a judged witnessed cell is the record
a run's judged cell is and a later re-judge (:func:`~threetears.evals.run.lifecycle.rejudge_result`)
reads it exactly as it reads one. Why at recording time and not by a later judging operation: the judge
is part of a run's context identity, stamped once when the run comes into being and never rewritten,
so a judge added to a stored run afterwards would either rewrite that key or leave the cells scored by
a judge the key does not name. The case stays template-less either way — a case filed under the
template would be one every launch of it runs.
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
    EvalTemplate,
    EvalTestCase,
    EvalTrace,
    JudgedArtifact,
)
from threetears.evals.contracts.spend import ExternalRateTable
from threetears.evals.contracts.world_events import WorldEvent
from threetears.evals.run.judge import JUDGE_REQUEST_SETTINGS
from threetears.evals.run.judge_service import JudgeService, judge_clients_for_run
from threetears.evals.run.launch import build_judge_service
from threetears.evals.run.rejudge import (
    judged_template,
    recorded_judge_configs,
    recorded_judge_pins,
    recorded_judged_dims,
)
from threetears.evals.run.runner import (
    DEFAULT_JUDGE_CONCURRENCY,
    assemble_completed_cell,
    hold_to_declaration,
    judge_witnessed_output,
)

#: The stop causes only the engine's simulator produces, which a witnessed session — no simulator in it —
#: cannot carry (:func:`record_witnessed_cell`).
_SIMULATOR_STOP_CAUSES = frozenset({ConversationStopCause.USER_DONE, ConversationStopCause.SIMULATOR_ERROR})


def stamp_witnessed_judge(
    host: EvalHost,
    run: EvalRun,
    template: EvalTemplate,
    *,
    judge_model: str,
    judged_artifact: JudgedArtifact,
    selection: dict[str, str] | None = None,
) -> EvalRun:
    """A witnessed run that names the template its cells are judged against, and the judge apparatus that scores them.

    The fields a launch stamps for a judged run, resolved the way a launch resolves them
    (:func:`~threetears.evals.run.launch.build_judge_service`: one config load per scored dim, the
    run-level pin, each dim's effective judge): ``template_id``, ``judge_model`` and its provenance,
    ``effective_judges`` (recorded), ``judge_config_ids`` and their provenance, ``judge_request_settings``
    and ``rubric_scales``. Call it on the run as built and BEFORE stamping its identity — the judge is
    part of the context it hashes — then record its cells with :func:`record_witnessed_cell`, which
    scores them from exactly these.

    Args:
        host: The host: where the judge configs are read and the judge clients come from.
        run: The witnessed run, unjudged and not yet identity-stamped.
        template: The template whose intent and rubric the judge reads; in the run's scope, and of the
            kind the run recorded.
        judge_model: The run-level judge pin, resolved.
        judged_artifact: What the kind's judge reads, which picks the dims it scores.
        selection: Optional ``{dim_id: config_id}`` naming configs per dim, as a launch's
            ``judge_config_ids`` does; the rest inherit each dim's active config.

    Returns:
        A copy of the run carrying the judge.

    Raises:
        ValueError: The run is not witnessed; it already names a judge or a template; its identity is
            already stamped; the template is in another scope or of another kind; ``judged_artifact``
            declares a kind no judge reads; or the host supplies no completion clients.
        ValidationFailedError: ``selection`` names an unscored dim, a config that does not load, or one
            authored for another dim.
    """
    if run.apparatus_provenance != "witnessed":
        raise ValueError(f"run {run.id} is {run.apparatus_provenance!r}; a launch stamps a commissioned run's judge")
    if run.judge_model is not None or run.template_id is not None:
        raise ValueError(
            f"run {run.id} already names judge {run.judge_model!r} and template {run.template_id!r}; a run's judge "
            "is stamped once, when it comes into being"
        )
    if run.context_key is not None:
        raise ValueError(
            f"run {run.id}'s identity is already stamped, and the judge is part of the context it hashes; stamp the "
            "judge first, then the identity"
        )
    if template.scope_id != run.scope_id:
        raise ValueError(
            f"template {template.id!r} is in scope {template.scope_id!r} and run {run.id} in {run.scope_id!r}; a "
            "run is judged against a template of its own scope"
        )
    if template.candidate_kind != run.candidate_kind:
        raise ValueError(
            f"template {template.id!r} is a {template.candidate_kind!r} template and run {run.id} recorded kind "
            f"{run.candidate_kind!r}; its intent and rubric were written for another kind's output"
        )
    judge = build_judge_service(host, template, judge_model, selection, judged_artifact=judged_artifact)
    return run.model_copy(
        update={
            "template_id": template.id,
            "judge_model": judge.model,
            "model_role_provenance": {**(run.model_role_provenance or {}), "judge": "chosen"},
            "effective_judges": judge.effective_judges,
            "effective_judges_source": "recorded",
            "judge_config_ids": judge.config_ids,
            "judge_config_provenance": judge.config_provenance,
            "judge_request_settings": JUDGE_REQUEST_SETTINGS,
            "rubric_scales": {dim.name: dim.scale for dim in template.rubric},
        }
    )


async def record_witnessed_cell(
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

    **Judged when its run names a judge, unjudged when it does not.** A run stamped by
    :func:`stamp_witnessed_judge` names the template its cells are judged against and the judge apparatus;
    the cell is then scored through the runner's own judge phase, from what the run recorded — its
    template as it stood when the run was created, each dim's recorded config, its judge pin, and the
    request settings it recorded, which must be the ones a judge call sends now — so the cell is the
    record a run's judged cell is, and a later re-judge reads it the same way. An output with nothing in
    it, or no evidence rendered, is not judged, as the runner gates it. A run naming no judge records the
    cell unjudged, with no scores.

    **What the host writes around it.** A witnessed session has no template that set it, so its case
    carries ``template_id=None`` whatever its run names: an :class:`~threetears.evals.contracts.models.EvalTestCase`
    with ``template_id=None``, its stimulus in ``variation_params`` and ``host_payload``, saved with
    ``host.storage.save_test_case`` — re-check and re-judge read it back by id, and no launch will run it,
    because a launch refuses a case whose template is not its own. Its run's ``template_id`` is None too,
    unless the run is judged, when it names the template the judge reads. A conversation its real participants
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
        ValueError: The run is not ``witnessed``; it names a template and no judge, or a judge and no
            template; ``test_case`` is not one of its cases, sits in another scope, or names a template;
            ``k_iteration`` is outside ``1..run.k_runs``; the output's stop cause is one only the engine's
            simulator produces; a judged run's kind declares nothing a judge reads; or a judged run's host
            supplies no completion clients.
        ValidationFailedError: A judged run's recorded apparatus cannot score the cell: it recorded no
            attribution, config set or request settings, or settings other than the ones a call sends
            now; its template was edited after it was created; or it recorded a judge for a dim this
            kind's cell is not scored on.
        NotFoundError: A judged run's template, or a judge config it recorded, does not load.
        CandidateKindDefect: ``output`` contradicts ``judged_artifact``.
    """
    if run.apparatus_provenance != "witnessed":
        raise ValueError(
            f"run {run.id} is {run.apparatus_provenance!r}: only a witnessed run's cells are recorded by its host. "
            "A commissioned run's cells are what its rig produced, and execute_run is what drives that rig"
        )
    if (run.judge_model is None) != (run.template_id is None):
        raise ValueError(
            f"run {run.id} names judge {run.judge_model!r} and template {run.template_id!r}: a witnessed run names a "
            "template exactly when it is judged against one — stamp both with stamp_witnessed_judge, or neither"
        )
    if test_case.id not in run.test_case_ids:
        raise ValueError(
            f"test case {test_case.id!r} is not one of run {run.id}'s cases; a run's cases are its denominator, "
            "so a cell outside them is an observation no read of the run would count"
        )
    if test_case.scope_id != run.scope_id:
        raise ValueError(
            f"test case {test_case.id!r} is in scope {test_case.scope_id!r} and run {run.id} in {run.scope_id!r}; a "
            "witnessed cell's case belongs where its run does"
        )
    if test_case.template_id is not None:
        raise ValueError(
            f"test case {test_case.id!r} is filed under template {test_case.template_id!r}; a witnessed case names "
            "none, because no template set the session and a case filed under one is a case every launch of it runs"
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
    judged, judge_ms = (None, None)
    if run.judge_model is not None:
        judge_service, template = _recorded_judge(host, run, judged_artifact)
        async with judge_service:
            judged, judge_ms = await judge_witnessed_output(
                template=template,
                test_case=test_case,
                output=output,
                judged_artifact=judged_artifact,
                judge_service=judge_service,
                concurrency=DEFAULT_JUDGE_CONCURRENCY,
                eval_run_id=run.id,
            )
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
        judge_model=run.judge_model,
        output=output,
        judged_artifact=judged_artifact,
        rate_table=external_rates,
        result_id=result_id,
        scored_at=scored_at,
        judged=judged,
        judge_ms=judge_ms,
        spans=spans,
        # Nothing here samples the host's job count: a witnessed session ran under no eval job.
        concurrent_eval_jobs=None,
        # No run ledger metered a witnessed session; the kind's own count, if any, is still folded.
        metered_cell=None,
        world_events=list(world_events) if world_events is not None else None,
        end_state=dict(end_state) if end_state is not None else None,
    )
    return result, trace


def _recorded_judge(host: EvalHost, run: EvalRun, judged_artifact: JudgedArtifact) -> tuple[JudgeService, EvalTemplate]:
    """The judge a judged witnessed run recorded, and the template it reads — refusing what cannot score the cell.

    Built from the run's own record through the checks a re-judge makes (:mod:`threetears.evals.run.rejudge`),
    so a cell is scored now under exactly the apparatus a re-judge would reproduce later.

    Args:
        host: The host: its store and its judge clients.
        run: The judged witnessed run.
        judged_artifact: What the kind's judge reads.

    Returns:
        The judge service and the template.

    Raises:
        ValueError: The kind declares nothing a judge reads, or the host supplies no completion clients.
        ValidationFailedError: The run's recorded apparatus cannot be reproduced (see the rejudge checks).
        NotFoundError: The template or a recorded config does not load.
    """
    if judged_artifact is JudgedArtifact.UNJUDGED:
        raise ValueError(
            f"run {run.id} names judge {run.judge_model!r} and kind {run.candidate_kind!r} declares nothing a judge "
            "reads; a run naming a judge claims every cell was scored by it"
        )
    judge_model = recorded_judge_pins(run, request_settings="today")
    storage = host.storage
    template = judged_template(storage, run, run.scope_id)
    dims = recorded_judged_dims(run, template, judged_artifact)
    configs = recorded_judge_configs(storage, run, dims, run.scope_id)
    # A dim the run recorded no config for is scored by the judge's built-in prompt, as a launch scores it.
    service = JudgeService(
        client_factory=judge_clients_for_run(host.completion_clients("a judged witnessed cell"), judge_model),
        configs=configs,
        failure_describer=host.failure_describer,
    )
    return service, template


__all__ = ["record_witnessed_cell", "stamp_witnessed_judge"]
