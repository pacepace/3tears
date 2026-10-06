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

**A judged witnessed run is held to a cost ceiling, as a launched run is.** The candidate's spend in a
witnessed session is the host's (real people, a rig nobody commissioned), but the judge's is the engine's,
and nothing else bounds it: the host drives recording, and re-recording a cell judges it again. So
:func:`stamp_witnessed_judge` records the run's ceiling exactly as a launch resolves one, and
:func:`record_witnessed_cell` checks it before each judged cell, against the judge spend the run's cells
already carry — the runner's between-cells check, read off the store (because the loop is the host's) plus
what the run's :class:`WitnessedJudging` judged and the host has not yet saved, one cell of a run at a time.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Literal

from threetears.evals.contracts.candidate_kind import CandidateOutput
from threetears.evals.contracts.errors import ValidationFailedError
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
from threetears.evals.run.budget import BudgetStoppedError, EvalRunCostCap
from threetears.evals.run.judge import JUDGE_REQUEST_SETTINGS
from threetears.evals.run.judge_service import JudgeService, judge_clients_for_run
from threetears.evals.run.launch import build_judge_service
from threetears.evals.run.rejudge import (
    judged_template,
    recorded_judge_configs,
    recorded_judge_pins,
    recorded_judged_dims,
)
from threetears.evals.run.offload import run_blocking
from threetears.evals.run.runner import (
    DEFAULT_JUDGE_CONCURRENCY,
    assemble_completed_cell,
    hold_to_declaration,
    judge_witnessed_output,
    refuse_engine_derived_host_measures,
    refuse_inner_agent_usage,
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
    configured_max_cost_usd: float,
    enforcement_enabled: bool,
    max_cost_usd: float | None = None,
) -> EvalRun:
    """A witnessed run that names the template its cells are judged against, the judge apparatus that scores them, and the ceiling it is held to.

    The fields a launch stamps for a judged run, resolved the way a launch resolves them
    (:func:`~threetears.evals.run.launch.build_judge_service`: one config load per scored dim, the
    run-level pin, each dim's effective judge): ``template_id``, ``judge_model`` and its provenance,
    ``effective_judges`` (recorded), ``judge_config_ids`` and their provenance, ``judge_request_settings``
    and ``rubric_scales``. Call it on the run as built and BEFORE stamping its identity — the judge is
    part of the context it hashes — then record its cells with :func:`record_witnessed_cell`, which
    scores them from exactly these.

    It also records the cost ceiling the run's judging is held to (``max_cost_usd`` and its origin),
    resolved from the same three values a launch resolves a run's from — the per-run override, the host's
    configured ceiling and whether the host enforces ceilings at all (:class:`~threetears.evals.run.budget.EvalRunCostCap`).

    Args:
        host: The host: where the judge configs are read and the judge clients come from.
        run: The witnessed run, unjudged and not yet identity-stamped.
        template: The template whose intent and rubric the judge reads; in the run's scope, and of the
            kind the run recorded.
        judge_model: The run-level judge pin, resolved.
        judged_artifact: What the kind's judge reads, which picks the dims it scores.
        selection: Optional ``{dim_id: config_id}`` naming configs per dim, as a launch's
            ``judge_config_ids`` does; the rest inherit each dim's active config.
        configured_max_cost_usd: The host's run cost ceiling, which the run inherits when ``max_cost_usd``
            names none (a launching host's ``LaunchSettings.max_cost_usd``).
        enforcement_enabled: Whether the host enforces eval ceilings; when it does not, the run records
            none and its judging is unbounded, as a launched run's is.
        max_cost_usd: Optional per-run override; must be ``> 0``.

    Returns:
        A copy of the run carrying the judge and its ceiling.

    Raises:
        ValueError: The run is not witnessed; it already names a judge or a template; its identity is
            already stamped; the template is in another scope or of another kind; ``judged_artifact``
            declares a kind no judge reads; or the host supplies no completion clients.
        ValidationFailedError: ``selection`` names an unscored dim, a config that does not load, or one
            authored for another dim; or ``max_cost_usd`` is not positive.
    """
    if max_cost_usd is not None and max_cost_usd <= 0:
        raise ValidationFailedError(f"max_cost_usd must be > 0 (got {max_cost_usd})")
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
            "max_cost_usd": EvalRunCostCap.resolve_effective_ceiling(
                max_cost_usd,
                configured_max_cost_usd=configured_max_cost_usd,
                enforcement_enabled=enforcement_enabled,
            ),
            "max_cost_usd_origin": EvalRunCostCap.resolve_ceiling_origin(
                max_cost_usd, enforcement_enabled=enforcement_enabled
            ),
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
    judging: WitnessedJudging | None = None,
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

    **Nothing is paid for a cell that cannot be recorded.** The refusals that would stop the record being
    built — a kind landing a measure the engine derives, a kind double-reporting its background work's
    spend, a variant identity the host's profile cannot derive — are made before the judge is called, as
    the runner makes them before its judge phase.

    **A judged cell is held to its run's ceiling.** Before judging, the run's judge spend so far is checked
    against the ceiling :func:`stamp_witnessed_judge` recorded, the way the runner checks its cost cap between
    cells: past it — or with any of that spend unpriced, under an enforced ceiling — the cell is refused before
    the judge is called. The spend counted is every cell the store holds and every cell judged through the
    run's ``judging`` that the host has not saved yet, and the check and the judgement it admits are one step
    per run: cells of one run recorded concurrently through one :class:`WitnessedJudging` are judged one at a
    time, each against everything judged before it, so the ceiling is overshot by at most the one cell that
    crosses it — the runner's bound. What it cannot see is a judgement made through another
    :class:`WitnessedJudging` (another process's, say) that its host has not saved: those share only the store,
    so save each pair as it is recorded.

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
        judging: For a judged run, the run's :class:`WitnessedJudging` — one per run, handed to every cell's
            recording, which serialises the cells' ceiling checks with the judgements they admit; required for a
            judged run and ignored for an unjudged one.

    Returns:
        The result and its trace, unsaved.

    Raises:
        ValueError: The run is not ``witnessed``; it names a template and no judge, or a judge and no
            template; it is judged and ``judging`` is missing or judges another run; ``test_case`` is not one of its cases, sits in another scope, or names a template;
            ``k_iteration`` is outside ``1..run.k_runs``; the output's stop cause is one only the engine's
            simulator produces; a judged run's kind declares nothing a judge reads; or a judged run's host
            supplies no completion clients.
        ValidationFailedError: A judged run's recorded apparatus cannot score the cell: it recorded no
            attribution, config set or request settings, or settings other than the ones a call sends
            now; its template was edited after it was created; or it recorded a judge for a dim this
            kind's cell is not scored on.
        NotFoundError: A judged run's template, or a judge config it recorded, does not load.
        CandidateKindDefect: ``output`` contradicts ``judged_artifact``.
        LeverCoordinateError: The host's profile cannot derive the run's variant identity.
        BudgetStoppedError: A judged run's saved cells already carry judge spend past its ceiling, or
            unpriced judge spend under an enforced one — raised before the judge is called.

    The refusals of a kind's own defects (a landed engine-derived measure, a double-reported background
    spend) raise ``ValueError`` too, before the judge is called.
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
    # Every refusal the assembly would make of this output, made BEFORE the judge is paid for — as the
    # runner makes them before its judge phase — so no judgement is bought for a record that cannot be
    # built. The assembly makes the two kind refusals again; the variant is handed to it, not re-derived.
    refuse_inner_agent_usage(output.telemetry.usage)
    refuse_engine_derived_host_measures(output.host_measures)
    # Read off the run exactly as execute_run reads it, so the two key a cell the same way.
    variant = resolve_variant_identity(run=run, profile=host.profile)

    def assemble(judged: Any, judge_ms: Any) -> tuple[EvalResult, EvalTrace]:
        return _assembled(
            run,
            test_case,
            output,
            candidate_kind=candidate_kind,
            k_iteration=k_iteration,
            variant=variant,
            judged_artifact=judged_artifact,
            external_rates=external_rates,
            result_id=result_id,
            scored_at=scored_at,
            judged=judged,
            judge_ms=judge_ms,
            spans=spans,
            world_events=world_events,
            end_state=end_state,
        )

    if run.judge_model is None:
        return assemble(None, None)
    # The ceiling check and the judgement it admits are one step per run: a second cell of the same run
    # waits here until the first has been judged and its spend counted, so cells recorded concurrently are
    # admitted one at a time, against everything judged before them — saved or not yet saved.
    if judging is None or judging.run_id != run.id:
        raise ValueError(
            f"run {run.id} is judged, and its cells are recorded through one WitnessedJudging for the run — "
            f"{'none was given' if judging is None else f'the one given judges run {judging.run_id}'}: build "
            f"WitnessedJudging({run.id!r}) once and hand it to every record_witnessed_cell of the run, so each "
            "cell's ceiling check counts every cell judged before it"
        )
    async with judging.admitting() as unsaved:
        await _refuse_past_the_judge_ceiling(host, run, unsaved)
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
        result, trace = assemble(judged, judge_ms)
        judging.count(result)
    return result, trace


def _assembled(
    run: EvalRun,
    test_case: EvalTestCase,
    output: CandidateOutput,
    *,
    candidate_kind: str,
    k_iteration: int,
    variant: Any,
    judged_artifact: JudgedArtifact,
    external_rates: ExternalRateTable | None,
    result_id: str,
    scored_at: str,
    judged: Any,
    judge_ms: Any,
    spans: CellTrace | None,
    world_events: Sequence[WorldEvent] | None,
    end_state: Mapping[str, Any] | None,
) -> tuple[EvalResult, EvalTrace]:
    """The witnessed cell through the runner's own assembly.

    Args:
        run: The witnessed run.
        test_case: The case.
        output: What the candidate produced.
        candidate_kind: The kind the run stamped.
        k_iteration: The repeat.
        variant: The run's variant identity.
        judged_artifact: What the kind's judge reads.
        external_rates: The rate table, or ``None``.
        result_id: The result's id.
        scored_at: When it was recorded.
        judged: The judge phase's outcome, or ``None`` unjudged.
        judge_ms: The judge phase's latency, or ``None``.
        spans: The host's trace record, or ``None``.
        world_events: What moved the world, or ``None``.
        end_state: The world it left, or ``None``.

    Returns:
        The result and its trace.
    """
    return assemble_completed_cell(
        scope_id=run.scope_id,
        eval_run_id=run.id,
        test_case_id=test_case.id,
        model=run.candidate_model,
        candidate_kind=candidate_kind,
        k_iteration=k_iteration,
        subject_id=run.subject_snapshot.subject_id,
        variant=variant,
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


def _judge_spend(result: EvalResult) -> float | None | Literal[False]:
    """What ``result``'s judging cost: the sum of its judge rows, ``None`` when any is unpriced, ``False`` for none.

    Args:
        result: A judged witnessed cell.

    Returns:
        The judge spend; ``None`` for an unpriced one; ``False`` when the result carries no judge row.
    """
    judging = [row.cost_usd for row in result.usage if row.role == "judge"]
    if not judging:
        return False
    return None if None in judging else math.fsum(cost for cost in judging if cost is not None)


class WitnessedJudging:
    """One judged witnessed run's judging in this process: its ceiling checks, one cell at a time, and what it judged unsaved.

    A host recording a judged witnessed run builds ONE of these for the run and hands it to every
    :func:`record_witnessed_cell` of the run's cells. It is what makes a cell's ceiling check and the
    judgement that check admits one step: a second cell of the run waits until the first has been judged and
    its spend counted, so cells recorded concurrently are admitted one at a time, each against every cell
    judged before it — the ones the store holds and the ones judged here that the host has not saved yet. The
    ceiling is then overshot by at most the one cell that crosses it, the bound the runner's between-cells check
    keeps. It is the host's to hold, as the recording loop is: the engine keeps no state of its own between
    calls.

    Attributes:
        run_id: The run whose cells it judges.
    """

    def __init__(self, run_id: str) -> None:
        """Judging for ``run_id``, nothing judged yet.

        Args:
            run_id: The judged witnessed run.
        """
        self.run_id = run_id
        self._lock = asyncio.Lock()
        # The judge spend of every cell judged here, by result id, until the store holds that result: the
        # spend a check reading the store alone would miss.
        self._unsaved: dict[str, float | None] = {}

    def count(self, result: EvalResult) -> None:
        """Count ``result``'s judging as spent, until the store holds it — what :func:`record_witnessed_cell` calls.

        Args:
            result: The cell just judged.
        """
        spend = _judge_spend(result)
        if spend is not False:
            self._unsaved[result.id] = spend

    @contextlib.asynccontextmanager
    async def admitting(self) -> AsyncIterator[dict[str, float | None]]:
        """Hold the run's ceiling check and the judgement it admits — what :func:`record_witnessed_cell` enters.

        Yields:
            The unsaved judge spend, by result id.
        """
        async with self._lock:
            yield self._unsaved


async def _refuse_past_the_judge_ceiling(host: EvalHost, run: EvalRun, unsaved: dict[str, float | None]) -> None:
    """Refuse to judge another of a judged witnessed run's cells once its judge spend has passed its ceiling.

    The runner's between-cells check (:meth:`~threetears.evals.run.budget.EvalRunCostCap.check`), over the
    judge spend of every cell of the run: the ones the store holds, and the ones this process judged that the
    host has not saved yet — the only spend of a witnessed cell the engine makes. A cell whose judge spend is
    unpriced counts as unpriced, never as $0, and stops an enforced ceiling.

    Args:
        host: The host, whose store holds the run's saved cells.
        run: The judged witnessed run.
        unsaved: The judge spend of the cells judged in this process the store did not hold yet, by result id;
            a cell the store now holds is dropped from it, since the store's copy is counted.

    Raises:
        ValueError: The run records no ceiling origin — it was not stamped by :func:`stamp_witnessed_judge`.
        BudgetStoppedError: The ceiling is enforced and the saved judge spend is past it, or unpriced.
    """
    if run.max_cost_usd_origin is None:
        raise ValueError(
            f"run {run.id} names judge {run.judge_model!r} and records no cost ceiling, and a judged witnessed run's "
            "judging is held to one; stamp it with stamp_witnessed_judge, which records the ceiling a launch would"
        )
    if run.max_cost_usd is None:
        # The host enforces no ceiling, and the run recorded that: uncapped, as a launched run would be.
        return
    saved = await run_blocking(host.blocking_executor, host.storage.query_eval_results_by_run, run.id, run.scope_id)
    for result in saved:
        unsaved.pop(result.id, None)
    cap = EvalRunCostCap(run.id, run.max_cost_usd, enabled=True)
    for result in saved:
        spend = _judge_spend(result)
        if spend is not False:
            cap.record(spend)
    for spend in unsaved.values():
        cap.record(spend)
    breach = cap.check()
    if breach is not None:
        raise BudgetStoppedError(len(saved) + len(unsaved), len(run.test_case_ids) * run.k_runs, breach)


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


__all__ = ["WitnessedJudging", "record_witnessed_cell", "stamp_witnessed_judge"]
