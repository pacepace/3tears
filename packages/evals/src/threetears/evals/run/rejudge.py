"""Re-judge the dimensions a result's judge phase failed to score.

A judge call can fail after the trial it scores has finished — a reply cut off at its output
cap, an unparseable score — and the cell then stores that dimension as missing and carries a
``judge_error`` that excludes the whole result from scoring. Re-running the trial to recover
one score would re-buy the candidate, the simulated user and every judge call that succeeded,
and would measure a different conversation. A re-judge asks the same judge the same question
again instead, from the evidence the cell's kind rendered for its first judge, which the cell's
trace stores — never by re-rendering it, which would need the kind's renderer as it was then.

This module holds which dims failed, how a re-judge's outcomes land on the result, and the one
answer to "can this result's judge input be rebuilt from what its run recorded?"
(:func:`reproducible_judge_inputs`) — asked by :func:`~threetears.evals.run.lifecycle.rejudge_result` before it pays
for a call and by the judge freeze before it renders one. Every refusal exists because a re-score
under an apparatus the run did not record would be a different measurement filed under the old one.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol

from threetears.evals.contracts.errors import NotFoundError, ValidationFailedError
from threetears.evals.contracts.scoring import boundary_dim_names
from threetears.evals.contracts.models import (
    NON_TERMINAL_RUN_STATUSES,
    OUTCOME_DIM_ID,
    TRANSCRIPT_DIM_ID,
    JudgedArtifact,
    JudgeRescore,
    scored_dim_ids,
)
from threetears.evals.run.judge import JUDGE_REQUEST_SETTINGS
from threetears.evals.run.judge_service import fold_judge_outcomes

if TYPE_CHECKING:
    from threetears.evals.contracts.storage import EvalStorage
    from threetears.evals.contracts.models import (
        EvalResult,
        EvalRun,
        EvalTemplate,
        EvalTestCase,
        EvalTrace,
        JudgeConfig,
        JudgeEvidence,
    )
    from threetears.evals.run.judge_service import JudgeOutcome

#: How a caller treats the run's recorded judge request settings. ``"today"``: the judge will be
#: CALLED, so the settings the client sends now must be the ones the run recorded, or the call
#: would be asked differently from the rest of the run. ``"as_recorded"``: nothing is sent (a
#: freeze renders prompts), so the recorded settings are honoured as they are and only their
#: absence refuses.
RequestSettingsPolicy = Literal["today", "as_recorded"]


class JudgeInputStore(Protocol):
    """The reads :func:`reproducible_judge_inputs` makes, and no more.

    Structural, so a host's own storage satisfies it by having the methods. The scope parameters
    are positional-only, which lets this port say ``scope_id`` — the engine's word for a partition
    it never interprets — while an implementation names what it partitions by.
    """

    def load_template(self, template_id: str, scope_id: str, /) -> EvalTemplate | None:
        """One template within a scope, or ``None`` when it does not resolve."""
        ...

    def load_judge_config(self, config_id: str, scope_id: str, /) -> JudgeConfig | None:
        """One versioned judge config within a scope, or ``None`` when it does not resolve."""
        ...

    def load_test_case(self, test_case_id: str, scope_id: str, /) -> EvalTestCase | None:
        """One test case within a scope, or ``None`` when it does not resolve."""
        ...

    def load_eval_trace(self, result_id: str, scope_id: str, /) -> EvalTrace | None:
        """One result's stored trace within a scope, or ``None`` when none is stored."""
        ...


class ReproducibleJudgeInputs(NamedTuple):
    """The stored records a result's judge calls are rebuilt from, each the one its run recorded."""

    template: EvalTemplate
    test_case: EvalTestCase
    #: What the first judge read, as the cell's kind rendered it — stored on the cell's trace.
    judge_evidence: JudgeEvidence
    #: The kind's declaration that picked the first judge's axes, stored beside the evidence.
    judged_artifact: JudgedArtifact
    #: The versioned config per dim, for the dims asked for that have one.
    configs: dict[str, JudgeConfig]
    #: Every dim this result was judged on, in the order the judge phase calls them.
    dims: list[str]
    #: The run's judge pin, which the check refuses a run without.
    judge_model: str


def reproducible_judge_inputs(
    storage: JudgeInputStore,
    result: EvalResult,
    run: EvalRun,
    scope_id: str,
    *,
    request_settings: RequestSettingsPolicy,
    config_dims: Collection[str] | None = None,
) -> ReproducibleJudgeInputs:
    """Load what a result's judge calls read, refusing any input its run did not record.

    The one check behind both a re-judge and a judge freeze: each input a re-render needs — the
    scored dim set, the judge pin, the versioned config per dim, the request settings, the
    template's intent and rubric, the test case, and the evidence the cell's kind rendered for
    its judge — must be the one the run recorded, or the result is refused naming it. Falling
    back to today's value for any of them would render a different question and file it under
    the old one.

    Args:
        storage: The eval store the result lives in.
        result: The result whose judge input is rebuilt.
        run: Its run.
        scope_id: The partition they live in.
        request_settings: How the run's recorded judge request settings are treated — see
            :data:`RequestSettingsPolicy`.
        config_dims: The dims whose recorded configs must load; ``None`` for every dim the run judged.

    Returns:
        The records.

    Raises:
        ValidationFailedError: An input the run recorded cannot be reproduced, naming it.
        NotFoundError: A record the run names (template, judge config, test case) does not load.
    """
    if run.status in NON_TERMINAL_RUN_STATUSES:
        raise ValidationFailedError(f"run '{run.id}' is {run.status} — its results are still being written")
    judge_model = recorded_judge_pins(run, request_settings=request_settings)
    template = judged_template(storage, run, scope_id)
    # What the first judge read, and the declaration that picked its axes, as the cell stored
    # them. Read before anything is derived from the dim set, because the declaration picks it.
    trace = storage.load_eval_trace(result.id, scope_id)
    if trace is None or trace.judge_evidence is None or trace.judged_artifact is None:
        raise ValidationFailedError(
            f"result '{result.id}' stores no judge evidence, so what its judge read cannot be sent again"
        )
    dims = recorded_judged_dims(run, template, trace.judged_artifact)
    # A result scored under a config its run did not record read a prompt nothing else names.
    assert run.judge_config_ids is not None  # recorded_judge_pins refuses a run without
    for dim, config_id in result.judge_config_ids.items():
        if run.judge_config_ids.get(dim) != config_id:
            raise ValidationFailedError(
                f"result '{result.id}' was scored on {dim} by config '{config_id}' while its run recorded "
                f"{run.judge_config_ids.get(dim)!r}, so which prompt the judge read is ambiguous"
            )
    configs = recorded_judge_configs(
        storage, run, dims if config_dims is None else [dim for dim in dims if dim in config_dims], scope_id
    )
    test_case = storage.load_test_case(result.test_case_id, scope_id)
    if test_case is None:
        raise NotFoundError("test_case", result.test_case_id)
    return ReproducibleJudgeInputs(
        template=template,
        test_case=test_case,
        judge_evidence=trace.judge_evidence,
        judged_artifact=trace.judged_artifact,
        configs=configs,
        dims=dims,
        judge_model=judge_model,
    )


def recorded_judge_pins(run: EvalRun, *, request_settings: RequestSettingsPolicy) -> str:
    """The run's judge pin, refusing a run whose judging apparatus it did not record.

    One check behind every path that scores a result under a run's recorded judge — a re-judge, a
    freeze, and the judging of a witnessed cell as it is recorded — so a score is never filed under a
    judge, a config set or request settings the run does not name.

    Args:
        run: The run.
        request_settings: How the run's recorded request settings are treated — see :data:`RequestSettingsPolicy`.

    Returns:
        The run's judge pin.

    Raises:
        ValidationFailedError: The run names no judge, recorded no attribution, no config set or no
            request settings, or (``"today"``) recorded settings other than the ones a call sends now.
    """
    if run.judge_model is None:
        raise ValidationFailedError(f"run '{run.id}' was not judged, so there is no judgement to reproduce")
    if not run.effective_judges:
        raise ValidationFailedError(
            f"run '{run.id}' recorded no judge attribution, so which judge scored which dim cannot be reproduced"
        )
    if run.judge_config_ids is None:
        raise ValidationFailedError(
            f"run '{run.id}' recorded no judge config set, so which prompt scored each dim is unknown"
        )
    if run.judge_request_settings is None:
        raise ValidationFailedError(
            f"run '{run.id}' recorded no judge request settings, so how the judge was asked is unknown"
        )
    if request_settings == "today" and run.judge_request_settings != JUDGE_REQUEST_SETTINGS:
        raise ValidationFailedError(
            f"run '{run.id}' was judged with request settings {run.judge_request_settings!r} and the judge client "
            f"now sends {JUDGE_REQUEST_SETTINGS!r}; a new call would be asked differently from the rest of the run"
        )
    return run.judge_model


def judged_template(storage: JudgeInputStore, run: EvalRun, scope_id: str) -> EvalTemplate:
    """The template whose intent and rubric a run's judge reads, as the run was judged against it.

    Args:
        storage: The eval store.
        run: The run.
        scope_id: The partition it lives in.

    Returns:
        The template.

    Raises:
        ValidationFailedError: The run names no template; the template was edited after the run was
            created, so its intent is not the one the run was judged against; or its rubric names one
            dim twice.
        NotFoundError: The template does not load.
    """
    if run.template_id is None:
        raise ValidationFailedError(f"run '{run.id}' is ad hoc, so there is no template to read its intent from")
    template = storage.load_template(run.template_id, scope_id)
    if template is None:
        raise NotFoundError("template", run.template_id)
    # Templates are edited in place, and the judge reads the template's intent (and a config-less
    # dim's own text): an edit after launch would change the question.
    if template.updated_at > run.created_at:
        raise ValidationFailedError(
            f"template '{template.id}' was edited at {template.updated_at}, after run '{run.id}' launched at "
            f"{run.created_at}, so the judge would read an intent the run was not judged against"
        )
    # Scores, configs and errors are keyed by dim NAME everywhere a result stores them, so a
    # rubric naming one dim twice cannot say which of the two a stored score belongs to.
    names = [dim.name for dim in template.rubric]
    if duplicated := sorted({name for name in names if names.count(name) > 1}):
        raise ValidationFailedError(
            f"template '{template.id}' names rubric dim(s) {', '.join(duplicated)} more than once, so which "
            "of them a stored score belongs to cannot be told apart"
        )
    return template


def recorded_judged_dims(run: EvalRun, template: EvalTemplate, judged_artifact: JudgedArtifact) -> list[str]:
    """Every dim a cell of this kind is judged on under the run's recorded attribution, in judge-phase order.

    Args:
        run: The run, whose ``effective_judges`` a caller has checked is recorded.
        template: Its template.
        judged_artifact: The kind's declaration, which picks the axes.

    Returns:
        The dims.

    Raises:
        ValidationFailedError: The run recorded a judge for a dim a cell of this kind is not scored on.
    """
    # The dims a cell of this kind is scored on, in judge-phase order — the rule the launch
    # attributed judges by, so a run that recorded a judge for a dim outside it is refused.
    order = scored_dim_ids([dim.name for dim in template.rubric], judged_artifact)
    recorded = run.effective_judges or {}
    if strays := sorted(set(recorded) - set(order)):
        raise ValidationFailedError(
            f"run '{run.id}' judged {', '.join(strays)}, which a {judged_artifact.value} cell of its "
            "template is not scored on"
        )
    return [dim for dim in order if dim in recorded]


def recorded_judge_configs(
    storage: JudgeInputStore, run: EvalRun, dims: Collection[str], scope_id: str
) -> dict[str, JudgeConfig]:
    """The versioned judge config the run recorded for each of ``dims`` that has one, loaded.

    Args:
        storage: The eval store.
        run: The run, whose ``judge_config_ids`` a caller has checked is recorded.
        dims: The dims whose configs to load.
        scope_id: The partition they live in.

    Returns:
        ``{dim_id: config}`` over the dims the run recorded a config for.

    Raises:
        NotFoundError: A recorded config does not load.
    """
    configs: dict[str, JudgeConfig] = {}
    for dim in dims:
        if (recorded_id := (run.judge_config_ids or {}).get(dim)) is None:
            continue
        config = storage.load_judge_config(recorded_id, scope_id)
        if config is None:
            raise NotFoundError("judge_config", recorded_id)
        configs[dim] = config
    return configs


def failed_judge_dims(result: EvalResult, scored_dims: list[str]) -> list[str]:
    """The dims a result's judging failed to answer, in the order given.

    A dim is failed when it holds neither a score nor a recorded "can't tell". **A can't-tell is an
    answer, not a failure**: the judge worked and said the evidence does not decide the dim, so it
    holds no score by contract (:class:`~threetears.evals.run.judge_service.JudgeOutcome`) and is
    never re-asked. Re-asking it would pay for a call nothing failed, and storing the fresh sample
    over it would file a second measurement under the first judge's record.

    Derived from what the result holds, never by parsing ``judge_error``: that field is a display
    string joined from per-dim messages, and a message may itself contain the separator.

    Args:
        result: The result whose judging is being inspected.
        scored_dims: Every dim the run scores, as its launch recorded them.

    Returns:
        Each dim in ``scored_dims`` with no stored score and no recorded can't-tell.
    """
    answered = {score.dim for score in result.rubric_scores} | set(result.judge_cannot_tell)
    if result.transcript_score is not None:
        answered.add(TRANSCRIPT_DIM_ID)
    if result.outcome_score is not None:
        answered.add(OUTCOME_DIM_ID)
    return [dim for dim in scored_dims if dim not in answered]


def apply_rejudge(
    result: EvalResult,
    template: EvalTemplate,
    outcomes: list[tuple[str, JudgeOutcome]],
    *,
    judge_model: str,
) -> EvalResult:
    """Land a re-judge's outcomes on a result where the judge phase would have put them.

    A score goes to its axis slot, or into ``rubric_scores`` in the template's rubric order;
    the configs, errors and spend are folded exactly as the phase folds them
    (:func:`~threetears.evals.run.judge_service.fold_judge_outcomes`); ``judge_error`` is recomposed
    from the dims that failed again, in the phase's own ``"<dim>: <error>"`` form, and cleared
    when none did. A :class:`~threetears.evals.contracts.models.JudgeRescore` recording all of it is appended.

    ``cost_usd`` and ``usage`` are left as they are — see ``JudgeRescore`` for why.

    Args:
        result: The stored result. Not mutated.
        template: The run's template, whose rubric order the scores are kept in.
        outcomes: ``(dim_id, outcome)`` for every dim re-asked, in dimension order.
        judge_model: The run's judge pin, recorded on the rescore.

    Returns:
        The result as it now stands.

    Raises:
        ValueError: ``outcomes`` is empty, or ``result`` carries no ``judge_error`` to record
            as the prior one — both are states the caller refuses before paying for a call.
    """
    if not outcomes:
        raise ValueError("a re-judge with no outcomes records nothing")
    if result.judge_error is None:
        raise ValueError(f"result {result.id!r} has no judge_error, so nothing failed to re-judge")

    folded = fold_judge_outcomes(outcomes)
    rubric_order = {dim.name: index for index, dim in reversed(list(enumerate(template.rubric)))}
    transcript_score = result.transcript_score
    outcome_score = result.outcome_score
    judge_reasoning = result.judge_reasoning
    rubric_scores = list(result.rubric_scores)
    scores: dict[str, int] = {}
    for dim_id, outcome in outcomes:
        score = outcome.score
        if score is None:
            continue
        scores[dim_id] = score.score
        if dim_id == TRANSCRIPT_DIM_ID:
            transcript_score = score
        elif dim_id == OUTCOME_DIM_ID:
            outcome_score = score
            # The result's ``judge_reasoning`` mirrors the outcome axis, as the phase sets it.
            judge_reasoning = score.reasoning
        else:
            rubric_scores.append(score)

    rubric_scores.sort(key=lambda s: rubric_order.get(s.dim, len(rubric_order)))
    config_ids = {**result.judge_config_ids, **folded.config_ids}
    errors = dict(folded.errors)
    rescore = JudgeRescore(
        dims=[dim_id for dim_id, _ in outcomes],
        prior_judge_error=result.judge_error,
        scores=scores,
        errors=errors,
        cannot_tell=folded.cannot_tell,
        judge_model=judge_model,
        judge_config_ids=folded.config_ids,
        usage=folded.usage,
        cost_usd=folded.cost_usd,
    )
    return result.model_copy(
        update={
            "transcript_score": transcript_score,
            "outcome_score": outcome_score,
            "judge_reasoning": judge_reasoning,
            "rubric_scores": rubric_scores,
            "judge_config_ids": config_ids,
            "judge_error": "; ".join(f"{dim_id}: {error}" for dim_id, error in folded.errors) or None,
            # A recorded can't-tell is never re-asked (``failed_judge_dims``), so it stands; a dim
            # that failed before and is now answered "can't tell" leaves the errors and joins it.
            "judge_cannot_tell": {**result.judge_cannot_tell, **folded.cannot_tell},
            # Stamped from the template's definitions, as the phase stamps them, so a guardrail's can't-tell
            # never drops the trial from the capability measures.
            "judge_cannot_tell_boundary": sorted(
                set(result.judge_cannot_tell_boundary)
                | {dim for dim in folded.cannot_tell if dim in boundary_dim_names(template.rubric)}
            ),
            "judge_rescores": [*result.judge_rescores, rescore],
        }
    )


if TYPE_CHECKING:

    def _eval_storage_satisfies_the_port(storage: EvalStorage) -> None:
        """Hold the engine's own store to this consumer's port, so a drifted signature fails typecheck."""
        store: JudgeInputStore = storage
        del store


__all__ = [
    "JudgeInputStore",
    "ReproducibleJudgeInputs",
    "RequestSettingsPolicy",
    "apply_rejudge",
    "failed_judge_dims",
    "judged_template",
    "recorded_judge_configs",
    "recorded_judge_pins",
    "recorded_judged_dims",
    "reproducible_judge_inputs",
]
