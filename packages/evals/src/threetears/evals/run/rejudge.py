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

**Claim criteria** are re-judged beside the dims, by the same rule (:func:`reproducible_claim_inputs`): from the claim
evidence the cell's trace stores, by the run's judge pin, and only under the definition the cell recorded for each
criterion (``EvalResult.claim_fingerprints``), refused by name when today's fingerprints otherwise. A lost criterion
is graded again whole; a completed grade is never asked again, a lost ground step on it included, since its truth
stands. A criterion whose tally waited on its trace (``trace_pending``) costs no call: its tally is recomputed from
the grade the trace stores (:func:`~threetears.evals.kernel.claims.grading.grade_tally`), and once the run has ended a
pending criterion whose trace was never stored is lost (``trace_lost``). Claims calls bind no inputs, so a replay keys
each by its prompt: a document changed since the recording is another call, and misses.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol

from threetears.evals.kernel.claims.grading import (
    GradedCriterion,
    criterion_fingerprints,
    grade_tally,
    refuse_unbuilt_claim_steps,
)
from threetears.evals.kernel.errors import NotFoundError, ValidationFailedError
from threetears.evals.kernel.scoring import boundary_dim_names
from threetears.evals.schema.models import (
    CLAIM_PENDING,
    DEFAULT_JUDGE_TEMPERATURE,
    NON_TERMINAL_RUN_STATUSES,
    OUTCOME_DIM_ID,
    TRANSCRIPT_DIM_ID,
    ClaimRescore,
    JudgedArtifact,
    JudgeRescore,
    judge_config_of,
    scored_dim_ids,
)
from threetears.evals.run.judge import JUDGE_REQUEST_SETTINGS
from threetears.evals.run.judge_service import fold_judge_outcomes, judge_config_ids_for
from threetears.evals.kernel.remedies import FIND_JUDGE_CONFIG, FIND_TEMPLATE, FIND_TEST_CASE, RELAUNCH_TO_REJUDGE

if TYPE_CHECKING:
    from threetears.evals.kernel.storage import EvalStorage
    from threetears.evals.schema.completion import CompletionClient
    from threetears.evals.schema.models import (
        ClaimEvidence,
        ClaimGrade,
        ClaimLoss,
        ClaimTally,
        ClientRequestSettings,
        RoleUsage,
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
    same_judge_dims: Collection[str] = (),
) -> ReproducibleJudgeInputs:
    """Load what a result's judge calls read, refusing any input its run did not record.

    The one check behind both a re-judge and a judge freeze: each input a re-render needs — the
    scored dim set, the judge pin, the versioned config per dim, the request settings, the
    template's intent and rubric, the test case, and the evidence the cell's kind rendered for
    its judge — must be the one the run recorded, or the result is refused naming it. Falling
    back to today's value for any of them would render a different question and file it under
    the old one.

    :param storage: The eval store the result lives in.
    :param result: The result whose judge input is rebuilt.
    :param run: Its run.
    :param scope_id: The partition they live in.
    :param request_settings: How the run's recorded judge request settings are treated — see
        :data:`RequestSettingsPolicy`.
    :param config_dims: The dims whose recorded configs must load; ``None`` for every dim the run judged.
    :param same_judge_dims: The dims the caller asks again AS the judge that first asked them (a re-judge, a judge
        repeat), rather than as a judge of its own choosing (a second judge, a freeze). Each must have been
        asked by the prompt the judge sends it today — the built-in prompt, or its recorded config with the
        engine's reply-format suffix: the result must record that prompt's identity for it, and it must be
        today's.
    :return: The records.
    :raises ValidationFailedError: An input the run recorded cannot be reproduced, naming it.
    :raises NotFoundError: A record the run names (template, judge config, test case) does not load.
    """
    if run.status in NON_TERMINAL_RUN_STATUSES:
        raise ValidationFailedError(
            f"run '{run.id}' is {run.status} — its results are still being written",
            remedy=f"wait for it to end (job_poll run:{run.id}), then ask again",
        )
    judge_model = recorded_judge_pins(run, request_settings=request_settings)
    template = judged_template(storage, run, scope_id)
    # What the first judge read, and the declaration that picked its axes, as the cell stored
    # them. Read before anything is derived from the dim set, because the declaration picks it.
    trace = storage.load_eval_trace(result.id, scope_id)
    if trace is None or trace.judge_evidence is None or trace.judged_artifact is None:
        raise ValidationFailedError(
            f"result '{result.id}' stores no judge evidence, so what its judge read cannot be sent again",
            remedy=RELAUNCH_TO_REJUDGE,
        )
    dims = recorded_judged_dims(run, template, trace.judged_artifact)
    # A result scored under a config its run did not record read a prompt nothing else names. A configured judge's
    # recorded identity names its config before the reply format's fingerprint, and a built-in prompt's identity
    # is the run's "no config" for its dim.
    assert run.judge_config_ids is not None  # recorded_judge_pins refuses a run without
    for dim, config_id in result.judge_config_ids.items():
        recorded = run.judge_config_ids.get(dim)
        if recorded != judge_config_of(config_id):
            raise ValidationFailedError(
                f"result '{result.id}' was scored on {dim} by config '{config_id}' while its run recorded "
                f"{recorded!r}, so which prompt the judge read is ambiguous",
                remedy=RELAUNCH_TO_REJUDGE,
            )
    _refuse_another_judge_prompt(
        result,
        run.judge_config_ids,
        [dim for dim in dims if dim in same_judge_dims],
        judge_config_ids_for(template.rubric, run.judge_config_ids, judged_artifact=trace.judged_artifact),
    )
    configs = recorded_judge_configs(
        storage, run, dims if config_dims is None else [dim for dim in dims if dim in config_dims], scope_id
    )
    test_case = storage.load_test_case(result.test_case_id, scope_id)
    if test_case is None:
        raise NotFoundError("test_case", result.test_case_id, remedy=FIND_TEST_CASE)
    return ReproducibleJudgeInputs(
        template=template,
        test_case=test_case,
        judge_evidence=trace.judge_evidence,
        judged_artifact=trace.judged_artifact,
        configs=configs,
        dims=dims,
        judge_model=judge_model,
    )


class ReproducibleClaimInputs(NamedTuple):
    """What a re-judge of a result's lost claim criteria reads, each as its run recorded it, and what it will do."""

    template: EvalTemplate
    test_case: EvalTestCase
    #: The run's judge pin, which every claims call is sent on.
    judge_model: str
    #: The cell's trace as stored, whose claim evidence a criterion is graded again from and whose grades a pending
    #: criterion's tally is recomputed from; ``None`` when none is stored.
    trace: EvalTrace | None
    #: The lost criteria to grade again whole, in the template's order.
    regrade: tuple[str, ...]
    #: The pending criteria (``trace_pending``) whose tally the stored grade gives, recomputed with no call.
    recovered: dict[str, ClaimTally]
    #: The pending criteria whose trace was never stored, lost now that the run has ended (``trace_lost``).
    unrecoverable: tuple[str, ...]

    @property
    def claim_evidence(self) -> ClaimEvidence | None:
        """The claim evidence the cell's trace stores, that a criterion is graded again from."""
        return None if self.trace is None else self.trace.claim_evidence


def reproducible_claim_inputs(
    storage: JudgeInputStore, result: EvalResult, run: EvalRun, scope_id: str, *, client: CompletionClient
) -> ReproducibleClaimInputs:
    """Load what a re-judge of ``result``'s lost claim criteria reads, refusing any input its run did not record.

    The claims half of :func:`reproducible_judge_inputs`, by the same rule: the run has ended, its judge pin and its
    request settings are today's (``request_settings="today"``, since a call is sent), its template was not edited
    since it launched, and its case loads. A lost criterion is graded again whole from the claim evidence the trace
    stores, only under the definition the cell recorded for it (``EvalResult.claim_fingerprints``): today's
    fingerprint otherwise, or none recorded, refuses it by name, as a dim asked by another prompt is refused. A
    completed grade is never asked again. A pending criterion (``trace_pending``) is recomputed from the grade the
    trace stores, or lost (``trace_lost``) when the trace is not stored.

    :param storage: The eval store the result lives in.
    :param result: The result whose claim criteria are re-judged.
    :param run: Its run.
    :param scope_id: The partition they live in.
    :param client: The judge's client every claims call would be sent on, which names its model; today's
        fingerprints are read through it.
    :return: The records, and what the re-judge will do with each lost criterion.
    :raises ValidationFailedError: The result lost no criterion; the run is still running or recorded no judge; a
        criterion it lost is no longer the template's, declares a step this build does not run, would grade under
        another definition than the cell recorded, or has no stored claim evidence to be graded again from; or a
        stored grade does not give a tally.
    :raises NotFoundError: The template or the case does not load.
    """
    if not result.claim_failures:
        raise ValidationFailedError(
            f"result '{result.id}' lost no claim criterion, so there is none to re-judge",
            remedy="read its claim readings as they are",
        )
    if run.status in NON_TERMINAL_RUN_STATUSES:
        raise ValidationFailedError(
            f"run '{run.id}' is {run.status} — its results are still being written",
            remedy=f"wait for it to end (job_poll run:{run.id}), then ask again",
        )
    judge_model = recorded_judge_pins(run, request_settings="today", claims_only=True)
    template = judged_template(storage, run, scope_id)
    criteria = {criterion.name: criterion for criterion in template.claim_criteria}
    if strays := sorted(set(result.claim_failures) - set(criteria)):
        raise ValidationFailedError(
            f"result '{result.id}' lost claim criteria {', '.join(map(repr, strays))}, which template "
            f"'{template.id}' no longer declares",
            remedy=RELAUNCH_TO_REJUDGE,
        )
    try:
        refuse_unbuilt_claim_steps(template)
    except ValueError as unbuilt:
        raise ValidationFailedError(str(unbuilt), remedy=RELAUNCH_TO_REJUDGE) from unbuilt
    trace = storage.load_eval_trace(result.id, scope_id)
    pending = [name for name in criteria if result.claim_failures.get(name) == CLAIM_PENDING]
    grades = {} if trace is None else trace.claim_grades
    recovered: dict[str, ClaimTally] = {}
    for name in pending:
        if name not in grades or trace is None or trace.claim_evidence is None:
            continue
        grade = grades[name]
        if (recorded := result.claim_fingerprints.get(name)) is not None and recorded != grade.criterion_fingerprint:
            raise ValidationFailedError(
                f"result '{result.id}' recorded claim criterion {name!r} under fingerprint {recorded[:12]}, and the "
                f"grade its trace stores under {grade.criterion_fingerprint[:12]}",
                remedy=RELAUNCH_TO_REJUDGE,
            )
        try:
            recovered[name] = grade_tally(grade, trace.claim_evidence, criteria[name])
        except ValueError as unreadable:
            raise ValidationFailedError(
                f"result '{result.id}''s stored grade under claim criterion {name!r} gives no tally: {unreadable}",
                remedy=RELAUNCH_TO_REJUDGE,
            ) from unreadable
    evidence = None if trace is None else trace.claim_evidence
    unrecoverable = tuple(name for name in pending if name not in recovered and evidence is None)
    regrade = tuple(
        name
        for name in criteria
        if name in result.claim_failures and name not in recovered and name not in unrecoverable
    )
    if regrade and evidence is None:
        raise ValidationFailedError(
            f"result '{result.id}' lost claim criteria {', '.join(map(repr, regrade))}, and its trace stores no claim "
            "evidence to grade them again from",
            remedy=RELAUNCH_TO_REJUDGE,
        )
    today = criterion_fingerprints(template, client)
    for name in regrade:
        recorded = result.claim_fingerprints.get(name)
        if recorded is None:
            raise ValidationFailedError(
                f"result '{result.id}' records no fingerprint for claim criterion {name!r}, so the definition it would "
                "be graded again under cannot be told to be the one its run graded",
                remedy=RELAUNCH_TO_REJUDGE,
            )
        if recorded != today[name]:
            raise ValidationFailedError(
                f"result '{result.id}' was due claim criterion {name!r} under fingerprint {recorded[:12]}, and it would "
                f"now be graded under {today[name][:12]}; a new grade would be a different instrument's",
                remedy=RELAUNCH_TO_REJUDGE,
            )
    test_case = storage.load_test_case(result.test_case_id, scope_id)
    if test_case is None:
        raise NotFoundError("test_case", result.test_case_id, remedy=FIND_TEST_CASE)
    return ReproducibleClaimInputs(
        template=template,
        test_case=test_case,
        judge_model=judge_model,
        trace=trace,
        regrade=regrade,
        recovered=recovered,
        unrecoverable=unrecoverable,
    )


def apply_claim_rejudge(
    result: EvalResult,
    inputs: ReproducibleClaimInputs,
    regraded: dict[str, GradedCriterion | ClaimLoss],
    *,
    usage: list[RoleUsage],
    cost_usd: float | None,
) -> tuple[EvalResult, EvalTrace | None]:
    """Land a claims re-judge on the result and its trace where the claims phase would have put it.

    Each criterion graded again lands its tally on the result and its grade on the trace, or its loss again; each
    pending criterion takes the tally its stored grade gives, or ``trace_lost`` when its trace was never stored.
    Every other criterion, tallied or lost, is left as it stands. A
    :class:`~threetears.evals.schema.models.ClaimRescore` recording all of it is appended.

    :param result: The stored result. Not mutated.
    :param inputs: What the re-judge read (:func:`reproducible_claim_inputs`).
    :param regraded: criterion -> its new grade, or its loss, for each of ``inputs.regrade``.
    :param usage: The re-judge's judge-role usage rows.
    :param cost_usd: What it cost; ``None`` when any call went unpriced.
    :return: The result as it now stands, and the trace to store first when a grade moved on it (``None`` when none
        did).
    :raises ValueError: ``regraded`` does not name exactly the criteria ``inputs`` grades again.
    """
    if set(regraded) != set(inputs.regrade):
        raise ValueError(
            f"a claims re-judge grades again {sorted(inputs.regrade)}, and its outcomes name {sorted(regraded)}"
        )
    tallies = dict(result.claim_tallies)
    failures = dict(result.claim_failures)
    lost: dict[str, ClaimLoss] = {}
    grades: dict[str, ClaimGrade] = {}
    for name, tally in inputs.recovered.items():
        failures.pop(name)
        tallies[name] = tally
    for name in inputs.unrecoverable:
        failures[name] = lost[name] = "trace_lost"
    for name, outcome in regraded.items():
        if isinstance(outcome, str):
            failures[name] = lost[name] = outcome
        else:
            failures.pop(name)
            tallies[name] = outcome.tally
            grades[name] = outcome.grade
    trace = inputs.trace
    if trace is not None and (grades or set(regraded) & set(trace.claim_grades)):
        kept = {name: grade for name, grade in trace.claim_grades.items() if name not in regraded}
        trace = trace.model_copy(update={"claim_grades": {**kept, **grades}})
    else:
        trace = None
    rescore = ClaimRescore(
        criteria=list(inputs.regrade),
        recovered=[*inputs.recovered, *inputs.unrecoverable],
        prior={
            name: result.claim_failures[name] for name in (*inputs.regrade, *inputs.recovered, *inputs.unrecoverable)
        },
        lost=lost,
        judge_model=inputs.judge_model,
        usage=usage,
        cost_usd=cost_usd,
    )
    # Every criterion moved is in exactly one of the two maps, and each new tally carries the fingerprint the cell
    # recorded for it (a regraded one was refused otherwise, a recovered one is its stored grade's).
    updated = result.model_copy(
        update={
            "claim_tallies": tallies,
            "claim_failures": failures,
            "claim_rescores": [*result.claim_rescores, rescore],
        }
    )
    return updated, trace


def _refuse_another_judge_prompt(
    result: EvalResult, run_configs: dict[str, str], dims: list[str], today: dict[str, str]
) -> None:
    """Refuse re-asking a dim with today's prompt when it is not the one that first asked it.

    What a judge reads is an instrument with an identity: the built-in wording
    (:func:`~threetears.evals.schema.models.builtin_judge_config_id`), and a config's prompt with the reply-format
    suffix the engine appends to it (:func:`~threetears.evals.schema.models.configured_judge_config_id`). Asked
    again under another wording, a dim's new score is a different judge's, filed beside its run's other scores as
    if it were the same one.

    :param result: The result whose dims are re-asked.
    :param run_configs: The config set its run recorded; a dim with none in it is asked with the built-in prompt.
    :param dims: The dims re-asked as the judge that first asked them.
    :param today: What each dim would be asked by now
        (:func:`~threetears.evals.run.judge_service.judge_config_ids_for`).
    :raises ValidationFailedError: A dim's recorded prompt is not today's, or the result records none for it.
    """
    for dim in dims:
        recorded = result.judge_config_ids.get(dim)
        if recorded is None:
            raise ValidationFailedError(
                f"result '{result.id}' records no judge for {dim}: no judge phase of its run asked it, so there is no "
                "first judge to ask it again as",
                remedy=RELAUNCH_TO_REJUDGE,
            )
        if recorded != today[dim]:
            raise ValidationFailedError(
                f"result '{result.id}' was judged on {dim} by prompt '{recorded}', and the judge would now ask it "
                f"with '{today[dim]}'; a new score would be a different judge's",
                remedy=RELAUNCH_TO_REJUDGE,
            )


def request_settings_drift(recorded: ClientRequestSettings | None) -> str:
    """How the judge request settings a run recorded it asked for differ from the ones a judge call asks now, in words.

    Two settings can differ where this build cannot see: a run a newer build judged may record a field of the
    settings this build does not know, which its read keeps, so the two compare unequal while every field this
    build declares agrees — and printing both would show two identical values. That case, and only that case, is
    named as a field this build does not know: settings that are merely read from a newer minor's run, with nothing
    kept on them, compare equal to today's and are said to match.

    :param recorded: The judge request settings the run recorded it asked for (``judge_asked_settings``; ``None`` when
        it recorded none).
    :return: A clause for a refusal: "asked its judge for request settings ... and a judge call now asks for ...".
    """
    if recorded == JUDGE_REQUEST_SETTINGS:
        return f"asked its judge for the request settings a judge call now asks for ({JUDGE_REQUEST_SETTINGS!r})"
    # Unequal, yet equal in every declared field: the difference is a field a newer build wrote, which equality
    # compares (the minor a shape was read at is not compared, so a run a newer minor wrote is not this case).
    if recorded is not None and recorded.model_dump() == JUDGE_REQUEST_SETTINGS.model_dump():
        return (
            "asked its judge for request settings that match the ones a judge call now asks for "
            f"({JUDGE_REQUEST_SETTINGS!r}) "
            "in every field this build knows, and carry a field it does not, which a newer build recorded"
        )
    return f"asked its judge for request settings {recorded!r} and a judge call now asks for {JUDGE_REQUEST_SETTINGS!r}"


def recorded_judge_pins(run: EvalRun, *, request_settings: RequestSettingsPolicy, claims_only: bool = False) -> str:
    """The run's judge pin, refusing a run whose judging apparatus it did not record.

    One check behind every path that scores a result under a run's recorded judge — a re-judge, a
    freeze, and the judging of a witnessed cell as it is recorded — so a score is never filed under a
    judge, a config set or request settings the run does not name.

    :param run: The run.
    :param request_settings: How the run's recorded request settings are treated — see :data:`RequestSettingsPolicy`.
    :param claims_only: Whether the judge is asked only its template's claim criteria, which the pin grades on the
        built-in prompts: then an attribution naming no dimension is the record of a judge that scores none.
    :return: The run's judge pin.
    :raises ValidationFailedError: The run names no judge, recorded no attribution, no config set or no
        request settings, or (``"today"``) recorded settings, or a judge temperature, other than the ones
        a call sends now.
    """
    if run.judge_model is None:
        raise ValidationFailedError(
            f"run '{run.id}' was not judged, so there is no judgement to reproduce",
            remedy="name a judged run, or launch the template with a judge (run_launch with judge_model)",
        )
    if not run.effective_judges and not (claims_only and run.effective_judges is not None):
        raise ValidationFailedError(
            f"run '{run.id}' recorded no judge attribution, so which judge scored which dim cannot be reproduced",
            remedy=RELAUNCH_TO_REJUDGE,
        )
    if run.judge_config_ids is None:
        raise ValidationFailedError(
            f"run '{run.id}' recorded no judge config set, so which prompt scored each dim is unknown",
            remedy=RELAUNCH_TO_REJUDGE,
        )
    if run.judge_asked_settings is None:
        raise ValidationFailedError(
            f"run '{run.id}' recorded no judge request settings, so how the judge was asked is unknown",
            remedy=RELAUNCH_TO_REJUDGE,
        )
    if request_settings == "today" and run.judge_asked_settings != JUDGE_REQUEST_SETTINGS:
        raise ValidationFailedError(
            f"run '{run.id}' {request_settings_drift(run.judge_asked_settings)}; a new call would be asked "
            "differently from the rest of the run",
            remedy=RELAUNCH_TO_REJUDGE,
        )
    # The judge service asks every dim with no config for DEFAULT_JUDGE_TEMPERATURE. A run that recorded
    # another would have its new scores sampled by a different judge from the rest of it (#633).
    if request_settings == "today" and run.judge_temperature != DEFAULT_JUDGE_TEMPERATURE:
        raise ValidationFailedError(
            f"run '{run.id}' was judged at temperature {run.judge_temperature} and a judge call now asks for "
            f"{DEFAULT_JUDGE_TEMPERATURE}; a new score would be sampled differently from the rest of the run",
            remedy=RELAUNCH_TO_REJUDGE,
        )
    return run.judge_model


def judged_template(storage: JudgeInputStore, run: EvalRun, scope_id: str) -> EvalTemplate:
    """The template whose intent and rubric a run's judge reads, as the run was judged against it.

    :param storage: The eval store.
    :param run: The run.
    :param scope_id: The partition it lives in.
    :return: The template.
    :raises ValidationFailedError: The run names no template; the template was edited after the run was
        created, so its intent is not the one the run was judged against; or its rubric names one
        dim twice.
    :raises NotFoundError: The template does not load.
    """
    if run.template_id is None:
        raise ValidationFailedError(
            f"run '{run.id}' is ad hoc, so there is no template to read its intent from",
            remedy="name a run launched from a template",
        )
    template = storage.load_template(run.template_id, scope_id)
    if template is None:
        raise NotFoundError("template", run.template_id, remedy=FIND_TEMPLATE)
    # Templates are edited in place, and the judge reads the template's intent (and a config-less
    # dim's own text): an edit after launch would change the question.
    if template.updated_at > run.created_at:
        raise ValidationFailedError(
            f"template '{template.id}' was edited at {template.updated_at}, after run '{run.id}' launched at "
            f"{run.created_at}, so the judge would read an intent the run was not judged against",
            remedy=RELAUNCH_TO_REJUDGE,
        )
    # Scores, configs and errors are keyed by dim NAME everywhere a result stores them, so a
    # rubric naming one dim twice cannot say which of the two a stored score belongs to.
    names = [dim.name for dim in template.rubric]
    if duplicated := sorted({name for name in names if names.count(name) > 1}):
        raise ValidationFailedError(
            f"template '{template.id}' names rubric dim(s) {', '.join(duplicated)} more than once, so which "
            "of them a stored score belongs to cannot be told apart",
            remedy="name each rubric dim once in the template, and launch it again",
        )
    return template


def recorded_judged_dims(run: EvalRun, template: EvalTemplate, judged_artifact: JudgedArtifact) -> list[str]:
    """Every dim a cell of this kind is judged on under the run's recorded attribution, in judge-phase order.

    :param run: The run, whose ``effective_judges`` a caller has checked is recorded.
    :param template: Its template.
    :param judged_artifact: The kind's declaration, which picks the axes.
    :return: The dims.
    :raises ValidationFailedError: The run recorded a judge for a dim a cell of this kind is not scored on.
    """
    # The dims a cell of this kind is scored on, in judge-phase order — the rule the launch
    # attributed judges by, so a run that recorded a judge for a dim outside it is refused.
    order = scored_dim_ids([dim.name for dim in template.rubric], judged_artifact)
    recorded = run.effective_judges or {}
    if strays := sorted(set(recorded) - set(order)):
        raise ValidationFailedError(
            f"run '{run.id}' judged {', '.join(strays)}, which a {judged_artifact.value} cell of its "
            "template is not scored on",
            remedy=RELAUNCH_TO_REJUDGE,
        )
    return [dim for dim in order if dim in recorded]


def recorded_judge_configs(
    storage: JudgeInputStore, run: EvalRun, dims: Collection[str], scope_id: str
) -> dict[str, JudgeConfig]:
    """The versioned judge config the run recorded for each of ``dims`` that has one, loaded.

    :param storage: The eval store.
    :param run: The run, whose ``judge_config_ids`` a caller has checked is recorded.
    :param dims: The dims whose configs to load.
    :param scope_id: The partition they live in.
    :return: ``{dim_id: config}`` over the dims the run recorded a config for.
    :raises NotFoundError: A recorded config does not load.
    """
    configs: dict[str, JudgeConfig] = {}
    for dim in dims:
        if (recorded_id := (run.judge_config_ids or {}).get(dim)) is None:
            continue
        config = storage.load_judge_config(recorded_id, scope_id)
        if config is None:
            raise NotFoundError("judge_config", recorded_id, remedy=FIND_JUDGE_CONFIG)
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

    :param result: The result whose judging is being inspected.
    :param scored_dims: Every dim the run scores, as its launch recorded them.
    :return: Each dim in ``scored_dims`` with no stored score and no recorded can't-tell.
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
    when none did; ``judge_cut_short`` drops every re-asked dim's old entry and records the ones cut off
    again. A :class:`~threetears.evals.schema.models.JudgeRescore` recording all of it is appended.

    ``cost_usd`` and ``usage`` are left as they are — see ``JudgeRescore`` for why.

    :param result: The stored result. Not mutated.
    :param template: The run's template, whose rubric order the scores are kept in.
    :param outcomes: ``(dim_id, outcome)`` for every dim re-asked, in dimension order.
    :param judge_model: The run's judge pin, recorded on the rescore.
    :return: The result as it now stands.
    :raises ValueError: ``outcomes`` is empty, or ``result`` carries no ``judge_error`` to record
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
    asked = {dim_id for dim_id, _ in outcomes}
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
            # Every dim re-asked leaves its old cut-off behind, whatever became of it now; one cut off again is
            # recorded afresh, with what THIS call was sent.
            "judge_cut_short": {
                **{dim: cut for dim, cut in result.judge_cut_short.items() if dim not in asked},
                **folded.cut_short,
            },
            # Stamped from the template's definitions, as the phase stamps them, so a guardrail's can't-tell
            # never drops the trial from the capability measures.
            "judge_cannot_tell_boundary": sorted(
                set(result.judge_cannot_tell_boundary)
                | {dim for dim in folded.cannot_tell if dim in boundary_dim_names(template.rubric)}
            ),
            "judge_rescores": [*result.judge_rescores, rescore],
            # Every dim the cost cap kept from being asked has been asked now; any it left unscored is this call's.
            "judging_set_aside": False,
        }
    )


if TYPE_CHECKING:

    def _eval_storage_satisfies_the_port(storage: EvalStorage) -> None:
        """Hold the engine's own store to this consumer's port, so a drifted signature fails typecheck."""
        store: JudgeInputStore = storage
        del store


__all__ = [
    "JudgeInputStore",
    "ReproducibleClaimInputs",
    "ReproducibleJudgeInputs",
    "RequestSettingsPolicy",
    "apply_claim_rejudge",
    "apply_rejudge",
    "failed_judge_dims",
    "judged_template",
    "recorded_judge_configs",
    "recorded_judge_pins",
    "recorded_judged_dims",
    "reproducible_claim_inputs",
    "reproducible_judge_inputs",
    "request_settings_drift",
]
