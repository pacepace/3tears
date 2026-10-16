"""Freezing judge cases: the step every judge campaign starts from (#628).

A judge campaign measures a judge configuration as its subject, and its cases are never generated: each is one
stored output and the criterion it was judged on, frozen with the person labels given on it, into a test case of a
``judge`` template (:func:`~threetears.evals.run.freeze_judge_cases`). :func:`judge_cases_freeze` is how a surface
takes that step, answering with the receipt every surface renders
(:class:`~threetears.evals.run.JudgeCaseFreezeReport`): the cases, the case set listing them, and every result it
could not freeze, with why. :func:`judge_case_author` writes a case by hand instead (:class:`JudgeCaseAuthor`):
a hard negative no run stored, its criterion named by the subject template that words it and its expected verdict
its label. :func:`judge_template_create` writes the judge template they are frozen into — the judge kind and an
intent, which is all a judge template is.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from threetears.evals.kernel.errors import ConflictError, NotFoundError, ValidationFailedError
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.judge_cases import JUDGE_KIND, AuthoredVerdict, judge_case_of
from threetears.evals.kernel.remedies import FIND_TEMPLATE
from threetears.evals.ops.host import OpsHost
from threetears.evals.ops.runs import LaunchArguments, TemplateLine
from threetears.evals.run.authoring import get_template
from threetears.evals.run.judge_kind import JudgeCaseFreezeReport, author_judge_case, freeze_judge_cases
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.models import EvalTemplate, JudgedArtifact, RubricScale


class JudgeCasesFreeze(EvalBaseModel):
    """What freezing judge cases names: the judge template, the judged runs (or the scope's labels), and optionally
    which dims and results.

    The one declaration of a freeze's arguments; the ``judge_cases_freeze`` action's parameters derive from it.
    """

    template_id: Annotated[str, LaunchArguments.model_fields["template_id"]]
    judged_run_ids: list[str] = Field(
        default_factory=list,
        description=(
            "The judged runs whose stored results to freeze, as runs_list names them; omitted with from_labels, and "
            "required without it."
        ),
    )
    from_labels: bool = Field(
        default=False,
        description=(
            "Freeze from the scope's person labels instead of from runs: one case per output and criterion, from the "
            "content each label carries, reading no run or result, so it works after the runs are gone. A label "
            "that carries no content is skipped by name."
        ),
    )
    criteria: list[str] | None = Field(
        default=None, description="The criteria to freeze; omitted freezes every dim each result was judged on."
    )
    judged_result_ids: list[str] | None = Field(
        default=None, description="The results to freeze; omitted freezes every result of the runs."
    )
    into_case_set: str | None = Field(
        default=None,
        min_length=1,
        description="A case set to mint the next version of, listing exactly the frozen cases — what a launch targets.",
    )

    @model_validator(mode="after")
    def _runs_or_labels(self) -> Self:
        """Refuse a freeze naming both runs and labels as its source, or neither."""
        if self.from_labels and (self.judged_run_ids or self.judged_result_ids is not None):
            raise ValueError("a freeze from labels names no judged_run_ids or judged_result_ids")
        if not self.from_labels and not self.judged_run_ids:
            raise ValueError("a freeze names at least one judged run in judged_run_ids, or freezes from_labels")
        return self


def judge_cases_freeze(host: EvalHost, freeze: JudgeCasesFreeze, scope_id: str) -> JudgeCaseFreezeReport:
    """Freeze stored judged outputs, their criteria and their labels into cases of a judge template.

    The template is loaded as every use of one is (:func:`~threetears.evals.run.get_template`), in the caller's
    scope, so a template or run of another scope does not resolve.

    Args:
        host: The host whose store holds the runs and receives the cases.
        freeze: What to freeze.
        scope_id: The scope the template and the runs live in, and the cases are stored in.

    Returns:
        The receipt.

    Raises:
        NotFoundError: The template or a run is not in the scope.
        ValidationFailedError: The template is not of the judge kind, or nothing could be frozen (see
            :func:`~threetears.evals.run.freeze_judge_cases`).
        StorageError: A case or the set could not be persisted.
    """
    return freeze_judge_cases(
        host.storage,
        template=get_template(host, freeze.template_id, scope_id),
        run_ids=freeze.judged_run_ids,
        scope_id=scope_id,
        dims=freeze.criteria,
        result_ids=freeze.judged_result_ids,
        case_set=freeze.into_case_set,
        from_labels=freeze.from_labels,
    )


class JudgeCaseAuthor(EvalBaseModel):
    """What authoring one judge case by hand names: the judge template, the output, its criterion and its verdict.

    The one declaration of the ``judge_case_author`` action's parameters. The criterion is named by the subject
    template whose rubric words it, never restated: its wording is what a judge profile is keyed by, so a case on a
    reworded copy would calibrate a criterion no campaign asks. The author is whoever authors it (the caller).
    """

    template_id: Annotated[str, LaunchArguments.model_fields["template_id"]]
    criterion_template_id: str = Field(
        min_length=1,
        description="The subject template whose rubric dimension the case is judged on, as templates_list names it.",
    )
    criterion: str = Field(min_length=1, description="The rubric dimension's name on that template.")
    output: str = Field(min_length=1, description="The output under review, as the judge reads it.")
    case_material: str = Field(min_length=1, description="What the output is judged against: its source, or its facts.")
    expected: AuthoredVerdict = Field(
        description=(
            "The verdict the output must get: 'pass' or 'fail' (or 1 / 0) on a pass/fail dimension, a score from 1 to "
            "5 on an ordinal one. Refused, naming both, when the dimension's scale cannot hold it."
        )
    )
    note: str = Field(min_length=1, description="Why the output must get that verdict, in the author's words.")
    scenario: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "The scenario the judge reads as what the case asks; omitted reads the criterion template's intent, as "
            "its runs' judge does."
        ),
    )
    subject: str | None = Field(
        default=None, description="Who or what produced the output, as the judge should know it; omitted for none."
    )
    judged_artifact: Literal["document", "transcript"] = Field(
        default="document",
        description="'document' (judged against its material) or 'transcript' (a conversation's).",
    )


def judge_case_author(host: EvalHost, case: JudgeCaseAuthor, scope_id: str, *, author: str) -> JudgeCaseFreezeReport:
    """Write one judge case by hand into a judge template: an output, its criterion, and the verdict it must get.

    The criterion is read from the template that words it, and the case is frozen as every authored case is
    (:func:`~threetears.evals.run.freeze_judge_cases`, ``authored=``): its verdict its one label, by ``author``.
    Authoring the same case again names the stored one. It mints no case set: a launch with none runs every live
    case of the template, and ``case_set_mint`` lists the ones a launch should run.

    Args:
        host: The host whose store holds both templates and receives the case.
        case: What to author.
        scope_id: The scope both templates live in, and the case is stored in.
        author: Who is authoring it: the label's rater.

    Returns:
        The receipt, listing the one case.

    Raises:
        NotFoundError: Either template is not in the scope.
        ValidationFailedError: The judge template is not of the judge kind, the criterion template has no such rubric
            dimension, or the dimension's scale cannot hold the verdict (naming both).
        StorageError: The case could not be persisted.
    """
    authored = author_judge_case(
        get_template(host, case.criterion_template_id, scope_id),
        case.criterion,
        output=case.output,
        case_material=case.case_material,
        expected=case.expected,
        note=case.note,
        author=author,
        scenario=case.scenario,
        subject=case.subject,
        judged_artifact=JudgedArtifact(case.judged_artifact),
    )
    return freeze_judge_cases(
        host.storage, template=get_template(host, case.template_id, scope_id), scope_id=scope_id, authored=[authored]
    )


class JudgeCaseEntry(EvalBaseModel):
    """One stored case of a judge template, as a listing shows it."""

    test_case_id: str = Field(description="The case's id: what a case set lists, and what case_set_mint takes.")
    criterion: str = Field(description="The criterion it is judged on: a rubric dim's name, or a reserved axis id.")
    scale: RubricScale = Field(description="The criterion's scale: 'pass_fail' or 'ordinal'.")
    authored_by: str | None = Field(
        description="Who wrote the case by hand; None for a case frozen from a stored result or a label."
    )
    expected: int | None = Field(
        description=(
            "An authored case's expected verdict as a stored score: 1 (pass) or 0 (fail), or 1 to 5. None for a case "
            "frozen from a stored result, whose verdicts are its people's labels (see labels)."
        )
    )
    labels: int = Field(description="How many person labels the case carries.")
    live: bool = Field(description="Whether a launch with no case set runs it: not retired.")


class UnreadableJudgeCase(EvalBaseModel):
    """A stored case of a judge template this build cannot read as a judge case, and why."""

    test_case_id: str
    reason: str


class JudgeCaseListing(EvalBaseModel):
    """A judge template's stored cases, in storage order: every id a case set over all of them must list."""

    template_id: str
    template_name: str
    cases: list[JudgeCaseEntry]
    unreadable: list[UnreadableJudgeCase] = Field(
        description="Cases that carry no readable judge case. A launch of the template refuses until they are re-frozen."
    )


def judge_cases_list(host: EvalHost, template_id: str, scope_id: str) -> JudgeCaseListing:
    """List a judge template's stored cases, in storage order, retired ones included.

    Reads only; spends and writes nothing. A case's name is stored with it (``EvalTestCase.name``, and a seeded
    case's corpus name as ``seed_name``) but is not part of this listing; ``cases_rename`` changes it.

    Args:
        host: The host whose store is read.
        template_id: The judge template.
        scope_id: The scope it lives in.

    Returns:
        The listing.

    Raises:
        NotFoundError: No template with that id in the scope.
        ValidationFailedError: The template is not of the judge kind.
    """
    template = host.storage.load_template(template_id, scope_id)
    if template is None:
        raise NotFoundError("template", template_id, remedy=FIND_TEMPLATE)
    if template.candidate_kind != JUDGE_KIND:
        raise ValidationFailedError(
            f"template {template.id!r} is of kind {template.candidate_kind!r}, not {JUDGE_KIND!r}, so it holds no "
            "judge cases",
            remedy=f"name a template of the {JUDGE_KIND!r} kind ({FIND_TEMPLATE})",
        )
    entries: list[JudgeCaseEntry] = []
    unreadable: list[UnreadableJudgeCase] = []
    for stored in host.storage.query_test_cases(scope_id, template_id=template.id):
        try:
            case = judge_case_of(stored)
        except ValueError as exc:
            unreadable.append(UnreadableJudgeCase(test_case_id=stored.id, reason=str(exc)))
            continue
        if case is None:
            unreadable.append(UnreadableJudgeCase(test_case_id=stored.id, reason="it carries no judge case"))
            continue
        authored = case.source.authored
        entries.append(
            JudgeCaseEntry(
                test_case_id=stored.id,
                criterion=case.dim,
                scale=case.scale,
                authored_by=authored.author if authored is not None else None,
                expected=case.labels[0].score if authored is not None else None,
                labels=len(case.labels),
                live=not stored.archived,
            )
        )
    return JudgeCaseListing(template_id=template.id, template_name=template.name, cases=entries, unreadable=unreadable)


class JudgeTemplateCreate(EvalBaseModel):
    """What authoring a judge template names: what it is called and what its judge campaign sets out to measure.

    Nothing else: a judge template is the judge kind and an intent, its cases carrying everything a trial reads.
    """

    template_name: str = Field(min_length=1, description="The judge template's name, unique in the scope.")
    intent: str = Field(
        min_length=1, description="What its judge campaign measures, in a sentence a reader of its report is told."
    )


def judge_template_create(host: OpsHost, create: JudgeTemplateCreate, scope_id: str) -> TemplateLine:
    """Write a template of the judge kind — what :func:`judge_cases_freeze` freezes cases into and a judge campaign launches.

    Wholly the package's: the judge kind, the name and the intent, and no field the judge kind would ignore — the
    dispatch refuses one (:attr:`~threetears.evals.run.LaunchableKind.ignored_template_fields`). So no host check
    has anything to read on it, and it is refused only where it could never run: on a host whose launch registry
    has no judge kind.

    Args:
        host: The launching host, whose registry must launch the judge kind and whose store receives the template.
        create: What the template is called and measures.
        scope_id: The scope it lives in.

    Returns:
        The template as written.

    Raises:
        ValidationFailedError: The host launches no judge kind.
        ConflictError: A template of that name is already in the scope.
        StorageError: The template could not be written.
    """
    if JUDGE_KIND not in host.launch.kinds:
        # The remedy is the host operator's, not the caller's: no action registers a kind. What the operator does is
        # said in the reason, where it names the host's own setup rather than a call the caller could make.
        raise ValidationFailedError(
            f"host {host.eval_host.profile.host_id!r} launches no {JUDGE_KIND!r} kind, so a judge template could never "
            f"run here: its LaunchHost registers no {JUDGE_KIND!r} kind (run.launchable_judge_kind) and its profile "
            "declares no JUDGE_KIND_CONTRACT",
            remedy=(
                "ask this host's operator to enable judge campaigns — a setting of the host, which no action changes; "
                "until then, judge_second and judge_drift_check measure a run's judge against a second one"
            ),
        )
    storage = host.eval_host.storage
    if storage.load_template_by_name(create.template_name, scope_id) is not None:
        raise ConflictError(
            f"a template named '{create.template_name}' already exists",
            remedy=f"name the judge template something else, or freeze into the existing one ({FIND_TEMPLATE})",
        )
    template = EvalTemplate(
        scope_id=scope_id,
        name=create.template_name,
        intent=create.intent,
        candidate_kind=JUDGE_KIND,
    )
    storage.save_template(template)
    return TemplateLine(
        id=template.id,
        name=template.name,
        revision=template.revision,
        candidate_kind=template.candidate_kind,
        intent=template.intent,
        archived=template.archived,
    )


__all__ = [
    "JudgeCaseAuthor",
    "JudgeCaseEntry",
    "JudgeCaseListing",
    "JudgeCasesFreeze",
    "JudgeTemplateCreate",
    "UnreadableJudgeCase",
    "judge_case_author",
    "judge_cases_freeze",
    "judge_cases_list",
    "judge_template_create",
]
