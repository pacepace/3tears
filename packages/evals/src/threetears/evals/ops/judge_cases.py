"""Freezing judge cases: the step every judge campaign starts from (#628).

A judge campaign measures a judge configuration as its subject, and its cases are never generated: each is one
stored output and the criterion it was judged on, frozen with the person labels given on it, into a test case of a
``judge`` template (:func:`~threetears.evals.run.freeze_judge_cases`). :func:`judge_cases_freeze` is how a surface
takes that step, answering with the receipt every surface renders
(:class:`~threetears.evals.run.JudgeCaseFreezeReport`): the cases, the case set listing them, and every result it
could not freeze, with why. :func:`judge_template_create` writes the judge template they are frozen into — the
judge kind and an intent, which is all a judge template is.
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, model_validator

from threetears.evals.kernel.errors import ConflictError, ValidationFailedError
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.judge_cases import JUDGE_KIND
from threetears.evals.kernel.remedies import FIND_TEMPLATE
from threetears.evals.ops.host import OpsHost
from threetears.evals.ops.runs import LaunchArguments, TemplateLine
from threetears.evals.run.authoring import get_template
from threetears.evals.run.judge_kind import JudgeCaseFreezeReport, freeze_judge_cases
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.models import EvalTemplate


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
        candidate_kind=template.candidate_kind,
        intent=template.intent,
        archived=template.archived,
    )


__all__ = ["JudgeCasesFreeze", "JudgeTemplateCreate", "judge_cases_freeze", "judge_template_create"]
