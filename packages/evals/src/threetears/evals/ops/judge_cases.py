"""Freezing judge cases: the step every judge campaign starts from (#628).

A judge campaign measures a judge configuration as its subject, and its cases are never generated: each is one
stored output and the criterion it was judged on, frozen with the person labels given on it, into a test case of a
``judge`` template (:func:`~threetears.evals.run.freeze_judge_cases`). :func:`judge_cases_freeze` is how a surface
takes that step, answering with the receipt every surface renders
(:class:`~threetears.evals.run.JudgeCaseFreezeReport`): the cases, the case set listing them, and every result it
could not freeze, with why.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from threetears.evals.kernel.host import EvalHost
from threetears.evals.ops.runs import LaunchArguments
from threetears.evals.run.authoring import get_template
from threetears.evals.run.judge_kind import JudgeCaseFreezeReport, freeze_judge_cases
from threetears.evals.schema.base import EvalBaseModel


class JudgeCasesFreeze(EvalBaseModel):
    """What freezing judge cases names: the judge template, the judged runs, and optionally which dims and results.

    The one declaration of a freeze's arguments; the ``judge_cases_freeze`` action's parameters derive from it.
    """

    template_id: Annotated[str, LaunchArguments.model_fields["template_id"]]
    judged_run_ids: list[str] = Field(
        min_length=1, description="The judged runs whose stored results to freeze, as runs_list names them."
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
    )


__all__ = ["JudgeCasesFreeze", "judge_cases_freeze"]
