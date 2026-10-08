"""The reporter case bank's operations: freeze a campaign into a case, list a template's cases, retire one.

A reporter run measures the analysis writer itself, and its cases are never generated: each is one
campaign's analysis bundle — and, optionally, the memo the campaign actually got — frozen into a test
case of an ``analysis_reporter`` template (:func:`~threetears.evals.analysis.freeze_reporter_case`). So
freezing is the step every reporter run starts from, and these operations are how a surface takes it:

- :func:`reporter_case_freeze` freezes one, answering with the receipt every surface renders
  (:class:`~threetears.evals.analysis.FrozenReporterCase`): the case, its bundle fingerprint and the
  ``limits`` the freeze recorded.
- :func:`reporter_cases_list` reads a template's bank — which case each (campaign, recorded memo) pair
  launches, which were superseded or retired, which this build cannot read and which pairs hold more
  than one live case.
- :func:`reporter_case_archive` retires a case, or restores one
  (:func:`~threetears.evals.analysis.set_reporter_case_archived`).

**The listing reads the bank without refusing on a case it cannot read.** The freeze, the launch and a
restore decide from liveness, and refuse while an unreadable case could change the answer; a listing
decides nothing, and it is how an operator finds that case, so it lists it rather than stopping.
"""

from __future__ import annotations

from typing import Annotated, Any, get_args

from pydantic import Field

from threetears.evals.analysis.reporter_bank import FrozenReporterCase, frozen_case_receipt, reporter_case_bank
from threetears.evals.analysis.reporter_curation import set_reporter_case_archived
from threetears.evals.analysis.reporter_kind import REPORTER_KIND, LabelDirection
from threetears.evals.analysis.service import freeze_reporter_case
from threetears.evals.contracts.base import EvalBaseModel
from threetears.evals.contracts.errors import NotFoundError, ValidationFailedError
from threetears.evals.contracts.host import EvalHost
from threetears.evals.contracts.models import EvalTemplate
from threetears.evals.ops.runs import LaunchArguments
from threetears.evals.run.authoring import get_template


class ReporterCaseFreeze(EvalBaseModel):
    """What freezing a reporter case names: the reporter template, the campaign, and optionally its memo and labels.

    The one declaration of a freeze's arguments. The ``reporter_case_freeze`` action's parameters derive from
    it, so the operation and what an agent is offered cannot drift apart; ``template_id`` and ``campaign_id``
    mean on it what they mean on every other action, which mounting the catalogue holds.
    """

    template_id: Annotated[str, LaunchArguments.model_fields["template_id"]]
    campaign_id: str = Field(min_length=1, description="A campaign's id, as campaigns_list names it.")
    recorded_analysis_id: str | None = Field(
        default=None,
        min_length=1,
        description="A stored analysis of the campaign whose memo the case pins as the one it got, as analyses_list "
        "names it; omitted freezes the bundle alone, which only a generating candidate can run.",
    )
    labels: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Reader verdicts on the recorded memo, each {dimension, direction, quote, note?}: a dimension the "
        f"template scores, a direction of {'|'.join(get_args(LabelDirection))}, and the reader's words verbatim. They "
        "need recorded_analysis_id; the freeze stamps each with the template's criterion for its dimension.",
    )
    supersedes: list[str] = Field(
        default_factory=list,
        description="The live case(s) of this campaign and memo the freeze replaces, by id — needed to revise a "
        "case's labels or re-freeze moved evidence; every live case of the pair when it holds several.",
    )


class ReporterCaseEntry(EvalBaseModel):
    """One readable case of a reporter template, with whether a launch runs it."""

    case: FrozenReporterCase
    live: bool = Field(description="Whether a launch runs it: nothing supersedes it and no operator retired it.")
    superseded_by: list[str] = Field(description="The cases that replaced it, by id; empty when none did.")


class UnreadableReporterCase(EvalBaseModel):
    """A stored case carrying a reporter case this build cannot read."""

    test_case_id: str
    reason: str


class AmbiguousReporterPair(EvalBaseModel):
    """A (campaign, recorded memo) pair holding more than one live case — which every launch of the template refuses."""

    campaign_id: str | None
    recorded_analysis_id: str | None
    live_case_ids: list[str]


class ReporterCaseListing(EvalBaseModel):
    """A reporter template's cases, in storage order."""

    template_id: str
    include_archived: bool = Field(description="Whether retired cases were listed; they are left out by default.")
    cases: list[ReporterCaseEntry]
    unreadable: list[UnreadableReporterCase] = Field(
        description="Cases this build cannot read. While any is listed, which case is live cannot be decided: one of "
        "them may supersede a case listed as live."
    )
    ambiguous: list[AmbiguousReporterPair] = Field(
        description="Pairs with more than one live case. A launch of the template refuses until one freeze naming "
        "every one of them in supersedes replaces them."
    )


def _reporter_template(template: EvalTemplate) -> EvalTemplate:
    """The template, refused unless it is of the reporter kind — the only kind whose bank these operations read."""
    if template.candidate_kind != REPORTER_KIND:
        raise ValidationFailedError(
            f"template {template.id!r} is of kind {template.candidate_kind!r}, not {REPORTER_KIND!r}, so it holds no "
            "reporter cases"
        )
    return template


def reporter_case_freeze(host: EvalHost, freeze: ReporterCaseFreeze, scope_id: str) -> FrozenReporterCase:
    """Freeze a campaign's analysis bundle, and optionally the memo it got, into a case of a reporter template.

    :func:`~threetears.evals.analysis.freeze_reporter_case` does the freezing, with every rule it holds: a
    case is a (campaign, recorded memo) pair, a freeze matching the pair's live case returns it, and one
    that differs is refused unless ``supersedes`` names the live case. The template is loaded as every use
    of one is (:func:`~threetears.evals.run.get_template`), in the caller's scope, so a template, campaign
    or analysis of another scope does not resolve.

    Args:
        host: The host whose store holds the campaign and receives the case.
        freeze: What to freeze.
        scope_id: The scope the template, the campaign and its runs live in, and the case is stored in.

    Returns:
        The receipt: the case — the pair's existing live case when the freeze matched it — its fingerprint
        and the limits it recorded.

    Raises:
        NotFoundError: The template, campaign or analysis is not in the scope.
        ValidationFailedError: The template is not of the reporter kind, or the freeze refused (see
            :func:`~threetears.evals.analysis.freeze_reporter_case`).
        StorageError: The case could not be persisted.
    """
    stored = freeze_reporter_case(
        host,
        template_id=freeze.template_id,
        campaign_id=freeze.campaign_id,
        scope_id=scope_id,
        analysis_id=freeze.recorded_analysis_id,
        labels=freeze.labels,
        supersedes=freeze.supersedes,
        load_template=lambda template_id: get_template(host, template_id, scope_id),
    )
    return frozen_case_receipt(stored)


def reporter_cases_list(
    host: EvalHost, template_id: str, scope_id: str, *, include_archived: bool = False
) -> ReporterCaseListing:
    """A reporter template's cases: each with whether a launch runs it, then what cannot be read or decided.

    Args:
        host: The host whose store is read.
        template_id: The reporter template.
        scope_id: The scope it lives in.
        include_archived: List retired cases too.

    Returns:
        The listing.

    Raises:
        NotFoundError: No template with that id in the scope.
        ValidationFailedError: The template is not of the reporter kind.
    """
    template = host.storage.load_template(template_id, scope_id)
    if template is None:
        raise NotFoundError("template", template_id)
    _reporter_template(template)
    bank = reporter_case_bank(host.storage.query_test_cases(scope_id, template_id=template_id))
    return ReporterCaseListing(
        template_id=template_id,
        include_archived=include_archived,
        cases=[
            ReporterCaseEntry(
                case=frozen_case_receipt(stored),
                live=bank.is_live(stored.id),
                superseded_by=list(bank.superseded_by.get(stored.id, ())),
            )
            for stored, _ in bank.cases
            if include_archived or not stored.archived
        ],
        unreadable=[UnreadableReporterCase(test_case_id=case_id, reason=why) for case_id, why in bank.unreadable],
        ambiguous=[
            AmbiguousReporterPair(
                campaign_id=pair.campaign_id,
                recorded_analysis_id=pair.recorded_analysis_id,
                live_case_ids=list(pair.live_case_ids),
            )
            for pair in bank.ambiguous
        ],
    )


def reporter_case_archive(
    host: EvalHost, test_case_id: str, scope_id: str, *, archived: bool, reason: str | None = None
) -> FrozenReporterCase:
    """Retire a reporter case, or restore one: a retired case is never launched again and stays readable.

    Args:
        host: The host whose store holds the case.
        test_case_id: The case.
        scope_id: The scope it lives in.
        archived: ``True`` retires it, ``False`` restores it.
        reason: Why it is retired; cleared on restore.

    Returns:
        The case's receipt, as persisted.

    Raises:
        NotFoundError: No case with that id in the scope.
        ValidationFailedError: It carries no reporter case, or restoring it would make a second live case of
            its pair (see :func:`~threetears.evals.analysis.set_reporter_case_archived`).
        StorageError: The write failed.
    """
    return frozen_case_receipt(
        set_reporter_case_archived(host.storage, test_case_id, scope_id, archived=archived, reason=reason)
    )


__all__ = [
    "AmbiguousReporterPair",
    "ReporterCaseEntry",
    "ReporterCaseFreeze",
    "ReporterCaseListing",
    "UnreadableReporterCase",
    "reporter_case_archive",
    "reporter_case_freeze",
    "reporter_cases_list",
]
