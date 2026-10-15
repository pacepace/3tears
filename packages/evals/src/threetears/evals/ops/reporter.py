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

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol, get_args

from pydantic import Field

from threetears.evals.analysis.reporter_bank import FrozenReporterCase, frozen_case_receipt, reporter_case_bank
from threetears.evals.analysis.reporter_curation import set_reporter_case_archived
from threetears.evals.analysis.reporter_contract import (
    ReporterOverlays,
    live_reporter_cases,
    refuse_undeliverable_reporter_template,
)
from threetears.evals.analysis.reporter_kind import (
    AS_RECORDED_MODEL,
    REPORTER_KIND,
    LabelDirection,
    ReporterCase,
    ReporterKind,
    reporter_case_of,
)
from threetears.evals.kernel.inner_ceilings import MAX_GENERATION_CALLS
from threetears.evals.analysis.service import freeze_reporter_case
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.kernel.errors import NotFoundError, ValidationFailedError
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.prompts.catalogue import ANALYSIS_GEN_PROMPT
from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.run.launch import (
    ArmPlan,
    CallCeiling,
    LaunchArgument,
    LaunchRequest,
    plan_judge,
    resolve_judge_pin,
)
from threetears.evals.schema.models import EvalTemplate, EvalTestCase, JudgedArtifact
from threetears.evals.ops.runs import LaunchArguments
from threetears.evals.run.authoring import get_template
from threetears.evals.kernel.remedies import FIND_TEMPLATE


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
            "reporter cases",
            remedy=f"name a template of the {REPORTER_KIND!r} kind ({FIND_TEMPLATE})",
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
        raise NotFoundError("template", template_id, remedy=FIND_TEMPLATE)
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


# --- launching a reporter arm -----------------------------------------------------------------------------------------


#: The launch arguments a reporter arm cannot honour, for its ``LaunchableKind``: it drives no simulated user,
#: calls no tool a cassette could record, and generates no case — each is frozen from a campaign.
REPORTER_UNHONOURED_LAUNCH_ARGUMENTS: frozenset[LaunchArgument] = frozenset(
    {"simulator_model", "cassette_mode", "n_variations"}
)

#: Which model a reporter launch asks the host to resolve: the generator writing the memos, or the judge scoring them.
ReporterModelRole = Literal["generator", "judge"]

#: The host's model resolution, as a reporter launch asks it: ``(role, named)`` returns the provider model the name
#: resolves to, or the role's default when ``named`` is ``None``. It raises
#: :class:`~threetears.evals.kernel.errors.ValidationFailedError` to refuse a name it does not know. It is called off
#: the event loop, so it may read the host's registry.
ReporterModelResolver = Callable[[ReporterModelRole, str | None], str]


class ReporterPrompts(Protocol):
    """The host's prompt registry, as a reporter launch reads it: what a preset of the generator's prompt resolves to.

    The slot is :data:`~threetears.evals.kernel.ANALYSIS_GEN_PROMPT`. Read freshly for each launch, so a preset an
    operator authored since the process started is seen.
    """

    def resolve(self, preset: str | None) -> str | None:
        """The text ``preset`` names, or the registry's system default for ``None``.

        **``None`` for a preset the registry does not hold — never the default in its place.** A registry that
        answered a missing name with the default would have the run bill the incumbent prompt under the
        candidate's name; the launch refuses instead.

        Args:
            preset: The preset's name, or ``None`` for the system default.

        Returns:
            The prompt text, or ``None`` when no prompt of that name is stored.
        """
        ...


@dataclass(frozen=True, kw_only=True)
class ReporterAdmitted:
    """What :func:`admit_reporter_launch` resolved of a reporter arm it can honour.

    Attributes:
        preset: The preset the launch named, or ``None`` for the system default.
        prompt: The text the preset resolved to — what a generating arm's
            :class:`~threetears.evals.analysis.ReporterKind` writes with, under the prompt id
            :data:`~threetears.evals.kernel.ANALYSIS_GEN_PROMPT`.
        candidate_model: The generator model the arm runs on, or
            :data:`~threetears.evals.analysis.AS_RECORDED_MODEL` for the replay.
        judge_model: The run-level judge pin, resolved.
        cases: The cases the arm runs, in run order.
    """

    preset: str | None
    prompt: str
    candidate_model: str
    judge_model: str
    cases: tuple[EvalTestCase, ...]

    @property
    def generates(self) -> bool:
        """Whether the arm writes memos — ``False`` for the replay, which builds no generator client."""
        return self.candidate_model != AS_RECORDED_MODEL


def _resolved_prompt(prompts: ReporterPrompts, preset: str | None) -> str:
    """The text a launch's preset resolves to, refused when the registry holds no such prompt."""
    text = prompts.resolve(preset)
    if text is None:
        if preset is None:
            raise ValidationFailedError(
                f"the prompt registry holds no system default {ANALYSIS_GEN_PROMPT!r} prompt",
                remedy="seed it from SEED_PROMPTS before launching a reporter",
            )
        raise ValidationFailedError(
            f"prompt_preset {preset!r}: no {ANALYSIS_GEN_PROMPT!r} preset of that name. Refused rather than resolved "
            "to the default, because the run would bill the incumbent prompt under the candidate's name",
            remedy="author the preset first, or launch without prompt_preset for the system default",
        )
    if not text.strip():
        raise ValidationFailedError(
            f"the {ANALYSIS_GEN_PROMPT!r} prompt {preset or 'system default'!r} resolves to no text, so every memo "
            "would be written to no instruction",
            remedy="give the prompt its text in the prompt registry, or name another preset",
        )
    return text


def _candidate_model(request: LaunchRequest, resolve_model: ReporterModelResolver) -> str:
    """The model the arm runs on: the replay, the generator role's default, or the model the launch named.

    An arm runs on the model the launch NAMED and records that name, so a name the host resolves to another
    model is an alias: two runs recording it could have run two models as its target moved. It is refused
    here, before the arm is priced, rather than by the launch tail after.
    """
    named = request.candidate_model
    if named == AS_RECORDED_MODEL:
        return named
    resolved = resolve_model("generator", named)
    if named is not None and resolved != named:
        raise ValidationFailedError(
            f"reporter model {named!r} is an alias of {resolved!r}; the run records the name the launch gave, and an "
            "alias's target moves",
            remedy=f"launch {resolved!r} by name",
        )
    return resolved


def _cases(request: LaunchRequest, storage: EvalStorage) -> list[tuple[EvalTestCase, ReporterCase]]:
    """The arm's cases: its launch's case set when it names one, else the template's live bank."""
    if request.case_set is None:
        return live_reporter_cases(storage, request.template, request.scope_id)
    read: list[tuple[EvalTestCase, ReporterCase]] = []
    unusable: list[str] = []
    for stored in request.case_set_cases:
        try:
            case = reporter_case_of(stored)
        except ValueError as exc:
            unusable.append(f"{stored.id!r} ({exc})")
            continue
        if case is None:
            unusable.append(f"{stored.id!r} (carries no reporter case)")
            continue
        read.append((stored, case))
    if unusable:
        raise ValidationFailedError(
            f"case set {request.case_set.ref.label!r} names case(s) no reporter can run: {', '.join(unusable)}",
            remedy="mint a version of the set without them (case_set_mint), or launch without case_set",
        )
    return read


def _refuse_a_replay_with_nothing_recorded(cases: Sequence[tuple[EvalTestCase, ReporterCase]]) -> None:
    """Refuse the as-recorded replay over a case that pins no recorded memo, which it would have nothing to replay of."""
    if unrecorded := [stored.id for stored, case in cases if case.recorded_memo is None]:
        raise ValidationFailedError(
            f"model {AS_RECORDED_MODEL!r} replays each case's recorded memo, and case(s) {', '.join(unrecorded)} pin "
            "none",
            remedy="launch it over cases that each froze the memo their campaign got, or launch a generator model",
        )


def admit_reporter_launch(
    request: LaunchRequest,
    *,
    storage: EvalStorage,
    prompts: ReporterPrompts,
    resolve_model: ReporterModelResolver,
) -> ReporterAdmitted:
    """Every refusal a reporter arm meets before it is priced, and what it resolved to — for its plan and its launcher alike.

    The host supplies only what the host knows: its prompt registry and its model resolution. Everything else is
    the kind's, in this order: the template (:func:`~threetears.evals.analysis.refuse_undeliverable_reporter_template`),
    the preset, the generator model, the cases, the replay's memos, then the judge. Blocking — it reads the case
    bank and both hooks — so a launcher calls it off the event loop.

    The cases are the launch's case set when it names one, each of which must carry a reporter case this build
    reads; otherwise the template's live bank (:func:`~threetears.evals.analysis.live_reporter_cases`), refused
    when empty or ambiguous.

    Args:
        request: The arm.
        storage: Where the template's cases are stored.
        prompts: The host's prompt registry, as just loaded.
        resolve_model: The host's model resolution.

    Returns:
        What the arm resolved to.

    Raises:
        ValidationFailedError: The template asks for what a reporter cannot honour; the preset is not stored, or
            no default is; the model is an alias, or the host refuses it; there is no live case, a pair holds two,
            a case of the bank does not read, or a case-set case carries no reporter case; or the replay is
            launched over a case pinning no recorded memo.
    """
    refuse_undeliverable_reporter_template(request.template)
    preset = request.overlays_as(ReporterOverlays).prompt_preset if request.overlays is not None else None
    prompt = _resolved_prompt(prompts, preset)
    candidate_model = _candidate_model(request, resolve_model)
    cases = _cases(request, storage)
    if candidate_model == AS_RECORDED_MODEL:
        _refuse_a_replay_with_nothing_recorded(cases)
    judge_model = request.judge_model or resolve_judge_pin(
        request, resolve_model("judge", None), candidate_model=candidate_model
    )
    return ReporterAdmitted(
        preset=preset,
        prompt=prompt,
        candidate_model=candidate_model,
        judge_model=judge_model,
        cases=tuple(stored for stored, _case in cases),
    )


def plan_reporter_arm(
    host: EvalHost,
    request: LaunchRequest,
    *,
    prompts: ReporterPrompts,
    resolve_model: ReporterModelResolver,
) -> ArmPlan:
    """A reporter arm's plan, for its ``LaunchableKind.plan_arm``: every refusal first, then what it will run.

    The arm runs its admitted cases on its admitted model, judged on the memo (``JudgedArtifact.DOCUMENT``) by the
    launch's judge or the judge role's default, with no simulator. A launcher re-admits with the same hooks, so
    the plan and the run cannot be built from different reads.

    Args:
        host: The host, where the cases and judge configs are read.
        request: The arm.
        prompts: The host's prompt registry, as just loaded.
        resolve_model: The host's model resolution.

    Returns:
        The plan.

    Raises:
        ValidationFailedError: Anything :func:`admit_reporter_launch` or
            :func:`~threetears.evals.run.plan_judge` refuses.
    """
    admitted = admit_reporter_launch(request, storage=host.storage, prompts=prompts, resolve_model=resolve_model)
    return ArmPlan(
        case_count=len(admitted.cases),
        candidate_model=admitted.candidate_model,
        judge=plan_judge(
            host,
            request.template,
            admitted.judge_model,
            request.judge_config_ids,
            judged_artifact=JudgedArtifact.DOCUMENT,
        ),
        simulator_model=None,
    )


async def price_largest_reporter_call(
    host: EvalHost,
    request: LaunchRequest,
    *,
    prompts: ReporterPrompts,
    resolve_model: ReporterModelResolver,
) -> CallCeiling | None:
    """A reporter arm's largest generator call, priced before launch, for its ``LaunchableKind.price_largest_call``.

    A generator call's input is known whole at plan time — each case's frozen bundle, the preset's text and the
    admitted model — so the arm is priced here rather than refused only when its run reaches the call, after the
    cases before it were paid for. Each case's first generator request is built as its cell builds it
    (:meth:`~threetears.evals.analysis.ReporterKind.price_case`, over the one reading of a case
    :meth:`~threetears.evals.analysis.ReporterKind.invoke` generates over) and priced on a client the host builds
    in the ``analysis`` role for the admitted model: the same factory call a launcher builds the run's generator
    with, so pass the same ``host`` your launcher does (one whose clients are pinned to the run's policy, if yours
    are). The largest is kept.

    **What stays unpriced.** The memo's judge call, whose prompt carries a memo that does not exist before launch,
    and the generation's repair round-trip, whose message carries the first call's output. Each is still held to
    the run's cap when it is reserved. The as-recorded replay calls no generator, so it prices nothing.

    Admits the arm again (:func:`admit_reporter_launch`, off the event loop) with the same hooks the plan is made
    with, so the plan and the price cannot be built from different reads.

    Args:
        host: The host the launcher builds the generator client on.
        request: The arm, as planned.
        prompts: The host's prompt registry, as just loaded.
        resolve_model: The host's model resolution.

    Returns:
        The largest generator call's ceiling, naming its case and what lowers it; ``None`` for the replay, or when
        no case's call could be priced (the client has no ``price_ceiling``, or answers ``None``).

    Raises:
        ValidationFailedError: Anything :func:`admit_reporter_launch` refuses.
        ValueError: The host supplies no completion clients, or its client priced a call at an amount no
            reservation could hold.
    """
    admitted = await run_blocking(
        host.blocking_executor,
        functools.partial(admit_reporter_launch, storage=host.storage, prompts=prompts, resolve_model=resolve_model),
        request,
    )
    if not admitted.generates:
        return None
    client = host.completion_clients("pricing a reporter arm")("analysis", admitted.candidate_model)
    try:
        kind = ReporterKind(
            prompt=admitted.prompt,
            prompt_id=ANALYSIS_GEN_PROMPT,
            prompt_version=None,
            client=client,
            model=admitted.candidate_model,
            host=host,
        )
        priced = await run_blocking(
            host.blocking_executor,
            lambda: [(ceiling, case.id) for case in admitted.cases if (ceiling := kind.price_case(case)) is not None],
        )
    finally:
        await client.aclose()
    if not priced:
        return None
    usd, case_id = max(priced)
    return CallCeiling(
        usd=usd,
        call=(
            f"the generator call on case {case_id!r} (the first request of up to {MAX_GENERATION_CALLS}: the memo, "
            f"then one repair if its output is refused) on {admitted.candidate_model!r}"
        ),
        lowered_by=(
            "a cheaper generator model; a smaller frozen bundle on the largest case (re-freeze it, or launch a case "
            "set without it); or, on the host, a generator-role client that re-sends a request fewer times (its "
            "retry policy multiplies every request's ceiling) or whose price_ceiling prices a smaller output cap"
        ),
    )


__all__ = [
    "REPORTER_UNHONOURED_LAUNCH_ARGUMENTS",
    "AmbiguousReporterPair",
    "ReporterAdmitted",
    "ReporterModelResolver",
    "ReporterModelRole",
    "ReporterPrompts",
    "ReporterCaseEntry",
    "ReporterCaseFreeze",
    "ReporterCaseListing",
    "UnreadableReporterCase",
    "admit_reporter_launch",
    "plan_reporter_arm",
    "price_largest_reporter_call",
    "reporter_case_archive",
    "reporter_case_freeze",
    "reporter_cases_list",
]
