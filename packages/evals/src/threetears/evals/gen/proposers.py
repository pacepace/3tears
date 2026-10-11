"""The rubric proposer: one completion call that drafts a template for operator review.

One proposer, run on one of two axes. On the ``capability`` axis it drafts a capability rubric
and its scenario axes; on the ``boundary`` axis it drafts a universal boundary battery and its
refusal dims. Either way it sends a system prompt and a two-feed user message — the subject
feed (what the subject under test is) and the catalog feed (the reusable rubric dims the
operator already trusts) — and validates the reply into a
:class:`~threetears.evals.schema.models.RubricProposal`. No draft is persisted: the operator
edits it and commits the parts they accept through the authoring operations. What IS written is
the call's spend: the call runs outside any run, so it is priced before it is made and ledgered
once it is, through the :class:`~threetears.evals.kernel.out_of_run.OutOfRunBudget` the host hands in.

**Planned, then admitted by the host, then made.** :func:`plan_draft` is the call as it will be sent, made
of nothing but the three strings; the host quotes it and admits it under its budget in its own call
(:meth:`~threetears.evals.kernel.out_of_run.OutOfRunBudget.admit`), refusing there what the cap cannot pay
for, and :func:`propose_draft` — in the job — makes exactly that admitted call and reads the draft.

**What arrives here is text.** The host renders the subject feed from its own subject, the
catalog feed from its rubric-dim store, and the system prompt from its prompt registry — each
for the axis it is proposing on — then hands this module the three strings and a
:class:`~threetears.evals.schema.completion.BoundCompletionClient`. So the proposer knows neither what
a subject is nor where a prompt is kept, and every host drafts with the same code whatever its
subject is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from pydantic import ValidationError

from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.schema.models import RubricAxis, RubricProposal
from threetears.evals.schema.out_of_run_spend import OutOfRunSpend
from threetears.evals.kernel.out_of_run import AdmittedCall, PlannedCall
from threetears.evals.schema.completion import JSON_OBJECT_RESPONSE_FORMAT
from threetears.evals.kernel.provider import extract_json
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.evals.kernel.out_of_run import OutOfRunBudget
    from threetears.evals.schema.completion import BoundCompletionClient

log = get_logger(__name__)

#: How a refusal names the proposer that produced the output it refuses, per axis.
_REFUSAL_SUBJECT: dict[str, str] = {
    "capability": "proposer",
    "boundary": "boundary proposer",
}

#: Output-token ceiling for the capability/boundary proposer. It drafts a full
#: rubric (template + several dims + variation axes + new-dim suggestions) in one
#: shot — far larger than the judge's per-dim score — so a 4096 default truncates
#: it mid-JSON (``finish_reason=length`` → unparseable object). This generous
#: ceiling lets the draft complete; cost scales with actual output, not the
#: ceiling. The host applies it when it builds the client it hands the proposer.
#: The judge's cap is its own, derived in ``threetears.evals.run.judge`` from its
#: reasoning budget plus its answer budget.
PROPOSER_MAX_TOKENS = 16384


class ProposedDraft(NamedTuple):
    """A rubric draft, and the ledger row of the call that wrote it."""

    proposal: RubricProposal
    #: What the drafting call cost as the provider reported it, the ceiling it was admitted at and the
    #: cap it was admitted under — written to the budget's store before the reply was read.
    spend: OutOfRunSpend


def plan_draft(*, system_prompt: str, subject_feed: str, catalog_feed: str) -> PlannedCall:
    """The one call a rubric proposal sends, on either axis, built before anything is priced or made.

    What a host quotes and admits under its out-of-run budget in its call
    (:meth:`~threetears.evals.kernel.out_of_run.OutOfRunBudget.quote` / ``admit`` with purpose ``proposer``),
    then hands, admitted, to :func:`propose_draft`. The user message is the subject feed, then the catalog
    feed, because the catalog is read against the subject: which dims to reuse depends on what the subject is.

    Args:
        system_prompt: The rendered system prompt for the axis being proposed on.
        subject_feed: The rendered description of the subject under test.
        catalog_feed: The rendered reusable catalog for the axis, each dim with its ``key`` so the model can
            name reuses in ``reused_dim_keys``.

    Returns:
        The planned call.
    """
    return PlannedCall(
        system=system_prompt,
        user=f"{subject_feed}\n{catalog_feed}".rstrip() + "\n",
        response_format=JSON_OBJECT_RESPONSE_FORMAT,
    )


async def propose_draft(
    client: BoundCompletionClient,
    admitted: AdmittedCall,
    *,
    budget: OutOfRunBudget,
    axis: RubricAxis,
    subject_id: str,
) -> ProposedDraft:
    """Make an admitted rubric proposal on ``axis`` and read its draft.

    On the ``capability`` axis the draft is a capability rubric and its scenario axes. On the
    ``boundary`` axis it is a *universal* boundary template carrying an adversarial
    ``conversation`` (an out-of-domain ask + injection / jailbreak / unsafe-request /
    character-break pressure), ``variation_axes`` to instantiate the out-of-domain subject, and
    the three refusal rubric dims (``boundary.correct`` / ``boundary.in_character`` /
    ``boundary.scope_discipline`` — namespaced because a judge config binds to a dim by name
    across every template). Domain width is judged by the model, not by code: the subject feed
    carries the subject's tool breadth and reasoning focus, and the boundary system prompt
    encodes the rule that for a broad-domain generalist 'out of domain' collapses toward unsafe
    or abusive rather than merely off-topic.

    The call is the one :func:`plan_draft` planned, which the host admitted under ``budget`` in its own
    call — so a proposal the cap cannot pay for was refused there, with nothing called, and this admits
    nothing. It makes the call on ``client``, whose spend ``budget`` ledgers as soon as it returns — before
    the reply is read, so a draft refused for its content still has its cost on record — and validates the
    reply into a :class:`~threetears.evals.schema.models.RubricProposal`, **returned for operator review —
    no draft is persisted**. The client is released when the call returns, on the refusal paths too: a
    proposal is the client's whole lifetime.

    Args:
        client: The completion client the call was admitted on — the host's client for the ``proposer`` role
            (:data:`~threetears.evals.kernel.host.CompletionRole`). This function owns it from here and
            releases it.
        admitted: The proposal's call, as ``budget`` admitted it on ``client`` (purpose ``proposer``).
        budget: The out-of-run budget that admitted the call, which ledgers it. Its ``subject_id``, when it
            names one, is the subject the draft is for.
        axis: Which catalog axis the draft is for; stamped onto every new-dim suggestion.
        subject_id: The subject the draft is for, named in the log line only.

    Returns:
        The validated :class:`~threetears.evals.schema.models.RubricProposal` draft, and its call's
        ledger row.

    Raises:
        ValueError: ``budget`` names another subject than ``subject_id``; ``admitted`` was admitted for
            another purpose than ``proposer``; or ``budget`` did not admit it on ``client`` (another
            budget's, already made, or a copy) — the client released, nothing called.
        ValidationFailedError: The completion was not JSON, or failed draft validation (the call's spend is
            already ledgered).
    """
    async with client:
        if budget.subject_id is not None and budget.subject_id != subject_id:
            raise ValueError(
                f"the budget ledgers its calls under subject {budget.subject_id!r} and the draft is for "
                f"{subject_id!r}; one budget per proposal, built for the subject it drafts"
            )
        if admitted.purpose != "proposer":
            raise ValueError(
                f"a proposal makes a call admitted for purpose 'proposer', and this one was admitted for "
                f"{admitted.purpose!r}; admit plan_draft's call with budget.admit(client, 'proposer', [call])"
            )
        recorded = await budget.generate(client, admitted)
    content: str = recorded.result.content
    spend = recorded.spend
    refused = _REFUSAL_SUBJECT[axis]
    cost = _spent(spend)
    try:
        payload = extract_json(content)
    except ValueError as e:
        raise ValidationFailedError(
            f"{refused} LLM output was not valid JSON ({cost}): {e}", remedy="propose again, or with another model"
        ) from e

    _coerce_new_dim_axis(payload, axis)
    try:
        proposal = RubricProposal(**payload)
    except ValidationError as e:
        raise ValidationFailedError(
            f"{refused} LLM output failed draft validation ({cost}): {e}", remedy="propose again, or with another model"
        ) from e

    if axis == "boundary":
        log.info(
            "eval.propose_boundary subject=%s actors=%d dims=%d reused=%d new=%d",
            subject_id,
            len(proposal.template.conversation.actors) if proposal.template.conversation is not None else 0,
            len(proposal.template.rubric),
            len(proposal.reused_dim_keys),
            len(proposal.new_dim_suggestions),
        )
    else:
        log.info(
            "eval.propose_rubric subject=%s reused=%d new=%d axes=%d",
            subject_id,
            len(proposal.reused_dim_keys),
            len(proposal.new_dim_suggestions),
            len(proposal.template.variation_axes),
        )
    return ProposedDraft(proposal, spend)


def _spent(spend: OutOfRunSpend) -> str:
    """What the refused draft's call cost, for the refusal to say — it was paid either way."""
    return "the call cost an unreported amount" if spend.cost_usd is None else f"the call cost ${spend.cost_usd:.4f}"


def _coerce_new_dim_axis(payload: Any, axis: RubricAxis) -> None:
    """Stamp the proposer's axis onto every new-dim suggestion and every drafted rubric dim in ``payload``.

    The axis ('capability' vs 'boundary') is determined by *which* axis the proposer ran on,
    not a judgment the drafting LLM should make. Models nonetheless sometimes
    echo a dim's own key/name into the ``axis`` field, which would fail the
    :data:`~threetears.evals.schema.models.RubricAxis` validation and reject an otherwise-valid
    draft. Overwriting it before validation removes that failure mode at the
    cause (the field is server-determined, so the LLM's value is never trusted).

    The drafted template's own rubric dims take it too: a boundary battery's dims are guardrails, and
    the judge stamps a dim's axis onto every score it gives, so a boundary dim drafted as capability
    would be averaged into the composite it must stay out of.
    """
    if not isinstance(payload, dict):
        return
    for sug in payload.get("new_dim_suggestions") or []:
        if isinstance(sug, dict):
            sug["axis"] = axis
            if isinstance(sug.get("dim"), dict):
                sug["dim"]["axis"] = axis
    template = payload.get("template")
    if isinstance(template, dict):
        for dim in template.get("rubric") or []:
            if isinstance(dim, dict):
                dim["axis"] = axis


__all__ = [
    "PROPOSER_MAX_TOKENS",
    "ProposedDraft",
    "plan_draft",
    "propose_draft",
]
