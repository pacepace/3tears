"""The rubric proposer: one completion call that drafts a template for operator review.

One proposer, run on one of two axes. On the ``capability`` axis it drafts a capability rubric
and its scenario axes; on the ``boundary`` axis it drafts a universal boundary battery and its
refusal dims. Either way it sends a system prompt and a two-feed user message — the subject
feed (what the subject under test is) and the catalog feed (the reusable rubric dims the
operator already trusts) — and validates the reply into a
:class:`~threetears.evals.contracts.models.RubricProposal`. Nothing is persisted: the operator
edits the draft and commits the parts they accept through the authoring operations.

**What arrives here is text.** The host renders the subject feed from its own subject, the
catalog feed from its rubric-dim store, and the system prompt from its prompt registry — each
for the axis it is proposing on — then hands this module the three strings and a
:class:`~threetears.evals.contracts.provider.CompletionClient`. So the proposer knows neither what
a subject is nor where a prompt is kept, and every host drafts with the same code whatever its
subject is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from threetears.evals.contracts.errors import ValidationFailedError
from threetears.evals.contracts.models import RubricAxis, RubricProposal
from threetears.evals.contracts.provider import JSON_OBJECT_RESPONSE_FORMAT, extract_json
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.evals.contracts.provider import CompletionClient

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


async def propose_draft(
    client: CompletionClient,
    *,
    axis: RubricAxis,
    subject_id: str,
    system_prompt: str,
    subject_feed: str,
    catalog_feed: str,
) -> RubricProposal:
    """Draft a rubric on ``axis`` from a subject feed and a catalog feed.

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

    One call on ``client`` renders both feeds into a strict-JSON draft, which is validated
    into a :class:`~threetears.evals.contracts.models.RubricProposal` and **returned for operator
    review — nothing is persisted**. The client is released when the call returns, on the
    refusal paths too: a proposal is the client's whole lifetime.

    Args:
        client: The completion client to draft with — the host's client for the ``proposer`` role
            (:data:`~threetears.evals.contracts.host.CompletionRole`). This function owns it from here
            and releases it.
        axis: Which catalog axis the draft is for; stamped onto every new-dim suggestion.
        subject_id: The subject the draft is for, named in the log line only.
        system_prompt: The rendered system prompt for ``axis``.
        subject_feed: The rendered description of the subject under test.
        catalog_feed: The rendered reusable catalog for ``axis``, each dim with its ``key``
            so the model can name reuses in ``reused_dim_keys``.

    Returns:
        The validated :class:`~threetears.evals.contracts.models.RubricProposal` draft.

    Raises:
        ValidationFailedError: The completion was not JSON, or failed draft validation.
    """
    refused = _REFUSAL_SUBJECT[axis]
    content = await _draft(
        client, system_prompt=system_prompt, user_prompt=_assemble_user_prompt(subject_feed, catalog_feed)
    )
    try:
        payload = extract_json(content)
    except ValueError as e:
        raise ValidationFailedError(f"{refused} LLM output was not valid JSON: {e}") from e

    _coerce_new_dim_axis(payload, axis)
    try:
        proposal = RubricProposal(**payload)
    except ValidationError as e:
        raise ValidationFailedError(f"{refused} LLM output failed draft validation: {e}") from e

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
    return proposal


async def _draft(client: CompletionClient, *, system_prompt: str, user_prompt: str) -> str:
    """Send the prompt pair in JSON-object mode and return the completion text.

    The ``async with`` releases the client on every exit: it owns a connection pool that
    garbage collection does not close deterministically, and one proposal is its whole
    lifetime.
    """
    async with client:
        response = await client.generate(
            system=system_prompt, user=user_prompt, response_format=JSON_OBJECT_RESPONSE_FORMAT
        )
    return response.content


def _assemble_user_prompt(subject_feed: str, catalog_feed: str) -> str:
    """Join the two feeds into the user message the proposer sends on either axis.

    The subject feed comes first because the catalog is read against it: which dims to
    reuse depends on what the subject is.
    """
    return f"{subject_feed}\n{catalog_feed}".rstrip() + "\n"


def _coerce_new_dim_axis(payload: Any, axis: RubricAxis) -> None:
    """Stamp the proposer's axis onto every new-dim suggestion in ``payload``.

    The axis ('capability' vs 'boundary') is determined by *which* axis the proposer ran on,
    not a judgment the drafting LLM should make. Models nonetheless sometimes
    echo a dim's own key/name into the ``axis`` field, which would fail the
    :data:`~threetears.evals.contracts.models.RubricAxis` validation and reject an otherwise-valid
    draft. Overwriting it before validation removes that failure mode at the
    cause (the field is server-determined, so the LLM's value is never trusted).
    """
    if not isinstance(payload, dict):
        return
    for sug in payload.get("new_dim_suggestions") or []:
        if isinstance(sug, dict):
            sug["axis"] = axis


__all__ = [
    "PROPOSER_MAX_TOKENS",
    "propose_draft",
]
