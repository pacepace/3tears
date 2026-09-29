"""the agent -> hub contract for anonymizing the audit rows an agent published.

An agent that erases a person (the survey engine erasing a respondent) holds its own
data, but the platform audit rows its events became live in the hub's audit table. The
agent asks; the hub answers. Neither owns this contract: the request and reply models,
the subject and the caller's client live here, beside the envelope those rows came from,
and both sides import them.

**The rule the hub applies** is the platform's one erasure rule for audit records
(:mod:`threetears.agent.audit.anonymize`): every row is KEPT and no id on it changes; its
``details`` go through :func:`~threetears.agent.audit.anonymize_details` under the row's
own ``event_type``, and its ``ip_address`` through
:func:`~threetears.agent.audit.anonymize_ip`.

**Hub responder obligations** -- what a responder must do to honour this contract:

1. Subscribe :meth:`threetears.nats.Subjects.hub_audit_anonymize`
   (``{ns}.hub.audit.anonymize``) in a queue group, and decode the body as
   :class:`AuditAnonymizeRequest`. A body that does not decode, or breaks its bounds,
   is answered ``INVALID_REQUEST``, echoing ``correlation_id`` whenever the body carried
   one the responder could read. A refusal it could not tie to a correlation id is
   answered with none: the client treats a refusal carrying NO correlation id as the
   refusal of the request it just sent, and a refusal carrying a DIFFERENT one as a
   stray reply, not an answer.
2. Verify ``identity_token`` exactly as every other forwarded-token subject does, and
   derive the calling AGENT from the verified claims. A token that does not verify is
   answered ``IDENTITY_UNVERIFIED``. The agent is NEVER taken from the body.
3. Compare the body's ``agent_id`` with the verified agent and answer ``AGENT_MISMATCH``
   when they differ, touching nothing. The body's copy exists only so the two can be
   compared: a request that believes it is someone else is refused, never silently
   re-scoped.
4. **Authorization rule: only rows whose agent is the verified caller.** Match rows
   ``WHERE <agent column> = <verified agent> AND actor_user_id = ANY(<actor_user_ids>)``,
   where the agent column is the one the audit consumer fills from the envelope's
   ``calling_agent_id`` (the agent whose pod published the event). Never widen the match
   by customer, by actor alone, or by any other body field.
5. For each matched row: ``details = anonymize_details(details, event_type=row.event_type)``
   and ``ip_address = anonymize_ip(ip_address)``. Change nothing else: not the row id,
   ``actor_user_id``, any agent or customer column, ``event_type``, ``action``,
   ``outcome``, correlation ids or timestamps. Never delete a row.
6. Evict every cache holding a changed row (L1, L2 and the cross-pod invalidation), as
   the table's collection does for any write.
7. Reply :class:`AuditAnonymizeReply` with ``success=True``, the request's
   ``correlation_id``, the VERIFIED ``agent_id``, ``rows_matched`` (rows the rule 4 match
   found) and ``rows_changed`` (rows whose stored content actually changed). Rows already
   anonymized match and do not change, which is what makes a retry safe: running the same
   request twice matches the same rows and changes none the second time.
8. A failure after verification is answered ``ANONYMIZE_FAILED`` with a message naming the
   cause for an operator; the caller retries, and the rule makes the retry safe.

``error_code`` vocabulary: :data:`AUDIT_ANONYMIZE_ERROR_CODES`.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Annotated, Final
from uuid import UUID, uuid7

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from threetears.nats import RequestError, Subjects
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.nats import NatsClient

__all__ = [
    "AUDIT_ANONYMIZE_ERROR_CODES",
    "DEFAULT_ANONYMIZE_TIMEOUT_SECONDS",
    "MAX_ANONYMIZE_ACTORS",
    "AuditAnonymization",
    "AuditAnonymizeError",
    "AuditAnonymizeRefusedError",
    "AuditAnonymizeReply",
    "AuditAnonymizeRequest",
    "AuditAnonymizeUnavailableError",
    "request_audit_anonymization",
]

log = get_logger(__name__)

#: actor ids one request may name. bounds the hub's match and the request's size; the
#: client sends a longer list in batches of this size.
MAX_ANONYMIZE_ACTORS: Final[int] = 500

#: seconds a pod waits for the hub's answer to one batch.
DEFAULT_ANONYMIZE_TIMEOUT_SECONDS: Final[float] = 30.0

#: every ``error_code`` a responder answers with.
#:
#: - ``INVALID_REQUEST`` -- the body did not decode, or broke its bounds
#: - ``IDENTITY_UNVERIFIED`` -- the forwarded identity token did not verify
#: - ``AGENT_MISMATCH`` -- the body names an agent other than the verified caller
#: - ``ANONYMIZE_FAILED`` -- the rewrite failed after verification; safe to retry
AUDIT_ANONYMIZE_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {"INVALID_REQUEST", "IDENTITY_UNVERIFIED", "AGENT_MISMATCH", "ANONYMIZE_FAILED"}
)

#: the codes a retry can get past. the client raises :class:`AuditAnonymizeUnavailableError`
#: for these and :class:`AuditAnonymizeRefusedError` for every other code, a code a newer hub
#: added included, because an unknown refusal is not known to be transient.
_RETRYABLE_ERROR_CODES: Final[frozenset[str]] = frozenset({"ANONYMIZE_FAILED"})


class AuditAnonymizeError(Exception):
    """base of every way an audit anonymization request does not complete."""


class AuditAnonymizeRefusedError(AuditAnonymizeError):
    """the hub answered and refused; retrying the same request will be refused again.

    raised for ``INVALID_REQUEST``, ``IDENTITY_UNVERIFIED``, ``AGENT_MISMATCH``, and any code
    a newer hub sends that this client does not know. ``ANONYMIZE_FAILED`` is not a refusal:
    the hub verified the caller and then failed, and that is
    :class:`AuditAnonymizeUnavailableError`.

    :ivar error_code: the hub's code
    :ivar error_message: the hub's description, for an operator
    :ivar correlation_id: the refused batch's correlation id
    """

    def __init__(self, error_code: str, error_message: str, *, correlation_id: UUID) -> None:
        """
        :param error_code: the hub's refusal code
        :ptype error_code: str
        :param error_message: the hub's description
        :ptype error_message: str
        :param correlation_id: the refused batch's correlation id
        :ptype correlation_id: UUID
        """
        self.error_code = error_code
        self.error_message = error_message
        self.correlation_id = correlation_id
        super().__init__(
            f"audit anonymization refused: {error_code}: {error_message} (correlation_id={correlation_id})"
        )


class AuditAnonymizeUnavailableError(AuditAnonymizeError):
    """no usable answer, or a hub failure after it verified the caller.

    covers no identity token, a timeout or transport failure, a reply that does not
    decode, a success without counts or for another agent, a reply to a different
    request, and the hub's ``ANONYMIZE_FAILED``. safe to retry: the hub's rule is
    idempotent, so a batch that did land and whose answer was lost, or that failed part
    way, changes nothing it already changed the second time.
    """


class AuditAnonymizeRequest(BaseModel):
    """an agent's request to anonymize the audit rows it published about some actors.

    :param identity_token: the agent's CURRENT hub-minted identity token; the hub verifies
        it and derives the calling agent from it
    :ptype identity_token: str
    :param correlation_id: echoed on the reply
    :ptype correlation_id: UUID
    :param agent_id: the agent the caller believes it is -- compared with the verified
        agent, never used to choose rows
    :ptype agent_id: UUID
    :param actor_user_ids: the actors whose rows to anonymize, 1 to
        :data:`MAX_ANONYMIZE_ACTORS` of them
    :ptype actor_user_ids: list[UUID]
    """

    model_config = ConfigDict(extra="forbid")

    identity_token: Annotated[str, Field(min_length=1)]
    correlation_id: UUID
    agent_id: UUID
    actor_user_ids: Annotated[list[UUID], Field(min_length=1, max_length=MAX_ANONYMIZE_ACTORS)]


class AuditAnonymizeReply(BaseModel):
    """the hub's answer: counts on success, a code and message on refusal.

    one model for both shapes, because it is the type the caller decodes into and a refusal
    must be readable off the same shape as a success.

    :param success: whether the rows were anonymized
    :ptype success: bool
    :param correlation_id: the request's correlation id
    :ptype correlation_id: UUID | None
    :param agent_id: the agent the hub VERIFIED (on success)
    :ptype agent_id: UUID | None
    :param rows_matched: rows of that agent naming those actors (on success)
    :ptype rows_matched: int | None
    :param rows_changed: of those, rows whose stored content changed (on success)
    :ptype rows_changed: int | None
    :param error_code: one of :data:`AUDIT_ANONYMIZE_ERROR_CODES` (on refusal)
    :ptype error_code: str | None
    :param error_message: a description for an operator (on refusal)
    :ptype error_message: str | None
    """

    success: bool
    correlation_id: UUID | None = None
    agent_id: UUID | None = None
    rows_matched: Annotated[int, Field(ge=0)] | None = None
    rows_changed: Annotated[int, Field(ge=0)] | None = None
    error_code: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class AuditAnonymization:
    """what an audit erasure touched: the one answer both erasure paths give.

    :func:`request_audit_anonymization` returns it for the hub's table, summed over its
    batches; :func:`threetears.agent.audit.persist.anonymize_audit_rows` returns it for a
    hub-less deployment's own table. A consumer supporting both deployments reads one type.

    :ivar rows_matched: rows naming the actors (on the hub path, rows of the calling agent)
    :ivar rows_changed: of those, rows whose stored content changed
    """

    rows_matched: int
    rows_changed: int


async def request_audit_anonymization(
    nats_client: NatsClient,
    *,
    identity_token: str,
    agent_id: UUID,
    actor_user_ids: Collection[UUID],
    timeout_seconds: float = DEFAULT_ANONYMIZE_TIMEOUT_SECONDS,
) -> AuditAnonymization:
    """ask the hub to anonymize the audit rows this agent published about ``actor_user_ids``.

    duplicates are dropped, and a list longer than :data:`MAX_ANONYMIZE_ACTORS` is sent in
    batches of that size, in order, with the counts summed. an empty list sends nothing.
    every batch is safe to retry, so a caller that sees
    :class:`AuditAnonymizeUnavailableError` retries the whole call.

    :param nats_client: this pod's connected NATS client
    :ptype nats_client: NatsClient
    :param identity_token: this pod's CURRENT hub identity token
    :ptype identity_token: str
    :param agent_id: this pod's own agent id, which the hub compares with the verified one
    :ptype agent_id: UUID
    :param actor_user_ids: the actors whose audit rows to anonymize
    :ptype actor_user_ids: Collection[UUID]
    :param timeout_seconds: seconds to wait for each batch's answer
    :ptype timeout_seconds: float
    :return: the rows matched and changed, summed over the batches
    :rtype: AuditAnonymization
    :raises AuditAnonymizeRefusedError: when the hub refuses a batch with ``INVALID_REQUEST``,
        ``IDENTITY_UNVERIFIED``, ``AGENT_MISMATCH`` or a code this client does not know
    :raises AuditAnonymizeUnavailableError: on no token, a transport failure or timeout, a
        reply that does not decode, a reply to a different request, a success without counts
        or for another agent, or the hub's ``ANONYMIZE_FAILED``
    """
    actors = list(dict.fromkeys(actor_user_ids))
    if not actors:
        return AuditAnonymization(rows_matched=0, rows_changed=0)
    if not identity_token:
        raise AuditAnonymizeUnavailableError("audit anonymization has no identity token to present")
    matched = changed = 0
    for start in range(0, len(actors), MAX_ANONYMIZE_ACTORS):
        reply = await _send(
            nats_client,
            AuditAnonymizeRequest(
                identity_token=identity_token,
                correlation_id=uuid7(),
                agent_id=agent_id,
                actor_user_ids=actors[start : start + MAX_ANONYMIZE_ACTORS],
            ),
            timeout_seconds=timeout_seconds,
        )
        matched += reply.rows_matched
        changed += reply.rows_changed
    log.info(
        "audit rows anonymized by the hub",
        extra={
            "extra_data": {
                "agent_id": str(agent_id),  # convert at border: log extra_data
                "actors": len(actors),
                "rows_matched": matched,
                "rows_changed": changed,
            }
        },
    )
    return AuditAnonymization(rows_matched=matched, rows_changed=changed)


async def _send(
    nats_client: NatsClient, request: AuditAnonymizeRequest, *, timeout_seconds: float
) -> AuditAnonymization:
    """send one batch and validate its reply.

    :param nats_client: this pod's connected NATS client
    :ptype nats_client: NatsClient
    :param request: the batch
    :ptype request: AuditAnonymizeRequest
    :param timeout_seconds: seconds to wait for the answer
    :ptype timeout_seconds: float
    :return: the batch's counts
    :rtype: AuditAnonymization
    :raises AuditAnonymizeRefusedError: when the hub refuses with a non-retryable code, its
        correlation id this batch's or absent (a body the hub could not decode carries none
        it could echo)
    :raises AuditAnonymizeUnavailableError: when there is no usable answer -- a reply
        carrying ANOTHER request's correlation id among them -- or the hub reports
        ``ANONYMIZE_FAILED``
    """
    correlation_id = request.correlation_id
    log.debug(
        "audit anonymization batch sent",
        extra={
            "extra_data": {
                "correlation_id": str(correlation_id),  # convert at border: log extra_data
                "actors": len(request.actor_user_ids),
            }
        },
    )
    try:
        raw = await nats_client.request_raw(
            subject=Subjects.hub_audit_anonymize(),
            payload=request.model_dump_json().encode("utf-8"),
            timeout=timedelta(seconds=timeout_seconds),
        )
    except RequestError as exc:
        raise AuditAnonymizeUnavailableError(
            f"audit anonymization request failed (correlation_id={correlation_id}): {exc}"
        ) from exc
    try:
        reply = AuditAnonymizeReply.model_validate_json(raw)
    except ValidationError as exc:
        raise AuditAnonymizeUnavailableError(
            f"audit anonymization reply did not decode (correlation_id={correlation_id}): {exc}"
        ) from exc
    # a refusal carrying NO correlation id is this batch's: a hub that could not decode the body had
    # no id to echo, and filing its INVALID_REQUEST as a stray would tell the caller to retry a request
    # the hub refuses every time. one carrying ANOTHER id -- a late answer to an earlier request, or a
    # hub bug -- is not an answer to this batch, and neither is a success under another id.
    answers_this_batch = reply.correlation_id == correlation_id or (not reply.success and reply.correlation_id is None)
    if not answers_this_batch:
        raise AuditAnonymizeUnavailableError(
            f"audit anonymization reply carried correlation_id={reply.correlation_id}, not this batch's "
            f"{correlation_id}"
        )
    if not reply.success and reply.error_code in _RETRYABLE_ERROR_CODES:
        raise AuditAnonymizeUnavailableError(
            f"audit anonymization failed hub-side (correlation_id={correlation_id}): "
            f"{reply.error_code}: {reply.error_message or 'no details'}"
        )
    if not reply.success:
        raise AuditAnonymizeRefusedError(
            reply.error_code or "UNKNOWN", reply.error_message or "no details", correlation_id=correlation_id
        )
    if reply.rows_matched is None or reply.rows_changed is None:
        raise AuditAnonymizeUnavailableError(
            f"audit anonymization reported success but carried no counts (correlation_id={correlation_id})"
        )
    if reply.agent_id != request.agent_id:
        raise AuditAnonymizeUnavailableError(
            "audit anonymization reply names a different agent than this pod; its counts are not this pod's "
            f"(correlation_id={correlation_id})"
        )
    log.debug(
        "audit anonymization batch answered",
        extra={
            "extra_data": {
                "correlation_id": str(correlation_id),  # convert at border: log extra_data
                "rows_matched": reply.rows_matched,
                "rows_changed": reply.rows_changed,
            }
        },
    )
    return AuditAnonymization(rows_matched=reply.rows_matched, rows_changed=reply.rows_changed)
