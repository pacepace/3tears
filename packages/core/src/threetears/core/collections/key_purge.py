"""a tool pod's purge of keys it retired from its own scope of the collections bucket, proven at the hub's door.

The contract is :mod:`threetears.nats.collection_key_requests`'s (the subject, the bounds, the codes,
the hub's half): a pod names keys relative to its scope, and the hub purges exactly those subjects
under the scope its VERIFIED token names. This module is the request as the door scheme reads it
(:mod:`threetears.core.security.door_request`), which the nats package cannot depend on:

- **Proven.** The pod presents its own hub identity token and, holding the key that token binds,
  proves possession of it over this request, its door (:data:`COLLECTION_KEYS_PURGE_DOOR`) and its
  operation (:data:`COLLECTION_KEYS_PURGE_OP`). The door's dated window decides how long it still
  takes the token alone, as every platform door's does.
- **A tolerant reader.** The hub reads the request with :class:`~threetears.core.wire.TolerantWireModel`,
  so a newer pod's added field is ignored rather than refusing the purge. Nothing a pod could add
  steers the purge's reach: the scope is composed from the verified token, never read from the body.

A hub that predates the proofs reads the request ``extra="forbid"`` and refuses one carrying a proof,
so a pod sends one only when given its key (``pop_key``); the release upgrades the hub first. The
strict request and client in :mod:`threetears.nats.collection_key_requests` stay for pods released
before this, until the door's window closes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated
from uuid import UUID, uuid7

from pydantic import Field, SecretStr, field_serializer
from threetears.nats.collection_key_requests import (
    COLLECTION_KEYS_REQUEST_ERROR_CODES,
    DEFAULT_COLLECTION_KEYS_REQUEST_TIMEOUT_SECONDS,
    MAX_PURGED_KEYS,
    SCOPED_KEY_PATTERN,
    CollectionKeysPurgeReply,
    CollectionKeysRequestRefusedError,
    CollectionKeysRequestUnavailableError,
)
from threetears.nats.hub_requests import ask_hub
from threetears.nats.subjects import Subjects
from threetears.observe import get_logger

from threetears.core.security.door_request import (
    COLLECTION_KEYS_PURGE_DOOR,
    COLLECTION_KEYS_PURGE_OP,
    DoorCredential,
    PopKey,
    ProvenDoorRequest,
)
from threetears.core.wire import TolerantWireModel

if TYPE_CHECKING:
    from threetears.nats import NatsClient

__all__ = ["CollectionKeysPurgeRequest", "purge_pod_collection_keys"]

log = get_logger(__name__)

#: the codes after which a retry is safe: a hub-side failure after verification
_RETRYABLE_ERROR_CODES = frozenset({"PURGE_FAILED"}) & COLLECTION_KEYS_REQUEST_ERROR_CODES


class CollectionKeysPurgeRequest(ProvenDoorRequest, TolerantWireModel):
    """a tool pod's ask that the hub purge keys from its own scope of the collections bucket.

    :param identity_token: the pod's CURRENT hub-minted identity token, redacted but on the wire
    :ptype identity_token: SecretStr
    :param correlation_id: echoed on the reply
    :ptype correlation_id: UUID
    :param keys: the keys, relative to the pod's scope (``{table}.{body}``)
    :ptype keys: list[str]
    :param identity_proof: the pod's proof of possession of the key its token binds, over this
        request; ``None`` sends the token alone
    :ptype identity_proof: str | None
    """

    identity_token: SecretStr
    correlation_id: UUID
    keys: Annotated[
        list[Annotated[str, Field(pattern=SCOPED_KEY_PATTERN, max_length=512)]],
        Field(min_length=1, max_length=MAX_PURGED_KEYS),
    ]

    @field_serializer("identity_token", when_used="json")
    def _emit_token_on_the_wire(self, value: SecretStr) -> str:
        """the token in clear on the wire, where the hub verifies it; redacted everywhere else.

        :param value: the token
        :ptype value: SecretStr
        :return: the token's value
        :rtype: str
        """
        return value.get_secret_value()


async def purge_pod_collection_keys(
    nats_client: NatsClient,
    *,
    identity_token: str,
    keys: list[str],
    pop_key: PopKey | None = None,
    timeout_seconds: float = DEFAULT_COLLECTION_KEYS_REQUEST_TIMEOUT_SECONDS,
) -> int:
    """ask the hub to purge ``keys`` from this pod's own scope of the collections bucket, proving its token.

    :param nats_client: this pod's connected NATS client
    :ptype nats_client: NatsClient
    :param identity_token: this pod's CURRENT hub identity token
    :ptype identity_token: str
    :param keys: the keys, relative to the pod's scope
    :ptype keys: list[str]
    :param pop_key: the key ``identity_token`` binds, to prove possession of it over the request;
        ``None`` (or a key the token does not bind) sends the token alone
    :ptype pop_key: PopKey | None
    :param timeout_seconds: seconds to wait for the answer
    :ptype timeout_seconds: float
    :return: how many keys were purged
    :rtype: int
    :raises CollectionKeysRequestRefusedError: when the hub refuses
    :raises CollectionKeysRequestUnavailableError: on no token, a hub that does not answer, a reply that
        does not answer this request, or the hub's ``PURGE_FAILED``
    """
    if not identity_token:
        raise CollectionKeysRequestUnavailableError("a collection keys purge has no identity token to present")
    request = DoorCredential.own(identity_token, pop_key).presented(
        CollectionKeysPurgeRequest(identity_token=SecretStr(identity_token), correlation_id=uuid7(), keys=keys),
        door=COLLECTION_KEYS_PURGE_DOOR,
        op=COLLECTION_KEYS_PURGE_OP,
    )
    reply = await ask_hub(
        nats_client,
        subject=Subjects.hub_collection_keys_purge(),
        request=request,
        reply_type=CollectionKeysPurgeReply,
        what="collection keys purge",
        timeout_seconds=timeout_seconds,
        unavailable=CollectionKeysRequestUnavailableError,
        refused=lambda refusal: CollectionKeysRequestRefusedError(
            refusal.error_code or "UNKNOWN", refusal.error_message or "no details"
        ),
        retryable=_RETRYABLE_ERROR_CODES,
    )
    return reply.purged or 0
