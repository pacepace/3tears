"""the tool pod -> hub contract for purging keys a pod retired from its own scope of the collections bucket.

A KV delete is a publish of a marker message, and on the shared ``{ns}-collections`` bucket (history
1, no ``max_age``) that marker is the key's one message from then on: a cache that retires every
answer of every old version leaves one marker per answer, forever. A JetStream stream purge filtered
to the key's exact subject removes every message of it and leaves nothing. A purge is a
stream-management verb, which no pod holds, so a pod asks the hub, which owns the bucket.

- **The pod** names its keys RELATIVE to its own scope (``{table}.{body}``) and forwards its
  ``identity_token`` (:func:`purge_pod_collection_keys`). It purges only keys it has already deleted
  (or that hold nothing it still serves): the purge carries no revision check.
- **The hub** verifies the token, composes each subject as ``$KV.{bucket}.{verified scope}.{key}``
  and purges exactly that subject (:func:`purge_scoped_keys`). A key is one literal subject below the
  scope: no wildcard, no empty token, so no request can reach another principal's keys.

**Rollout.** A hub older than this contract has no responder on the subject (and an older hub's grant
does not name it), so the request is refused or times out; :class:`CollectionKeysRequestUnavailableError`
then tells the pod to keep the markers, which cost space and nothing else.

**Hub responder obligations:**

1. Subscribe :meth:`threetears.nats.Subjects.hub_collection_keys_purge` in a queue group; a body that
   does not decode or breaks its bounds is answered ``INVALID_REQUEST``.
2. Verify ``identity_token`` as every forwarded-token subject does; a token that does not verify, or
   names anything but a tool pod, is answered ``IDENTITY_REFUSED``.
3. Purge under the VERIFIED pod's scope only; a failure after verification is ``PURGE_FAILED``, and a
   retry is safe (purging an absent subject purges nothing).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Final
from uuid import UUID, uuid7

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_serializer
from threetears.observe import get_logger

from threetears.nats.hub_requests import ask_hub
from threetears.nats.subject_permissions import kv_stream_name
from threetears.nats.subjects import Subjects

if TYPE_CHECKING:
    from threetears.nats.client import NatsClient

__all__ = [
    "COLLECTION_KEYS_REQUEST_ERROR_CODES",
    "DEFAULT_COLLECTION_KEYS_REQUEST_TIMEOUT_SECONDS",
    "MAX_PURGED_KEYS",
    "SCOPED_KEY_PATTERN",
    "CollectionKeysPurgeReply",
    "CollectionKeysPurgeRequest",
    "CollectionKeysRequestError",
    "CollectionKeysRequestRefusedError",
    "CollectionKeysRequestUnavailableError",
    "purge_pod_collection_keys",
    "purge_scoped_keys",
]

log = get_logger(__name__)

#: keys one request may name
MAX_PURGED_KEYS: Final[int] = 1000

#: seconds a pod waits for the hub: short, because a hub that does not know the subject never answers,
#: and a retirement waiting on it is waiting on nothing
DEFAULT_COLLECTION_KEYS_REQUEST_TIMEOUT_SECONDS: Final[float] = 5.0

#: a key below the pod's scope: dot-separated tokens of the KV key grammar, none empty, so never a
#: wildcard (``*``, ``>``) and never a token that climbs out of the scope
SCOPED_KEY_PATTERN: Final[str] = r"^[-/_=a-zA-Z0-9]+(\.[-/_=a-zA-Z0-9]+)*$"

#: the codes that are a hub-side failure after verification: no usable answer, and a retry is safe
_RETRYABLE_ERROR_CODES: Final[frozenset[str]] = frozenset({"PURGE_FAILED"})

#: every ``error_code`` a responder answers with
COLLECTION_KEYS_REQUEST_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {"INVALID_REQUEST", "IDENTITY_REFUSED", "PURGE_FAILED"}
)


class CollectionKeysRequestError(Exception):
    """base of every way a purge request to the hub does not complete."""


class CollectionKeysRequestRefusedError(CollectionKeysRequestError):
    """the hub answered and refused.

    :ivar error_code: the hub's code
    :ivar error_message: the hub's description
    """

    def __init__(self, error_code: str, error_message: str) -> None:
        """
        :param error_code: the hub's refusal code
        :ptype error_code: str
        :param error_message: the hub's description
        :ptype error_message: str
        """
        self.error_code = error_code
        self.error_message = error_message
        super().__init__(f"collection keys purge refused: {error_code}: {error_message}")


class CollectionKeysRequestUnavailableError(CollectionKeysRequestError):
    """no usable answer: a hub that does not know the request, a timeout, or a hub-side failure."""


class CollectionKeysPurgeRequest(BaseModel):
    """a tool pod's ask that the hub purge keys from its own scope of the collections bucket.

    :param identity_token: the pod's CURRENT hub-minted identity token, redacted but on the wire
    :ptype identity_token: SecretStr
    :param correlation_id: echoed on the reply
    :ptype correlation_id: UUID
    :param keys: the keys, relative to the pod's scope (``{table}.{body}``)
    :ptype keys: list[str]
    """

    model_config = ConfigDict(extra="forbid")

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


class CollectionKeysPurgeReply(BaseModel):
    """the hub's answer to a purge.

    :param success: whether every named key's subject was purged
    :ptype success: bool
    :param correlation_id: the request's correlation id
    :ptype correlation_id: UUID | None
    :param purged: how many keys were purged (on success)
    :ptype purged: int | None
    :param error_code: one of :data:`COLLECTION_KEYS_REQUEST_ERROR_CODES` (on refusal)
    :ptype error_code: str | None
    :param error_message: a description for an operator (on refusal)
    :ptype error_message: str | None
    """

    success: bool
    correlation_id: UUID | None = None
    purged: int | None = None
    error_code: str | None = None
    error_message: str | None = None


async def purge_pod_collection_keys(
    nats_client: NatsClient,
    *,
    identity_token: str,
    keys: list[str],
    timeout_seconds: float = DEFAULT_COLLECTION_KEYS_REQUEST_TIMEOUT_SECONDS,
) -> int:
    """ask the hub to purge ``keys`` from this pod's own scope of the collections bucket.

    :param nats_client: this pod's connected NATS client
    :ptype nats_client: NatsClient
    :param identity_token: this pod's CURRENT hub identity token
    :ptype identity_token: str
    :param keys: the keys, relative to the pod's scope
    :ptype keys: list[str]
    :param timeout_seconds: seconds to wait for the answer
    :ptype timeout_seconds: float
    :return: how many keys were purged
    :rtype: int
    :raises CollectionKeysRequestRefusedError: when the hub refuses
    :raises CollectionKeysRequestUnavailableError: on no token, a hub that does not answer (one older
        than this contract among them), a reply that does not decode or does not answer this request
        (a success with no correlation id among them), or the hub's ``PURGE_FAILED``
    """
    if not identity_token:
        raise CollectionKeysRequestUnavailableError("a collection keys purge has no identity token to present")
    request = CollectionKeysPurgeRequest(identity_token=SecretStr(identity_token), correlation_id=uuid7(), keys=keys)
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


async def purge_scoped_keys(jetstream: Any, *, bucket: str, scope: str, keys: list[str]) -> int:
    """purge each key's exact subject in ``bucket``, under ``scope``: the hub's half, after it verified the scope.

    :param jetstream: the hub's JetStream context (``purge_stream(name, subject=)``)
    :ptype jetstream: Any
    :param bucket: the bucket's full name, ``{ns}-collections``
    :ptype bucket: str
    :param scope: the VERIFIED caller's key scope
    :ptype scope: str
    :param keys: the keys relative to that scope, each already matching :data:`SCOPED_KEY_PATTERN`
    :ptype keys: list[str]
    :return: how many keys were purged
    :rtype: int
    :raises ValueError: when a key is not one literal subject below the scope
    """
    checked = CollectionKeysPurgeRequest.model_validate(
        {"identity_token": "checked", "correlation_id": uuid7(), "keys": keys}
    )
    for key in checked.keys:
        await jetstream.purge_stream(kv_stream_name(bucket), subject=f"$KV.{bucket}.{scope}.{key}")
    return len(checked.keys)
