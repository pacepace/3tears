"""the tool pod -> hub contract for a CDN-cached index that moved to a new version.

A pod that serves an index a CDN caches -- the newest version of a customer's report, read by every viewer
through the CDN for a second at a time -- moves it when its data moves. Until the CDN's copy expires, a
viewer can read the old index; the hub owns the CDN and purges the customer's cached index on the pod's
word, so the move reaches viewers at once.

- **The pod** reports the customer, the conversion and the version the index now names, and the scope
  of the index when it serves more than one (:func:`report_cache_index_moved`), forwarding its
  ``identity_token``. Nothing in it is per viewer.
- **The hub** verifies the token and purges the cached index the verified pod serves for that customer.
  A ``success`` means the purge was accepted and queued (coalesced, retried), not that it has finished.

**Rollout.** A hub older than this contract has no responder on the subject (and an older hub's grant does
not name it), so the request is refused or times out; :class:`CacheIndexRequestUnavailableError` then tells
the pod the CDN's copy expires on its own, as before.

**Hub responder obligations:**

1. Subscribe :meth:`threetears.nats.Subjects.hub_cache_index_moved` in a queue group; a body that does not
   decode or breaks its bounds is answered ``INVALID_REQUEST``, as is a ``customer_id`` that is no existing
   customer's, and a non-empty ``scope`` while the hub purges only a pod's one global index.
2. Verify ``identity_token`` as every forwarded-token subject does; a token that does not verify, or names
   anything but a tool pod, is answered ``IDENTITY_REFUSED``.
3. Purge only what the VERIFIED pod serves: a pod that does not own the index's provider node is answered
   ``INDEX_NOT_OWNED``. A failure after verification, or a hub with no purge configured, is ``PURGE_FAILED``,
   and a retry is safe (purging a purged index purges nothing). ``success`` means the purge was accepted
   and queued.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Final
from uuid import UUID, uuid7

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_serializer

from threetears.nats.hub_requests import ask_hub
from threetears.nats.subjects import Subjects

if TYPE_CHECKING:
    from threetears.nats.client import NatsClient

__all__ = [
    "CACHE_INDEX_REQUEST_ERROR_CODES",
    "DEFAULT_CACHE_INDEX_REQUEST_TIMEOUT_SECONDS",
    "CacheIndexMovedReply",
    "CacheIndexMovedRequest",
    "CacheIndexRequestError",
    "CacheIndexRequestRefusedError",
    "CacheIndexRequestUnavailableError",
    "report_cache_index_moved",
]

#: seconds a pod waits for the hub: short, because a hub that does not know the subject never answers,
#: and the CDN's copy expires on its own
DEFAULT_CACHE_INDEX_REQUEST_TIMEOUT_SECONDS: Final[float] = 5.0

#: the codes that are a hub-side failure after verification: no usable answer, and a retry is safe
_RETRYABLE_ERROR_CODES: Final[frozenset[str]] = frozenset({"PURGE_FAILED"})

#: every ``error_code`` a responder answers with
CACHE_INDEX_REQUEST_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {"INVALID_REQUEST", "IDENTITY_REFUSED", "INDEX_NOT_OWNED", "PURGE_FAILED"}
)

#: a name the request carries: printable, bounded
_Name = Annotated[str, Field(min_length=1, max_length=256, pattern=r"^[^\s]+$")]


class CacheIndexRequestError(Exception):
    """base of every way an index-moved report to the hub does not complete."""


class CacheIndexRequestRefusedError(CacheIndexRequestError):
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
        super().__init__(f"cache index moved refused: {error_code}: {error_message}")


class CacheIndexRequestUnavailableError(CacheIndexRequestError):
    """no usable answer: a hub that does not know the request, a timeout, or a hub-side failure."""


class CacheIndexMovedRequest(BaseModel):
    """a tool pod's word that the CDN-cached index it serves for a customer moved to a new version.

    :param identity_token: the pod's CURRENT hub-minted identity token, redacted but on the wire
    :ptype identity_token: SecretStr
    :param correlation_id: echoed on the reply
    :ptype correlation_id: UUID
    :param customer_id: the customer whose index moved
    :ptype customer_id: UUID
    :param conversion: the conversion the index is under
    :ptype conversion: str
    :param version: the version the index now names
    :ptype version: str
    :param scope: which of the pod's indexes moved, ``""`` for its one global index (the only one a hub
        purges today; it refuses any other)
    :ptype scope: str
    """

    model_config = ConfigDict(extra="forbid")

    identity_token: SecretStr
    correlation_id: UUID
    customer_id: UUID
    conversion: _Name
    version: _Name
    scope: Annotated[str, Field(max_length=256)] = ""

    @field_serializer("identity_token", when_used="json")
    def _emit_token_on_the_wire(self, value: SecretStr) -> str:
        """the token in clear on the wire, where the hub verifies it; redacted everywhere else.

        :param value: the token
        :ptype value: SecretStr
        :return: the token's value
        :rtype: str
        """
        return value.get_secret_value()


class CacheIndexMovedReply(BaseModel):
    """the hub's answer to an index-moved report.

    :param success: whether the customer's cached index was purged
    :ptype success: bool
    :param correlation_id: the request's correlation id
    :ptype correlation_id: UUID | None
    :param error_code: one of :data:`CACHE_INDEX_REQUEST_ERROR_CODES` (on refusal)
    :ptype error_code: str | None
    :param error_message: a description for an operator (on refusal)
    :ptype error_message: str | None
    """

    success: bool
    correlation_id: UUID | None = None
    error_code: str | None = None
    error_message: str | None = None


async def report_cache_index_moved(
    nats_client: NatsClient,
    *,
    identity_token: str,
    customer_id: UUID,
    conversion: str,
    version: str,
    scope: str = "",
    timeout_seconds: float = DEFAULT_CACHE_INDEX_REQUEST_TIMEOUT_SECONDS,
) -> CacheIndexMovedReply:
    """tell the hub that the CDN-cached index this pod serves for ``customer_id`` moved to ``version``.

    :param nats_client: this pod's connected NATS client
    :ptype nats_client: NatsClient
    :param identity_token: this pod's CURRENT hub identity token
    :ptype identity_token: str
    :param customer_id: the customer whose index moved
    :ptype customer_id: UUID
    :param conversion: the conversion the index is under
    :ptype conversion: str
    :param version: the version the index now names
    :ptype version: str
    :param scope: which of the pod's indexes moved, ``""`` for its one global index
    :ptype scope: str
    :param timeout_seconds: seconds to wait for the answer
    :ptype timeout_seconds: float
    :return: the hub's reply, a success: the purge was accepted and queued
    :rtype: CacheIndexMovedReply
    :raises CacheIndexRequestRefusedError: when the hub refuses (``INVALID_REQUEST``, ``IDENTITY_REFUSED``,
        ``INDEX_NOT_OWNED``)
    :raises CacheIndexRequestUnavailableError: on no token, a hub that does not answer (one older than this
        contract among them), a reply that does not decode or does not answer this request, or the hub's
        ``PURGE_FAILED``
    """
    if not identity_token:
        raise CacheIndexRequestUnavailableError("a cache index moved report has no identity token to present")
    request = CacheIndexMovedRequest(
        identity_token=SecretStr(identity_token),
        correlation_id=uuid7(),
        customer_id=customer_id,
        conversion=conversion,
        version=version,
        scope=scope,
    )
    return await ask_hub(
        nats_client,
        subject=Subjects.hub_cache_index_moved(),
        request=request,
        reply_type=CacheIndexMovedReply,
        what="cache index moved report",
        timeout_seconds=timeout_seconds,
        unavailable=CacheIndexRequestUnavailableError,
        refused=lambda refusal: CacheIndexRequestRefusedError(
            refusal.error_code or "UNKNOWN", refusal.error_message or "no details"
        ),
        retryable=_RETRYABLE_ERROR_CODES,
    )
