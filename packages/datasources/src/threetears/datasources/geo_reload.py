"""the tool pod -> hub contract for reporting reloaded geography shapes.

A tool pod that registers platform geography layers (``geo:`` in its data section)
writes each new generation of their shapes into its own tables, stamping every row
with the generation in the layer's ``version_column``, and then reports it. The hub
answers by moving each layer's tile version to that generation. A tile's address
carries its version and a tile build reads only rows of that generation, so clients
move to the new tiles at once and no cache is purged.

The request and reply models, the subject and the pod's client live here, beside
:class:`~threetears.datasources.geo_config.GeoConfig`, and both sides import them.

**The generation rule.** A layer's tile version starts at 1 and only moves forward.
A pod reports a generation it has fully written:

- the version + 1 or more: the version moves to it;
- equal to the version: nothing changes (a retry after a lost reply);
- below the version: refused ``GENERATION_BEHIND``, with the current version in the
  reply, since tiles would read a generation clients no longer ask for. The pod
  writes its next generation above that version.

A pod picks its next generation as one above both its own highest stamped generation
and the version the hub last answered.

**Retention.** The hub serves a layer's tiles at its current version and the
:data:`RETAINED_GENERATIONS` - 1 before it, since a client holds a TileJSON (and so a
version) for a short while after a reload; it refuses any other version, built or not.
So a pod keeps the rows of its last :data:`RETAINED_GENERATIONS` reported generations
and may delete older ones once the hub has answered the report that superseded them.

**The pod's duties,** in order: write every row of the new generation, each stamped
with it in the layer's ``version_column``; report it; on success, delete generations
older than the retained ones. A load that fails part way is never reported, so its
rows are never read.

**Hub responder obligations:**

1. Subscribe :meth:`threetears.nats.Subjects.hub_geo_layers_reloaded` in a queue group
   and decode :class:`GeoLayersReloadedRequest`; a body that does not decode or breaks
   its bounds is answered ``INVALID_REQUEST``.
2. Verify ``identity_token`` as every forwarded-token subject does; a token that does
   not verify, or names anything but a tool pod, is answered ``IDENTITY_REFUSED``.
3. Every named layer must be registered (``LAYER_NOT_REGISTERED``) under a provider
   namespace the verified pod owns (``LAYER_NOT_OWNED``), and every generation must
   be within :data:`MAX_GENERATION_STEP` of the layer's version (which is 1 before it
   has ever moved)
   (``GENERATION_OUT_OF_RANGE``) and not below it (``GENERATION_BEHIND``). These are
   checked for all layers before any version moves; a refusal moves none.
4. Move each layer's version to its generation, only after the check above, and reply
   :class:`GeoLayersReloadedReply` with ``success=True``, the request's
   ``correlation_id`` and each layer's version.
5. A failure after verification is answered ``RELOAD_FAILED``; the rule makes the
   retry safe.
6. Serve tiles only at versions from the current one back through
   :data:`RETAINED_GENERATIONS` of them, refusing any other without caching the refusal:
   a version not yet reached has no rows, and one past retention may have lost them,
   and a tile built from no rows would be cached as an empty map.

``error_code`` vocabulary: :data:`GEO_RELOAD_ERROR_CODES`.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from typing import TYPE_CHECKING, Annotated, Final
from uuid import UUID, uuid7

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_serializer
from threetears.nats.errors import RequestError
from threetears.nats.subjects import Subjects
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.nats import NatsClient

__all__ = [
    "DEFAULT_GEO_RELOAD_TIMEOUT_SECONDS",
    "GEO_RELOAD_ERROR_CODES",
    "MAX_GENERATION_STEP",
    "MAX_RELOADED_LAYERS",
    "RETAINED_GENERATIONS",
    "GeoLayersReloadedReply",
    "GeoLayersReloadedRequest",
    "GeoReloadError",
    "GeoReloadRefusedError",
    "GeoReloadUnavailableError",
    "report_geo_layers_reloaded",
]

log = get_logger(__name__)

#: layers one request may name; a provider registers a handful
MAX_RELOADED_LAYERS: Final[int] = 100

#: how far one report may move a layer's version. a step this large is a mis-stamped
#: generation, not a run of missed reports, and each step is a durable write and a broadcast.
MAX_GENERATION_STEP: Final[int] = 1000

#: generations of a layer the hub serves tiles for: the current version and the ones just before
#: it. a pod keeps the rows of this many of its latest reported generations.
RETAINED_GENERATIONS: Final[int] = 2

#: seconds a pod waits for the hub's answer
DEFAULT_GEO_RELOAD_TIMEOUT_SECONDS: Final[float] = 30.0

#: every ``error_code`` a responder answers with.
#:
#: - ``INVALID_REQUEST`` -- the body did not decode, or broke its bounds
#: - ``IDENTITY_REFUSED`` -- the forwarded token did not verify, or names no tool pod
#: - ``LAYER_NOT_REGISTERED`` -- no platform layer has that name
#: - ``LAYER_NOT_OWNED`` -- the layer is registered under a namespace the pod does not own
#: - ``GENERATION_BEHIND`` -- a generation below the layer's version; the reply carries the version
#: - ``GENERATION_OUT_OF_RANGE`` -- a generation more than :data:`MAX_GENERATION_STEP` past the version
#: - ``RELOAD_FAILED`` -- the version could not be moved after verification; safe to retry
GEO_RELOAD_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {
        "INVALID_REQUEST",
        "IDENTITY_REFUSED",
        "LAYER_NOT_REGISTERED",
        "LAYER_NOT_OWNED",
        "GENERATION_BEHIND",
        "GENERATION_OUT_OF_RANGE",
        "RELOAD_FAILED",
    }
)

#: the codes a retry can get past; every other code, one a newer hub added included, is a refusal
_RETRYABLE_ERROR_CODES: Final[frozenset[str]] = frozenset({"RELOAD_FAILED"})


class GeoReloadError(Exception):
    """base of every way a reload report does not complete."""


class GeoReloadRefusedError(GeoReloadError):
    """the hub answered and refused; the same report will be refused again.

    :ivar error_code: the hub's code
    :ivar error_message: the hub's description, for an operator
    :ivar versions: each named layer's current version, when the hub reported them
        (``GENERATION_BEHIND``)
    """

    def __init__(self, error_code: str, error_message: str, *, versions: Mapping[str, int] | None = None) -> None:
        """
        :param error_code: the hub's refusal code
        :ptype error_code: str
        :param error_message: the hub's description
        :ptype error_message: str
        :param versions: the layers' current versions, when reported
        :ptype versions: Mapping[str, int] | None
        """
        self.error_code = error_code
        self.error_message = error_message
        self.versions = dict(versions or {})
        super().__init__(f"geography reload refused: {error_code}: {error_message}")


class GeoReloadUnavailableError(GeoReloadError):
    """no usable answer, or a hub failure after it verified the pod; safe to retry."""


class GeoLayersReloadedRequest(BaseModel):
    """a tool pod's report that it has written a new generation of some layers' shapes.

    :param identity_token: the pod's CURRENT hub-minted identity token; a ``SecretStr`` so a
        log line or traceback showing the request shows no token
    :ptype identity_token: SecretStr
    :param correlation_id: echoed on the reply
    :ptype correlation_id: UUID
    :param generations: layer name -> the generation now fully written
    :ptype generations: dict[str, int]
    """

    model_config = ConfigDict(extra="forbid")

    identity_token: SecretStr
    correlation_id: UUID
    generations: Annotated[
        dict[Annotated[str, Field(min_length=1)], Annotated[int, Field(ge=1)]],
        Field(min_length=1, max_length=MAX_RELOADED_LAYERS),
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


class GeoLayersReloadedReply(BaseModel):
    """the hub's answer: each layer's version on success, a code and message on refusal.

    :param success: whether every layer's version stands at its reported generation
    :ptype success: bool
    :param correlation_id: the request's correlation id
    :ptype correlation_id: UUID | None
    :param versions: each named layer's tile version (on success, and on ``GENERATION_BEHIND``)
    :ptype versions: dict[str, int] | None
    :param error_code: one of :data:`GEO_RELOAD_ERROR_CODES` (on refusal)
    :ptype error_code: str | None
    :param error_message: a description for an operator (on refusal)
    :ptype error_message: str | None
    """

    success: bool
    correlation_id: UUID | None = None
    versions: dict[str, int] | None = None
    error_code: str | None = None
    error_message: str | None = None


async def report_geo_layers_reloaded(
    nats_client: NatsClient,
    *,
    identity_token: str,
    generations: Mapping[str, int],
    timeout_seconds: float = DEFAULT_GEO_RELOAD_TIMEOUT_SECONDS,
) -> dict[str, int]:
    """report that these layers' generations are fully written; return each layer's tile version.

    :param nats_client: this pod's connected NATS client
    :ptype nats_client: NatsClient
    :param identity_token: this pod's CURRENT hub identity token
    :ptype identity_token: str
    :param generations: layer name -> the generation now written
    :ptype generations: Mapping[str, int]
    :param timeout_seconds: seconds to wait for the answer
    :ptype timeout_seconds: float
    :return: each layer's tile version, which equals its reported generation
    :rtype: dict[str, int]
    :raises GeoReloadRefusedError: when the hub refuses with a non-retryable code
    :raises GeoReloadUnavailableError: on no token, a transport failure or timeout, a reply
        that does not decode or answers another request, or the hub's ``RELOAD_FAILED``
    """
    if not identity_token:
        raise GeoReloadUnavailableError("a geography reload report has no identity token to present")
    request = GeoLayersReloadedRequest(
        identity_token=SecretStr(identity_token), correlation_id=uuid7(), generations=dict(generations)
    )
    correlation_id = request.correlation_id
    try:
        raw = await nats_client.request_raw(
            subject=Subjects.hub_geo_layers_reloaded(),
            payload=request.model_dump_json().encode("utf-8"),
            timeout=timedelta(seconds=timeout_seconds),
        )
    except RequestError as exc:
        raise GeoReloadUnavailableError(
            f"geography reload report failed (correlation_id={correlation_id}): {exc}"
        ) from exc
    try:
        reply = GeoLayersReloadedReply.model_validate_json(raw)
    except ValidationError as exc:
        raise GeoReloadUnavailableError(
            f"geography reload reply did not decode (correlation_id={correlation_id}): {exc}"
        ) from exc
    # a refusal with no correlation id is this request's: a body the hub could not decode had none to echo
    if reply.correlation_id != correlation_id and (reply.success or reply.correlation_id is not None):
        raise GeoReloadUnavailableError(
            f"geography reload reply carried correlation_id={reply.correlation_id}, not {correlation_id}"
        )
    if not reply.success and reply.error_code in _RETRYABLE_ERROR_CODES:
        raise GeoReloadUnavailableError(
            f"geography reload failed hub-side (correlation_id={correlation_id}): "
            f"{reply.error_code}: {reply.error_message or 'no details'}"
        )
    if not reply.success:
        raise GeoReloadRefusedError(
            reply.error_code or "UNKNOWN", reply.error_message or "no details", versions=reply.versions
        )
    versions = reply.versions or {}
    if any(versions.get(name) != generation for name, generation in request.generations.items()):
        raise GeoReloadUnavailableError(
            f"geography reload reply does not carry the reported generations (correlation_id={correlation_id})"
        )
    log.info(
        "geography layers reloaded",
        extra={"extra_data": {"versions": versions, "correlation_id": str(correlation_id)}},  # convert at border
    )
    return versions
