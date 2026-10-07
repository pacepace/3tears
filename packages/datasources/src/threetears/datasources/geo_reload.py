"""the tool pod -> hub contract for reporting reloaded geography shapes.

A tool pod that registers platform geography layers (``geo:`` in its data section)
writes each new generation of their shapes into its own tables, stamping every row
with the generation in the layer's ``version_column``, and then reports it. The hub
answers by moving each layer's tile version to that generation. A tile's address
carries its version and a tile build reads only rows of that generation, so clients
move to the new tiles at once and no cache is purged.

The request and reply models, the subject and the pod's client live here, beside
:class:`~threetears.datasources.geo_config.GeoConfig`, and both sides import them.

**Nothing is served before the first report.** A layer has no tile version until a pod
first reports one, and until then the hub serves nothing for it -- no TileJSON, no tile.
Its rows before that are a load in progress, or one that failed part way, and a tile
built from them would be cached as immutable at an address that stays current. The
first successful report sets the version; a layer the epoch store has never moved has
none.

**The generation rule.** Once set, a layer's tile version only moves forward.
A pod reports a generation it has fully written:

- the version + 1 or more: the version moves to it;
- equal to the version: nothing changes (a retry after a lost reply);
- below the version: refused ``GENERATION_BEHIND``, with the current version in the
  reply, since tiles would read a generation clients no longer ask for. The pod
  writes its next generation above that version.

A pod picks its next generation as one above both its own highest stamped generation
and the version the hub last answered. Generations therefore skip: a load that fails
part way leaves rows stamped with a generation nobody reported, and the next load is
stamped above it.

**Retention.** The hub serves a layer's tiles at exactly two versions: its current
one, and its PREVIOUS one -- the version the current one replaced when the hub moved
it, which a client still holds in a TileJSON for a short while after a reload. The
previous version is RECORDED when the version moves, never inferred: because
generations skip, the version before 7 may be 3, and 6 may be a failed load's partial
rows that were never reported and must never be read. The hub refuses every other
version, built or not. The reply to a successful report carries each layer's previous
version, and the pod keeps the rows of exactly those two generations
(:func:`generations_to_delete` names the rest).

**The pod's duties,** in order: write every row of the new generation, each stamped
with it in the layer's ``version_column``; report it; on success, delete every stamped
generation but the reported one and the reply's previous one -- older reported ones and
failed loads' alike. A load that fails part way is never reported, so its rows are never
read, and the next successful report deletes them.

**Hub responder obligations:**

1. Subscribe :meth:`threetears.nats.Subjects.hub_geo_layers_reloaded` in a queue group
   and decode :class:`GeoLayersReloadedRequest`; a body that does not decode or breaks
   its bounds is answered ``INVALID_REQUEST``.
2. Verify ``identity_token`` as every forwarded-token subject does; a token that does
   not verify, or names anything but a tool pod, is answered ``IDENTITY_REFUSED``.
3. Every named layer must be registered (``LAYER_NOT_REGISTERED``) under a provider
   namespace the verified pod owns (``LAYER_NOT_OWNED``), and every generation must
   be at most :data:`MAX_GENERATION_STEP` past the layer's version
   (``GENERATION_OUT_OF_RANGE``) and not below it (``GENERATION_BEHIND``). A layer that
   has never been reported has no version (``versions`` answers epoch ``0``), so its
   first report is accepted at any generation from 1 to :data:`MAX_GENERATION_STEP` and
   sets its version (obligations 4 and 6). These are checked for all layers before any
   version moves; a refusal moves none.
4. Move each layer's version to its generation, only after the check above, recording
   the version it replaced (:meth:`threetears.epoch.EpochClient.advance_to` keeps it as
   ``previous``), and reply :class:`GeoLayersReloadedReply` with ``success=True``, the
   request's ``correlation_id``, each layer's version and each layer's previous version.
   A report equal to the version moves nothing, and answers the previous version already
   recorded.
5. A failure after verification is answered ``RELOAD_FAILED``; the rule makes the
   retry safe.
6. Serve tiles only at a layer's current version and its recorded previous one
   (:meth:`threetears.epoch.EpochClient.versions`), refusing any other without caching the
   refusal: a version not yet reached has no rows, an older one may have lost them, a
   skipped one is a failed load's, and a tile built from no rows would be cached as an
   empty map. A layer never reported (``versions`` answers epoch ``0``) serves nothing --
   neither tiles nor a TileJSON naming a version.

``error_code`` vocabulary: :data:`GEO_RELOAD_ERROR_CODES`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
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
    "GeoLayersReloadedReply",
    "GeoLayersReloadedRequest",
    "GeoReloadError",
    "GeoReloadRefusedError",
    "GeoReloadUnavailableError",
    "LayerVersions",
    "generations_to_delete",
    "report_geo_layers_reloaded",
]

log = get_logger(__name__)

#: layers one request may name; a provider registers a handful
MAX_RELOADED_LAYERS: Final[int] = 100

#: how far one report may move a layer's version. a step this large is a mis-stamped
#: generation, not a run of missed reports, and each step is a durable write and a broadcast.
MAX_GENERATION_STEP: Final[int] = 1000

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


class LayerVersions(BaseModel):
    """a layer's tile version after a report, and the version it replaced.

    :param version: the layer's current tile version, its reported generation
    :ptype version: int
    :param previous: the version the hub held before ``version``, which it still serves;
        ``None`` when there was none (the layer's first move)
    :ptype previous: int | None
    """

    model_config = ConfigDict(frozen=True)

    version: int
    previous: int | None


def generations_to_delete(stamped: Iterable[int], *, version: int, previous: int | None) -> frozenset[int]:
    """the stamped generations a pod deletes after a successful report: all but the two the hub serves.

    Everything else goes -- older reported generations, and a failed load's rows stamped with a
    generation nobody reported, whether below the version or between it and the previous one.

    :param stamped: every generation the layer's rows are stamped with
    :ptype stamped: Iterable[int]
    :param version: the layer's version, from the reply
    :ptype version: int
    :param previous: the layer's previous version, from the reply
    :ptype previous: int | None
    :return: the generations whose rows to delete
    :rtype: frozenset[int]
    """
    return frozenset(stamped) - {version, previous}


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
    :param previous_versions: each named layer's previous version, the one its current version
        replaced, ``None`` for a layer that has moved once (on success)
    :ptype previous_versions: dict[str, int | None] | None
    :param error_code: one of :data:`GEO_RELOAD_ERROR_CODES` (on refusal)
    :ptype error_code: str | None
    :param error_message: a description for an operator (on refusal)
    :ptype error_message: str | None
    """

    success: bool
    correlation_id: UUID | None = None
    versions: dict[str, int] | None = None
    previous_versions: dict[str, int | None] | None = None
    error_code: str | None = None
    error_message: str | None = None


async def report_geo_layers_reloaded(
    nats_client: NatsClient,
    *,
    identity_token: str,
    generations: Mapping[str, int],
    timeout_seconds: float = DEFAULT_GEO_RELOAD_TIMEOUT_SECONDS,
) -> dict[str, LayerVersions]:
    """report that these layers' generations are fully written; return each layer's versions.

    :param nats_client: this pod's connected NATS client
    :ptype nats_client: NatsClient
    :param identity_token: this pod's CURRENT hub identity token
    :ptype identity_token: str
    :param generations: layer name -> the generation now written
    :ptype generations: Mapping[str, int]
    :param timeout_seconds: seconds to wait for the answer
    :ptype timeout_seconds: float
    :return: each layer's version, which equals its reported generation, and its previous one;
        the pod keeps those two generations' rows and deletes the rest (:func:`generations_to_delete`)
    :rtype: dict[str, LayerVersions]
    :raises GeoReloadRefusedError: when the hub refuses with a non-retryable code, or
        ``INVALID_REQUEST`` without asking it when ``generations`` breaks the request's bounds
        (none, too many, an empty name, a generation below 1)
    :raises GeoReloadUnavailableError: on no token, a transport failure or timeout, a reply
        that does not decode or answers another request, or the hub's ``RELOAD_FAILED``
    """
    if not identity_token:
        raise GeoReloadUnavailableError("a geography reload report has no identity token to present")
    try:
        request = GeoLayersReloadedRequest(
            identity_token=SecretStr(identity_token), correlation_id=uuid7(), generations=dict(generations)
        )
    except ValidationError as exc:
        # the hub would refuse the same body, so it is refused here with the hub's code
        raise GeoReloadRefusedError("INVALID_REQUEST", f"geography reload report is not valid: {exc}") from exc
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
    previous = reply.previous_versions or {}
    if any(versions.get(name) != generation for name, generation in request.generations.items()):
        raise GeoReloadUnavailableError(
            f"geography reload reply does not carry the reported generations (correlation_id={correlation_id})"
        )
    # without it the pod cannot know which older generation the hub still serves, so it could delete
    # rows clients are reading; a reply that leaves it out is not trusted
    if any(name not in previous for name in request.generations):
        raise GeoReloadUnavailableError(
            f"geography reload reply does not carry each layer's previous version (correlation_id={correlation_id})"
        )
    result = {name: LayerVersions(version=versions[name], previous=previous[name]) for name in request.generations}
    log.info(
        "geography layers reloaded",
        extra={
            "extra_data": {
                "versions": versions,
                "previous_versions": previous,
                "correlation_id": str(correlation_id),  # convert at border: a log line
            }
        },
    )
    return result
