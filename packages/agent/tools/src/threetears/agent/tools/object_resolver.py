"""Consumer-side hub caller: resolve an object id to its stored key.

The read-side, identity-bound counterpart of the SDK's ``HubObjectCataloger``.
A consuming tool holds an object id (an LLM arg, a cross-turn reference) but not
the ``s3_key``; :class:`HubObjectResolver` asks the hub to resolve it -- tenant
safely -- to the key the tool then streams down or presigns.

Authentication rides the per-call ``identity_token`` the invoking agent already
carries on the call context (a hub-minted, EdDSA-signed assertion the hub
verifies in-process). The hub derives the owning customer from the VERIFIED
``customer_id`` claim -- never an unauthenticated request field -- so a tool can
never resolve an object outside its tenant, and this pod (a pure-``threetears``
tool server) needs no hub session of its own.

Shape mirrors :class:`~threetears.core.security.jwks_provider.CachedHubJwksProvider`
-- a pod-side hub caller, cached, fail-closed -- but NOT its refresh loop: an
object's id -> key mapping is immutable once committed, so a resolved key never
rotates. The cache is fill-on-resolve, keyed by the VERIFIED
``(customer_id, object_id)`` (no cross-tenant reuse); failures are never cached.

**Two caches, and which one you get is a property of the pod, not of this class.**
A pod that declared Collection tables has a three-tier stack, and
:class:`ToolServerBootstrap` hands this resolver the runtime's
:class:`~threetears.agent.tools.object_resolution_collection.ObjectResolutionCollection`
-- L1 in this process, L2 shared with every replica under the pod's own key scope, so a
resolution one replica paid for serves all of them. A pod that declared none keeps the
historical behaviour: a process-local ``dict`` with bounded FIFO eviction, which is a
real cache for a single replica and no cache at all for the second one.

Fail-closed: a transport error, a hub error reply (identity unverified / object
not found), or a malformed success reply raises :class:`ResolveObjectError` and
caches nothing. A consuming tool must never proceed on an unresolved or
cross-tenant key.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Protocol
from uuid import UUID, uuid7

from pydantic import BaseModel
from threetears.media.contracts import ObjectHandle
from threetears.nats import NatsClient, RequestError, Subjects
from threetears.observe import get_logger

__all__ = [
    "HubObjectResolver",
    "ObjectResolutionCache",
    "ObjectResolveRequestModel",
    "ObjectResolveResponseModel",
    "ObjectResolver",
    "ResolveObjectError",
]

_log = get_logger(__name__)

_DEFAULT_CACHE_MAX = 512


class ResolveObjectError(RuntimeError):
    """A consuming tool could not resolve an object id to its stored key.

    Carries a structural reason only (transport failure / hub rejection /
    malformed reply) -- never object bytes or key material.
    """


class ObjectResolveRequestModel(BaseModel):
    """outbound object-resolve request to the hub.

    carries the invoking agent's ``identity_token`` as the caller proof. the
    owning customer is NOT sent: the hub derives it server-side from the
    VERIFIED token claim (never an unauthenticated request field), so a tool can
    never resolve an object under a customer it does not own. hand-mirrors the
    hub's ``ObjectResolveRequest`` (each side owns its model; they agree on the
    wire, like the secret + commit request pairs).

    :param identity_token: the hub-minted identity assertion the invoking agent
        carries on the call context; the hub verifies it + derives the customer
    :ptype identity_token: str
    :param correlation_id: transport correlation id echoed in the reply (NOT a
        replay control -- resolve replay is bounded by token expiry + session
        liveness hub-side)
    :ptype correlation_id: UUID
    :param object_id: the object id to resolve
    :ptype object_id: UUID
    """

    identity_token: str
    correlation_id: UUID
    object_id: UUID


class ObjectResolveResponseModel(BaseModel):
    """inbound object-resolve response from the hub.

    one permissive model absorbs both the success and the error reply (the
    fields present differ), matching the SDK's ``ObjectCommitResponseModel``
    pattern: check ``success`` then read the success fields.

    :param success: whether the object was resolved
    :ptype success: bool
    :param correlation_id: correlation id matching the request
    :ptype correlation_id: UUID | None
    :param s3_key: the resolved object-store key (on success)
    :ptype s3_key: str | None
    :param mime_type: stored object MIME type (on success)
    :ptype mime_type: str | None
    :param size_bytes: stored object size in bytes (on success)
    :ptype size_bytes: int | None
    :param error_code: machine-readable error code (on failure)
    :ptype error_code: str | None
    :param error_message: human-readable error description (on failure)
    :ptype error_message: str | None
    """

    success: bool
    correlation_id: UUID | None = None
    s3_key: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    error_code: str | None = None
    error_message: str | None = None


class ObjectResolutionCache(Protocol):
    """the two-tier store a resolved mapping is remembered in.

    Declared structurally rather than imported so this module keeps no dependency on the
    collection that satisfies it; the concrete implementation is
    :class:`~threetears.agent.tools.object_resolution_collection.ObjectResolutionCollection`,
    and a test may substitute anything of this shape.
    """

    async def lookup(self, customer_id: UUID, object_id: UUID) -> "ObjectHandle | None":
        """return the remembered handle for a verified pair, or ``None`` on a miss.

        :param customer_id: the VERIFIED owning customer
        :ptype customer_id: UUID
        :param object_id: the object whose stored key is wanted
        :ptype object_id: UUID
        :return: the handle, or ``None``
        :rtype: ObjectHandle | None
        """
        ...

    async def remember(self, customer_id: UUID, handle: "ObjectHandle") -> None:
        """record a resolved handle so this pod's replicas need not re-resolve it.

        :param customer_id: the VERIFIED owning customer
        :ptype customer_id: UUID
        :param handle: the resolved handle
        :ptype handle: ObjectHandle
        :return: nothing
        :rtype: None
        """
        ...


class ObjectResolver(Protocol):
    """resolves an object id to its stored key under the verified tenant.

    the abstraction the tool server installs on every call scope so consuming
    tools reach a resolver through ``current_scope`` -- the same way they reach
    the object store -- without per-tool constructor plumbing. tests inject a
    fake; production wires :class:`HubObjectResolver`.
    """

    async def resolve(self, object_id: UUID, *, customer_id: UUID, identity_token: str) -> ObjectHandle:
        """resolve ``object_id`` to a handle carrying its stored key, or raise.

        :param object_id: the object id to resolve
        :ptype object_id: UUID
        :param customer_id: the VERIFIED owning customer (cache scoping only;
            NOT sent to the hub -- the hub derives the customer from the token)
        :ptype customer_id: UUID
        :param identity_token: the caller proof forwarded to the hub
        :ptype identity_token: str
        :return: a handle with the resolved ``s3_key`` / ``mime_type`` /
            ``size_bytes`` (``summary`` / ``category`` are ``None`` -- resolve
            returns only the storage-locating fields)
        :rtype: ObjectHandle
        :raises ResolveObjectError: transport failure, hub rejection, or a
            malformed success reply
        """
        ...


class HubObjectResolver:
    """resolves object ids to stored keys over NATS, tenant-safely (Path-2).

    a per-pod resolver the tool server self-provisions from its NATS client (it
    needs no S3 creds, unlike the object store) and installs on every call
    scope. holds only the NATS client + an id -> handle cache; there is no
    background loop, because a committed object's id -> key mapping never
    changes.

    :param nats_client: connected canonical NATS wrapper client
    :ptype nats_client: NatsClient
    :param request_timeout_seconds: the resolve request/reply timeout in seconds
    :ptype request_timeout_seconds: float
    :param cache_max: the most id -> handle mappings the fallback in-process
        cache retains before FIFO eviction; keeps it bounded on a long-lived
        pod. ignored when ``resolution_cache`` is supplied, which bounds itself
        through the pod's L1 tier
    :ptype cache_max: int
    :param resolution_cache: the pod's two-tier resolution store, when it has
        one. supplied it REPLACES the in-process dict, so a mapping resolved on
        one replica is served to every replica. ``None`` -> the historical
        pod-local cache
    :ptype resolution_cache: ObjectResolutionCache | None
    """

    def __init__(
        self,
        nats_client: NatsClient,
        *,
        request_timeout_seconds: float,
        cache_max: int = _DEFAULT_CACHE_MAX,
        resolution_cache: ObjectResolutionCache | None = None,
    ) -> None:
        self._nc = nats_client
        self._timeout = request_timeout_seconds
        self._cache_max = cache_max
        self._resolution_cache = resolution_cache
        # keyed by the VERIFIED (customer_id, object_id): a mapping resolved for
        # one tenant is never reused for another. insertion-ordered for FIFO
        # eviction once the cap is reached. unused when a resolution_cache was
        # supplied -- its L1 tier is this, per process, plus a shared L2.
        self._cache: dict[tuple[UUID, UUID], ObjectHandle] = {}

    async def resolve(self, object_id: UUID, *, customer_id: UUID, identity_token: str) -> ObjectHandle:
        """resolve ``object_id`` to a handle carrying its stored key, or raise.

        Serves from cache when the ``(customer_id, object_id)`` mapping is known
        (immutable once committed); otherwise sends a resolve request carrying
        the ``identity_token`` and caches the success. Fail-closed on transport
        error, an error reply, or a success reply missing the key.

        :param object_id: the object id to resolve
        :ptype object_id: UUID
        :param customer_id: the VERIFIED owning customer, for cache scoping only
        :ptype customer_id: UUID
        :param identity_token: the caller proof forwarded to the hub
        :ptype identity_token: str
        :return: a handle with the resolved ``s3_key`` / ``mime_type`` /
            ``size_bytes``
        :rtype: ObjectHandle
        :raises ResolveObjectError: transport failure, hub rejection, or a
            malformed success reply
        """
        cached = await self._remembered(customer_id, object_id)
        if cached is not None:
            return cached
        request = ObjectResolveRequestModel(
            identity_token=identity_token,
            correlation_id=uuid7(),
            object_id=object_id,
        )
        try:
            response = await self._nc.request(
                subject=Subjects.hub_object_resolve(),
                message=request,
                response_type=ObjectResolveResponseModel,
                timeout=timedelta(seconds=self._timeout),
            )
        except RequestError as exc:
            raise ResolveObjectError(f"object resolve request failed: {exc}") from exc
        if not response.success:
            # a rejected resolve is never cached: the object may be committed
            # later, or the token refreshed. surface the hub's reason.
            raise ResolveObjectError(f"object resolve rejected: {response.error_code}: {response.error_message}")
        if response.s3_key is None:
            raise ResolveObjectError("object resolve reply reported success but carried no s3_key")
        handle = ObjectHandle(
            object_id=object_id,
            s3_key=response.s3_key,
            mime_type=response.mime_type or "application/octet-stream",
            size_bytes=response.size_bytes or 0,
            summary=None,
            category=None,
        )
        await self._remember(customer_id, handle)
        _log.debug(
            "resolved object id to stored key",
            # object_id is the safe handle; the s3_key is NOT logged (it embeds
            # the customer_id + owning-context label).
            extra={"extra_data": {"object_id": str(object_id)}},  # convert at border: resolve log extra_data
        )
        return handle

    async def _remembered(self, customer_id: UUID, object_id: UUID) -> ObjectHandle | None:
        """read the resolved handle back out of whichever cache this pod has.

        :param customer_id: the VERIFIED owning customer
        :ptype customer_id: UUID
        :param object_id: the object id being resolved
        :ptype object_id: UUID
        :return: the cached handle, or ``None`` on a miss
        :rtype: ObjectHandle | None
        """
        if self._resolution_cache is not None:
            return await self._resolution_cache.lookup(customer_id, object_id)
        return self._cache.get((customer_id, object_id))

    async def _remember(self, customer_id: UUID, handle: ObjectHandle) -> None:
        """record a freshly-resolved handle in whichever cache this pod has.

        :param customer_id: the VERIFIED owning customer
        :ptype customer_id: UUID
        :param handle: the resolved handle
        :ptype handle: ObjectHandle
        :return: nothing
        :rtype: None
        """
        if self._resolution_cache is not None:
            await self._resolution_cache.remember(customer_id, handle)
            return
        self._cache_put((customer_id, handle.object_id), handle)

    def _cache_put(self, key: tuple[UUID, UUID], handle: ObjectHandle) -> None:
        """Store ``handle`` under ``key``, evicting the oldest entry when full.

        :param key: the VERIFIED ``(customer_id, object_id)`` cache key
        :ptype key: tuple[UUID, UUID]
        :param handle: the resolved handle to cache
        :ptype handle: ObjectHandle
        :return: nothing
        :rtype: None
        """
        if key in self._cache:
            return
        if len(self._cache) >= self._cache_max:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = handle
