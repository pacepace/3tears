"""Consumer-side helpers: stream a stored object back down, or presign it.

A consuming tool -- one that needs the bytes of a stored artifact (re-read a
prior scan, render from a saved capture) or a delivery URL for one -- calls
these from inside its ``execute``. They are the read-side mirror of
:mod:`~threetears.agent.tools.produce`.

The object store and the VERIFIED caller identity both ride on the per-call
:class:`~threetears.agent.tools.call_scope.ToolCallScope` the tool server
installs around every dispatch. These helpers read them through
:func:`~threetears.agent.tools.call_scope.current_scope`, ASSERT the key sits
under the verified customer's prefix (tenant isolation), then stream / presign
via the injected store. The bytes stream straight from the store to the tool;
they never cross NATS or touch the agent.

Fail-closed -- raising :class:`ConsumeObjectError` -- when invoked outside a
call scope, when the pod was wired with no object store, when the context
carries no verified customer, or when the key is not owned by that customer. A
tool must never read or deliver an object outside its tenant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, AsyncIterator
from uuid import UUID

from threetears.observe import get_logger

from threetears.agent.tools.call_scope import current_scope

if TYPE_CHECKING:
    from threetears.media.contracts import ObjectHandle, ObjectStore

__all__ = [
    "ConsumeObjectError",
    "open_object_stream",
    "presigned_object_url",
    "resolve_object",
]

_log = get_logger(__name__)

_DEFAULT_PRESIGN_TTL_SECONDS = 300


class ConsumeObjectError(RuntimeError):
    """A consuming tool could not read or deliver an object from the store.

    Carries a structural reason only (no call scope / no object store / no
    verified customer / cross-tenant key) -- never object bytes or key material.
    """


def _scoped_store_for(s3_key: str) -> "ObjectStore":
    """Resolve the pod's object store for ``s3_key`` under the verified tenant.

    The single fail-closed gate every consume path passes through. It requires,
    in order: a call scope, a wired object store, a verified ``customer_id`` on
    the call context, and that ``s3_key`` sit under that customer's
    ``<customer_id>/`` prefix (the scope-first key layout, ``keys.py``).

    The prefix check is the tenant-isolation control. A consuming tool is handed
    a key from one of two sources -- a produced :class:`ObjectHandle`, or an id
    resolved via the hub resolver -- and must never read another customer's
    object regardless of which. It is asserted here even when the key came from
    a hub resolve that already checked ownership: defense in depth on the
    direct-handle path, and it mirrors exactly the ``<customer_id>/`` prefix the
    producer wrote (``build_object_key``) and the hub catalog asserts on commit.

    :param s3_key: the object key to read or deliver
    :ptype s3_key: str
    :return: the injected object store, validated for this tenant + key
    :rtype: ObjectStore
    :raises ConsumeObjectError: no scope / no store / no verified customer, or
        the key is not owned by the verified customer
    """
    scope = current_scope()
    if scope is None:
        raise ConsumeObjectError(
            "object consume helper called outside a ToolServer call scope; a "
            "consuming tool runs inside enter_call_scope"
        )
    store = scope.object_store
    if store is None:
        raise ConsumeObjectError(
            "the current call scope carries no object store; the tool pod was "
            "not wired with one (no object store configured)"
        )
    customer_id = scope.context.customer_id
    if customer_id is None:
        raise ConsumeObjectError(
            "the call context carries no verified customer_id; refusing to read an untenanted object"
        )
    prefix = f"{customer_id}/"
    if not s3_key.startswith(prefix):
        # tenant-isolation breach: the key names a DIFFERENT customer's prefix
        # than the verified caller. log the verified customer + the claimed
        # leading segment for forensics -- never the whole key (it embeds the
        # foreign customer_id + owning-context label).
        claimed = s3_key.split("/", 1)[0]
        _log.warning(
            "refusing cross-tenant object access",
            extra={
                "extra_data": {
                    "verified_customer_id": str(customer_id),  # convert at border: security log extra_data
                    "claimed_prefix": claimed,  # convert at border: security log extra_data
                }
            },
        )
        raise ConsumeObjectError("the object key is not owned by the verified customer; refusing cross-tenant access")
    return store


def open_object_stream(s3_key: str) -> AsyncIterator[bytes]:
    """Open a stored object for streaming read under the verified tenant.

    Validates the call scope + tenant ownership of ``s3_key`` EAGERLY (at call
    time, before the iterator is handed back), then returns the store's byte
    stream. The bytes stream straight from the object store to the caller in
    chunks -- never buffered whole, never across NATS.

    Eager validation means a caller that never iterates still fails closed on a
    bad scope or a cross-tenant key; the store reference is captured here, so
    iteration is unaffected by a later scope reset.

    :param s3_key: tenant-scoped key of the object to read (from a produced
        :class:`ObjectHandle`, or resolved from an object id via the resolver)
    :ptype s3_key: str
    :return: async iterator over the object's bytes, in chunks
    :rtype: AsyncIterator[bytes]
    :raises ConsumeObjectError: no scope / no store / no verified customer, or
        the key is not owned by the verified customer
    """
    store = _scoped_store_for(s3_key)
    _log.debug("opening object stream for read")
    return store.open_read(s3_key)


async def presigned_object_url(s3_key: str, *, expires_in: int = _DEFAULT_PRESIGN_TTL_SECONDS) -> str:
    """Presigned GET URL for a stored object, under the verified tenant.

    The delivery path: hand the caller a short-lived URL so a client fetches the
    bytes directly from the store -- the bytes never cross the agent. Same
    tenant-ownership assertion as :func:`open_object_stream`.

    :param s3_key: tenant-scoped key of the object to deliver
    :ptype s3_key: str
    :param expires_in: URL validity in seconds
    :ptype expires_in: int
    :return: a presigned GET URL for the object
    :rtype: str
    :raises ConsumeObjectError: no scope / no store / no verified customer, or
        the key is not owned by the verified customer
    """
    store = _scoped_store_for(s3_key)
    url = await store.presigned_get_url(s3_key, expires_in=expires_in)
    _log.debug("presigned object for delivery")
    return url


async def resolve_object(object_id: UUID) -> "ObjectHandle":
    """Resolve an object id to a handle carrying its stored key, tenant-safely.

    The tenant-safety keystone of the consume path. Reads the pod's resolver +
    the VERIFIED identity off the current call scope and asks the hub to map
    ``object_id`` to its key under the verified customer. A bare object id is an
    untrusted input (an LLM tool arg, a cross-turn reference); only the hub --
    which owns the objects table -- can safely decide whether the verified
    customer owns it, so the resolve is where cross-tenant access is refused for
    the id-not-handle path. The caller then hands ``handle.s3_key`` to
    :func:`open_object_stream` or :func:`presigned_object_url` (which re-assert
    the tenant prefix, defense in depth).

    Authentication uses the invoking agent's ``identity_token`` from the call
    context -- the hub verifies it and derives the customer from the verified
    claim -- so this pure-``threetears`` pod needs no hub session of its own.

    :param object_id: the object id to resolve (untrusted; the hub authorizes it
        against the verified customer)
    :ptype object_id: UUID
    :return: a handle whose ``s3_key`` locates the object for streaming/delivery
        (``summary`` / ``category`` are ``None`` -- resolve returns only the
        storage-locating fields)
    :rtype: ObjectHandle
    :raises ConsumeObjectError: no scope / no resolver wired / no verified
        customer / no identity_token to authenticate the resolve
    :raises ResolveObjectError: the hub rejected the resolve (identity
        unverified, or the customer does not own the object) or it failed in
        transit -- raised by the resolver and propagated
    """
    scope = current_scope()
    if scope is None:
        raise ConsumeObjectError(
            "object consume helper called outside a ToolServer call scope; a "
            "consuming tool runs inside enter_call_scope"
        )
    resolver = scope.object_resolver
    if resolver is None:
        raise ConsumeObjectError(
            "the current call scope carries no object resolver; the tool pod was not wired with one (no NATS client)"
        )
    customer_id = scope.context.customer_id
    if customer_id is None:
        raise ConsumeObjectError(
            "the call context carries no verified customer_id; refusing to resolve an untenanted object"
        )
    identity_token = scope.context.identity_token
    if identity_token is None:
        raise ConsumeObjectError("the call context carries no identity_token; cannot authenticate the object resolve")
    return await resolver.resolve(object_id, customer_id=customer_id, identity_token=identity_token)
