"""Publish and subscribe the rbac invalidation subjects, bound to an :class:`AclCache`.

:mod:`threetears.agent.acl.invalidation` defines the wire format; this module moves it. Both
halves live here because they are one contract: a publisher that fires ``assignment.invalidate``
and a subscriber that answers it by calling :meth:`AclCache.invalidate_group` have to agree, and
splitting them across consuming apps is how they stop agreeing.

**Why this exists at all.** An :class:`AclCache` is per-pod and TTL-bounded, so without a bus a
revoked grant keeps authorizing on every pod that has it cached until the entry ages out --
seconds during which a user who has just been removed can still act. The TTL is the floor on how
wrong the answer can be; this is what makes it near-zero instead. A deployment that never wires
the bus is not merely un-optimised, it is one whose revocations do not take effect when they are
issued, which is usually not what "revoke" is taken to mean.

**Publish locally too.** :func:`subscribe_acl_invalidation` binds a plain (non-queue) subscription,
so the publishing pod receives its own broadcast and evicts on the same path as every other pod --
one code path, not a local special case that can drift from the remote one. Callers that cannot
tolerate the round trip (or that must stay correct with the broker down) should ALSO evict locally
before publishing; the operations are idempotent, so doing both is safe.

The publish helpers take a narrow :class:`AclInvalidationPublisher` rather than the whole
``NatsClient``, matching :class:`threetears.nats.kv.JetStreamPublisher`'s reasoning: a signature
naming the entire client forces every test with a recording double through a cast, which puts a
hole in exactly the tests that exist to prove the right thing was published.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel
from threetears.nats import Subjects
from threetears.observe import get_logger

from threetears.agent.acl.invalidation import (
    AssignmentInvalidatePayload,
    MembershipInvalidatePayload,
    RoleInvalidatePayload,
)

if TYPE_CHECKING:
    from threetears.agent.acl.cache import AclCache

__all__ = [
    "AclInvalidationPublisher",
    "AclInvalidationSubscriber",
    "publish_assignment_invalidation",
    "publish_membership_invalidation",
    "publish_role_invalidation",
    "subscribe_acl_invalidation",
]

log = get_logger(__name__)


@runtime_checkable
class AclInvalidationPublisher(Protocol):
    """The slice of :class:`~threetears.nats.NatsClient` an invalidation producer needs."""

    async def publish(self, *, subject: Any, message: BaseModel) -> None: ...


@runtime_checkable
class AclInvalidationSubscriber(Protocol):
    """The slice of :class:`~threetears.nats.NatsClient` an invalidation consumer needs."""

    async def subscribe_typed(self, *, subject: Any, cb: Any, message_type: Any) -> Any: ...


async def publish_membership_invalidation(
    nats: AclInvalidationPublisher | None,
    *,
    actor_type: Literal["user", "agent"],
    actor_id: UUID,
) -> None:
    """Announce that an actor's group membership changed.

    :param nats: the publisher; ``None`` is an explicit no-op for a deployment with no broker,
        which then falls back to TTL expiry.
    :ptype nats: AclInvalidationPublisher | None
    :param actor_type: ``"user"`` or ``"agent"``. Typed as the payload's own ``Literal`` rather
        than ``str``: declaring ``str`` and silencing the mismatch with a ``type: ignore`` removed
        the only static signal that would catch a bad value at a call site, and turned it into a
        pydantic ``ValidationError`` raised AFTER the caller's DB write had already committed.
    :ptype actor_type: Literal["user", "agent"]
    :param actor_id: the actor whose memberships changed.
    :ptype actor_id: UUID
    :return: nothing
    :rtype: None
    """
    if nats is None:
        return
    payload = MembershipInvalidatePayload(actor_type=actor_type, actor_id=actor_id)
    await nats.publish(subject=Subjects.acl_invalidate("membership"), message=payload)


async def publish_assignment_invalidation(nats: AclInvalidationPublisher | None, *, group_id: UUID) -> None:
    """Announce that a group's role assignments changed.

    :param nats: the publisher; ``None`` is an explicit no-op.
    :ptype nats: AclInvalidationPublisher | None
    :param group_id: the group whose assignments changed.
    :ptype group_id: UUID
    :return: nothing
    :rtype: None
    """
    if nats is None:
        return
    await nats.publish(
        subject=Subjects.acl_invalidate("assignment"),
        message=AssignmentInvalidatePayload(group_id=group_id),
    )


async def publish_role_invalidation(nats: AclInvalidationPublisher | None) -> None:
    """Announce that a role DEFINITION changed (its permissions were edited).

    Subscribers clear every layer: any group holding an assignment to that role may now decide
    differently, and there is no index from role to cached entries to do better.

    :param nats: the publisher; ``None`` is an explicit no-op.
    :ptype nats: AclInvalidationPublisher | None
    :return: nothing
    :rtype: None
    """
    if nats is None:
        return
    await nats.publish(subject=Subjects.acl_invalidate("role"), message=RoleInvalidatePayload())


async def subscribe_acl_invalidation(nats: AclInvalidationSubscriber, cache: AclCache) -> list[Any]:
    """Bind ``cache`` to the three invalidation subjects. Returns the subscriptions.

    Plain subscriptions, NOT queue groups: every pod must evict, so every pod must receive. A
    queue group would deliver each broadcast to exactly one member, leaving every other pod
    serving the stale decision it was told to drop -- the precise failure this bus prevents.

    :param nats: the connected client.
    :ptype nats: AclInvalidationSubscriber
    :param cache: the per-pod cache to evict from.
    :ptype cache: AclCache
    :return: the three subscriptions, for the caller to unsubscribe at shutdown.
    :rtype: list[Any]
    """

    async def _on_membership(payload: MembershipInvalidatePayload) -> None:
        cache.invalidate_membership_for_actor(payload.actor_type, payload.actor_id)
        log.debug(
            "acl membership cache invalidated",
            extra={"extra_data": {"actor_type": payload.actor_type, "actor_id": str(payload.actor_id)}},
        )

    async def _on_assignment(payload: AssignmentInvalidatePayload) -> None:
        cache.invalidate_group(payload.group_id)
        log.debug("acl assignment cache invalidated", extra={"extra_data": {"group_id": str(payload.group_id)}})

    async def _on_role(_payload: RoleInvalidatePayload) -> None:
        cache.invalidate_all()
        log.info("acl cache fully invalidated (a role definition changed)")

    bindings: tuple[tuple[Literal["membership", "assignment", "role"], Any, Any], ...] = (
        ("membership", _on_membership, MembershipInvalidatePayload),
        ("assignment", _on_assignment, AssignmentInvalidatePayload),
        ("role", _on_role, RoleInvalidatePayload),
    )
    bound: list[Any] = []
    try:
        for kind, handler, payload_type in bindings:
            bound.append(
                await nats.subscribe_typed(
                    subject=Subjects.acl_invalidate(kind),
                    cb=handler,
                    message_type=payload_type,
                )
            )
    except BaseException:
        # Unwind, or a failed startup orphans the subscriptions that DID bind: the caller receives
        # no handles (the exception propagates instead of a list), so nothing can ever unsubscribe
        # them and they hold the AclCache alive for the process lifetime.
        for subscription in bound:
            try:
                await subscription.unsubscribe()
            except Exception:  # noqa: BLE001 -- the original failure is the one worth raising
                log.warning("could not unwind an acl invalidation subscription", exc_info=True)
        raise
    return bound
