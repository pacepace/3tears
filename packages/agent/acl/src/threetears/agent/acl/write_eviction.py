"""evict what an rbac write changed, from the cache that will answer the next question.

an :class:`AclCache` holds an actor's memberships and a group's per-namespace
contribution for its whole TTL. a helper that WRITES a ``group_members`` or
``role_assignments`` row while it holds that cache has therefore just made
the cache wrong, and the very next evaluation -- usually the caller's own
next request, on the same pod -- is answered from the stale entry. for a
grant that means a denial for up to the TTL immediately after the grant
was made; for a revocation it means the revoked actor keeps acting.

this module is the one rule every such helper follows, so it cannot be
restated slightly differently in each: evict locally first (the local
answer must be right even with no broker at all), then broadcast on the
invalidation bus when a publisher is available so every other pod evicts
the same entries.

the eviction is exactly what the bus subscriber does for the same payload
(:func:`~threetears.agent.acl.invalidation_bus.subscribe_acl_invalidation`):
a membership change drops that actor's membership entry, and an assignment
change drops every assignment-layer entry naming the group. the local
eviction and the remote one cannot drift because they call the same two
:class:`AclCache` methods.

this module does not import the bus at load time. the bus needs the NATS
client (the ``3tears-agent-acl[bus]`` extra) and a consumer with no broker
must still be able to evict locally; the publish helpers are imported only
when a publisher was actually handed in, which is also the only case in
which the extra is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from threetears.observe import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from threetears.agent.acl.cache import AclCache
    from threetears.agent.acl.invalidation_bus import AclInvalidationPublisher

__all__ = ["evict_after_rbac_write"]

log = get_logger(__name__)


async def evict_after_rbac_write(
    cache: AclCache,
    publisher: AclInvalidationPublisher | None = None,
    *,
    member_actors: Sequence[tuple[Literal["user", "agent"], UUID]] = (),
    group_ids: Sequence[UUID] = (),
) -> None:
    """evict the entries an rbac write made stale, locally and then across pods.

    call it after the write has committed, naming what the write changed:

    - a ``group_members`` row added or removed for a user or agent -> that
      actor in ``member_actors``
    - a ``role_assignments`` row added or removed, or a ``groups`` row
      created -> that group in ``group_ids``

    naming nothing is a no-op, which is what lets an idempotent ensure call
    this with only what it actually wrote: an ensure that found every row
    already present changed nothing, evicts nothing and publishes nothing,
    so running it on every request costs neither cache churn nor bus traffic.

    a publish failure is logged and does not raise. by the time this runs the
    write has committed and the local cache is already correct, so raising
    would report a failure for work that succeeded -- and a retry would find
    every row present, write nothing, and so publish nothing either. what
    the failure costs is bounded: other pods serve their cached entry until
    it ages out, which is the no-broker behaviour, and the warning names the
    entries so an operator can see which grant is lagging.

    :param cache: the cache the writing process evaluates against
    :ptype cache: AclCache
    :param publisher: the invalidation-bus publisher, or ``None`` for a
        process with no broker (local eviction only; other processes fall
        back to TTL expiry)
    :ptype publisher: AclInvalidationPublisher | None
    :param member_actors: ``(actor_type, actor_id)`` pairs whose group
        membership the write changed
    :ptype member_actors: Sequence[tuple[Literal["user", "agent"], UUID]]
    :param group_ids: groups whose role assignments the write changed, or
        that the write created
    :ptype group_ids: Sequence[UUID]
    :return: nothing
    :rtype: None
    :raises ImportError: when a publisher is supplied but the bus extra
        (``3tears-agent-acl[bus]``) is not installed -- a wiring error that
        must not degrade silently into TTL-only eviction
    """
    for actor_type, actor_id in member_actors:
        cache.invalidate_membership_for_actor(actor_type, actor_id)
    for group_id in group_ids:
        cache.invalidate_group(group_id)
    if publisher is not None and (member_actors or group_ids):
        await _broadcast(publisher, member_actors=member_actors, group_ids=group_ids)


async def _broadcast(
    publisher: AclInvalidationPublisher,
    *,
    member_actors: Sequence[tuple[Literal["user", "agent"], UUID]],
    group_ids: Sequence[UUID],
) -> None:
    """publish one invalidation per evicted entry, logging rather than raising on failure.

    :param publisher: the invalidation-bus publisher
    :ptype publisher: AclInvalidationPublisher
    :param member_actors: actors whose membership entries were evicted
    :ptype member_actors: Sequence[tuple[Literal["user", "agent"], UUID]]
    :param group_ids: groups whose assignment entries were evicted
    :ptype group_ids: Sequence[UUID]
    :return: nothing
    :rtype: None
    :raises ImportError: when the bus extra is not installed
    """
    # deferred: the bus needs the NATS client, which only a process that
    # handed in a publisher is required to have installed.
    from threetears.agent.acl.invalidation_bus import (  # noqa: PLC0415 -- optional [bus] extra
        publish_assignment_invalidation,
        publish_membership_invalidation,
    )

    try:
        for actor_type, actor_id in member_actors:
            await publish_membership_invalidation(publisher, actor_type=actor_type, actor_id=actor_id)
        for group_id in group_ids:
            await publish_assignment_invalidation(publisher, group_id=group_id)
    except (
        Exception
    ):  # prawduct:allow prawduct/broad-except -- the write committed and the local cache is correct; see docstring
        log.warning(
            "rbac write committed and evicted locally, but the cross-pod invalidation broadcast failed; "
            "other pods serve their cached decision until it expires",
            exc_info=True,
            extra={
                "extra_data": {
                    "member_actors": [f"{actor_type}:{actor_id}" for actor_type, actor_id in member_actors],
                    "group_ids": [str(group_id) for group_id in group_ids],
                }
            },
        )
