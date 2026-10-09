"""evict what an rbac write changed, from the cache the writer will ask next.

a helper that WRITES a ``group_members`` or ``role_assignments`` row while it
holds an :class:`AclCache` has just made that cache wrong, and the very next
evaluation -- usually the caller's own next request, on the same process --
would be answered from the stale entry: a denial straight after a grant, or a
revoked actor still acting.

every other process learns of the write from its row broadcast and the
table's write generation (``threetears.agent.acl.generation_follow``), and so
does the writer's own cache when it is followed on the registry the write went
through. this evicts the writer's cache at once all the same, because the
cache the helper was handed need not be bound to that registry, and an entry
computed from a read that began before the write must not be the one answered.

a membership change drops that actor's membership entry, and an assignment
change drops every assignment-layer entry naming the group: the same two
:class:`AclCache` methods a row broadcast for those rows reaches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from threetears.agent.acl.cache import AclCache

__all__ = ["evict_after_rbac_write"]


def evict_after_rbac_write(
    cache: AclCache,
    *,
    member_actors: Sequence[tuple[Literal["user", "agent"], UUID]] = (),
    group_ids: Sequence[UUID] = (),
) -> None:
    """evict the entries an rbac write made stale from the writer's own cache.

    call it after the write has committed, naming what the write changed:

    - a ``group_members`` row added or removed for a user or agent -> that
      actor in ``member_actors``
    - a ``role_assignments`` row added or removed, or a ``groups`` row
      created -> that group in ``group_ids``

    naming nothing is a no-op, which is what lets an idempotent ensure call
    this with only what it actually wrote.

    :param cache: the cache the writing process evaluates against
    :ptype cache: AclCache
    :param member_actors: ``(actor_type, actor_id)`` pairs whose group
        membership the write changed
    :ptype member_actors: Sequence[tuple[Literal["user", "agent"], UUID]]
    :param group_ids: groups whose role assignments the write changed, or
        that the write created
    :ptype group_ids: Sequence[UUID]
    :return: nothing
    :rtype: None
    """
    for actor_type, actor_id in member_actors:
        cache.invalidate_membership_for_actor(actor_type, actor_id)
    for group_id in group_ids:
        cache.invalidate_group(group_id)
