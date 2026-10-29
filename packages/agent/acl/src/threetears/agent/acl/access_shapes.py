"""a person's access, as the evaluator sees it, is a function of their membership *shape*.

The access evaluator reads a person through one thing only: their direct membership rows
(:meth:`MembershipLoader.load_for_user` -- each row's ``group_id`` and the membership's
``customer_id``, which the cross-customer wall reads). Roles are assigned to groups, never to people
(:attr:`RoleAssignment.group_id`), and everything past the direct groups -- nesting, assignments,
roles, namespaces, the agent side -- is shared by everyone. So two people whose direct memberships
are the same get the same answer to every access question, and an answer derived once for that set
serves them all. At a sign-in surge that is the difference between one derivation per person and
one per distinct set of groups.

- :class:`AccessShape` -- the sorted set of ``(group_id, membership customer_id)`` a person's rows
  name, and a stable digest of it.
- :class:`ShapeActors` -- a :class:`MembershipLoader` wrapping the real one, under which each shape
  has a synthetic actor whose memberships ARE the shape. A derivation evaluates as that actor, so
  its answer depends on the shape and nothing else of any person: no race between reading a person's
  memberships and deriving with them, and the answer is right for everyone with the shape.
- :func:`access_footprint` -- the shared rows an answer for a shape can reach (groups, roles,
  agents), which is what a change must name to make the answer stale
  (:mod:`threetears.agent.acl.shape_answers`).
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final
from uuid import UUID

from threetears.observe import get_logger

from threetears.agent.acl.cache import ActorMembershipKey
from threetears.agent.acl.evaluator import _parent_rows
from threetears.agent.acl.loader import HeldAssignmentsLoader, MembershipLoader, ParentGroupsLoader
from threetears.agent.acl.types import MAX_GROUP_MEMBERSHIP_DEPTH, GroupMembership, MemberType

if TYPE_CHECKING:
    from threetears.agent.acl.cache import AclCache

__all__ = [
    "AccessFootprint",
    "AccessShape",
    "ShapeActors",
    "access_footprint",
]

log = get_logger(__name__)

#: how many shapes a :class:`ShapeActors` holds an actor for unless told otherwise. A shape is a few
#: group ids, and a customer has far fewer distinct sets of groups than people
_DEFAULT_MAX_SHAPE_ACTORS: Final = 100_000

#: the first bytes of every synthetic actor id, in a version-8 UUID; real principals are version 4 or 7
_ACTOR_MARKER: Final = bytes.fromhex("a55e5ac7")


@dataclass(frozen=True, slots=True)
class AccessShape:
    """the direct memberships a person's access is a function of: sorted ``(group_id, customer_id)``.

    :ivar memberships: each direct membership's group and the membership's customer (``None`` for a
        platform-scoped row), sorted and without duplicates
    """

    memberships: tuple[tuple[UUID, UUID | None], ...]

    @classmethod
    def of(cls, rows: Iterable[GroupMembership]) -> AccessShape:
        """the shape a person's membership rows name; rows that are not a user's are ignored.

        :param rows: the person's rows, as :meth:`MembershipLoader.load_for_user` returns them
        :ptype rows: Iterable[GroupMembership]
        :return: the shape
        :rtype: AccessShape
        """
        pairs = {(row.group_id, row.customer_id) for row in rows if row.member_type == MemberType.USER}
        return cls(memberships=tuple(sorted(pairs, key=lambda pair: (pair[0].hex, pair[1].hex if pair[1] else ""))))

    @property
    def digest(self) -> str:
        """a stable digest of the shape: equal shapes, equal digests, in every process.

        :return: sha256 hex
        :rtype: str
        """
        text = "\x1e".join(f"{group.hex}\x1f{customer.hex if customer else ''}" for group, customer in self.memberships)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @property
    def group_ids(self) -> frozenset[UUID]:
        """the groups the shape is a direct member of.

        :return: the group ids
        :rtype: frozenset[UUID]
        """
        return frozenset(group for group, _customer in self.memberships)

    def rows_for(self, actor_id: UUID) -> tuple[GroupMembership, ...]:
        """the shape's memberships as rows naming ``actor_id`` as the user member.

        :param actor_id: the actor the rows name
        :ptype actor_id: UUID
        :return: the rows
        :rtype: tuple[GroupMembership, ...]
        """
        return tuple(
            GroupMembership(group_id=group, member_type=MemberType.USER, member_id=actor_id, customer_id=customer)
            for group, customer in self.memberships
        )


def _shape_actor_id(shape: AccessShape) -> UUID:
    """the synthetic actor id of ``shape``: the same in every process, and never a real principal's.

    :param shape: the shape
    :ptype shape: AccessShape
    :return: a version-8 UUID beginning with the actor marker
    :rtype: UUID
    """
    raw = bytearray(hashlib.sha256(b"access-shape-actor\x00" + shape.digest.encode("ascii")).digest()[:16])
    raw[0:4] = _ACTOR_MARKER
    raw[6] = (raw[6] & 0x0F) | 0x80
    raw[8] = (raw[8] & 0x3F) | 0x80
    return UUID(bytes=bytes(raw))


def _is_shape_actor(actor_id: UUID) -> bool:
    """whether ``actor_id`` is a synthetic shape actor rather than a principal.

    :param actor_id: the id
    :ptype actor_id: UUID
    :return: ``True`` for a shape actor
    :rtype: bool
    """
    return actor_id.version == 8 and actor_id.bytes[:4] == _ACTOR_MARKER


class _ShapeActorForgotten(LookupError):
    """a shape actor's memberships were asked for after its shape was let go.

    Raised rather than answered from the database, which holds no row for a synthetic actor: an empty
    answer would be read as "a member of nothing" and kept.
    """


class ShapeActors:
    """the membership loader the evaluator reads through, with an actor per shape beside the real people.

    Every question about a real principal goes to ``inner``. A shape actor (:meth:`actor_for`) is
    answered from its shape, which never changes: an actor's memberships are its shape by definition,
    so a membership-layer entry the access cache keeps for it is never stale. Shapes are held up to
    ``max_shapes``, least recently used dropped first; a dropped shape's actor is refused
    (:class:`LookupError`) until :meth:`actor_for` names it again. Thread-safe.

    :param inner: the real membership loader
    :ptype inner: MembershipLoader
    :param max_shapes: the most shapes held at once
    :ptype max_shapes: int
    :raises ValueError: when ``max_shapes`` is below 1
    """

    def __init__(self, inner: MembershipLoader, *, max_shapes: int = _DEFAULT_MAX_SHAPE_ACTORS) -> None:
        if max_shapes < 1:
            raise ValueError(f"a shape-actor loader holds at least one shape, not {max_shapes}")
        self._inner = inner
        self._max_shapes = max_shapes
        self._shapes: OrderedDict[UUID, AccessShape] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def inner(self) -> MembershipLoader:
        """the real membership loader.

        :return: it
        :rtype: MembershipLoader
        """
        return self._inner

    def actor_for(self, shape: AccessShape) -> UUID:
        """the actor whose memberships are ``shape``, held from now on.

        :param shape: the shape
        :ptype shape: AccessShape
        :return: its actor id
        :rtype: UUID
        """
        actor = _shape_actor_id(shape)
        with self._lock:
            self._shapes[actor] = shape
            self._shapes.move_to_end(actor)
            while len(self._shapes) > self._max_shapes:
                self._shapes.popitem(last=False)
        return actor

    def _shape_of(self, actor_id: UUID) -> AccessShape:
        with self._lock:
            shape = self._shapes.get(actor_id)
            if shape is not None:
                self._shapes.move_to_end(actor_id)
        if shape is None:
            raise _ShapeActorForgotten(f"shape actor {actor_id} is not held; name its shape again before asking")
        return shape

    async def load_for_user(self, user_id: UUID) -> tuple[GroupMembership, ...]:
        """a person's rows from ``inner``; a shape actor's from its shape.

        :param user_id: the person or shape actor
        :ptype user_id: UUID
        :return: the rows
        :rtype: tuple[GroupMembership, ...]
        :raises LookupError: for a shape actor not held (it is never read from the database as a member of nothing)
        """
        if _is_shape_actor(user_id):
            return self._shape_of(user_id).rows_for(user_id)
        return tuple(await self._inner.load_for_user(user_id))

    async def load_for_agent(self, agent_id: UUID) -> tuple[GroupMembership, ...]:
        """an agent's rows, from ``inner``.

        :param agent_id: the agent
        :ptype agent_id: UUID
        :return: the rows
        :rtype: tuple[GroupMembership, ...]
        """
        return tuple(await self._inner.load_for_agent(agent_id))

    async def load_for_group(self, group_id: UUID) -> tuple[GroupMembership, ...]:
        """a group's parent rows, from ``inner``.

        :param group_id: the group
        :ptype group_id: UUID
        :return: the rows
        :rtype: tuple[GroupMembership, ...]
        """
        return tuple(await self._inner.load_for_group(group_id))

    async def load_for_groups(self, group_ids: tuple[UUID, ...]) -> tuple[GroupMembership, ...]:
        """many groups' parent rows: one read when ``inner`` can, else one per group.

        :param group_ids: the groups
        :ptype group_ids: tuple[UUID, ...]
        :return: every parent row naming one of them
        :rtype: tuple[GroupMembership, ...]
        """
        if isinstance(self._inner, ParentGroupsLoader):
            return tuple(await self._inner.load_for_groups(group_ids))
        rows: list[GroupMembership] = []
        for group_id in group_ids:
            rows.extend(await self._inner.load_for_group(group_id))
        return tuple(rows)


#: the dep every answer has on every role when its roles cannot be listed
ANY_ROLE: Final = "r:*"


def group_dep(group_id: UUID) -> str:
    """the dep a change to a group's rows, nesting or assignments names.

    :param group_id: the group
    :ptype group_id: UUID
    :return: the dep
    :rtype: str
    """
    return f"g:{group_id.hex}"


def role_dep(role_id: UUID) -> str:
    """the dep a change to a role's row names.

    :param role_id: the role
    :ptype role_id: UUID
    :return: the dep
    :rtype: str
    """
    return f"r:{role_id.hex}"


def agent_dep(agent_id: UUID) -> str:
    """the dep a change to an agent's own membership rows names.

    :param agent_id: the agent
    :ptype agent_id: UUID
    :return: the dep
    :rtype: str
    """
    return f"a:{agent_id.hex}"


def table_dep(table: str) -> str:
    """the dep a change of unknown or table-wide reach names: every answer derived from the table has it.

    :param table: the table
    :ptype table: str
    :return: the dep
    :rtype: str
    """
    return f"t:{table}"


@dataclass(frozen=True, slots=True)
class AccessFootprint:
    """the shared rows an answer for one shape can reach.

    :ivar groups: every group reachable from the shape's and the agents' direct groups, through
        nesting, with no customer filter (a superset only over-invalidates)
    :ivar roles: the roles those groups hold assignments of; ``None`` when they could not be listed,
        which makes every role a dep
    :ivar agents: the signing agents whose own memberships the answer read
    """

    groups: frozenset[UUID]
    roles: frozenset[UUID] | None
    agents: frozenset[UUID]

    def deps(self, tables: Iterable[str]) -> frozenset[str]:
        """the deps an answer with this footprint has, beside a whole-table dep on each of ``tables``.

        :param tables: every table the answer is derived from
        :ptype tables: Iterable[str]
        :return: the deps
        :rtype: frozenset[str]
        """
        roles = (ANY_ROLE,) if self.roles is None else tuple(role_dep(role) for role in self.roles)
        return frozenset(
            (
                *(group_dep(group) for group in self.groups),
                *roles,
                *(agent_dep(agent) for agent in self.agents),
                *(table_dep(table) for table in tables),
            )
        )


async def _agent_rows(cache: AclCache, agent_id: UUID) -> tuple[GroupMembership, ...]:
    """an agent's direct membership rows, through the access cache's membership layer.

    :param cache: the access cache
    :ptype cache: AclCache
    :param agent_id: the agent
    :ptype agent_id: UUID
    :return: the rows
    :rtype: tuple[GroupMembership, ...]
    """
    key = ActorMembershipKey(actor_kind="agent", actor_id=agent_id)
    entry = cache.get_membership(key)
    if entry is not None:
        return entry.memberships
    fence = cache.read_fence()
    rows = tuple(await cache.membership_loader.load_for_agent(agent_id))
    cache.put_membership(key, rows, fence=fence)
    return rows


async def access_footprint(cache: AclCache, *, shape: AccessShape, agent_ids: Collection[UUID]) -> AccessFootprint:
    """the groups, roles and agents an answer for ``shape`` with these signing agents can reach.

    Read through the access cache's own layers (and its loaders on a miss), widening the shape's and
    the agents' direct groups through nesting up to :data:`MAX_GROUP_MEMBERSHIP_DEPTH`, as the
    evaluator does but with no customer filter, then listing the roles the groups hold assignments
    of in one read. Every change that could widen the footprint names a group already in it (a new
    nesting names its child, a new assignment its group), so a footprint read before a change is
    made stale by that change, never silently incomplete.

    :param cache: the access cache the answer is evaluated through
    :ptype cache: AclCache
    :param shape: the shape
    :ptype shape: AccessShape
    :param agent_ids: the agents whose side the answer evaluates
    :ptype agent_ids: Collection[UUID]
    :return: the footprint
    :rtype: AccessFootprint
    """
    direct: dict[UUID, None] = dict.fromkeys(sorted(shape.group_ids, key=lambda group: group.hex))
    for agent_id in agent_ids:
        for row in await _agent_rows(cache, agent_id):
            if row.member_type == MemberType.AGENT and row.member_id == agent_id:
                direct.setdefault(row.group_id, None)
    seen: dict[UUID, None] = dict(direct)
    frontier = tuple(direct)
    for _ in range(MAX_GROUP_MEMBERSHIP_DEPTH - 1):
        if not frontier:
            break
        parents = await _parent_rows(frontier, cache)
        widened: list[UUID] = []
        for group_id in frontier:
            for row in parents[group_id]:
                if row.member_type == MemberType.GROUP and row.member_id == group_id and row.group_id not in seen:
                    seen[row.group_id] = None
                    widened.append(row.group_id)
        frontier = tuple(widened)
    groups = frozenset(seen)
    roles: frozenset[UUID] | None = None
    loader = cache.grant_loader
    if isinstance(loader, HeldAssignmentsLoader):
        held = await loader.load_assignments_held_by_groups(tuple(seen)) if seen else ()
        roles = frozenset(assignment.role_id for assignment in held)
    return AccessFootprint(groups=groups, roles=roles, agents=frozenset(agent_ids))
