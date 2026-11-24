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
- :class:`AccessShapeInterner` -- one canonical :class:`AccessShape` per equal shape, so a process
  holding many people holds only their few distinct shapes.
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
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final
from uuid import UUID

from threetears.core.frozen_state import frozen
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
    "AccessShapeInterner",
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


#: how many distinct shapes an :class:`AccessShapeInterner` holds before it starts afresh
_DEFAULT_MAX_INTERNED_SHAPES: Final = 50_000


class AccessShapeInterner:
    """one canonical :class:`AccessShape` per equal shape: equal shapes in, the identical object out.

    A replica keeping each person's shape keeps far fewer distinct shapes than people (measured, 50k
    people over 500 shapes: 10.2 MB interned, 57.3 MB not), so every equal shape is handed back as
    the one object first seen.

    The canonical shapes are held as immutable chunks (read-only snapshots keyed by the shape itself),
    oldest first, and the tuple of them is replaced whole, never changed in place. A new distinct
    shape appends a one-entry chunk and a chunk no bigger than the one after it is merged into it, so
    there are O(log shapes) chunks and each shape is copied O(log shapes) times: one whole snapshot
    copied per new shape was 30 s at 50k distinct, the chunks take well under a second. A hit is a
    plain read, newest chunk first, with no lock; only a new shape takes the lock. Past
    ``max_shapes`` it starts afresh (an earlier shape is then simply not shared with later ones;
    nothing is ever wrong, only less shared). Thread-safe.

    :param max_shapes: the most distinct shapes held
    :ptype max_shapes: int
    :raises ValueError: when ``max_shapes`` is below 1
    """

    def __init__(self, *, max_shapes: int = _DEFAULT_MAX_INTERNED_SHAPES) -> None:
        if max_shapes < 1:
            raise ValueError(f"a shape interner holds at least one shape, not {max_shapes}")
        self._max_shapes = max_shapes
        self._chunks: tuple[Mapping[AccessShape, AccessShape], ...] = ()
        self._held = 0
        self._lock = threading.Lock()

    @property
    def held(self) -> int:
        """how many distinct shapes are held.

        :return: the count
        :rtype: int
        """
        return self._held

    @property
    def chunks(self) -> int:
        """how many immutable chunks hold the shapes: O(log held).

        :return: the count
        :rtype: int
        """
        return len(self._chunks)

    def intern(self, shape: AccessShape) -> AccessShape:
        """the canonical shape equal to ``shape``: the one held, or ``shape`` itself, held from now on.

        :param shape: the shape
        :ptype shape: AccessShape
        :return: the canonical shape
        :rtype: AccessShape
        """
        held = _find(self._chunks, shape)
        if held is not None:
            return held
        with self._lock:
            chunks = self._chunks
            held = _find(chunks, shape)
            if held is not None:
                return held
            count = self._held
            if count >= self._max_shapes:
                chunks, count = (), 0
            self._chunks = _with_chunk(chunks, shape, shape)
            self._held = count + 1
        return shape


def _find[K, V](chunks: tuple[Mapping[K, V], ...], key: K) -> V | None:
    """``key``'s value in ``chunks``, newest chunk first.

    :param chunks: immutable chunks, oldest first
    :ptype chunks: tuple[Mapping[K, V], ...]
    :param key: the key
    :ptype key: K
    :return: its value, or ``None``
    :rtype: V | None
    """
    for chunk in reversed(chunks):
        held = chunk.get(key)
        if held is not None:
            return held
    return None


def _with_chunk[K, V](chunks: tuple[Mapping[K, V], ...], key: K, value: V) -> tuple[Mapping[K, V], ...]:
    """``chunks`` with ``key`` added as the newest one-entry chunk, a chunk no bigger than the one after it
    merged into it: O(log entries) chunks, each entry copied O(log entries) times. Nothing is changed in place.

    :param chunks: immutable chunks, oldest first, none holding ``key``
    :ptype chunks: tuple[Mapping[K, V], ...]
    :param key: the key
    :ptype key: K
    :param value: its value
    :ptype value: V
    :return: the next chunks
    :rtype: tuple[Mapping[K, V], ...]
    """
    chunks = (*chunks, frozen(((key, value),)))
    while len(chunks) > 1 and len(chunks[-2]) <= len(chunks[-1]):
        chunks = (*chunks[:-2], frozen({**chunks[-2], **chunks[-1]}))
    return chunks


@dataclass(frozen=True, slots=True)
class _Generations:
    """a :class:`ShapeActors`' held shapes by actor: the young generation and the one before it, replaced whole.

    :ivar young: the shapes named since the last turn, as immutable chunks oldest first
    :ivar young_held: how many shapes ``young`` holds
    :ivar old: the generation before, dropped whole at the next turn
    :ivar old_held: how many shapes ``old`` holds
    """

    young: tuple[Mapping[UUID, AccessShape], ...] = ()
    young_held: int = 0
    old: tuple[Mapping[UUID, AccessShape], ...] = ()
    old_held: int = 0

    def find(self, actor: UUID) -> tuple[AccessShape | None, bool]:
        """the actor's shape, and whether it was in the young generation.

        :param actor: the actor
        :ptype actor: UUID
        :return: the shape (``None`` when not held) and whether it is young
        :rtype: tuple[AccessShape | None, bool]
        """
        young = _find(self.young, actor)
        if young is not None:
            return young, True
        return _find(self.old, actor), False

    def with_young(self, actor: UUID, shape: AccessShape, max_shapes: int) -> _Generations:
        """these generations with ``actor`` named now, holding at most ``max_shapes``.

        The young generation turns old once it holds half of ``max_shapes`` (the old one dropped
        whole), so between half and all of ``max_shapes`` are held and every shape dropped was named
        less recently than every shape kept.

        :param actor: an actor not in the young generation
        :ptype actor: UUID
        :param shape: its shape
        :ptype shape: AccessShape
        :param max_shapes: the most shapes held
        :ptype max_shapes: int
        :return: the next generations
        :rtype: _Generations
        """
        now = self
        if now.young_held >= max(1, max_shapes // 2):
            now = _Generations(old=now.young, old_held=now.young_held)
        if now.young_held + now.old_held >= max_shapes:
            now = _Generations(young=now.young, young_held=now.young_held)
        return _Generations(
            young=_with_chunk(now.young, actor, shape),
            young_held=now.young_held + 1,
            old=now.old,
            old_held=now.old_held,
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
    ``max_shapes`` in two generations of immutable chunks (:class:`_Generations`, replaced whole): a
    shape named or read is in the young one, and when it holds half of ``max_shapes`` it turns old and
    the old one is dropped whole, so the least recently used go first. A dropped shape's actor is
    refused (:class:`LookupError`) until :meth:`actor_for` names it again. A read of a young shape is a
    plain read; naming a shape anew, or reading an old one, takes the lock. Thread-safe.

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
        self._shapes = _Generations()
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
        self._touch(actor, shape)
        return actor

    def _touch(self, actor: UUID, shape: AccessShape) -> None:
        if self._shapes.find(actor)[1]:
            return
        with self._lock:
            if not self._shapes.find(actor)[1]:
                self._shapes = self._shapes.with_young(actor, shape, self._max_shapes)

    def _shape_of(self, actor_id: UUID) -> AccessShape:
        shape, young = self._shapes.find(actor_id)
        if shape is None:
            raise _ShapeActorForgotten(f"shape actor {actor_id} is not held; name its shape again before asking")
        if not young:
            self._touch(actor_id, shape)
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
