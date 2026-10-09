"""three-layer in-process cache for the rbac evaluator.

the evaluator is pure-functional: it never holds state across calls.
this cache sits in front of the loaders the evaluator depends on so
the production hot path serves authorization decisions from process
memory most of the time.

three explicit layers (do not unify; the layers exist so invalidation
fans out correctly):

- **membership layer** — keyed by ``("user", user_id)`` /
  ``("agent", agent_id)`` / ``("group", group_id)``; value is the tuple
  of membership rows the actor holds. the ``group`` kind holds a child
  group's PARENT groups for the depth-capped group-in-group walk --
  deliberately its own key, never expanded into any actor's entry, so
  changing one group's parents invalidates exactly one key. invalidated
  by membership-change events.
- **assignment-per-namespace layer** — keyed by
  ``(group_id, namespace_id)``; value is the action set that group
  contributes for the specific namespace, plus the trail rows that
  produced it. invalidated by assignment-change events targeting the
  group + namespace, by role-change events affecting any role the
  group holds, and by membership changes that retire the group from
  the actor (handled at the membership layer).
- **assignment-per-type-customer layer** — keyed by
  ``(group_id, namespace_type, customer_id)``; value is the action
  set the group contributes for any namespace of that type within
  that customer (when at least one of its assignments uses the
  ``type_customer`` scope). invalidated by the same triggers as the
  per-namespace layer.

cache is process-local. two pods have independent caches, and nothing
cached here has an age: an entry stays until a write that reaches it is
heard, or the cache stops being trusted.

**row by row, from the access tables' write generations.** the four
tables every entry is derived from (``groups``, ``group_members``,
``roles``, ``role_assignments``) carry write generations, and
:mod:`threetears.agent.acl.generation_follow` hands each of their row
broadcasts to the ``evict_*_row`` method of that table, which evicts
exactly the entries the row reaches. a layer is emptied only when the
reach of a change is unknown: a row that does not say which member or
group it names, or a table dropped because a broadcast was missed.

**trusted only while followed.** a cache handed to
:func:`~threetears.agent.acl.generation_follow.follow_access_tables` serves
and keeps entries only while every table's watch is running
(:attr:`AclCache.trusted`). a follower whose watches are failing cannot judge
a missed broadcast, so the cache asks its loaders every time and is emptied,
and nothing held across the failure is served after it. once its follower
stops, it is never trusted again. a cache nobody ever followed is trusted:
that is a scratch cache, scoped to one request (a dry run, a test), which no
write can reach while it lives; a cache that outlives a request must be
followed.

**the read fence.** an entry computed from rows read before an eviction
must not be stored after it: the eviction would be undone. a caller
takes :meth:`AclCache.read_fence` before its loader reads and passes it
to ``put_*``; a store whose fence an eviction has since moved is
skipped, and the next lookup reads again.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from threading import RLock
from typing import TYPE_CHECKING
from uuid import UUID

from threetears.agent.acl.loader import GrantLoader, MembershipLoader
from threetears.agent.acl.types import GroupMembership, Trail
from threetears.observe import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "AclCache",
    "ActorMembershipEntry",
    "ActorMembershipKey",
    "GroupNamespaceEntry",
    "GroupNamespaceKey",
    "GroupTypeCustomerEntry",
    "GroupTypeCustomerKey",
]

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# key + entry shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActorMembershipKey:
    """key into the membership layer.

    :ivar actor_kind: ``"user"``, ``"agent"`` or ``"group"`` — drives
        the loader method called on miss (``"group"`` resolves a child
        group's parent groups for the depth-capped walk)
    :ivar actor_id: caller UUID
    """

    actor_kind: str
    actor_id: UUID


@dataclass(frozen=True)
class GroupNamespaceKey:
    """key into the per-namespace assignment layer.

    :ivar group_id: group whose contribution is cached
    :ivar namespace_id: target namespace
    """

    group_id: UUID
    namespace_id: UUID


@dataclass(frozen=True)
class GroupTypeCustomerKey:
    """key into the type+customer assignment layer.

    :ivar group_id: group whose contribution is cached
    :ivar namespace_type: namespace type discriminator
    :ivar customer_id: customer the namespace belongs to
    """

    group_id: UUID
    namespace_type: str
    customer_id: UUID


@dataclass(frozen=True)
class ActorMembershipEntry:
    """value stored in the membership layer.

    stores the full membership rows, not just group ids, so the
    evaluator's cross-customer + member-type filter can run against
    cached state without a loader round-trip on the hot path. group
    ids are derivable from the rows; the row carries the
    ``customer_id`` discriminator the filter needs.

    :ivar memberships: tuple of :class:`GroupMembership` rows for the
        actor (deterministically ordered by the loader); on a cache
        hit the evaluator filters this in process for the
        namespace's customer + member type
    :ivar date_cached: utc moment the entry was minted; for diagnosis
        only, nothing expires on it
    """

    memberships: tuple[GroupMembership, ...]
    date_cached: datetime


@dataclass(frozen=True)
class GroupNamespaceEntry:
    """value stored in the per-namespace assignment layer.

    :ivar actions: action set this group contributes for the
        namespace (may be empty if every assignment was filtered out)
    :ivar trails: trail rows the per-namespace resolution produced;
        cached so trail-mode lookups via the explain api can pull
        the same rows the decision-mode lookup used
    :ivar date_cached: utc moment the entry was minted; for diagnosis only
    :ivar role_ids: every role the resolution read for the group's covering
        assignments, so an edit of any of them evicts this entry and no
        other; ``None`` when the caller did not say, which a role edit
        treats as reaching it
    """

    actions: frozenset[str]
    trails: tuple[Trail, ...]
    date_cached: datetime
    role_ids: frozenset[UUID] | None = None


@dataclass(frozen=True)
class GroupTypeCustomerEntry:
    """value stored in the type+customer assignment layer.

    same shape as :class:`GroupNamespaceEntry` but keyed by
    ``(group, type, customer)`` so a single group's broadly-scoped
    assignment populates one entry that serves every namespace of the
    type+customer combination.

    :ivar actions: action set this group contributes for the
        type+customer scope
    :ivar trails: trail rows the resolution produced
    :ivar date_cached: utc moment the entry was minted; for diagnosis only
    """

    actions: frozenset[str]
    trails: tuple[Trail, ...]
    date_cached: datetime


# ---------------------------------------------------------------------------
# cache class
# ---------------------------------------------------------------------------


class AclCache:
    """three-layer cache for the rbac evaluator, bundling loaders.

    the cache carries the two loader handles the evaluator depends on
    alongside its three in-process layers. production wiring (broker
    + every agent pod) hands a single :class:`AclCache` instance to
    every authorization call site: the cache holds the entries, the
    loaders resolve misses, and the evaluator reads both through the
    public :attr:`membership_loader` + :attr:`grant_loader` attributes.

    layers are explicitly separated so invalidation can target one
    layer without disturbing the others. all three layers share one
    ``RLock`` so multi-step "lookup-or-insert" sequences run
    atomically without giving up the cache in the middle.

    instances are process-local. one cache per process is the
    expected deployment shape: the broker has one, each agent pod
    has one, and each is followed on the access tables
    (:func:`~threetears.agent.acl.generation_follow.follow_access_tables`),
    which is what evicts it when another process writes.

    :param membership_loader: actor -> groups resolver consumed by the
        evaluator on membership-layer misses
    :ptype membership_loader: MembershipLoader
    :param grant_loader: groups -> assignments + roles + groups
        resolver consumed by the evaluator on assignment-layer misses
    :ptype grant_loader: GrantLoader
    """

    def __init__(
        self,
        *,
        membership_loader: MembershipLoader,
        grant_loader: GrantLoader,
    ) -> None:
        self.membership_loader = membership_loader
        self.grant_loader = grant_loader
        self._membership: dict[ActorMembershipKey, ActorMembershipEntry] = {}
        self._group_namespace: dict[GroupNamespaceKey, GroupNamespaceEntry] = {}
        self._group_type_customer: dict[
            GroupTypeCustomerKey,
            GroupTypeCustomerEntry,
        ] = {}
        self._lock = RLock()
        # moved by every eviction; see :meth:`read_fence`
        self._evictions = 0
        self._fence_skipped_stores = 0
        # who follows this cache: never set for a scratch cache; see :meth:`followed_by`
        self._watching: Callable[[], bool] | None = None
        self._ever_followed = False

    # -----------------------------------------------------------------
    # trust
    # -----------------------------------------------------------------

    def followed_by(self, watching: Callable[[], bool] | None) -> None:
        """say who follows the cache, or ``None`` once nobody does any more.

        Set by :func:`~threetears.agent.acl.generation_follow.follow_access_tables`; a caller does
        not call it. Once followed, the cache is :attr:`trusted` only while ``watching`` answers
        ``True``, and after ``None`` never again.

        :param watching: answers whether every table's watch is running now; ``None`` when the
            follower has stopped
        :ptype watching: Callable[[], bool] | None
        :return: nothing
        :rtype: None
        """
        with self._lock:
            self._watching = watching
            self._ever_followed = True
            if watching is None:
                self._empty_locked()

    @property
    def trusted(self) -> bool:
        """whether entries may be served and stored now.

        A cache nobody ever followed is trusted (a scratch cache, see the module docstring). A
        followed one is trusted while every watch is running; losing trust empties it, so an entry
        held while a broadcast could go unjudged is never served after.

        :return: ``True`` when the cache may answer from what it holds
        :rtype: bool
        """
        with self._lock:
            return self._trusted_locked()

    def _trusted_locked(self) -> bool:
        """see :attr:`trusted`; the caller holds the lock.

        :return: whether the cache is trusted
        :rtype: bool
        """
        if not self._ever_followed:
            return True
        watching = self._watching
        trusted = watching is not None and watching()
        if not trusted and self._size_locked():
            log.warning(
                "acl cache not trusted: its access-table watches are not running, so it is emptied and "
                "asks its loaders until they are",
                extra={"extra_data": {"entries_dropped": self._size_locked()}},
            )
            self._empty_locked()
        return trusted

    def _empty_locked(self) -> None:
        """drop every entry and move the fence; the caller holds the lock.

        :return: nothing
        :rtype: None
        """
        self._evictions += 1
        self._membership.clear()
        self._group_namespace.clear()
        self._group_type_customer.clear()

    def _size_locked(self) -> int:
        """total entries; the caller holds the lock.

        :return: the count
        :rtype: int
        """
        return len(self._membership) + len(self._group_namespace) + len(self._group_type_customer)

    # -----------------------------------------------------------------
    # the read fence
    # -----------------------------------------------------------------

    def read_fence(self) -> int:
        """take before reading what an entry will be computed from; pass to ``put_*``.

        :return: a token that an eviction after this call invalidates
        :rtype: int
        """
        with self._lock:
            return self._evictions

    def _fence_moved(self, fence: int | None) -> bool:
        """whether an eviction landed since ``fence`` was taken. caller holds the lock.

        :param fence: from :meth:`read_fence`, or ``None`` when the caller took none
        :ptype fence: int | None
        :return: ``True`` when the entry must not be stored
        :rtype: bool
        """
        moved = fence is not None and fence != self._evictions
        if moved:
            self._fence_skipped_stores += 1
            log.debug(
                "acl cache entry not stored: an eviction landed while it was being computed",
                extra={"extra_data": {"fence_skipped_stores": self._fence_skipped_stores}},
            )
        return moved

    @property
    def fence_skipped_stores(self) -> int:
        """how many computed entries were not stored because an eviction overtook their read.

        A count that climbs with every lookup means evictions arrive faster than entries can be
        computed, so the cache is not caching.

        :return: the count since the cache was built
        :rtype: int
        """
        with self._lock:
            return self._fence_skipped_stores

    # -----------------------------------------------------------------
    # membership layer
    # -----------------------------------------------------------------

    def get_membership(
        self,
        key: ActorMembershipKey,
    ) -> ActorMembershipEntry | None:
        """lookup an actor's group ids; returns None on a miss, or while the cache is not trusted.

        :param key: actor identity tuple
        :ptype key: ActorMembershipKey
        :return: cached entry or None
        :rtype: ActorMembershipEntry | None
        """
        with self._lock:
            result = self._membership.get(key) if self._trusted_locked() else None
        return result

    def put_membership(
        self,
        key: ActorMembershipKey,
        memberships: tuple[GroupMembership, ...],
        *,
        fence: int | None = None,
    ) -> ActorMembershipEntry:
        """insert or replace a membership entry.

        :param key: actor identity tuple
        :ptype key: ActorMembershipKey
        :param memberships: tuple of :class:`GroupMembership` rows
            the actor belongs to
        :ptype memberships: tuple[GroupMembership, ...]
        :param fence: :meth:`read_fence` as taken before the rows were read;
            the entry is not stored when an eviction has landed since
        :ptype fence: int | None
        :return: the entry (with freshly-stamped ``date_cached``), stored
            unless the fence moved or the cache is not trusted
        :rtype: ActorMembershipEntry
        """
        entry = ActorMembershipEntry(
            memberships=memberships,
            date_cached=datetime.now(UTC),
        )
        with self._lock:
            if self._trusted_locked() and not self._fence_moved(fence):
                self._membership[key] = entry
        return entry

    def invalidate_membership(self, key: ActorMembershipKey) -> None:
        """drop a single actor's cached group ids.

        a ``group_members`` row naming the actor reaches here. drops
        only the matching entry; the assignment layers stay populated because their
        keys are independent of actor identity.

        :param key: actor identity to evict
        :ptype key: ActorMembershipKey
        :return: nothing
        :rtype: None
        """
        with self._lock:
            self._evictions += 1
            self._membership.pop(key, None)

    def invalidate_membership_for_actor(
        self,
        actor_kind: str,
        actor_id: UUID,
    ) -> None:
        """drop every membership entry for an ``(actor_kind, actor_id)`` pair.

        convenience wrapper that builds the key from the parts; useful
        for callers that do not have a :class:`ActorMembershipKey` in
        hand.

        :param actor_kind: ``"user"`` or ``"agent"``
        :ptype actor_kind: str
        :param actor_id: actor UUID
        :ptype actor_id: UUID
        :return: nothing
        :rtype: None
        """
        self.invalidate_membership(
            ActorMembershipKey(actor_kind=actor_kind, actor_id=actor_id),
        )

    # -----------------------------------------------------------------
    # per-namespace layer
    # -----------------------------------------------------------------

    def get_group_namespace(
        self,
        key: GroupNamespaceKey,
    ) -> GroupNamespaceEntry | None:
        """lookup a per-namespace contribution; returns None on a miss, or while not trusted.

        :param key: ``(group_id, namespace_id)`` tuple
        :ptype key: GroupNamespaceKey
        :return: cached entry or None
        :rtype: GroupNamespaceEntry | None
        """
        with self._lock:
            result = self._group_namespace.get(key) if self._trusted_locked() else None
        return result

    def put_group_namespace(
        self,
        key: GroupNamespaceKey,
        actions: frozenset[str],
        trails: tuple[Trail, ...],
        *,
        role_ids: frozenset[UUID] | None = None,
        fence: int | None = None,
    ) -> GroupNamespaceEntry:
        """insert or replace a per-namespace entry.

        :param key: ``(group_id, namespace_id)`` tuple
        :ptype key: GroupNamespaceKey
        :param actions: action set the group contributes
        :ptype actions: frozenset[str]
        :param trails: trail rows produced during resolution
        :ptype trails: tuple[Trail, ...]
        :param role_ids: every role the resolution read; ``None`` when unknown
        :ptype role_ids: frozenset[UUID] | None
        :param fence: :meth:`read_fence` as taken before the rows were read;
            the entry is not stored when an eviction has landed since
        :ptype fence: int | None
        :return: the entry, stored unless the fence moved or the cache is not trusted
        :rtype: GroupNamespaceEntry
        """
        entry = GroupNamespaceEntry(
            actions=actions,
            trails=trails,
            date_cached=datetime.now(UTC),
            role_ids=role_ids,
        )
        with self._lock:
            if self._trusted_locked() and not self._fence_moved(fence):
                self._group_namespace[key] = entry
        return entry

    def invalidate_group_namespace(self, key: GroupNamespaceKey) -> None:
        """drop a single per-namespace entry.

        :param key: ``(group_id, namespace_id)`` tuple
        :ptype key: GroupNamespaceKey
        :return: nothing
        :rtype: None
        """
        with self._lock:
            self._evictions += 1
            self._group_namespace.pop(key, None)

    def invalidate_namespace(self, namespace_id: UUID) -> None:
        """drop every per-namespace entry whose key names ``namespace_id``.

        emitted in response to assignment-change events that affect
        a specific namespace (a new namespace-scope assignment, or a
        deletion of one).

        :param namespace_id: namespace to evict for every group
        :ptype namespace_id: UUID
        :return: nothing
        :rtype: None
        """
        with self._lock:
            self._evictions += 1
            doomed = [key for key in self._group_namespace if key.namespace_id == namespace_id]
            for key in doomed:
                del self._group_namespace[key]

    def invalidate_group(self, group_id: UUID) -> None:
        """drop every entry that names ``group_id`` in either assignment layer.

        a ``role_assignments`` row granting to the group reaches here,
        as does a ``groups`` row for it.

        does not touch the membership layer (callers also want a
        membership-layer invalidation for any actor that was in the
        group; that is a separate fan-out).

        :param group_id: group to evict from both assignment layers
        :ptype group_id: UUID
        :return: nothing
        :rtype: None
        """
        with self._lock:
            self._evictions += 1
            ns_doomed = [ns_key for ns_key in self._group_namespace if ns_key.group_id == group_id]
            for ns_key in ns_doomed:
                del self._group_namespace[ns_key]
            tc_doomed = [tc_key for tc_key in self._group_type_customer if tc_key.group_id == group_id]
            for tc_key in tc_doomed:
                del self._group_type_customer[tc_key]

    # -----------------------------------------------------------------
    # type+customer layer
    # -----------------------------------------------------------------

    def get_group_type_customer(
        self,
        key: GroupTypeCustomerKey,
    ) -> GroupTypeCustomerEntry | None:
        """lookup a type+customer contribution; returns None on a miss, or while not trusted.

        :param key: ``(group_id, namespace_type, customer_id)`` tuple
        :ptype key: GroupTypeCustomerKey
        :return: cached entry or None
        :rtype: GroupTypeCustomerEntry | None
        """
        with self._lock:
            result = self._group_type_customer.get(key) if self._trusted_locked() else None
        return result

    def put_group_type_customer(
        self,
        key: GroupTypeCustomerKey,
        actions: frozenset[str],
        trails: tuple[Trail, ...],
        *,
        fence: int | None = None,
    ) -> GroupTypeCustomerEntry:
        """insert or replace a type+customer entry.

        :param key: ``(group_id, namespace_type, customer_id)`` tuple
        :ptype key: GroupTypeCustomerKey
        :param actions: action set the group contributes for the
            type+customer combination
        :ptype actions: frozenset[str]
        :param trails: trail rows produced during resolution
        :ptype trails: tuple[Trail, ...]
        :param fence: :meth:`read_fence` as taken before the rows were read;
            the entry is not stored when an eviction has landed since
        :ptype fence: int | None
        :return: the entry, stored unless the fence moved or the cache is not trusted
        :rtype: GroupTypeCustomerEntry
        """
        entry = GroupTypeCustomerEntry(
            actions=actions,
            trails=trails,
            date_cached=datetime.now(UTC),
        )
        with self._lock:
            if self._trusted_locked() and not self._fence_moved(fence):
                self._group_type_customer[key] = entry
        return entry

    def invalidate_group_type_customer(
        self,
        key: GroupTypeCustomerKey,
    ) -> None:
        """drop a single type+customer entry.

        :param key: tuple to evict
        :ptype key: GroupTypeCustomerKey
        :return: nothing
        :rtype: None
        """
        with self._lock:
            self._evictions += 1
            self._group_type_customer.pop(key, None)

    # -----------------------------------------------------------------
    # bulk operations
    # -----------------------------------------------------------------

    def invalidate_all(self) -> None:
        """clear every layer.

        for a change whose reach is unknown; a heard, named change
        evicts row by row instead (``evict_*_row``).

        :return: nothing
        :rtype: None
        """
        with self._lock:
            self._empty_locked()

    # -----------------------------------------------------------------
    # row by row, from the access tables
    # -----------------------------------------------------------------

    def evict_group_member_row(self, member_kind: str | None, member_id: UUID | None) -> None:
        """a ``group_members`` row changed: evict the membership entry of the member it names.

        a group's row (``member_kind == "group"``) evicts that child group's
        parent entry and nothing else; every actor beneath it walks the
        parents at read. a row that does not name its member (an older
        writer, a raw path) has an unknown reach, so the membership layer is
        emptied; the assignment layers are not derived from this table.

        :param member_kind: the row's ``member_type``, or ``None`` when unknown
        :ptype member_kind: str | None
        :param member_id: the row's ``member_id``, or ``None`` when unknown
        :ptype member_id: UUID | None
        :return: nothing
        :rtype: None
        """
        if member_kind is None or member_id is None:
            self.drop_membership_layer()
        else:
            self.invalidate_membership_for_actor(member_kind, member_id)

    def evict_role_assignment_row(self, group_id: UUID | None) -> None:
        """a ``role_assignments`` row changed: evict the assignment entries of the group it grants to.

        :param group_id: the row's ``group_id``, or ``None`` when the row did
            not say, which empties both assignment layers
        :ptype group_id: UUID | None
        :return: nothing
        :rtype: None
        """
        if group_id is None:
            self.drop_assignment_layers()
        else:
            self.invalidate_group(group_id)

    def evict_role_row(self, role_id: UUID) -> None:
        """a ``roles`` row changed: evict every assignment entry whose resolution read that role.

        an entry that did not record its roles is evicted too, as is every
        type+customer entry, which records none.

        :param role_id: the role
        :ptype role_id: UUID
        :return: nothing
        :rtype: None
        """
        with self._lock:
            self._evictions += 1
            doomed = [
                key
                for key, entry in self._group_namespace.items()
                if entry.role_ids is None or role_id in entry.role_ids
            ]
            for key in doomed:
                del self._group_namespace[key]
            self._group_type_customer.clear()

    def evict_group_row(self, group_id: UUID) -> None:
        """a ``groups`` row changed: evict what was resolved through that group.

        its assignment entries; its own parent entry (``("group", id)``); and
        every membership entry that names it. deleting a group cascades its
        memberships away in the database, so an actor entry still naming it
        would otherwise walk to the deleted group's parents.

        :param group_id: the group
        :ptype group_id: UUID
        :return: nothing
        :rtype: None
        """
        with self._lock:
            self.invalidate_group(group_id)
            self._membership.pop(ActorMembershipKey(actor_kind="group", actor_id=group_id), None)
            doomed = [
                key
                for key, entry in self._membership.items()
                if any(membership.group_id == group_id for membership in entry.memberships)
            ]
            for key in doomed:
                del self._membership[key]

    def drop_membership_layer(self) -> None:
        """empty the membership layer, for a change to ``group_members`` whose reach is unknown.

        :return: nothing
        :rtype: None
        """
        with self._lock:
            self._evictions += 1
            self._membership.clear()

    def drop_assignment_layers(self) -> None:
        """empty both assignment layers, for a change to ``role_assignments`` or ``roles`` whose reach is unknown.

        :return: nothing
        :rtype: None
        """
        with self._lock:
            self._evictions += 1
            self._group_namespace.clear()
            self._group_type_customer.clear()

    @property
    def size(self) -> int:
        """total entry count across the three layers.

        :return: sum of layer sizes
        :rtype: int
        """
        with self._lock:
            result = self._size_locked()
        return result

    @property
    def membership_size(self) -> int:
        """entry count for the membership layer only.

        :return: len of the membership dict
        :rtype: int
        """
        with self._lock:
            result = len(self._membership)
        return result

    @property
    def group_namespace_size(self) -> int:
        """entry count for the per-namespace layer only.

        :return: len of the per-namespace dict
        :rtype: int
        """
        with self._lock:
            result = len(self._group_namespace)
        return result

    @property
    def group_type_customer_size(self) -> int:
        """entry count for the type+customer layer only.

        :return: len of the type+customer dict
        :rtype: int
        """
        with self._lock:
            result = len(self._group_type_customer)
        return result
