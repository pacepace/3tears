"""three-layer in-process ttl cache for the rbac evaluator.

the evaluator is pure-functional: it never holds state across calls.
this cache sits in front of the loaders the evaluator depends on so
the production hot path serves authorization decisions from process
memory most of the time.

three explicit layers (do not unify; the layers exist so invalidation
fans out correctly):

- **membership layer** — keyed by ``("user", user_id)`` /
  ``("agent", agent_id)``; value is the tuple of group ids the actor
  belongs to. invalidated by membership-change events.
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

each layer enforces its own ttl. layer entries carry a freshness
timestamp; lookup re-checks ttl on every hit so a stale entry is
evicted before being served.

cache is process-local. two pods have independent caches; cross-pod
invalidation is the responsibility of the caller's pub/sub layer
(broker publishes ``{ns}.acl.<layer>.invalidate`` events on every
mutation; subscribers translate those events to local
:meth:`AclCache.invalidate_*` calls).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import RLock
from uuid import UUID

from threetears.agent.acl.types import Trail
from threetears.observe import get_logger

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

    :ivar actor_kind: ``"user"`` or ``"agent"`` — drives the loader
        method called on miss
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

    :ivar group_ids: tuple of group UUIDs the actor belongs to
        (deterministically ordered by the loader)
    :ivar date_cached: utc moment the entry was minted; ttl is
        measured against this
    """

    group_ids: tuple[UUID, ...]
    date_cached: datetime


@dataclass(frozen=True)
class GroupNamespaceEntry:
    """value stored in the per-namespace assignment layer.

    :ivar actions: action set this group contributes for the
        namespace (may be empty if every assignment was filtered out)
    :ivar trails: trail rows the per-namespace resolution produced;
        cached so trail-mode lookups via the explain api can pull
        the same rows the decision-mode lookup used
    :ivar date_cached: utc moment the entry was minted
    """

    actions: frozenset[str]
    trails: tuple[Trail, ...]
    date_cached: datetime


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
    :ivar date_cached: utc moment the entry was minted
    """

    actions: frozenset[str]
    trails: tuple[Trail, ...]
    date_cached: datetime


# ---------------------------------------------------------------------------
# cache class
# ---------------------------------------------------------------------------


class AclCache:
    """three-layer ttl cache for the rbac evaluator.

    layers are explicitly separated so invalidation can target one
    layer without disturbing the others. all three layers share one
    ``ttl`` value and one ``RLock`` so multi-step
    "lookup-or-insert" sequences run atomically without giving up the
    cache in the middle.

    instances are process-local. one cache per process is the
    expected deployment shape: the broker has one, each agent pod
    has one. cross-process invalidation is the caller's job (publish
    invalidation events on whatever bus already exists).

    :param ttl_seconds: how long an entry stays fresh; defaults to
        sixty seconds. lookups past the ttl evict the entry and
        return ``None``.
    """

    def __init__(self, ttl_seconds: int = 60) -> None:
        self._ttl = timedelta(seconds=ttl_seconds)
        self._membership: dict[ActorMembershipKey, ActorMembershipEntry] = {}
        self._group_namespace: dict[GroupNamespaceKey, GroupNamespaceEntry] = {}
        self._group_type_customer: dict[
            GroupTypeCustomerKey, GroupTypeCustomerEntry,
        ] = {}
        self._lock = RLock()

    # -----------------------------------------------------------------
    # membership layer
    # -----------------------------------------------------------------

    def get_membership(
        self, key: ActorMembershipKey,
    ) -> ActorMembershipEntry | None:
        """lookup an actor's group ids; returns None on miss or expiry.

        :param key: actor identity tuple
        :ptype key: ActorMembershipKey
        :return: cached entry or None
        :rtype: ActorMembershipEntry | None
        """
        with self._lock:
            entry = self._membership.get(key)
            if entry is None:
                result: ActorMembershipEntry | None = None
            elif self._is_expired(entry.date_cached):
                del self._membership[key]
                result = None
            else:
                result = entry
        return result

    def put_membership(
        self,
        key: ActorMembershipKey,
        group_ids: tuple[UUID, ...],
    ) -> ActorMembershipEntry:
        """insert or replace a membership entry.

        :param key: actor identity tuple
        :ptype key: ActorMembershipKey
        :param group_ids: tuple of group UUIDs the actor belongs to
        :ptype group_ids: tuple[UUID, ...]
        :return: stored entry (with freshly-stamped ``date_cached``)
        :rtype: ActorMembershipEntry
        """
        entry = ActorMembershipEntry(
            group_ids=group_ids, date_cached=datetime.now(UTC),
        )
        with self._lock:
            self._membership[key] = entry
        return entry

    def invalidate_membership(self, key: ActorMembershipKey) -> None:
        """drop a single actor's cached group ids.

        emitted in response to ``{ns}.acl.membership.invalidate``
        events naming a specific actor. drops only the matching
        entry; the assignment layers stay populated because their
        keys are independent of actor identity.

        :param key: actor identity to evict
        :ptype key: ActorMembershipKey
        :return: nothing
        :rtype: None
        """
        with self._lock:
            self._membership.pop(key, None)

    def invalidate_membership_for_actor(
        self, actor_kind: str, actor_id: UUID,
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
        self, key: GroupNamespaceKey,
    ) -> GroupNamespaceEntry | None:
        """lookup a per-namespace contribution; returns None on miss / expiry.

        :param key: ``(group_id, namespace_id)`` tuple
        :ptype key: GroupNamespaceKey
        :return: cached entry or None
        :rtype: GroupNamespaceEntry | None
        """
        with self._lock:
            entry = self._group_namespace.get(key)
            if entry is None:
                result: GroupNamespaceEntry | None = None
            elif self._is_expired(entry.date_cached):
                del self._group_namespace[key]
                result = None
            else:
                result = entry
        return result

    def put_group_namespace(
        self,
        key: GroupNamespaceKey,
        actions: frozenset[str],
        trails: tuple[Trail, ...],
    ) -> GroupNamespaceEntry:
        """insert or replace a per-namespace entry.

        :param key: ``(group_id, namespace_id)`` tuple
        :ptype key: GroupNamespaceKey
        :param actions: action set the group contributes
        :ptype actions: frozenset[str]
        :param trails: trail rows produced during resolution
        :ptype trails: tuple[Trail, ...]
        :return: stored entry
        :rtype: GroupNamespaceEntry
        """
        entry = GroupNamespaceEntry(
            actions=actions, trails=trails, date_cached=datetime.now(UTC),
        )
        with self._lock:
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
            doomed = [
                key for key in self._group_namespace
                if key.namespace_id == namespace_id
            ]
            for key in doomed:
                del self._group_namespace[key]

    def invalidate_group(self, group_id: UUID) -> None:
        """drop every entry that names ``group_id`` in either assignment layer.

        emitted in response to role-change events affecting a role the
        group holds, or to assignment-change events targeting the
        group, or to membership-change events that drop the group
        entirely.

        does not touch the membership layer (callers also want a
        membership-layer invalidation for any actor that was in the
        group; that is a separate fan-out).

        :param group_id: group to evict from both assignment layers
        :ptype group_id: UUID
        :return: nothing
        :rtype: None
        """
        with self._lock:
            ns_doomed = [
                ns_key for ns_key in self._group_namespace
                if ns_key.group_id == group_id
            ]
            for ns_key in ns_doomed:
                del self._group_namespace[ns_key]
            tc_doomed = [
                tc_key for tc_key in self._group_type_customer
                if tc_key.group_id == group_id
            ]
            for tc_key in tc_doomed:
                del self._group_type_customer[tc_key]

    # -----------------------------------------------------------------
    # type+customer layer
    # -----------------------------------------------------------------

    def get_group_type_customer(
        self, key: GroupTypeCustomerKey,
    ) -> GroupTypeCustomerEntry | None:
        """lookup a type+customer contribution; returns None on miss / expiry.

        :param key: ``(group_id, namespace_type, customer_id)`` tuple
        :ptype key: GroupTypeCustomerKey
        :return: cached entry or None
        :rtype: GroupTypeCustomerEntry | None
        """
        with self._lock:
            entry = self._group_type_customer.get(key)
            if entry is None:
                result: GroupTypeCustomerEntry | None = None
            elif self._is_expired(entry.date_cached):
                del self._group_type_customer[key]
                result = None
            else:
                result = entry
        return result

    def put_group_type_customer(
        self,
        key: GroupTypeCustomerKey,
        actions: frozenset[str],
        trails: tuple[Trail, ...],
    ) -> GroupTypeCustomerEntry:
        """insert or replace a type+customer entry.

        :param key: ``(group_id, namespace_type, customer_id)`` tuple
        :ptype key: GroupTypeCustomerKey
        :param actions: action set the group contributes for the
            type+customer combination
        :ptype actions: frozenset[str]
        :param trails: trail rows produced during resolution
        :ptype trails: tuple[Trail, ...]
        :return: stored entry
        :rtype: GroupTypeCustomerEntry
        """
        entry = GroupTypeCustomerEntry(
            actions=actions, trails=trails, date_cached=datetime.now(UTC),
        )
        with self._lock:
            self._group_type_customer[key] = entry
        return entry

    def invalidate_group_type_customer(
        self, key: GroupTypeCustomerKey,
    ) -> None:
        """drop a single type+customer entry.

        :param key: tuple to evict
        :ptype key: GroupTypeCustomerKey
        :return: nothing
        :rtype: None
        """
        with self._lock:
            self._group_type_customer.pop(key, None)

    # -----------------------------------------------------------------
    # bulk operations
    # -----------------------------------------------------------------

    def invalidate_all(self) -> None:
        """clear every layer.

        emitted on role-definition changes (a role's permissions
        were edited; the safe move is to drop everything because
        every assignment that references the role is now stale and
        we have no fast index for that). also useful for tests.

        :return: nothing
        :rtype: None
        """
        with self._lock:
            self._membership.clear()
            self._group_namespace.clear()
            self._group_type_customer.clear()

    @property
    def size(self) -> int:
        """total entry count across the three layers.

        :return: sum of layer sizes
        :rtype: int
        """
        with self._lock:
            result = (
                len(self._membership)
                + len(self._group_namespace)
                + len(self._group_type_customer)
            )
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

    # -----------------------------------------------------------------
    # internal helpers
    # -----------------------------------------------------------------

    def _is_expired(self, date_cached: datetime) -> bool:
        """true iff the entry's age exceeds the ttl.

        :param date_cached: timestamp the entry was minted
        :ptype date_cached: datetime
        :return: whether the entry has aged past ttl
        :rtype: bool
        """
        age = datetime.now(UTC) - date_cached
        return age >= self._ttl
