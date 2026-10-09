"""a per-caller answer derived from the access tables, kept in this process and dropped by their generations.

A tool pod that asks the hub something about its caller -- which tool namespaces ``namespace.discover``
lists for them, say -- gets an answer computed from ``groups``, ``group_members``, ``roles``,
``role_assignments`` and ``namespaces``. Asking on every call costs a hub round trip per call; caching
the answer with an age would serve a withdrawn grant until it expired. :class:`CallerAccessCache` keeps
one answer per caller, with no age of any kind, and is dropped by the tables it is derived from
(epoch-task-06, "Derived Caches"):

- a ``group_members`` row naming a user or an agent drops the entries of the callers that are that
  principal, and nobody else's;
- any other row -- a nested group's membership, a group, a role, an assignment, a namespace -- drops
  every entry, because an entry keyed by the person records nothing of the groups, grants and
  namespaces its answer passed through;
- a row whose broadcast does not say what it names drops every entry, and is counted as degraded;
- a table dropped in this process (a missed broadcast, a replaced bucket) drops every entry.

**The read fence.** An answer whose question was asked before an eviction is not stored: take
:meth:`CallerAccessCache.read_fence` before asking, and pass it to :meth:`CallerAccessCache.put`.
Otherwise a change landing while the hub answers would be evicted from an empty cache and its stale
answer stored after it.

:class:`CallerNamespaces` is the answer this was built for: the namespaces the hub's
``namespace.discover`` lists for a caller, asked with the caller's own tokens, failing closed.

Bind and follow it in one call with :func:`follow_caller_access_cache`, once the registry's
invalidation listener is running; it hears the rows, and the follower drops what a missed broadcast
left stale. Binding alone (:func:`bind_caller_cache_to_access_tables`) never catches a missed one.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Generic, Protocol, TypeVar
from uuid import UUID, uuid4

from threetears.observe import get_logger

from threetears.agent.acl.access_tables import ACCESS_TABLES, DegradedEvictions
from threetears.agent.acl.collections import GroupMemberCollection
from threetears.agent.acl.types import MemberType

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from threetears.core.collections.registry import CacheInvalidationMessage, CollectionRegistry

__all__ = [
    "CallerAccessCache",
    "CallerKey",
    "CallerNamespaces",
    "CallerNamespacesUnavailable",
    "NamespaceDiscovery",
    "bind_caller_cache_to_access_tables",
]

log = get_logger(__name__)

V = TypeVar("V")

_GROUP_MEMBERS: Final = GroupMemberCollection.schema.name

#: the member kinds a caller is made of; a ``group`` member's row is a nesting change and drops all
_CALLER_KINDS: Final = frozenset({MemberType.USER.value, MemberType.AGENT.value})


@dataclass(frozen=True)
class CallerKey:
    """one caller, as a verified call names them: the agent it came through and the person behind it.

    :ivar agent_id: the verified agent (for a REST or app reader, the platform's ingress principal)
    :ivar user_id: the verified person, or ``None`` for a call with no person behind it
    """

    agent_id: UUID
    user_id: UUID | None


class CallerAccessCache(Generic[V]):
    """one answer per caller, derived from the access tables, with no age; see the module docstring.

    Thread-safe: the registry's listener evicts on the event loop while a tool may read from a worker
    thread.
    """

    def __init__(self) -> None:
        """start empty.

        :return: nothing
        :rtype: None
        """
        self._entries: dict[CallerKey, V] = {}
        self._lock = threading.Lock()
        # moved by every eviction; see read_fence
        self._evictions = 0
        self._fence_skipped_stores = 0

    def read_fence(self) -> int:
        """the fence to take before asking the question whose answer :meth:`put` will store.

        :return: the eviction count now
        :rtype: int
        """
        with self._lock:
            return self._evictions

    def get(self, caller: CallerKey) -> V | None:
        """the caller's answer, when one is held.

        :param caller: the caller
        :ptype caller: CallerKey
        :return: the answer, or ``None``
        :rtype: V | None
        """
        with self._lock:
            return self._entries.get(caller)

    def put(self, caller: CallerKey, value: V, *, fence: int) -> bool:
        """hold ``value`` as the caller's answer, unless an eviction landed since ``fence`` was taken.

        :param caller: the caller
        :ptype caller: CallerKey
        :param value: the answer
        :ptype value: V
        :param fence: :meth:`read_fence` as taken before the question was asked
        :ptype fence: int
        :return: whether it was stored
        :rtype: bool
        """
        with self._lock:
            stored = fence == self._evictions
            if stored:
                self._entries[caller] = value
            else:
                self._fence_skipped_stores += 1
        if not stored:
            log.debug("a caller's answer was not stored: an access-table change landed while it was asked")
        return stored

    @property
    def size(self) -> int:
        """how many callers have an answer held.

        :return: the count
        :rtype: int
        """
        with self._lock:
            return len(self._entries)

    @property
    def fence_skipped_stores(self) -> int:
        """answers not stored because a change landed while they were asked.

        :return: the count
        :rtype: int
        """
        with self._lock:
            return self._fence_skipped_stores

    def evict_principal(self, member_type: str, member_id: UUID) -> int:
        """drop the answers of every caller that is this principal: the person, or the agent.

        :param member_type: ``user`` or ``agent``
        :ptype member_type: str
        :param member_id: the principal
        :ptype member_id: UUID
        :return: how many answers were dropped
        :rtype: int
        """
        with self._lock:
            self._evictions += 1
            if member_type == MemberType.USER.value:
                doomed = [key for key in self._entries if key.user_id == member_id]
            else:
                doomed = [key for key in self._entries if key.agent_id == member_id]
            for key in doomed:
                del self._entries[key]
        return len(doomed)

    def invalidate_all(self) -> None:
        """drop every answer.

        :return: nothing
        :rtype: None
        """
        with self._lock:
            self._evictions += 1
            self._entries.clear()


def bind_caller_cache_to_access_tables(
    registry: CollectionRegistry,
    cache: CallerAccessCache[object],
    *,
    degraded: DegradedEvictions | None = None,
) -> Callable[[], None]:
    """make ``cache`` a cache derived from the access tables on ``registry``.

    A consumer calls :func:`threetears.agent.acl.generation_follow.follow_caller_access_cache`, which
    binds through this and follows the tables.

    :param registry: the registry whose invalidation listener hears the tables' row broadcasts
    :ptype registry: CollectionRegistry
    :param cache: the cache to drop from
    :ptype cache: CallerAccessCache
    :param degraded: where rows that did not say what they name are counted; a new record when ``None``
    :ptype degraded: DegradedEvictions | None
    :return: a call that removes the registrations
    :rtype: Callable[[], None]
    """
    record = degraded if degraded is not None else DegradedEvictions()

    def on_group_member(message: CacheInvalidationMessage) -> None:
        columns = message.columns or {}
        kind = columns.get("member_type")
        member = _uuid(columns.get("member_id"))
        if kind in _CALLER_KINDS and member is not None:
            cache.evict_principal(kind, member)
        elif kind == MemberType.GROUP.value and member is not None:
            # a group nested in another: which callers it reaches is not recorded per entry
            cache.invalidate_all()
        else:
            record.record(_GROUP_MEMBERS, "no member_type and member_id it parses")
            cache.invalidate_all()

    def on_other(_message: CacheInvalidationMessage) -> None:
        cache.invalidate_all()

    removers = [
        registry.register_derived_cache(
            table,
            on_row=on_group_member if table == _GROUP_MEMBERS else on_other,
            on_table_dropped=cache.invalidate_all,
        )
        for table in ACCESS_TABLES
    ]

    def remove() -> None:
        for remover in removers:
            remover()

    return remove


def _uuid(value: str | None) -> UUID | None:
    """a broadcast's string value as a UUID, or ``None`` when absent or not one.

    :param value: the value as the broadcast carried it
    :ptype value: str | None
    :return: the UUID, or ``None``
    :rtype: UUID | None
    """
    result: UUID | None = None
    if value is not None:
        try:
            result = UUID(value)
        except ValueError:
            result = None
    return result


class NamespaceDiscovery(Protocol):
    """the hub's ``namespace.discover``, as ``threetears.agent.tools.namespace_discovery_client`` asks it."""

    async def discover(
        self,
        *,
        correlation_id: UUID,
        identity_token: str | None = None,
        user_identity_token: str | None = None,
        namespace_type: Any = None,
    ) -> Sequence[Any]:
        """the namespaces the caller whose tokens these are can see; each has a ``name``."""
        ...


class CallerNamespacesUnavailable(Exception):
    """the caller's namespaces could not be had; ``reason`` is for the log, never for the caller.

    :param reason: why
    :ptype reason: str
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class CallerNamespaces:
    """the namespaces of one type the hub lists for a caller, asked once per caller and kept until the
    access tables say otherwise.

    The hub answers ``namespace.discover`` about whoever signed the tokens forwarded to it, so the
    answer is about THIS caller; nothing here names a principal. The answer is kept in a
    :class:`CallerAccessCache` keyed by the verified agent and person, which the pod binds and follows
    (:func:`threetears.agent.acl.generation_follow.follow_caller_access_cache`). It fails closed: a pod
    not connected yet, a call with no token, a failed or refused discovery all raise
    :class:`CallerNamespacesUnavailable`, never an empty answer in place of an error.

    :param discovery: returns the pod's discovery client once it is connected, ``None`` before
    :ptype discovery: Callable[[], NamespaceDiscovery | None]
    :param cache: where answers are kept; the pod's followed cache
    :ptype cache: CallerAccessCache[frozenset[str]]
    :param namespace_type: the namespace type asked for (``tool``); ``None`` asks for every type
    :ptype namespace_type: str | None
    """

    def __init__(
        self,
        discovery: Callable[[], NamespaceDiscovery | None],
        cache: CallerAccessCache[frozenset[str]],
        *,
        namespace_type: str | None = "tool",
    ) -> None:
        self._discovery = discovery
        self._cache = cache
        self._namespace_type = namespace_type

    async def names_for(
        self,
        *,
        agent_id: UUID | None,
        user_id: UUID | None,
        identity_token: str | None,
        user_identity_token: str | None,
        correlation_id: UUID | None = None,
    ) -> frozenset[str]:
        """the names of the namespaces the hub lists for this caller.

        :param agent_id: the call's verified agent
        :ptype agent_id: UUID | None
        :param user_id: the call's verified person, or ``None``
        :ptype user_id: UUID | None
        :param identity_token: the identity token the call arrived with, forwarded verbatim
        :ptype identity_token: str | None
        :param user_identity_token: the person's assertion the call arrived with, forwarded verbatim
        :ptype user_identity_token: str | None
        :param correlation_id: the call's correlation id, for tracing; a fresh one when ``None``
        :ptype correlation_id: UUID | None
        :return: the namespace names
        :rtype: frozenset[str]
        :raises CallerNamespacesUnavailable: when the answer cannot be had
        """
        if agent_id is None:
            raise CallerNamespacesUnavailable("the call names no verified agent")
        key = CallerKey(agent_id=agent_id, user_id=user_id)
        held = self._cache.get(key)
        if held is not None:
            return held
        client = self._discovery()
        if client is None:
            raise CallerNamespacesUnavailable("the pod is not connected to the hub yet")
        if not identity_token and not user_identity_token:
            raise CallerNamespacesUnavailable("the call carried no hub credential to ask discovery with")
        fence = self._cache.read_fence()
        try:
            items = await client.discover(
                correlation_id=correlation_id if correlation_id is not None else uuid4(),
                identity_token=identity_token,
                user_identity_token=user_identity_token,
                namespace_type=self._namespace_type,
            )
        # prawduct:allow prawduct/broad-except -- the hub's answer is the authority; any failure to have it is a refusal
        except Exception as exc:  # noqa: BLE001
            raise CallerNamespacesUnavailable(f"namespace.discover failed: {type(exc).__name__}: {exc}") from exc
        names = frozenset(str(item.name) for item in items)
        self._cache.put(key, names, fence=fence)
        return names
