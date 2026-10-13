"""follow the access tables' write generations, bound to the cache that is derived from them.

A registry that follows a table keeps a mark of the generations it has heard every row of, and
drops the table -- its rows, its scans, every cache derived from it -- when the epoch bucket shows
an advance it did not hear (epoch-task-06). The access tables are followed by watching their
generation keys, the owner's decision for these tables: a missed broadcast is noticed when the
generation moves, not at a scheduled pass. The watches are the family's generic follower,
:class:`threetears.epoch.TableFollower`, which follows any table; what is here is the half that
belongs to access control.

**One call for a consumer: :func:`follow_access_tables`.** It binds an :class:`AclCache` to the
tables' row broadcasts and follows them, and returns one handle whose :meth:`AccessTableFollowing.stop`
undoes both. Each half is unsafe alone: binding without following never drops what a missed
broadcast left stale, and following without binding drops the registry's rows but never tells the
cache. :func:`follow_caller_access_cache` does the same for a per-caller cache.

Needs ``3tears-epoch``, which comes with the ``3tears-agent-acl[follow]`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from threetears.epoch import (
    DEFAULT_BROADCAST_GRACE,
    DEFAULT_WATCH_RESTART_DELAY,
    MAX_WATCH_RESTART_DELAY,
    GenerationWatcher,
    TableFollower,
)

from threetears.agent.acl.access_tables import ACCESS_TABLES, DegradedEvictions, bind_acl_cache_to_access_tables
from threetears.agent.acl.caller_cache import CallerAccessCache, bind_caller_cache_to_access_tables

if TYPE_CHECKING:
    from collections.abc import Callable

    from threetears.core.collections.registry import CollectionRegistry

    from threetears.agent.acl.cache import AclCache

__all__ = [
    "AccessTableFollowing",
    "follow_access_tables",
    "follow_caller_access_cache",
]


@dataclass(frozen=True)
class AccessTableFollowing:
    """an :class:`AclCache` bound to the access tables and following them; :meth:`stop` undoes both.

    :ivar follower: the watches, whose :attr:`~threetears.epoch.TableFollower.health` a caller can report
    :ivar degraded: the unknown-reach evictions counted for this cache
    """

    follower: TableFollower
    degraded: DegradedEvictions
    _unbind: Callable[[], None]

    @property
    def healthy(self) -> bool:
        """whether every table's watch is running and pushed since it last failed.

        :return: the follower's health
        :rtype: bool
        """
        return self.follower.healthy

    async def stop(self) -> None:
        """stop the watches, then unbind the cache. Idempotent.

        :return: nothing
        :rtype: None
        """
        await self.follower.stop()
        self._unbind()


def follow_access_tables(
    registry: CollectionRegistry,
    cache: AclCache,
    reader: GenerationWatcher,
    *,
    grace: timedelta = DEFAULT_BROADCAST_GRACE,
    restart_delay: timedelta = DEFAULT_WATCH_RESTART_DELAY,
    max_restart_delay: timedelta = MAX_WATCH_RESTART_DELAY,
) -> AccessTableFollowing:
    """bind ``cache`` to the access tables' row broadcasts on ``registry`` and follow the tables.

    The one call a consumer makes, once ``registry``'s invalidation listener is running: it is what
    hears the rows. Stop the returned handle before stopping the listener. From here the cache serves
    and keeps entries only while every watch is running (:attr:`AclCache.trusted`), and once the
    handle stops it never does again.

    :param registry: the registry whose listener hears the rows, and which follows the tables
    :ptype registry: CollectionRegistry
    :param cache: the cache to evict from
    :ptype cache: AclCache
    :param reader: reads and watches the epoch bucket
    :ptype reader: GenerationWatcher
    :param grace: how long a watch waits for an advance's rows before judging them missed
    :ptype grace: timedelta
    :param restart_delay: a failing watch's first wait before it starts again
    :ptype restart_delay: timedelta
    :param max_restart_delay: the longest wait between a failing watch's attempts
    :ptype max_restart_delay: timedelta
    :return: the handle that stops both
    :rtype: AccessTableFollowing
    :raises RuntimeError: when the registry's invalidation listener is not running
    """
    following = _bind_and_follow(
        registry,
        lambda degraded: _acl_cache_bound_and_followed(registry, cache, degraded),
        reader,
        grace=grace,
        restart_delay=restart_delay,
        max_restart_delay=max_restart_delay,
    )
    cache.followed_by(lambda: following.follower.watching)
    return following


def _acl_cache_bound_and_followed(
    registry: CollectionRegistry, cache: AclCache, degraded: DegradedEvictions
) -> Callable[[], None]:
    """bind the acl cache; the remover also tells it nobody follows it any more, so it never trusts again.

    :param registry: the registry
    :ptype registry: CollectionRegistry
    :param cache: the acl cache
    :ptype cache: AclCache
    :param degraded: where unknown-reach rows are counted
    :ptype degraded: DegradedEvictions
    :return: the call that unbinds it and marks it unfollowed
    :rtype: Callable[[], None]
    """
    unbind = bind_acl_cache_to_access_tables(registry, cache, degraded=degraded)

    def remove() -> None:
        cache.followed_by(None)
        unbind()

    return remove


def follow_caller_access_cache(
    registry: CollectionRegistry,
    cache: CallerAccessCache[Any],
    reader: GenerationWatcher,
    *,
    grace: timedelta = DEFAULT_BROADCAST_GRACE,
    restart_delay: timedelta = DEFAULT_WATCH_RESTART_DELAY,
    max_restart_delay: timedelta = MAX_WATCH_RESTART_DELAY,
) -> AccessTableFollowing:
    """bind a per-caller cache to the access tables' row broadcasts on ``registry`` and follow the tables.

    What a tool pod calls for an answer it keeps per caller (``threetears.agent.acl.caller_cache``),
    once its registry's invalidation listener is running. Stop the returned handle before the listener.

    :param registry: the registry whose listener hears the rows, and which follows the tables
    :ptype registry: CollectionRegistry
    :param cache: the per-caller cache to drop from
    :ptype cache: CallerAccessCache
    :param reader: reads and watches the epoch bucket
    :ptype reader: GenerationWatcher
    :param grace: how long a watch waits for an advance's rows before judging them missed
    :ptype grace: timedelta
    :param restart_delay: a failing watch's first wait before it starts again
    :ptype restart_delay: timedelta
    :param max_restart_delay: the longest wait between a failing watch's attempts
    :ptype max_restart_delay: timedelta
    :return: the handle that stops both
    :rtype: AccessTableFollowing
    :raises RuntimeError: when the registry's invalidation listener is not running
    """
    following = _bind_and_follow(
        registry,
        lambda degraded: _bound_and_followed(registry, cache, degraded),
        reader,
        grace=grace,
        restart_delay=restart_delay,
        max_restart_delay=max_restart_delay,
    )
    cache.followed_by(lambda: following.follower.watching)
    return following


def _bound_and_followed(
    registry: CollectionRegistry, cache: CallerAccessCache[Any], degraded: DegradedEvictions
) -> Callable[[], None]:
    """bind the per-caller cache; the remover also tells it nobody follows it any more.

    :param registry: the registry
    :ptype registry: CollectionRegistry
    :param cache: the per-caller cache
    :ptype cache: CallerAccessCache
    :param degraded: where unknown-reach rows are counted
    :ptype degraded: DegradedEvictions
    :return: the call that unbinds it and marks it unfollowed
    :rtype: Callable[[], None]
    """
    unbind = bind_caller_cache_to_access_tables(registry, cache, degraded=degraded)

    def remove() -> None:
        cache.followed_by(None)
        unbind()

    return remove


def _bind_and_follow(
    registry: CollectionRegistry,
    bind: Callable[[DegradedEvictions], Callable[[], None]],
    reader: GenerationWatcher,
    *,
    grace: timedelta,
    restart_delay: timedelta,
    max_restart_delay: timedelta,
) -> AccessTableFollowing:
    """bind a derived cache through ``bind`` and follow the access tables; see :func:`follow_access_tables`.

    :param registry: the registry
    :ptype registry: CollectionRegistry
    :param bind: registers the cache on the registry, counting into the record it is given, and
        returns the call that removes the registrations
    :ptype bind: Callable[[DegradedEvictions], Callable[[], None]]
    :param reader: reads and watches the epoch bucket
    :ptype reader: GenerationWatcher
    :param grace: how long a watch waits for an advance's rows
    :ptype grace: timedelta
    :param restart_delay: a failing watch's first wait
    :ptype restart_delay: timedelta
    :param max_restart_delay: the longest wait between attempts
    :ptype max_restart_delay: timedelta
    :return: the handle that stops both
    :rtype: AccessTableFollowing
    :raises RuntimeError: when the registry's invalidation listener is not running
    """
    if not registry.invalidation_listener_running:
        raise RuntimeError(
            "following the access tables needs the registry's invalidation listener running first: it hears "
            "the rows each watch judges against, and an advance judged without them reads as missed"
        )
    degraded = DegradedEvictions()
    unbind = bind(degraded)
    follower = TableFollower(
        registry,
        reader,
        ACCESS_TABLES,
        grace=grace,
        restart_delay=restart_delay,
        max_restart_delay=max_restart_delay,
    )
    follower.start()
    return AccessTableFollowing(follower=follower, degraded=degraded, _unbind=unbind)
