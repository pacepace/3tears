"""which database an L3 answers from: YugabyteDB, or Postgres.

A reader that spells a statement for one of them asks here, once a read. A direct pool, or anything with an
asyncpg-shaped ``fetch`` that reaches the database, is asked ``SELECT version()``: YugabyteDB's says ``-YB-``. The rail's proxy cannot ask that (the hub's broker refuses a pod
any function it does not model), so it reads the capability names the broker sends on every reply
(:meth:`~threetears.core.backends.nats_proxy.NatsProxyL3Backend.hub_capabilities`): a broker over YugabyteDB names
:data:`_YUGABYTE_DIALECT_CAPABILITY`. A broker built before it names none, and is answered as Postgres, which keeps
every statement in the form it had.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final

import asyncpg
from asyncpg.pool import PoolConnectionProxy

from threetears.observe import FrozenWeakKeys, get_logger
from threetears.observe.metrics import gauge

__all__ = ["check_index_hints", "known_on_yugabyte", "l3_on_yugabyte"]

log = get_logger(__name__)

#: the gauge saying a YugabyteDB connection has no pg_hint_plan (1) or has it (0): the keyed helpers' index hints
#: (``yugabyte_index_hint``) are then plain comments and YugabyteDB may plan a page off its declared index
_INDEX_HINT_UNAVAILABLE: Final = "threetears_yb_index_hint_unavailable"

#: a GUC pg_hint_plan registers when it is loaded (``shared_preload_libraries``); NULL (missing_ok) when it is not
_HINT_SETTING_SQL: Final = "SELECT current_setting('pg_hint_plan.enable_hint', true) AS enabled"

#: whether this process has logged a missing pg_hint_plan already: one error a process, the gauge every connection
_HINT_MISSING_LOGGED: bool = False

#: the method (``async def answers_from_yugabyte(self) -> bool``) a reader standing in front of another defines to say
#: which database the one behind it is, so no ``SELECT version()`` passes through it (a recorder would record it)
_DIALECT_OF_READER: Final = "answers_from_yugabyte"

#: the capability a hub's broker names in its replies' ``hub_capabilities`` when its database is YugabyteDB (the hub's
#: ``DIALECT_CAPABILITY_PREFIX`` and the database it reads from ``version()`` at start)
_YUGABYTE_DIALECT_CAPABILITY: Final = "dialect:yugabyte"

#: what ``version()`` says on YugabyteDB (``PostgreSQL 15.12-YB-2026.1.0.0-b0 ...``)
_YUGABYTE_VERSION_MARK: Final = "-YB-"


async def l3_on_yugabyte(l3: Any) -> bool:
    """whether ``l3`` answers from YugabyteDB, asked each time: a reader asks once a whole read (beside it, one round
    trip), and the answer is kept nowhere a second caller could fill it twice.

    :param l3: the rail's proxy (``hub_capabilities``), a reader in front of another (``answers_from_yugabyte``), or
        anything with an asyncpg-shaped ``fetch``
    :ptype l3: Any
    :return: whether its database is YugabyteDB
    :rtype: bool
    """
    # looked up on the class: a mock's instance answers any name it is asked
    if callable(getattr(type(l3), "hub_capabilities", None)):
        return _YUGABYTE_DIALECT_CAPABILITY in await l3.hub_capabilities()
    # a reader in front of another (a test's recorder of the statements a helper sends) answers for the one behind it,
    # asking it nothing it would record
    if callable(getattr(type(l3), _DIALECT_OF_READER, None)):
        return bool(await getattr(l3, _DIALECT_OF_READER)())
    rows = await l3.fetch("SELECT version() AS v")
    return bool(rows) and _YUGABYTE_VERSION_MARK in str(rows[0]["v"])


#: readers whose database was asked once (a pool, the rail's proxy): a reader's database does not change under it.
#: Held weakly, replaced whole
_ASKED: FrozenWeakKeys[Any, bool] = FrozenWeakKeys()

#: asyncpg pools whose database was asked once, by ``id``, each held beside its answer. A pool is slotted with no
#: ``__weakref__``, so it cannot sit in :data:`_ASKED`; before this every keyed read through a pool -- the hub's and
#: every service's L3 -- paid a ``SELECT version()`` round trip (plus the pool's reset on release). Held strongly, a
#: pool's id cannot be reused while it is here; a closed pool is dropped when the next pool is asked, so the map holds
#: the process's live pools (one or two) and never grows with closed ones. (test_known_dialect, 3tears work/h4-dialect)
_ASKED_POOLS: Mapping[int, tuple[asyncpg.Pool, bool]] = MappingProxyType({})


async def known_on_yugabyte(l3: Any) -> bool:
    """whether ``l3`` answers from YugabyteDB, with no round trip where it can be known locally: an asyncpg
    connection reads the ``server_version`` its server sent at startup (YugabyteDB's carries ``-YB-``); any other
    reader (a pool, the rail's proxy) is asked once, by :func:`l3_on_yugabyte`, and its answer kept for it (an
    asyncpg pool, which cannot be held weakly, in :data:`_ASKED_POOLS`).

    For a statement whose spelling differs by database on every call (a listing's page past its cursor), where
    :func:`l3_on_yugabyte`'s question each time would be a round trip a page.

    :param l3: the reader
    :ptype l3: Any
    :return: whether its database is YugabyteDB
    :rtype: bool
    """
    if isinstance(l3, asyncpg.Connection | PoolConnectionProxy):
        return _YUGABYTE_VERSION_MARK in str(l3.get_settings().server_version)
    if isinstance(l3, asyncpg.Pool):
        return await _pool_on_yugabyte(l3)
    if not type(l3).__weakrefoffset__:
        # a reader that cannot be held weakly is asked each time
        return await l3_on_yugabyte(l3)
    known = _ASKED.get(l3)
    if known is not None:
        return known
    answer = await l3_on_yugabyte(l3)
    _keep_answer(l3, answer)
    return answer


def _keep_answer(l3: Any, answer: bool) -> None:
    """keep ``l3``'s answer: :data:`_ASKED` replaced whole, read here after the question's await so an answer kept for
    another reader meanwhile stays. Two callers that both find a reader unasked each ask it once and keep the same
    answer: no object is built, nothing is registered twice.

    :param l3: the reader
    :ptype l3: Any
    :param answer: whether its database is YugabyteDB
    :ptype answer: bool
    :return: nothing
    :rtype: None
    """
    global _ASKED
    _ASKED = _ASKED.with_entry(l3, answer)


async def _pool_on_yugabyte(pool: asyncpg.Pool) -> bool:
    """whether ``pool`` answers from YugabyteDB, asked once a pool (:data:`_ASKED_POOLS`).

    :param pool: the pool
    :ptype pool: asyncpg.Pool
    :return: whether its database is YugabyteDB
    :rtype: bool
    """
    global _ASKED_POOLS
    held = _ASKED_POOLS.get(id(pool))
    if held is not None and held[0] is pool:
        return held[1]
    answer = await l3_on_yugabyte(pool)
    live = {key: other for key, other in _ASKED_POOLS.items() if not other[0].is_closing()}
    _ASKED_POOLS = MappingProxyType({**live, id(pool): (pool, answer)})
    return answer


async def check_index_hints(conn: Any) -> bool | None:
    """on a new YugabyteDB connection, whether pg_hint_plan is loaded, said by the gauge
    ``threetears_yb_index_hint_unavailable`` (1 missing, 0 loaded) and, the first time it is missing in a process,
    an error log.

    Asked once a connection, from the pool's ``init=`` hook (:func:`~threetears.core.collections.init_connection`),
    never on a read: the keyed helpers keep sending their hints, which a database without pg_hint_plan reads as
    comments, so a page is still answered, perhaps through another index -- what the gauge lets an alert see. A
    PostgreSQL connection, or anything with no asyncpg ``get_settings``, is not asked.

    :param conn: the new connection
    :ptype conn: Any
    :return: whether the hints are honoured; None when ``conn`` is not a YugabyteDB connection
    :rtype: bool | None
    """
    # looked up on the class: a mock's instance answers any name it is asked
    settings = getattr(type(conn), "get_settings", None)
    if not callable(settings) or _YUGABYTE_VERSION_MARK not in str(conn.get_settings().server_version):
        return None
    loaded = await conn.fetchval(_HINT_SETTING_SQL) is not None
    gauge(_INDEX_HINT_UNAVAILABLE, description="1 when a YugabyteDB connection has no pg_hint_plan loaded").set(
        0 if loaded else 1
    )
    global _HINT_MISSING_LOGGED
    if not loaded and not _HINT_MISSING_LOGGED:
        _HINT_MISSING_LOGGED = True
        log.error(
            "pg_hint_plan is not loaded on this YugabyteDB: the keyed helpers' index hints are ignored and a page may "
            "be planned off its declared index; add pg_hint_plan to ysql_pg_conf shared_preload_libraries",
            extra={"metric": _INDEX_HINT_UNAVAILABLE},
        )
    return loaded
