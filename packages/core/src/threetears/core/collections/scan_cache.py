"""L1-backed cache for visibility-filtered scans.

The by-pk Collection cache cannot express "which rows may this caller see" --
that predicate is a cross-table JOIN against ``role_assignments`` /
``group_members``. That limitation is real and correctly documented at each scan
site.

The error is the conclusion drawn from it. "Cannot use the by-pk cache" quietly
became "do not cache at all", so a stable, rarely-changing row set is re-fetched
from L3 on EVERY turn because the authorization decision over it is dynamic. On
cobalt-dev that meant two cross-table JOINs per agent turn, over NATS, against
distributed Yugabyte, under a 5s request timeout -- and when it blew that
timeout the agent proceeded with no governed knowledge at all.

This caches the scan RESULT under a caller-derived key and evicts it when any
table the scan reads is written. Storage is the pod's :class:`L1Backend` -- the
same machinery the by-pk cache uses -- NOT a module-level dict: a dict is
neither async nor thread safe, and more importantly it is per-process, so every
other pod would serve staleness with nothing to correct it. Multi-pod eviction
is the whole reason the invalidation broadcast exists.

**The dependency declaration is load-bearing for SECURITY, not just freshness.**
A scan whose result depends on ``role_assignments`` must be evicted when a grant
is revoked. Declaring only the data table would leave a revoked caller seeing
rows a broadcast for the grant never evicted.

**Nothing here ages; an entry is served only while every table it depends on is
followed** (epoch-task-06, the owner's ruling of 2026-10-09). A broadcast evicts the
entries of its table; a broadcast that never arrives is caught by the table's write
generation, which the registry follows, and the follower drops the table, scans
included. So an entry is stored and served only while every dependency is followed
with its watch running (``trusted``, supplied by the registry); otherwise the scan
reads L3 every time. A dependency nobody follows is a scan nobody may cache.

**A result is stored only if nothing it depends on was evicted while it was being
read.** Eviction drops what is stored; it cannot drop what has not been stored yet.
A scan that read L3 before a write committed, and reaches :meth:`ScanCache.put`
after that write's eviction ran, would otherwise store the pre-write result where
the eviction can no longer see it -- served with nothing left to correct it. So a reader takes a :class:`ScanReadToken` from
:meth:`ScanCache.begin_read` BEFORE it queries, and ``put`` refuses the store when
any table the token names has been evicted since. The eviction and the refusal
are counted per table, so an unrelated write never costs a reader its cache.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sqlalchemy import Column, MetaData, String, Table, Text
from threetears.observe import get_logger

from threetears.core.backends.schema_sql import json_default
from threetears.core import keyed
from threetears.core.keyed import Sort, inline_l1, run_inline
from threetears.core.eviction import Declared, Policy
from threetears.core.cache.l1_lru import L1_EVICTION_INFO
from threetears.core.frozen_state import frozen, with_entry

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from threetears.core.cache.base import L1Backend

__all__ = ["ScanCache", "ScanCacheKey", "ScanReadToken"]

log = get_logger(__name__)

SCAN_CACHE_METADATA = MetaData()

_scan_cache_table = Table(
    "collection_scan_cache",
    SCAN_CACHE_METADATA,
    Column("key", String, primary_key=True),
    Column("owner_table", Text, nullable=False),
    Column("depends_on", Text, nullable=False),
    Column("payload", Text, nullable=False),
    info={L1_EVICTION_INFO: Declared(Policy.EVICTABLE, "a cached scan; a miss scans again")},
)

#: each stored entry's dependencies, one row a (table, entry): an eviction reads the entries of its table by this key
#: (2026-11-14, no-whole-reads drive). It read every entry of ``collection_scan_cache`` and kept those whose JSON list
#: named the table, on every broadcast this pod hears; now it reads only the evicted table's own rows, by key
_DEPS_TABLE = "collection_scan_cache_deps"
_DEPS_KEY = ("dep_table", "key")

_scan_cache_deps_table = Table(
    _DEPS_TABLE,
    SCAN_CACHE_METADATA,
    Column("dep_table", String, primary_key=True),
    Column("key", String, primary_key=True),
    info={
        L1_EVICTION_INFO: Declared(
            Policy.KEEP,
            "which scans depend on which table: an invalidation reads it to find the scans to drop, so it must be whole",
        )
    },
)

#: the entries one statement of an eviction reads (``keyset_rows``' page)
_DEPS_PAGE = 1_000


class _Dependents:
    """Which tables of one L1 backend are known to have no stored scan depending on them.

    WHY (2026-11-15): every local write and every broadcast runs :meth:`ScanCache.drop_for_table`, and
    almost none of them is for a table a cached scan depends on. The keyed read of the dependency rows
    cost about 8.4 us a write on SQLite against about 2 us for the whole-cache select it replaced (2026-11-15;
    on a loaded 10-core host 69 us against 12 us), all of it the keyed helper building and running a statement that
    finds nothing. A table an eviction has emptied is marked clean here, and an eviction of a clean table skips L1
    entirely (about 0.3 us); a store unmarks every table it depends on BEFORE it writes its rows. Nothing here ages:
    a mark is made only by an eviction that found the table's rows gone, and removed by the next store naming it, so
    it is exactly as current as the rows (no TTL, no count). The speed test is
    ``test_scan_cache_write_cost.py``.

    Kept per :class:`ScanCache`, beside its eviction counts and on the same footing: one cache owns its backend's
    scan tables (a registry's, :attr:`~threetears.core.collections.registry.CollectionRegistry.scan_cache`), and
    the in-flight check in :meth:`ScanCache.put` already orders only that cache's reads against its own evictions.
    A table never marked (a new cache over a backend that already held rows) is read on its first eviction, as
    before.

    One lock orders a store against an eviction: a store's unmark-and-write and an eviction's read-delete-and-mark
    run whole, one at a time, so an eviction never marks a table clean past a row a store wrote while it read.

    :ivar clean: the tables known to have no dependency row in the backend
    :ivar lock: held across a store's rows and an eviction's read, delete and mark
    """

    __slots__ = ("clean", "lock")

    def __init__(self) -> None:
        """start knowing nothing clean: each table's first eviction reads it."""
        self.clean: set[str] = set()
        self.lock = threading.Lock()


def _never_trusted(_tables: Sequence[str]) -> bool:
    """the trust of a cache nothing follows: none, so it caches nothing.

    :param _tables: the tables a scan depends on
    :ptype _tables: Sequence[str]
    :return: ``False``
    :rtype: bool
    """
    return False


class ScanCacheKey:
    """The identity of one cached scan.

    A scan's result depends on WHO is asking as much as on what is stored, so
    the caller's identity is part of the key. Two callers with different grants
    must never share an entry.

    :ivar owner_table: the collection's table, for routing evictions
    :ivar parts: the caller-derived values that make this scan distinct
    """

    __slots__ = ("owner_table", "parts")

    def __init__(self, owner_table: str, *parts: Any) -> None:
        """build a scan key from the collection's table and the caller's identity.

        :param owner_table: the collection's table name
        :ptype owner_table: str
        :param parts: values that distinguish this scan (caller id, filters)
        :ptype parts: Any
        """
        self.owner_table = owner_table
        self.parts = tuple("" if p is None else str(p) for p in parts)

    def as_string(self) -> str:
        """render the key for storage.

        :return: a stable string key
        :rtype: str
        """
        return "|".join((self.owner_table, *self.parts))


@dataclass(frozen=True, slots=True)
class ScanReadToken:
    """What a scan depended on, and how often each dependency had been evicted, when it began.

    Taken from :meth:`ScanCache.begin_read` BEFORE the scan queries L3 and presented to
    :meth:`ScanCache.put` after. The eviction counts are the snapshot ``put`` compares
    against: a count that moved means a write to that table committed and was evicted
    while the scan was in flight, so its result may predate the write.

    :ivar depends_on: every table whose write must evict the stored result -- including
        the RBAC tables the visibility predicate reads
    :ivar evictions: the per-table eviction count at the start of the read, in
        ``depends_on`` order
    :ivar issuer: the cache that issued the token; a token is only meaningful to the
        cache whose counts it snapshotted
    """

    depends_on: tuple[str, ...]
    evictions: tuple[int, ...]
    issuer: ScanCache = field(repr=False, compare=False)


class ScanCache:
    """Stores visibility-scan results in L1, evicted by table dependency.

    :ivar _l1: the pod's L1 backend, or ``None`` (caching disabled)
    :ivar _trusted: whether every table of a dependency list is followed with its watch running
    :ivar _evictions: per-table count of :meth:`drop_for_table` calls, the clock a
        :class:`ScanReadToken` is checked against
    """

    __slots__ = ("_dependents", "_evictions", "_l1", "_trusted")

    def __init__(
        self, l1_backend: L1Backend | None, *, trusted: Callable[[Sequence[str]], bool] = _never_trusted
    ) -> None:
        """initialize the scan cache over an L1 backend.

        :param l1_backend: the pod's L1 backend; ``None`` disables caching
        :ptype l1_backend: L1Backend | None
        :param trusted: whether every one of the given tables is followed with its watch running
            (the registry's :meth:`~threetears.core.collections.registry.CollectionRegistry.tables_trusted`);
            by default nothing is, and nothing is cached
        :ptype trusted: Callable[[Sequence[str]], bool]
        """
        self._l1 = l1_backend
        self._trusted = trusted
        # In process, not in L1, on purpose: the counts order THIS pod's reads against
        # THIS pod's evictions, and both happen in this process. Every pod's evictions
        # reach it through `drop_for_table` -- its own writes via `publish_invalidation`,
        # everyone else's via the listener -- so a local count sees all of them.
        self._evictions: Mapping[str, int] = frozen()
        self._dependents = None if self._l1 is None else _Dependents()
        if self._l1 is not None and not (
            self._l1.has_table("collection_scan_cache") and self._l1.has_table(_DEPS_TABLE)
        ):
            self._l1.initialize(SCAN_CACHE_METADATA)

    def get(self, key: ScanCacheKey) -> list[dict[str, Any]] | None:
        """return the cached rows for ``key``, or ``None`` on a miss or while a dependency is not followed.

        :param key: the scan identity
        :ptype key: ScanCacheKey
        :return: the cached rows, or ``None``
        :rtype: list[dict[str, Any]] | None
        """
        hit: list[dict[str, Any]] | None = None
        if self._l1 is not None:
            row = self._l1.select_by_id("collection_scan_cache", key.as_string(), "key")
            if row is not None and self._trusted(json.loads(row["depends_on"])):
                hit = json.loads(row["payload"])
        return hit

    def begin_read(self, depends_on: tuple[str, ...]) -> ScanReadToken:
        """snapshot the eviction counts of ``depends_on``, before the scan queries.

        Must be called BEFORE the read, not after. A write evicted between this call and
        the read only costs one refused store; a write evicted between the read and this
        call would be invisible to the check in :meth:`put`, which is the defect the
        token exists to close.

        :param depends_on: every table whose write must evict the result --
            including the RBAC tables the visibility predicate reads
        :ptype depends_on: tuple[str, ...]
        :return: the token to present to :meth:`put`
        :rtype: ScanReadToken
        """
        return ScanReadToken(
            depends_on=depends_on,
            evictions=tuple(self._evictions.get(table, 0) for table in depends_on),
            issuer=self,
        )

    def put(
        self,
        key: ScanCacheKey,
        rows: list[dict[str, Any]],
        *,
        token: ScanReadToken,
    ) -> bool:
        """store a scan result under ``key``, unless a dependency was evicted during the read or is not followed.

        A refused store is not an error: the rows are still the caller's answer for this
        read, they are just not safe to answer the NEXT read with, because a write the
        eviction was about may be missing from them.

        :param key: the scan identity
        :ptype key: ScanCacheKey
        :param rows: the scan's result rows
        :ptype rows: list[dict[str, Any]]
        :param token: the token :meth:`begin_read` issued before the scan queried
        :ptype token: ScanReadToken
        :return: whether the result was stored
        :rtype: bool
        :raises ValueError: if ``token`` was issued by a different cache
        """
        if token.issuer is not self:
            raise ValueError(
                "scan read token was issued by a different ScanCache; its eviction counts mean nothing here"
            )
        current = tuple(self._evictions.get(table, 0) for table in token.depends_on)
        overtaken = current != token.evictions
        stored = False
        if overtaken:
            log.debug(
                "Scan result not cached: a dependency was evicted during the read",
                extra={
                    "extra_data": {
                        "owner_table": key.owner_table,
                        "evicted": [
                            table
                            for table, then, now in zip(token.depends_on, token.evictions, current, strict=True)
                            if then != now
                        ],
                    },
                },
            )
        elif self._l1 is not None and self._dependents is not None and self._trusted(token.depends_on):
            with self._dependents.lock:
                # unmarked before the rows are written, under the lock an eviction marks by (see _Dependents)
                self._dependents.clean.difference_update(token.depends_on)
                self._store(key, rows, token.depends_on)
            stored = True
        return stored

    def _store(self, key: ScanCacheKey, rows: list[dict[str, Any]], depends_on: tuple[str, ...]) -> None:
        """write one entry and its dependency rows.

        :param key: the scan identity
        :ptype key: ScanCacheKey
        :param rows: the scan's result rows
        :ptype rows: list[dict[str, Any]]
        :param depends_on: the tables whose write evicts it
        :ptype depends_on: tuple[str, ...]
        :return: nothing
        :rtype: None
        """
        if self._l1 is not None:
            self._l1.upsert(
                "collection_scan_cache",
                {
                    "key": key.as_string(),
                    "owner_table": key.owner_table,
                    "depends_on": json.dumps(list(depends_on)),
                    "payload": json.dumps(rows, default=json_default),
                },
                "key",
            )
            # a row a dependency (a scan names a few tables); a process-local cache's own index, no L3 row
            for table in depends_on:
                self._l1.upsert(_DEPS_TABLE, {"dep_table": table, "key": key.as_string()}, _DEPS_KEY)

    def drop_for_table(self, table: str) -> int:
        """evict every entry that declared ``table`` as a dependency.

        Called from the invalidation listener, so a write to ``concepts`` drops
        concept scans AND a write to ``role_assignments`` drops every scan whose
        visibility predicate reads it. The second is the security-relevant one.

        :param table: the table that was written
        :ptype table: str
        Counts the eviction whether or not anything was stored, so a scan that is still
        reading when this runs cannot store its result afterwards (see :meth:`put`).

        :return: how many entries were evicted
        :rtype: int
        """
        self._evictions = with_entry(self._evictions, table, self._evictions.get(table, 0) + 1)
        dropped = 0
        if self._l1 is not None and self._dependents is not None and self._l1.has_table(_DEPS_TABLE):
            # a table no stored scan depends on has nothing to read (WHY and numbers at _Dependents)
            with self._dependents.lock:
                if table not in self._dependents.clean:
                    dropped = self._drop_dependents(table)
                    self._dependents.clean.add(table)
        return dropped

    def _drop_dependents(self, table: str) -> int:
        """delete every entry depending on ``table``, with every dependency row naming them.

        :param table: the table that was written
        :ptype table: str
        :return: how many entries were deleted
        :rtype: int
        """
        dropped = 0
        if self._l1 is not None:
            # the table's entries, by the dependency key (``dep_table`` leads it), never every entry
            keys = [
                row["key"]
                for row in run_inline(
                    keyed.keyset_rows(
                        inline_l1(self._l1),
                        _DEPS_TABLE,
                        columns=["key"],
                        by={"dep_table": table},
                        order=[Sort("key", cast=None)],
                        page=_DEPS_PAGE,
                    )
                )
            ]
            # each entry's dependency rows go with it, so no row is left naming an entry that is gone
            entries = self._l1.select_batch("collection_scan_cache", keys, "key", columns=["key", "depends_on"])
            links = {(table, key) for key in keys}
            links |= {(dep, entry["key"]) for entry in entries for dep in json.loads(entry["depends_on"])}
            # one write for every entry: the SQLite L1 locks and commits once, not once an entry
            self._l1.delete_many_by_id_across(
                [("collection_scan_cache", keys, "key"), (_DEPS_TABLE, sorted(links), _DEPS_KEY)]
            )
            dropped = len(entries)
        return dropped
