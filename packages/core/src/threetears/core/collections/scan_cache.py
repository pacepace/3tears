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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sqlalchemy import Column, MetaData, String, Table, Text
from threetears.observe import get_logger

from threetears.core.backends.schema_sql import json_default

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
)


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

    __slots__ = ("_evictions", "_l1", "_trusted")

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
        self._evictions: dict[str, int] = {}
        if self._l1 is not None and not self._l1.has_table("collection_scan_cache"):
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
        elif self._l1 is not None and self._trusted(token.depends_on):
            self._l1.upsert(
                "collection_scan_cache",
                {
                    "key": key.as_string(),
                    "owner_table": key.owner_table,
                    "depends_on": json.dumps(list(token.depends_on)),
                    "payload": json.dumps(rows, default=json_default),
                },
                "key",
            )
            stored = True
        return stored

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
        self._evictions[table] = self._evictions.get(table, 0) + 1
        dropped = 0
        if self._l1 is not None and self._l1.has_table("collection_scan_cache"):
            rows = self._l1.execute_query("SELECT key, depends_on FROM collection_scan_cache")
            for row in rows:
                if table in json.loads(row["depends_on"]):
                    self._l1.delete_by_id("collection_scan_cache", row["key"], "key")
                    dropped += 1
        return dropped
