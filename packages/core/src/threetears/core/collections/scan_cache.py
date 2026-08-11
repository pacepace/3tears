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
rows until the TTL lapsed. The TTL is a backstop for anything the broadcast
misses, never the primary mechanism.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from sqlalchemy import Column, MetaData, String, Table, Text
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.core.cache.base import L1Backend

__all__ = ["ScanCache", "ScanCacheKey"]

log = get_logger(__name__)

SCAN_CACHE_METADATA = MetaData()

_scan_cache_table = Table(
    "collection_scan_cache",
    SCAN_CACHE_METADATA,
    Column("key", String, primary_key=True),
    Column("owner_table", Text, nullable=False),
    Column("depends_on", Text, nullable=False),
    Column("payload", Text, nullable=False),
    Column("stored_at_monotonic", Text, nullable=False),
)

#: Backstop only. Eviction is driven by the invalidation broadcast; this bounds
#: the damage from a broadcast that never arrives (a pod that missed a message,
#: a partial rollout). Deliberately short -- an RBAC change that slips the
#: broadcast must not outlive a coffee break.
DEFAULT_SCAN_TTL_SECONDS: float = 60.0


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


class ScanCache:
    """Stores visibility-scan results in L1, evicted by table dependency.

    :ivar _l1: the pod's L1 backend, or ``None`` (caching disabled)
    :ivar _ttl_seconds: backstop expiry
    """

    __slots__ = ("_l1", "_ttl_seconds")

    def __init__(self, l1_backend: L1Backend | None, *, ttl_seconds: float = DEFAULT_SCAN_TTL_SECONDS) -> None:
        """initialize the scan cache over an L1 backend.

        :param l1_backend: the pod's L1 backend; ``None`` disables caching
        :ptype l1_backend: L1Backend | None
        :param ttl_seconds: backstop expiry for entries the broadcast misses
        :ptype ttl_seconds: float
        """
        self._l1 = l1_backend
        self._ttl_seconds = ttl_seconds
        if self._l1 is not None and not self._l1.has_table("collection_scan_cache"):
            self._l1.initialize(SCAN_CACHE_METADATA)

    def get(self, key: ScanCacheKey, *, now_monotonic: float) -> list[dict[str, Any]] | None:
        """return the cached rows for ``key``, or ``None`` on miss or expiry.

        :param key: the scan identity
        :ptype key: ScanCacheKey
        :param now_monotonic: caller-supplied monotonic clock reading
        :ptype now_monotonic: float
        :return: the cached rows, or ``None``
        :rtype: list[dict[str, Any]] | None
        """
        hit: list[dict[str, Any]] | None = None
        if self._l1 is not None:
            row = self._l1.select_by_id("collection_scan_cache", key.as_string(), "key")
            if row is not None:
                stored_at = float(row["stored_at_monotonic"])
                if now_monotonic - stored_at <= self._ttl_seconds:
                    hit = json.loads(row["payload"])
                else:
                    # expired: drop it now rather than leave a tombstone that
                    # every later read has to re-evaluate.
                    self._l1.delete_by_id("collection_scan_cache", key.as_string(), "key")
        return hit

    def put(
        self,
        key: ScanCacheKey,
        rows: list[dict[str, Any]],
        *,
        depends_on: tuple[str, ...],
        now_monotonic: float,
    ) -> None:
        """store a scan result under ``key``.

        :param key: the scan identity
        :ptype key: ScanCacheKey
        :param rows: the scan's result rows
        :ptype rows: list[dict[str, Any]]
        :param depends_on: every table whose write must evict this entry --
            including the RBAC tables the visibility predicate reads
        :ptype depends_on: tuple[str, ...]
        :param now_monotonic: caller-supplied monotonic clock reading
        :ptype now_monotonic: float
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
                    "payload": json.dumps(rows, default=str),
                    "stored_at_monotonic": str(now_monotonic),
                },
                "key",
            )

    def drop_for_table(self, table: str) -> int:
        """evict every entry that declared ``table`` as a dependency.

        Called from the invalidation listener, so a write to ``concepts`` drops
        concept scans AND a write to ``role_assignments`` drops every scan whose
        visibility predicate reads it. The second is the security-relevant one.

        :param table: the table that was written
        :ptype table: str
        :return: how many entries were evicted
        :rtype: int
        """
        dropped = 0
        if self._l1 is not None and self._l1.has_table("collection_scan_cache"):
            rows = self._l1.execute_query("SELECT key, depends_on FROM collection_scan_cache")
            for row in rows:
                if table in json.loads(row["depends_on"]):
                    self._l1.delete_by_id("collection_scan_cache", row["key"], "key")
                    dropped += 1
        return dropped
