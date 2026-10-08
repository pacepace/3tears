"""per-pod feature cache with a SQLite R-Tree bbox index.

a tile build asks one question: *which features fall inside this rectangle?*
without PostGIS there is no spatial index in the database, so answering it
from L3 means a bbox range scan per tile. adjacent tiles overlap heavily in
source features, so a pod building a run of neighbouring tiles asks
near-identical questions dozens of times over.

so features are cached per pod and indexed locally, and the index answers
the read: a covered region's rows come back from an R-Tree query, then an
exact rectangle test (the R-Tree stores 32-bit floats, rounded outward, so it
may over-answer and never under-answers). SQLite's R-Tree module
is built in -- unlike SpatiaLite, which is a genuine local-dev build headache
on macOS -- and it lives alongside the collection's own managed table on the
same connection pool, exactly as the platform's caching rules require. this
is a :class:`BaseCollection` subclass rather than a bespoke wrapper around a
``SQLiteBackend`` for the same reason.

scope: the cache is *region*-scoped, not dataset-scoped. warming an entire
dataset into L1 is fine for a few thousand locations and impossible for
~180k precincts, so a pod holds what it has touched and fetches the rest --
and what it holds is bounded (:attr:`FeatureCache.max_cached_rows`), the
least recently read chunk going first, rows and index entries together. with
no L1 bound it holds nothing: every read is one loader call.

the R-Tree needs integer keys and features are keyed by
``(layer, source_version, feature_id)``, so a companion map table assigns a
surrogate rowid per feature key. that indirection is the price of the built-in
module; it is one extra table, not a second cache. each key also carries a
token minted per instance: several caches of one scope share the L1's tables
(the hub builds one per layer), and each keeps its rows in memory, so each
indexes and evicts only its own entries. an instance's entries are deleted
when it is garbage-collected.
"""

from __future__ import annotations

import secrets
import weakref
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, ClassVar

from threetears.core.collections.base import BaseCollection
from threetears.core.entities.base import BaseEntity
from threetears.geo.scope import check_cache_scope
from threetears.geo.tiles import BoundingBox, TileId, bounds_to_tile_range, tile_bounds
from threetears.observe import get_logger, traced

__all__ = ["FeatureCache", "FeatureEntity", "FeatureLoader"]

log = get_logger(__name__)

#: signature of the L3 read this cache sits in front of: given a layer,
#: source generation and rectangle, return the source rows inside it. the
#: caller owns the query, because only the caller knows the datasource.
FeatureLoader = Callable[[str, int, BoundingBox], Awaitable[list[dict[str, Any]]]]

#: one covered chunk: ``(layer, source_version, chunk tile key)``
_ChunkMarker = tuple[str, int, tuple[int, int, int]]


class FeatureEntity(BaseEntity):
    """one cached source feature row."""

    primary_key_field = "feature_id"


class FeatureCache(BaseCollection[FeatureEntity]):
    """L1 cache of source features with a local R-Tree bbox index.

    :param loader: async callable fetching source rows for a rectangle
    :ptype loader: FeatureLoader
    :param bounds_of: extracts a row's bounding rectangle. supplied by the
        caller because only the layer declaration knows which column holds
        geometry, and whether it is WKB or a lon/lat pair
    :ptype bounds_of: Callable[[dict[str, Any]], BoundingBox]
    :param feature_id_column: column holding each row's stable identity
    :ptype feature_id_column: str
    :param cache_scope: the tile source this cache serves; names its tables, so another
        source's features of the same layer name are never read (see :mod:`threetears.geo.scope`)
    :ptype cache_scope: str
    :param max_cached_rows: the most row entries this pod holds across its covered chunks (a
        row in two chunks counts twice, an empty chunk once); ``None`` takes
        :attr:`max_cached_rows` from the class. past it the least recently read chunk is evicted,
        rows and spatial index together, and is loaded again when next asked for. one read may
        newly hold at most half of it, counted across all the chunks it loads: the chunks past
        that are answered from the read's own rows and not held, so no read -- a dense chunk, many
        small ones, or a wide rectangle -- flushes the working set
    :ptype max_cached_rows: int | None
    :raises ValueError: for a cache scope that is not a lowercase identifier, or a
        ``max_cached_rows`` below 1
    """

    primary_key_column: tuple[str, ...] = ("layer", "source_version", "feature_id")

    #: zoom of the chunks this cache loads and tracks coverage by. z8 is
    #: roughly metro-sized: coarse enough that a run of z12-z14 tiles shares
    #: one chunk, fine enough that a single chunk is not a whole country's
    #: worth of geometry.
    chunk_zoom: ClassVar[int] = 8

    #: the most uncovered chunks one read loads one at a time. a rectangle spanning more -- a z3
    #: tile is 32x32 of them, z0 is 65,536 -- is ONE loader call for the rectangle instead, and the
    #: chunks it fully contains are covered from its rows. sixteen keeps a z12-z14 tile, which
    #: spans one to four chunks, on the chunk path that lets its neighbours share the load.
    max_chunk_reads: ClassVar[int] = 16

    #: the default bound on held row entries; see the ``max_cached_rows`` constructor argument.
    #: a placeholder, not a measurement: it holds a few hundred z8 chunks of dense precinct-level
    #: geometry, or about half of a 180k-row layer, and every entry is one row's geometry and
    #: attributes in memory. size it to the pod's memory and the layer's row size.
    max_cached_rows: ClassVar[int] = 100_000

    def __init__(
        self,
        *args: Any,
        loader: FeatureLoader,
        bounds_of: Callable[[dict[str, Any]], BoundingBox],
        feature_id_column: str,
        cache_scope: str,
        max_cached_rows: int | None = None,
        **kwargs: Any,
    ) -> None:
        # before the base constructor: it reads table_name to resolve and register every tier
        self._cache_scope = check_cache_scope(cache_scope)
        row_limit = type(self).max_cached_rows if max_cached_rows is None else max_cached_rows
        if row_limit < 1:
            raise ValueError(f"max_cached_rows must be at least 1, got {row_limit}")
        super().__init__(*args, **kwargs)
        self._loader = loader
        self._bounds_of = bounds_of
        self.feature_id_column = feature_id_column
        self._rtree_ready = False
        self._row_limit = row_limit
        # the chunks this pod has fully covered, each with the ids of the features intersecting
        # it, least recently read first. coverage is what lets a hit be trusted: without it the
        # R-Tree can only say what is held, never what is complete. only filled with an L1 bound,
        # and bounded by ``_row_limit`` row entries (``_held``).
        self._chunks: OrderedDict[_ChunkMarker, tuple[Any, ...]] = OrderedDict()
        self._held = 0
        # each held feature's row by its R-Tree feature key, and how many held chunks carry it: a
        # feature leaves the rows and the index only when the last chunk holding it is evicted
        self._rows: dict[str, dict[str, Any]] = {}
        self._holders: dict[str, int] = {}
        # chunks a read in progress depends on, which eviction skips until that read has answered
        self._pins: dict[_ChunkMarker, int] = {}
        # leads every R-Tree key this instance writes, so instances sharing the tables never read
        # or evict each other's entries
        self._instance_key = secrets.token_hex(6)

    @property
    def table_name(self) -> str:
        return f"geo_features_{self._cache_scope}"

    @property
    def entity_class(self) -> type[FeatureEntity]:
        return FeatureEntity

    # ------------------------------------------------------------------
    # R-Tree companion
    # ------------------------------------------------------------------

    @property
    def _rtree_table(self) -> str:
        return f"{self.table_name}_rtree"

    @property
    def _map_table(self) -> str:
        return f"{self.table_name}_rtree_map"

    def ensure_index(self) -> None:
        """create the R-Tree and its key map if absent.

        idempotent and cheap after the first call. built lazily rather than at
        construction so a collection that never runs a spatial query never
        pays for the virtual table.
        """
        if self._rtree_ready:
            return
        backend = self._l1
        if backend is None:
            # L1 is optional in the framework; without it there is nothing to
            # index and every lookup falls through to the loader.
            log.debug("no L1 backend bound to %s; spatial index disabled", self.table_name)
            return
        conn = backend.get_connection()
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {self._rtree_table} USING rtree(id, min_x, max_x, min_y, max_y)"
        )
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self._map_table} ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  feature_key TEXT NOT NULL UNIQUE"
            ")"
        )
        conn.commit()
        self._rtree_ready = True
        weakref.finalize(
            self, _drop_index_entries, backend, self._rtree_table, self._map_table, f"{self._instance_key}\x1f"
        )

    def _feature_key(self, layer: str, source_version: int, feature_id: Any) -> str:
        return f"{self._instance_key}\x1f{layer}\x1f{source_version}\x1f{feature_id}"

    def index_feature(self, layer: str, source_version: int, feature_id: Any, bounds: BoundingBox) -> None:
        """record one feature's bounds in the R-Tree.

        :param layer: geo layer name
        :ptype layer: str
        :param source_version: generation the row belongs to
        :ptype source_version: int
        :param feature_id: the feature's stable identity
        :ptype feature_id: Any
        :param bounds: the feature's bounding rectangle
        :ptype bounds: BoundingBox
        """
        self._index_many(layer, source_version, [(feature_id, bounds)])

    def indexed_keys_in_bbox(self, layer: str, source_version: int, bounds: BoundingBox) -> list[str]:
        """return cached feature ids whose bounds intersect ``bounds``.

        the R-Tree answers on *overlap*, which is the same edge-inclusive
        predicate :meth:`BoundingBox.intersects` and the L3 bbox-column query
        use -- all three have to agree or a feature appears in one path and
        not another.

        :param layer: geo layer name
        :ptype layer: str
        :param source_version: generation to read
        :ptype source_version: int
        :param bounds: query rectangle
        :ptype bounds: BoundingBox
        :return: feature ids present in this pod's cache and inside the rectangle
        :rtype: list[str]
        """
        return [key.split("\x1f", 3)[3] for key in self._keys_in_bbox(layer, source_version, bounds)]

    def _keys_in_bbox(self, layer: str, source_version: int, bounds: BoundingBox) -> list[str]:
        """the R-Tree feature keys of one layer and generation whose bounds overlap ``bounds``."""
        self.ensure_index()
        backend = self._l1
        if backend is None or not self._rtree_ready:
            return []
        prefix = self._feature_key(layer, source_version, "")
        rows = backend.execute_query(
            f"SELECT m.feature_key AS feature_key FROM {self._rtree_table} r "
            f"JOIN {self._map_table} m ON m.id = r.id "
            "WHERE r.max_x >= ? AND r.min_x <= ? AND r.max_y >= ? AND r.min_y <= ? "
            # a prefix compare, not LIKE: a layer name may carry the LIKE wildcards % and _
            "AND substr(m.feature_key, 1, ?) = ?",
            (bounds.min_lon, bounds.max_lon, bounds.min_lat, bounds.max_lat, len(prefix), prefix),
        )
        return [str(row["feature_key"]) for row in rows]

    # ------------------------------------------------------------------
    # read path
    # ------------------------------------------------------------------

    @traced
    async def features_in_bbox(self, layer: str, source_version: int, bounds: BoundingBox) -> list[dict[str, Any]]:
        """return every source feature intersecting ``bounds``.

        every path answers by one rule: rows with no feature id are dropped, and every row is
        tested against the rectangle exactly. a tile built from the answer is cached as
        immutable, so the same rectangle must yield the same features whichever path serves it.

        **with no L1 bound this is one loader call for the rectangle**, and nothing is kept: there
        is nothing to hold rows in or evict them from, so a chunk sweep would only multiply the
        reads.

        with an L1, the R-Tree alone cannot answer this. it can say which features a pod
        *holds* inside a rectangle, but not whether it holds *all* of them --
        and a tile built from a silently partial set is wrong rather than
        slow, then cached as immutable. so the cache tracks **coverage**: the
        chunks it has fully loaded. once every chunk under the rectangle is covered, the R-Tree
        answers which held features overlap it.

        the chunk is the same trick the whole design rests on, applied one
        level up. rather than loading each tile's own rectangle, the cache
        loads the coarse tile containing it (:data:`chunk_zoom`) and records
        that chunk as covered. a run of neighbouring tiles then falls inside
        one already-loaded chunk, so the first tile of a region pays one L3
        read and its neighbours pay none -- which is the actual saving,
        because adjacent tiles overlap almost entirely in source features.

        a rectangle spanning several chunks loads each uncovered chunk it touches, so correctness
        never depends on the caller's rectangle happening to fit -- up to
        :attr:`max_chunk_reads` of them. past that the rectangle is read in ONE loader call, and
        the chunks it wholly contains are covered from those rows; a chunk it only overlaps (its
        edge) is not, since the rows outside the rectangle were never read.

        what is held is bounded (``max_cached_rows``) and evicted least recently read first, never
        while a read still depends on it. one read newly holds at most half the bound, summed over
        every chunk it loads: the chunks past that, and a wide rectangle whose entries would pass
        it, are answered from the read's own rows and not held.

        :param layer: geo layer name
        :ptype layer: str
        :param source_version: generation to read
        :ptype source_version: int
        :param bounds: query rectangle
        :ptype bounds: BoundingBox
        :return: source rows intersecting the rectangle
        :rtype: list[dict[str, Any]]
        """
        if self.l1_backend is None:
            rows = self._identified(await self._loader(layer, source_version, bounds))
            return self._within(rows.values(), bounds)
        chunks = self._chunks_for(bounds)
        uncovered = sum(1 for chunk in chunks if (layer, source_version, chunk.key) not in self._chunks)
        if uncovered > self.max_chunk_reads:
            rows = self._identified(await self._loader(layer, source_version, bounds))
            self._hold_rectangle(layer, source_version, bounds, rows)
            self._evict()
            return self._within(rows.values(), bounds)
        pinned: list[_ChunkMarker] = []
        # rows of chunks this read cannot hold, answered from here instead of from the index
        transient: dict[str, dict[str, Any]] = {}
        # entries this read has newly held: together they may not pass half the bound
        claimed = 0
        try:
            for chunk in chunks:
                marker = (layer, source_version, chunk.key)
                if marker in self._chunks:
                    self._chunks.move_to_end(marker)
                else:
                    loaded = self._identified(await self._loader(layer, source_version, tile_bounds(chunk)))
                    log.debug(
                        "chunk loaded: layer=%s version=%s chunk=%s rows=%d",
                        layer,
                        source_version,
                        chunk,
                        len(loaded),
                    )
                    weight = max(1, len(loaded))
                    if claimed + weight > self._read_cap:
                        log.debug(
                            "chunk not held: layer=%s version=%s chunk=%s rows=%d would take this read's "
                            "entries past half the bound of %d",
                            layer,
                            source_version,
                            chunk,
                            len(loaded),
                            self._row_limit,
                        )
                        for feature_id, row in loaded.items():
                            transient[self._feature_key(layer, source_version, feature_id)] = row
                        continue
                    claimed += weight
                    self._index_many(layer, source_version, self._hold(marker, loaded))
                self._pins[marker] = self._pins.get(marker, 0) + 1
                pinned.append(marker)
            found = dict(transient)
            for key in self._keys_in_bbox(layer, source_version, bounds):
                row = self._rows.get(key)
                if row is not None:
                    found.setdefault(key, row)
            return self._within(found.values(), bounds)
        finally:
            for marker in pinned:
                count = self._pins[marker] - 1
                if count:
                    self._pins[marker] = count
                else:
                    del self._pins[marker]
            self._evict()

    @property
    def _read_cap(self) -> int:
        """the most row entries one read may hold: half the bound, so it never flushes the rest."""
        return max(1, self._row_limit // 2)

    def _identified(self, rows: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
        """``rows`` by feature id, dropping any with none: a row with no identity is not a feature."""
        found: dict[Any, dict[str, Any]] = {}
        for row in rows:
            feature_id = row.get(self.feature_id_column)
            if feature_id is None:
                continue
            # keyed by the raw identity, not a string form, so a UUID stays a UUID
            found.setdefault(feature_id, row)
        if len(found) < len(rows):
            log.debug("dropped %d source rows with no %s", len(rows) - len(found), self.feature_id_column)
        return found

    def _within(self, rows: Iterable[dict[str, Any]], bounds: BoundingBox) -> list[dict[str, Any]]:
        """the rows whose own bounds intersect ``bounds``, edge-inclusive."""
        return [row for row in rows if self._row_bounds(row).intersects(bounds)]

    def _chunks_for(self, bounds: BoundingBox) -> list[TileId]:
        """coarse tiles covering ``bounds``."""
        min_x, min_y, max_x, max_y = bounds_to_tile_range(bounds, self.chunk_zoom)
        return [TileId(z=self.chunk_zoom, x=x, y=y) for x in range(min_x, max_x + 1) for y in range(min_y, max_y + 1)]

    def _hold_rectangle(
        self, layer: str, source_version: int, bounds: BoundingBox, rows: dict[Any, dict[str, Any]]
    ) -> None:
        """cover every chunk ``bounds`` wholly contains, from one rectangle's rows.

        each contained chunk gets exactly the rows that intersect it, which is what loading it on
        its own would have returned -- an empty one included, which is coverage too (ocean). the
        whole rectangle is held or none of it, counted as the bound counts, and indexed in one
        transaction.
        """
        zoom = self.chunk_zoom
        min_x, min_y, max_x, max_y = bounds_to_tile_range(bounds, zoom)
        contained: dict[tuple[int, int], BoundingBox] = {}
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if (layer, source_version, (zoom, x, y)) in self._chunks:
                    continue
                chunk_bounds = tile_bounds(TileId(z=zoom, x=x, y=y))
                if (
                    chunk_bounds.min_lon >= bounds.min_lon
                    and chunk_bounds.max_lon <= bounds.max_lon
                    and chunk_bounds.min_lat >= bounds.min_lat
                    and chunk_bounds.max_lat <= bounds.max_lat
                ):
                    contained[(x, y)] = chunk_bounds
        if not contained:
            return
        if len(contained) > self._read_cap:
            # every contained chunk costs at least one entry, so this read cannot fit
            self._log_wide_not_held(layer, source_version, len(rows), len(contained))
            return
        per_chunk: dict[tuple[int, int], dict[Any, dict[str, Any]]] = {xy: {} for xy in contained}
        for feature_id, row in rows.items():
            row_bounds = self._row_bounds(row)
            row_min_x, row_min_y, row_max_x, row_max_y = bounds_to_tile_range(row_bounds, zoom)
            # one chunk wider on every side: a row whose edge lies exactly on a chunk boundary
            # touches the chunk beyond it too, and the loader's edge-inclusive query returns it
            # there, so the held set must as well
            for x in range(row_min_x - 1, row_max_x + 2):
                for y in range(row_min_y - 1, row_max_y + 2):
                    chunk_bounds = contained.get((x, y))
                    if chunk_bounds is not None and row_bounds.intersects(chunk_bounds):
                        per_chunk[(x, y)][feature_id] = row
        entries = sum(max(1, len(held)) for held in per_chunk.values())
        if entries > self._read_cap:
            self._log_wide_not_held(layer, source_version, len(rows), entries)
            return
        fresh: list[tuple[Any, BoundingBox]] = []
        for (x, y), held in per_chunk.items():
            fresh.extend(self._hold((layer, source_version, (zoom, x, y)), held))
        self._index_many(layer, source_version, fresh)
        log.debug(
            "wide read held: layer=%s version=%s rows=%d chunks covered=%d entries=%d",
            layer,
            source_version,
            len(rows),
            len(per_chunk),
            entries,
        )

    def _log_wide_not_held(self, layer: str, source_version: int, rows: int, entries: int) -> None:
        log.debug(
            "wide read not held: layer=%s version=%s rows=%d need %d entries, over half the bound of %d",
            layer,
            source_version,
            rows,
            entries,
            self._row_limit,
        )

    def _hold(self, marker: _ChunkMarker, rows: dict[Any, dict[str, Any]]) -> list[tuple[Any, BoundingBox]]:
        """record ``marker`` as covered by ``rows``; return the features new to the pod, to index.

        does not evict: the caller evicts once its read no longer depends on what it holds.
        """
        if marker in self._chunks:
            # two reads loaded the same chunk at once; the later one replaces the earlier
            self._release(marker)
        layer, source_version, _key = marker
        self._chunks[marker] = tuple(rows)
        self._held += max(1, len(rows))
        fresh: list[tuple[Any, BoundingBox]] = []
        for feature_id, row in rows.items():
            key = self._feature_key(layer, source_version, feature_id)
            count = self._holders.get(key, 0)
            if count == 0:
                self._rows[key] = row
                fresh.append((feature_id, self._row_bounds(row)))
            self._holders[key] = count + 1
        return fresh

    def _evict(self) -> None:
        """release least recently read chunks no read depends on, until within the bound."""
        evicted = 0
        while self._held > self._row_limit:
            victim = next((marker for marker in self._chunks if marker not in self._pins), None)
            if victim is None:
                break
            self._release(victim)
            evicted += 1
        if evicted:
            log.debug(
                "feature cache evicted %d chunk(s); holding %d entries in %d chunk(s), bound %d",
                evicted,
                self._held,
                len(self._chunks),
                self._row_limit,
            )

    def _release(self, marker: _ChunkMarker) -> None:
        """evict one covered chunk, dropping each feature no other held chunk carries."""
        feature_ids = self._chunks.pop(marker)
        self._held -= max(1, len(feature_ids))
        layer, source_version, _key = marker
        gone: list[str] = []
        for feature_id in feature_ids:
            key = self._feature_key(layer, source_version, feature_id)
            count = self._holders.get(key, 0) - 1
            if count > 0:
                self._holders[key] = count
            else:
                self._holders.pop(key, None)
                self._rows.pop(key, None)
                gone.append(key)
        self._unindex_keys(gone)

    def _index_many(self, layer: str, source_version: int, features: list[tuple[Any, BoundingBox]]) -> None:
        """record features' bounds in the R-Tree, in one transaction."""
        if not features:
            return
        self.ensure_index()
        backend = self._l1
        if backend is None or not self._rtree_ready:
            return
        keys = [(self._feature_key(layer, source_version, feature_id),) for feature_id, _bounds in features]
        conn = backend.get_connection()
        conn.executemany(f"INSERT OR IGNORE INTO {self._map_table} (feature_key) VALUES (?)", keys)
        conn.executemany(
            f"INSERT OR REPLACE INTO {self._rtree_table} (id, min_x, max_x, min_y, max_y) "
            f"SELECT id, ?, ?, ?, ? FROM {self._map_table} WHERE feature_key = ?",
            [
                (b.min_lon, b.max_lon, b.min_lat, b.max_lat, key)
                for (_feature_id, b), (key,) in zip(features, keys, strict=True)
            ],
        )
        conn.commit()

    def _unindex_keys(self, keys: list[str]) -> None:
        """drop features from the spatial index by feature key, in one transaction."""
        if not keys:
            return
        backend = self._l1
        if backend is None or not self._rtree_ready:
            return
        params = [(key,) for key in keys]
        conn = backend.get_connection()
        conn.executemany(
            f"DELETE FROM {self._rtree_table} WHERE id IN (SELECT id FROM {self._map_table} WHERE feature_key = ?)",
            params,
        )
        conn.executemany(f"DELETE FROM {self._map_table} WHERE feature_key = ?", params)
        conn.commit()

    def _row_bounds(self, row: dict[str, Any]) -> BoundingBox:
        """the row's own bounding rectangle, via the caller-supplied extractor."""
        return self._bounds_of(row)

    # ------------------------------------------------------------------
    # BaseCollection contract
    # ------------------------------------------------------------------

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        """single-feature L3 read.

        not used by the tile path, which reads by rectangle rather than by
        id. present because the base class requires it, and returning ``None``
        is the honest answer: this cache has no by-id L3 query to issue, since
        the loader is rectangle-shaped.
        """
        return None

    async def save_to_store(self, data: dict[str, Any], original_timestamp: Any = None, *, conn: Any = None) -> int:
        """no-op: source features are owned by the datasource, not by this cache.

        writing here would mean this cache had become a second copy of the
        source of truth.
        """
        return 0

    async def delete_from_store(self, entity_id: Any) -> None:
        """no-op, for the same reason as :meth:`save_to_store`."""
        return None

    def serialize(self, data: dict[str, Any]) -> bytes:
        from threetears.core.serialization import serialize_to_json

        return serialize_to_json(data)

    def deserialize(self, data: bytes) -> dict[str, Any]:
        from threetears.core.serialization import deserialize_from_json

        # no declared field types: a cached feature row's columns are whatever
        # the datasource's layer declaration selected, which varies per layer
        # and is not known to this class. values round-trip as their JSON
        # types, which is sufficient -- geometry travels as WKB hex and
        # attributes are already coerced to MVT scalars downstream.
        result: dict[str, Any] = deserialize_from_json(data, {})
        return result


def _drop_index_entries(backend: Any, rtree_table: str, map_table: str, prefix: str) -> None:
    """delete a collected :class:`FeatureCache`'s R-Tree entries; best effort, never raises.

    run by :func:`weakref.finalize`, so it may run on any thread -- whichever one triggers the
    collection -- during interpreter shutdown, or after the L1 has closed, and SQLite may refuse
    the connection from there. the worst case is orphaned entries: rows in an in-memory table that
    no live cache reads, since every key carries the collected instance's token.
    """
    try:
        conn = backend.get_connection()
        params = (len(prefix), prefix)
        conn.execute(
            f"DELETE FROM {rtree_table} WHERE id IN (SELECT id FROM {map_table} WHERE substr(feature_key, 1, ?) = ?)",
            params,
        )
        conn.execute(f"DELETE FROM {map_table} WHERE substr(feature_key, 1, ?) = ?", params)
        conn.commit()
    # prawduct:allow prawduct/broad-except -- a finalizer must not raise; whatever the L1 says on
    # the way down, the entries are unread and the table is in memory
    except Exception:  # noqa: BLE001
        log.debug("could not drop a collected feature cache's index entries", exc_info=True)
