"""per-pod feature cache with a SQLite R-Tree bbox index.

a tile build asks one question: *which features fall inside this rectangle?*
without PostGIS there is no spatial index in the database, so answering it
from L3 means a bbox range scan per tile. adjacent tiles overlap heavily in
source features, so a pod building a run of neighbouring tiles asks
near-identical questions dozens of times over.

so features are cached per pod and indexed locally. SQLite's R-Tree module
is built in -- unlike SpatiaLite, which is a genuine local-dev build headache
on macOS -- and it lives alongside the collection's own managed table on the
same connection pool, exactly as the platform's caching rules require. this
is a :class:`BaseCollection` subclass rather than a bespoke wrapper around a
``SQLiteBackend`` for the same reason.

scope: the cache is *region*-scoped, not dataset-scoped. warming an entire
dataset into L1 is fine for a few thousand locations and impossible for
~180k precincts, so a pod holds what it has touched and fetches the rest --
and what it holds is bounded (:attr:`FeatureCache.max_cached_rows`), the
least recently read chunk going first. with no L1 bound it holds nothing:
every read is one loader call.

the R-Tree needs integer keys and features are keyed by
``(layer, source_version, feature_id)``, so a companion map table assigns a
surrogate rowid per feature key. that indirection is the price of the built-in
module; it is one extra table, not a second cache.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Awaitable, Callable
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
        rows and spatial index together, and is loaded again when next asked for
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

    #: the default bound on held row entries; see the ``max_cached_rows`` constructor argument
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
        # the chunks this pod has fully covered, each with the rows intersecting it, least
        # recently read first. coverage is what lets a hit be trusted: without it the R-Tree can
        # only say what is held, never what is complete. only filled with an L1 bound, and
        # bounded by ``_row_limit`` row entries (``_held``).
        self._chunks: OrderedDict[_ChunkMarker, dict[Any, dict[str, Any]]] = OrderedDict()
        self._held = 0
        # how many held chunks carry each feature, so one leaves the spatial index only when the
        # last chunk holding it is evicted
        self._holders: dict[tuple[str, int, Any], int] = {}

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

    @staticmethod
    def _feature_key(layer: str, source_version: int, feature_id: Any) -> str:
        return f"{layer}\x1f{source_version}\x1f{feature_id}"

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
        self.ensure_index()
        backend = self._l1
        if backend is None or not self._rtree_ready:
            return
        key = self._feature_key(layer, source_version, feature_id)
        conn = backend.get_connection()
        conn.execute(f"INSERT OR IGNORE INTO {self._map_table} (feature_key) VALUES (?)", (key,))
        rows = backend.execute_query(f"SELECT id FROM {self._map_table} WHERE feature_key = ?", (key,))
        if not rows:
            return
        conn.execute(
            f"INSERT OR REPLACE INTO {self._rtree_table} (id, min_x, max_x, min_y, max_y) VALUES (?, ?, ?, ?, ?)",
            (rows[0]["id"], bounds.min_lon, bounds.max_lon, bounds.min_lat, bounds.max_lat),
        )
        conn.commit()

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
        self.ensure_index()
        backend = self._l1
        if backend is None or not self._rtree_ready:
            return []
        prefix = f"{layer}\x1f{source_version}\x1f"
        rows = backend.execute_query(
            f"SELECT m.feature_key AS feature_key FROM {self._rtree_table} r "
            f"JOIN {self._map_table} m ON m.id = r.id "
            "WHERE r.max_x >= ? AND r.min_x <= ? AND r.max_y >= ? AND r.min_y <= ? "
            "AND m.feature_key LIKE ?",
            (bounds.min_lon, bounds.max_lon, bounds.min_lat, bounds.max_lat, f"{prefix}%"),
        )
        return [str(row["feature_key"]).split("\x1f", 2)[2] for row in rows]

    # ------------------------------------------------------------------
    # read path
    # ------------------------------------------------------------------

    @traced
    async def features_in_bbox(self, layer: str, source_version: int, bounds: BoundingBox) -> list[dict[str, Any]]:
        """return every source feature intersecting ``bounds``.

        **with no L1 bound this is one loader call for the rectangle**, and nothing is kept: there
        is nothing to hold rows in or evict them from, so a chunk sweep would only multiply the
        reads (live, a z0 tile swept 65,536 chunks and then read the world anyway).

        with an L1, the R-Tree alone cannot answer this. it can say which features a pod
        *holds* inside a rectangle, but not whether it holds *all* of them --
        and a tile built from a silently partial set is wrong rather than
        slow, then cached as immutable. so the cache tracks **coverage**: the
        chunks it has fully loaded.

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

        what is held is bounded (``max_cached_rows``) and evicted least recently read first. a
        read never depends on its own chunks surviving: it answers from the rows it gathered.

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
            return await self._loader(layer, source_version, bounds)
        chunks = self._chunks_for(bounds)
        uncovered = sum(1 for chunk in chunks if (layer, source_version, chunk.key) not in self._chunks)
        if uncovered > self.max_chunk_reads:
            rows = await self._loader(layer, source_version, bounds)
            self._hold_rectangle(layer, source_version, bounds, rows)
            return [row for row in rows if self._row_bounds(row).intersects(bounds)]
        found: dict[Any, dict[str, Any]] = {}
        for chunk in chunks:
            for feature_id, row in (await self._chunk_rows(layer, source_version, chunk)).items():
                found.setdefault(feature_id, row)
        return [row for row in found.values() if self._row_bounds(row).intersects(bounds)]

    def _chunks_for(self, bounds: BoundingBox) -> list[TileId]:
        """coarse tiles covering ``bounds``."""
        min_x, min_y, max_x, max_y = bounds_to_tile_range(bounds, self.chunk_zoom)
        return [TileId(z=self.chunk_zoom, x=x, y=y) for x in range(min_x, max_x + 1) for y in range(min_y, max_y + 1)]

    async def _chunk_rows(self, layer: str, source_version: int, chunk: TileId) -> dict[Any, dict[str, Any]]:
        """one chunk's rows by feature id: held ones (marked recently read), else loaded and held."""
        marker = (layer, source_version, chunk.key)
        held = self._chunks.get(marker)
        if held is not None:
            self._chunks.move_to_end(marker)
            return held
        rows = await self._loader(layer, source_version, tile_bounds(chunk))
        loaded: dict[Any, dict[str, Any]] = {}
        for row in rows:
            feature_id = row.get(self.feature_id_column)
            if feature_id is None:
                continue
            # keyed by the raw identity, not a string form: this dict only
            # dedupes rows and is never looked up by a caller's key, so a
            # UUID stays a UUID.
            loaded[feature_id] = row
        self._hold(marker, loaded)
        log.debug(
            "chunk loaded: layer=%s version=%s chunk=%s rows=%d",
            layer,
            source_version,
            chunk,
            len(rows),
        )
        return loaded

    def _hold_rectangle(self, layer: str, source_version: int, bounds: BoundingBox, rows: list[dict[str, Any]]) -> None:
        """cover every chunk ``bounds`` wholly contains, from one rectangle's rows.

        each contained chunk gets exactly the rows that intersect it, which is what loading it on
        its own would have returned -- an empty one included, which is coverage too (ocean). a
        rectangle carrying more rows than the bound holds nothing: holding it would only evict
        everything else and then most of itself.
        """
        if len(rows) > self._row_limit:
            log.debug(
                "wide read not held: layer=%s version=%s rows=%d exceed the bound of %d",
                layer,
                source_version,
                len(rows),
                self._row_limit,
            )
            return
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
        per_chunk: dict[tuple[int, int], dict[Any, dict[str, Any]]] = {xy: {} for xy in contained}
        for row in rows:
            feature_id = row.get(self.feature_id_column)
            if feature_id is None:
                continue
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
        for (x, y), held in per_chunk.items():
            self._hold((layer, source_version, (zoom, x, y)), held)
        log.debug(
            "wide read held: layer=%s version=%s rows=%d chunks covered=%d",
            layer,
            source_version,
            len(rows),
            len(per_chunk),
        )

    def _hold(self, marker: _ChunkMarker, rows: dict[Any, dict[str, Any]]) -> None:
        """record ``marker`` as covered by ``rows``, index what is new, then evict past the bound."""
        if marker in self._chunks:
            # two reads loaded the same chunk at once; the later one replaces the earlier
            self._release(marker)
        layer, source_version, _key = marker
        self._chunks[marker] = rows
        self._held += max(1, len(rows))
        fresh: list[tuple[Any, BoundingBox]] = []
        for feature_id, row in rows.items():
            holder = (layer, source_version, feature_id)
            count = self._holders.get(holder, 0)
            if count == 0:
                fresh.append((feature_id, self._row_bounds(row)))
            self._holders[holder] = count + 1
        self._index_many(layer, source_version, fresh)
        while self._held > self._row_limit and self._chunks:
            self._release(next(iter(self._chunks)))

    def _release(self, marker: _ChunkMarker) -> None:
        """evict one covered chunk, dropping from the spatial index each feature no other chunk holds."""
        rows = self._chunks.pop(marker)
        self._held -= max(1, len(rows))
        layer, source_version, _key = marker
        gone: list[Any] = []
        for feature_id in rows:
            holder = (layer, source_version, feature_id)
            count = self._holders.get(holder, 0) - 1
            if count > 0:
                self._holders[holder] = count
            else:
                self._holders.pop(holder, None)
                gone.append(feature_id)
        self._unindex_many(layer, source_version, gone)

    def _index_many(self, layer: str, source_version: int, features: list[tuple[Any, BoundingBox]]) -> None:
        """:meth:`index_feature` for many features, in one transaction."""
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

    def _unindex_many(self, layer: str, source_version: int, feature_ids: list[Any]) -> None:
        """drop features from the spatial index, in one transaction."""
        if not feature_ids:
            return
        backend = self._l1
        if backend is None or not self._rtree_ready:
            return
        keys = [(self._feature_key(layer, source_version, feature_id),) for feature_id in feature_ids]
        conn = backend.get_connection()
        conn.executemany(
            f"DELETE FROM {self._rtree_table} WHERE id IN (SELECT id FROM {self._map_table} WHERE feature_key = ?)",
            keys,
        )
        conn.executemany(f"DELETE FROM {self._map_table} WHERE feature_key = ?", keys)
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
