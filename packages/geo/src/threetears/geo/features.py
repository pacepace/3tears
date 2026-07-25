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
~180k precincts, so a pod holds what it has touched and fetches the rest.

the R-Tree needs integer keys and features are keyed by
``(layer, source_version, feature_id)``, so a companion map table assigns a
surrogate rowid per feature key. that indirection is the price of the built-in
module; it is one extra table, not a second cache.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, ClassVar

from threetears.core.collections.base import BaseCollection
from threetears.core.entities.base import BaseEntity
from threetears.geo.tiles import BoundingBox, TileId, bounds_to_tile_range, tile_bounds
from threetears.observe import get_logger, traced

__all__ = ["FeatureCache", "FeatureEntity", "FeatureLoader"]

log = get_logger(__name__)

#: signature of the L3 read this cache sits in front of: given a layer,
#: source generation and rectangle, return the source rows inside it. the
#: caller owns the query, because only the caller knows the datasource.
FeatureLoader = Callable[[str, int, BoundingBox], Awaitable[list[dict[str, Any]]]]


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
    """

    primary_key_column: tuple[str, ...] = ("layer", "source_version", "feature_id")

    #: zoom of the chunks this cache loads and tracks coverage by. z8 is
    #: roughly metro-sized: coarse enough that a run of z12-z14 tiles shares
    #: one chunk, fine enough that a single chunk is not a whole country's
    #: worth of geometry.
    chunk_zoom: ClassVar[int] = 8

    def __init__(
        self,
        *args: Any,
        loader: FeatureLoader,
        bounds_of: Callable[[dict[str, Any]], BoundingBox],
        feature_id_column: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._loader = loader
        self._bounds_of = bounds_of
        self.feature_id_column = feature_id_column
        self._rtree_ready = False
        # chunks fully loaded by this pod, and the rows they brought. the
        # coverage set is what lets a hit be trusted: without it the R-Tree
        # can only say what is held, never what is complete.
        self._covered: set[tuple[str, int, tuple[int, int, int]]] = set()
        self._rows: dict[tuple[str, int], dict[Any, dict[str, Any]]] = {}

    @property
    def table_name(self) -> str:
        return "geo_features"

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

        the R-Tree alone cannot answer this. it can say which features a pod
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

        a rectangle spanning more than one chunk loads each uncovered chunk
        it touches, so correctness never depends on the caller's rectangle
        happening to fit.

        :param layer: geo layer name
        :ptype layer: str
        :param source_version: generation to read
        :ptype source_version: int
        :param bounds: query rectangle
        :ptype bounds: BoundingBox
        :return: source rows intersecting the rectangle
        :rtype: list[dict[str, Any]]
        """
        chunks = self._chunks_for(bounds)
        for chunk in chunks:
            await self._ensure_chunk_loaded(layer, source_version, chunk)
        cached = self._cached_rows(layer, source_version)
        if cached is None:
            # no L1 to hold anything; the loader is the only source of truth.
            return await self._loader(layer, source_version, bounds)
        return [row for row in cached if self._row_bounds(row).intersects(bounds)]

    def _chunks_for(self, bounds: BoundingBox) -> list[TileId]:
        """coarse tiles covering ``bounds``."""
        min_x, min_y, max_x, max_y = bounds_to_tile_range(bounds, self.chunk_zoom)
        return [TileId(z=self.chunk_zoom, x=x, y=y) for x in range(min_x, max_x + 1) for y in range(min_y, max_y + 1)]

    async def _ensure_chunk_loaded(self, layer: str, source_version: int, chunk: TileId) -> None:
        """load and index one chunk unless it is already covered."""
        marker = (layer, source_version, chunk.key)
        if marker in self._covered:
            return
        rows = await self._loader(layer, source_version, tile_bounds(chunk))
        bucket = self._rows.setdefault((layer, source_version), {})
        for row in rows:
            feature_id = row.get(self.feature_id_column)
            if feature_id is None:
                continue
            # keyed by the raw identity, not a string form: this dict only
            # dedupes rows within a chunk and is never looked up by key, so
            # a UUID stays a UUID.
            bucket[feature_id] = row
            bounds = self._row_bounds(row)
            self.index_feature(layer, source_version, feature_id, bounds)
        self._covered.add(marker)
        log.debug(
            "chunk loaded: layer=%s version=%s chunk=%s rows=%d",
            layer,
            source_version,
            chunk,
            len(rows),
        )

    def _cached_rows(self, layer: str, source_version: int) -> list[dict[str, Any]] | None:
        if self._l1 is None:
            return None
        return list(self._rows.get((layer, source_version), {}).values())

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
