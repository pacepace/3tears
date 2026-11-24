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
surrogate id per feature key (:class:`~threetears.core.cache.rtree.SQLiteRTree`
keeps both). several caches of one scope share the L1 (the hub builds one per
layer), and each keeps its rows in memory, so each has a tree and map of its
own and indexes and evicts only its own entries. an instance's tables are
dropped when it is garbage-collected.
"""

from __future__ import annotations

import secrets
import weakref
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, ClassVar

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from threetears.core.cache.rtree import SQLiteRTree
from threetears.core.collections.base import BaseCollection
from threetears.core.entities.base import BaseEntity
from threetears.core.frozen_state import EMPTY, with_entry, with_newest, without_entry
from threetears.core.serialization import deserialize_from_json, serialize_to_json
from threetears.geo.bounds import BOUNDS_COLUMNS, BoundsOf
from threetears.geo.scope import check_cache_scope
from threetears.geo.tiles import MAX_MERCATOR_LATITUDE, BoundingBox, TileId, bounds_to_tile_range, tile_bounds
from threetears.geo.workers import read_bounds
from threetears.observe import contain, get_logger, traced

__all__ = ["BOUNDS_COLUMNS", "BoundsOf", "FeatureCache", "FeatureEntity", "FeatureLoader"]

log = get_logger(__name__)

#: signature of the L3 read this cache sits in front of: given a layer,
#: source generation and rectangle, return the source rows inside it as one Arrow table.
#: the caller owns the query, because only the caller knows the datasource.
FeatureLoader = Callable[[str, int, BoundingBox], Awaitable[pa.Table]]


#: one covered chunk: ``(layer, source_version, chunk tile key)``
_ChunkMarker = tuple[str, int, tuple[int, int, int]]

#: the columns a held chunk carries beside the source's own: each row's R-Tree feature key and
#: bounds. a source column of one of these names is refused rather than shadowed
_KEY = "__feature_cache_key"
_MIN_LON = "__feature_cache_min_lon"
_MIN_LAT = "__feature_cache_min_lat"
_MAX_LON = "__feature_cache_max_lon"
_MAX_LAT = "__feature_cache_max_lat"
_BOUNDS = (_MIN_LON, _MIN_LAT, _MAX_LON, _MAX_LAT)
_OWN_COLUMNS = (_KEY, *_BOUNDS)


class FeatureEntity(BaseEntity):
    """one cached source feature row."""

    primary_key_field = "feature_id"


class FeatureCache(BaseCollection[FeatureEntity]):
    """L1 cache of source features with a local R-Tree bbox index.

    :param loader: async callable fetching source rows for a rectangle
    :ptype loader: FeatureLoader
    :param bounds_of: every row's bounding rectangle, for a loaded table at once (see
        :data:`BoundsOf`). supplied by the caller because only the layer declaration knows which
        column holds geometry, and whether it is WKB or a lon/lat pair
    :ptype bounds_of: BoundsOf
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
        bounds_of: BoundsOf,
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
        # the chunks this pod has fully covered, each with the rows of the features intersecting it
        # as one immutable Arrow table (the source's columns plus each row's feature key and
        # bounds), least recently read first. coverage is what lets a hit be trusted: without it
        # the R-Tree can only say what is held, never what is complete. only filled with an L1
        # bound, and bounded by ``_row_limit`` row entries (``_held``). a read-only snapshot
        # replaced whole (threetears.core.frozen_state): one entry a chunk, a few hundred at the
        # default bound. a feature in two chunks is a row in each; it leaves the index when no
        # held chunk next to the one evicted still carries it (see :meth:`_carried_nearby`)
        self._chunks: Mapping[_ChunkMarker, pa.Table] = EMPTY
        self._held = 0
        # chunks a read in progress depends on, which eviction skips until that read has answered;
        # a read-only snapshot replaced whole
        self._pins: Mapping[_ChunkMarker, int] = EMPTY
        # names this instance's own R-Tree, so instances sharing the L1 never read or evict each
        # other's entries
        self._instance_key = secrets.token_hex(6)
        self._tree: SQLiteRTree | None = None

    @property
    def table_name(self) -> str:
        return f"geo_features_{self._cache_scope}"

    @property
    def entity_class(self) -> type[FeatureEntity]:
        return FeatureEntity

    # ------------------------------------------------------------------
    # R-Tree companion
    # ------------------------------------------------------------------

    # each instance holds its own R-Tree and key map, so a collected instance's entries go by dropping
    # its two tables, never by finding its rows among another instance's
    @property
    def _rtree_table(self) -> str:
        return f"{self.table_name}_{self._instance_key}_rtree"

    @property
    def _map_table(self) -> str:
        return f"{self.table_name}_{self._instance_key}_rtree_map"

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
        tree = SQLiteRTree(backend, self._rtree_table, self._map_table)
        tree.create()
        self._tree = tree
        self._rtree_ready = True
        weakref.finalize(self, _drop_index_entries, tree)

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
        self._index_many([(self._feature_key(layer, source_version, feature_id), bounds)])

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
        tree = self._tree
        if tree is None:
            return []
        prefix = self._feature_key(layer, source_version, "")
        # the tree holds every layer and generation this instance indexed: the prefix keeps the asked one's (a plain
        # prefix, so a layer name's own % or _ is no wildcard)
        return tree.overlapping(bounds.min_lon, bounds.max_lon, bounds.min_lat, bounds.max_lat, prefix=prefix)

    # ------------------------------------------------------------------
    # read path
    # ------------------------------------------------------------------

    @traced
    async def features_in_bbox(self, layer: str, source_version: int, bounds: BoundingBox) -> pa.Table:
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
        :return: source rows intersecting the rectangle, with the loader's columns
        :rtype: pa.Table
        """
        if self.l1_backend is None:
            rows = await self._prepared(layer, source_version, await self._loader(layer, source_version, bounds))
            return _answer(rows, bounds)
        chunks = self._chunks_for(bounds)
        uncovered = sum(1 for chunk in chunks if (layer, source_version, chunk.key) not in self._chunks)
        if uncovered > self.max_chunk_reads:
            rows = await self._prepared(layer, source_version, await self._loader(layer, source_version, bounds))
            self._hold_rectangle(layer, source_version, bounds, rows)
            self._evict()
            return _answer(rows, bounds)
        pinned: list[_ChunkMarker] = []
        # rows of chunks this read cannot hold, answered from here instead of from the index
        transient: list[pa.Table] = []
        # entries this read has newly held: together they may not pass half the bound
        claimed = 0
        try:
            for chunk in chunks:
                marker = (layer, source_version, chunk.key)
                held = self._chunks.get(marker)
                if held is not None:
                    self._chunks = with_newest(self._chunks, marker, held)
                else:
                    loaded = await self._prepared(
                        layer, source_version, await self._loader(layer, source_version, tile_bounds(chunk))
                    )
                    log.debug(
                        "chunk loaded: layer=%s version=%s chunk=%s rows=%d",
                        layer,
                        source_version,
                        chunk,
                        loaded.num_rows,
                    )
                    weight = max(1, loaded.num_rows)
                    if claimed + weight > self._read_cap:
                        log.debug(
                            "chunk not held: layer=%s version=%s chunk=%s rows=%d would take this read's "
                            "entries past half the bound of %d",
                            layer,
                            source_version,
                            chunk,
                            loaded.num_rows,
                            self._row_limit,
                        )
                        transient.append(loaded)
                        continue
                    claimed += weight
                    self._index_rows(self._hold(marker, loaded))
                self._pins = with_entry(self._pins, marker, self._pins.get(marker, 0) + 1)
                pinned.append(marker)
            # every feature overlapping the rectangle overlaps a chunk under it, which this read
            # either holds (pinned: no eviction can take it before the answer) or answers from
            # ``transient``; the R-Tree says which held rows overlap, in its order
            keys = pa.array(self._keys_in_bbox(layer, source_version, bounds), type=pa.string())
            found = [*transient, *(_rows_by_key(self._chunks[marker], keys) for marker in pinned)]
            return _answer(_first_per_key(_concat(found)), bounds)
        finally:
            for marker in pinned:
                count = self._pins[marker] - 1
                if count:
                    self._pins = with_entry(self._pins, marker, count)
                else:
                    self._pins = without_entry(self._pins, marker)
            self._evict()

    @property
    def _read_cap(self) -> int:
        """the most row entries one read may hold: half the bound, so it never flushes the rest."""
        return max(1, self._row_limit // 2)

    async def _prepared(self, layer: str, source_version: int, table: pa.Table) -> pa.Table:
        """a loader's ``table`` as the cache holds it: one row per feature, each with its key and bounds.

        a row with no feature id is dropped (a row with no identity is not a feature), and a feature
        answered twice is kept once, first first. the feature key is the R-Tree's, made from the
        id's string form, as it always was.

        :raises ValueError: when the source carries a column the cache adds, or ``bounds_of``
            answers other than one row of :data:`BOUNDS_COLUMNS` per row
        """
        clash = [name for name in _OWN_COLUMNS if name in table.column_names]
        if clash:
            raise ValueError(f"source rows carry the feature cache's own column(s) {clash}")
        loaded = table.num_rows
        if self.feature_id_column not in table.column_names:
            table = table.schema.empty_table()
            ids: pa.ChunkedArray | pa.Array = pa.array([], type=pa.string())
        else:
            ids = table.column(self.feature_id_column)
            if ids.null_count:
                table = table.filter(pc.is_valid(ids))
            table = _first_per(table, table.column(self.feature_id_column))
            ids = table.column(self.feature_id_column)
        if table.num_rows < loaded:
            log.debug("dropped %d source rows with no %s", loaded - table.num_rows, self.feature_id_column)
        # off the loop: a layer's own geometry columns are read in a worker process over shared memory, any other
        # callable on a thread (threetears.geo.workers.read_bounds)
        found = await read_bounds(self._bounds_of, table)
        if found.num_rows != table.num_rows or any(name not in found.column_names for name in BOUNDS_COLUMNS):
            raise ValueError(
                f"bounds_of answered {found.num_rows} row(s) of {found.column_names} for {table.num_rows} row(s); "
                f"it must answer one row of {list(BOUNDS_COLUMNS)} per row"
            )
        prefix = self._feature_key(layer, source_version, "")
        table = table.append_column(_KEY, pc.binary_join_element_wise(prefix, pc.cast(ids, pa.string()), ""))
        for own, name in zip(_BOUNDS, BOUNDS_COLUMNS, strict=True):
            table = table.append_column(own, pc.cast(found.column(name), pa.float64()))
        return table

    def _chunks_for(self, bounds: BoundingBox) -> list[TileId]:
        """coarse tiles covering ``bounds``."""
        min_x, min_y, max_x, max_y = bounds_to_tile_range(bounds, self.chunk_zoom)
        return [TileId(z=self.chunk_zoom, x=x, y=y) for x in range(min_x, max_x + 1) for y in range(min_y, max_y + 1)]

    def _hold_rectangle(self, layer: str, source_version: int, bounds: BoundingBox, rows: pa.Table) -> None:
        """cover every chunk ``bounds`` wholly contains, from one rectangle's prepared rows.

        each contained chunk gets exactly the rows that intersect it, which is what loading it on
        its own would have returned -- an empty one included, which is coverage too (ocean). the
        whole rectangle is held or none of it, counted as the bound counts, and indexed in one
        transaction.
        """
        zoom = self.chunk_zoom
        tile_range = bounds_to_tile_range(bounds, zoom)
        min_x, min_y, max_x, max_y = tile_range
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
            self._log_wide_not_held(layer, source_version, rows.num_rows, len(contained))
            return
        per_chunk = _split_by_chunk(rows, contained, tile_range, zoom)
        entries = sum(max(1, held.num_rows) for held in per_chunk.values())
        if entries > self._read_cap:
            self._log_wide_not_held(layer, source_version, rows.num_rows, entries)
            return
        fresh = [self._hold((layer, source_version, (zoom, x, y)), held) for (x, y), held in per_chunk.items()]
        self._index_rows(_concat([part.select(list(_OWN_COLUMNS)) for part in fresh]))
        log.debug(
            "wide read held: layer=%s version=%s rows=%d chunks covered=%d entries=%d",
            layer,
            source_version,
            rows.num_rows,
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

    def _hold(self, marker: _ChunkMarker, rows: pa.Table) -> pa.Table:
        """record ``marker`` as covered by ``rows``; return the rows whose features are new to the pod, to index.

        does not evict: the caller evicts once its read no longer depends on what it holds.
        """
        if marker in self._chunks:
            # two reads loaded the same chunk at once; the later one replaces the earlier
            self._release(marker)
        layer, source_version, _key = marker
        nearby = self._carried_nearby(layer, source_version, rows)
        self._chunks = with_newest(self._chunks, marker, rows)
        self._held += max(1, rows.num_rows)
        return _without_keys(rows, nearby)

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
        """evict one covered chunk, dropping from the index each feature no other held chunk carries."""
        rows = self._chunks[marker]
        self._chunks = without_entry(self._chunks, marker)
        self._held -= max(1, rows.num_rows)
        layer, source_version, _key = marker
        gone = _without_keys(rows, self._carried_nearby(layer, source_version, rows))
        self._unindex_keys(gone.column(_KEY).to_pylist())

    def _carried_nearby(self, layer: str, source_version: int, rows: pa.Table) -> pa.Array | None:
        """the feature keys held chunks near ``rows`` carry, or ``None`` when none is held there.

        a chunk carrying one of ``rows``' features intersects that feature, so it lies within the
        features' chunk range one chunk wider on every side -- the rule :meth:`_hold_rectangle`
        splits rows by, a feature whose edge lies on a chunk boundary touching the chunk beyond.
        only those chunks are asked, never a count per feature.
        """
        if not rows.num_rows:
            return None
        edges = [pc.min(rows.column(_MIN_LON)), pc.min(rows.column(_MIN_LAT))]
        edges += [pc.max(rows.column(_MAX_LON)), pc.max(rows.column(_MAX_LAT))]
        min_lon, min_lat, max_lon, max_lat = (edge.as_py() for edge in edges)
        if min_lon is None or min_lat is None or max_lon is None or max_lat is None:
            return None  # no row has bounds, so none is indexed
        min_x, min_y, max_x, max_y = bounds_to_tile_range(
            BoundingBox(min_lon, min_lat, max_lon, max_lat), self.chunk_zoom
        )
        parts = [
            chunk
            for (held_layer, held_version, (_z, x, y)), held in self._chunks.items()
            if held_layer == layer
            and held_version == source_version
            and min_x - 1 <= x <= max_x + 1
            and min_y - 1 <= y <= max_y + 1
            for chunk in held.column(_KEY).chunks
        ]
        if not parts:
            return None
        return pa.concat_arrays(parts)

    def _index_many(self, features: list[tuple[str, BoundingBox]]) -> None:
        """record features' bounds in the R-Tree by feature key, in one transaction."""
        if not features:
            return
        self.ensure_index()
        tree = self._tree
        if tree is None:
            return
        tree.put([(key, b.min_lon, b.max_lon, b.min_lat, b.max_lat) for key, b in features])

    def _index_rows(self, rows: pa.Table) -> None:
        """record prepared rows' bounds in the R-Tree by feature key, in one transaction; a row with no
        bounds intersects nothing and is not indexed."""
        boxes = rows.select(list(_OWN_COLUMNS)).drop_null()
        if not boxes.num_rows:
            return
        self.ensure_index()
        tree = self._tree
        if tree is None:
            return
        # the tree's order: key, then x (lon) bounds, then y (lat) bounds
        columns = [boxes.column(name).to_pylist() for name in (_KEY, _MIN_LON, _MAX_LON, _MIN_LAT, _MAX_LAT)]
        tree.put(list(zip(*columns, strict=True)))

    def _unindex_keys(self, keys: list[str]) -> None:
        """drop features from the spatial index by feature key, in one transaction."""
        tree = self._tree
        if not keys or tree is None:
            return
        tree.remove(keys)

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
        return serialize_to_json(data)

    def deserialize(self, data: bytes) -> dict[str, Any]:
        # no declared field types: a cached feature row's columns are whatever
        # the datasource's layer declaration selected, which varies per layer
        # and is not known to this class. values round-trip as their JSON
        # types, which is sufficient -- geometry travels as WKB hex and
        # attributes are already coerced to MVT scalars downstream.
        result: dict[str, Any] = deserialize_from_json(data, {})
        return result


#: an empty table of the cache's own columns, for a concatenation of nothing
_EMPTY_OWN = pa.schema([(_KEY, pa.string()), *((name, pa.float64()) for name in _BOUNDS)]).empty_table()


def _concat(tables: list[pa.Table]) -> pa.Table:
    """``tables`` as one, columns unified by name: two loads of one layer may type a column apart
    (all null in one), and a column one lacks is null in its rows."""
    if not tables:
        return _EMPTY_OWN
    if len(tables) == 1:
        return tables[0]
    return pa.concat_tables(tables, promote_options="permissive")


def _first_per(rows: pa.Table, ids: pa.ChunkedArray | pa.Array) -> pa.Table:
    """``rows`` with each value of ``ids`` kept once, at its first row, in row order."""
    if rows.num_rows < 2:
        return rows
    values = ids.combine_chunks() if isinstance(ids, pa.ChunkedArray) else ids
    unique = pc.unique(values)
    if len(unique) == len(values):
        return rows
    first = pc.index_in(unique, value_set=values)
    return rows.take(pc.take(first, pc.sort_indices(first)))


def _first_per_key(rows: pa.Table) -> pa.Table:
    """``rows`` with each feature key kept once, at its first row."""
    return _first_per(rows, rows.column(_KEY))


def _rows_by_key(rows: pa.Table, keys: pa.Array) -> pa.Table:
    """the rows of ``rows`` carrying ``keys``, in ``keys``' order; a key it does not carry is skipped."""
    positions = pc.drop_null(pc.index_in(keys, value_set=rows.column(_KEY).combine_chunks()))
    return rows.take(positions)


def _without_keys(rows: pa.Table, keys: pa.Array | None) -> pa.Table:
    """the rows of ``rows`` whose feature key is not among ``keys``."""
    if keys is None or not len(keys):
        return rows
    return rows.filter(pc.invert(pc.is_in(rows.column(_KEY), value_set=keys)))


def _answer(rows: pa.Table, bounds: BoundingBox) -> pa.Table:
    """the prepared rows whose own bounds intersect ``bounds``, edge-inclusive as
    :meth:`BoundingBox.intersects`, with the source's columns only. a row with no bounds intersects
    nothing."""
    overlaps = pc.and_(
        pc.and_(
            pc.less_equal(rows.column(_MIN_LON), bounds.max_lon),
            pc.greater_equal(rows.column(_MAX_LON), bounds.min_lon),
        ),
        pc.and_(
            pc.less_equal(rows.column(_MIN_LAT), bounds.max_lat),
            pc.greater_equal(rows.column(_MAX_LAT), bounds.min_lat),
        ),
    )
    return rows.filter(overlaps).drop_columns(list(_OWN_COLUMNS))


def _tile_x(lon: np.ndarray, n: int) -> np.ndarray:
    """:func:`~threetears.geo.tiles.tile_for_point`'s column, for an array of longitudes."""
    return np.minimum(((lon + 180.0) / 360.0 * n).astype(np.int64), n - 1)


def _tile_y(lat: np.ndarray, n: int) -> np.ndarray:
    """:func:`~threetears.geo.tiles.tile_for_point`'s row, for an array of latitudes."""
    lat_rad = np.radians(np.clip(lat, -MAX_MERCATOR_LATITUDE, MAX_MERCATOR_LATITUDE))
    rows: np.ndarray = np.minimum(((1.0 - np.arcsinh(np.tan(lat_rad)) / np.pi) / 2.0 * n).astype(np.int64), n - 1)
    return rows


def _split_by_chunk(
    rows: pa.Table,
    contained: Mapping[tuple[int, int], BoundingBox],
    tile_range: tuple[int, int, int, int],
    zoom: int,
) -> dict[tuple[int, int], pa.Table]:
    """each ``contained`` chunk's rows: the prepared ``rows`` intersecting it, in row order.

    each row is tried against the chunks of its own chunk range one chunk wider on every side -- a
    row whose edge lies exactly on a chunk boundary touches the chunk beyond it too, and the
    loader's edge-inclusive query returns it there, so the held set must as well -- then tested
    exactly against the chunk's bounds. vectorised: one (row, chunk) pair per candidate, never a
    Python object per row. each chunk's rows are a copy, so a held chunk never keeps the whole
    rectangle's buffers alive.
    """
    n = 1 << zoom
    range_min_x, range_min_y, range_max_x, range_max_y = tile_range
    columns = [rows.column(name).to_numpy() for name in _BOUNDS]
    has_bounds = np.flatnonzero(~np.isnan(np.stack(columns)).any(axis=0))
    min_lon, min_lat, max_lon, max_lat = (column[has_bounds] for column in columns)
    first_x = np.maximum(_tile_x(min_lon, n) - 1, range_min_x)
    last_x = np.minimum(_tile_x(max_lon, n) + 1, range_max_x)
    first_y = np.maximum(_tile_y(max_lat, n) - 1, range_min_y)
    last_y = np.minimum(_tile_y(min_lat, n) + 1, range_max_y)
    width = np.maximum(last_x - first_x + 1, 0)
    pairs = width * np.maximum(last_y - first_y + 1, 0)
    row = np.repeat(np.arange(len(has_bounds)), pairs)
    step = np.arange(int(pairs.sum())) - np.repeat(np.cumsum(pairs) - pairs, pairs)
    pair_x = first_x[row] + step % np.maximum(width[row], 1)
    pair_y = first_y[row] + step // np.maximum(width[row], 1)
    order = sorted(contained)
    codes = np.array([x * n + y for x, y in order], dtype=np.int64)
    chunk_bounds = [np.array([getattr(contained[xy], name) for xy in order]) for name in BOUNDS_COLUMNS]
    found = np.minimum(np.searchsorted(codes, pair_x * n + pair_y), len(codes) - 1)
    keep = codes[found] == pair_x * n + pair_y
    keep &= min_lon[row] <= chunk_bounds[2][found]
    keep &= max_lon[row] >= chunk_bounds[0][found]
    keep &= min_lat[row] <= chunk_bounds[3][found]
    keep &= max_lat[row] >= chunk_bounds[1][found]
    kept_row, kept_chunk = row[keep], found[keep]
    by_chunk = np.lexsort((kept_row, kept_chunk))
    kept_row, kept_chunk = kept_row[by_chunk], kept_chunk[by_chunk]
    edges = np.searchsorted(kept_chunk, np.arange(len(codes) + 1))
    return {
        xy: rows.take(pa.array(has_bounds[kept_row[edges[i] : edges[i + 1]]], type=pa.int64()))
        for i, xy in enumerate(order)
    }


def _drop_index_entries(tree: SQLiteRTree) -> None:
    """drop a collected :class:`FeatureCache`'s own R-Tree; best effort, never raises.

    run by :func:`weakref.finalize`, so it may run on any thread -- whichever one triggers the
    collection -- during interpreter shutdown, or after the L1 has closed, and SQLite may refuse
    the connection from there. the worst case is orphaned entries: rows in an in-memory table that
    no live cache reads, since the tables are the collected instance's own.
    """
    try:
        tree.drop()
    # prawduct:allow prawduct/broad-except -- a finalizer must not raise; whatever the L1 says on
    # the way down, the entries are unread and the table is in memory
    except Exception as exc:  # noqa: BLE001
        contain(exc, log=log, what="could not drop a collected feature cache's index entries")
