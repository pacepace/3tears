"""tests for the R-Tree-backed feature cache, against a real SQLiteBackend.

these run the actual SQLite R-Tree virtual table rather than a stand-in.
that matters: the module is only worth having if the built-in rtree is
present and behaves the way the design assumes, and the sole honest way to
know is to create one and query it. a mocked index would assert my beliefs
about SQLite rather than SQLite's behaviour.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import Column as SAColumn
from sqlalchemy import Integer, MetaData, String, Table

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.geo.features import FeatureCache
from threetears.geo.tiles import BoundingBox, TileId, tile_bounds, tile_for_point

#: the tile source these caches serve
SCOPE = "ds_test"


async def _empty_loader(layer: str, source_version: int, bounds: BoundingBox) -> list[dict[str, Any]]:
    return []


def _row_bounds(row: dict[str, Any]) -> BoundingBox:
    """test rows carry their bounds directly; real layers decode geometry."""
    return row["bounds"]


@pytest.fixture
def cache(request: pytest.FixtureRequest) -> FeatureCache:
    """a FeatureCache bound to a real in-memory SQLite L1.

    the backend name is per-test: SQLiteBackend opens a *named* shared
    in-memory database, so two backends built with the same name are two
    handles on one database. that is deliberate in production (pods share an
    L1 per process) and would silently leak state between tests.
    """
    metadata = MetaData()
    Table(
        f"geo_features_{SCOPE}",
        metadata,
        SAColumn("layer", String, primary_key=True),
        SAColumn("source_version", Integer, primary_key=True),
        SAColumn("feature_id", String, primary_key=True),
    )
    backend = SQLiteBackend(f"geo_features_test_{abs(hash(request.node.nodeid))}")
    backend.initialize(metadata)

    registry = CollectionRegistry()
    registry.configure(l1_backend=backend, l2_client=None, l3_pool=None)
    return FeatureCache(
        registry,
        DefaultCoreConfig(),
        None,
        None,
        loader=_empty_loader,
        bounds_of=_row_bounds,
        feature_id_column="feature_id",
        cache_scope=SCOPE,
    )


class TestRTreeAvailability:
    def test_sqlite_ships_the_rtree_module(self, cache: FeatureCache) -> None:
        """the assumption the whole module rests on.

        SQLite's R-Tree is a compile-time option. SpatiaLite -- the usual
        alternative -- is a documented local-dev build headache on macOS,
        which is why the built-in module was chosen. if it is absent here,
        everything below is unreachable.
        """
        cache.ensure_index()
        assert cache.indexed_keys_in_bbox("l", 1, BoundingBox(-1, -1, 1, 1)) == []

    def test_ensure_index_is_idempotent(self, cache: FeatureCache) -> None:
        cache.ensure_index()
        cache.ensure_index()
        cache.index_feature("l", 1, "a", BoundingBox(-1, -1, 1, 1))
        assert cache.indexed_keys_in_bbox("l", 1, BoundingBox(-1, -1, 1, 1)) == ["a"]


class TestSpatialQueries:
    def test_returns_features_inside_the_rectangle(self, cache: FeatureCache) -> None:
        cache.index_feature("tracts", 1, "inside", BoundingBox(-112.2, 33.3, -112.0, 33.5))
        cache.index_feature("tracts", 1, "far-away", BoundingBox(-80.0, 40.0, -79.0, 41.0))
        found = cache.indexed_keys_in_bbox("tracts", 1, BoundingBox(-112.5, 33.0, -111.5, 34.0))
        assert found == ["inside"]

    def test_partial_overlap_counts(self, cache: FeatureCache) -> None:
        # a tract straddling a tile edge belongs to both tiles; dropping it
        # from either leaves a visible gap along the seam.
        cache.index_feature("tracts", 1, "straddles", BoundingBox(-112.2, 33.3, -112.0, 33.5))
        found = cache.indexed_keys_in_bbox("tracts", 1, BoundingBox(-112.1, 33.4, -111.0, 34.0))
        assert found == ["straddles"]

    def test_edge_contact_counts(self, cache: FeatureCache) -> None:
        # matches BoundingBox.intersects and the L3 bbox-column predicate.
        # all three have to agree or a feature appears via one path and not
        # another.
        cache.index_feature("tracts", 1, "touching", BoundingBox(-112.2, 33.3, -112.0, 33.5))
        assert cache.indexed_keys_in_bbox("tracts", 1, BoundingBox(-112.0, 33.5, -111.0, 34.0)) == ["touching"]

    def test_disjoint_features_are_excluded(self, cache: FeatureCache) -> None:
        cache.index_feature("tracts", 1, "elsewhere", BoundingBox(-80.0, 40.0, -79.0, 41.0))
        assert cache.indexed_keys_in_bbox("tracts", 1, BoundingBox(-112.5, 33.0, -111.5, 34.0)) == []

    def test_reindexing_a_feature_moves_it(self, cache: FeatureCache) -> None:
        # a corrected boundary must not leave the old footprint behind, or
        # the feature answers queries for a place it no longer occupies.
        cache.index_feature("tracts", 1, "moved", BoundingBox(-112.2, 33.3, -112.0, 33.5))
        cache.index_feature("tracts", 1, "moved", BoundingBox(-80.0, 40.0, -79.0, 41.0))
        assert cache.indexed_keys_in_bbox("tracts", 1, BoundingBox(-112.5, 33.0, -111.5, 34.0)) == []
        assert cache.indexed_keys_in_bbox("tracts", 1, BoundingBox(-81.0, 39.0, -78.0, 42.0)) == ["moved"]


class TestGenerationAndLayerIsolation:
    def test_generations_do_not_bleed_into_each_other(self, cache: FeatureCache) -> None:
        """the property that makes warm-then-flip safe.

        two generations are resident at once by design, so a build of
        version 1 that saw version 2's rows would emit a tile mixing
        vintages -- and then cache it as immutable.
        """
        box = BoundingBox(-112.2, 33.3, -112.0, 33.5)
        cache.index_feature("tracts", 1, "old-shape", box)
        cache.index_feature("tracts", 2, "new-shape", box)
        assert cache.indexed_keys_in_bbox("tracts", 1, box) == ["old-shape"]
        assert cache.indexed_keys_in_bbox("tracts", 2, box) == ["new-shape"]

    def test_layers_do_not_bleed_into_each_other(self, cache: FeatureCache) -> None:
        box = BoundingBox(-112.2, 33.3, -112.0, 33.5)
        cache.index_feature("tracts", 1, "a-tract", box)
        cache.index_feature("locations", 1, "a-location", box)
        assert cache.indexed_keys_in_bbox("tracts", 1, box) == ["a-tract"]
        assert cache.indexed_keys_in_bbox("locations", 1, box) == ["a-location"]

    def test_feature_ids_containing_the_separator_do_not_corrupt_lookup(self, cache: FeatureCache) -> None:
        # keys are packed into one string; a feature id is caller data, so
        # the separator has to be one that cannot appear in it.
        box = BoundingBox(-112.2, 33.3, -112.0, 33.5)
        cache.index_feature("tracts", 1, "04013-010101", box)
        assert cache.indexed_keys_in_bbox("tracts", 1, box) == ["04013-010101"]


class TestWithoutL1:
    def test_spatial_calls_degrade_quietly_when_no_l1_is_bound(self) -> None:
        # L1 is optional in the framework. without it there is nothing to
        # index, and the caller falls through to the loader rather than
        # crashing.
        registry = CollectionRegistry()
        registry.configure(l1_backend=None, l2_client=None, l3_pool=None)
        cache = FeatureCache(
            registry,
            DefaultCoreConfig(),
            None,
            None,
            loader=_empty_loader,
            bounds_of=_row_bounds,
            feature_id_column="feature_id",
            cache_scope=SCOPE,
        )
        cache.index_feature("tracts", 1, "x", BoundingBox(-1, -1, 1, 1))
        assert cache.indexed_keys_in_bbox("tracts", 1, BoundingBox(-1, -1, 1, 1)) == []


class TestChunkCoverage:
    """the R-Tree only pays off if a warm chunk actually skips the L3 read.

    without coverage tracking the index can say what a pod holds but never
    whether it holds *all* of a region, so every build would have to reload
    anyway and the index would be decoration.
    """

    @staticmethod
    def _counting_cache(request: pytest.FixtureRequest, rows: list[dict[str, Any]]) -> tuple[FeatureCache, list[Any]]:
        calls: list[Any] = []

        async def _loader(layer: str, source_version: int, bounds: BoundingBox) -> list[dict[str, Any]]:
            calls.append(bounds)
            return [row for row in rows if row["bounds"].intersects(bounds)]

        metadata = MetaData()
        Table(
            f"geo_features_{SCOPE}",
            metadata,
            SAColumn("layer", String, primary_key=True),
            SAColumn("source_version", Integer, primary_key=True),
            SAColumn("feature_id", String, primary_key=True),
        )
        backend = SQLiteBackend(f"geo_chunk_{abs(hash(request.node.nodeid))}")
        backend.initialize(metadata)
        registry = CollectionRegistry()
        registry.configure(l1_backend=backend, l2_client=None, l3_pool=None)
        cache = FeatureCache(
            registry,
            DefaultCoreConfig(),
            None,
            None,
            loader=_loader,
            bounds_of=_row_bounds,
            feature_id_column="feature_id",
            cache_scope=SCOPE,
        )
        return cache, calls

    async def test_neighbouring_tiles_share_one_chunk_load(self, request: pytest.FixtureRequest) -> None:
        """the actual saving: adjacent tiles overlap almost entirely."""
        rows = [
            {"feature_id": "a", "bounds": BoundingBox(-112.10, 33.40, -112.09, 33.41)},
            {"feature_id": "b", "bounds": BoundingBox(-112.08, 33.42, -112.07, 33.43)},
        ]
        cache, calls = self._counting_cache(request, rows)

        # two adjacent z12 tiles inside one z8 chunk
        first = tile_for_point(-112.10, 33.40, 12)
        second = TileId(z=12, x=first.x + 1, y=first.y)
        await cache.features_in_bbox("tracts", 1, tile_bounds(first))
        await cache.features_in_bbox("tracts", 1, tile_bounds(second))

        assert len(calls) == 1, f"expected one chunk load for two neighbouring tiles, got {len(calls)}"

    async def test_a_warm_chunk_serves_without_touching_l3(self, request: pytest.FixtureRequest) -> None:
        rows = [{"feature_id": "a", "bounds": BoundingBox(-112.10, 33.40, -112.09, 33.41)}]
        cache, calls = self._counting_cache(request, rows)
        tile = tile_for_point(-112.10, 33.40, 12)

        first = await cache.features_in_bbox("tracts", 1, tile_bounds(tile))
        second = await cache.features_in_bbox("tracts", 1, tile_bounds(tile))

        assert len(calls) == 1
        assert first == second
        assert [row["feature_id"] for row in second] == ["a"]

    async def test_results_are_filtered_to_the_requested_rectangle(self, request: pytest.FixtureRequest) -> None:
        """a chunk is coarser than a tile, so its extra rows must not leak.

        returning everything in the chunk would put features from kilometres
        away into a tile that does not contain them.
        """
        near = BoundingBox(-112.10, 33.40, -112.09, 33.41)
        far = BoundingBox(-111.50, 33.90, -111.49, 33.91)
        rows = [{"feature_id": "near", "bounds": near}, {"feature_id": "far", "bounds": far}]
        cache, _ = self._counting_cache(request, rows)

        tile = tile_for_point(-112.10, 33.40, 12)
        found = await cache.features_in_bbox("tracts", 1, tile_bounds(tile))
        assert [row["feature_id"] for row in found] == ["near"]

    async def test_a_different_generation_is_loaded_separately(self, request: pytest.FixtureRequest) -> None:
        # coverage is per generation: reusing version 1's chunk for version 2
        # would build a tile from the wrong vintage and cache it as immutable.
        rows = [{"feature_id": "a", "bounds": BoundingBox(-112.10, 33.40, -112.09, 33.41)}]
        cache, calls = self._counting_cache(request, rows)
        tile = tile_for_point(-112.10, 33.40, 12)

        await cache.features_in_bbox("tracts", 1, tile_bounds(tile))
        await cache.features_in_bbox("tracts", 2, tile_bounds(tile))
        assert len(calls) == 2

    async def test_a_rectangle_spanning_chunks_loads_each_one(self, request: pytest.FixtureRequest) -> None:
        # correctness must not depend on the caller's rectangle happening to
        # fit inside a single chunk.
        cache, calls = self._counting_cache(request, [])
        wide = BoundingBox(-113.0, 33.0, -111.0, 34.0)
        await cache.features_in_bbox("tracts", 1, wide)
        assert len(calls) > 1


def _recording_loader(rows: list[dict[str, Any]]) -> tuple[Any, list[BoundingBox]]:
    """a loader answering ``rows`` by rectangle, and the list of every rectangle it was asked for."""
    calls: list[BoundingBox] = []

    async def _loader(layer: str, source_version: int, bounds: BoundingBox) -> list[dict[str, Any]]:
        calls.append(bounds)
        return [row for row in rows if row["bounds"].intersects(bounds)]

    return _loader, calls


def _cache_over(request: pytest.FixtureRequest, loader: Any, *, l1: bool = True, **kwargs: Any) -> FeatureCache:
    """a FeatureCache over ``loader``, with a real per-test SQLite L1 or with none."""
    backend: SQLiteBackend | None = None
    if l1:
        metadata = MetaData()
        Table(
            f"geo_features_{SCOPE}",
            metadata,
            SAColumn("layer", String, primary_key=True),
            SAColumn("source_version", Integer, primary_key=True),
            SAColumn("feature_id", String, primary_key=True),
        )
        backend = SQLiteBackend(f"geo_wide_{abs(hash(request.node.nodeid))}")
        backend.initialize(metadata)
    registry = CollectionRegistry()
    registry.configure(l1_backend=backend, l2_client=None, l3_pool=None)
    return FeatureCache(
        registry,
        DefaultCoreConfig(),
        None,
        None,
        loader=loader,
        bounds_of=_row_bounds,
        feature_id_column="feature_id",
        cache_scope=SCOPE,
        **kwargs,
    )


#: the z3 tile over Phoenix: 32x32 z8 chunks, far past what a chunk sweep should walk
_Z3 = TileId(z=3, x=1, y=3)


def _inside_z3(z: int = 12) -> TileId:
    """a tile well inside :data:`_Z3`, off every z8 chunk boundary, so it falls in one chunk."""
    scale = 1 << (z - _Z3.z)
    return TileId(z=z, x=_Z3.x * scale + scale // 2 + 3, y=_Z3.y * scale + scale // 2 + 3)


class TestWithoutL1TheLoaderIsAskedOnce:
    """with no L1 nothing can be held, so a chunk sweep is pure cost: one loader call per z8 chunk
    under the rectangle (1,025 for a z3 tile, 65,537 for z0, live) and then the whole rectangle
    again. the loader is the only source of truth, so it is asked exactly once."""

    async def test_a_z3_rectangle_is_one_loader_call(self, request: pytest.FixtureRequest) -> None:
        row = {"feature_id": "a", "bounds": BoundingBox(-112.10, 33.40, -112.09, 33.41)}
        loader, calls = _recording_loader([row])
        cache = _cache_over(request, loader, l1=False)

        found = await cache.features_in_bbox("tracts", 1, tile_bounds(_Z3))

        assert len(calls) == 1, f"expected one loader call with no L1, got {len(calls)}"
        assert calls[0] == tile_bounds(_Z3)
        assert [r["feature_id"] for r in found] == ["a"]

    async def test_nothing_is_kept_between_reads(self, request: pytest.FixtureRequest) -> None:
        # with no L1 there is nothing to evict from, so anything kept would grow for the life of
        # the pod: the same rectangle twice is the loader twice
        loader, calls = _recording_loader([{"feature_id": "a", "bounds": BoundingBox(-112.10, 33.40, -112.09, 33.41)}])
        cache = _cache_over(request, loader, l1=False)
        tile = tile_bounds(_inside_z3())

        await cache.features_in_bbox("tracts", 1, tile)
        await cache.features_in_bbox("tracts", 1, tile)

        assert len(calls) == 2


class TestAWideRectangleIsOneLoad:
    """a rectangle spanning more than :attr:`FeatureCache.max_chunk_reads` uncovered chunks is one
    loader call for the rectangle; the chunks it fully contains are then covered."""

    async def test_a_z3_rectangle_is_one_loader_call(self, request: pytest.FixtureRequest) -> None:
        loader, calls = _recording_loader([])
        cache = _cache_over(request, loader)

        await cache.features_in_bbox("tracts", 1, tile_bounds(_Z3))

        assert len(calls) == 1, f"expected one loader call for a z3 rectangle, got {len(calls)}"
        assert calls[0] == tile_bounds(_Z3)

    async def test_it_returns_exactly_the_rows_in_the_rectangle(self, request: pytest.FixtureRequest) -> None:
        inside = {"feature_id": "in", "bounds": BoundingBox(-112.10, 33.40, -112.09, 33.41)}
        outside = {"feature_id": "out", "bounds": BoundingBox(-80.0, 40.0, -79.0, 41.0)}
        loader, _ = _recording_loader([inside, outside])
        cache = _cache_over(request, loader)

        found = await cache.features_in_bbox("tracts", 1, tile_bounds(_Z3))
        assert [r["feature_id"] for r in found] == ["in"]

    async def test_a_tile_inside_it_is_served_with_no_loader_call(self, request: pytest.FixtureRequest) -> None:
        tile = _inside_z3()
        bounds = tile_bounds(tile)
        near = {
            "feature_id": "near",
            "bounds": BoundingBox(bounds.min_lon, bounds.min_lat, bounds.min_lon + 1e-4, bounds.min_lat + 1e-4),
        }
        elsewhere = {"feature_id": "elsewhere", "bounds": BoundingBox(-112.9, 34.9, -112.8, 35.0)}
        loader, calls = _recording_loader([near, elsewhere])
        cache = _cache_over(request, loader)

        await cache.features_in_bbox("tracts", 1, tile_bounds(_Z3))
        found = await cache.features_in_bbox("tracts", 1, bounds)

        assert len(calls) == 1, f"a z12 tile inside a loaded z3 rectangle went back to the loader ({len(calls)} calls)"
        assert [r["feature_id"] for r in found] == ["near"]

    async def test_a_tile_whose_chunk_crosses_the_edge_loads_that_chunk(self, request: pytest.FixtureRequest) -> None:
        # a z12 tile on the z3 tile's eastern edge touches the z8 chunk beyond it, which the z3 load
        # did not contain and so did not cover: that chunk alone is loaded
        scale = 1 << (12 - _Z3.z)
        edge = TileId(z=12, x=(_Z3.x + 1) * scale - 1, y=_Z3.y * scale + scale // 2)
        loader, calls = _recording_loader([])
        cache = _cache_over(request, loader)

        await cache.features_in_bbox("tracts", 1, tile_bounds(_Z3))
        await cache.features_in_bbox("tracts", 1, tile_bounds(edge))

        assert len(calls) == 2, f"expected the one uncovered chunk to load, got {len(calls) - 1} loads"
        beyond = TileId(z=FeatureCache.chunk_zoom, x=(_Z3.x + 1) * (1 << (FeatureCache.chunk_zoom - _Z3.z)), y=0)
        assert calls[1].min_lon == tile_bounds(beyond).min_lon

    async def test_a_feature_straddling_the_edge_is_still_served(self, request: pytest.FixtureRequest) -> None:
        # a row crossing the z3 edge belongs to the chunks on both sides of it; the inside chunk must
        # still answer it once covered by the wide load
        inner = tile_bounds(_inside_z3())
        west_of_z3 = tile_bounds(_Z3).min_lon - 10.0
        straddler = {
            "feature_id": "wide",
            "bounds": BoundingBox(west_of_z3, inner.min_lat, inner.max_lon, inner.max_lat),
        }
        loader, calls = _recording_loader([straddler])
        cache = _cache_over(request, loader)

        await cache.features_in_bbox("tracts", 1, tile_bounds(_Z3))
        found = await cache.features_in_bbox("tracts", 1, tile_bounds(_inside_z3()))

        assert len(calls) == 1
        assert [r["feature_id"] for r in found] == ["wide"]


class TestHeldRowsAreBounded:
    """the rows a pod holds are bounded and evicted, least recently used chunk first."""

    @staticmethod
    def _two_chunks() -> tuple[TileId, TileId, list[dict[str, Any]]]:
        first = _inside_z3()
        second = TileId(z=12, x=first.x + 64, y=first.y)  # a different z8 chunk
        rows = []
        for name, tile in (("a", first), ("b", second)):
            bounds = tile_bounds(tile)
            rows.append(
                {
                    "feature_id": name,
                    "bounds": BoundingBox(bounds.min_lon, bounds.min_lat, bounds.min_lon + 1e-4, bounds.min_lat + 1e-4),
                }
            )
        return first, second, rows

    async def test_a_chunk_past_the_bound_is_evicted_and_reloaded(self, request: pytest.FixtureRequest) -> None:
        first, second, rows = self._two_chunks()
        loader, calls = _recording_loader(rows)
        cache = _cache_over(request, loader, max_cached_rows=1)

        await cache.features_in_bbox("tracts", 1, tile_bounds(first))
        await cache.features_in_bbox("tracts", 1, tile_bounds(second))
        again = await cache.features_in_bbox("tracts", 1, tile_bounds(first))

        assert len(calls) == 3, "the first chunk should have been evicted to hold the second"
        assert [r["feature_id"] for r in again] == ["a"]

    async def test_within_the_bound_both_stay_held(self, request: pytest.FixtureRequest) -> None:
        first, second, rows = self._two_chunks()
        loader, calls = _recording_loader(rows)
        cache = _cache_over(request, loader, max_cached_rows=10)

        await cache.features_in_bbox("tracts", 1, tile_bounds(first))
        await cache.features_in_bbox("tracts", 1, tile_bounds(second))
        await cache.features_in_bbox("tracts", 1, tile_bounds(first))

        assert len(calls) == 2

    async def test_an_evicted_feature_leaves_the_spatial_index(self, request: pytest.FixtureRequest) -> None:
        # the R-Tree is in L1 too, so it is bounded with the rows rather than growing behind them
        first, second, rows = self._two_chunks()
        loader, _ = _recording_loader(rows)
        cache = _cache_over(request, loader, max_cached_rows=1)

        await cache.features_in_bbox("tracts", 1, tile_bounds(first))
        assert cache.indexed_keys_in_bbox("tracts", 1, tile_bounds(first)) == ["a"]
        await cache.features_in_bbox("tracts", 1, tile_bounds(second))
        assert cache.indexed_keys_in_bbox("tracts", 1, tile_bounds(first)) == []
        assert cache.indexed_keys_in_bbox("tracts", 1, tile_bounds(second)) == ["b"]

    def test_a_bound_below_one_is_refused(self, request: pytest.FixtureRequest) -> None:
        with pytest.raises(ValueError, match="max_cached_rows"):
            _cache_over(request, _empty_loader, max_cached_rows=0)
