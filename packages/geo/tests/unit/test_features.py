"""tests for the R-Tree-backed feature cache, against a real SQLiteBackend.

these run the actual SQLite R-Tree virtual table rather than a stand-in.
that matters: the module is only worth having if the built-in rtree is
present and behaves the way the design assumes, and the sole honest way to
know is to create one and query it. a mocked index would assert my beliefs
about SQLite rather than SQLite's behaviour.
"""

from __future__ import annotations

import asyncio
import gc
import random
import sqlite3
from typing import Any

import pytest
from sqlalchemy import Column as SAColumn
from sqlalchemy import Integer, MetaData, String, Table

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.geo.features import FeatureCache
from threetears.geo.tiles import BoundingBox, TileId, tile_bounds, tile_for_point
from threetears.core.testing.l1_tables import L1_EVICTION_INFO, TEST_L1_TABLE

#: the tile source these caches serve
SCOPE = "ds_test"


async def _empty_loader(layer: str, source_version: int, bounds: BoundingBox) -> list[dict[str, Any]]:
    return []


def _row_bounds(row: dict[str, Any]) -> BoundingBox:
    """test rows carry their bounds directly; real layers decode geometry."""
    bounds = row["bounds"]
    assert isinstance(bounds, BoundingBox)
    return bounds


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
        info={L1_EVICTION_INFO: TEST_L1_TABLE},
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
            info={L1_EVICTION_INFO: TEST_L1_TABLE},
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
            info={L1_EVICTION_INFO: TEST_L1_TABLE},
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


def _point_row(feature_id: str, tile: TileId) -> dict[str, Any]:
    """a tiny row at the inner corner of ``tile``, inside it and inside its z8 chunk."""
    bounds = tile_bounds(tile)
    lon = bounds.min_lon + (bounds.max_lon - bounds.min_lon) / 4
    lat = bounds.min_lat + (bounds.max_lat - bounds.min_lat) / 4
    return {"feature_id": feature_id, "bounds": BoundingBox(lon, lat, lon + 1e-6, lat + 1e-6)}


def _chunk_tile(dx: int) -> TileId:
    """a z12 tile in the z8 chunk ``dx`` chunks east of :func:`_inside_z3`'s, off every chunk edge."""
    base = _inside_z3()
    return TileId(z=12, x=base.x + 16 * dx, y=base.y)


class TestEvictionOrderAndSharing:
    """the eviction mechanics the bound rests on, each pinned so removing it fails a test."""

    async def test_the_least_recently_read_chunk_goes_first(self, request: pytest.FixtureRequest) -> None:
        a, b, c = _chunk_tile(0), _chunk_tile(1), _chunk_tile(2)
        loader, calls = _recording_loader([_point_row("a", a), _point_row("b", b), _point_row("c", c)])
        cache = _cache_over(request, loader, max_cached_rows=2)

        await cache.features_in_bbox("tracts", 1, tile_bounds(a))
        await cache.features_in_bbox("tracts", 1, tile_bounds(b))
        await cache.features_in_bbox("tracts", 1, tile_bounds(a))  # a is now the most recent
        await cache.features_in_bbox("tracts", 1, tile_bounds(c))  # evicts b, not a
        assert len(calls) == 3

        await cache.features_in_bbox("tracts", 1, tile_bounds(a))
        assert len(calls) == 3, "the chunk read most recently was evicted (FIFO, not LRU)"
        await cache.features_in_bbox("tracts", 1, tile_bounds(b))
        assert len(calls) == 4, "the least recently read chunk was kept"

    async def test_a_feature_in_two_chunks_survives_one_of_them_leaving(self, request: pytest.FixtureRequest) -> None:
        a, b, c = _chunk_tile(0), _chunk_tile(1), _chunk_tile(2)
        a_bounds, b_bounds = tile_bounds(a), tile_bounds(b)
        shared = {
            "feature_id": "s",
            "bounds": BoundingBox(a_bounds.min_lon, a_bounds.min_lat, b_bounds.max_lon, b_bounds.max_lat),
        }
        loader, calls = _recording_loader([shared, _point_row("c", c)])
        cache = _cache_over(request, loader, max_cached_rows=2)

        await cache.features_in_bbox("tracts", 1, tile_bounds(a))
        await cache.features_in_bbox("tracts", 1, tile_bounds(b))
        await cache.features_in_bbox("tracts", 1, tile_bounds(c))  # evicts a, which also held "s"

        found = await cache.features_in_bbox("tracts", 1, b_bounds)
        assert len(calls) == 3, "b should still be held"
        assert [row["feature_id"] for row in found] == ["s"], "a feature b still holds was dropped with a"
        assert cache.indexed_keys_in_bbox("tracts", 1, b_bounds) == ["s"]

    async def test_eviction_is_logged(self, request: pytest.FixtureRequest, caplog: pytest.LogCaptureFixture) -> None:
        a, b = _chunk_tile(0), _chunk_tile(1)
        loader, _ = _recording_loader([_point_row("a", a), _point_row("b", b)])
        cache = _cache_over(request, loader, max_cached_rows=1)

        with caplog.at_level("DEBUG", logger="threetears.geo.features"):
            await cache.features_in_bbox("tracts", 1, tile_bounds(a))
            await cache.features_in_bbox("tracts", 1, tile_bounds(b))

        assert any("evicted" in record.getMessage() for record in caplog.records)


class TestOneReadNeverFlushesTheWorkingSet:
    """a read whose own rows would claim more than half the bound is answered and not held, so one
    dense chunk or one wide rectangle cannot evict everything the pod was holding for its
    neighbours. the guard counts what the bound counts: row entries, one per empty chunk."""

    async def test_an_oversized_chunk_is_answered_and_not_held(self, request: pytest.FixtureRequest) -> None:
        small, dense = _chunk_tile(0), _chunk_tile(1)
        rows = [_point_row("small", small)] + [_point_row(f"d{i}", dense) for i in range(3)]
        loader, calls = _recording_loader(rows)
        cache = _cache_over(request, loader, max_cached_rows=2)

        await cache.features_in_bbox("tracts", 1, tile_bounds(small))
        found = await cache.features_in_bbox("tracts", 1, tile_bounds(dense))
        assert sorted(row["feature_id"] for row in found) == ["d0", "d1", "d2"]

        await cache.features_in_bbox("tracts", 1, tile_bounds(small))
        assert len(calls) == 2, "the dense chunk flushed the chunk held before it"
        await cache.features_in_bbox("tracts", 1, tile_bounds(dense))
        assert len(calls) == 3, "a chunk over the bound was held"

    async def test_a_wide_read_with_more_rows_than_the_bound_is_not_held(self, request: pytest.FixtureRequest) -> None:
        # 20 chunks (over max_chunk_reads, so one wide read) holding 24 entries: within the bound
        # of 40, over the half of it one read may claim
        north_west = tile_bounds(TileId(z=8, x=48, y=112))
        south_east = tile_bounds(TileId(z=8, x=52, y=115))
        wide = BoundingBox(north_west.min_lon, south_east.min_lat, south_east.max_lon, north_west.max_lat)
        inner = TileId(z=12, x=50 * 16 + 5, y=113 * 16 + 5)
        loader, calls = _recording_loader([_point_row(f"r{i}", inner) for i in range(5)])
        cache = _cache_over(request, loader, max_cached_rows=40)

        found = await cache.features_in_bbox("tracts", 1, wide)
        assert len(calls) == 1
        assert len(found) == 5
        await cache.features_in_bbox("tracts", 1, tile_bounds(inner))
        assert len(calls) == 2, "a wide read over half the bound covered its chunks"

    async def test_a_wide_read_counts_its_empty_chunks_against_the_bound(self, request: pytest.FixtureRequest) -> None:
        # one row, but 1,024 contained chunks: by rows it fits a bound of 100, by entries it does not
        held, inner = _chunk_tile(40), _inside_z3()
        loader, calls = _recording_loader([_point_row("held", held), _point_row("inner", inner)])
        cache = _cache_over(request, loader, max_cached_rows=100)

        await cache.features_in_bbox("tracts", 1, tile_bounds(held))
        await cache.features_in_bbox("tracts", 1, tile_bounds(_Z3))
        await cache.features_in_bbox("tracts", 1, tile_bounds(held))
        assert len(calls) == 2, "the wide read's empty chunks flushed the chunk held before it"
        await cache.features_in_bbox("tracts", 1, tile_bounds(inner))
        assert len(calls) == 3


class TestOneRuleForEveryReadPath:
    """the same rectangle yields the same features whichever path answers it: rows with no feature
    id are dropped and every row is filtered to the rectangle, with or without an L1, narrow or
    wide. a tile built from either is cached as immutable, so the paths must agree."""

    @pytest.mark.parametrize("path", ["no-l1", "chunk", "wide"])
    async def test_id_less_and_outside_rows_are_dropped(self, request: pytest.FixtureRequest, path: str) -> None:
        tile = _inside_z3()
        good = _point_row("good", tile)
        no_id = {"feature_id": None, "bounds": good["bounds"]}
        outside = {"feature_id": "outside", "bounds": BoundingBox(10.0, 10.0, 11.0, 11.0)}

        async def _loader(layer: str, source_version: int, bounds: BoundingBox) -> list[dict[str, Any]]:
            # a loader answering a little more than asked, as a coarse bbox-column query can
            return [good, no_id, outside]

        cache = _cache_over(request, _loader, l1=path != "no-l1")
        bounds = tile_bounds(_Z3) if path == "wide" else tile_bounds(tile)

        found = await cache.features_in_bbox("tracts", 1, bounds)
        assert [row["feature_id"] for row in found] == ["good"]


class TestAReadInProgressKeepsWhatItDependsOn:
    async def test_a_chunk_a_read_holds_survives_eviction_while_it_waits(self, request: pytest.FixtureRequest) -> None:
        # read A covers chunk a, then waits on chunk b's load; other reads push the cache past its
        # bound meanwhile. a is the least recently read chunk, but A still depends on it
        a_tile, b_tile, c_tile, d_tile = _chunk_tile(0), _chunk_tile(1), _chunk_tile(2), _chunk_tile(3)
        rows = [_point_row(name, tile) for name, tile in (("a", a_tile), ("b", b_tile), ("c", c_tile), ("d", d_tile))]
        b_chunk = tile_bounds(TileId(z=FeatureCache.chunk_zoom, x=b_tile.x // 16, y=b_tile.y // 16))
        b_released = asyncio.Event()
        b_requested = asyncio.Event()

        async def _loader(layer: str, source_version: int, bounds: BoundingBox) -> list[dict[str, Any]]:
            if bounds == b_chunk:
                b_requested.set()
                await b_released.wait()
            return [row for row in rows if row["bounds"].intersects(bounds)]

        cache = _cache_over(request, _loader, max_cached_rows=2)
        a_bounds, b_bounds = tile_bounds(a_tile), tile_bounds(b_tile)
        spanning = BoundingBox(a_bounds.min_lon, a_bounds.min_lat, b_bounds.max_lon, b_bounds.max_lat)

        read_a = asyncio.create_task(cache.features_in_bbox("tracts", 1, spanning))
        await b_requested.wait()
        await cache.features_in_bbox("tracts", 1, tile_bounds(c_tile))
        await cache.features_in_bbox("tracts", 1, tile_bounds(d_tile))
        b_released.set()

        found = await read_a
        assert sorted(row["feature_id"] for row in found) == ["a", "b"], "a chunk the read held was evicted under it"


class TestLayersAndInstancesDoNotLeak:
    def test_a_layer_name_with_like_wildcards_matches_only_itself(self, request: pytest.FixtureRequest) -> None:
        cache = _cache_over(request, _empty_loader)
        box = BoundingBox(-1, -1, 1, 1)
        cache.index_feature("axb", 1, "x-feature", box)
        cache.index_feature("a_b", 1, "underscore-feature", box)

        assert cache.indexed_keys_in_bbox("a_b", 1, box) == ["underscore-feature"]
        assert cache.indexed_keys_in_bbox("axb", 1, box) == ["x-feature"]
        assert cache.indexed_keys_in_bbox("a%b", 1, box) == []

    async def test_two_caches_of_one_scope_on_one_l1_do_not_evict_each_others_index(
        self, request: pytest.FixtureRequest
    ) -> None:
        # both caches open the same named SQLite database, so they share its tables
        a_tile, b_tile = _chunk_tile(0), _chunk_tile(1)
        loader, calls = _recording_loader([_point_row("f", a_tile), _point_row("g", b_tile)])
        first = _cache_over(request, loader)
        second = _cache_over(request, loader, max_cached_rows=1)

        await first.features_in_bbox("tracts", 1, tile_bounds(a_tile))
        await second.features_in_bbox("tracts", 1, tile_bounds(a_tile))
        await second.features_in_bbox("tracts", 1, tile_bounds(b_tile))  # evicts its own copy of "f"

        found = await first.features_in_bbox("tracts", 1, tile_bounds(a_tile))
        assert len(calls) == 3, "the first cache's chunk should still be covered"
        assert [row["feature_id"] for row in found] == ["f"], "one cache's eviction removed the other's index entry"


class TestOneReadClaimsAtMostHalfTheBound:
    async def test_a_read_over_several_small_chunks_is_counted_whole(self, request: pytest.FixtureRequest) -> None:
        # four one-row chunks in one read, against a bound of 4: each fits half the bound alone,
        # together they would claim all of it and flush the chunk held before
        held = _chunk_tile(8)
        tiles = [_chunk_tile(i) for i in range(4)]
        rows = [_point_row("held", held)] + [_point_row(f"r{i}", tile) for i, tile in enumerate(tiles)]
        loader, calls = _recording_loader(rows)
        cache = _cache_over(request, loader, max_cached_rows=4)
        first, last = tile_bounds(tiles[0]), tile_bounds(tiles[-1])
        spanning = BoundingBox(first.min_lon, first.min_lat, last.max_lon, last.max_lat)

        await cache.features_in_bbox("tracts", 1, tile_bounds(held))
        found = await cache.features_in_bbox("tracts", 1, spanning)
        assert sorted(row["feature_id"] for row in found) == ["r0", "r1", "r2", "r3"]

        loads = len(calls)
        await cache.features_in_bbox("tracts", 1, tile_bounds(held))
        assert len(calls) == loads, "one read's chunks together flushed the chunk held before it"


class TestACollectedCache:
    async def test_leaves_no_index_entries(self, request: pytest.FixtureRequest) -> None:
        # caches of one scope share the L1, each with its own R-Tree, so one that is dropped must
        # take its entries -- and only its own: a second live cache on the same L1 keeps every
        # entry it holds
        tile = _chunk_tile(0)
        loader, calls = _recording_loader([_point_row("f", tile)])
        cache = _cache_over(request, loader)
        survivor = _cache_over(request, loader)
        backend = SQLiteBackend(f"geo_wide_{abs(hash(request.node.nodeid))}")
        backend.initialize(MetaData())
        await cache.features_in_bbox("tracts", 1, tile_bounds(tile))
        await survivor.features_in_bbox("tracts", 1, tile_bounds(tile))

        def entries() -> int:
            trees = backend.execute_query(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE ?",
                (f"geo_features_{SCOPE}_%_rtree",),
            )
            return sum(int(backend.execute_query(f"SELECT count(*) AS n FROM {row['name']}")[0]["n"]) for row in trees)

        assert entries() == 2

        del cache
        gc.collect()

        assert entries() == 1
        assert survivor.indexed_keys_in_bbox("tracts", 1, tile_bounds(tile)) == ["f"]
        found = await survivor.features_in_bbox("tracts", 1, tile_bounds(tile))
        assert [row["feature_id"] for row in found] == ["f"]
        assert len(calls) == 2, "the surviving cache's chunk should still be covered"


#: the R-Tree statements FeatureCache sent before its key map moved into the instance (2026-11-15): the tree and a
#: ``(id, feature_key)`` map table, written and read through the map
_OLD_INDEX_DDL = (
    "CREATE VIRTUAL TABLE old_rtree USING rtree(id, min_x, max_x, min_y, max_y)",
    "CREATE TABLE old_map (id INTEGER PRIMARY KEY AUTOINCREMENT, feature_key TEXT NOT NULL UNIQUE)",
)
_OLD_MAP_INSERT = "INSERT OR IGNORE INTO old_map (feature_key) VALUES (?)"
_OLD_TREE_PUT = (
    "INSERT OR REPLACE INTO old_rtree (id, min_x, max_x, min_y, max_y) SELECT id, ?, ?, ?, ? FROM old_map "
    "WHERE feature_key = ?"
)
_OLD_READ = (
    "SELECT m.feature_key AS feature_key FROM old_rtree r JOIN old_map m ON m.id = r.id "
    "WHERE r.max_x >= ? AND r.min_x <= ? AND r.max_y >= ? AND r.min_y <= ? AND substr(m.feature_key, 1, ?) = ?"
)

#: every box any test row lies in
_WORLD = BoundingBox(-180.0, -85.0, 180.0, 85.0)


def _tree_entries(backend: Any) -> int:
    """every entry the L1's feature-cache R-Trees hold, found by name as the finalizer test finds them."""
    trees = backend.execute_query(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE ?", (f"geo_features_{SCOPE}_%_rtree",)
    )
    return sum(int(backend.execute_query(f"SELECT count(*) AS n FROM {row['name']}")[0]["n"]) for row in trees)


class TestTheIndexAnswersTheOldStatements:
    def test_every_box_answers_what_the_map_table_answered(self, cache: FeatureCache) -> None:
        """Flips if indexed_keys_in_bbox answers other feature ids than the old tree-and-map statements did over the
        same writes: a box bound dropped or flipped, another layer or generation let in (a layer name carrying
        LIKE's % and _ included), or a re-indexed feature's old box kept beside its new one."""
        rng = random.Random(20261115)
        old = sqlite3.connect(":memory:")
        for ddl in _OLD_INDEX_DDL:
            old.execute(ddl)
        layers = ["tracts", "a_b", "axb", "t%s", "tracts2"]

        def box() -> BoundingBox:
            lon, lat = rng.uniform(-120.0, -70.0), rng.uniform(25.0, 48.0)
            return BoundingBox(lon, lat, lon + rng.uniform(0.0, 4.0), lat + rng.uniform(0.0, 4.0))

        def index(layer: str, version: int, feature_id: str, bounds: BoundingBox) -> None:
            cache.index_feature(layer, version, feature_id, bounds)
            # the old key, less the instance's token (one instance: the same for every key)
            key = f"{layer}\x1f{version}\x1f{feature_id}"
            old.execute(_OLD_MAP_INSERT, (key,))
            old.execute(_OLD_TREE_PUT, (bounds.min_lon, bounds.max_lon, bounds.min_lat, bounds.max_lat, key))

        features = [(rng.choice(layers), rng.choice((1, 2)), f"f{n}") for n in range(400)]
        for layer, version, feature_id in features:
            index(layer, version, feature_id, box())
        for layer, version, feature_id in rng.sample(features, 60):  # moved: indexed again under a new box
            index(layer, version, feature_id, box())
        assert cache.l1_backend is not None
        assert _tree_entries(cache.l1_backend) == old.execute("SELECT count(*) FROM old_rtree").fetchone()[0]

        checked = answered = 0
        for _ in range(300):
            asked = box()
            for layer in layers:
                for version in (1, 2):
                    prefix = f"{layer}\x1f{version}\x1f"
                    expected = sorted(
                        str(row[0]).split("\x1f", 2)[2]
                        for row in old.execute(
                            _OLD_READ,
                            (asked.min_lon, asked.max_lon, asked.min_lat, asked.max_lat, len(prefix), prefix),
                        ).fetchall()
                    )
                    assert sorted(cache.indexed_keys_in_bbox(layer, version, asked)) == expected, (
                        layer,
                        version,
                        asked,
                    )
                    checked += 1
                    answered += bool(expected)
        assert answered > checked // 10, "too few boxes answered anything to compare"

    async def test_an_evicted_chunk_s_features_leave_the_tree_itself(self, request: pytest.FixtureRequest) -> None:
        """Flips if eviction stops answering a feature but leaves its entry in the tree (the tree growing behind
        the bound): the tree holds exactly the features the cache answers, after every read."""
        tiles = [_chunk_tile(dx) for dx in range(6)]
        rows = [_point_row(f"f{n}-{dx}", tile) for dx, tile in enumerate(tiles) for n in range(3)]
        loader, _ = _recording_loader(rows)
        cache = _cache_over(request, loader, max_cached_rows=6)
        for tile in [*tiles, *tiles[:2]]:
            await cache.features_in_bbox("tracts", 1, tile_bounds(tile))
            assert cache.l1_backend is not None
            held = cache.indexed_keys_in_bbox("tracts", 1, _WORLD)
            assert 0 < len(held) <= 6
            assert _tree_entries(cache.l1_backend) == len(held)


class TestAnL1HoldingAnOlderCachesTables:
    async def test_a_new_cache_reads_only_its_own_tree(self, request: pytest.FixtureRequest) -> None:
        """the upgrade path: the SQLite L1 is a named in-memory database (memdb), so a pod's new process starts
        with no tables and nothing is migrated -- each cache creates its own tree on first use. Should an older
        cache's tree and key map still sit in the same L1, a new cache never reads them: flips if it answers their
        entries or they stop it indexing and reading its own."""
        tile = _chunk_tile(0)
        loader, _ = _recording_loader([_point_row("mine", tile)])
        cache = _cache_over(request, loader)
        backend = cache.l1_backend
        assert backend is not None
        conn = backend.get_connection()
        stale = f"geo_features_{SCOPE}_0123456789ab"
        conn.execute(f"CREATE VIRTUAL TABLE {stale}_rtree USING rtree(id, min_x, max_x, min_y, max_y)")
        conn.execute(
            f"CREATE TABLE {stale}_rtree_map (id INTEGER PRIMARY KEY AUTOINCREMENT, feature_key TEXT NOT NULL UNIQUE)"
        )
        conn.execute(
            f"INSERT INTO {stale}_rtree_map (id, feature_key) VALUES (1, ?)", ("0123456789ab\x1ftracts\x1f1\x1fold",)
        )
        b = tile_bounds(tile)
        conn.execute(f"INSERT INTO {stale}_rtree VALUES (1, ?, ?, ?, ?)", (b.min_lon, b.max_lon, b.min_lat, b.max_lat))
        conn.commit()

        found = await cache.features_in_bbox("tracts", 1, tile_bounds(tile))

        assert [row["feature_id"] for row in found] == ["mine"]
        assert cache.indexed_keys_in_bbox("tracts", 1, tile_bounds(tile)) == ["mine"]
