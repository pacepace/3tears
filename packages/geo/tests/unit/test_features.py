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
        "geo_features",
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
            "geo_features",
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
