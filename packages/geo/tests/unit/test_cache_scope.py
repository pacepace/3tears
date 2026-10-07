"""two tile sources never answer for each other through a shared cache tier.

a customer's datasource layer and a platform layer can carry the same name, and both
start at generation 1, so their tiles share an address. every cache tier the framework
keys by table -- the pod-local L1, the NATS L2, the cross-pod build lock, the registry
entry -- must therefore be scoped by the source, or whichever built a tile first answers
for both, across tenants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mapbox_vector_tile
import pytest
from shapely.geometry import Polygon
from sqlalchemy import Column as SAColumn
from sqlalchemy import Integer, LargeBinary, MetaData, String, Table
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.geo import FeatureCache, LayerDefinition, TileCollection, check_cache_scope
from threetears.geo.bands import AggregateSpec, FeatureSpec
from threetears.geo.tiles import BoundingBox, tile_bounds, tile_for_point
from threetears.object_store.filesystem import FilesystemObjectStore

_LON, _LAT = -112.07, 33.45
_TILE = tile_for_point(_LON, _LAT, 10)
_KEY = ("census_tracts", 1, _TILE.z, _TILE.x, _TILE.y)
_SCOPES = ("ds_customer", "ns_platform")


def _layer() -> LayerDefinition:
    return LayerDefinition(
        name="census_tracts",
        feature_id_column="geoid",
        geometry_column="geometry_wkb",
        aggregate=AggregateSpec(rollup_column="state_fips", measures={"total_unreg": "sum"}),
        features=FeatureSpec(attributes=("total_unreg",), feature_id_column="geoid"),
        crossover_zoom=9,
        minzoom=4,
        maxzoom=14,
    )


def _rows(count: int) -> list[dict[str, Any]]:
    bounds = tile_bounds(_TILE)
    out = []
    for n in range(count):
        lon = bounds.min_lon + 0.001 * (n + 1)
        lat = bounds.min_lat + 0.001 * (n + 1)
        box = Polygon([(lon, lat), (lon + 0.0002, lat), (lon + 0.0002, lat + 0.0002), (lon, lat + 0.0002)])
        out.append({"geoid": f"{n:04d}", "geometry_wkb": box.wkb, "total_unreg": n, "state_fips": "04"})
    return out


def _shared_l1(name: str) -> SQLiteBackend:
    """one pod's L1, holding both sources' tables, as a hub serving both would."""
    metadata = MetaData()
    # the unscoped names too, so a collection that ignored its scope would run and share
    # one table rather than fail on a missing one: the leak is what these tests detect
    for suffix in [f"_{scope}" for scope in _SCOPES] + [""]:
        Table(
            f"geo_tiles{suffix}",
            metadata,
            SAColumn("layer", String, primary_key=True),
            SAColumn("version", Integer, primary_key=True),
            SAColumn("z", Integer, primary_key=True),
            SAColumn("x", Integer, primary_key=True),
            SAColumn("y", Integer, primary_key=True),
            SAColumn("mvt", LargeBinary),
        )
        Table(
            f"geo_features{suffix}",
            metadata,
            SAColumn("layer", String, primary_key=True),
            SAColumn("source_version", Integer, primary_key=True),
            SAColumn("feature_id", String, primary_key=True),
        )
    backend = SQLiteBackend(name)
    backend.initialize(metadata)
    return backend


def _source(
    registry: CollectionRegistry, tmp_path: Path, scope: str, rows: list[dict[str, Any]], calls: list[str]
) -> TileCollection:
    async def _loader(layer: str, version: int, bounds: BoundingBox) -> list[dict[str, Any]]:
        calls.append(scope)
        return list(rows)

    return TileCollection(
        registry,
        DefaultCoreConfig(),
        None,
        None,
        layers={"census_tracts": _layer()},
        loader=_loader,
        # each source's own durable tier, so only a shared cache tier could leak
        object_store=FilesystemObjectStore(tmp_path / scope),
        datasource_name=scope,
        cache_scope=scope,
    )


class TestTiles:
    async def test_a_tile_one_source_built_is_not_served_to_another_through_l1(
        self, tmp_path: Path, request: pytest.FixtureRequest
    ) -> None:
        registry = CollectionRegistry()
        registry.configure(
            l1_backend=_shared_l1(f"scope_l1_{abs(hash(request.node.nodeid))}"), l2_client=None, l3_pool=None
        )
        calls: list[str] = []
        customer = _source(registry, tmp_path, "ds_customer", _rows(2), calls)
        platform = _source(registry, tmp_path, "ns_platform", _rows(4), calls)

        assert await customer.get(_KEY) is not None
        theirs = await platform.get(_KEY)
        assert theirs is not None
        assert calls == ["ds_customer", "ns_platform"], f"a source was served another's tile: {calls}"
        assert len(mapbox_vector_tile.decode(theirs.mvt)["census_tracts"]["features"]) == 4

    def test_each_source_registers_and_locks_under_its_own_name(self, tmp_path: Path) -> None:
        registry = CollectionRegistry()
        registry.configure(l1_backend=None, l2_client=None, l3_pool=None)
        customer = _source(registry, tmp_path, "ds_customer", [], [])
        platform = _source(registry, tmp_path, "ns_platform", [], [])
        assert registry.get_collection(customer.table_name) is customer
        assert registry.get_collection(platform.table_name) is platform
        assert customer.build_lock_key(_KEY) != platform.build_lock_key(_KEY)


class TestFeatures:
    async def test_features_one_source_cached_are_not_read_by_another(self, request: pytest.FixtureRequest) -> None:
        registry = CollectionRegistry()
        registry.configure(
            l1_backend=_shared_l1(f"scope_features_{abs(hash(request.node.nodeid))}"), l2_client=None, l3_pool=None
        )

        def _cache(scope: str) -> FeatureCache:
            async def _loader(layer: str, source_version: int, bounds: BoundingBox) -> list[dict[str, Any]]:
                return []

            return FeatureCache(
                registry,
                DefaultCoreConfig(),
                None,
                None,
                loader=_loader,
                bounds_of=lambda row: BoundingBox(-1, -1, 1, 1),
                feature_id_column="geoid",
                cache_scope=scope,
            )

        customer, platform = _cache("ds_customer"), _cache("ns_platform")
        customer.index_feature("census_tracts", 1, "0001", BoundingBox(-1, -1, 1, 1))
        assert customer.indexed_keys_in_bbox("census_tracts", 1, BoundingBox(-1, -1, 1, 1)) != []
        assert platform.indexed_keys_in_bbox("census_tracts", 1, BoundingBox(-1, -1, 1, 1)) == []
        assert customer.table_name != platform.table_name


class TestTheScopeItself:
    @pytest.mark.parametrize(
        "scope",
        ["", "Ds_upper", "1starts_with_digit", "has-hyphen", "has space", 'quote"', "x" * 49, "ds_ok;drop"],
    )
    def test_an_unusable_scope_is_refused(self, scope: str) -> None:
        with pytest.raises(ValueError, match="cache scope"):
            check_cache_scope(scope)

    def test_a_datasource_and_a_namespace_identity_are_accepted(self) -> None:
        hex32 = "0123456789abcdef0123456789abcdef"
        assert check_cache_scope(f"ds_{hex32}") == f"ds_{hex32}"
        assert check_cache_scope(f"ns_{hex32}") == f"ns_{hex32}"

    def test_a_collection_refuses_an_unusable_scope_before_touching_the_registry(self, tmp_path: Path) -> None:
        registry = CollectionRegistry()
        registry.configure(l1_backend=None, l2_client=None, l3_pool=None)
        with pytest.raises(ValueError, match="cache scope"):
            _source(registry, tmp_path, "bad scope", [], [])
        assert registry.get_collection("geo_tiles_bad scope") is None
