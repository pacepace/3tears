"""tests for :class:`TileCollection`, end to end from source rows to MVT bytes.

the durable tier is a real :class:`FilesystemObjectStore` over a tmp
directory rather than a mock, so the streaming put/read contract is
genuinely exercised. tiles round-trip through the actual MVT decoder --
asserting on bytes I produced with my own encoder would only prove the two
agree with each other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mapbox_vector_tile
import pytest
from shapely.geometry import Polygon

from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.geo.bands import AggregateSpec, FeatureSpec
from threetears.geo.collection import LayerDefinition, TileCollection, ViewportRequest
from threetears.geo.tiles import BoundingBox, TileId, tile_bounds
from threetears.object_store.filesystem import FilesystemObjectStore

_PHOENIX_LON = -112.07
_PHOENIX_LAT = 33.45


def _tracts_layer(**overrides: Any) -> LayerDefinition:
    defaults: dict[str, Any] = {
        "name": "census_tracts",
        "feature_id_column": "geoid",
        "geometry_column": "geometry_wkb",
        "aggregate": AggregateSpec(rollup_column="state_fips", measures={"total_unreg": "sum"}),
        "features": FeatureSpec(attributes=("total_unreg",), feature_id_column="geoid"),
        "crossover_zoom": 9,
        "minzoom": 4,
        "maxzoom": 14,
    }
    defaults.update(overrides)
    return LayerDefinition(**defaults)


def _tract_row(geoid: str, *, lon: float, lat: float, unreg: int = 100, state: str = "04") -> dict[str, Any]:
    box = Polygon([(lon, lat), (lon + 0.01, lat), (lon + 0.01, lat + 0.01), (lon, lat + 0.01)])
    return {"geoid": geoid, "geometry_wkb": box.wkb, "total_unreg": unreg, "state_fips": state}


def _make_collection(
    tmp_path: Path,
    *,
    rows: list[dict[str, Any]] | None = None,
    layers: dict[str, LayerDefinition] | None = None,
) -> tuple[TileCollection, list[tuple[str, int, BoundingBox]]]:
    """build a collection over a real filesystem object store.

    returns the collection plus a list recording every loader call, so tests
    can assert on how often source data was read.
    """
    calls: list[tuple[str, int, BoundingBox]] = []
    source = rows if rows is not None else []

    async def _loader(layer: str, version: int, bounds: BoundingBox) -> list[dict[str, Any]]:
        calls.append((layer, version, bounds))
        return list(source)

    registry = CollectionRegistry()
    registry.configure(l1_backend=None, l2_client=None, l3_pool=None)
    collection = TileCollection(
        registry,
        DefaultCoreConfig(),
        None,
        None,
        layers=layers if layers is not None else {"census_tracts": _tracts_layer()},
        loader=_loader,
        object_store=FilesystemObjectStore(tmp_path),
        datasource_name="aibotsmap-data",
    )
    return collection, calls


class TestLayerDefinition:
    def test_rejects_a_layer_declaring_no_geometry(self) -> None:
        # a layer with no geometry produces empty tiles forever, which on a
        # map is indistinguishable from empty geography.
        with pytest.raises(ValueError, match="declares no geometry"):
            LayerDefinition(name="broken", feature_id_column="id")

    def test_accepts_a_point_layer(self) -> None:
        layer = LayerDefinition(
            name="locations",
            feature_id_column="id",
            longitude_column="longitude",
            latitude_column="latitude",
        )
        geometry = layer.geometry_of({"longitude": -112.0, "latitude": 33.4})
        assert geometry is not None
        assert (geometry.x, geometry.y) == (-112.0, 33.4)

    def test_point_layer_tolerates_missing_coordinates(self) -> None:
        layer = LayerDefinition(
            name="locations",
            feature_id_column="id",
            longitude_column="longitude",
            latitude_column="latitude",
        )
        assert layer.geometry_of({"longitude": None, "latitude": None}) is None


class TestQuantization:
    def test_viewports_in_one_tile_share_a_key(self, tmp_path: Path) -> None:
        """the property the whole design rests on.

        two users looking at slightly different points in the same tile must
        produce the same cache key, or nothing is ever shared between them.
        """
        collection, _ = _make_collection(tmp_path)
        first = collection.derive_key(ViewportRequest("census_tracts", _PHOENIX_LON, _PHOENIX_LAT, 10, 3))
        second = collection.derive_key(ViewportRequest("census_tracts", _PHOENIX_LON + 0.001, _PHOENIX_LAT, 10, 3))
        assert first == second

    def test_different_versions_are_different_keys(self, tmp_path: Path) -> None:
        # a rebuild is a new address, not an invalidation.
        collection, _ = _make_collection(tmp_path)
        v3 = collection.derive_key(ViewportRequest("census_tracts", _PHOENIX_LON, _PHOENIX_LAT, 10, 3))
        v4 = collection.derive_key(ViewportRequest("census_tracts", _PHOENIX_LON, _PHOENIX_LAT, 10, 4))
        assert v3 != v4

    def test_different_layers_are_different_keys(self, tmp_path: Path) -> None:
        collection, _ = _make_collection(tmp_path)
        a = collection.derive_key(ViewportRequest("census_tracts", _PHOENIX_LON, _PHOENIX_LAT, 10, 3))
        b = collection.derive_key(ViewportRequest("locations", _PHOENIX_LON, _PHOENIX_LAT, 10, 3))
        assert a != b


class TestObjectKey:
    def test_is_derivable_from_the_address_alone(self, tmp_path: Path) -> None:
        # a CDN fetching a tile has only the URL and cannot consult a
        # database to translate it into an opaque object id.
        collection, _ = _make_collection(tmp_path)
        key = collection.object_key(("census_tracts", 3, 8, 40, 98))
        assert key == "shared/tiles/aibotsmap-data/census-tracts/v3/8/40/98.mvt"

    def test_keeps_the_extension(self, tmp_path: Path) -> None:
        collection, _ = _make_collection(tmp_path)
        assert collection.object_key(("census_tracts", 1, 0, 0, 0)).endswith(".mvt")

    def test_versions_do_not_share_a_key(self, tmp_path: Path) -> None:
        collection, _ = _make_collection(tmp_path)
        assert collection.object_key(("census_tracts", 3, 8, 40, 98)) != collection.object_key(
            ("census_tracts", 4, 8, 40, 98)
        )


class TestBuild:
    async def test_builds_a_decodable_tile(self, tmp_path: Path) -> None:
        tile = TileId(z=12, x=812, y=1620)
        bounds = tile_bounds(tile)
        rows = [_tract_row("04013010101", lon=bounds.min_lon + 0.001, lat=bounds.min_lat + 0.001, unreg=250)]
        collection, _ = _make_collection(tmp_path, rows=rows)

        built = await collection.compute(("census_tracts", 3, tile.z, tile.x, tile.y))
        assert built is not None

        decoded = mapbox_vector_tile.decode(built["mvt"])
        feature = decoded["census_tracts"]["features"][0]
        assert feature["properties"]["total_unreg"] == 250
        # a geoid is not a uint64, so it travels as a property rather than as
        # the wire-level MVT id -- see TestFeatureIdentity below.
        assert feature["properties"]["geoid"] == "04013010101"

    async def test_low_zoom_aggregates_rather_than_sampling(self, tmp_path: Path) -> None:
        """below the crossover a tile carries rollups with real totals.

        the alternative -- individual features simplified and dropped -- shows
        an arbitrary sample of the data and reads as though it were all of it.
        """
        tile = TileId(z=5, x=6, y=12)
        bounds = tile_bounds(tile)
        lon = bounds.min_lon + 0.1
        lat = bounds.min_lat + 0.1
        rows = [
            _tract_row("a", lon=lon, lat=lat, unreg=100),
            _tract_row("b", lon=lon + 0.02, lat=lat, unreg=250),
        ]
        collection, _ = _make_collection(tmp_path, rows=rows)

        built = await collection.compute(("census_tracts", 3, tile.z, tile.x, tile.y))
        assert built is not None
        decoded = mapbox_vector_tile.decode(built["mvt"])
        features = decoded["census_tracts"]["features"]
        assert len(features) == 1, "expected one rollup, not per-tract features"
        assert features[0]["properties"]["total_unreg"] == 350
        assert features[0]["properties"]["count"] == 2

    async def test_high_zoom_emits_individual_features(self, tmp_path: Path) -> None:
        tile = TileId(z=12, x=812, y=1620)
        bounds = tile_bounds(tile)
        rows = [
            _tract_row("a", lon=bounds.min_lon + 0.001, lat=bounds.min_lat + 0.001),
            _tract_row("b", lon=bounds.min_lon + 0.002, lat=bounds.min_lat + 0.002),
        ]
        collection, _ = _make_collection(tmp_path, rows=rows)
        built = await collection.compute(("census_tracts", 3, tile.z, tile.x, tile.y))
        assert built is not None
        assert len(mapbox_vector_tile.decode(built["mvt"])["census_tracts"]["features"]) == 2

    async def test_undeclared_layer_is_a_miss_not_an_error(self, tmp_path: Path) -> None:
        collection, _ = _make_collection(tmp_path)
        assert await collection.compute(("no_such_layer", 3, 10, 1, 1)) is None

    async def test_zoom_outside_the_layer_range_is_a_miss(self, tmp_path: Path) -> None:
        # a client asking for a zoom the layer does not serve should get
        # nothing to draw, not a failure.
        collection, _ = _make_collection(tmp_path)
        assert await collection.compute(("census_tracts", 3, 2, 1, 1)) is None

    async def test_empty_region_still_builds_a_tile(self, tmp_path: Path) -> None:
        # ocean is a legitimate, cacheable answer; returning None would make
        # every request for it rebuild forever.
        collection, _ = _make_collection(tmp_path, rows=[])
        built = await collection.compute(("census_tracts", 3, 12, 812, 1620))
        assert built is not None
        assert isinstance(built["mvt"], bytes)


class TestFeatureIdentity:
    """MVT feature ids are uint64 by specification.

    a string id -- a census geoid, a UUID -- is silently coerced to 0 by the
    encoder. that would collapse every feature in a tile onto one id and
    break any client-side join keyed on it, without raising anything. so
    non-integer ids travel as a property and the client binds through
    MapLibre's ``promoteId``.
    """

    async def test_string_ids_survive_as_a_property(self, tmp_path: Path) -> None:
        tile = TileId(z=12, x=812, y=1620)
        bounds = tile_bounds(tile)
        rows = [
            _tract_row("04013010101", lon=bounds.min_lon + 0.001, lat=bounds.min_lat + 0.001),
            _tract_row("04013010102", lon=bounds.min_lon + 0.003, lat=bounds.min_lat + 0.003),
        ]
        collection, _ = _make_collection(tmp_path, rows=rows)
        built = await collection.compute(("census_tracts", 3, tile.z, tile.x, tile.y))
        assert built is not None

        features = mapbox_vector_tile.decode(built["mvt"])["census_tracts"]["features"]
        geoids = {feature["properties"]["geoid"] for feature in features}
        assert geoids == {"04013010101", "04013010102"}, "ids must stay distinct per feature"

    async def test_string_ids_are_not_written_to_the_wire_id(self, tmp_path: Path) -> None:
        tile = TileId(z=12, x=812, y=1620)
        bounds = tile_bounds(tile)
        rows = [_tract_row("04013010101", lon=bounds.min_lon + 0.001, lat=bounds.min_lat + 0.001)]
        collection, _ = _make_collection(tmp_path, rows=rows)
        built = await collection.compute(("census_tracts", 3, tile.z, tile.x, tile.y))
        assert built is not None
        feature = mapbox_vector_tile.decode(built["mvt"])["census_tracts"]["features"][0]
        # absent, or at most a meaningless 0 -- never a corrupted stand-in
        # that a client might mistake for the real identity.
        assert feature.get("id", 0) == 0

    async def test_integer_ids_do_reach_the_wire_id(self, tmp_path: Path) -> None:
        tile = TileId(z=12, x=812, y=1620)
        bounds = tile_bounds(tile)
        layer = _tracts_layer(
            name="numbered",
            feature_id_column="row_id",
            features=FeatureSpec(attributes=("row_id",), feature_id_column="row_id"),
        )
        row = _tract_row("x", lon=bounds.min_lon + 0.001, lat=bounds.min_lat + 0.001)
        row["row_id"] = 4242
        collection, _ = _make_collection(tmp_path, rows=[row], layers={"numbered": layer})
        built = await collection.compute(("numbered", 3, tile.z, tile.x, tile.y))
        assert built is not None
        feature = mapbox_vector_tile.decode(built["mvt"])["numbered"]["features"][0]
        assert feature["id"] == 4242


class TestDurableTier:
    async def test_round_trips_through_the_object_store(self, tmp_path: Path) -> None:
        tile = TileId(z=12, x=812, y=1620)
        bounds = tile_bounds(tile)
        rows = [_tract_row("04013010101", lon=bounds.min_lon + 0.001, lat=bounds.min_lat + 0.001)]
        collection, _ = _make_collection(tmp_path, rows=rows)

        built = await collection.compute(("census_tracts", 3, tile.z, tile.x, tile.y))
        assert built is not None
        await collection.save_to_store(built)

        loaded = await collection.load_derived(("census_tracts", 3, tile.z, tile.x, tile.y))
        assert loaded is not None
        assert loaded["mvt"] == built["mvt"]

    async def test_absent_tile_reads_as_a_miss(self, tmp_path: Path) -> None:
        collection, _ = _make_collection(tmp_path)
        assert await collection.load_derived(("census_tracts", 3, 12, 1, 1)) is None

    async def test_a_built_tile_is_not_rebuilt(self, tmp_path: Path) -> None:
        """the entire point of the durable tier."""
        tile = TileId(z=12, x=812, y=1620)
        bounds = tile_bounds(tile)
        rows = [_tract_row("a", lon=bounds.min_lon + 0.001, lat=bounds.min_lat + 0.001)]
        collection, calls = _make_collection(tmp_path, rows=rows)
        key = ("census_tracts", 3, tile.z, tile.x, tile.y)

        first = await collection.fetch_from_store(key)
        second = await collection.fetch_from_store(key)

        assert first is not None
        assert second is not None
        assert first["mvt"] == second["mvt"]
        assert len(calls) == 1, f"source was read {len(calls)} times for one tile"

    async def test_delete_reclaims_the_artifact(self, tmp_path: Path) -> None:
        # generation reclamation deletes by version; a tile that survived
        # would keep serving a vintage whose source rows are gone.
        tile = TileId(z=12, x=812, y=1620)
        collection, _ = _make_collection(tmp_path)
        key = ("census_tracts", 3, tile.z, tile.x, tile.y)
        await collection.fetch_from_store(key)
        assert await collection.load_derived(key) is not None

        await collection.delete_from_store(key)
        assert await collection.load_derived(key) is None

    async def test_versions_are_independent_artifacts(self, tmp_path: Path) -> None:
        tile = TileId(z=12, x=812, y=1620)
        collection, _ = _make_collection(tmp_path)
        await collection.fetch_from_store(("census_tracts", 3, tile.z, tile.x, tile.y))
        assert await collection.load_derived(("census_tracts", 4, tile.z, tile.x, tile.y)) is None


class TestSerialization:
    def test_round_trips_a_payload_containing_the_separator(self, tmp_path: Path) -> None:
        """MVT is protobuf and may contain the separator byte anywhere.

        an unbounded split would truncate the payload at whatever byte
        happened to match, producing a tile that decodes to garbage rather
        than failing loudly.
        """
        collection, _ = _make_collection(tmp_path)
        row = {"layer": "census_tracts", "version": 3, "z": 12, "x": 812, "y": 1620, "mvt": b"\x1a\x1f\x1f\x00mvt"}
        assert collection.deserialize(collection.serialize(row)) == row

    def test_round_trips_an_empty_payload(self, tmp_path: Path) -> None:
        collection, _ = _make_collection(tmp_path)
        row = {"layer": "l", "version": 1, "z": 0, "x": 0, "y": 0, "mvt": b""}
        assert collection.deserialize(collection.serialize(row)) == row
