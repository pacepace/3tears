"""three-tier integration test for tiles: real SQLite L1, real NATS KV L2.

the unit tests configure ``l1_backend=None, l2_client=None``, so every read
falls straight through to the durable tier. that leaves the two tiers this
platform exists to provide entirely unexercised, and the L2 path in
particular is where a tile is most likely to break: MVT is protobuf, it
travels through a serialize/deserialize round trip that is *not* JSON, and
the KV value has a size ceiling.

so this runs the real thing. one NATS container, two collections standing
in for two pods, and assertions on what actually crosses between them.

requires docker; gated by ``pytest.mark.integration``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
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
from threetears.geo.bands import AggregateSpec, FeatureSpec
from threetears.geo.collection import LayerDefinition, TileCollection
from threetears.geo.tiles import BoundingBox, TileId, tile_bounds
from threetears.nats import NatsClient, set_default_namespace
from threetears.object_store.filesystem import FilesystemObjectStore

pytestmark = pytest.mark.integration

_TILE = TileId(z=12, x=812, y=1620)


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
    out: list[dict[str, Any]] = []
    for n in range(count):
        lon = bounds.min_lon + 0.0005 * n
        lat = bounds.min_lat + 0.0005 * n
        box = Polygon([(lon, lat), (lon + 0.0002, lat), (lon + 0.0002, lat + 0.0002), (lon, lat + 0.0002)])
        out.append({"geoid": f"0401301{n:04d}", "geometry_wkb": box.wkb, "total_unreg": 100 + n, "state_fips": "04"})
    return out


def _metadata() -> MetaData:
    metadata = MetaData()
    Table(
        "geo_tiles",
        metadata,
        SAColumn("layer", String, primary_key=True),
        SAColumn("version", Integer, primary_key=True),
        SAColumn("z", Integer, primary_key=True),
        SAColumn("x", Integer, primary_key=True),
        SAColumn("y", Integer, primary_key=True),
        SAColumn("mvt", LargeBinary),
    )
    return metadata


@pytest.fixture
async def nats_client(nats_container: str) -> AsyncIterator[NatsClient]:
    set_default_namespace("3tears")
    client = await NatsClient.connect(
        nats_url=nats_container,
        nats_subject_namespace="3tears",
        client_name="geo-tile-tier-test",
    )
    try:
        yield client
    finally:
        await client.shutdown()


def _make_pod(
    tmp_path: Path,
    nats_client: NatsClient,
    *,
    name: str,
    rows: list[dict[str, Any]],
    calls: list[str],
    isolation: str,
) -> TileCollection:
    """one pod: its own SQLite L1, the shared NATS L2, the shared object store.

    ``isolation`` scopes the L1 database name so two tests do not share one.
    SQLiteBackend opens a *named* shared in-memory database, which is
    deliberate in production -- one L1 per process -- and would otherwise
    leak a warm cache from one test into the next. The L2 key space is
    isolated separately, by giving each test its own tile version.
    """

    async def _loader(layer: str, version: int, bounds: BoundingBox) -> list[dict[str, Any]]:
        calls.append(name)
        return list(rows)

    backend = SQLiteBackend(f"geo_tiles_{isolation}_{name}")
    backend.initialize(_metadata())
    registry = CollectionRegistry()
    registry.configure(l1_backend=backend, l2_client=nats_client, l3_pool=None)
    return TileCollection(
        registry,
        DefaultCoreConfig(),
        nats_client,
        None,
        layers={"census_tracts": _layer()},
        loader=_loader,
        object_store=FilesystemObjectStore(tmp_path),
        datasource_name="aibotsmap-data",
    )


class TestL2RoundTrip:
    async def test_mvt_survives_the_l2_serialize_round_trip(
        self,
        tmp_path: Path,
        nats_client: NatsClient,
    ) -> None:
        """a tile must come back off L2 byte-identical and still decodable.

        MVT is protobuf: arbitrary bytes, including the separator this
        collection's own framing uses. a serialization bug here does not
        raise -- it yields a tile that decodes to nothing, or to garbage.
        """
        calls: list[str] = []
        pod = _make_pod(tmp_path, nats_client, name="a", rows=_rows(3), calls=calls, isolation="roundtrip")
        key = ("census_tracts", 101, _TILE.z, _TILE.x, _TILE.y)

        built = await pod.compute(key)
        assert built is not None

        restored = pod.deserialize(pod.serialize(built))
        assert restored["mvt"] == built["mvt"]
        decoded = mapbox_vector_tile.decode(restored["mvt"])
        assert len(decoded["census_tracts"]["features"]) == 3

    async def test_a_tile_reaches_l2_and_serves_a_second_pod(
        self,
        tmp_path: Path,
        nats_client: NatsClient,
    ) -> None:
        """the cross-pod payoff: pod A builds, pod B serves without building.

        pod B has its own empty L1, so a hit can only have come from the
        shared tiers.
        """
        calls: list[str] = []
        rows = _rows(3)
        pod_a = _make_pod(tmp_path, nats_client, name="a", rows=rows, calls=calls, isolation="crosspod")
        pod_b = _make_pod(tmp_path, nats_client, name="b", rows=rows, calls=calls, isolation="crosspod")
        key = ("census_tracts", 102, _TILE.z, _TILE.x, _TILE.y)

        first = await pod_a.get(key)
        assert first is not None
        assert calls == ["a"]

        second = await pod_b.get(key)
        assert second is not None
        assert calls == ["a"], f"pod b rebuilt the tile: {calls}"

    async def test_l1_hit_avoids_every_lower_tier(
        self,
        tmp_path: Path,
        nats_client: NatsClient,
    ) -> None:
        calls: list[str] = []
        pod = _make_pod(tmp_path, nats_client, name="a", rows=_rows(2), calls=calls, isolation="l1hit")
        key = ("census_tracts", 103, _TILE.z, _TILE.x, _TILE.y)

        await pod.get(key)
        await pod.get(key)
        await pod.get(key)
        assert calls == ["a"], f"source read {len(calls)} times for one tile"


class TestPayloadSize:
    async def test_a_dense_tile_stays_within_the_kv_ceiling(
        self,
        tmp_path: Path,
        nats_client: NatsClient,
    ) -> None:
        """the feature cap exists to keep tiles inside the L2 payload ceiling.

        an uncapped dense tile would exceed it, and the L2 write would fail
        at runtime on exactly the busiest tiles -- the ones most worth
        caching. this asserts the cap does its job on a genuinely dense
        input rather than trusting the arithmetic.
        """
        calls: list[str] = []
        pod = _make_pod(tmp_path, nats_client, name="dense", rows=_rows(2000), calls=calls, isolation="dense")
        built = await pod.compute(("census_tracts", 104, _TILE.z, _TILE.x, _TILE.y))
        assert built is not None
        # NATS max_payload here is 16MB; MVT guidance is to stay well under
        # 500KB for renderability. assert the far tighter of the two.
        assert len(built["mvt"]) < 500_000, f"tile is {len(built['mvt'])} bytes"
