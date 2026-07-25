"""unit tests for MVT projection and encoding.

the orientation test is the important one: a mirrored tile renders as
scrambled fragments rather than an obviously upside-down map, so it is easy
to ship and hard to diagnose. round-tripping through the decoder is the only
honest check that the bytes mean what we think.
"""

from __future__ import annotations

import mapbox_vector_tile
import pytest
from shapely.geometry import Point, Polygon

from threetears.geo.bands import TileFeature
from threetears.geo.mvt import encode_tile, project_to_tile
from threetears.geo.tiles import TILE_EXTENT, TileId, tile_bounds


class TestProjection:
    def test_tile_corners_map_to_the_extent_corners(self) -> None:
        tile = TileId(z=6, x=12, y=25)
        bounds = tile_bounds(tile)
        north_west = project_to_tile(Point(bounds.min_lon, bounds.max_lat), tile)
        south_east = project_to_tile(Point(bounds.max_lon, bounds.min_lat), tile)
        assert north_west.x == pytest.approx(0.0, abs=1e-6)
        assert north_west.y == pytest.approx(0.0, abs=1e-6)
        assert south_east.x == pytest.approx(TILE_EXTENT, abs=1e-6)
        assert south_east.y == pytest.approx(TILE_EXTENT, abs=1e-6)

    def test_local_y_increases_southward(self) -> None:
        # tile-local coordinates run downward from the top-left while latitude
        # runs northward. inverting this mirrors every tile about its own
        # horizontal axis.
        tile = TileId(z=6, x=12, y=25)
        bounds = tile_bounds(tile)
        mid_lon = (bounds.min_lon + bounds.max_lon) / 2
        northern = project_to_tile(Point(mid_lon, bounds.max_lat - 0.01), tile)
        southern = project_to_tile(Point(mid_lon, bounds.min_lat + 0.01), tile)
        assert northern.y < southern.y

    def test_local_x_increases_eastward(self) -> None:
        tile = TileId(z=6, x=12, y=25)
        bounds = tile_bounds(tile)
        mid_lat = (bounds.min_lat + bounds.max_lat) / 2
        west = project_to_tile(Point(bounds.min_lon + 0.01, mid_lat), tile)
        east = project_to_tile(Point(bounds.max_lon - 0.01, mid_lat), tile)
        assert west.x < east.x

    def test_latitude_is_projected_not_linearly_interpolated(self) -> None:
        # latitude is non-linear in Mercator; a linear map would put the
        # midpoint latitude exactly at the tile's vertical centre.
        tile = TileId(z=1, x=0, y=0)
        bounds = tile_bounds(tile)
        mid_lat = (bounds.min_lat + bounds.max_lat) / 2
        projected = project_to_tile(Point(0.0, mid_lat), tile)
        assert projected.y != pytest.approx(TILE_EXTENT / 2, abs=1.0)


class TestEncoding:
    def test_round_trips_through_the_decoder(self) -> None:
        tile = TileId(z=6, x=12, y=25)
        bounds = tile_bounds(tile)
        mid_lon = (bounds.min_lon + bounds.max_lon) / 2
        mid_lat = (bounds.min_lat + bounds.max_lat) / 2
        features = [
            TileFeature(
                geometry=Point(mid_lon, mid_lat),
                attributes={"name": "Site A", "score": 80},
                feature_id=7,
            )
        ]
        decoded = mapbox_vector_tile.decode(encode_tile({"locations": features}, tile))
        assert "locations" in decoded
        properties = decoded["locations"]["features"][0]["properties"]
        assert properties["name"] == "Site A"
        assert properties["score"] == 80

    def test_feature_id_is_promoted(self) -> None:
        # MapLibre binds volatile values to static geometry via the feature
        # id; without it the election-night join has nothing to key on.
        tile = TileId(z=6, x=12, y=25)
        bounds = tile_bounds(tile)
        features = [
            TileFeature(
                geometry=Point((bounds.min_lon + bounds.max_lon) / 2, (bounds.min_lat + bounds.max_lat) / 2),
                attributes={},
                feature_id=42,
            )
        ]
        decoded = mapbox_vector_tile.decode(encode_tile({"tracts": features}, tile))
        assert decoded["tracts"]["features"][0]["id"] == 42

    def test_multiple_layers_share_one_tile(self) -> None:
        tile = TileId(z=6, x=12, y=25)
        bounds = tile_bounds(tile)
        centre = Point((bounds.min_lon + bounds.max_lon) / 2, (bounds.min_lat + bounds.max_lat) / 2)
        decoded = mapbox_vector_tile.decode(
            encode_tile(
                {
                    "locations": [TileFeature(geometry=centre, attributes={"a": 1})],
                    "tracts": [TileFeature(geometry=centre, attributes={"b": 2})],
                },
                tile,
            )
        )
        assert set(decoded) == {"locations", "tracts"}

    def test_polygon_survives_encoding(self) -> None:
        tile = TileId(z=6, x=12, y=25)
        bounds = tile_bounds(tile)
        lon_step = (bounds.max_lon - bounds.min_lon) / 4
        lat_step = (bounds.max_lat - bounds.min_lat) / 4
        polygon = Polygon(
            [
                (bounds.min_lon + lon_step, bounds.min_lat + lat_step),
                (bounds.min_lon + 3 * lon_step, bounds.min_lat + lat_step),
                (bounds.min_lon + 3 * lon_step, bounds.min_lat + 3 * lat_step),
                (bounds.min_lon + lon_step, bounds.min_lat + 3 * lat_step),
            ]
        )
        decoded = mapbox_vector_tile.decode(
            encode_tile({"tracts": [TileFeature(geometry=polygon, attributes={"geoid": "04013"})]}, tile)
        )
        feature = decoded["tracts"]["features"][0]
        assert feature["geometry"]["type"] == "Polygon"
        assert feature["properties"]["geoid"] == "04013"

    def test_empty_layer_encodes_without_error(self) -> None:
        # a tile covering ocean is a legitimate, cacheable empty result, not
        # a failure to build.
        assert isinstance(encode_tile({"locations": []}, TileId(z=6, x=12, y=25)), bytes)
