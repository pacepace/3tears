"""unit tests for MVT projection and encoding.

the orientation test is the important one: a mirrored tile renders as
scrambled fragments rather than an obviously upside-down map, so it is easy
to ship and hard to diagnose. round-tripping through the decoder is the only
honest check that the bytes mean what we think.
"""

from __future__ import annotations

import mapbox_vector_tile
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon

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


class TestTheWireIsYDown:
    """the encoded tile's own coordinates, as a renderer reads them, put north at the top.

    The MVT spec's y runs downward. Decoding with the library's default flips y back, so a
    tile flipped on the way out round-trips cleanly through it and every test of the
    encoder agreed with itself while every map drew each tile mirrored. These read the
    wire as MapLibre does: ``y_coord_down=True`` leaves the coordinates as encoded.
    """

    def test_a_point_near_the_northern_edge_is_near_the_top(self) -> None:
        tile = TileId(z=6, x=17, y=24)
        bounds = tile_bounds(tile)
        mid_lon = (bounds.min_lon + bounds.max_lon) / 2
        north = bounds.max_lat - (bounds.max_lat - bounds.min_lat) * 0.05
        features = [TileFeature(geometry=Point(mid_lon, north), attributes={}, feature_id=1)]
        decoded = mapbox_vector_tile.decode(
            encode_tile({"pts": features}, tile), default_options={"y_coord_down": True}
        )
        _, y = decoded["pts"]["features"][0]["geometry"]["coordinates"]
        assert y < TILE_EXTENT * 0.2

    def test_the_encoded_y_is_the_projection_unchanged(self) -> None:
        tile = TileId(z=6, x=17, y=24)
        bounds = tile_bounds(tile)
        point = Point((bounds.min_lon + bounds.max_lon) / 2, bounds.min_lat + (bounds.max_lat - bounds.min_lat) * 0.3)
        expected = project_to_tile(point, tile)
        decoded = mapbox_vector_tile.decode(
            encode_tile({"pts": [TileFeature(geometry=point, attributes={}, feature_id=1)]}, tile),
            default_options={"y_coord_down": True},
        )
        _, y = decoded["pts"]["features"][0]["geometry"]["coordinates"]
        assert abs(y - expected.y) <= 1


#: the margin past the tile's edge a tile's geometry may reach (the MVT buffer)
BUFFER = 64


def _wire_coordinates(geometry: object) -> list[tuple[float, float]]:
    """every coordinate pair of a decoded geometry's coordinates, however nested."""
    if isinstance(geometry, (list, tuple)) and geometry and isinstance(geometry[0], (int, float)):
        return [(geometry[0], geometry[1])]
    found: list[tuple[float, float]] = []
    if isinstance(geometry, (list, tuple)):
        for part in geometry:
            found.extend(_wire_coordinates(part))
    return found


def _decode(payload: bytes) -> dict:
    return mapbox_vector_tile.decode(payload, default_options={"y_coord_down": True})


def _within_buffer(coordinates: list[tuple[float, float]]) -> bool:
    return all(-BUFFER <= c <= TILE_EXTENT + BUFFER for pair in coordinates for c in pair)


#: Aleutians West Census Area, in outline: a chain of islands across the
#: antimeridian, stored (as the census stores it) as one MultiPolygon whose parts
#: sit at both ends of the longitude range. Unclipped, the eastern islands land
#: about eight tile widths east of a western tile's edge, past the 16-bit range a
#: renderer reads tile coordinates in ("Geometry exceeds allowed extent").
def _aleutians() -> MultiPolygon:
    def island(lon: float) -> Polygon:
        return Polygon([(lon, 51.5), (lon + 0.8, 51.5), (lon + 0.8, 52.2), (lon, 52.2)])

    return MultiPolygon([island(lon) for lon in (-178.9, -176.5, -173.0, 172.5, 175.0, 178.6)])


class TestClippedToTheTileAndItsBuffer:
    """a tile carries each shape only as far as its own square and a small margin.

    Features are chosen by their bounds, so a shape far larger than the tile (a
    state at z10, a county that crosses the antimeridian at z3) arrives whole;
    projected unclipped, it runs tiles past the edge: heavy to ship and, past
    16 bits, refused by the renderer.
    """

    def test_a_shape_larger_than_the_tile_stays_within_the_buffer(self) -> None:
        tile = TileId(z=8, x=70, y=100)
        bounds = tile_bounds(tile)
        big = Polygon(
            [
                (bounds.min_lon - 5, bounds.min_lat - 5),
                (bounds.max_lon + 5, bounds.min_lat - 5),
                (bounds.max_lon + 5, bounds.max_lat + 5),
                (bounds.min_lon - 5, bounds.max_lat + 5),
            ]
        )
        decoded = _decode(encode_tile({"tracts": [TileFeature(geometry=big, attributes={}, feature_id=1)]}, tile))
        coordinates = _wire_coordinates(decoded["tracts"]["features"][0]["geometry"]["coordinates"])
        assert _within_buffer(coordinates)
        # still covers the whole tile: the corners of the clip box
        xs = [x for x, _ in coordinates]
        ys = [y for _, y in coordinates]
        assert min(xs) <= 0 and max(xs) >= TILE_EXTENT
        assert min(ys) <= 0 and max(ys) >= TILE_EXTENT

    def test_a_shape_across_the_antimeridian_keeps_only_its_near_side(self) -> None:
        aleutians = TileFeature(geometry=_aleutians(), attributes={"name": "Aleutians West"}, feature_id=2016)
        west = _decode(encode_tile({"counties": [aleutians]}, TileId(z=3, x=0, y=2)))
        east = _decode(encode_tile({"counties": [aleutians]}, TileId(z=3, x=7, y=2)))
        for decoded in (west, east):
            features = decoded["counties"]["features"]
            assert len(features) == 1
            coordinates = _wire_coordinates(features[0]["geometry"]["coordinates"])
            assert _within_buffer(coordinates)
        # the western tile holds the western islands at its western edge, the
        # eastern tile the eastern ones at its eastern edge
        west_xs = [x for x, _ in _wire_coordinates(west["counties"]["features"][0]["geometry"]["coordinates"])]
        east_xs = [x for x, _ in _wire_coordinates(east["counties"]["features"][0]["geometry"]["coordinates"])]
        assert max(west_xs) < TILE_EXTENT / 3
        assert min(east_xs) > TILE_EXTENT * 2 / 3

    def test_a_shape_whose_bounds_reach_the_tile_but_whose_parts_do_not_is_left_out(self) -> None:
        # the islands' bounds span the globe, so every tile in their row picks them
        # up; one in the middle of the Atlantic holds none of them
        aleutians = TileFeature(geometry=_aleutians(), attributes={}, feature_id=2016)
        decoded = _decode(encode_tile({"counties": [aleutians]}, TileId(z=3, x=3, y=2)))
        assert decoded.get("counties", {"features": []})["features"] == []
