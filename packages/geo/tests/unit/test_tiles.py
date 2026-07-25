"""unit tests for slippy-map tile addressing.

the axis-orientation tests are the load-bearing ones: XYZ and TMS differ only
in the direction of ``y``, and confusing them renders a vertically mirrored
map that looks plausible at a glance.
"""

from __future__ import annotations

import pytest

from threetears.geo.tiles import (
    MAX_MERCATOR_LATITUDE,
    BoundingBox,
    TileId,
    bounds_to_tile_range,
    tile_bounds,
    tile_for_point,
)


class TestTileId:
    def test_key_is_the_collection_primary_key_tuple(self) -> None:
        assert TileId(z=4, x=3, y=6).key == (4, 3, 6)

    def test_rejects_indices_outside_the_grid(self) -> None:
        # zoom 2 is a 4x4 grid, so index 4 does not exist.
        with pytest.raises(ValueError, match="outside the 4x4 grid"):
            TileId(z=2, x=4, y=0)

    def test_rejects_negative_zoom(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            TileId(z=-1, x=0, y=0)


class TestOrientation:
    """y increases southward. this is XYZ, not TMS."""

    def test_northern_hemisphere_is_the_top_row_at_zoom_one(self) -> None:
        # zoom 1 splits the world into four; a northern point must land in
        # row 0, a southern point in row 1. under TMS this is reversed.
        northern = tile_for_point(longitude=-70.0, latitude=45.0, zoom=1)
        southern = tile_for_point(longitude=-70.0, latitude=-45.0, zoom=1)
        assert northern.y == 0
        assert southern.y == 1

    def test_tile_row_zero_bounds_the_north(self) -> None:
        top = tile_bounds(TileId(z=1, x=0, y=0))
        bottom = tile_bounds(TileId(z=1, x=0, y=1))
        assert top.max_lat == pytest.approx(MAX_MERCATOR_LATITUDE)
        assert top.min_lat == pytest.approx(0.0, abs=1e-9)
        assert bottom.max_lat == pytest.approx(0.0, abs=1e-9)
        assert bottom.min_lat == pytest.approx(-MAX_MERCATOR_LATITUDE)

    def test_x_increases_eastward(self) -> None:
        west = tile_for_point(longitude=-170.0, latitude=0.0, zoom=2)
        east = tile_for_point(longitude=170.0, latitude=0.0, zoom=2)
        assert west.x < east.x


class TestRoundTrip:
    @pytest.mark.parametrize("zoom", [0, 1, 4, 10, 14])
    def test_point_lands_inside_the_bounds_of_its_own_tile(self, zoom: int) -> None:
        longitude, latitude = -112.07, 33.45
        tile = tile_for_point(longitude, latitude, zoom)
        bounds = tile_bounds(tile)
        assert bounds.min_lon <= longitude <= bounds.max_lon
        assert bounds.min_lat <= latitude <= bounds.max_lat

    def test_zoom_zero_is_the_whole_world(self) -> None:
        bounds = tile_bounds(TileId(z=0, x=0, y=0))
        assert bounds.min_lon == pytest.approx(-180.0)
        assert bounds.max_lon == pytest.approx(180.0)
        assert bounds.max_lat == pytest.approx(MAX_MERCATOR_LATITUDE)


class TestMercatorClamp:
    def test_polar_latitudes_clamp_rather_than_diverge(self) -> None:
        # Mercator cannot represent the poles; without the clamp this is a
        # math domain error rather than a tile.
        assert tile_for_point(0.0, 89.9, zoom=3).y == 0
        assert tile_for_point(0.0, -89.9, zoom=3).y == 7

    def test_eastern_and_southern_edges_stay_in_range(self) -> None:
        # exactly on the edge computes to index n, one past the last tile.
        assert tile_for_point(180.0, -MAX_MERCATOR_LATITUDE, zoom=3).key == (3, 7, 7)


class TestBoundingBox:
    def test_overlapping_rectangles_intersect(self) -> None:
        a = BoundingBox(-112.2, 33.3, -112.0, 33.5)
        b = BoundingBox(-112.1, 33.4, -111.9, 33.6)
        assert a.intersects(b)
        assert b.intersects(a)

    def test_disjoint_rectangles_do_not(self) -> None:
        a = BoundingBox(-112.2, 33.3, -112.1, 33.4)
        b = BoundingBox(-100.0, 40.0, -99.0, 41.0)
        assert not a.intersects(b)

    def test_edge_contact_counts_as_intersection(self) -> None:
        # inclusive on the edges, matching the SQL predicate a build issues
        # against the materialised bbox columns.
        a = BoundingBox(-112.2, 33.3, -112.0, 33.5)
        b = BoundingBox(-112.0, 33.5, -111.0, 34.0)
        assert a.intersects(b)


class TestTileRange:
    def test_range_covers_the_requested_region(self) -> None:
        bounds = BoundingBox(min_lon=-113.0, min_lat=33.0, max_lon=-111.0, max_lat=34.0)
        min_x, min_y, max_x, max_y = bounds_to_tile_range(bounds, zoom=8)
        assert min_x <= max_x
        # y counts southward, so the northern edge yields the SMALLER row.
        assert min_y <= max_y
        corner = tile_for_point(bounds.min_lon, bounds.max_lat, 8)
        assert (corner.x, corner.y) == (min_x, min_y)

    def test_whole_world_at_zoom_two_is_the_full_grid(self) -> None:
        bounds = BoundingBox(-180.0, -MAX_MERCATOR_LATITUDE, 180.0, MAX_MERCATOR_LATITUDE)
        assert bounds_to_tile_range(bounds, zoom=2) == (0, 0, 3, 3)
