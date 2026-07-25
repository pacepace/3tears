"""slippy-map tile addressing and the geometry projection that goes with it.

the tiling scheme is stated explicitly rather than left implicit, because the
two common conventions differ only in the direction of one axis and silently
render an upside-down map when confused:

- **Web Mercator (EPSG:3857)**, the projection every consumer web map uses.
- **XYZ orientation**: ``y`` increases *southward* from the top-left origin.
  this is the OpenStreetMap / MapLibre / Google convention. it is NOT TMS,
  whose ``y`` increases northward.
- source coordinates are **WGS84 (EPSG:4326)** longitude/latitude and are
  projected here, at build time.
- MVT geometry is expressed in tile-local integer coordinates over a
  4096-unit extent, the format default.

Mercator cannot represent the poles, so latitude is clamped to
±85.0511287798066° -- the value whose projection is exactly square with the
longitude range, and therefore the implicit vertical bound of every tile
pyramid in this scheme.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

__all__ = [
    "MAX_MERCATOR_LATITUDE",
    "TILE_EXTENT",
    "BoundingBox",
    "TileId",
    "bounds_to_tile_range",
    "tile_bounds",
    "tile_for_point",
]

#: the latitude bound of the Web Mercator square. beyond this the projection
#: diverges, so every tile pyramid in this scheme is implicitly clipped here.
MAX_MERCATOR_LATITUDE: Final = 85.0511287798066

#: MVT tile-local coordinate extent. 4096 is the format default and is not
#: varied per layer -- a non-default extent has to be communicated to every
#: consumer, and there is no benefit here worth that coupling.
TILE_EXTENT: Final = 4096


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """a WGS84 longitude/latitude rectangle.

    :param min_lon: western edge, degrees
    :ptype min_lon: float
    :param min_lat: southern edge, degrees
    :ptype min_lat: float
    :param max_lon: eastern edge, degrees
    :ptype max_lon: float
    :param max_lat: northern edge, degrees
    :ptype max_lat: float
    """

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def intersects(self, other: BoundingBox) -> bool:
        """true when the two rectangles overlap, edges inclusive.

        the predicate a tile build issues against materialised bbox columns,
        expressed here so the SQL and the in-process filter cannot disagree.

        :param other: rectangle to test against
        :ptype other: BoundingBox
        :return: whether the rectangles overlap
        :rtype: bool
        """
        return (
            self.min_lon <= other.max_lon
            and self.max_lon >= other.min_lon
            and self.min_lat <= other.max_lat
            and self.max_lat >= other.min_lat
        )


@dataclass(frozen=True, slots=True)
class TileId:
    """one tile address in XYZ orientation.

    :param z: zoom level
    :ptype z: int
    :param x: column, increasing eastward
    :ptype x: int
    :param y: row, increasing **southward** from the top-left origin
    :ptype y: int
    """

    z: int
    x: int
    y: int

    def __post_init__(self) -> None:
        if self.z < 0:
            raise ValueError(f"zoom must be non-negative, got {self.z}")
        limit = 1 << self.z
        if not (0 <= self.x < limit and 0 <= self.y < limit):
            raise ValueError(f"tile ({self.z}/{self.x}/{self.y}) is outside the {limit}x{limit} grid at zoom {self.z}")

    @property
    def key(self) -> tuple[int, int, int]:
        """``(z, x, y)`` tuple, for use as a collection primary key.

        :return: the tile's key tuple
        :rtype: tuple[int, int, int]
        """
        return (self.z, self.x, self.y)

    def __str__(self) -> str:
        return f"{self.z}/{self.x}/{self.y}"


def tile_for_point(longitude: float, latitude: float, zoom: int) -> TileId:
    """return the tile containing a WGS84 point at ``zoom``.

    :param longitude: degrees east
    :ptype longitude: float
    :param latitude: degrees north; clamped to the Mercator bound
    :ptype latitude: float
    :param zoom: zoom level
    :ptype zoom: int
    :return: the containing tile
    :rtype: TileId
    """
    n = 1 << zoom
    clamped_lat = max(-MAX_MERCATOR_LATITUDE, min(MAX_MERCATOR_LATITUDE, latitude))
    lat_rad = math.radians(clamped_lat)
    x = int((longitude + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    # a point exactly on the eastern or southern edge of the grid computes to
    # index n, which is one past the last tile.
    return TileId(z=zoom, x=min(x, n - 1), y=min(y, n - 1))


def tile_bounds(tile: TileId) -> BoundingBox:
    """return the WGS84 rectangle a tile covers.

    the inverse of :func:`tile_for_point`, and the query window a build uses
    to select source features.

    :param tile: tile address
    :ptype tile: TileId
    :return: the tile's geographic bounds
    :rtype: BoundingBox
    """
    n = 1 << tile.z
    min_lon = tile.x / n * 360.0 - 180.0
    max_lon = (tile.x + 1) / n * 360.0 - 180.0
    # y increases southward, so row y is the NORTHERN edge and y+1 the
    # southern one -- the sign flip that distinguishes XYZ from TMS.
    max_lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * tile.y / n))))
    min_lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (tile.y + 1) / n))))
    return BoundingBox(min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat)


def bounds_to_tile_range(bounds: BoundingBox, zoom: int) -> tuple[int, int, int, int]:
    """return ``(min_x, min_y, max_x, max_y)`` covering ``bounds`` at ``zoom``.

    inclusive on every edge. used by the precompute pass to enumerate the
    tiles a region occupies without walking the whole pyramid.

    :param bounds: geographic rectangle
    :ptype bounds: BoundingBox
    :param zoom: zoom level
    :ptype zoom: int
    :return: inclusive tile index range
    :rtype: tuple[int, int, int, int]
    """
    top_left = tile_for_point(bounds.min_lon, bounds.max_lat, zoom)
    bottom_right = tile_for_point(bounds.max_lon, bounds.min_lat, zoom)
    return (top_left.x, top_left.y, bottom_right.x, bottom_right.y)
