"""WKB/EWKB decoding and the bounds math a build needs.

decoding is deliberately forgiving: one malformed row must not abort a tile
build, let alone a whole precompute pass over millions of tiles. a geometry
that cannot be read is logged and skipped, which loses one feature rather
than the map.

bounds are computed here as well as materialised into indexed source columns
at ingest. those serve different purposes and both are needed: the indexed
columns let the database return only the rows a tile might contain (there is
no spatial index without PostGIS, so an unindexed bbox query is a full scan
plus a Python-side test per tile), and this function is what computes them at
ingest and what re-checks a candidate row against the exact tile window.
"""

from __future__ import annotations

from typing import Any

from shapely import wkb as shapely_wkb
from shapely.geometry.base import BaseGeometry

from threetears.geo.tiles import BoundingBox
from threetears.observe import get_logger

__all__ = [
    "decode_geometry",
    "geometry_bounds",
    "point_geometry",
]

log = get_logger(__name__)


def decode_geometry(raw: bytes | bytearray | memoryview | None) -> BaseGeometry | None:
    """decode WKB or EWKB bytes, returning ``None`` when unreadable.

    EWKB (the PostGIS-flavoured variant carrying an SRID flag) is accepted
    directly -- shapely reads the SRID-tagged form without a separate column
    or a pre-strip step.

    :param raw: WKB/EWKB bytes, or ``None``
    :ptype raw: bytes | bytearray | memoryview | None
    :return: decoded geometry, or ``None`` if absent, empty or undecodable
    :rtype: BaseGeometry | None
    """
    if not raw:
        return None
    try:
        geometry = shapely_wkb.loads(bytes(raw))
    except Exception:
        # one bad row must not abort a build; skipping loses a feature,
        # raising loses the tile.
        log.warning("skipping undecodable geometry (%d bytes)", len(bytes(raw)))
        return None
    if geometry.is_empty:
        return None
    return geometry


def geometry_bounds(geometry: BaseGeometry | None) -> BoundingBox | None:
    """return a geometry's bounding rectangle, or ``None`` when it has none.

    :param geometry: decoded geometry
    :ptype geometry: BaseGeometry | None
    :return: bounds, or ``None`` for absent/empty geometry
    :rtype: BoundingBox | None
    """
    if geometry is None or geometry.is_empty:
        return None
    min_lon, min_lat, max_lon, max_lat = geometry.bounds
    return BoundingBox(
        min_lon=float(min_lon),
        min_lat=float(min_lat),
        max_lon=float(max_lon),
        max_lat=float(max_lat),
    )


def point_geometry(row: dict[str, Any], longitude_column: str, latitude_column: str) -> BaseGeometry | None:
    """build a point geometry from a lon/lat column pair.

    the other declared geometry shape besides a WKB blob. handled here so a
    build treats point and polygon layers identically from this call onward.

    :param row: source row
    :ptype row: dict[str, Any]
    :param longitude_column: column holding degrees east
    :ptype longitude_column: str
    :param latitude_column: column holding degrees north
    :ptype latitude_column: str
    :return: a point, or ``None`` when either coordinate is absent or unparseable
    :rtype: BaseGeometry | None
    """
    from shapely.geometry import Point

    longitude = row.get(longitude_column)
    latitude = row.get(latitude_column)
    if longitude is None or latitude is None:
        return None
    try:
        return Point(float(longitude), float(latitude))
    except TypeError, ValueError:
        log.warning("skipping row with unparseable coordinates: %r/%r", longitude, latitude)
        return None
