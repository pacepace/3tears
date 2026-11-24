"""WKB/EWKB decoding and the bounds math a build needs.

one malformed row must not abort a tile build, let alone a whole precompute
pass over millions of tiles. but neither may it vanish: a geometry that cannot
be read raises :class:`UndecodableGeometry`, and the band builders skip that
row and COUNT it on the band's result (``BandResult.undecodable``), so a tile
missing features says how many, rather than reading as complete.

bounds are computed here as well as materialised into indexed source columns
at ingest. those serve different purposes and both are needed: the indexed
columns let the database return only the rows a tile might contain (there is
no spatial index without PostGIS, so an unindexed bbox query is a full scan
plus a Python-side test per tile), and this function is what computes them at
ingest and what re-checks a candidate row against the exact tile window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import shapely
from shapely import wkb as shapely_wkb
from shapely.errors import ShapelyError
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from threetears.geo.features import BOUNDS_COLUMNS
from threetears.geo.tiles import BoundingBox
from threetears.observe import get_logger

__all__ = [
    "RowGeometries",
    "UndecodableGeometry",
    "decode_geometry",
    "geometry_bounds",
    "point_bounds",
    "point_geometries",
    "point_geometry",
    "wkb_bounds",
    "wkb_geometries",
]

log = get_logger(__name__)


class UndecodableGeometry(ValueError):
    """a row's geometry is present but cannot be read.

    distinct from "no geometry" (``None``): the row HAS one and it is broken, so dropping it
    without a word would make the tile read as complete.
    """


def decode_geometry(raw: bytes | bytearray | memoryview | None) -> BaseGeometry | None:
    """decode WKB or EWKB bytes; ``None`` only when there is nothing to decode.

    EWKB (the PostGIS-flavoured variant carrying an SRID flag) is accepted
    directly -- shapely reads the SRID-tagged form without a separate column
    or a pre-strip step.

    :param raw: WKB/EWKB bytes, or ``None``
    :ptype raw: bytes | bytearray | memoryview | None
    :return: decoded geometry, or ``None`` if absent or empty
    :rtype: BaseGeometry | None
    :raises UndecodableGeometry: when the bytes are not valid WKB/EWKB
    """
    if not raw:
        return None
    data = bytes(raw)
    try:
        geometry = shapely_wkb.loads(data)
    except (ShapelyError, ValueError, TypeError) as exc:
        raise UndecodableGeometry(f"{len(data)} bytes are not valid WKB/EWKB: {exc}") from exc
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
    :return: a point, or ``None`` when either coordinate is absent
    :rtype: BaseGeometry | None
    :raises UndecodableGeometry: when a coordinate is present but is not a number
    """
    longitude = row.get(longitude_column)
    latitude = row.get(latitude_column)
    if longitude is None or latitude is None:
        return None
    try:
        return Point(float(longitude), float(latitude))
    except TypeError, ValueError:
        raise UndecodableGeometry(
            f"coordinates {longitude_column}={longitude!r}, {latitude_column}={latitude!r} are not numbers"
        ) from None


# ----------------------------------------------------------------------
# whole columns at once: what a build and a bounds read use
# ----------------------------------------------------------------------

#: a coordinate written as text that reads as a number, as ``float()`` reads one (after trimming): digits with an
#: optional point and exponent, or inf / infinity / nan in any case
_NUMERIC_TEXT: Final = r"(?i)^[+-]?((\d+\.?\d*|\.\d+)(e[+-]?\d+)?|inf|infinity|nan)$"


@dataclass(frozen=True, slots=True)
class RowGeometries:
    """every row's geometry, read from a whole table at once.

    :param shapes: one shapely geometry per row, ``None`` where the row has none (absent, empty, or unreadable)
    :ptype shapes: np.ndarray
    :param undecodable: per row, whether it HAS a geometry that could not be read (:class:`UndecodableGeometry`
        a row at a time): such a row is left out of a tile and counted, never passed off as absent
    :ptype undecodable: np.ndarray
    """

    shapes: np.ndarray
    undecodable: np.ndarray


def _none_of(rows: int) -> RowGeometries:
    """no geometry on any of ``rows`` rows, none unreadable."""
    return RowGeometries(shapes=np.full(rows, None, dtype=object), undecodable=np.zeros(rows, dtype=bool))


def wkb_geometries(table: pa.Table, column: str) -> RowGeometries:
    """decode a WKB/EWKB column whole, as :func:`decode_geometry` reads one value: absent bytes, empty bytes and an
    empty geometry are no geometry; bytes that do not decode are undecodable.

    shapely decodes the column in one call (its C loop); text is read as its bytes.

    :param table: source rows
    :ptype table: pa.Table
    :param column: the geometry column; a table without it has no geometry on any row
    :ptype column: str
    :return: every row's geometry
    :rtype: RowGeometries
    """
    if column not in table.column_names:
        return _none_of(table.num_rows)
    raw = table.column(column)
    if pa.types.is_string(raw.type) or pa.types.is_large_string(raw.type):
        raw = pc.cast(raw, pa.binary())
    if not (pa.types.is_binary(raw.type) or pa.types.is_large_binary(raw.type)):
        # a value of any other type is not WKB: present and unreadable
        return RowGeometries(
            shapes=np.full(table.num_rows, None, dtype=object),
            undecodable=pc.is_valid(raw).to_numpy(zero_copy_only=False),
        )
    present = pc.fill_null(pc.greater(pc.binary_length(raw), 0), False).to_numpy(zero_copy_only=False)
    shapes = shapely.from_wkb(raw.to_numpy(zero_copy_only=False), on_invalid="ignore")
    undecodable = present & shapely.is_missing(shapes)
    shapes[shapely.is_empty(shapes)] = None
    return RowGeometries(shapes=shapes, undecodable=undecodable)


#: a coordinate's state on a row (:func:`_coordinates`): none, present but not a number, a number
_ABSENT: Final = 0
_UNREADABLE: Final = 1
_NUMBER: Final = 2


def _coordinates(table: pa.Table, column: str) -> tuple[np.ndarray, np.ndarray]:
    """a coordinate column whole: each row's value as float64, and its state (:data:`_ABSENT`,
    :data:`_UNREADABLE`, :data:`_NUMBER`).

    a value reads as ``float()`` reads it -- a number of any kind (bool and decimal included), or text that spells
    one; any other present value is unreadable.

    :param table: source rows
    :ptype table: pa.Table
    :param column: the coordinate column; a table without it has none on any row
    :ptype column: str
    :return: ``(values, state)``: float64, NaN where not a number; int8 states
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    rows = table.num_rows
    if column not in table.column_names:
        return np.full(rows, np.nan), np.full(rows, _ABSENT, dtype=np.int8)
    values = table.column(column)
    present = pc.is_valid(values).to_numpy(zero_copy_only=False)
    kind = values.type
    if pa.types.is_string(kind) or pa.types.is_large_string(kind):
        trimmed = pc.utf8_trim_whitespace(values)
        numeric = pc.match_substring_regex(trimmed, _NUMERIC_TEXT)
        values = pc.cast(pc.if_else(numeric, trimmed, pa.scalar(None, trimmed.type)), pa.float64())
    elif (
        pa.types.is_integer(kind)
        or pa.types.is_floating(kind)
        or pa.types.is_decimal(kind)
        or pa.types.is_boolean(kind)
        or pa.types.is_null(kind)
    ):
        values = pc.cast(values, pa.float64(), safe=False)
    else:
        return np.full(rows, np.nan), np.where(present, _UNREADABLE, _ABSENT).astype(np.int8)
    number = pc.is_valid(values).to_numpy(zero_copy_only=False)
    floats = pc.fill_null(values, np.nan).to_numpy(zero_copy_only=False).astype(np.float64)
    state = np.where(number, _NUMBER, np.where(present, _UNREADABLE, _ABSENT)).astype(np.int8)
    return floats, state


def point_geometries(table: pa.Table, longitude_column: str, latitude_column: str) -> RowGeometries:
    """points from a lon/lat column pair, whole, as :func:`point_geometry` reads one row: a row missing either
    coordinate has none; a row with both, either not a number, is undecodable.

    :param table: source rows
    :ptype table: pa.Table
    :param longitude_column: degrees east
    :ptype longitude_column: str
    :param latitude_column: degrees north
    :ptype latitude_column: str
    :return: every row's point
    :rtype: RowGeometries
    """
    lon, lon_state = _coordinates(table, longitude_column)
    lat, lat_state = _coordinates(table, latitude_column)
    both = (lon_state != _ABSENT) & (lat_state != _ABSENT)
    readable = (lon_state == _NUMBER) & (lat_state == _NUMBER)
    drawn = both & readable
    shapes = np.full(table.num_rows, None, dtype=object)
    shapes[drawn] = shapely.points(lon[drawn], lat[drawn])
    return RowGeometries(shapes=shapes, undecodable=both & ~readable)


def _bounds_table(shapes: np.ndarray) -> pa.Table:
    """the four :data:`BOUNDS_COLUMNS` of ``shapes``, null where a shape is ``None``."""
    found = shapely.bounds(shapes).astype(np.float64)
    return pa.table({name: pa.array(found[:, index], from_pandas=True) for index, name in enumerate(BOUNDS_COLUMNS)})


def wkb_bounds(table: pa.Table, column: str) -> pa.Table:
    """every row's bounding rectangle from a WKB/EWKB column, whole: a
    :data:`~threetears.geo.features.BoundsOf` for a polygon-shaped layer.

    a row whose geometry is absent, empty or will not decode answers null bounds, which intersect no tile, so the
    row is never drawn rather than failing a whole build.

    :param table: source rows
    :ptype table: pa.Table
    :param column: the geometry column
    :ptype column: str
    :return: the four float64 :data:`~threetears.geo.features.BOUNDS_COLUMNS`, one row per row
    :rtype: pa.Table
    """
    return _bounds_table(wkb_geometries(table, column).shapes)


def point_bounds(table: pa.Table, longitude_column: str, latitude_column: str) -> pa.Table:
    """every row's bounds from a lon/lat column pair, whole: a :data:`~threetears.geo.features.BoundsOf` for a
    point-shaped layer. a row without both coordinates as finite numbers answers null bounds.

    :param table: source rows
    :ptype table: pa.Table
    :param longitude_column: degrees east
    :ptype longitude_column: str
    :param latitude_column: degrees north
    :ptype latitude_column: str
    :return: the four float64 :data:`~threetears.geo.features.BOUNDS_COLUMNS`, one row per row
    :rtype: pa.Table
    """
    lon, lon_state = _coordinates(table, longitude_column)
    lat, lat_state = _coordinates(table, latitude_column)
    usable = (lon_state == _NUMBER) & (lat_state == _NUMBER) & np.isfinite(lon) & np.isfinite(lat)
    lon_column = pa.array(lon, mask=~usable)
    lat_column = pa.array(lat, mask=~usable)
    return pa.table(dict(zip(BOUNDS_COLUMNS, (lon_column, lat_column, lon_column, lat_column), strict=True)))
