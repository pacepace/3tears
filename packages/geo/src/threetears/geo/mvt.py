"""MVT encoding: the job ``ST_AsMVT`` would do, done in Python.

geometry arrives in WGS84 and must leave in tile-local integer coordinates
over the 4096-unit extent, with the tile's own bounds as the frame. that
projection is the encoder's real work; ``mapbox_vector_tile`` handles the
protobuf once the coordinates are in the right space.

the y flip lives here. tile-local coordinates increase *downward* from the
top-left, matching the XYZ addressing in :mod:`threetears.geo.tiles`, while
latitude increases northward. getting this wrong produces a tile that renders
mirrored about its own horizontal axis -- and because each tile is
individually mirrored, the result looks like scrambled fragments rather than
an obviously upside-down map, which makes it harder to diagnose than it
sounds.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import mapbox_vector_tile
from shapely import clip_by_rect
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from threetears.geo.bands import TileFeature
from threetears.geo.tiles import TILE_EXTENT, MAX_MERCATOR_LATITUDE, TileId, tile_bounds
from threetears.observe import get_logger

__all__ = ["TILE_BUFFER", "clip_to_tile", "encode_tile", "project_to_tile"]

log = get_logger(__name__)

#: how far past its own square, in tile units, a tile carries geometry. Features are
#: picked by their bounds, so a shape much larger than the tile -- a state at a high
#: zoom, or a county whose islands sit at both ends of the longitude range (Aleutians
#: West, across the antimeridian) -- would otherwise arrive whole, running tiles past
#: the edge: heavy to ship and, past the 16-bit range renderers read tile coordinates
#: in, refused ("Geometry exceeds allowed extent, reduce your vector tile buffer
#: size"). The margin keeps strokes and fills seamless where tiles meet; 64 of 4096
#: is the usual choice for data layers.
TILE_BUFFER: int = 64


def _mercator_y(latitude: float) -> float:
    """project latitude to the Web Mercator unit square (0 at north pole)."""
    clamped = max(-MAX_MERCATOR_LATITUDE, min(MAX_MERCATOR_LATITUDE, latitude))
    return (1.0 - math.asinh(math.tan(math.radians(clamped))) / math.pi) / 2.0


def project_to_tile(geometry: BaseGeometry, tile: TileId) -> BaseGeometry:
    """project WGS84 geometry into ``tile``'s local 4096-unit coordinate space.

    :param geometry: geometry in WGS84 degrees
    :ptype geometry: BaseGeometry
    :param tile: the tile providing the coordinate frame
    :ptype tile: TileId
    :return: geometry in tile-local integer-ranged coordinates
    :rtype: BaseGeometry
    """
    bounds = tile_bounds(tile)
    lon_span = bounds.max_lon - bounds.min_lon
    # y is projected through Mercator rather than linearly interpolated in
    # latitude: latitude is not linear in the projection, so a linear map
    # would skew geometry increasingly toward the tile's edges.
    top = _mercator_y(bounds.max_lat)
    bottom = _mercator_y(bounds.min_lat)
    y_span = bottom - top

    def _project(x: Any, y: Any, z: Any = None) -> tuple[float, float]:
        local_x = (x - bounds.min_lon) / lon_span * TILE_EXTENT
        # (mercator_y - top) already increases southward, so no extra flip:
        # the downward direction comes from the projection itself.
        local_y = (_mercator_y(y) - top) / y_span * TILE_EXTENT
        return (local_x, local_y)

    return transform(_project, geometry)


def clip_to_tile(projected: BaseGeometry, buffer: int = TILE_BUFFER) -> BaseGeometry:
    """cut tile-local geometry to the tile's square and a ``buffer`` around it.

    a shape across the antimeridian is stored with parts at both ends of the
    longitude range; projected into a tile at one end, the parts at the other land
    far outside it, so the cut leaves each tile only its own side's parts.

    :param projected: geometry already in the tile's local coordinates
    :ptype projected: BaseGeometry
    :param buffer: the margin kept past each edge, in tile units
    :ptype buffer: int
    :return: the part within the square and its margin; empty when nothing is
    :rtype: BaseGeometry
    """
    low = -float(buffer)
    high = float(TILE_EXTENT + buffer)
    minx, miny, maxx, maxy = projected.bounds
    if minx >= low and miny >= low and maxx <= high and maxy <= high:
        return projected
    clipped: BaseGeometry = clip_by_rect(projected, low, low, high, high)
    return clipped


def encode_tile(layers: dict[str, Sequence[TileFeature]], tile: TileId) -> bytes:
    """encode one or more named layers into a single MVT tile.

    a tile carrying several layers is the format's own design and is why the
    static geometry / volatile attribute split works: boundaries and any other
    static layer ship together, addressed once and cached once.

    :param layers: layer name to its features, in WGS84
    :ptype layers: dict[str, Sequence[TileFeature]]
    :param tile: the tile being encoded
    :ptype tile: TileId
    :return: encoded MVT bytes
    :rtype: bytes
    """
    encoded_layers: list[dict[str, Any]] = []
    for name, features in layers.items():
        encoded_features: list[dict[str, Any]] = []
        for feature in features:
            projected = clip_to_tile(project_to_tile(feature.geometry, tile))
            if projected.is_empty:
                continue
            entry: dict[str, Any] = {
                "geometry": projected,
                "properties": feature.attributes,
            }
            # MVT feature ids are uint64 by specification. a non-integer id --
            # a census geoid, a UUID -- is silently coerced to 0 by the
            # encoder, which would collapse every feature in a tile onto one
            # id and break any client-side join keyed on it. so only integer
            # ids become the wire-level id; everything else travels as a
            # property, which is exactly what MapLibre's ``promoteId`` is for.
            # the id is always present in properties either way, so a client
            # has one consistent place to look.
            if isinstance(feature.feature_id, int) and not isinstance(feature.feature_id, bool):
                entry["id"] = feature.feature_id
            encoded_features.append(entry)
        encoded_layers.append({"name": name, "features": encoded_features})

    # ``default_options`` rather than the legacy ``extents=`` kwarg, which the
    # library deprecated in 2.x. ``y_coord_down``: the projection above already puts y
    # downward, as the format wants; left at its default the library flips it a second time,
    # and every tile renders mirrored about its own middle. Its decoder flips back by default,
    # so a round trip through it cannot see this -- tests read the wire y-down
    result: bytes = mapbox_vector_tile.encode(
        encoded_layers, default_options={"extents": TILE_EXTENT, "y_coord_down": True}
    )
    return result
