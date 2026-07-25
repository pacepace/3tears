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
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from threetears.geo.bands import TileFeature
from threetears.geo.tiles import TILE_EXTENT, MAX_MERCATOR_LATITUDE, TileId, tile_bounds
from threetears.observe import get_logger

__all__ = ["encode_tile", "project_to_tile"]

log = get_logger(__name__)


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
            projected = project_to_tile(feature.geometry, tile)
            if projected.is_empty:
                continue
            entry: dict[str, Any] = {
                "geometry": projected,
                "properties": feature.attributes,
            }
            if feature.feature_id is not None:
                # promoted to the MVT feature id so MapLibre's feature-state
                # can bind volatile values to this static geometry.
                entry["id"] = feature.feature_id
            encoded_features.append(entry)
        encoded_layers.append({"name": name, "features": encoded_features})

    # ``default_options`` rather than the legacy ``extents=`` kwarg, which the
    # library deprecated in 2.x.
    result: bytes = mapbox_vector_tile.encode(encoded_layers, default_options={"extents": TILE_EXTENT})
    return result
