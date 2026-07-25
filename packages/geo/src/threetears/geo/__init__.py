"""3tears-geo: slippy-map tile geometry in application code.

no PostGIS is available on YugabyteDB, so the work ``ST_Intersects`` /
``ST_Simplify`` / ``ST_AsMVT`` would do in SQL happens here instead --
shapely for the geometry, mapbox-vector-tile for the encoding.
"""

from threetears.geo.attributes import (
    MVT_SCALAR_TYPES,
    UnsupportedAttributeError,
    coerce_attribute,
    coerce_attributes,
    validate_attribute_value,
)
from threetears.geo.tiles import (
    MAX_MERCATOR_LATITUDE,
    TILE_EXTENT,
    BoundingBox,
    TileId,
    bounds_to_tile_range,
    tile_bounds,
    tile_for_point,
)

__all__ = [
    "MAX_MERCATOR_LATITUDE",
    "MVT_SCALAR_TYPES",
    "TILE_EXTENT",
    "BoundingBox",
    "TileId",
    "UnsupportedAttributeError",
    "bounds_to_tile_range",
    "coerce_attribute",
    "coerce_attributes",
    "tile_bounds",
    "tile_for_point",
    "validate_attribute_value",
]
