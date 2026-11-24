"""the shape of a row-bounds read: what a :class:`~threetears.geo.features.FeatureCache` asks of a layer's rows.

its own module so the geometry readers, the feature cache and the worker jobs all name it without importing each
other.
"""

from __future__ import annotations

from collections.abc import Callable

import pyarrow as pa

__all__ = ["BOUNDS_COLUMNS", "BoundsOf"]

#: the columns a :data:`BoundsOf` answers, in :class:`~threetears.geo.tiles.BoundingBox` field order
BOUNDS_COLUMNS: tuple[str, str, str, str] = ("min_lon", "min_lat", "max_lon", "max_lat")

#: every row's bounding rectangle, for a whole table at once: a table of the four float
#: :data:`BOUNDS_COLUMNS`, one row per row of its argument, in its order. a row whose geometry
#: will not decode may answer nulls: it then intersects nothing
BoundsOf = Callable[[pa.Table], pa.Table]
