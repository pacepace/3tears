"""building layers from other layers: cutting, replacing, and checking coverage.

a layer here is a sequence of :class:`KeyedFeature`: a stable id, a geometry in
WGS84 longitude/latitude, and a **key** that partitions the layer (a state, a
country) so operations only ever pair features that share it. two layers drawn
from different sources disagree at their boundaries by a few meters, so cutting
one by the other leaves artifacts: thin slivers along shared lines and small
fragments where coastlines differ. each part of a cut is judged on its own and
dropped when it is narrower or smaller than the caller's thresholds. the
thresholds belong to the data, not to this module: measure them on the layers
in hand (the slivers' widths and areas against the smallest real piece) and
pass them in.

measurements are taken in a local equirectangular projection around each
shape's own latitude: accurate to well under a percent for anything the size of
a county, which is the scale these thresholds work at.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from shapely import STRtree, affinity, get_parts
from shapely.errors import GEOSException
from shapely.geometry import MultiPolygon
from shapely.geometry.base import BaseGeometry

from threetears.observe import get_logger

__all__ = [
    "KeyedFeature",
    "LayerOverlayError",
    "Overlay",
    "area_km2",
    "cut",
    "mean_width_m",
    "missing_features",
    "replace_features",
]

log = get_logger(__name__)

#: meters in one degree of latitude (and of longitude at the equator)
_METERS_PER_DEGREE = 111_320.0


class LayerOverlayError(ValueError):
    """a layer operation that cannot be carried out, naming the features involved."""


@dataclass(frozen=True, slots=True)
class KeyedFeature:
    """one feature of a layer, with the key that partitions the layer.

    :param feature_id: the feature's stable id
    :ptype feature_id: str
    :param key: the partition it belongs to (such as a state); features only meet within a key
    :ptype key: str
    :param geometry: its shape, in WGS84 longitude/latitude
    :ptype geometry: BaseGeometry
    """

    feature_id: str
    key: str
    geometry: BaseGeometry


@dataclass(frozen=True, slots=True)
class Overlay:
    """the result of cutting one layer by another.

    :param pieces: the pieces, each id ``<coarse id><separator><fine id>``
    :ptype pieces: tuple[KeyedFeature, ...]
    :param slivers_dropped: how many parts were judged artifacts and dropped
    :ptype slivers_dropped: int
    :param without_pieces: coarse features no fine feature of their key covers at all
    :ptype without_pieces: tuple[str, ...]
    """

    pieces: tuple[KeyedFeature, ...]
    slivers_dropped: int
    without_pieces: tuple[str, ...]


def _local(geometry: BaseGeometry) -> BaseGeometry:
    """the shape scaled to a local equirectangular projection around its own latitude, still in degrees.

    :param geometry: the shape, in longitude and latitude
    :ptype geometry: BaseGeometry
    :return: the shape with longitude shrunk by the cosine of its latitude
    :rtype: BaseGeometry
    """
    return affinity.scale(geometry, xfact=math.cos(math.radians(geometry.centroid.y)), yfact=1.0, origin=(0, 0))


def mean_width_m(geometry: BaseGeometry) -> float:
    """a shape's mean width in meters: its area over half its perimeter.

    for a long thin shape that is its width; for a square, half its side.

    :param geometry: the shape, in degrees
    :ptype geometry: BaseGeometry
    :return: the mean width, in meters; 0 for a shape with no perimeter
    :rtype: float
    """
    local = _local(geometry)
    return float(local.area / (local.length / 2)) * _METERS_PER_DEGREE if local.length else 0.0


def area_km2(geometry: BaseGeometry) -> float:
    """a shape's area in square kilometers.

    :param geometry: the shape, in degrees
    :ptype geometry: BaseGeometry
    :return: the area, in km²
    :rtype: float
    """
    return float(_local(geometry).area) * _METERS_PER_DEGREE**2 / 1e6


def _without_artifacts(
    intersection: BaseGeometry, min_width_m: float, min_area_km2: float
) -> tuple[BaseGeometry | None, int]:
    """an intersection's areas, each judged on its own, with the artifacts among them dropped.

    judged as one shape, a detached sliver would ride along with a real piece
    (stretching its bounding box and pulling its centroid), or drag a real
    piece's average width under the line. lines and points where two shapes
    only touch are not areas and are not counted.

    :param intersection: a coarse feature cut by a fine one
    :ptype intersection: BaseGeometry
    :param min_width_m: a part narrower than this on average is an artifact
    :ptype min_width_m: float
    :param min_area_km2: a part smaller than this is an artifact
    :ptype min_area_km2: float
    :return: the parts kept (None when none is), and how many were dropped
    :rtype: tuple[BaseGeometry | None, int]
    """
    areas = [part for part in get_parts(intersection) if part.area > 0]
    kept = [part for part in areas if mean_width_m(part) >= min_width_m and area_km2(part) >= min_area_km2]
    piece = (kept[0] if len(kept) == 1 else MultiPolygon(kept)) if kept else None
    return piece, len(areas) - len(kept)


def _intersection(coarse: KeyedFeature, fine: KeyedFeature) -> BaseGeometry:
    """the part of ``coarse`` inside ``fine``.

    :param coarse: the feature being cut
    :ptype coarse: KeyedFeature
    :param fine: the feature cutting it
    :ptype fine: KeyedFeature
    :return: the intersection, possibly empty or a shared edge
    :rtype: BaseGeometry
    :raises LayerOverlayError: when the geometry engine cannot intersect the two
    """
    try:
        result = coarse.geometry.intersection(fine.geometry)
    except GEOSException as exc:
        raise LayerOverlayError(f"cannot cut {coarse.feature_id!r} by {fine.feature_id!r}") from exc
    return result


def cut(
    coarse: Sequence[KeyedFeature],
    fine: Sequence[KeyedFeature],
    *,
    min_width_m: float,
    min_area_km2: float,
    separator: str = "-",
) -> Overlay:
    """each coarse feature cut by the fine features of its key that it overlaps.

    :param coarse: the layer being cut (such as counties)
    :ptype coarse: Sequence[KeyedFeature]
    :param fine: the layer cutting it (such as districts)
    :ptype fine: Sequence[KeyedFeature]
    :param min_width_m: a part narrower than this on average is an artifact and dropped
    :ptype min_width_m: float
    :param min_area_km2: a part smaller than this is an artifact and dropped
    :ptype min_area_km2: float
    :param separator: joins the two ids into a piece's id
    :ptype separator: str
    :return: the pieces, sorted by id, with what was dropped and what found no piece
    :rtype: Overlay
    :raises LayerOverlayError: when two shapes cannot be intersected
    """
    tree = STRtree([f.geometry for f in fine])
    pieces: list[KeyedFeature] = []
    slivers = 0
    without: list[str] = []
    for feature in coarse:
        found = 0
        for index in tree.query(feature.geometry, predicate="intersects"):
            other = fine[int(index)]
            if other.key != feature.key:
                continue
            piece, dropped = _without_artifacts(_intersection(feature, other), min_width_m, min_area_km2)
            slivers += dropped
            if piece is not None:
                found += 1
                pieces.append(KeyedFeature(f"{feature.feature_id}{separator}{other.feature_id}", feature.key, piece))
        if not found:
            without.append(feature.feature_id)
    log.info(
        "cut a layer by another",
        extra={"extra_data": {"pieces": len(pieces), "slivers_dropped": slivers, "without_pieces": len(without)}},
    )
    if without:
        log.warning(
            "features with no piece after cutting",
            extra={"extra_data": {"count": len(without), "features": without[:50]}},
        )
    return Overlay(
        pieces=tuple(sorted(pieces, key=lambda f: f.feature_id)),
        slivers_dropped=slivers,
        without_pieces=tuple(without),
    )


def replace_features(
    base: Iterable[KeyedFeature], replacements: Iterable[KeyedFeature], keys: frozenset[str]
) -> list[KeyedFeature]:
    """a layer with every feature of ``keys`` replaced by the replacements.

    the replacements must cover exactly ``keys``, so no key is ever left with
    neither source's features or both.

    :param base: the layer
    :ptype base: Iterable[KeyedFeature]
    :param replacements: the features that replace those of ``keys``
    :ptype replacements: Iterable[KeyedFeature]
    :param keys: the keys whose features are replaced
    :ptype keys: frozenset[str]
    :return: the layer, sorted by id
    :rtype: list[KeyedFeature]
    :raises LayerOverlayError: when the replacements and ``keys`` disagree
    """
    replacing = list(replacements)
    found = {f.key for f in replacing}
    if found != keys:
        raise LayerOverlayError(f"replacements cover keys {sorted(found)} but the keys replaced are {sorted(keys)}")
    kept = [f for f in base if f.key not in keys]
    log.info(
        "replaced a layer's features for a set of keys",
        extra={"extra_data": {"keys": len(keys), "replacements": len(replacing), "kept": len(kept)}},
    )
    return sorted([*kept, *replacing], key=lambda f: f.feature_id)


def missing_features(layer: Iterable[str], needed: Iterable[str]) -> tuple[str, ...]:
    """the feature ids a consumer needs that a layer has no feature for.

    a consumer joins its values to a layer by id; an id with no feature is a
    region that silently disappears from its map.

    :param layer: the layer's feature ids
    :ptype layer: Iterable[str]
    :param needed: the ids the consumer joins on
    :ptype needed: Iterable[str]
    :return: the needed ids with no feature, sorted
    :rtype: tuple[str, ...]
    """
    return tuple(sorted(set(needed) - set(layer)))
