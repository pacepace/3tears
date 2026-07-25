"""zoom bands: aggregate below the crossover, individual features above.

low zoom is not simplified high zoom. a tile at z4 spans a large fraction of
a country; rendering it by simplifying and dropping individual features
produces a map that is both expensive to build and wrong to look at, because
whichever features survive the drop are an arbitrary sample of the ones that
did not. a national view showing 4,000 of 180,000 precincts is not a coarse
view of the data -- it is a different, misleading dataset.

so every layer has two bands:

- the **aggregate band** groups source rows into buckets and emits one
  feature per bucket carrying measures rather than source geometry. a
  national view is then a rollup with real totals.
- the **feature band** emits individual features, simplified per zoom and
  capped in count.

`bl-ds-ai-lcv-registration` arrived at the same split independently: its
precomputed z4-z10 band holds cluster aggregates and individual features
appear only from z11.

the cap in the feature band is a hard limit rather than advice. an uncapped
tile in a dense metro can reach tens of megabytes, which blows the NATS
payload ceiling, defeats the L2 hot band, and produces a tile no browser
renders acceptably. when it binds, features are dropped by a declared ranking
column so the survivors are the important ones rather than whichever the
query returned first, and the tile records that it was truncated.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from shapely.geometry.base import BaseGeometry

from threetears.geo.tiles import TileId
from threetears.observe import get_logger

__all__ = [
    "AggregateSpec",
    "BandResult",
    "FeatureSpec",
    "TileFeature",
    "aggregate_band",
    "feature_band",
    "simplification_tolerance",
]

log = get_logger(__name__)

#: tolerance at zoom 0, in degrees, halved per zoom level. chosen so that at
#: any zoom the tolerance is roughly one tile-extent unit: simplifying below
#: that removes vertices the tile grid cannot express anyway, and simplifying
#: above it is visible.
_BASE_TOLERANCE_DEGREES: Final = 360.0 / 4096.0


@dataclass(frozen=True, slots=True)
class TileFeature:
    """one feature destined for MVT encoding.

    :param geometry: the feature's geometry, in WGS84
    :ptype geometry: BaseGeometry
    :param attributes: MVT-safe attribute values
    :ptype attributes: dict[str, Any]
    :param feature_id: stable identity, used as the MVT feature id so a
        client can bind volatile values to static geometry
    :ptype feature_id: str | int | None
    """

    geometry: BaseGeometry
    attributes: dict[str, Any]
    feature_id: str | int | None = None


@dataclass(frozen=True, slots=True)
class BandResult:
    """the outcome of building one band for one tile.

    :param features: features to encode
    :ptype features: list[TileFeature]
    :param truncated: whether the feature cap dropped anything
    :ptype truncated: bool
    :param dropped: how many features the cap dropped
    :ptype dropped: int
    """

    features: list[TileFeature]
    truncated: bool = False
    dropped: int = 0


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """declared shape of a layer's feature band.

    :param attributes: columns carried onto each feature
    :ptype attributes: tuple[str, ...]
    :param feature_id_column: column holding the stable feature identity
    :ptype feature_id_column: str | None
    :param max_features_per_tile: hard cap; see the module docstring
    :ptype max_features_per_tile: int
    :param rank_column: column ranking features for the cap, descending. when
        absent the cap drops in whatever order the source returned, which is
        arbitrary -- declare one for any layer where the cap is likely to bind
    :ptype rank_column: str | None
    """

    attributes: tuple[str, ...] = ()
    feature_id_column: str | None = None
    max_features_per_tile: int = 4000
    rank_column: str | None = None


@dataclass(frozen=True, slots=True)
class AggregateSpec:
    """declared shape of a layer's aggregate band.

    :param rollup_column: column naming the coarser geography rows group into
        (state FIPS, county id). when absent, rows group by the tile itself,
        which is the fallback for a layer with no declared coarser geography
    :ptype rollup_column: str | None
    :param measures: column name to aggregation, one of ``sum``/``mean``/``count``
    :ptype measures: dict[str, str]
    """

    rollup_column: str | None = None
    measures: dict[str, str] = field(default_factory=dict)


def simplification_tolerance(zoom: int) -> float:
    """return the Douglas-Peucker tolerance for ``zoom``, in degrees.

    halves per zoom level, so the tolerance tracks the ground resolution a
    tile can actually express. simplifying more than this is visible;
    simplifying less keeps vertices the 4096-unit grid quantizes away
    regardless, at full payload cost.

    :param zoom: zoom level
    :ptype zoom: int
    :return: tolerance in degrees
    :rtype: float
    """
    return _BASE_TOLERANCE_DEGREES / (1 << zoom)


def feature_band(
    rows: Iterable[dict[str, Any]],
    *,
    tile: TileId,
    spec: FeatureSpec,
    geometry_of: Callable[[dict[str, Any]], BaseGeometry | None],
    attributes_of: Callable[[dict[str, Any], tuple[str, ...]], dict[str, Any]],
) -> BandResult:
    """build the individual-feature band for one tile.

    simplifies each geometry to the zoom's tolerance, then applies the cap.
    the cap is applied *after* geometry work because ranking must consider
    every candidate -- capping first would rank an arbitrary subset.

    :param rows: candidate source rows, already bbox-filtered by the caller
    :ptype rows: Iterable[dict[str, Any]]
    :param tile: the tile being built
    :ptype tile: TileId
    :param spec: the layer's declared feature band
    :ptype spec: FeatureSpec
    :param geometry_of: extracts geometry from a row (WKB column or lon/lat pair)
    :ptype geometry_of: Callable[[dict[str, Any]], BaseGeometry | None]
    :param attributes_of: projects declared columns into MVT-safe attributes
    :ptype attributes_of: Callable[[dict[str, Any], tuple[str, ...]], dict[str, Any]]
    :return: the band's features plus truncation accounting
    :rtype: BandResult
    """
    tolerance = simplification_tolerance(tile.z)
    candidates: list[tuple[float, TileFeature]] = []
    for row in rows:
        geometry = geometry_of(row)
        if geometry is None:
            continue
        simplified = geometry.simplify(tolerance, preserve_topology=True)
        if simplified.is_empty:
            # simplification can collapse a sliver entirely; that is the
            # correct outcome at this zoom, not an error.
            continue
        feature_id = row.get(spec.feature_id_column) if spec.feature_id_column else None
        candidates.append(
            (
                _rank_of(row, spec.rank_column),
                TileFeature(
                    geometry=simplified,
                    attributes=attributes_of(row, spec.attributes),
                    feature_id=feature_id,
                ),
            )
        )

    if len(candidates) <= spec.max_features_per_tile:
        return BandResult(features=[feature for _, feature in candidates])

    # descending rank: keep the most important, drop the rest.
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    dropped = len(candidates) - spec.max_features_per_tile
    kept = [feature for _, feature in candidates[: spec.max_features_per_tile]]
    log.info(
        "tile %s: feature cap bound, kept %d of %d (ranked by %s)",
        tile,
        len(kept),
        len(candidates),
        spec.rank_column or "source order",
    )
    return BandResult(features=kept, truncated=True, dropped=dropped)


def aggregate_band(
    rows: Iterable[dict[str, Any]],
    *,
    tile: TileId,
    spec: AggregateSpec,
    geometry_of: Callable[[dict[str, Any]], BaseGeometry | None],
) -> BandResult:
    """build the rollup band for one tile.

    groups rows by the declared coarser geography (or by the tile itself when
    none is declared), and emits one point feature per group at the group's
    centroid, carrying the declared measures plus a ``count``.

    the centroid is the mean of member centroids rather than the centre of the
    tile, so a group's marker sits on its members instead of in whatever empty
    space the tile happens to centre on.

    :param rows: candidate source rows, already bbox-filtered by the caller
    :ptype rows: Iterable[dict[str, Any]]
    :param tile: the tile being built
    :ptype tile: TileId
    :param spec: the layer's declared aggregate band
    :ptype spec: AggregateSpec
    :param geometry_of: extracts geometry from a row
    :ptype geometry_of: Callable[[dict[str, Any]], BaseGeometry | None]
    :return: one feature per group
    :rtype: BandResult
    """
    from shapely.geometry import Point

    groups: dict[Any, list[tuple[dict[str, Any], BaseGeometry]]] = {}
    for row in rows:
        geometry = geometry_of(row)
        if geometry is None:
            continue
        bucket = row.get(spec.rollup_column) if spec.rollup_column else tile.key
        groups.setdefault(bucket, []).append((row, geometry))

    features: list[TileFeature] = []
    for bucket, members in groups.items():
        centroids = [geometry.centroid for _, geometry in members]
        mean_x = sum(point.x for point in centroids) / len(centroids)
        mean_y = sum(point.y for point in centroids) / len(centroids)
        attributes: dict[str, Any] = {"count": len(members)}
        if spec.rollup_column is not None:
            attributes[spec.rollup_column] = bucket
        for column, how in spec.measures.items():
            aggregated = _aggregate([row.get(column) for row, _ in members], how)
            if aggregated is not None:
                attributes[column] = aggregated
        features.append(
            TileFeature(
                geometry=Point(mean_x, mean_y),
                attributes=attributes,
                # the rollup id is the join key for volatile values in this
                # band, exactly as feature_id is in the feature band.
                feature_id=bucket if isinstance(bucket, (str, int)) else None,
            )
        )
    return BandResult(features=features)


def _rank_of(row: dict[str, Any], rank_column: str | None) -> float:
    """numeric rank for cap ordering; unrankable rows sort last."""
    if rank_column is None:
        return 0.0
    value = row.get(rank_column)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float("-inf")
    return float(value)


def _aggregate(values: Sequence[Any], how: str) -> float | int | None:
    """apply one declared aggregation, ignoring nulls.

    nulls are skipped rather than counted as zero: a tract with no measurement
    must not drag a regional mean toward zero, which is the same reasoning
    that omits null attributes from a tile rather than encoding them.
    """
    if how == "count":
        return sum(1 for value in values if value is not None)
    numeric = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if not numeric:
        return None
    if how == "sum":
        return sum(numeric)
    if how == "mean":
        return sum(numeric) / len(numeric)
    raise ValueError(f"unknown aggregation {how!r}; expected one of sum/mean/count")
