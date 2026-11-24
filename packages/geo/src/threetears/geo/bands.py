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

import math
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import shapely
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from threetears.geo.attributes import coerce_attribute
from threetears.geo.geometry import RowGeometries
from threetears.geo.tiles import TileId
from threetears.observe import get_logger

__all__ = [
    "AggregateSpec",
    "BandResult",
    "FeatureSpec",
    "TileFeature",
    "UNDECODABLE_PROPERTY",
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

#: the property an aggregate feature carries when some of its bucket's source rows were left
#: out because their geometry could not be read: how many. its ``count`` and measures then
#: total only the rows that were read, and this says by how many rows they fall short, in the
#: stored tile itself rather than only in a log line. absent on a feature that lost nothing,
#: so a healthy tile encodes exactly as before and a client reads it as ``?? 0``. the same
#: name carries a tile's whole count on its :data:`~threetears.geo.mvt.PARTIAL_LAYER_SUFFIX` layer.
UNDECODABLE_PROPERTY: Final = "undecodable"


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
    :param undecodable: how many source rows were left out because their geometry could not be
        read (:attr:`~threetears.geo.geometry.RowGeometries.undecodable`). a malformed row costs its
        own feature, never the tile, and is counted here rather than passed off as absent; the
        encoded tile carries the count in its :data:`~threetears.geo.mvt.PARTIAL_LAYER_SUFFIX` layer
    :ptype undecodable: int
    :param first_undecodable: the feature id (or rollup key) of the first such row, so the warning
        points at a row an operator can look up; ``None`` when none was left out or the row has no id
    :ptype first_undecodable: Any
    """

    features: list[TileFeature]
    truncated: bool = False
    dropped: int = 0
    undecodable: int = 0
    first_undecodable: Any = None


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
    :raises ValueError: when the rollup column or a measure is named ``undecodable``
        (:data:`UNDECODABLE_PROPERTY`), which would overwrite the partial-bucket marker or be
        overwritten by it
    """

    rollup_column: str | None = None
    measures: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rollup_column == UNDECODABLE_PROPERTY or UNDECODABLE_PROPERTY in self.measures:
            raise ValueError(
                f"{UNDECODABLE_PROPERTY!r} is reserved for the partial-bucket marker on aggregate features"
            )


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


def feature_band(table: pa.Table, geometries: RowGeometries, *, tile: TileId, spec: FeatureSpec) -> BandResult:
    """build the individual-feature band for one tile, over whole columns.

    simplifies each geometry to the zoom's tolerance, then applies the cap.
    the cap is applied *after* geometry work because ranking must consider
    every candidate -- capping first would rank an arbitrary subset. only the
    features kept become Python objects, as the MVT encoder takes them; the
    candidates stay columns.

    :param table: candidate source rows, already bbox-filtered by the caller
    :ptype table: pa.Table
    :param geometries: every row's geometry (the layer's
        :meth:`~threetears.geo.collection.LayerDefinition.geometries_of`)
    :ptype geometries: RowGeometries
    :param tile: the tile being built
    :ptype tile: TileId
    :param spec: the layer's declared feature band
    :ptype spec: FeatureSpec
    :return: the band's features plus truncation accounting
    :rtype: BandResult
    """
    shapes = geometries.shapes
    undecodable = int(np.count_nonzero(geometries.undecodable))
    first_undecodable: Any = None
    if undecodable and spec.feature_id_column is not None:
        first_undecodable = _value_at(table, spec.feature_id_column, int(np.argmax(geometries.undecodable)))
    drawn = ~shapely.is_missing(shapes)
    simplified = np.full(len(shapes), None, dtype=object)
    simplified[drawn] = shapely.simplify(shapes[drawn], simplification_tolerance(tile.z), preserve_topology=True)
    # simplification can collapse a sliver entirely; that is the correct outcome at this zoom, not an error.
    candidates = np.flatnonzero(drawn & ~shapely.is_empty(simplified))

    if len(candidates) <= spec.max_features_per_tile:
        return BandResult(
            features=_tile_features(table, simplified, candidates, spec),
            undecodable=undecodable,
            first_undecodable=first_undecodable,
        )

    # descending rank, ties in source order (a stable sort of the negated rank): keep the most important
    order = np.argsort(-_ranks(table, spec.rank_column)[candidates], kind="stable")
    kept_rows = candidates[order[: spec.max_features_per_tile]]
    dropped = len(candidates) - spec.max_features_per_tile
    kept = _tile_features(table, simplified, kept_rows, spec)
    log.info(
        "tile %s: feature cap bound, kept %d of %d (ranked by %s)",
        tile,
        len(kept),
        len(candidates),
        spec.rank_column or "source order",
    )
    return BandResult(
        features=kept, truncated=True, dropped=dropped, undecodable=undecodable, first_undecodable=first_undecodable
    )


def _tile_features(table: pa.Table, shapes: np.ndarray, rows: np.ndarray, spec: FeatureSpec) -> list[TileFeature]:
    """the features of ``rows``, in their order: each one's geometry, attributes and id.

    one Python object per KEPT feature, as the MVT encoder takes them; the rows' columns are read for these rows
    only.
    """
    if not len(rows):
        return []
    picked = table.take(pa.array(rows, type=pa.int64()))
    ids = _column_values(picked, spec.feature_id_column)
    columns = [(name, _column_values(picked, name)) for name in spec.attributes]
    features: list[TileFeature] = []
    for position, row in enumerate(rows.tolist()):
        feature_id = ids[position]
        attributes: dict[str, Any] = {}
        for name, values in columns:
            coerced = coerce_attribute(values[position])
            if coerced is not None:
                attributes[name] = coerced
        if spec.feature_id_column is not None and feature_id is not None:
            # the id always travels as a property, not only as the wire-level
            # MVT id: that id is uint64 by specification, so a geoid or UUID
            # cannot be carried there. a client binds volatile values via
            # MapLibre's promoteId against this property.
            attributes.setdefault(spec.feature_id_column, feature_id)
        features.append(TileFeature(geometry=shapes[row], attributes=attributes, feature_id=feature_id))
    return features


def _column_values(table: pa.Table, column: str | None) -> list[Any]:
    """one column's values for the (few, picked) rows of ``table``; all ``None`` when it is not there."""
    if column is None or column not in table.column_names:
        return [None] * table.num_rows
    values: list[Any] = table.column(column).to_pylist()
    return values


def _value_at(table: pa.Table, column: str, row: int) -> Any:
    """one row's value of one column, ``None`` when the table has no such column."""
    if column not in table.column_names:
        return None
    return table.column(column)[row].as_py()


def aggregate_band(table: pa.Table, geometries: RowGeometries, *, tile: TileId, spec: AggregateSpec) -> BandResult:
    """build the rollup band for one tile, over whole columns.

    groups rows by the declared coarser geography (or by the tile itself when
    none is declared), and emits one point feature per group at the group's
    centroid, carrying the declared measures plus a ``count``. groups come in
    the order of their first row that has a geometry.

    a row whose geometry cannot be read is left out and counted against its bucket: the
    bucket's feature carries :data:`UNDECODABLE_PROPERTY`, so a total that undercounts says so
    in the tile. a bucket none of whose rows could be read has no position and so no feature;
    its rows are still counted on the result's ``undecodable``, which the encoder turns into the
    tile's :data:`~threetears.geo.mvt.PARTIAL_LAYER_SUFFIX` layer.

    the centroid is the mean of member centroids rather than the centre of the
    tile, so a group's marker sits on its members instead of in whatever empty
    space the tile happens to centre on. its means and sums add in row order as
    Python's ``sum`` does (:func:`_python_sum`), so a tile is the same bytes it
    was when the band read a row at a time.

    :param table: candidate source rows, already bbox-filtered by the caller
    :ptype table: pa.Table
    :param geometries: every row's geometry
    :ptype geometries: RowGeometries
    :param tile: the tile being built
    :ptype tile: TileId
    :param spec: the layer's declared aggregate band
    :ptype spec: AggregateSpec
    :return: one feature per group
    :rtype: BandResult
    """
    shapes, unread = geometries.shapes, geometries.undecodable
    codes, buckets = _bucket_codes(table, spec, tile)
    undecodable = int(np.count_nonzero(unread))
    first_undecodable: Any = None
    if undecodable and spec.rollup_column is not None:
        first_undecodable = buckets[int(codes[int(np.argmax(unread))])].as_py()
    unread_by_bucket = np.bincount(codes[unread], minlength=len(buckets))

    drawn = np.flatnonzero(~shapely.is_missing(shapes))
    if not len(drawn):
        return BandResult(features=[], undecodable=undecodable, first_undecodable=first_undecodable)
    centroids = shapely.centroid(shapes[drawn])
    xs, ys = shapely.get_x(centroids), shapely.get_y(centroids)
    drawn_codes = codes[drawn]
    # each group's members, in row order (a stable sort by group), and the groups in order of their first member
    by_group = np.argsort(drawn_codes, kind="stable")
    group_codes, starts = np.unique(drawn_codes[by_group], return_index=True)
    member_runs = np.split(by_group, starts[1:])
    firsts = np.array([run[0] for run in member_runs])
    measures = [(column, how, _measure_column(table, column)) for column, how in spec.measures.items()]

    features: list[TileFeature] = []
    for group in np.argsort(firsts, kind="stable").tolist():
        members = member_runs[group]
        code = int(group_codes[group])
        bucket = buckets[code].as_py() if spec.rollup_column is not None else tile.key
        count = len(members)
        mean_x = _python_sum(xs[members]) / count
        mean_y = _python_sum(ys[members]) / count
        attributes: dict[str, Any] = {"count": count}
        if spec.rollup_column is not None:
            attributes[spec.rollup_column] = bucket
        for column, how, (numbers, valid, numeric) in measures:
            aggregated = _aggregate(numbers, valid, numeric, drawn[members], how)
            if aggregated is not None:
                attributes[column] = aggregated
        lost = int(unread_by_bucket[code])
        if lost:
            attributes[UNDECODABLE_PROPERTY] = lost
        features.append(
            TileFeature(
                geometry=Point(mean_x, mean_y),
                attributes=attributes,
                # the rollup id is the join key for volatile values in this
                # band, exactly as feature_id is in the feature band.
                feature_id=bucket if isinstance(bucket, (str, int)) else None,
            )
        )
    return BandResult(features=features, undecodable=undecodable, first_undecodable=first_undecodable)


def _bucket_codes(table: pa.Table, spec: AggregateSpec, tile: TileId) -> tuple[np.ndarray, pa.Array]:
    """each row's group, as an index into the groups' values: the rollup column dictionary-encoded (a null is a
    group of its own; a table without the column is one null group), or one group when none is declared.

    :return: ``(codes, values)``: int64 codes, one per row, and the groups' values
    :rtype: tuple[np.ndarray, pa.Array]
    """
    rows = table.num_rows
    if spec.rollup_column is None:
        return np.zeros(rows, dtype=np.int64), pa.array([None])
    if spec.rollup_column not in table.column_names:
        return np.zeros(rows, dtype=np.int64), pa.array([None])
    column = table.column(spec.rollup_column)
    if pa.types.is_null(column.type):
        return np.zeros(rows, dtype=np.int64), pa.array([None])
    encoded = pc.dictionary_encode(column.combine_chunks(), null_encoding="encode")
    return encoded.indices.to_numpy(zero_copy_only=False).astype(np.int64), encoded.dictionary


def _measure_column(table: pa.Table, column: str) -> tuple[np.ndarray, np.ndarray, bool]:
    """a measure column whole: its values as float64, which rows have one, and whether it is numeric.

    a measure sums and averages only int and float values (never bool, decimal or text), as it always has; a
    column of another type still counts its non-null rows.

    :return: ``(numbers, valid, numeric)``
    :rtype: tuple[np.ndarray, np.ndarray, bool]
    """
    rows = table.num_rows
    if column not in table.column_names:
        return np.full(rows, np.nan), np.zeros(rows, dtype=bool), False
    values = table.column(column)
    valid = pc.is_valid(values).to_numpy(zero_copy_only=False)
    if not (pa.types.is_integer(values.type) or pa.types.is_floating(values.type)):
        return np.full(rows, np.nan), valid, False
    numbers = pc.fill_null(pc.cast(values, pa.float64(), safe=False), np.nan).to_numpy(zero_copy_only=False)
    return numbers.astype(np.float64), valid, True


def _ranks(table: pa.Table, rank_column: str | None) -> np.ndarray:
    """every row's rank for cap ordering, as float64; unrankable rows (null, absent, not an int or float column)
    rank -inf and sort last. with no rank column every row ranks 0, so the cap keeps source order."""
    rows = table.num_rows
    if rank_column is None:
        return np.zeros(rows)
    if rank_column not in table.column_names:
        return np.full(rows, -np.inf)
    values = table.column(rank_column)
    if not (pa.types.is_integer(values.type) or pa.types.is_floating(values.type)):
        return np.full(rows, -np.inf)
    ranked = pc.fill_null(pc.cast(values, pa.float64(), safe=False), -np.inf)
    floats: np.ndarray = ranked.to_numpy(zero_copy_only=False).astype(np.float64)
    return floats


def _python_sum(values: np.ndarray) -> float:
    """``sum()`` of these floats exactly as Python 3.12+ adds them: in order, with Neumaier compensation.

    the running total is a sequential cumulative sum; each step's lost low part is computed elementwise from the
    totals either side of it, and those parts add sequentially, then onto the total when non-zero and finite --
    CPython's own steps, so the answer is bit for bit what the band's ``sum`` gave (a pairwise numpy sum is not).
    the running total starts at 0.0 as Python's does, so a leading -0.0 is +0.0.

    :param values: float64, at least one
    :ptype values: np.ndarray
    :return: their sum
    :rtype: float
    """
    with np.errstate(all="ignore"):
        totals = np.cumsum(values) + 0.0
        before = np.concatenate(([0.0], totals[:-1]))
        lost = np.where(np.abs(before) >= np.abs(values), (before - totals) + values, (values - totals) + before)
        compensation = float(np.cumsum(lost)[-1])
    total = float(totals[-1])
    if compensation and math.isfinite(compensation):
        total += compensation
    return total


def _aggregate(numbers: np.ndarray, valid: np.ndarray, numeric: bool, rows: np.ndarray, how: str) -> float | int | None:
    """apply one declared aggregation over ``rows`` of a measure column, ignoring nulls.

    nulls are skipped rather than counted as zero: a tract with no measurement
    must not drag a regional mean toward zero, which is the same reasoning
    that omits null attributes from a tile rather than encoding them.
    """
    if how == "count":
        return int(np.count_nonzero(valid[rows]))
    if not numeric:
        return None
    present = rows[valid[rows]]
    if not len(present):
        return None
    if how == "sum":
        return _python_sum(numbers[present])
    if how == "mean":
        return _python_sum(numbers[present]) / len(present)
    raise ValueError(f"unknown aggregation {how!r}; expected one of sum/mean/count")
