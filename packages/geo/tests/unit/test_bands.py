"""unit tests for the two zoom bands.

the cap-ordering and null-aggregation tests are the ones that matter: both
are cases where a plausible implementation silently produces a misleading
map rather than an error.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from shapely.geometry import Point, Polygon

from threetears.geo.attributes import coerce_attribute
from threetears.geo.geometry import RowGeometries
from threetears.geo.bands import (
    UNDECODABLE_PROPERTY,
    AggregateSpec,
    FeatureSpec,
    aggregate_band,
    feature_band,
    simplification_tolerance,
)
from threetears.geo.tiles import TileId


def _point_of(row: dict[str, Any]) -> Point | None:
    if row.get("lon") is None:
        return None
    return Point(row["lon"], row["lat"])


def _table(rows: list[dict[str, Any]]) -> pa.Table:
    """the rows as the band reads them, without the test's own geometry and marker keys."""
    return pa.Table.from_pylist([{k: v for k, v in row.items() if k not in ("geom", "bad")} for row in rows])


def _geometries(rows: list[dict[str, Any]]) -> RowGeometries:
    """each row's point (or its ``geom``), and a row marked ``bad`` as one whose geometry could not be read."""
    shapes = np.full(len(rows), None, dtype=object)
    for index, row in enumerate(rows):
        if not row.get("bad"):
            shapes[index] = row["geom"] if "geom" in row else _point_of(row)
    return RowGeometries(shapes=shapes, undecodable=np.array([bool(row.get("bad")) for row in rows], dtype=bool))


def _rows(count: int, **extra: Any) -> list[dict[str, Any]]:
    return [{"id": n, "lon": -112.0 + n * 0.001, "lat": 33.4, **extra} for n in range(count)]


class TestSimplificationTolerance:
    def test_halves_per_zoom(self) -> None:
        assert simplification_tolerance(5) == pytest.approx(simplification_tolerance(4) / 2)

    def test_tighter_at_higher_zoom(self) -> None:
        assert simplification_tolerance(14) < simplification_tolerance(4)


class TestUndecodableRows:
    def test_feature_band_counts_the_rows_it_could_not_read(self) -> None:
        rows = _rows(2) + [{"id": 9, "lon": 0, "lat": 0, "bad": True}]
        result = feature_band(
            _table(rows),
            _geometries(rows),
            tile=TileId(z=12, x=100, y=200),
            spec=FeatureSpec(attributes=("id",), feature_id_column="id"),
        )
        assert len(result.features) == 2
        assert result.undecodable == 1

    def test_aggregate_band_counts_the_rows_it_could_not_read(self) -> None:
        rows = _rows(2) + [{"id": 9, "lon": 0, "lat": 0, "bad": True}]
        result = aggregate_band(
            _table(rows),
            _geometries(rows),
            tile=TileId(z=5, x=6, y=12),
            spec=AggregateSpec(rollup_column="state"),
        )
        assert result.undecodable == 1

    def test_aggregate_feature_carries_its_own_buckets_undecodable_rows(self) -> None:
        # the marker is per bucket: a state's total falls short by its own unread rows, and a
        # state that lost none is unmarked.
        rows = [
            {"id": 1, "lon": -112.0, "lat": 33.4, "state": "04", "unreg": 100},
            {"id": 2, "lon": 0, "lat": 0, "state": "04", "unreg": 200, "bad": True},
            {"id": 3, "lon": 0, "lat": 0, "state": "04", "unreg": 300, "bad": True},
            {"id": 4, "lon": -115.0, "lat": 36.0, "state": "32", "unreg": 50},
        ]
        result = aggregate_band(
            _table(rows),
            _geometries(rows),
            tile=TileId(z=4, x=3, y=6),
            spec=AggregateSpec(rollup_column="state", measures={"unreg": "sum"}),
        )
        by_state = {f.attributes["state"]: f.attributes for f in result.features}
        assert by_state["04"]["unreg"] == 100
        assert by_state["04"][UNDECODABLE_PROPERTY] == 2
        assert UNDECODABLE_PROPERTY not in by_state["32"]
        assert result.undecodable == 2

    def test_without_a_rollup_column_the_tile_bucket_is_marked(self) -> None:
        rows = _rows(3) + [{"id": 9, "lon": 0, "lat": 0, "bad": True}]
        result = aggregate_band(
            _table(rows),
            _geometries(rows),
            tile=TileId(z=4, x=3, y=6),
            spec=AggregateSpec(),
        )
        (feature,) = result.features
        assert feature.attributes["count"] == 3
        assert feature.attributes[UNDECODABLE_PROPERTY] == 1

    def test_a_bucket_with_no_readable_row_has_no_feature_but_is_still_counted(self) -> None:
        # nothing gives it a position, so there is nothing to stamp; the result's count still has it.
        rows = [
            {"id": 1, "lon": -112.0, "lat": 33.4, "state": "04"},
            {"id": 2, "lon": 0, "lat": 0, "state": "32", "bad": True},
        ]
        result = aggregate_band(
            _table(rows),
            _geometries(rows),
            tile=TileId(z=4, x=3, y=6),
            spec=AggregateSpec(rollup_column="state"),
        )
        assert [f.attributes["state"] for f in result.features] == ["04"]
        assert UNDECODABLE_PROPERTY not in result.features[0].attributes
        assert result.undecodable == 1

    @pytest.mark.parametrize(
        "spec_kwargs",
        [{"rollup_column": UNDECODABLE_PROPERTY}, {"measures": {UNDECODABLE_PROPERTY: "sum"}}],
    )
    def test_the_marker_name_is_reserved(self, spec_kwargs: dict[str, Any]) -> None:
        # a source column of that name would overwrite the marker, or be overwritten by it.
        with pytest.raises(ValueError, match="reserved"):
            AggregateSpec(**spec_kwargs)


class TestFeatureBand:
    def test_emits_one_feature_per_row(self) -> None:
        result = feature_band(
            _table(_rows(3)),
            _geometries(_rows(3)),
            tile=TileId(z=12, x=100, y=200),
            spec=FeatureSpec(attributes=("id",), feature_id_column="id"),
        )
        assert len(result.features) == 3
        assert not result.truncated
        assert [f.feature_id for f in result.features] == [0, 1, 2]

    def test_rows_without_geometry_are_skipped_not_fatal(self) -> None:
        rows = _rows(2) + [{"id": 99, "lon": None, "lat": None}]
        result = feature_band(
            _table(rows),
            _geometries(rows),
            tile=TileId(z=12, x=100, y=200),
            spec=FeatureSpec(attributes=("id",)),
        )
        assert len(result.features) == 2

    def test_cap_keeps_the_highest_ranked(self) -> None:
        # the cap must drop by declared importance; dropping in source order
        # would keep an arbitrary subset and misrepresent a dense area.
        rows = [{"id": n, "lon": -112.0 + n * 0.001, "lat": 33.4, "score": n} for n in range(10)]
        result = feature_band(
            _table(rows),
            _geometries(rows),
            tile=TileId(z=12, x=100, y=200),
            spec=FeatureSpec(attributes=("id", "score"), max_features_per_tile=3, rank_column="score"),
        )
        assert result.truncated
        assert result.dropped == 7
        assert sorted(f.attributes["score"] for f in result.features) == [7, 8, 9]

    def test_cap_reports_truncation(self) -> None:
        # a silently truncated tile reads as "this is all the data there is".
        result = feature_band(
            _table(_rows(50)),
            _geometries(_rows(50)),
            tile=TileId(z=12, x=100, y=200),
            spec=FeatureSpec(attributes=("id",), max_features_per_tile=10),
        )
        assert result.truncated
        assert result.dropped == 40

    def test_under_the_cap_is_not_marked_truncated(self) -> None:
        result = feature_band(
            _table(_rows(5)),
            _geometries(_rows(5)),
            tile=TileId(z=12, x=100, y=200),
            spec=FeatureSpec(attributes=("id",), max_features_per_tile=10),
        )
        assert not result.truncated
        assert result.dropped == 0

    def test_geometry_is_simplified_at_low_zoom(self) -> None:
        # a many-vertex polygon should lose vertices the tile grid cannot
        # express anyway.
        detailed = Polygon([(-112.0 + i * 0.0001, 33.4 + (i % 2) * 0.00001) for i in range(60)] + [(-112.0, 33.4)])
        rows = [{"id": 1, "geom": detailed}]
        result = feature_band(
            _table(rows),
            _geometries(rows),
            tile=TileId(z=4, x=3, y=6),
            spec=FeatureSpec(attributes=("id",)),
        )
        simplified = result.features[0].geometry
        assert len(simplified.exterior.coords) < len(detailed.exterior.coords)


class TestAggregateBand:
    def test_groups_by_declared_rollup_column(self) -> None:
        rows = [
            {"id": 1, "lon": -112.0, "lat": 33.4, "state": "04", "unreg": 100},
            {"id": 2, "lon": -112.1, "lat": 33.5, "state": "04", "unreg": 200},
            {"id": 3, "lon": -115.0, "lat": 36.0, "state": "32", "unreg": 50},
        ]
        result = aggregate_band(
            _table(rows),
            _geometries(rows),
            tile=TileId(z=4, x=3, y=6),
            spec=AggregateSpec(rollup_column="state", measures={"unreg": "sum"}),
        )
        by_state = {f.attributes["state"]: f.attributes for f in result.features}
        assert by_state["04"]["unreg"] == 300
        assert by_state["04"]["count"] == 2
        assert by_state["32"]["unreg"] == 50

    def test_rollup_id_is_promoted_for_client_side_joins(self) -> None:
        # the aggregate band needs a join key for volatile values exactly as
        # the feature band does.
        rows = [{"id": 1, "lon": -112.0, "lat": 33.4, "state": "04"}]
        result = aggregate_band(
            _table(rows),
            _geometries(rows),
            tile=TileId(z=4, x=3, y=6),
            spec=AggregateSpec(rollup_column="state"),
        )
        assert result.features[0].feature_id == "04"

    def test_centroid_sits_on_members_not_tile_centre(self) -> None:
        # a marker in the empty middle of a tile misrepresents where the data is.
        rows = [
            {"id": 1, "lon": -112.0, "lat": 33.4, "state": "04"},
            {"id": 2, "lon": -112.2, "lat": 33.6, "state": "04"},
        ]
        result = aggregate_band(
            _table(rows),
            _geometries(rows),
            tile=TileId(z=4, x=3, y=6),
            spec=AggregateSpec(rollup_column="state"),
        )
        centroid = result.features[0].geometry
        assert centroid.x == pytest.approx(-112.1)
        assert centroid.y == pytest.approx(33.5)

    def test_mean_ignores_nulls_rather_than_treating_them_as_zero(self) -> None:
        # counting a missing measurement as zero drags the regional mean down
        # and shades the map with data that was never collected.
        rows = [
            {"id": 1, "lon": -112.0, "lat": 33.4, "state": "04", "approval": 0.6},
            {"id": 2, "lon": -112.1, "lat": 33.5, "state": "04", "approval": None},
        ]
        result = aggregate_band(
            _table(rows),
            _geometries(rows),
            tile=TileId(z=4, x=3, y=6),
            spec=AggregateSpec(rollup_column="state", measures={"approval": "mean"}),
        )
        assert result.features[0].attributes["approval"] == pytest.approx(0.6)

    def test_all_null_measure_is_omitted_entirely(self) -> None:
        rows = [{"id": 1, "lon": -112.0, "lat": 33.4, "state": "04", "approval": None}]
        result = aggregate_band(
            _table(rows),
            _geometries(rows),
            tile=TileId(z=4, x=3, y=6),
            spec=AggregateSpec(rollup_column="state", measures={"approval": "mean"}),
        )
        assert "approval" not in result.features[0].attributes

    def test_count_measure_counts_non_null(self) -> None:
        rows = [
            {"id": 1, "lon": -112.0, "lat": 33.4, "state": "04", "approval": 0.6},
            {"id": 2, "lon": -112.1, "lat": 33.5, "state": "04", "approval": None},
        ]
        result = aggregate_band(
            _table(rows),
            _geometries(rows),
            tile=TileId(z=4, x=3, y=6),
            spec=AggregateSpec(rollup_column="state", measures={"approval": "count"}),
        )
        assert result.features[0].attributes["approval"] == 1

    def test_without_a_rollup_column_rows_group_by_tile(self) -> None:
        result = aggregate_band(
            _table(_rows(4)),
            _geometries(_rows(4)),
            tile=TileId(z=4, x=3, y=6),
            spec=AggregateSpec(),
        )
        assert len(result.features) == 1
        assert result.features[0].attributes["count"] == 4

    def test_unknown_aggregation_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown aggregation"):
            aggregate_band(
                _table(_rows(2, state="04", v=1)),
                _geometries(_rows(2, state="04", v=1)),
                tile=TileId(z=4, x=3, y=6),
                spec=AggregateSpec(rollup_column="state", measures={"v": "median"}),
            )


class TestBandsReadColumns:
    """the bands read whole columns: only the features a tile keeps become Python objects."""

    def _polygons(self, count: int) -> tuple[pa.Table, RowGeometries]:
        rows = [
            {"id": n, "lon": -112.0 + n * 0.0001, "lat": 33.4, "score": n % 97, "state": f"{n % 3:02d}", "v": n}
            for n in range(count)
        ]
        return _table(rows), _geometries(rows)

    def test_a_capped_feature_band_reads_values_only_for_the_features_it_keeps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Flips if the band reads its candidate rows' values a row at a time: 5,000 candidates under a cap of 10
        coerce the attribute values of those 10 only."""
        table, geometries = self._polygons(5_000)
        coerced: list[Any] = []
        real_coerce = coerce_attribute

        def _coerce(value: Any) -> Any:
            coerced.append(value)
            return real_coerce(value)

        monkeypatch.setattr("threetears.geo.bands.coerce_attribute", _coerce)
        result = feature_band(
            table,
            geometries,
            tile=TileId(z=12, x=100, y=200),
            spec=FeatureSpec(attributes=("id", "score"), max_features_per_tile=10, rank_column="score"),
        )
        assert len(result.features) == 10 and result.dropped == 4_990
        assert sorted(coerced[1::2]) == [96] * 10
        assert len(coerced) == 10 * 2

    def test_an_aggregate_band_reads_no_row_as_a_python_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Flips if the rollup reads its rows a row at a time: 5,000 rows in 3 groups coerce no value."""

        def _refuse(*_: Any) -> Any:
            raise AssertionError("the aggregate band read rows as Python values")

        monkeypatch.setattr("threetears.geo.bands.coerce_attribute", _refuse)
        table, geometries = self._polygons(5_000)
        result = aggregate_band(
            table,
            geometries,
            tile=TileId(z=4, x=3, y=6),
            spec=AggregateSpec(rollup_column="state", measures={"v": "sum"}),
        )
        assert sorted(f.attributes["count"] for f in result.features) == [1666, 1667, 1667]

    def test_sums_add_as_python_s_sum_does(self) -> None:
        """Flips if a measure's sum is a pairwise or plain running sum: Python's sum compensates, and the tile's bytes
        carry its exact answer."""
        values = [0.1, 1e16, 0.1, -1e16, 0.3, 1e-9, 7.7]
        rows = [{"id": n, "lon": -112.0, "lat": 33.4, "state": "04", "v": value} for n, value in enumerate(values)]
        result = aggregate_band(
            _table(rows),
            _geometries(rows),
            tile=TileId(z=4, x=3, y=6),
            spec=AggregateSpec(rollup_column="state", measures={"v": "sum"}),
        )
        # the case is one a running sum gets wrong (8.000000001, not 8.200000001)
        assert sum(values) != float(np.cumsum(values)[-1])
        assert result.features[0].attributes["v"] == sum(values)
