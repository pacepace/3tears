"""unit tests for the two zoom bands.

the cap-ordering and null-aggregation tests are the ones that matter: both
are cases where a plausible implementation silently produces a misleading
map rather than an error.
"""

from __future__ import annotations

from typing import Any

import pytest
from shapely.geometry import Point, Polygon

from threetears.geo.attributes import coerce_attributes
from threetears.geo.bands import (
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


def _rows(count: int, **extra: Any) -> list[dict[str, Any]]:
    return [{"id": n, "lon": -112.0 + n * 0.001, "lat": 33.4, **extra} for n in range(count)]


class TestSimplificationTolerance:
    def test_halves_per_zoom(self) -> None:
        assert simplification_tolerance(5) == pytest.approx(simplification_tolerance(4) / 2)

    def test_tighter_at_higher_zoom(self) -> None:
        assert simplification_tolerance(14) < simplification_tolerance(4)


class TestFeatureBand:
    def test_emits_one_feature_per_row(self) -> None:
        result = feature_band(
            _rows(3),
            tile=TileId(z=12, x=100, y=200),
            spec=FeatureSpec(attributes=("id",), feature_id_column="id"),
            geometry_of=_point_of,
            attributes_of=coerce_attributes,
        )
        assert len(result.features) == 3
        assert not result.truncated
        assert [f.feature_id for f in result.features] == [0, 1, 2]

    def test_rows_without_geometry_are_skipped_not_fatal(self) -> None:
        rows = _rows(2) + [{"id": 99, "lon": None, "lat": None}]
        result = feature_band(
            rows,
            tile=TileId(z=12, x=100, y=200),
            spec=FeatureSpec(attributes=("id",)),
            geometry_of=_point_of,
            attributes_of=coerce_attributes,
        )
        assert len(result.features) == 2

    def test_cap_keeps_the_highest_ranked(self) -> None:
        # the cap must drop by declared importance; dropping in source order
        # would keep an arbitrary subset and misrepresent a dense area.
        rows = [{"id": n, "lon": -112.0 + n * 0.001, "lat": 33.4, "score": n} for n in range(10)]
        result = feature_band(
            rows,
            tile=TileId(z=12, x=100, y=200),
            spec=FeatureSpec(attributes=("id", "score"), max_features_per_tile=3, rank_column="score"),
            geometry_of=_point_of,
            attributes_of=coerce_attributes,
        )
        assert result.truncated
        assert result.dropped == 7
        assert sorted(f.attributes["score"] for f in result.features) == [7, 8, 9]

    def test_cap_reports_truncation(self) -> None:
        # a silently truncated tile reads as "this is all the data there is".
        result = feature_band(
            _rows(50),
            tile=TileId(z=12, x=100, y=200),
            spec=FeatureSpec(attributes=("id",), max_features_per_tile=10),
            geometry_of=_point_of,
            attributes_of=coerce_attributes,
        )
        assert result.truncated
        assert result.dropped == 40

    def test_under_the_cap_is_not_marked_truncated(self) -> None:
        result = feature_band(
            _rows(5),
            tile=TileId(z=12, x=100, y=200),
            spec=FeatureSpec(attributes=("id",), max_features_per_tile=10),
            geometry_of=_point_of,
            attributes_of=coerce_attributes,
        )
        assert not result.truncated
        assert result.dropped == 0

    def test_geometry_is_simplified_at_low_zoom(self) -> None:
        # a many-vertex polygon should lose vertices the tile grid cannot
        # express anyway.
        detailed = Polygon([(-112.0 + i * 0.0001, 33.4 + (i % 2) * 0.00001) for i in range(60)] + [(-112.0, 33.4)])
        rows = [{"id": 1, "geom": detailed}]
        result = feature_band(
            rows,
            tile=TileId(z=4, x=3, y=6),
            spec=FeatureSpec(attributes=("id",)),
            geometry_of=lambda row: row["geom"],
            attributes_of=coerce_attributes,
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
            rows,
            tile=TileId(z=4, x=3, y=6),
            spec=AggregateSpec(rollup_column="state", measures={"unreg": "sum"}),
            geometry_of=_point_of,
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
            rows,
            tile=TileId(z=4, x=3, y=6),
            spec=AggregateSpec(rollup_column="state"),
            geometry_of=_point_of,
        )
        assert result.features[0].feature_id == "04"

    def test_centroid_sits_on_members_not_tile_centre(self) -> None:
        # a marker in the empty middle of a tile misrepresents where the data is.
        rows = [
            {"id": 1, "lon": -112.0, "lat": 33.4, "state": "04"},
            {"id": 2, "lon": -112.2, "lat": 33.6, "state": "04"},
        ]
        result = aggregate_band(
            rows,
            tile=TileId(z=4, x=3, y=6),
            spec=AggregateSpec(rollup_column="state"),
            geometry_of=_point_of,
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
            rows,
            tile=TileId(z=4, x=3, y=6),
            spec=AggregateSpec(rollup_column="state", measures={"approval": "mean"}),
            geometry_of=_point_of,
        )
        assert result.features[0].attributes["approval"] == pytest.approx(0.6)

    def test_all_null_measure_is_omitted_entirely(self) -> None:
        rows = [{"id": 1, "lon": -112.0, "lat": 33.4, "state": "04", "approval": None}]
        result = aggregate_band(
            rows,
            tile=TileId(z=4, x=3, y=6),
            spec=AggregateSpec(rollup_column="state", measures={"approval": "mean"}),
            geometry_of=_point_of,
        )
        assert "approval" not in result.features[0].attributes

    def test_count_measure_counts_non_null(self) -> None:
        rows = [
            {"id": 1, "lon": -112.0, "lat": 33.4, "state": "04", "approval": 0.6},
            {"id": 2, "lon": -112.1, "lat": 33.5, "state": "04", "approval": None},
        ]
        result = aggregate_band(
            rows,
            tile=TileId(z=4, x=3, y=6),
            spec=AggregateSpec(rollup_column="state", measures={"approval": "count"}),
            geometry_of=_point_of,
        )
        assert result.features[0].attributes["approval"] == 1

    def test_without_a_rollup_column_rows_group_by_tile(self) -> None:
        result = aggregate_band(
            _rows(4),
            tile=TileId(z=4, x=3, y=6),
            spec=AggregateSpec(),
            geometry_of=_point_of,
        )
        assert len(result.features) == 1
        assert result.features[0].attributes["count"] == 4

    def test_unknown_aggregation_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown aggregation"):
            aggregate_band(
                _rows(2, state="04", v=1),
                tile=TileId(z=4, x=3, y=6),
                spec=AggregateSpec(rollup_column="state", measures={"v": "median"}),
                geometry_of=_point_of,
            )
