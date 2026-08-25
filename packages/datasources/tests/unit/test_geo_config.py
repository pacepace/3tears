"""tests for the declarative geo layer block.

these validators exist to turn silent map defects into load-time failures.
almost every one guards a case that renders *something* rather than raising
-- an empty layer, a map that is all rollups, an overlap query matching
nothing -- which is why they are validators and not documentation.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from threetears.datasources.config import DatasourceConfig
from threetears.datasources.geo_config import (
    CacheClassConfig,
    GeoConfig,
    GeoLayerConfig,
    GeometryConfig,
    GeometryKind,
    MeasureAggregation,
)


def _layer(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "name": "census_tracts",
        "table": "census_tracts",
        "feature_id": "geoid",
        "geometry": {"kind": "wkb", "column": "geometry_wkb"},
    }
    base.update(overrides)
    return base


class TestGeometryDeclaration:
    def test_wkb_requires_a_column(self) -> None:
        # a layer whose geometry cannot be read produces empty tiles forever,
        # and an empty tile looks exactly like empty geography on a map.
        with pytest.raises(ValidationError, match="requires 'column'"):
            GeometryConfig(kind=GeometryKind.WKB)

    def test_lonlat_requires_both_coordinates(self) -> None:
        with pytest.raises(ValidationError, match="requires both"):
            GeometryConfig(kind=GeometryKind.LONLAT, longitude="lon")

    def test_wkb_shape_is_accepted(self) -> None:
        geometry = GeometryConfig(kind=GeometryKind.WKB, column="geometry_wkb")
        assert geometry.column == "geometry_wkb"

    def test_lonlat_shape_is_accepted(self) -> None:
        geometry = GeometryConfig(kind=GeometryKind.LONLAT, longitude="longitude", latitude="latitude")
        assert (geometry.longitude, geometry.latitude) == ("longitude", "latitude")


class TestZoomCoherence:
    def test_inverted_range_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="exceeds maxzoom"):
            GeoLayerConfig(**_layer(minzoom=12, maxzoom=4))  # type: ignore[arg-type]

    def test_crossover_below_the_range_is_rejected(self) -> None:
        # a crossover under the floor makes the layer all features, silently
        # disabling the rollup band that low zoom depends on.
        with pytest.raises(ValidationError, match="outside the served range"):
            GeoLayerConfig(**_layer(minzoom=6, maxzoom=14, crossover_zoom=3))  # type: ignore[arg-type]

    def test_crossover_above_the_range_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="outside the served range"):
            GeoLayerConfig(**_layer(minzoom=6, maxzoom=14, crossover_zoom=20))  # type: ignore[arg-type]

    def test_crossover_at_a_boundary_is_allowed(self) -> None:
        layer = GeoLayerConfig(**_layer(minzoom=4, maxzoom=14, crossover_zoom=14))  # type: ignore[arg-type]
        assert layer.crossover_zoom == 14


class TestBboxColumns:
    def test_defaults_match_the_migration(self) -> None:
        layer = GeoLayerConfig(**_layer())  # type: ignore[arg-type]
        assert layer.bbox_columns == ("bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy")

    def test_duplicate_columns_are_rejected(self) -> None:
        # a duplicate collapses the rectangle onto a line or a point, so the
        # build's overlap query silently matches almost nothing.
        with pytest.raises(ValidationError, match="four distinct columns"):
            GeoLayerConfig(**_layer(bbox_columns=("a", "a", "b", "c")))  # type: ignore[arg-type]


class TestBands:
    def test_measures_take_declared_aggregations(self) -> None:
        layer = GeoLayerConfig(
            **_layer(aggregate={"rollup_by": "state_fips", "measures": {"total_unreg": "sum"}})  # type: ignore[arg-type]
        )
        assert layer.aggregate.measures["total_unreg"] is MeasureAggregation.SUM

    def test_unknown_aggregation_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GeoLayerConfig(**_layer(aggregate={"measures": {"x": "median"}}))  # type: ignore[arg-type]

    def test_feature_cap_must_be_positive(self) -> None:
        # a zero cap would emit empty tiles everywhere.
        with pytest.raises(ValidationError):
            GeoLayerConfig(**_layer(features={"max_features_per_tile": 0}))  # type: ignore[arg-type]

    def test_cap_defaults_to_a_renderable_size(self) -> None:
        assert GeoLayerConfig(**_layer()).features.max_features_per_tile == 4000  # type: ignore[arg-type]


class TestCacheClass:
    def test_defaults_to_inherit(self) -> None:
        # sensitivity is stated once, on the datasource. inheriting keeps it
        # that way; a default of anything else would be a second source of
        # truth with a guess in it.
        assert GeoLayerConfig(**_layer()).cache is CacheClassConfig.INHERIT  # type: ignore[arg-type]

    def test_a_layer_may_declare_private(self) -> None:
        layer = GeoLayerConfig(**_layer(cache="private"))  # type: ignore[arg-type]
        assert layer.cache is CacheClassConfig.PRIVATE

    def test_the_name_this_package_exports_is_the_promoted_enum_itself(self) -> None:
        # promoted to threetears.core.http_cache when a second consumer (the
        # tool REST affordance) arrived. this asserts identity rather than
        # equality: a copy would be a second vocabulary, which is the thing
        # the promotion exists to prevent.
        from threetears.core.http_cache import CacheClass

        assert CacheClassConfig is CacheClass


class TestGeoBlock:
    def test_duplicate_layer_names_are_rejected(self) -> None:
        # the name is a path segment in every tile URL and the MVT layer
        # name, so a duplicate makes one of the two unaddressable.
        with pytest.raises(ValidationError, match="duplicate geo layer name"):
            GeoConfig(layers=[GeoLayerConfig(**_layer()), GeoLayerConfig(**_layer())])  # type: ignore[arg-type]

    def test_lookup_by_name(self) -> None:
        block = GeoConfig(layers=[GeoLayerConfig(**_layer())])  # type: ignore[arg-type]
        assert block.layer("census_tracts") is not None
        assert block.layer("nope") is None

    def test_typos_are_rejected_rather_than_ignored(self) -> None:
        # extra="forbid" throughout: a silently-ignored key ships the default
        # and the author never learns their declaration did nothing.
        with pytest.raises(ValidationError):
            GeoLayerConfig(**_layer(crossver_zoom=8))  # type: ignore[arg-type]


class TestDatasourceIntegration:
    def test_a_datasource_without_geo_is_unchanged(self) -> None:
        # the overwhelming majority of datasources are not geographic and
        # must not have to say so.
        config = DatasourceConfig(name="reporting", access_mode="read")
        assert config.geo is None

    def test_geo_parses_from_the_datasource_yaml_shape(self) -> None:
        config = DatasourceConfig(
            name="aibotsmap-data",
            access_mode="read",
            schemas=["aibots_map"],
            geo={  # type: ignore[arg-type]
                "layers": [
                    {
                        "name": "census_tracts",
                        "table": "census_tracts",
                        "feature_id": "geoid",
                        "cache": "public",
                        "geometry": {"kind": "wkb", "column": "geometry_wkb"},
                        "minzoom": 4,
                        "maxzoom": 14,
                        "crossover_zoom": 9,
                        "aggregate": {"rollup_by": "state_fips", "measures": {"total_unreg": "sum"}},
                        "features": {"attributes": ["total_unreg"], "rank_by": "total_unreg"},
                    },
                    {
                        "name": "locations",
                        "table": "locations",
                        "feature_id": "id",
                        "cache": "private",
                        "geometry": {"kind": "lonlat", "longitude": "longitude", "latitude": "latitude"},
                    },
                ]
            },
        )
        assert config.geo is not None
        tracts = config.geo.layer("census_tracts")
        locations = config.geo.layer("locations")
        assert tracts is not None and locations is not None
        assert tracts.cache is CacheClassConfig.PUBLIC
        assert locations.cache is CacheClassConfig.PRIVATE
        assert locations.geometry.kind is GeometryKind.LONLAT
