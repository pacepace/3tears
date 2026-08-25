"""declarative geo layers on a datasource.

every mapping application is the same pipeline: a table holds geometry and
attributes, and a client wants tiles. what varies is which table, which
column carries the geometry, which attributes ride along, and how
aggressively to reduce at low zoom. that is configuration, not code -- so it
is declared here, in the datasource definition the product already writes,
and the product writes no map plumbing at all.

this lives beside :class:`~threetears.datasources.config.DatasourceConfig`
rather than in ``3tears-geo`` on purpose. the *declaration* is part of what a
datasource is, and this package is the single source of truth for that; the
*production* it drives lives in ``3tears-geo``, which this package does not
depend on. the SDK validates YAML against these models without pulling in
shapely.

sensitivity is deliberately absent as a free-standing field. a datasource
already records how exposed it is (``visibility`` alongside a nullable
``customer_id``), and a second place to say the same thing is a second place
for it to be wrong. a layer may only *narrow* what it inherits.

``CacheClassConfig`` is that narrow-only vocabulary. it no longer lives here:
a second consumer arrived (the inbound REST affordance a tool may declare)
and neither package may import the other, so the enum was promoted to
:mod:`threetears.core.http_cache` as ``CacheClass`` and is re-exported below
under the name this package has always used. one enum object, two names.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from threetears.core.http_cache import CacheClass as CacheClassConfig

__all__ = [
    "AggregateBandConfig",
    "CacheClassConfig",
    "FeatureBandConfig",
    "GeoConfig",
    "GeoLayerConfig",
    "GeometryConfig",
    "GeometryKind",
    "MeasureAggregation",
]


class GeometryKind(StrEnum):
    """how a layer's geometry is stored."""

    #: a WKB/EWKB column, the usual shape for polygons
    WKB = "wkb"
    #: a longitude/latitude column pair, the usual shape for points
    LONLAT = "lonlat"


class MeasureAggregation(StrEnum):
    """how a measure combines when rows roll up."""

    SUM = "sum"
    MEAN = "mean"
    COUNT = "count"


class GeometryConfig(BaseModel):
    """where a layer's geometry lives.

    :param kind: storage shape
    :ptype kind: GeometryKind
    :param column: WKB/EWKB column, required when ``kind`` is ``wkb``
    :ptype column: str | None
    :param longitude: longitude column, required when ``kind`` is ``lonlat``
    :ptype longitude: str | None
    :param latitude: latitude column, required when ``kind`` is ``lonlat``
    :ptype latitude: str | None
    """

    kind: GeometryKind
    column: str | None = None
    longitude: str | None = None
    latitude: str | None = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def columns_match_kind(self) -> GeometryConfig:
        """require the columns the declared kind actually needs.

        a layer whose geometry cannot be read produces empty tiles forever,
        and an empty tile is indistinguishable from empty geography once it
        reaches a map -- so this fails at load rather than at render.

        :return: the validated config
        :rtype: GeometryConfig
        :raises ValueError: when the declared kind's columns are missing
        """
        if self.kind is GeometryKind.WKB and not self.column:
            raise ValueError("geometry kind 'wkb' requires 'column'")
        if self.kind is GeometryKind.LONLAT and not (self.longitude and self.latitude):
            raise ValueError("geometry kind 'lonlat' requires both 'longitude' and 'latitude'")
        return self


class AggregateBandConfig(BaseModel):
    """the low-zoom band, where rows roll up rather than being sampled.

    :param rollup_by: coarser-geography column rows group into (state FIPS,
        county id). absent groups by the tile itself, the fallback for a
        layer with no declared coarser geography
    :ptype rollup_by: str | None
    :param measures: column name to aggregation
    :ptype measures: dict[str, MeasureAggregation]
    """

    rollup_by: str | None = None
    measures: dict[str, MeasureAggregation] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class FeatureBandConfig(BaseModel):
    """the high-zoom band, where individual features are drawn.

    :param attributes: columns carried onto each feature
    :ptype attributes: list[str]
    :param max_features_per_tile: hard cap. an uncapped tile in a dense
        metro reaches tens of megabytes, exceeds the L2 payload ceiling and
        renders badly
    :ptype max_features_per_tile: int
    :param rank_by: column ranking features for the cap, descending. absent
        means the cap drops in whatever order the source returned, which is
        arbitrary -- declare one wherever the cap is likely to bind
    :ptype rank_by: str | None
    :param simplify_tolerance: override for the per-zoom simplification
        tolerance in degrees at zoom 0; absent derives it
    :ptype simplify_tolerance: float | None
    """

    attributes: list[str] = Field(default_factory=list)
    max_features_per_tile: int = Field(default=4000, gt=0)
    rank_by: str | None = None
    simplify_tolerance: float | None = Field(default=None, gt=0)

    model_config = ConfigDict(extra="forbid")


class GeoLayerConfig(BaseModel):
    """one tileable layer on a datasource.

    :param name: layer name, unique within the datasource; appears in the
        tile URL and as the MVT layer name
    :ptype name: str
    :param table: source table
    :ptype table: str
    :param feature_id: column holding each feature's stable identity
    :ptype feature_id: str
    :param geometry: where the geometry lives
    :ptype geometry: GeometryConfig
    :param cache: how far the layer's tiles may travel
    :ptype cache: CacheClassConfig
    :param minzoom: lowest zoom served
    :ptype minzoom: int
    :param maxzoom: highest zoom served
    :ptype maxzoom: int
    :param crossover_zoom: highest zoom served by the aggregate band; above
        it individual features are drawn
    :ptype crossover_zoom: int
    :param aggregate: the low-zoom band
    :ptype aggregate: AggregateBandConfig
    :param features: the high-zoom band
    :ptype features: FeatureBandConfig
    :param version_column: column stamping which source generation a row
        belongs to, so a build reads one coherent vintage and a tile built
        during a reseed is not a torn read
    :ptype version_column: str
    :param bbox_columns: the four materialised bounding-box columns. without
        PostGIS there is no spatial index, so a tile build's "features in
        this rectangle" query would otherwise be a full scan per tile
    :ptype bbox_columns: tuple[str, str, str, str]
    """

    name: str
    table: str
    feature_id: str
    geometry: GeometryConfig
    cache: CacheClassConfig = CacheClassConfig.INHERIT
    minzoom: int = Field(default=0, ge=0, le=24)
    maxzoom: int = Field(default=16, ge=0, le=24)
    crossover_zoom: int = Field(default=10, ge=0, le=24)
    aggregate: AggregateBandConfig = Field(default_factory=AggregateBandConfig)
    features: FeatureBandConfig = Field(default_factory=FeatureBandConfig)
    version_column: str = "source_version"
    bbox_columns: tuple[str, str, str, str] = ("bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy")

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def zooms_are_coherent(self) -> GeoLayerConfig:
        """require a usable zoom range with the crossover inside it.

        a crossover outside the range silently disables one band: below the
        range's floor the layer is all features, above its ceiling all
        rollups. either is a map that quietly does not do what was declared.

        :return: the validated config
        :rtype: GeoLayerConfig
        :raises ValueError: when the zooms cannot describe two bands
        """
        if self.minzoom > self.maxzoom:
            raise ValueError(f"layer {self.name!r}: minzoom {self.minzoom} exceeds maxzoom {self.maxzoom}")
        if not (self.minzoom <= self.crossover_zoom <= self.maxzoom):
            raise ValueError(
                f"layer {self.name!r}: crossover_zoom {self.crossover_zoom} is outside "
                f"the served range {self.minzoom}-{self.maxzoom}, which would disable a band"
            )
        return self

    @model_validator(mode="after")
    def declared_columns_are_distinct(self) -> GeoLayerConfig:
        """the four bbox columns must name four different columns.

        a duplicate collapses the rectangle onto a line or a point, so the
        build's overlap query silently matches almost nothing.

        :return: the validated config
        :rtype: GeoLayerConfig
        :raises ValueError: when bbox columns repeat
        """
        if len(set(self.bbox_columns)) != len(self.bbox_columns):
            raise ValueError(f"layer {self.name!r}: bbox_columns must name four distinct columns")
        return self


class GeoConfig(BaseModel):
    """the ``geo:`` block on a datasource definition.

    :param layers: tileable layers
    :ptype layers: list[GeoLayerConfig]
    """

    layers: list[GeoLayerConfig] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def layer_names_are_unique(self) -> GeoConfig:
        """layer names must be unique within a datasource.

        the name is a path segment in every tile URL and the MVT layer name,
        so a duplicate makes one of the two unaddressable.

        :return: the validated config
        :rtype: GeoConfig
        :raises ValueError: when two layers share a name
        """
        names = [layer.name for layer in self.layers]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"duplicate geo layer name(s): {', '.join(sorted(duplicates))}")
        return self

    def layer(self, name: str) -> GeoLayerConfig | None:
        """return a declared layer by name.

        :param name: layer name
        :ptype name: str
        :return: the layer, or ``None`` when undeclared
        :rtype: GeoLayerConfig | None
        """
        return next((layer for layer in self.layers if layer.name == name), None)

    def model_dump_yaml_safe(self) -> dict[str, Any]:
        """dump in the shape the datasource YAML uses.

        :return: plain dict with enums as their string values
        :rtype: dict[str, Any]
        """
        return self.model_dump(mode="json", exclude_defaults=True)
