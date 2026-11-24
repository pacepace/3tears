"""the tile collection: quantized key, computed value, object-store durable tier.

this is where the pieces meet. :class:`DerivedCollection` supplies the
quantization contract and the two single-flight gates;
:mod:`threetears.geo.bands` and :mod:`threetears.geo.mvt` supply the
production; the object store supplies the durable tier that outlives any one
pod and that a CDN can ultimately be pointed at.

the key is ``(layer, version, z, x, y)``. every part of it is load-bearing:

- ``z/x/y`` is the quantization -- the thing that turns an unbounded space of
  viewports into a bounded space of addresses, which is what makes any of
  this cacheable across pods at all.
- ``version`` makes the value immutable. a tile is a pure function of its
  key, so a rebuild is a *new address* rather than an invalidation, and the
  superseded generation ages out of downstream caches without anyone purging
  anything.
- ``layer`` keeps two layers of one datasource from sharing a cache entry.

because the value is immutable, a build must read a single coherent
generation of source data. that is what the ``source_version`` stamp on the
rows is for, and why the version is in the key rather than beside it: a tile
built from generation N is a different artifact from one built from N+1, and
conflating them would let a rebuild silently serve a mixture.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Final

import pyarrow as pa

from threetears.core.collections.derived import DerivedCollection
from threetears.core.entities.base import BaseEntity
from threetears.geo.bands import AggregateSpec, BandResult, FeatureSpec, aggregate_band, feature_band
from threetears.geo.features import FeatureCache
from threetears.geo.geometry import (
    RowGeometries,
    decode_geometry,
    point_geometries,
    point_geometry,
    wkb_geometries,
)
from threetears.geo.mvt import encode_tile
from threetears.geo.scope import check_cache_scope
from threetears.geo.tiles import BoundingBox, TileId, tile_bounds, tile_for_point
from threetears.geo.workers import INLINE_ROWS, GeometryColumns, mapped_rows, ran_inline, rows_of, run_over_rows
from threetears.interfaces.media.keys import build_object_key
from threetears.observe import counter, get_logger, traced
from shapely.geometry.base import BaseGeometry

__all__ = ["LayerDefinition", "TileEntity", "TileCollection", "ViewportRequest"]

log = get_logger(__name__)

_MVT_CONTENT_TYPE = "application/vnd.mapbox-vector-tile"

#: object-store error codes meaning "no such object". checked by string
#: rather than by exception class so this package does not take a dependency
#: on botocore purely to name one error -- the S3 backend is an optional
#: extra, and a filesystem or in-memory store must work without it.
_MISSING_OBJECT_CODES = frozenset({"NoSuchKey", "NotFound", "404"})


def _is_missing_object(exc: BaseException) -> bool:
    """true when ``exc`` means the object is absent rather than unreachable.

    the distinction matters more than it looks: absent means build it,
    unreachable means stop. conflating them turns an outage into a rebuild
    storm aimed at the component that is already failing.

    :param exc: exception raised by an object-store read
    :ptype exc: BaseException
    :return: whether the object is simply not there
    :rtype: bool
    """
    if isinstance(exc, FileNotFoundError):
        return True
    # botocore's ClientError shape, matched structurally so no import is needed
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = str(response.get("Error", {}).get("Code", ""))
        status = str(response.get("ResponseMetadata", {}).get("HTTPStatusCode", ""))
        return code in _MISSING_OBJECT_CODES or status == "404"
    return False


@dataclass(frozen=True, slots=True)
class ViewportRequest:
    """what a caller actually has: a place, a zoom, and a layer.

    quantized into a tile key by :meth:`TileCollection.derive_key`, so no
    caller ever computes a tile address itself and no two callers can
    disagree about which tile a point belongs to.
    """

    layer: str
    longitude: float
    latitude: float
    zoom: int
    version: int


@dataclass(frozen=True, slots=True)
class LayerDefinition:
    """the declared shape of one geo layer.

    :param name: layer name, unique within its datasource
    :ptype name: str
    :param feature_id_column: column holding the stable feature identity;
        becomes the MVT feature id, and is the join key a client uses to bind
        volatile values to static geometry
    :ptype feature_id_column: str
    :param geometry_column: column holding WKB/EWKB geometry, when the layer
        is polygon-shaped
    :ptype geometry_column: str | None
    :param longitude_column: lon column, when the layer is point-shaped
    :ptype longitude_column: str | None
    :param latitude_column: lat column, when the layer is point-shaped
    :ptype latitude_column: str | None
    :param aggregate: rollup band spec, used at and below ``crossover_zoom``
    :ptype aggregate: AggregateSpec
    :param features: feature band spec, used above ``crossover_zoom``
    :ptype features: FeatureSpec
    :param crossover_zoom: highest zoom served by the aggregate band
    :ptype crossover_zoom: int
    :param minzoom: lowest zoom this layer serves
    :ptype minzoom: int
    :param maxzoom: highest zoom this layer serves
    :ptype maxzoom: int
    """

    name: str
    feature_id_column: str
    geometry_column: str | None = None
    longitude_column: str | None = None
    latitude_column: str | None = None
    aggregate: AggregateSpec = AggregateSpec()
    features: FeatureSpec = FeatureSpec()
    crossover_zoom: int = 10
    minzoom: int = 0
    maxzoom: int = 16

    def __post_init__(self) -> None:
        has_wkb = self.geometry_column is not None
        has_point = self.longitude_column is not None and self.latitude_column is not None
        if not has_wkb and not has_point:
            raise ValueError(
                f"layer {self.name!r} declares no geometry: supply geometry_column, "
                "or both longitude_column and latitude_column"
            )

    def geometry_of(self, row: dict[str, Any]) -> BaseGeometry | None:
        """extract this layer's geometry from a source row.

        the one place the two declared geometry shapes -- a WKB blob and a
        lon/lat pair -- converge, so every caller downstream handles one kind
        of thing.

        :param row: source row
        :ptype row: dict[str, Any]
        :return: geometry, or ``None`` when the row carries none
        :rtype: BaseGeometry | None
        """
        if self.geometry_column is not None:
            return decode_geometry(row.get(self.geometry_column))
        assert self.longitude_column is not None and self.latitude_column is not None  # noqa: S101 - __post_init__
        return point_geometry(row, self.longitude_column, self.latitude_column)

    def geometries_of(self, table: pa.Table) -> RowGeometries:
        """every row's geometry, from a whole table at once: :meth:`geometry_of` over columns.

        :param table: source rows
        :ptype table: pa.Table
        :return: each row's geometry, and which rows have one that could not be read
        :rtype: RowGeometries
        """
        if self.geometry_column is not None:
            return wkb_geometries(table, self.geometry_column)
        assert self.longitude_column is not None and self.latitude_column is not None  # noqa: S101 - __post_init__
        return point_geometries(table, self.longitude_column, self.latitude_column)

    @property
    def bounds_of(self) -> GeometryColumns:
        """this layer's :data:`~threetears.geo.features.BoundsOf`, for its
        :class:`~threetears.geo.features.FeatureCache`: every row's bounding rectangle, from a whole table at once,
        null where the row's geometry is absent, empty or will not decode. plain data, so the cache reads it in a
        worker process.

        :return: this layer's geometry columns
        :rtype: GeometryColumns
        """
        return GeometryColumns(self.geometry_column, self.longitude_column, self.latitude_column)

    def columns(self) -> tuple[str, ...]:
        """every source column a build of this layer reads, once each, in declaration order.

        :return: the column names
        :rtype: tuple[str, ...]
        """
        named = (
            self.feature_id_column,
            self.geometry_column,
            self.longitude_column,
            self.latitude_column,
            self.aggregate.rollup_column,
            *self.aggregate.measures,
            *self.features.attributes,
            self.features.feature_id_column,
            self.features.rank_column,
        )
        return tuple(dict.fromkeys(name for name in named if name is not None))


class TileEntity(BaseEntity):
    """one built tile.

    the tile's key is the 5-tuple ``(layer, version, z, x, y)`` declared
    on :class:`TileCollection`; :class:`BaseEntity` derives the
    addressing ``_id`` from that declaration, and ``addressing_id``
    exposes it.

    ``primary_key_field`` names ``z`` rather than ``layer``: it selects
    what the scalar ``.id`` surfaces, and a layer NAME identifies a whole
    layer rather than a row, so it was the one component guaranteed to be
    wrong. no component of a 5-part key is a row identity on its own --
    read ``addressing_id`` when identity is what is wanted.
    """

    primary_key_field = "z"


#: fetches source rows for a layer, generation and rectangle, as one Arrow table: the same
#: callable as :data:`~threetears.geo.features.FeatureLoader`. the collection does not own the
#: query because only the caller knows the datasource.
SourceLoader = Callable[[str, int, BoundingBox], Awaitable[pa.Table]]


class TileCollection(DerivedCollection[TileEntity]):
    """three-tier collection over built MVT tiles.

    :param layers: declared layers, keyed by name
    :ptype layers: dict[str, LayerDefinition]
    :param loader: async callable returning source rows for a rectangle
    :ptype loader: SourceLoader
    :param object_store: durable tier for built tiles
    :ptype object_store: Any
    :param datasource_name: names the object-key namespace, so two
        datasources declaring a layer of the same name cannot collide
    :ptype datasource_name: str
    :param cache_scope: the tile source's cache identity: names the collection's table, and
        through it the L1 table, the L2 keys, the build locks and the registry entry, so a
        layer of the same name from another source is never served from this one's caches
        (see :mod:`threetears.geo.scope`)
    :ptype cache_scope: str
    :param feature_caches: per-layer source-feature caches. omitted means
        every build reads L3 directly: correct, just slower
    :ptype feature_caches: dict[str, FeatureCache] | None
    """

    primary_key_column: tuple[str, ...] = ("layer", "version", "z", "x", "y")

    #: its own bucket rather than the shared default: a tile build is
    #: seconds of CPU, far longer than the config-reload work the default
    #: bucket's TTL was sized for, and a bucket pins one TTL for every key.
    build_lock_bucket: ClassVar[str] = "geo-tile-build-locks"

    def __init__(
        self,
        *args: Any,
        layers: dict[str, LayerDefinition],
        loader: SourceLoader,
        object_store: Any,
        datasource_name: str,
        cache_scope: str,
        feature_caches: dict[str, FeatureCache] | None = None,
        **kwargs: Any,
    ) -> None:
        # before the base constructor: it reads table_name to resolve and register every tier
        self._cache_scope = check_cache_scope(cache_scope)
        super().__init__(*args, **kwargs)
        self._layers = layers
        self._loader = loader
        self._object_store = object_store
        self._datasource_name = datasource_name
        # optional per-layer feature caches. absent is a supported
        # configuration -- builds then read L3 per tile, which is correct and
        # only slower -- so a deployment without an L1 backend still works. A read-only copy: the map never changes
        # after construction.
        self._feature_caches = MappingProxyType(dict(feature_caches or {}))

    @property
    def table_name(self) -> str:
        return f"geo_tiles_{self._cache_scope}"

    @property
    def entity_class(self) -> type[TileEntity]:
        return TileEntity

    def layer(self, name: str) -> LayerDefinition | None:
        """return a declared layer, or ``None``.

        :param name: layer name
        :ptype name: str
        :return: the layer definition, or ``None`` if undeclared
        :rtype: LayerDefinition | None
        """
        return self._layers.get(name)

    # ------------------------------------------------------------------
    # quantization
    # ------------------------------------------------------------------

    def derive_key(self, request: Any) -> Any:
        """quantize a viewport onto a tile address.

        :param request: the viewport being looked at
        :ptype request: ViewportRequest
        :return: ``(layer, version, z, x, y)``
        :rtype: tuple[str, int, int, int, int]
        """
        viewport: ViewportRequest = request
        tile = tile_for_point(viewport.longitude, viewport.latitude, viewport.zoom)
        return (viewport.layer, viewport.version, tile.z, tile.x, tile.y)

    def object_key(self, entity_id: Any) -> str:
        """object-store key for one tile, derivable from its address alone.

        that derivability is the requirement: a reader fetching a tile has
        only the URL, and cannot consult a database to translate it into an
        opaque object id.

        :param entity_id: ``(layer, version, z, x, y)``
        :ptype entity_id: Any
        :return: object key
        :rtype: str
        """
        layer, version, z, x, y = self.normalize_pk(entity_id)
        return build_object_key(
            customer_id=None,
            scope="tiles",
            category=self._datasource_name,
            path=f"{layer}/v{version}/{z}/{x}/{y}.mvt",
        )

    # ------------------------------------------------------------------
    # durable tier
    # ------------------------------------------------------------------

    async def load_derived(self, entity_id: Any) -> dict[str, Any] | None:
        """read an already-built tile from the object store.

        an absent object is a miss and returns ``None``. **every other
        failure propagates**, deliberately: a credentials error, a network
        partition or a permissions change would otherwise be indistinguishable
        from "not built yet", and this collection's answer to that is to
        rebuild. a store that is failing every read would then be asked to
        absorb a rebuild of every tile requested, at exactly the moment it is
        least able to, and the underlying fault would never surface.
        """
        key = self.object_key(entity_id)
        try:
            chunks = [chunk async for chunk in self._object_store.open_read(key)]
        except Exception as exc:
            if _is_missing_object(exc):
                return None
            raise
        layer, version, z, x, y = self.normalize_pk(entity_id)
        return {"layer": layer, "version": version, "z": z, "x": x, "y": y, "mvt": b"".join(chunks)}

    async def save_to_store(self, data: dict[str, Any], original_timestamp: Any = None, *, conn: Any = None) -> int:
        """write a built tile to the object store."""
        payload: bytes = data["mvt"]
        key = self.object_key((data["layer"], data["version"], data["z"], data["x"], data["y"]))

        async def _body() -> AsyncIterator[bytes]:
            yield payload

        await self._object_store.put(key, _body(), content_type=_MVT_CONTENT_TYPE, size=len(payload))
        return 1

    async def delete_from_store(self, entity_id: Any) -> None:
        """drop one tile artifact, for generation reclamation."""
        await self._object_store.delete(self.object_key(entity_id))

    # ------------------------------------------------------------------
    # production
    # ------------------------------------------------------------------

    @traced
    async def compute(self, entity_id: Any) -> dict[str, Any] | None:
        """build one tile from source features.

        returns ``None`` for an undeclared layer or an out-of-range zoom.
        that is a miss rather than an error: a client asking for a zoom the
        layer does not serve should get nothing to draw, not a failure.

        a tile built over source rows whose geometry could not be read, in either band, carries
        the layer's :data:`~threetears.geo.mvt.PARTIAL_LAYER_SUFFIX` companion layer saying how
        many, besides the counter and the warning.
        """
        layer_name, version, z, x, y = self.normalize_pk(entity_id)
        definition = self._layers.get(layer_name)
        if definition is None:
            log.warning("tile requested for undeclared layer %r", layer_name)
            return None
        if not (definition.minzoom <= z <= definition.maxzoom):
            return None

        tile = TileId(z=z, x=x, y=y)
        rows = await self._features_for(definition, version, tile)
        # the build is CPU (decode, simplify, project, encode), so it runs in a worker process over a shared-memory
        # hand-off of the rows and the loop only waits; a tile of a few rows builds here, where the round trip would
        # cost the loop more than the build (threetears.geo.workers)
        rows = rows.select([name for name in definition.columns() if name in rows.column_names])
        if rows.num_rows <= INLINE_ROWS:
            ran_inline("tile")
            answer = _build(rows, definition, tile)
        else:
            answer = await run_over_rows(_build_tile, rows, _layer_plain(definition), z, x, y, work="tile")
        payload, band = _built(answer)

        if band.undecodable:
            counter(
                "geo_undecodable_rows_total",
                description="source rows left out of a tile because their geometry could not be read",
                label_names=("layer",),
            ).labels(layer=layer_name).inc(band.undecodable)
            log.warning(
                "tile %s/%s left out %d source rows whose geometry could not be read (first: %r)",
                layer_name,
                tile,
                band.undecodable,
                band.first_undecodable,
                extra={
                    "extra_data": {
                        "layer": layer_name,
                        "undecodable": band.undecodable,
                        "first_undecodable": str(band.first_undecodable),
                    }
                },
            )
        if band.truncated:
            log.info(
                "tile %s/%s dropped %d features to the per-tile cap",
                layer_name,
                tile,
                band.dropped,
            )
        return {"layer": layer_name, "version": version, "z": z, "x": x, "y": y, "mvt": payload}

    async def _features_for(self, definition: LayerDefinition, version: int, tile: TileId) -> pa.Table:
        """source rows for one tile, through the per-pod feature cache.

        the cache loads coarse chunks and tracks which it has fully covered,
        so a run of neighbouring tiles -- which overlap almost entirely in
        source features -- pays one L3 read between them rather than one
        each. without a cache bound, this falls straight through to the
        loader, which is correct and merely slower.
        """
        cache = self._feature_caches.get(definition.name)
        if cache is None:
            return await self._loader(definition.name, version, tile_bounds(tile))
        return await cache.features_in_bbox(definition.name, version, tile_bounds(tile))

    # ------------------------------------------------------------------
    # serialization for L1 / L2
    # ------------------------------------------------------------------

    def serialize(self, data: dict[str, Any]) -> bytes:
        """encode a tile row for L2.

        the MVT payload is already bytes and is the overwhelming majority of
        the row, so it is stored raw with a short header rather than
        base64-wrapped inside JSON -- which would inflate every cached tile
        by a third for no benefit.
        """
        header = f"{data['layer']}\x1f{data['version']}\x1f{data['z']}\x1f{data['x']}\x1f{data['y']}\x1f".encode()
        payload: bytes = data["mvt"]
        return header + payload

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """decode a tile row from L2.

        splits on the first five separators only. MVT is protobuf and may
        contain the separator byte anywhere, so an unbounded split would
        truncate the payload at whatever byte happened to match.
        """
        parts = data.split(b"\x1f", 5)
        return {
            "layer": parts[0].decode(),
            "version": int(parts[1]),
            "z": int(parts[2]),
            "x": int(parts[3]),
            "y": int(parts[4]),
            "mvt": parts[5],
        }


# ----------------------------------------------------------------------
# the build, as a worker process runs it: plain data in and out
# ----------------------------------------------------------------------


def _layer_plain(definition: LayerDefinition) -> tuple[Any, ...]:
    """a layer as plain data, for :func:`_layer_of` in a worker."""
    aggregate, features = definition.aggregate, definition.features
    return (
        definition.name,
        definition.feature_id_column,
        definition.geometry_column,
        definition.longitude_column,
        definition.latitude_column,
        (aggregate.rollup_column, tuple(aggregate.measures.items())),
        (features.attributes, features.feature_id_column, features.max_features_per_tile, features.rank_column),
        definition.crossover_zoom,
        definition.minzoom,
        definition.maxzoom,
    )


def _layer_of(plain: tuple[Any, ...]) -> LayerDefinition:
    """the layer :func:`_layer_plain` made plain."""
    name, feature_id, geometry, longitude, latitude, aggregate, features, crossover, minzoom, maxzoom = plain
    rollup, measures = aggregate
    attributes, features_id, cap, rank = features
    return LayerDefinition(
        name=name,
        feature_id_column=feature_id,
        geometry_column=geometry,
        longitude_column=longitude,
        latitude_column=latitude,
        aggregate=AggregateSpec(rollup_column=rollup, measures=dict(measures)),
        features=FeatureSpec(
            attributes=tuple(attributes), feature_id_column=features_id, max_features_per_tile=cap, rank_column=rank
        ),
        crossover_zoom=crossover,
        minzoom=minzoom,
        maxzoom=maxzoom,
    )


#: the values that cross back from a worker as they are; any other first-undecodable id crosses as its text
_PLAIN_IDS: Final = (str, int, float, bool, bytes, type(None))


def _build_tile(
    handoff: tuple[str, str, tuple[str, ...], int], layer: tuple[Any, ...], z: int, x: int, y: int
) -> tuple[Any, ...]:
    """build one tile in a worker process, from its rows mapped out of the shared-memory hand-off.

    :param handoff: the rows' hand-off (:func:`~threetears.geo.workers.run_over_rows`)
    :ptype handoff: tuple[str, str, tuple[str, ...], int]
    :param layer: the layer, as :func:`_layer_plain` made it
    :ptype layer: tuple[Any, ...]
    :return: as :func:`_build`
    :rtype: tuple[Any, ...]
    """
    with mapped_rows(handoff) as mapped:
        return _build(rows_of(mapped), _layer_of(layer), TileId(z=z, x=x, y=y))


def _build(table: pa.Table, definition: LayerDefinition, tile: TileId) -> tuple[Any, ...]:
    """build one tile from its source rows: the bands over whole columns, then the encoding.

    below the crossover a tile spans too much ground for individual
    features to mean anything, so it carries rollups with real totals
    instead of an arbitrary surviving sample.

    :param table: the tile's source rows
    :ptype table: pa.Table
    :param definition: the layer
    :ptype definition: LayerDefinition
    :param tile: the tile
    :ptype tile: TileId
    :return: ``(mvt, truncated, dropped, undecodable, first_undecodable)``: plain data
    :rtype: tuple[Any, ...]
    """
    geometries = definition.geometries_of(table)
    if tile.z <= definition.crossover_zoom:
        band = aggregate_band(table, geometries, tile=tile, spec=definition.aggregate)
    else:
        band = feature_band(table, geometries, tile=tile, spec=definition.features)
    # both bands meet here, so this is where a tile built over unreadable rows is marked:
    # the per-bucket property covers only buckets that kept a feature
    payload = encode_tile({definition.name: band.features}, tile, undecodable={definition.name: band.undecodable})
    first = band.first_undecodable if isinstance(band.first_undecodable, _PLAIN_IDS) else str(band.first_undecodable)
    return (payload, band.truncated, band.dropped, band.undecodable, first)


def _built(answer: tuple[Any, ...]) -> tuple[bytes, BandResult]:
    """a :func:`_build_tile` answer: the tile's bytes, and its band's accounting (no features: they stayed in the
    worker)."""
    payload, truncated, dropped, undecodable, first = answer
    return payload, BandResult(
        features=[], truncated=truncated, dropped=dropped, undecodable=undecodable, first_undecodable=first
    )
