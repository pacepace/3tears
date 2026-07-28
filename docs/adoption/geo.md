# 3tears-geo

`threetears.geo` -- slippy-map vector tiles built in application code: tile
addressing, zoom bands, geometry projection, and MVT encoding, with a durable
tile cache.

## Problem

PostGIS is not available on YugabyteDB. So the work `ST_Intersects`,
`ST_Simplify` and `ST_AsMVT` would do in the database has to happen somewhere
else, and "somewhere else" is where map backends usually go wrong.

Three failures follow, and none of them announce themselves. **The tiling
convention is a coin flip**: Web Mercator and TMS differ only in the direction of
one axis, so confusing them renders an upside-down map that looks plausible until
somebody recognises a coastline. **Low zoom is not simplified high zoom**: a z4
tile spans a large fraction of a country, and building it by simplifying and
dropping individual features is both expensive and wrong to look at, because
whichever features survive the drop are an arbitrary sample rather than a summary.
And **without a spatial index there is no cheap answer to "which features are in
this rectangle?"** -- the one question every tile build asks -- so a naive backend
does a bbox range scan per tile while adjacent tiles ask for heavily overlapping
sets of the same features.

## What it does

- **Tile addressing and projection** (`TileId`, `tile_bounds`,
  `tile_for_point`, `bounds_to_tile_range`, `project_to_tile`) -- the scheme is
  stated explicitly rather than left implicit, because the cost of getting it
  wrong is a silently inverted map.
- **Zoom bands** (`aggregate_band`, `feature_band`, `AggregateSpec`,
  `FeatureSpec`, `BandResult`) -- aggregate below a crossover zoom, individual
  features above it. Two different products for two different questions, rather
  than one pipeline degraded.
- **MVT encoding** (`encode_tile`, `TILE_EXTENT`) -- geometry arrives in WGS84
  and leaves in tile-local integer coordinates over the 4096-unit extent, framed
  by the tile's own bounds. That projection is the real work;
  `mapbox_vector_tile` handles the protobuf once the coordinates are right.
- **A per-pod feature cache with a SQLite R-Tree bbox index** (`FeatureCache`,
  `FeatureLoader`) -- the spatial index the database cannot provide, sized to the
  fact that neighbouring tiles reuse most of their source features.
- **A durable tile collection** (`TileCollection`, `TileEntity`,
  `ViewportRequest`) -- a quantized key, a computed value, and a durable tier that
  outlives any one pod, built on `core`'s `DerivedCollection` so the quantization
  contract and single-flight gates are not reimplemented. The object store is
  constructor-injected and typed loosely on purpose: this package never imports
  one, so it is not a dependency and any store exposing `put` / `open_read` will
  do.
- **Attribute coercion** (`coerce_attribute`, `coerce_attributes`,
  `MVT_SCALAR_TYPES`, `UnsupportedAttributeError`) -- MVT carries a narrow scalar
  type set, so what a layer may legally attach is checked rather than assumed.

## Design philosophy

**State the convention.** Where two conventions differ by one sign and fail
silently, the code names which one it implements and why, because the failure is
invisible to every test that does not look at a rendered map.

**Different zooms are different questions.** The zoom band is a first-class
concept rather than a threshold buried in a simplifier, because "summarise this
region" and "show me these features" are not the same operation performed at
different strengths.

**Compute once, share the result.** Tiles are expensive and requested
repeatedly, so the collection quantizes its key, gates concurrent builds, and
persists to an object store -- one pod's work serves every later request from any
pod.

**No spatial database, no pretending otherwise.** The R-Tree cache is per pod and
deliberately so: it is a working set for a run of neighbouring tiles, not a
replacement for the index the database does not have.

## When to adopt

Adopt it when you are serving vector tiles from a Postgres-compatible database
without PostGIS, which is the situation it was built for. Adopt the tile
addressing alone if you only need correct slippy-map arithmetic and want the
convention question settled by something that has already been wrong once.

Do **not** adopt it if you have PostGIS: `ST_AsMVT` in the database will beat
this, and this exists because that option was closed. It needs somewhere durable
to put built tiles -- an object store, which you supply -- and it brings `shapely`
and `mapbox-vector-tile`, real dependencies rather than pure Python.

## Composes with

- [`core`](core.md) -- `DerivedCollection` supplies the quantization contract and
  the single-flight gates the tile collection is built on.
- [`object-store`](object-store.md) -- the usual choice for the injected durable
  tier, though nothing here imports it; anything with the same two methods works.
- [`media-contracts`](media-contracts.md) -- a declared dependency, for the
  contracts that keep this decoupled from whoever produces the source features.
- [`observe`](observe.md) -- structured logging throughout.

## Install

```bash
pip install 3tears-geo
```
