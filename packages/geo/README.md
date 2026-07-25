# 3tears-geo

Slippy-map tile geometry for the 3tears platform: WKB decoding, zoom bands,
and MVT encoding — all in application code.

## Why this exists in Python rather than in SQL

Every off-the-shelf tile server (Martin, pg_tileserv, Tegola) assumes
PostGIS in the database and calls `ST_AsMVT`. YugabyteDB ships no postgis
extension — verified empirically against `yugabytedb/yugabyte:2025.2.1.0-b141`,
where `CREATE EXTENSION postgis` fails outright with no control file on the
image. There is therefore no `ST_Intersects`, no `ST_Simplify`, and no
`ST_AsMVT` to call.

So this package does that work: `shapely` for the geometry, and
`mapbox-vector-tile` for the encoding. The relevant prior art is Tippecanoe
and supercluster, not the Postgres tile servers.

## The tiling scheme, stated explicitly

Leaving this implicit guarantees a defect, because the two common
conventions differ only in the direction of one axis:

- **Web Mercator (EPSG:3857)**
- **XYZ orientation** — `y` increases *southward* from the top-left origin.
  OpenStreetMap / MapLibre / Google. **Not TMS**, whose `y` increases
  northward.
- Source coordinates are **WGS84 (EPSG:4326)**, projected at build time.
- MVT geometry uses tile-local integer coordinates over a **4096-unit
  extent**, the format default, not varied per layer.

Latitude clamps to ±85.0511287798066°, the bound of the Mercator square.

## Public surface

Imported via `from threetears.geo import …`:

- **tiles** — `TileId`, `BoundingBox`, `tile_for_point`, `tile_bounds`,
  `bounds_to_tile_range`, `TILE_EXTENT`, `MAX_MERCATOR_LATITUDE`.
- **attributes** — `coerce_attribute`, `coerce_attributes`,
  `validate_attribute_value`, `UnsupportedAttributeError`.
- **geometry** — `decode_geometry`, `geometry_bounds`, `point_geometry`.
- **bands** — `feature_band`, `aggregate_band`, `FeatureSpec`,
  `AggregateSpec`, `TileFeature`, `simplification_tolerance`.
- **mvt** — `encode_tile`, `project_to_tile`.
- **features** — `FeatureCache` (per-pod source cache + R-Tree).
- **collection** — `TileCollection`, `LayerDefinition`, `ViewportRequest`.

## Zoom bands: aggregate below, features above

Low zoom is not simplified high zoom. A z4 tile spans a large fraction of a
country; rendering it by simplifying and dropping individual features leaves
an arbitrary sample of whichever survived. A national view showing 4,000 of
180,000 precincts is not a coarse view of the data — it is a different and
misleading dataset.

So each layer declares a crossover. Below it, rows roll up to a coarser
declared geography and each bucket becomes one feature carrying real totals.
Above it, individual features are simplified per zoom and capped in count.
`bl-ds-ai-lcv-registration` reached the same split independently: its
precomputed z4–z10 band holds cluster aggregates, with individual features
only from z11.

The cap is a hard limit, not advice. An uncapped tile in a dense metro
reaches tens of megabytes, which exceeds the NATS payload ceiling, defeats
the L2 hot band, and renders badly. When it binds, features drop by a
declared ranking column so the survivors are the important ones, and the
result records that it was truncated — a silently capped tile reads as
"this is all the data there is".

## Feature ids are not MVT ids

MVT feature ids are **uint64 by specification**. A census geoid or a UUID is
silently coerced to `0` by the encoder — no error — which would collapse
every feature in a tile onto one id and break any client-side join. So only
genuine integers reach the wire-level id; everything else travels as a
property, which is exactly what MapLibre's `promoteId` is for. The id is
always present as a property either way, so a client has one place to look.

## The R-Tree earns its place via chunk coverage

`FeatureCache` indexes cached source features in a SQLite R-Tree (built in,
unlike SpatiaLite) on the same connection pool as its own managed table, as
a `BaseCollection` subclass rather than a bespoke `SQLiteBackend` wrapper.

An index alone cannot answer "which features are in this rectangle": it can
say what a pod *holds*, never whether it holds *all* of them, and a tile
built from a partial set is wrong rather than slow — then cached as
immutable. So the cache tracks coverage by **chunk**: it loads the coarse
tile containing a request and records that chunk as covered. A run of
neighbouring tiles, which overlap almost entirely in source features, then
pays one L3 read between them instead of one each.

### Attribute coercion is fixed, not per-caller

MVT carries only strings, numbers and booleans, so every other SQL type
needs a declared mapping. Two cases are quietly lossy if chosen badly:

| SQL | MVT | Note |
|---|---|---|
| integer / numeric / double | number | doubles are IEEE754 |
| text / varchar | string | |
| boolean | bool | checked before `int`, since `bool` subclasses it |
| **NULL** | **key omitted** | MVT has no null |
| timestamp / date | string | ISO-8601 UTC; naive stamps read as UTC |
| JSONB / array / bytes | **rejected** | project a scalar column instead |

Omitting the key for NULL is the only faithful encoding, and it puts a real
obligation on the client: a style expression reading that attribute must
supply its own fallback. The upside is that "no data" and "a genuine zero"
stay distinguishable in the tile — which matters most on a choropleth, where
collapsing them shades unmeasured regions as though they were measured.

## Versioning policy

`3tears-geo` versions in lockstep with the rest of the 3tears monorepo:
every package shares one version, tracking the framework git tag. All
packages move together.
