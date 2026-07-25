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
