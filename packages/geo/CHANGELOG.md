# Changelog

All notable changes to `3tears-geo` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the package version moves in **lockstep** with the rest of the
3tears monorepo (every package tracks the framework git tag; see
`README.md` "Versioning policy").

## [0.19.0]

First release.

### Added

- **Tile addressing** -- Web Mercator EPSG:3857, XYZ orientation with `y` increasing
  southward from the top-left. Stated explicitly rather than left implicit: XYZ and TMS
  differ only in that axis, and confusing them renders a vertically mirrored map that
  looks plausible enough to ship. Latitude clamps to the Mercator bound instead of
  diverging at the poles.
- **Attribute coercion** -- a fixed SQL-to-MVT mapping. Two cases are quietly lossy if
  chosen per-caller: NULL becomes an *omitted key*, since MVT has no null and collapsing
  it to zero shades unmeasured regions on a choropleth as though they had been surveyed;
  and `bool` is checked before `int`, which it subclasses, so `True` does not degrade to
  `1` and stop matching style expressions. Structured values are rejected at
  registration rather than stringified into an attribute no style expression can use.
- **Geometry** -- WKB/EWKB decoding via shapely, and bounds. Decoding is forgiving by
  design: one malformed row is logged and skipped rather than aborting a build, which
  would lose the map instead of one feature.
- **Zoom bands** -- aggregate below a declared crossover, individual features above.
  Low zoom is not simplified high zoom; dropping features at z4 leaves an arbitrary
  sample of whichever survived. The feature cap is a hard limit applied *after* geometry
  work, because ranking must consider every candidate, and truncation is reported rather
  than silent -- a silently capped tile reads as "this is all the data there is".
- **MVT encoding** -- projection into tile-local coordinates through Mercator rather
  than linear interpolation, which would skew geometry toward tile edges. Non-integer
  feature ids travel as a property: MVT ids are uint64 by specification, so a census
  geoid is silently coerced to `0`, which would collapse every feature in a tile onto
  one id and break any client-side join.
- **`FeatureCache`** -- a per-pod cache of source features with a companion SQLite
  R-Tree, on the same connection pool, as a `BaseCollection` subclass rather than a
  bespoke wrapper. An index alone cannot answer "which features are in this rectangle":
  it can say what a pod *holds*, never whether it holds *all* of them, and a tile built
  from a partial set is wrong rather than slow, then cached as immutable. So the cache
  tracks coverage by chunk -- the same quantization trick one level up.
- **`TileCollection`** -- a `DerivedCollection` keyed on `(layer, version, z, x, y)`
  with an object-store durable tier. `version` makes the value immutable, so a rebuild
  is a *new address* rather than an invalidation and the superseded generation ages out
  of downstream caches without anyone purging anything.

### Notes

- No PostGIS. Verified empirically that YugabyteDB ships no postgis extension, so the
  work `ST_Intersects` / `ST_Simplify` / `ST_AsMVT` would do in SQL happens here. The
  applicable prior art is Tippecanoe and supercluster, not Martin / pg_tileserv /
  Tegola, all of which assume `ST_AsMVT` in the database.
