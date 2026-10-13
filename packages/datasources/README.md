# 3tears-datasources

Datasource entities, three-tier collections, namespace helpers, the
agent-yaml-facing `DatasourceConfig`, and the `Driver` abstraction
(plus concrete asyncpg + Redshift drivers) for the 3tears platform.

This is the single source of truth for "what is a datasource" across
every 3tears consumer.

## Public surface

Imported via `from threetears.datasources import …`:

- **entities** -- `DataSourceEntity`, `DataSourceTableEntity`,
  `DataSourceColumnEntity`, `DataSourceRelationEntity`,
  `TableTemplateEntity` + the `DataSourceType`,
  `DataSourceAccessMode`, `DataSourceStatus` enums.
- **collections** -- `DataSourceCollection`,
  `DataSourceTableCollection`, `DataSourceColumnCollection`,
  `DataSourceRelationCollection`, `TableTemplateCollection`
  (three-tier `SchemaBackedCollection` / `BaseCollection` subclasses
  with L1/L2/L3 caching + `_publish_invalidation` on save).
- **namespace** -- `DATASOURCE_NAMESPACE_TYPE`,
  `datasource_namespace_id(uuid) -> uuid`,
  `datasource_namespace_name(name) -> str`.
- **config** -- `DatasourceConfig` (the agent-yaml-facing model the
  SDK validates against) plus the per-driver `ConnectionConfig`
  discriminated union.
- **query wire** -- `DatasourceQueryClient`, `DatasourceQueryRequest`,
  `DatasourceQueryResponse`, `DatasourceQueryResult`, `DatasourceQueryError`.
  How a process that holds no warehouse credential (a tool pod) queries a
  hub-served datasource on its own identity: it publishes on
  `{ns}.datasource.{name}.query` carrying its hub-minted token, the hub
  verifies the token, evaluates the caller's grant on the datasource
  namespace, runs the query on the driver it already holds, and answers
  with rows or a typed refusal. The client hands back the rows and
  `truncated`, set when the hub cut the result at its row cap, so a caller
  deriving state from a full read can refuse a prefix rather than treat the
  missing rows as absent. Rows only otherwise -- the model-facing rendering
  with honesty imperatives is the datasource tool's job.
- **geo config** -- `GeoConfig`, `GeoLayerConfig`, `GeometryConfig`,
  `GeometryKind`, `FeatureBandConfig`, `AggregateBandConfig`,
  `CacheClassConfig`, `MeasureAggregation`, `layer_name_fits`.

`threetears.datasources.__all__` is the full list. The surfaces below are
imported from their own modules.

## Reading a relation by parts

`threetears.datasources.partitioned_read` keeps a relation current when the
source rebuilds it whole but it divides into parts that change independently
(one state's results, one customer's rows):

- `fingerprint_parts` fingerprints every part, over the columns it is given.
- `read_parts` reads the parts asked for, at most `concurrency` at a time
  (`DEFAULT_PART_CONCURRENCY`); the first part that fails cancels the rest
  and is raised as itself.
- `read_partitions` reads every part of a relation (or the parts asked
  for) over the query rail, divided by one column, pages read side by side,
  and hands the parts back one at a time.

A caller compares each part's fingerprint with the one recorded when it was
last read and reads only the parts that moved. `DatasourceQueryClient` gains
`relation_fingerprint` and `relation_fingerprint_groups` for this, and
`read_all` (in `query_client`) reads one part whole through the query rail.

## Warehouse exports

A large relation read through the query rail costs a round trip per page. A
warehouse that can write a query's result to object storage does the same
work in seconds, and the reader takes the files from the bucket.

- `threetears.datasources.export` -- what may be exported, and where. The
  caller sends a plain `SELECT` and a short relative destination, never the
  `UNLOAD` or a bucket. `ExportConfig` (on the datasource's connection
  config) names the bucket, the prefix and the role; `check_destination`,
  `check_embeddable` and `redshift_unload_statement` fix the statement
  around the `SELECT`; a `SELECT` that cannot be embedded safely is refused
  (`ExportRefusedError`). A datasource with no `ExportConfig` cannot export.
- `threetears.datasources.export_read` -- `export_part` asks the hub for the
  export (`DatasourceQueryClient.export`), fingerprints the part before and
  after, and reads the parquet files through an object store the caller
  built. The export is proven whole before a row is handed back
  (`IncompleteExportError` otherwise) and is then always deleted through the
  hub (`DatasourceQueryClient.delete_export`), on every way out.
- `threetears.datasources.partitioned_export` -- `export_partitions` reads a
  relation's chosen parts through one export divided by a column, handing
  the parts back one at a time, with the same proof and the same deletion.

Reading an export needs `pip install '3tears-datasources[export]'` (pyarrow).

## Reporting reloaded geography layers

`threetears.datasources.geo_reload` is the tool pod to hub contract for
platform geography layers. A pod writes a new generation of a layer's shapes
into its own tables, stamping each row with the generation, and reports it
with `report_geo_layers_reloaded(nats_client, identity_token=...,
generations=...)`. The hub moves each layer's tile version to that generation
and answers each layer's `LayerVersions` (`version`, `previous`);
`generations_to_delete` names the stamped generations the pod then deletes,
all but those two. Nothing is served for a layer before its first report, and
a version only moves forward: a generation below it is refused
`GENERATION_BEHIND`. The request and reply models live in the same module and
the hub imports them.

## Drivers

The `Driver` ABC + `create_driver(config, *, hub_l3_pool=None)` factory
are the entry point. Concrete drivers are accessed via the
factory, NOT imported directly:

```python
from threetears.datasources.drivers import create_driver
driver = create_driver(config.connection_config)
try:
    rows = await driver.fetch("SELECT * FROM customers WHERE id = $1", customer_id)
finally:
    await driver.close()
```

Driver implementations live behind extras keys:

- `pip install '3tears-datasources[redshift]'` for `RedshiftDriver`
- `pip install '3tears-datasources[snowflake]'` for `SnowflakeDriver`
  (stub today; full impl tracked separately)
- `pip install '3tears-datasources[bigquery]'` for `BigQueryDriver`
  (stub today; full impl tracked separately)

Postgres / Yugabyte / agent_internal coverage uses `asyncpg` which is
a hard dep (no extras key required).

See `IMPLEMENTING_DRIVERS.md` for the contract every new driver must
satisfy.

## Versioning policy

`3tears-datasources` versions in **lockstep** with the rest of the
3tears monorepo: every package shares one version, which tracks the
framework git tag. All packages move together, and every intra-family
dependency in `pyproject.toml` is bounded to that one minor version.

See `CHANGELOG.md` for the full version history.
