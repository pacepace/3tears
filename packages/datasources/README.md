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

### A pod's own exports

A pod deletes each export once its load has proven it. A pod killed in the
middle of a load never gets to, and the only other record of what it wrote is
the hub's (`platform.datasource_export_objects`, one row per export the hub
ran and has not deleted). `DatasourceQueryClient.list_own_exports` reads that
record back for the calling pod, so the next run can delete what the last one
left:

```python
try:
    left = await client.list_own_exports("reports-warehouse")
except DatasourceExportListUnsupportedError:
    left = []  # a hub that predates the ask: nothing can be cleared through it; log and carry on
for export in left:
    await client.delete_export("reports-warehouse", destination=export.destination)
```

It returns every `DatasourceExportRecord` (`destination`, `object_prefix`,
`date_created`), read to the last page. Nothing ends an export but its
deletion: there is no timer, and listing changes nothing.

#### The hub's contract for the ask

The ask rides the datasource's existing query subject
(`{ns}.datasource.<name>.query`) as one more field of
`DatasourceQueryRequest`, beside `query`, `fingerprint`, `export` and
`export_delete`. No subject and no NATS grant is added.

Request (one page):

```json
{"correlation_id": "<uuid>", "identity_token": "<token>", "user_identity_token": null,
 "export_list": {"cursor": null}}
```

`cursor` is `null` for the first page and the previous page's `next_cursor`,
unchanged, for each page after it.

Reply (`DatasourceQueryResponse`):

```json
{"success": true, "correlation_id": "<the request's>",
 "export_list": {"exports": [{"destination": "enr/r1/VA",
                              "object_prefix": "exports/<namespace id>/enr/r1/VA/",
                              "date_created": "2026-10-13T09:00:00Z"}],
                 "next_cursor": null}}
```

What the hub does, in order:

1. **Verify the caller**, exactly as for every ask on the subject. A token
   that does not verify is `IDENTITY_REFUSED`.
2. **Resolve the datasource** from the subject. None served there is
   `DATASOURCE_NOT_FOUND`.
3. **Admit a tool pod, on its identity alone.** A principal that is not a
   tool pod is `EXPORT_NOT_GRANTED`. Neither `datasource.read` nor the export
   grant is asked, for the reason the delete asks neither: a pod whose grant
   was withdrawn must still be able to find, and so delete, what it was let
   write. The datasource needs no export configured to answer.
4. **Answer for the verified caller only.** The rows returned are those of
   `datasource_export_objects` whose `datasource_name` is the subject's
   datasource and whose `tool_pod_id` is the principal the verified token
   names. The request has no field naming a pod and forbids unknown fields,
   so there is no way to ask about another pod; the hub must not add one.
5. **Page by destination.** Rows are ordered by `destination`, ascending
   (unique within a datasource). The cursor is opaque to the caller; build it
   with `threetears.core.pagination.Keyset(columns=("destination",),
   descending=False)`: read `page_size + 1` rows after the cursor and let
   `Keyset.page` cut the page and mint `next_cursor`. `next_cursor` is `null`
   on the last page, and a pod with no recorded export gets
   `{"exports": [], "next_cursor": null}` with `success: true`. A page
   carries at most 200 exports, so the largest reply fits a 1 MiB message. A
   cursor that does not decode is `EXPORT_REFUSED`.
6. **Never answer `MALFORMED_REQUEST` to a request it parsed.** That code is
   how the client recognises a hub that does not know the ask (below), so a
   hub that does know it answers every refusal with another code.

The hub reads these models from this package, so the hub change that moves
its 3tears pin to the release carrying them ships the handler with it.

#### Against a hub that predates the ask

Pods and hubs upgrade at different times. A hub on an earlier release reads
the request with a model that forbids unknown fields, so `export_list` fails
to parse and it answers, before verifying anything:

```json
{"success": false, "error_code": "MALFORMED_REQUEST",
 "error_message": "malformed datasource query request", "correlation_id": null}
```

A hub that has the models but no handler reads the request, finds no ask it
serves and answers `MALFORMED_REQUEST` as well, with the correlation id. The
client turns `MALFORMED_REQUEST` in answer to this ask into
`DatasourceExportListUnsupportedError` (a `DatasourceQueryError` whose
`error_code` is `EXPORT_LIST_UNSUPPORTED`) and logs it at WARNING. It is
raised rather than returned as an empty list because the hub named no
export: "none recorded" and "could not ask" must not look alike. A caller
that lists in order to clear catches it and clears nothing; the leftover
exports stay on the operator's surface, as before the ask existed. The same
holds part-way through a hub's roll: a later page answered by an older
replica raises it, and no part of the list is returned.

Every other ask a newer client sends is unchanged on the wire (`export_list`
is left off a request that does not make it), so pod and hub upgrade in
either order.

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
