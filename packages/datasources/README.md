# 3tears-datasources

Datasource entities, three-tier collections, namespace helpers, the
agent-yaml-facing `DatasourceConfig`, and the `Driver` abstraction
(plus concrete asyncpg + Redshift drivers) for the 3tears platform.

This is the single source of truth for "what is a datasource" across
every 3tears consumer (the Hub, agent pods, future products).

## Public surface

Imported via `from threetears.datasources import …`:

- **entities** — `DataSourceEntity`, `DataSourceTableEntity`,
  `DataSourceColumnEntity`, `DataSourceRelationEntity`,
  `TableTemplateEntity` + the `DataSourceType`,
  `DataSourceAccessMode`, `DataSourceStatus` enums.
- **collections** — `DataSourceCollection`,
  `DataSourceTableCollection`, `DataSourceColumnCollection`,
  `DataSourceRelationCollection`, `TableTemplateCollection`
  (three-tier `SchemaBackedCollection` / `BaseCollection` subclasses
  with L1/L2/L3 caching + `_publish_invalidation` on save).
- **namespace** — `DATASOURCE_NAMESPACE_TYPE`,
  `datasource_namespace_id(uuid) -> uuid`,
  `datasource_namespace_name(name) -> str`.
- **config** — `DatasourceConfig` (the agent-yaml-facing model the
  SDK validates against) plus the per-driver `ConnectionConfig`
  discriminated union (lands in `datasource-task-08`).

## Drivers

The `Driver` ABC + `create_driver(config, *, hub_l3_pool=None)` factory
land in `datasource-task-09`. Concrete drivers are accessed via the
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
satisfy (lands in `datasource-task-09`).

## Versioning policy

`3tears-datasources` adopts SemVer **independently** of the 3tears
core release cadence. Other 3tears packages share core's version; this
package is explicitly multi-consumer (Hub + future products) so version
lockstep would force every core breaking change to bump this package
and every downstream consumer.

The `pyproject.toml` pins compatible 3tears core via a range
(`3tears>=0.9.0,<1.0`), not a single pinned version.

- **Major bumps** (`1.0 → 2.0`): Driver ABC shape changes, Collection
  PK changes, `ConnectionConfig` structural breaks.
- **Minor bumps** (`0.1 → 0.2`): additive Driver methods (`fetch_iter`,
  new high-level helpers), new driver type, new `ConnectionConfig`
  field with default.
- **Patch bumps**: bug fixes, doc updates, internal refactors.

See `CHANGELOG.md` for the full version history.
