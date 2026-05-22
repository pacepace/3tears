# Changelog

All notable changes to `3tears-datasources` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the package adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
**independently** of 3tears core's release cadence (see `README.md`
"Versioning policy").

## [Unreleased]

Driver abstraction additions land here as `datasource-task-09` through
`datasource-task-12` execute:

- `Driver` ABC + `create_driver(config)` factory (shard 09)
- `AsyncpgDriver` for postgres / yugabyte / agent_internal (shard 10)
- `RedshiftDriver` for Redshift via `redshift-connector` (shard 11)
- `SnowflakeDriver` + `BigQueryDriver` stubs (shard 12)

## [0.1.0] — initial release

### Added

- `threetears.datasources` package home for the datasource entity +
  collection + namespace + config primitives previously inlined in
  Hub (`aibots/hub/datasources/{entities,collections,schema_collections,namespace,schemas}.py`)
  and the agent SDK (`aibots_agents/devx/schema/agent_config.py:DatasourceConfig`).
- `threetears.datasources.entities` — `DataSourceEntity`,
  `DataSourceTableEntity`, `DataSourceColumnEntity`,
  `DataSourceRelationEntity`, `TableTemplateEntity` + the
  `DataSourceType`, `DataSourceAccessMode`, `DataSourceStatus` enums.
  All entities subclass `threetears.core.entities.base.BaseEntity` and
  preserve the composite-PK / single-PK shape from their Hub origins
  byte-for-byte.
- `threetears.datasources.collections` — `DataSourceCollection`
  (`SchemaBackedCollection` subclass), `DataSourceTableCollection`,
  `DataSourceColumnCollection`, `DataSourceRelationCollection`,
  `TableTemplateCollection` (all `BaseCollection` subclasses with the
  same `serialize / deserialize / fetch_from_postgres /
  save_to_postgres / delete_from_postgres` shape).
- `threetears.datasources.namespace` — `DATASOURCE_NAMESPACE_TYPE`,
  `datasource_namespace_id`, `datasource_namespace_name` helpers.
- `threetears.datasources.config` — `DatasourceConfig` (agent-yaml
  shape). The discriminated-union `ConnectionConfig` lands in
  `datasource-task-08`.
- Companion promotion: `threetears.core.utils.pg_pool_kwargs` — the
  shared `asyncpg.create_pool` kwargs helper, DSN redactor, and
  startup-timeout wrapper used by both Hub L3 and the future
  `AsyncpgDriver`. Promoted from Hub's `aibots/hub/common/pg_pool.py`
  so neither owns a divergent copy.

### Out of scope (intentionally, for 0.1.0)

- Concrete driver implementations (shards 09-12).
- Hub admin API DTOs (`DataSourceCreateRequest`, `DataSourceResponse`,
  etc.) — those are Hub-API contracts, not framework primitives, and
  stay in Hub as `aibots/hub/datasources/hub_api.py`.
- Per-table ACL via the `datasource_table` namespace — already lives
  in Hub's template routes; separate concern.

### Migration notes

`3tears-datasources` is a brand-new package; no prior version exists.
Hub + agent-SDK consumers flip imports from the old Hub paths to
`threetears.datasources.*` in the same migration PR (shards 07 + 08).
No backward-compat shims are provided.
