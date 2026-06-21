# Changelog

All notable changes to `3tears-datasources` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the package version moves in **lockstep** with the rest of the
3tears monorepo (every package tracks the framework git tag; see
`README.md` "Versioning policy").

## [0.12.3]

### Added

- Per-column value-coverage probe: classifies a column as unloaded when every
  value is zero across the table (the `UNLOADED_COLUMN` source the hub mirrors
  into datasource read results), with driver-coverage tests.

### Fixed

- Redshift: re-apply `search_path` on every connection acquisition, so a pooled
  connection never serves a stale path left by a prior caller.

### Changed

- Centralize JSONB handling through native binding under the codec
  (`collections-task-04`, Option B), simplifying the collection write path; an
  enforcement drift guard prevents a new column bypassing it.

## [0.12.2]

### Added

- `DataSourceSchemaDigest` entity + `DataSourceSchemaDigestCollection` — a
  three-tier collection for the materialized documented-schema digest, one row
  per datasource, addressed by primary key `datasource_id` for a by-pk hot-L1
  read with L2/L3 fallback and cross-pod invalidation. The `tables` projection
  is stored as JSONB.

### Fixed

- JSONB write double-encode in the digest tables: a pre-`json.dumps`'d string
  bound as `::jsonb` was re-encoded to a scalar by the text-format jsonb codec;
  writes now text-cast (`::text::jsonb`). Covered by a real-L1 round-trip test.

## [0.11.0]

### Added

- Platform-sharing primitives: a flat datasource primary key, a
  `visibility` field, and `origin_datasource_id` lineage so a customer
  datasource can inherit a platform-shared datasource's schema docs and
  governed knowledge (concepts / entries) rather than re-documenting them.

## [0.10.2]

### Added

- New `allowed_schemas: list[str]` field on
  `RedshiftConnectionConfig`, `PostgresConnectionConfig`, and
  `YugabyteConnectionConfig`. Drivers thread the value onto the
  connection's `search_path` at open time so agents can write
  unqualified table names (`SELECT … FROM report_geofacts_joined_data`)
  instead of having to fully qualify every reference. Empty default
  preserves the backend's default `search_path`.
- `build_search_path_value` and `build_set_search_path_sql`
  helpers in `threetears.datasources.drivers._util` with
  identifier-quoting suitable for adversarial schema names.
- Redshift driver issues `SET search_path TO "<schemas>"` via
  `cursor.execute` after the existing `SET statement_timeout`
  block on every connection open. Cached connection lifecycle
  applies the SET once per backend session.
- asyncpg driver passes `server_settings={"search_path": "..."}`
  through `create_pool`. The value rides the pgwire STARTUP
  packet and is preserved across `DISCARD ALL` (the default
  asyncpg pool-release reset), where an `init=` callback would
  have been wiped. The live testcontainer pass caught the reset
  issue before release.
- 8 new unit tests (4 Redshift, 4 asyncpg) pin the SQL / kwargs
  both drivers ship to the client library, including the
  identifier-quoting paths. 4 new live integration tests (2
  Redshift against `central-reporting`, 2 asyncpg against the
  testcontainer) prove unqualified-table-name resolution
  end-to-end and confirm empty `allowed_schemas` leaves the
  default `search_path` intact.

## [0.10.1]

### Fixed

- `RedshiftDriver` now runs `ROLLBACK` on a query error before
  returning the connection to its cache. `redshift_connector` uses
  the DB-API default of `autocommit=False`, so a failed statement
  leaves the connection's implicit transaction in `aborted` state;
  without the rollback the next caller to acquire that cached
  connection trips `25P02: current transaction is aborted, commands
  ignored until end of transaction block` on every subsequent
  statement until eviction. The fix lives in `_acquire_and_run`, the
  central wrapper every query method routes through (so `fetch`,
  `execute`, and `fetch_iter` all benefit). If the rollback itself
  raises, the connection is evicted instead of released and a
  WARNING is logged; the ORIGINAL query exception is what propagates
  to the caller. Covered by `TestRollbackOnError` unit tests
  (mocked-cursor positive / failure / two-fetch end-to-end shapes)
  and one live integration test against `central-reporting`.

## [0.9.1]

### Changed

- Credentials are now referenced by a `scheme://locator` string
  (`password_ref` / `credentials_json_ref`) instead of an env-var
  name (`password_env` / `credentials_json_env`). The value is
  resolved at use time by a pluggable backend in
  `threetears.datasources.secrets`. Shipped backends: `env://NAME`
  (process env) and `k8s://rel/path` (projected-Secret file under
  `AIBOTS_DATASOURCE_SECRETS_DIR`). `vault://`, `aws-secretsmanager://`
  and `gcp-sm://` are registered but raise until implemented.
- Package version realigned to the monorepo lockstep (`0.9.1`); the
  earlier independent-SemVer experiment (`0.1.x`) is retired.

## [Unreleased]

Future enhancements after the initial driver migration ships:

- Snowflake + BigQuery concrete driver implementations (stubs today
  per shard 12 — the ABC supports both shapes; implementations land
  when a consumer needs them).
- `Collection.save_entities_batched` to restore the per-batch write
  performance that shard-13's per-row `save_entity` flow gave up
  (preserving the L2-invalidation contract was non-negotiable; the
  batched variant ships as a 3tears-core enhancement when profiling
  data justifies it).
- `Driver.execute` return-value extension (currently `None`;
  callers that need a row count use `RETURNING *` + the read tool).
- Lift the Hub-side `introspect_if_changed` orchestrator helpers
  (probe + diff coordination) into `threetears.datasources.introspection`
  when a second 3tears consumer wires up a change-driven introspector.

## [0.1.0] — initial release

initial release shipping the full datasource driver-abstraction
migration (shards 07–15 in the `14-eng-ai-bot/docs/datasource-task-*.md`
series). drops in as the canonical home for datasource primitives
across every 3tears consumer.

### Driver layer (shards 09–12)

- `threetears.datasources.drivers.Driver` ABC: the contract every
  concrete driver implements. async-only surface (`fetch`,
  `execute`, `fetch_iter`, `list_tables`, `list_columns`,
  `table_hashes`, `test_connection`, `close`). `fetch_iter` carries
  a working default impl (yields from `fetch`); native streaming
  is the concrete-driver responsibility.
- `threetears.datasources.drivers.create_driver(config, *, hub_l3_pool=None, datasource_name="unknown")`:
  the factory. dispatches on `config.datasource_type` to the right
  concrete driver class. lazy backend imports — importing the
  `threetears.datasources` package roots does NOT pull
  `redshift_connector` / `snowflake-connector-python` /
  `google-cloud-bigquery`. one runtime audit verifies the contract
  on every test run.
- Shared helpers (`drivers/base.py`, `_util.py`, `_sync_bridge.py`):
  - `Driver._with_cancellation(coro_fn, cancel_callback=...)` — the
    canonical pattern every concrete driver routes through for
    `CancelledError` propagation to the backend cancel hook.
  - `_translate_placeholders(sql, target_style)` — `$1` ->
    `%s`/`:1`/`@p1`. asyncpg style is a no-op. handles `$10` vs
    `$1`, escaped `$$`, string-literal `'$1'`.
  - `AsyncSyncBridge(max_workers, name)` — bounded
    `ThreadPoolExecutor` + cancel-aware `to_thread_with_cancel`.
    Redshift / Snowflake / BigQuery drivers share one
    implementation rather than three drifting copies.
  - `@_observed(driver_type=...)` decorator — auto-emits OTel
    `datasource.driver.query.duration` (histogram) +
    `datasource.driver.error` (counter) on every wrapped method.
- `AsyncpgDriver` (shard 10) — covers POSTGRES / YUGABYTE /
  AGENT_INTERNAL. server-side cursor streaming via
  `conn.cursor()` inside `conn.transaction()`. AGENT_INTERNAL
  borrows Hub's L3 pool via `external_pool=`. cancellation uses
  `Connection.cancel()` (NOT `terminate()`) so the connection
  returns to the pool clean.
- `RedshiftDriver` (shard 11) — covers REDSHIFT via
  `redshift-connector` (AWS's official driver; `asyncpg` against
  Redshift's pg-protocol quirks never completed in production).
  bounded connection cache (`connection_cache_size`) amortises the
  ~1-3s TLS+auth handshake. `weakref.finalize` registers GC-time
  cache drain for pod-crash mitigation. cancellation uses
  `Connection.close()` wrapped in `asyncio.wait_for(_, 5.0)` (the
  lib has no `cancel()` API; closing the connection is the only
  in-flight-abort primitive). live integration test against the
  `central-reporting` warehouse is CI-required.
- `SnowflakeDriver` + `BigQueryDriver` (shard 12) — stub
  implementations. every abstract method raises
  `NotImplementedError` with a roadmap-pointing message; module
  docstrings codify the future implementation shape so the next
  contributor reuses `AsyncSyncBridge` / `_with_cancellation` /
  `_translate_placeholders` instead of reinventing.

### Configuration layer (shards 07–08)

- `threetears.datasources.config.DatasourceConfig`: agent.yaml-facing
  config. carries `name`, `access_mode`, `schemas`, and a nested
  `connection_config: ConnectionConfig` field.
- `threetears.datasources.config.ConnectionConfig`: discriminated
  union (Pydantic `Annotated[Union[...], Field(discriminator=
  "datasource_type")]`) routing to per-backend members:
  `PostgresConnectionConfig`, `YugabyteConnectionConfig`,
  `RedshiftConnectionConfig`, `SnowflakeConnectionConfig`,
  `BigQueryConnectionConfig`, `AgentInternalConnectionConfig`.
- Each member declares pool / executor / timeout knobs as Pydantic
  fields with documented defaults. The enforcement test
  `tests/enforcement/test_no_hardcoded_pool_params.py` fails the
  build on any concrete driver that inlines a banned-kwarg literal
  (`min_size`, `max_size`, `command_timeout`, `connection_cache_size`,
  `executor_max_workers`, ...).
- Passwords pass as env-var NAMES (`password_env: str`), NEVER
  resolved values. `resolve_password() -> SecretStr` is the only
  way to read the secret; drivers unwrap `.get_secret_value()`
  at the last moment when handing to the backend lib. Exception
  sanitization (`raise X from None`) breaks the cause chain so
  backend errors don't surface the password.

### Entity + collection + namespace layer (shard 07)

- `threetears.datasources.entities` — `DataSourceEntity`,
  `DataSourceTableEntity`, `DataSourceColumnEntity`,
  `DataSourceRelationEntity`, `TableTemplateEntity` + the
  `DataSourceType`, `DataSourceAccessMode`, `DataSourceStatus`
  enums. All entities subclass `threetears.core.entities.base.BaseEntity`
  and preserve the composite-PK / single-PK shape from their
  Hub origins byte-for-byte.
- `threetears.datasources.collections` — `DataSourceCollection`
  (`SchemaBackedCollection` subclass), `DataSourceTableCollection`,
  `DataSourceColumnCollection`, `DataSourceRelationCollection`,
  `TableTemplateCollection`. `get_by_natural_key(...)` on the
  table + column collections supports the introspector's "insert
  vs update" decision. `DataSourceTableCollection` carries
  `column_hash` natively (the Tier-2 probe digest).
- `threetears.datasources.namespace` — `DATASOURCE_NAMESPACE_TYPE`,
  `datasource_namespace_id`, `datasource_namespace_name` helpers.

### Introspection helpers (shard 02 + 03, lifted from Hub)

- `threetears.datasources.introspection.compute_column_hash(cols) -> str`:
  Python-side cross-language Tier-2 hash. byte-identical to the
  driver-side SQL `MD5(STRING_AGG(column_name || ':' || data_type
  || ':' || COALESCE(is_nullable, ''), ',' ORDER BY ordinal_position))`.
- `IntrospectionDiff` (frozen dataclass) — carries work lists
  (`tables_to_introspect`, `tables_to_delete`) + summary counts
  (`tables_checked`, `tables_unchanged`, `tables_changed`,
  `tables_added`, `tables_removed`, `columns_*`, `elapsed_ms`).
  `has_changes` property short-circuits the orchestrator when
  there's no work to do.
- `compute_introspection_diff(warehouse_hashes, stored_hashes)`:
  pure function. classifies every key into the right work list +
  counter bucket. null stored hashes are forced re-introspect
  (matches the migration-backfill sentinel semantics).

### Companion 3tears-core promotion (shard 07)

- `threetears.core.utils.pg_pool_kwargs` — promoted from Hub. the
  shared `asyncpg.create_pool` kwargs helper, DSN redactor, and
  startup-timeout wrapper. previously lived in Hub's
  `aibots/hub/common/pg_pool.py`; shard 15 deletes the Hub copy
  entirely. consumers import directly from the core helper.

### Out of scope (intentionally, for 0.1.0)

- Hub admin API DTOs (`DataSourceCreateRequest`, `DataSourceResponse`,
  etc.) — those are Hub-API contracts, not framework primitives;
  they stay in Hub as `aibots/hub/datasources/hub_api.py`.
- Per-table ACL via the `datasource_table` namespace — already
  lives in Hub's template routes; separate concern.
- Snowflake + BigQuery concrete driver implementations (stubs only
  today per shard 12 — see Unreleased section).

### Migration notes

`3tears-datasources` is a brand-new package; no prior version exists.
Hub + agent-SDK consumers flip imports from the old Hub paths to
`threetears.datasources.*` in the same migration PR (shards 07 + 08).
No backward-compat shims are provided.

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
