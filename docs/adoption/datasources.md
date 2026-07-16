# 3tears-datasources

`threetears.datasources` -- the **capability-source registry**: entities,
collections, namespace helpers, agent-yaml config model, and a driver
abstraction for PostgreSQL, Redshift, Snowflake, and BigQuery backends.

> **The package is named after its original and largest kind.** Since Fork-1
> (`gu-task-08`) the `platform.datasources` registry holds *capability
> sources* -- database datasources, external API imports, and MCP imports --
> discriminated by a `kind` field. The package, table, and module names still
> say "datasource"; the public API says `CapabilitySource*`. Read "what is a
> datasource" below as "what is a capability source" wherever the two differ.

## Problem

Without a single definition of "what is a capability source," every consumer
ends up with its own connection-config shape, its own driver-selection logic,
and its own way of coupling to a specific warehouse client library -- making
it expensive to add a new backend, a new kind of source, or swap one out
later.

## What it does

- `CapabilitySourceEntity` / `CapabilitySourceCollection` on top of `core`'s
  three-tier caching, discriminated by `CapabilitySourceKind`
  (`datasource` | `api_import` | `mcp_import`).
- **Kind-conditional config storage**: a `datasource`-kind row carries an
  encrypted JSON `connection_config`; `api_import` / `mcp_import` rows carry a
  `threetears.core.security.secret_refs` `scheme://locator` in the same
  physical column.
- Datasource-table entities and collections (`platform.datasource_tables`) --
  3tears models the shape; the host application's migrations create it.
- An agent-yaml config model for declaring sources declaratively.
- A `Driver` abstraction covering PostgreSQL and Redshift (implemented) plus
  Snowflake and BigQuery (currently stubs that raise `NotImplementedError`
  on every call), accessed only through a factory -- never imported
  directly. The driver axis (`DataSourceType`) applies **only** to
  `kind='datasource'` rows.

## Design philosophy

`datasources` is the single source of truth for "what is a capability source"
across every 3tears consumer. Two axes stay separate: `kind` says *what sort
of source a row is*; `DataSourceType` says *which driver reaches it*, and is
meaningful only for `datasource`-kind rows. Drivers are gated behind extras
(`datasources[redshift]`, `[snowflake]`, `[bigquery]`) and reached only
through the factory, so a consumer never hard-couples to a specific
warehouse client library. Swapping or adding a backend does not touch
consumer code.

Governed knowledge anchors here: a playbook entry or concept binds to a
required capability-source id, and that row's `customer_id` is what carries
RBAC into the knowledge layer. See
[`agent-knowledge`](agent-knowledge.md).

## When to adopt

Any app that connects to more than one kind of external data warehouse,
registers external API or MCP capability sources, or wants source
configuration to be declarative rather than hand-wired per integration.

## Composes with

- [`core`](core.md) -- the three-tier entity/collection base.
- [`observe`](observe.md) -- logging throughout the driver layer.

## Install

```bash
pip install 3tears-datasources
# extras as needed:
pip install "3tears-datasources[redshift]"
pip install "3tears-datasources[snowflake]"
pip install "3tears-datasources[bigquery]"
```
