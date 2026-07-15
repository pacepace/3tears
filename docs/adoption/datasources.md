# 3tears-datasources

`threetears.datasources` -- datasource entities, collections, namespace
helpers, agent-yaml config model, and a driver abstraction for PostgreSQL,
Redshift, Snowflake, and BigQuery backends.

## Problem

Without a single definition of "what is a datasource," every consumer ends
up with its own connection-config shape, its own driver-selection logic, and
its own way of coupling to a specific warehouse client library -- making it
expensive to add a new backend or swap one out later.

## What it does

- Datasource entities and collections on top of `core`'s three-tier caching.
- An agent-yaml config model for declaring datasources declaratively.
- A `Driver` abstraction covering PostgreSQL and Redshift (implemented) plus
  Snowflake and BigQuery (currently stubs that raise `NotImplementedError`
  on every call), accessed only through a factory -- never imported
  directly.

## Design philosophy

`datasources` is the single source of truth for "what is a datasource"
across every 3tears consumer. Drivers are gated behind extras
(`datasources[redshift]`, `[snowflake]`, `[bigquery]`) and reached only
through the factory, so a consumer never hard-couples to a specific
warehouse client library. Swapping or adding a backend does not touch
consumer code.

## When to adopt

Any app that connects to more than one kind of external data warehouse, or
wants datasource configuration to be declarative rather than hand-wired per
integration.

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
