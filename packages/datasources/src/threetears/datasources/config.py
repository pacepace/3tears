"""agent.yaml-facing datasource configuration model + per-driver connection configs.

:class:`DatasourceConfig` is the developer-facing pydantic shape every
agent.yaml ``datasources:`` block validates against. it is framework-
level (not Hub-specific): future 3tears products consume the same
model so the developer-facing config shape stays consistent.

Hub admin DTOs (``DataSourceCreateRequest``, ``DataSourceResponse``,
etc.) explicitly STAY in Hub (``aibots/hub/datasources/hub_api.py``)
because they're API contracts, not framework primitives. only the
agent-yaml shape lives here.

per-driver :class:`ConnectionConfig` members (added in
``datasource-task-08``):

- :class:`PostgresConnectionConfig` -- standard postgres backend; uses
  the asyncpg driver, asyncpg-flavored pool sizing.
- :class:`YugabyteConnectionConfig` -- shape-identical to postgres
  (same asyncpg driver) with the enum kept distinct so operators see
  which backend they target.
- :class:`RedshiftConnectionConfig` -- AWS Redshift via the
  ``redshift-connector`` lib; bounded executor + small connection
  cache instead of an asyncpg pool.
- :class:`SnowflakeConnectionConfig` -- Snowflake via the
  ``snowflake-connector-python`` lib (driver implementation lands as
  a stub in shard 12, full impl tracked separately).
- :class:`BigQueryConnectionConfig` -- BigQuery via
  ``google-cloud-bigquery`` (stateless HTTPS; stub in shard 12).
- :class:`AgentInternalConnectionConfig` -- sentinel for agent-created
  tables (data-task-01). carries no external connection identity; the
  driver borrows Hub's L3 pool via the factory's ``hub_l3_pool=``
  kwarg.

``ConnectionConfig`` itself is the discriminated union keyed on
``datasource_type``; pydantic routes incoming dicts to the right
member via the canonical pattern documented in
``packages/langgraph/src/threetears/langgraph/streaming.py``.

field-naming discipline (documented for the next contributor):

- ``password_env`` and similar ``*_env`` fields carry env-var NAMES,
  not the secret value itself. they are typed ``str``. drivers
  resolve the env var via :meth:`resolve_password` (or
  :meth:`resolve_credentials_json` for BigQuery) which returns
  :class:`SecretStr`. the secret value is only ever held inside a
  ``SecretStr`` and unwrapped at the last possible moment when handed
  to the backend lib.
- fields that hold *resolved* secrets (none today; concrete drivers
  may add them when they cache a connection) MUST be typed
  :class:`SecretStr`. an enforcement test
  (``tests/enforcement/test_secrets_typed.py``) catches drift.
- pool sizing / executor sizing / timeout fields MUST have an
  explicit default with a ``Field(default=N, description="...")``
  docstring. concrete drivers may NOT inline literals for these
  kwargs; the enforcement test
  (``tests/enforcement/test_no_hardcoded_pool_params.py``) catches
  drift in ``drivers/``.
"""

from __future__ import annotations

import os
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from threetears.datasources.entities import DataSourceType

__all__ = [
    "AgentInternalConnectionConfig",
    "BigQueryConnectionConfig",
    "ConnectionConfig",
    "DatasourceConfig",
    "PostgresConnectionConfig",
    "RedshiftConnectionConfig",
    "SnowflakeConnectionConfig",
    "YugabyteConnectionConfig",
]


# regex pattern for valid environment variable names. duplicated from
# the SDK's ``agent_config.py`` so this module has no SDK-side import
# (the SDK depends on this package, not the other way around).
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# valid access mode values, mirrored from
# :class:`threetears.datasources.entities.DataSourceAccessMode`. kept
# as a literal set here so the config validator stays a pure pydantic
# field check without pulling the enum module at validation time.
_VALID_ACCESS_MODES = frozenset({"read", "write", "readwrite"})


# ---------------------------------------------------------------------------
# Helpers for env-var-name -> SecretStr resolution
# ---------------------------------------------------------------------------


def _resolve_env_to_secret(env_var_name: str) -> SecretStr:
    """read the named env var and wrap it in :class:`SecretStr`.

    raised at the last possible moment when handing the credential to
    the backend lib (see ``datasource-task-10`` / ``11`` for driver
    use sites). intermediate variables holding the resolved value as
    ``str`` are not safe — they can land in pydantic ``ValidationError``
    tracebacks or in ``repr()`` output.

    :param env_var_name: documented name of the env var carrying the secret
    :ptype env_var_name: str
    :return: ``SecretStr`` wrapping the resolved value
    :rtype: SecretStr
    :raises ValueError: if the env var is not set; the message names
        the env var (which is the safe-to-log identifier) but never
        the secret value
    """
    raw = os.environ.get(env_var_name)
    if raw is None:
        raise ValueError(
            f"environment variable {env_var_name!r} is not set. "
            f"set it in your shell or .env file before running."
        )
    return SecretStr(raw)


# ---------------------------------------------------------------------------
# Per-driver ConnectionConfig members
# ---------------------------------------------------------------------------


class PostgresConnectionConfig(BaseModel):
    """postgres backend connection config.

    drives the :class:`AsyncpgDriver` for standard PostgreSQL targets.
    asyncpg's pool semantics map directly: ``pool_min_size`` and
    ``pool_max_size`` bound the connection pool, ``command_timeout_seconds``
    caps any single query.

    :param datasource_type: discriminator; must be ``DataSourceType.POSTGRES``
    :param host: postgres host name or IP
    :param port: postgres port (default 5432)
    :param database: database name to connect to
    :param username: postgres user
    :param password_env: env-var NAME holding the password (NOT the
        secret itself; drivers resolve via :meth:`resolve_password`)
    :param pool_min_size: asyncpg pool floor; min open connections per
        driver instance. trade-off: lower = lighter idle footprint;
        higher = no warm-up cost on first query
    :param pool_max_size: asyncpg pool ceiling. trade-off: lower =
        less warehouse load + slower under burst; higher = faster
        bursts + more warehouse-side connections
    :param command_timeout_seconds: per-query asyncpg timeout. trade-off:
        too low = innocent slow queries get cancelled; too high = a
        runaway query holds a connection for the duration
    """

    model_config = ConfigDict(populate_by_name=True)

    datasource_type: Literal[DataSourceType.POSTGRES]
    host: str
    port: int = Field(default=5432, description="postgres port")
    database: str
    username: str | None = Field(
        default=None,
        description="postgres user; None falls back to the connection's default user",
    )
    password_env: str | None = Field(
        default=None,
        description="env-var NAME holding the postgres password (NOT the secret); "
        "None for local dev / trust-auth setups where no password is required",
    )
    pool_min_size: int = Field(
        default=1,
        description="asyncpg pool floor; lower=lighter idle, higher=no warm-up cost on first query",
    )
    pool_max_size: int = Field(
        default=5,
        description="asyncpg pool ceiling; lower=less warehouse load, higher=faster bursts",
    )
    command_timeout_seconds: int = Field(
        default=120,
        description="per-query asyncpg timeout; balances slow-but-valid queries vs runaway protection",
    )

    @field_validator("password_env")
    @classmethod
    def password_env_must_be_valid_name(cls, value: str | None) -> str | None:
        if value is not None and not _ENV_VAR_NAME_RE.match(value):
            raise ValueError(
                f"invalid password_env {value!r}: must be a valid environment variable name"
            )
        return value

    def resolve_password(self) -> SecretStr:
        """resolve ``password_env`` to a :class:`SecretStr` at use time.

        :return: ``SecretStr`` wrapping the resolved password
        :rtype: SecretStr
        :raises ValueError: if ``password_env`` is None (no env var configured)
            or the named env var is not set
        """
        if self.password_env is None:
            raise ValueError(
                "password_env is None; no env-var configured for this datasource. "
                "set password_env in agent.yaml's connection_config block."
            )
        return _resolve_env_to_secret(self.password_env)


class YugabyteConnectionConfig(BaseModel):
    """YugabyteDB connection config.

    shape-identical to :class:`PostgresConnectionConfig` (the asyncpg
    driver handles both). keeps a distinct discriminator so operators
    can see which backend a datasource targets. default port is
    YugabyteDB's pgwire-compatible 5433.
    """

    model_config = ConfigDict(populate_by_name=True)

    datasource_type: Literal[DataSourceType.YUGABYTE]
    host: str
    port: int = Field(default=5433, description="yugabyte pgwire port")
    database: str
    username: str | None = Field(
        default=None,
        description="yugabyte user; None falls back to the connection's default user",
    )
    password_env: str | None = Field(
        default=None,
        description="env-var NAME holding the yugabyte password (NOT the secret); "
        "None for local dev / trust-auth setups where no password is required",
    )
    pool_min_size: int = Field(
        default=1,
        description="asyncpg pool floor; lower=lighter idle, higher=no warm-up cost on first query",
    )
    pool_max_size: int = Field(
        default=5,
        description="asyncpg pool ceiling; lower=less warehouse load, higher=faster bursts",
    )
    command_timeout_seconds: int = Field(
        default=120,
        description="per-query asyncpg timeout; balances slow-but-valid queries vs runaway protection",
    )

    @field_validator("password_env")
    @classmethod
    def password_env_must_be_valid_name(cls, value: str | None) -> str | None:
        if value is not None and not _ENV_VAR_NAME_RE.match(value):
            raise ValueError(
                f"invalid password_env {value!r}: must be a valid environment variable name"
            )
        return value

    def resolve_password(self) -> SecretStr:
        """resolve ``password_env`` to a :class:`SecretStr` at use time.

        :raises ValueError: if ``password_env`` is None or the env var is not set
        """
        if self.password_env is None:
            raise ValueError(
                "password_env is None; no env-var configured for this datasource. "
                "set password_env in agent.yaml's connection_config block."
            )
        return _resolve_env_to_secret(self.password_env)


class RedshiftConnectionConfig(BaseModel):
    """Amazon Redshift connection config.

    drives the :class:`RedshiftDriver` (shard 11). Redshift's pg-protocol
    quirks make ``asyncpg`` unusable against ``information_schema.columns``
    in practice; the driver uses the AWS ``redshift-connector`` lib
    behind a bounded ``ThreadPoolExecutor`` + small connection cache
    (Redshift TLS+auth takes 1-3s/connection).

    no pool here: the driver holds a deque of warm connections sized
    by :attr:`connection_cache_size` and dispatches blocking calls
    through an ``AsyncSyncBridge`` sized by :attr:`executor_max_workers`.

    :param datasource_type: discriminator; must be ``DataSourceType.REDSHIFT``
    :param host: Redshift cluster endpoint
    :param port: Redshift cluster port (default 5439)
    :param database: database name
    :param username: redshift user
    :param password_env: env-var NAME holding the password
    :param executor_max_workers: bounded ThreadPoolExecutor size (via
        ``AsyncSyncBridge``). trade-off: lower = serialises queries
        more, less Redshift WLM pressure; higher = more concurrent
        bridge threads + more WLM slots in use
    :param connection_cache_size: keep this many warm
        ``redshift_connector.Connection`` objects per driver. trade-off:
        lower = cold-start TLS cost on most queries; higher = more
        idle Redshift sessions held
    :param query_timeout_seconds: Redshift-side ``statement_timeout``
        (in ms server-side; driver converts). trade-off: same as
        ``command_timeout_seconds`` above
    """

    model_config = ConfigDict(populate_by_name=True)

    datasource_type: Literal[DataSourceType.REDSHIFT]
    host: str
    port: int = Field(default=5439, description="redshift cluster port")
    database: str
    username: str | None = Field(
        default=None,
        description="redshift user; required for production but None is accepted "
        "so the discriminated-union shape mirrors PostgresConnectionConfig",
    )
    password_env: str | None = Field(
        default=None,
        description="env-var NAME holding the redshift password (NOT the secret); "
        "None during local fixtures only — drivers raise at use time",
    )
    executor_max_workers: int = Field(
        default=10,
        description="bounded ThreadPoolExecutor size for the async-sync bridge",
    )
    connection_cache_size: int = Field(
        default=3,
        description="warm redshift_connector connections kept per driver",
    )
    query_timeout_seconds: int = Field(
        default=300,
        description="redshift statement_timeout; caps individual queries",
    )

    @field_validator("password_env")
    @classmethod
    def password_env_must_be_valid_name(cls, value: str | None) -> str | None:
        if value is not None and not _ENV_VAR_NAME_RE.match(value):
            raise ValueError(
                f"invalid password_env {value!r}: must be a valid environment variable name"
            )
        return value

    def resolve_password(self) -> SecretStr:
        """resolve ``password_env`` to a :class:`SecretStr` at use time.

        :raises ValueError: if ``password_env`` is None or the env var is not set
        """
        if self.password_env is None:
            raise ValueError(
                "password_env is None; no env-var configured for this datasource. "
                "set password_env in agent.yaml's connection_config block."
            )
        return _resolve_env_to_secret(self.password_env)


class SnowflakeConnectionConfig(BaseModel):
    """Snowflake connection config.

    drives the :class:`SnowflakeDriver` stub today (shard 12); full
    implementation tracked separately. uses
    ``snowflake-connector-python`` behind the same ``AsyncSyncBridge``
    pattern as Redshift.

    :param datasource_type: discriminator; must be ``DataSourceType.SNOWFLAKE``
    :param account: Snowflake account identifier (no .snowflakecomputing.com suffix)
    :param warehouse: virtual warehouse name to query against
    :param user: snowflake user
    :param password_env: env-var NAME holding the password
    :param role: optional snowflake role; None uses the user's default
    :param pool_size: warm connection pool size; Snowflake auth is
        expensive so a small pool amortises it
    :param query_timeout_seconds: per-query timeout (Snowflake-side)
    """

    model_config = ConfigDict(populate_by_name=True)

    datasource_type: Literal[DataSourceType.SNOWFLAKE]
    account: str
    warehouse: str
    user: str
    password_env: str = Field(
        description="env-var NAME holding the snowflake password (NOT the secret)",
    )
    role: str | None = Field(
        default=None,
        description="optional snowflake role; None uses user default",
    )
    pool_size: int = Field(
        default=5,
        description="warm snowflake connections kept per driver",
    )
    query_timeout_seconds: int = Field(
        default=300,
        description="snowflake-side per-query timeout",
    )

    @field_validator("password_env")
    @classmethod
    def password_env_must_be_valid_name(cls, value: str) -> str:
        if not _ENV_VAR_NAME_RE.match(value):
            raise ValueError(
                f"invalid password_env {value!r}: must be a valid environment variable name"
            )
        return value

    def resolve_password(self) -> SecretStr:
        """resolve ``password_env`` to a :class:`SecretStr` at use time."""
        return _resolve_env_to_secret(self.password_env)


class BigQueryConnectionConfig(BaseModel):
    """Google BigQuery connection config.

    drives the :class:`BigQueryDriver` stub today (shard 12); full
    implementation tracked separately. BigQuery is stateless HTTPS via
    ``google-cloud-bigquery`` — no pool needed, just one
    ``bigquery.Client`` per driver. async-sync bridge wraps the sync
    REST calls.

    :param datasource_type: discriminator; must be ``DataSourceType.BIGQUERY``
    :param project_id: GCP project hosting the BigQuery datasets
    :param credentials_json_env: env-var NAME holding the service-account
        JSON blob. resolved via :meth:`resolve_credentials_json` at use
        time so the blob never lives in a ``str`` variable that could
        leak into logs.
    :param executor_max_workers: bounded ThreadPoolExecutor size for
        wrapping the sync ``google-cloud-bigquery`` REST calls
    :param query_timeout_seconds: ``QueryJobConfig.job_timeout_ms``
        ceiling (driver converts seconds -> ms)
    """

    model_config = ConfigDict(populate_by_name=True)

    datasource_type: Literal[DataSourceType.BIGQUERY]
    project_id: str
    credentials_json_env: str = Field(
        description="env-var NAME holding the GCP service-account JSON blob (NOT the secret)",
    )
    executor_max_workers: int = Field(
        default=10,
        description="bounded ThreadPoolExecutor size for the async-sync bridge",
    )
    query_timeout_seconds: int = Field(
        default=300,
        description="BigQuery job_timeout_ms upper bound (driver converts s -> ms)",
    )

    @field_validator("credentials_json_env")
    @classmethod
    def credentials_json_env_must_be_valid_name(cls, value: str) -> str:
        if not _ENV_VAR_NAME_RE.match(value):
            raise ValueError(
                f"invalid credentials_json_env {value!r}: must be a valid environment variable name"
            )
        return value

    def resolve_credentials_json(self) -> SecretStr:
        """resolve ``credentials_json_env`` to a :class:`SecretStr` at use time."""
        return _resolve_env_to_secret(self.credentials_json_env)


class AgentInternalConnectionConfig(BaseModel):
    """sentinel config for agent-created tables (data-task-01).

    carries no external connection identity because the driver
    BORROWS Hub's L3 pool (the same yugabyte cluster that backs the
    Hub itself). this config is Hub-coupled by construction —
    documented as such; if a second use case for borrowed pools
    appears, lift to a generic ``BorrowedPoolConnectionConfig`` then.

    :param datasource_type: discriminator; must be ``DataSourceType.AGENT_INTERNAL``
    :param schema_name: the ``agent_<hex>`` schema in Hub's L3 that
        agent-created tables live in. immutable per the v056 db
        constraint
    """

    model_config = ConfigDict(populate_by_name=True)

    datasource_type: Literal[DataSourceType.AGENT_INTERNAL]
    schema_name: str = Field(
        description="agent_<hex> schema in Hub's L3 owning this datasource's tables",
    )


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------


#: per-driver connection-config discriminated union keyed on ``datasource_type``.
#:
#: pydantic dispatches incoming dicts to the right member based on the
#: ``datasource_type`` field; mistyped values raise a ``ValidationError``
#: that names the union's valid members. canonical 3tears precedent:
#: ``packages/langgraph/src/threetears/langgraph/streaming.py``.
ConnectionConfig = Annotated[
    PostgresConnectionConfig
    | YugabyteConnectionConfig
    | RedshiftConnectionConfig
    | SnowflakeConnectionConfig
    | BigQueryConnectionConfig
    | AgentInternalConnectionConfig,
    Field(discriminator="datasource_type"),
]


# ---------------------------------------------------------------------------
# Agent.yaml-facing DatasourceConfig
# ---------------------------------------------------------------------------


class DatasourceConfig(BaseModel):
    """configuration for a single external data source declared in agent.yaml.

    shape (post-``datasource-task-08``):

    .. code-block:: yaml

        datasources:
          - name: central-reporting
            access_mode: read
            schemas: [reporting_prod]
            connection_config:
              datasource_type: redshift
              host: cluster.region.redshift.amazonaws.com
              port: 5439
              database: analytics
              username: ots_user
              password_env: OTS_REDSHIFT_PASSWORD
              # optional tuning (defaults shown):
              # executor_max_workers: 10
              # connection_cache_size: 3
              # query_timeout_seconds: 300

    the connection-shape fields live inside :attr:`connection_config`
    (a discriminated union); top-level fields are limited to the
    datasource's *identity* (name, access mode, schema allow-list).
    no flat fields are kept for backward compat — pre-shard-08
    ``agent.yaml`` files must be rewritten by hand at upgrade time
    (one-time migration).

    :param name: human-readable name for this data source
    :ptype name: str
    :param access_mode: tool registration mode (read / write / readwrite)
    :ptype access_mode: str
    :param schemas: database schemas exposed to agents (whitelist;
        empty means "all schemas the warehouse account can see")
    :ptype schemas: list[str]
    :param connection_config: per-driver connection config; pydantic
        dispatches on ``datasource_type`` to the right
        :class:`ConnectionConfig` member
    :ptype connection_config: ConnectionConfig
    """

    name: str
    access_mode: str = "readwrite"
    schemas: list[str] = Field(default_factory=list)
    connection_config: ConnectionConfig

    @field_validator("access_mode")
    @classmethod
    def access_mode_must_be_valid(cls, value: str) -> str:
        """validates that ``access_mode`` is one of read / write / readwrite.

        :param value: access mode string to validate
        :ptype value: str
        :return: validated access mode string
        :rtype: str
        :raises ValueError: if access mode is not valid
        """
        if value not in _VALID_ACCESS_MODES:
            raise ValueError(
                f"invalid access_mode {value!r}: must be one of read, write, readwrite"
            )
        return value

    @property
    def datasource_type(self) -> Any:
        """convenience accessor for the nested datasource_type.

        ``ds_config.datasource_type`` is the common need; the value
        actually lives in ``ds_config.connection_config.datasource_type``.
        the property keeps existing callers concise without re-exposing
        the rest of the flat surface.

        :return: the connection's ``DataSourceType`` enum member
        :rtype: DataSourceType
        """
        return self.connection_config.datasource_type
