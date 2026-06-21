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

- ``password_ref`` and similar ``*_ref`` fields carry a
  ``scheme://locator`` secret REFERENCE (e.g. ``env://PG_PASSWORD``,
  ``k8s://central-reporting/password``), not the secret value itself.
  they are typed ``str``. drivers resolve the reference via
  :meth:`resolve_password` (or :meth:`resolve_credentials_json` for
  BigQuery) which dispatches to the backend in
  :mod:`threetears.datasources.secrets` and returns :class:`SecretStr`.
  the secret value is only ever held inside a ``SecretStr`` and
  unwrapped at the last possible moment when handed to the backend lib.
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

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

from threetears.datasources.entities import DataSourceType
from threetears.datasources.secrets import resolve_secret, validate_ref

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


# valid access mode values, mirrored from
# :class:`threetears.datasources.entities.DataSourceAccessMode`. kept
# as a literal set here so the config validator stays a pure pydantic
# field check without pulling the enum module at validation time.
_VALID_ACCESS_MODES = frozenset({"read", "write", "readwrite"})

# credential resolution lives in :mod:`threetears.datasources.secrets`.
# config carries a ``scheme://locator`` reference (``password_ref`` /
# ``credentials_json_ref``); the value is resolved at use time via
# :func:`resolve_secret`. field validators call :func:`validate_ref`
# so a malformed reference fails at config-load, not at first query.


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
    :param password_ref: secret reference (scheme://locator, NOT the
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
    password_ref: str | None = Field(
        default=None,
        description="secret reference (scheme://locator, e.g. 'env://PG_PASSWORD') "
        "for the postgres password; None for local dev / trust-auth setups where "
        "no password is required",
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
    allowed_schemas: list[str] = Field(
        default_factory=list,
        description="schemas to set on the connection's ``search_path`` at open "
        "time so agents do not have to qualify table names in SQL. typically "
        "threaded from the higher-level ``DatasourceConfig.schemas`` at driver "
        'construction time. empty list means "do not issue a SET search_path; '
        "the backend's default applies\".",
    )

    @field_validator("password_ref")
    @classmethod
    def password_ref_must_be_valid(cls, value: str | None) -> str | None:
        return value if value is None else validate_ref(value)

    def resolve_password(self) -> SecretStr:
        """resolve ``password_ref`` to a :class:`SecretStr` at use time.

        :return: ``SecretStr`` wrapping the resolved password
        :rtype: SecretStr
        :raises ValueError: if ``password_ref`` is None (no credential
            configured) or the reference cannot be resolved
        """
        if self.password_ref is None:
            raise ValueError(
                "password_ref is None; no credential configured for this "
                "datasource. set password_ref (e.g. env://PG_PASSWORD) in "
                "the datasource's connection_config."
            )
        return resolve_secret(self.password_ref)


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
    password_ref: str | None = Field(
        default=None,
        description="secret reference (scheme://locator, e.g. 'env://YB_PASSWORD') "
        "for the yugabyte password; None for local dev / trust-auth setups where "
        "no password is required",
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
    allowed_schemas: list[str] = Field(
        default_factory=list,
        description="schemas to set on the connection's ``search_path`` at open "
        "time so agents do not have to qualify table names in SQL. typically "
        "threaded from the higher-level ``DatasourceConfig.schemas`` at driver "
        'construction time. empty list means "do not issue a SET search_path; '
        "the backend's default applies\".",
    )

    @field_validator("password_ref")
    @classmethod
    def password_ref_must_be_valid(cls, value: str | None) -> str | None:
        return value if value is None else validate_ref(value)

    def resolve_password(self) -> SecretStr:
        """resolve ``password_ref`` to a :class:`SecretStr` at use time.

        :raises ValueError: if ``password_ref`` is None or the reference
            cannot be resolved
        """
        if self.password_ref is None:
            raise ValueError(
                "password_ref is None; no credential configured for this "
                "datasource. set password_ref (e.g. env://YB_PASSWORD) in "
                "the datasource's connection_config."
            )
        return resolve_secret(self.password_ref)


class RedshiftConnectionConfig(BaseModel):
    """Amazon Redshift connection config.

    drives the :class:`RedshiftDriver` (shard 11). Redshift's pg-protocol
    quirks make ``asyncpg`` unusable against ``information_schema.columns``
    in practice; the driver uses the AWS ``redshift-connector`` lib
    behind a bounded ``ThreadPoolExecutor`` + small connection cache
    (Redshift TLS+auth takes 1-3s/connection).

    the warm-connection deque acts as a bounded connection pool: sized to
    the Redshift user's ``CONNECTION LIMIT`` via :attr:`connection_cache_size`
    (which defaults to :attr:`executor_max_workers` so they stay equal), it
    reuses warm connections for concurrent queries and queues any query past
    the bound on the ``AsyncSyncBridge`` executor, rather than opening a fresh
    connection that would exceed the limit. set both to the user's connection
    limit per-datasource.

    :param datasource_type: discriminator; must be ``DataSourceType.REDSHIFT``
    :param host: Redshift cluster endpoint
    :param port: Redshift cluster port (default 5439)
    :param database: database name
    :param username: redshift user
    :param password_ref: secret reference (scheme://locator) for the password
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
    password_ref: str | None = Field(
        default=None,
        description="secret reference (scheme://locator, e.g. 'env://REDSHIFT_PASSWORD') "
        "for the redshift password; None during local fixtures only — drivers raise "
        "at use time",
    )
    executor_max_workers: int = Field(
        default=5,
        description="bounded ThreadPoolExecutor size for the async-sync bridge; the "
        "effective max concurrent connections. MUST be <= the Redshift user's "
        "CONNECTION LIMIT, else concurrent queries past the limit fail with "
        "'too many connections'. conservative default sized to a typical tight "
        "per-user limit; raise per-datasource when the user allows more",
    )
    connection_cache_size: int = Field(
        default=5,
        description="warm redshift_connector connections kept per driver. defaults to "
        "executor_max_workers when not set (see validator) so every concurrent query "
        "reuses a warm connection instead of opening a fresh one past the cache -- the "
        "deque then behaves as a proper bounded pool. a smaller cache than workers "
        "forces fresh opens under load and overshoots the connection limit",
    )
    query_timeout_seconds: int = Field(
        default=300,
        description="redshift statement_timeout; caps individual queries",
    )
    allowed_schemas: list[str] = Field(
        default_factory=list,
        description="schemas to set on the connection's ``search_path`` at open "
        "time so agents do not have to qualify table names in SQL. typically "
        "threaded from the higher-level ``DatasourceConfig.schemas`` at driver "
        'construction time. empty list means "do not issue a SET search_path; '
        "the backend's default applies\". order is preserved (matches the "
        "Redshift / Postgres precedence semantics: leftmost schema wins on "
        "unqualified-name resolution).",
    )

    @field_validator("password_ref")
    @classmethod
    def password_ref_must_be_valid(cls, value: str | None) -> str | None:
        return value if value is None else validate_ref(value)

    @model_validator(mode="after")
    def _cache_defaults_to_workers(self) -> "RedshiftConnectionConfig":
        """default ``connection_cache_size`` to ``executor_max_workers`` when unset.

        a cache smaller than the worker count forces a fresh connection open on
        every concurrent query past the cache -- which overshoots a tight per-user
        Redshift ``CONNECTION LIMIT`` and fails with "too many connections". keeping
        cache == workers (both sized to the user's limit) makes the warm-connection
        deque behave as a proper bounded pool: concurrent queries reuse warm
        connections, and a query past the bound queues on the executor instead of
        opening a doomed connection. set ``connection_cache_size`` explicitly to
        diverge from ``executor_max_workers``.

        :return: self, with ``connection_cache_size`` aligned to workers when unset
        :rtype: RedshiftConnectionConfig
        """
        if "connection_cache_size" not in self.model_fields_set:
            self.connection_cache_size = self.executor_max_workers
        return self

    def resolve_password(self) -> SecretStr:
        """resolve ``password_ref`` to a :class:`SecretStr` at use time.

        :raises ValueError: if ``password_ref`` is None or the reference
            cannot be resolved
        """
        if self.password_ref is None:
            raise ValueError(
                "password_ref is None; no credential configured for this "
                "datasource. set password_ref (e.g. env://REDSHIFT_PASSWORD) "
                "in the datasource's connection_config."
            )
        return resolve_secret(self.password_ref)


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
    :param password_ref: secret reference (scheme://locator) for the password
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
    password_ref: str = Field(
        description="secret reference (scheme://locator, e.g. 'env://SNOWFLAKE_PASSWORD') "
        "for the snowflake password (NOT the secret itself)",
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

    @field_validator("password_ref")
    @classmethod
    def password_ref_must_be_valid(cls, value: str) -> str:
        return validate_ref(value)

    def resolve_password(self) -> SecretStr:
        """resolve ``password_ref`` to a :class:`SecretStr` at use time."""
        return resolve_secret(self.password_ref)


class BigQueryConnectionConfig(BaseModel):
    """Google BigQuery connection config.

    drives the :class:`BigQueryDriver` stub today (shard 12); full
    implementation tracked separately. BigQuery is stateless HTTPS via
    ``google-cloud-bigquery`` — no pool needed, just one
    ``bigquery.Client`` per driver. async-sync bridge wraps the sync
    REST calls.

    :param datasource_type: discriminator; must be ``DataSourceType.BIGQUERY``
    :param project_id: GCP project hosting the BigQuery datasets
    :param credentials_json_ref: secret reference (scheme://locator) for
        the service-account JSON blob. resolved via
        :meth:`resolve_credentials_json` at use time so the blob never
        lives in a ``str`` variable that could leak into logs.
    :param executor_max_workers: bounded ThreadPoolExecutor size for
        wrapping the sync ``google-cloud-bigquery`` REST calls
    :param query_timeout_seconds: ``QueryJobConfig.job_timeout_ms``
        ceiling (driver converts seconds -> ms)
    """

    model_config = ConfigDict(populate_by_name=True)

    datasource_type: Literal[DataSourceType.BIGQUERY]
    project_id: str
    credentials_json_ref: str = Field(
        description="secret reference (scheme://locator, e.g. 'env://GCP_SA_JSON') "
        "for the GCP service-account JSON blob (NOT the secret itself)",
    )
    executor_max_workers: int = Field(
        default=10,
        description="bounded ThreadPoolExecutor size for the async-sync bridge",
    )
    query_timeout_seconds: int = Field(
        default=300,
        description="BigQuery job_timeout_ms upper bound (driver converts s -> ms)",
    )

    @field_validator("credentials_json_ref")
    @classmethod
    def credentials_json_ref_must_be_valid(cls, value: str) -> str:
        return validate_ref(value)

    def resolve_credentials_json(self) -> SecretStr:
        """resolve ``credentials_json_ref`` to a :class:`SecretStr` at use time."""
        return resolve_secret(self.credentials_json_ref)


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
    """configuration for a single declared data source.

    one model serves two surfaces post-``datasource-task-05``:

    1. **definition shape** (``aibots datasource apply -f <file>``
       and the legacy / deprecated inline-in-``agent.yaml``
       pattern). carries the full connection identity:

       .. code-block:: yaml

           - name: central-reporting
             access_mode: read
             schemas: [reporting_prod]
             connection_config:
               datasource_type: redshift
               host: cluster.region.redshift.amazonaws.com
               port: 5439
               database: analytics
               username: ots_user
               password_ref: env://OTS_REDSHIFT_PASSWORD

    2. **reference shape** (canonical in ``agent.yaml``
       post-``datasource-task-05``). identity-only; the connection
       config lives in a sibling ``datasources/<name>.yaml`` applied
       via ``aibots datasource apply``:

       .. code-block:: yaml

           - name: central-reporting
             access_mode: read

    the model accepts both by making ``connection_config`` optional.
    ``schemas`` is also optional in both shapes (empty means "all
    schemas the warehouse account can see").

    the post-shard-01 deprecation predicate (
    :func:`aibots_agents.devx.datasource_provision._has_inline_connection_config`
    ) keys off ``connection_config is not None`` -- a populated
    inline block in ``agent.yaml`` triggers a one-shot
    ``DeprecationWarning`` pointing at the new CLI; the reference
    shape stays silent.

    :param name: human-readable name for this data source
    :ptype name: str
    :param access_mode: tool registration mode (read / write / readwrite)
    :ptype access_mode: str
    :param schemas: database schemas exposed to agents (whitelist;
        empty means "all schemas the warehouse account can see").
        IGNORED on the reference shape -- the canonical
        ``schemas`` field for that datasource is read from the
        applied definition YAML
    :ptype schemas: list[str]
    :param connection_config: per-driver connection config (
        discriminated on ``datasource_type``). ``None`` on the
        reference shape; required on the definition shape
    :ptype connection_config: ConnectionConfig | None
    """

    name: str
    access_mode: str = "readwrite"
    schemas: list[str] = Field(default_factory=list)
    connection_config: ConnectionConfig | None = None

    # reject extra keys so a typo in the reference shape ("acccess_mode")
    # surfaces at load time instead of silently shipping the default.
    model_config = ConfigDict(extra="forbid")

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
            raise ValueError(f"invalid access_mode {value!r}: must be one of read, write, readwrite")
        return value

    @property
    def is_reference(self) -> bool:
        """``True`` when the entry is the reference-only shape (no inline config).

        :return: reference-only shape indicator
        :rtype: bool
        """
        return self.connection_config is None

    @property
    def datasource_type(self) -> Any:
        """convenience accessor for the nested datasource_type.

        only valid on the definition shape -- the reference shape
        does not carry an inline connection config, and the type
        information lives Hub-side. callers that need the type for
        a reference must read it from the resolved
        :class:`DataSourceResponse`.

        :return: the connection's ``DataSourceType`` enum member
        :rtype: DataSourceType
        :raises AttributeError: when called on the reference shape
        """
        if self.connection_config is None:
            raise AttributeError(
                f"datasource {self.name!r} is the reference-only shape; "
                f"no inline connection_config. read datasource_type from "
                f"the resolved DataSourceResponse instead.",
            )
        return self.connection_config.datasource_type
