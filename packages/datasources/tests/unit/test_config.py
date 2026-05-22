"""tests for threetears.datasources.config.

covers:

- ``DatasourceConfig`` new (post-shard-08) nested ``connection_config:`` shape
- six per-driver ``ConnectionConfig`` members + discriminator routing
- default-value pinning for every documented pool/executor/timeout knob
- ``SecretStr`` semantics on ``resolve_password`` /
  ``resolve_credentials_json`` (env-var-name -> opaque secret)
- secret-redaction sanity: ``repr(config)`` and ``str(config)`` do not
  expose either the env-var-resolved password OR the env-var name
  itself in a misleading way
- access-mode + ``password_env`` validators
- round-trip via ``model_dump`` / ``model_validate``
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from threetears.datasources.config import (
    AgentInternalConnectionConfig,
    BigQueryConnectionConfig,
    ConnectionConfig,
    DatasourceConfig,
    PostgresConnectionConfig,
    RedshiftConnectionConfig,
    SnowflakeConnectionConfig,
    YugabyteConnectionConfig,
)
from threetears.datasources.entities import DataSourceType


# ---------------------------------------------------------------------------
# Per-driver ConnectionConfig members
# ---------------------------------------------------------------------------


class TestPostgresConnectionConfig:
    """postgres config: asyncpg-flavored knobs with documented defaults."""

    def test_minimal(self) -> None:
        cfg = PostgresConnectionConfig(
            datasource_type=DataSourceType.POSTGRES,
            host="h",
            database="d",
            username="u",
            password_env="PW_ENV",
        )
        assert cfg.port == 5432
        assert cfg.pool_min_size == 1
        assert cfg.pool_max_size == 5
        assert cfg.command_timeout_seconds == 120

    def test_resolve_password_returns_secret_str(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MY_PW_ENV", "horse-battery-staple")
        cfg = PostgresConnectionConfig(
            datasource_type=DataSourceType.POSTGRES,
            host="h", database="d", username="u",
            password_env="MY_PW_ENV",
        )
        secret = cfg.resolve_password()
        assert isinstance(secret, SecretStr)
        assert secret.get_secret_value() == "horse-battery-staple"

    def test_resolve_password_raises_when_env_unset(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("MISSING_PW_ENV", raising=False)
        cfg = PostgresConnectionConfig(
            datasource_type=DataSourceType.POSTGRES,
            host="h", database="d", username="u",
            password_env="MISSING_PW_ENV",
        )
        with pytest.raises(ValueError, match="MISSING_PW_ENV"):
            cfg.resolve_password()

    def test_password_env_validator_rejects_garbage(self) -> None:
        with pytest.raises(ValidationError):
            PostgresConnectionConfig(
                datasource_type=DataSourceType.POSTGRES,
                host="h", database="d", username="u",
                password_env="has space",
            )


class TestYugabyteConnectionConfig:
    """yugabyte shape-mirrors postgres but uses port 5433 by default."""

    def test_default_port_is_5433(self) -> None:
        cfg = YugabyteConnectionConfig(
            datasource_type=DataSourceType.YUGABYTE,
            host="h", database="d", username="u",
            password_env="PW",
        )
        assert cfg.port == 5433

    def test_shape_matches_postgres_knobs(self) -> None:
        # field-name parity: pool_min_size, pool_max_size, command_timeout_seconds
        cfg = YugabyteConnectionConfig(
            datasource_type=DataSourceType.YUGABYTE,
            host="h", database="d", username="u",
            password_env="PW",
        )
        assert hasattr(cfg, "pool_min_size")
        assert hasattr(cfg, "pool_max_size")
        assert hasattr(cfg, "command_timeout_seconds")


class TestRedshiftConnectionConfig:
    """redshift: executor + cache (no pool); 5439 port."""

    def test_minimal(self) -> None:
        cfg = RedshiftConnectionConfig(
            datasource_type=DataSourceType.REDSHIFT,
            host="cluster.region.redshift.amazonaws.com",
            database="analytics",
            username="ots_user",
            password_env="OTS_REDSHIFT_PASSWORD",
        )
        assert cfg.port == 5439
        assert cfg.executor_max_workers == 10
        assert cfg.connection_cache_size == 3
        assert cfg.query_timeout_seconds == 300

    def test_no_pool_knobs(self) -> None:
        # negative assertion: redshift does NOT take pool_min_size /
        # pool_max_size; symmetry-by-naming would hide real differences
        cfg = RedshiftConnectionConfig(
            datasource_type=DataSourceType.REDSHIFT,
            host="h", database="d", username="u",
            password_env="PW",
        )
        assert not hasattr(cfg, "pool_min_size")
        assert not hasattr(cfg, "pool_max_size")


class TestSnowflakeConnectionConfig:
    """snowflake: account / warehouse / optional role; pool-sized."""

    def test_minimal(self) -> None:
        cfg = SnowflakeConnectionConfig(
            datasource_type=DataSourceType.SNOWFLAKE,
            account="abc12345",
            warehouse="WH_ANALYTICS",
            user="ots_user",
            password_env="SF_PW",
        )
        assert cfg.role is None
        assert cfg.pool_size == 5
        assert cfg.query_timeout_seconds == 300

    def test_role_carries_when_set(self) -> None:
        cfg = SnowflakeConnectionConfig(
            datasource_type=DataSourceType.SNOWFLAKE,
            account="acc", warehouse="WH", user="u",
            password_env="SF_PW",
            role="ANALYST_RO",
        )
        assert cfg.role == "ANALYST_RO"


class TestBigQueryConnectionConfig:
    """bigquery: stateless HTTPS; credentials_json_env not password_env."""

    def test_minimal(self) -> None:
        cfg = BigQueryConnectionConfig(
            datasource_type=DataSourceType.BIGQUERY,
            project_id="my-project",
            credentials_json_env="GCP_SA_JSON",
        )
        assert cfg.executor_max_workers == 10
        assert cfg.query_timeout_seconds == 300

    def test_resolve_credentials_json_returns_secret_str(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        blob = '{"type":"service_account","private_key":"FAKE"}'
        monkeypatch.setenv("GCP_SA_JSON", blob)
        cfg = BigQueryConnectionConfig(
            datasource_type=DataSourceType.BIGQUERY,
            project_id="p", credentials_json_env="GCP_SA_JSON",
        )
        secret = cfg.resolve_credentials_json()
        assert isinstance(secret, SecretStr)
        assert secret.get_secret_value() == blob

    def test_no_password_env(self) -> None:
        # negative assertion: bigquery doesn't take password_env (the
        # SA-JSON blob is the credential); resists "let me add it
        # for symmetry" drift
        cfg = BigQueryConnectionConfig(
            datasource_type=DataSourceType.BIGQUERY,
            project_id="p", credentials_json_env="GCP_SA_JSON",
        )
        assert not hasattr(cfg, "password_env")


class TestAgentInternalConnectionConfig:
    """agent_internal: no external identity; just the schema_name."""

    def test_minimal(self) -> None:
        cfg = AgentInternalConnectionConfig(
            datasource_type=DataSourceType.AGENT_INTERNAL,
            schema_name="agent_abc123",
        )
        assert cfg.schema_name == "agent_abc123"

    def test_no_connection_identity(self) -> None:
        cfg = AgentInternalConnectionConfig(
            datasource_type=DataSourceType.AGENT_INTERNAL,
            schema_name="agent_xyz",
        )
        # documented invariant: no host, port, database, username,
        # password_env on this config — the driver borrows Hub's L3
        # pool via the factory's hub_l3_pool= kwarg
        assert not hasattr(cfg, "host")
        assert not hasattr(cfg, "port")
        assert not hasattr(cfg, "database")
        assert not hasattr(cfg, "username")
        assert not hasattr(cfg, "password_env")


# ---------------------------------------------------------------------------
# Discriminator routing
# ---------------------------------------------------------------------------


class TestConnectionConfigDiscriminator:
    """pydantic routes incoming dicts to the right member on ``datasource_type``."""

    def test_routes_postgres(self) -> None:
        cfg = _validate({
            "datasource_type": "postgres",
            "host": "h", "database": "d", "username": "u", "password_env": "PW",
        })
        assert isinstance(cfg, PostgresConnectionConfig)

    def test_routes_yugabyte(self) -> None:
        cfg = _validate({
            "datasource_type": "yugabyte",
            "host": "h", "database": "d", "username": "u", "password_env": "PW",
        })
        assert isinstance(cfg, YugabyteConnectionConfig)

    def test_routes_redshift(self) -> None:
        cfg = _validate({
            "datasource_type": "redshift",
            "host": "h", "database": "d", "username": "u", "password_env": "PW",
        })
        assert isinstance(cfg, RedshiftConnectionConfig)

    def test_routes_snowflake(self) -> None:
        cfg = _validate({
            "datasource_type": "snowflake",
            "account": "a", "warehouse": "w", "user": "u", "password_env": "PW",
        })
        assert isinstance(cfg, SnowflakeConnectionConfig)

    def test_routes_bigquery(self) -> None:
        cfg = _validate({
            "datasource_type": "bigquery",
            "project_id": "p", "credentials_json_env": "GCP",
        })
        assert isinstance(cfg, BigQueryConnectionConfig)

    def test_routes_agent_internal(self) -> None:
        cfg = _validate({
            "datasource_type": "agent_internal",
            "schema_name": "agent_abc",
        })
        assert isinstance(cfg, AgentInternalConnectionConfig)

    def test_unknown_discriminator_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _validate({
                "datasource_type": "duckdb",  # not in the union
                "host": "h", "database": "d", "username": "u", "password_env": "PW",
            })


def _validate(raw: dict) -> ConnectionConfig:
    """validate a raw dict through the discriminated union for tests.

    pydantic v2 doesn't expose a top-level ``model_validate`` on an
    ``Annotated[Union, Field(discriminator=...)]``; round-trip through
    a wrapper model.
    """
    from pydantic import BaseModel

    class _Wrap(BaseModel):
        cfg: ConnectionConfig

    return _Wrap.model_validate({"cfg": raw}).cfg


# ---------------------------------------------------------------------------
# DatasourceConfig (post-shard-08 nested shape)
# ---------------------------------------------------------------------------


class TestDatasourceConfigNestedShape:
    """post-shard-08 shape: connection_config is nested, not flat."""

    def test_minimal_redshift(self) -> None:
        cfg = DatasourceConfig.model_validate({
            "name": "central-reporting",
            "schemas": ["reporting_prod"],
            "access_mode": "read",
            "connection_config": {
                "datasource_type": "redshift",
                "host": "h",
                "database": "d",
                "username": "u",
                "password_env": "OTS_REDSHIFT_PASSWORD",
            },
        })
        assert cfg.name == "central-reporting"
        assert cfg.access_mode == "read"
        assert isinstance(cfg.connection_config, RedshiftConnectionConfig)
        # convenience property exposes the nested datasource_type
        assert cfg.datasource_type == DataSourceType.REDSHIFT

    def test_no_flat_fields_accepted(self) -> None:
        # pre-shard-08 flat shape MUST fail (no aliasing)
        with pytest.raises(ValidationError):
            DatasourceConfig.model_validate({
                "name": "x",
                "type": "redshift",
                "host": "h",
                "database": "d",
                "username": "u",
                "password_env": "PW",
            })

    def test_access_mode_validator(self) -> None:
        with pytest.raises(ValidationError):
            DatasourceConfig.model_validate({
                "name": "x",
                "access_mode": "admin",  # invalid
                "connection_config": {
                    "datasource_type": "redshift",
                    "host": "h", "database": "d", "username": "u",
                    "password_env": "PW",
                },
            })


# ---------------------------------------------------------------------------
# Secret-redaction sanity
# ---------------------------------------------------------------------------


class TestSecretRedaction:
    """repr() / str() over a populated config don't expose secret values."""

    def test_resolved_secret_str_redacted(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MY_PW_ENV", "horse-battery-staple")
        cfg = PostgresConnectionConfig(
            datasource_type=DataSourceType.POSTGRES,
            host="h", database="d", username="u",
            password_env="MY_PW_ENV",
        )
        secret = cfg.resolve_password()
        # SecretStr's repr / str redacts to '**********'
        assert "horse-battery-staple" not in repr(secret)
        assert "horse-battery-staple" not in str(secret)

    def test_password_env_name_appears_but_not_resolved_value(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # the env-var NAME is safe to log; the resolved VALUE is not.
        # repr(config) shows the name; resolve_password() returns the
        # SecretStr that redacts.
        monkeypatch.setenv("MY_PW_ENV", "horse-battery-staple")
        cfg = PostgresConnectionConfig(
            datasource_type=DataSourceType.POSTGRES,
            host="h", database="d", username="u",
            password_env="MY_PW_ENV",
        )
        rendered = repr(cfg)
        # env-var NAME is fine to surface
        assert "MY_PW_ENV" in rendered
        # resolved value MUST NOT appear in any debug rendering of
        # the config itself
        assert "horse-battery-staple" not in rendered


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """``model_dump_json`` -> ``model_validate_json`` lossless for every member."""

    @pytest.mark.parametrize(
        "raw",
        [
            {"datasource_type": "postgres", "host": "h", "database": "d", "username": "u", "password_env": "PW"},
            {"datasource_type": "yugabyte", "host": "h", "database": "d", "username": "u", "password_env": "PW"},
            {"datasource_type": "redshift", "host": "h", "database": "d", "username": "u", "password_env": "PW"},
            {"datasource_type": "snowflake", "account": "a", "warehouse": "w", "user": "u", "password_env": "PW"},
            {"datasource_type": "bigquery", "project_id": "p", "credentials_json_env": "GCP"},
            {"datasource_type": "agent_internal", "schema_name": "agent_abc"},
        ],
    )
    def test_each_member_roundtrips(self, raw: dict) -> None:
        original = _validate(raw)
        dumped = original.model_dump(mode="json")
        restored = _validate(dumped)
        assert restored == original

    def test_datasource_config_roundtrips(self) -> None:
        original = DatasourceConfig.model_validate({
            "name": "central-reporting",
            "schemas": ["reporting_prod"],
            "access_mode": "read",
            "connection_config": {
                "datasource_type": "redshift",
                "host": "h.example.com",
                "port": 5439,
                "database": "analytics",
                "username": "ots_user",
                "password_env": "OTS_REDSHIFT_PASSWORD",
            },
        })
        dumped = original.model_dump(mode="json")
        restored = DatasourceConfig.model_validate(dumped)
        assert restored == original
