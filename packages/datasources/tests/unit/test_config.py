"""tests for threetears.datasources.config.

covers:

- ``DatasourceConfig`` new (post-shard-08) nested ``connection_config:`` shape
- six per-driver ``ConnectionConfig`` members + discriminator routing
- default-value pinning for every documented pool/executor/timeout knob
- ``SecretStr`` semantics on ``resolve_password`` /
  ``resolve_credentials_json`` (``scheme://locator`` ref -> opaque secret)
- secret-redaction sanity: ``repr(config)`` and ``str(config)`` do not
  expose the resolved password; the reference string itself is safe
- access-mode + ``password_ref`` validators
- round-trip via ``model_dump`` / ``model_validate``
- the stored form (only the fields that were set) reads back unchanged and
  stays readable by an earlier release
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError, create_model

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
            password_ref="env://PW_ENV",
        )
        assert cfg.port == 5432
        assert cfg.pool_min_size == 1
        assert cfg.pool_max_size == 5
        assert cfg.command_timeout_seconds == 120

    def test_password_ref_optional(self) -> None:
        # trust-auth / local dev: no credential at all
        cfg = PostgresConnectionConfig(
            datasource_type=DataSourceType.POSTGRES,
            host="h",
            database="d",
            username="u",
        )
        assert cfg.password_ref is None

    def test_resolve_password_returns_secret_str(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MY_PW_ENV", "horse-battery-staple")
        cfg = PostgresConnectionConfig(
            datasource_type=DataSourceType.POSTGRES,
            host="h",
            database="d",
            username="u",
            password_ref="env://MY_PW_ENV",
        )
        secret = cfg.resolve_password()
        assert isinstance(secret, SecretStr)
        assert secret.get_secret_value() == "horse-battery-staple"

    def test_resolve_password_raises_when_env_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("MISSING_PW_ENV", raising=False)
        cfg = PostgresConnectionConfig(
            datasource_type=DataSourceType.POSTGRES,
            host="h",
            database="d",
            username="u",
            password_ref="env://MISSING_PW_ENV",
        )
        with pytest.raises(ValueError, match="MISSING_PW_ENV"):
            cfg.resolve_password()

    def test_resolve_password_raises_when_ref_none(self) -> None:
        cfg = PostgresConnectionConfig(
            datasource_type=DataSourceType.POSTGRES,
            host="h",
            database="d",
            username="u",
        )
        with pytest.raises(ValueError, match="password_ref is None"):
            cfg.resolve_password()

    def test_password_ref_validator_rejects_bare_name(self) -> None:
        # a bare env-var name is no longer a valid reference; it must
        # carry a scheme (``env://NAME``).
        with pytest.raises(ValidationError):
            PostgresConnectionConfig(
                datasource_type=DataSourceType.POSTGRES,
                host="h",
                database="d",
                username="u",
                password_ref="PW_ENV",
            )

    def test_password_ref_validator_rejects_garbage_env_name(self) -> None:
        with pytest.raises(ValidationError):
            PostgresConnectionConfig(
                datasource_type=DataSourceType.POSTGRES,
                host="h",
                database="d",
                username="u",
                password_ref="env://has space",
            )

    def test_password_ref_validator_rejects_unknown_scheme(self) -> None:
        with pytest.raises(ValidationError):
            PostgresConnectionConfig(
                datasource_type=DataSourceType.POSTGRES,
                host="h",
                database="d",
                username="u",
                password_ref="duckdb://nope",
            )

    def test_k8s_ref_accepted(self) -> None:
        # the k8s scheme validates at load time without touching the fs
        cfg = PostgresConnectionConfig(
            datasource_type=DataSourceType.POSTGRES,
            host="h",
            database="d",
            username="u",
            password_ref="k8s://central-reporting/password",
        )
        assert cfg.password_ref == "k8s://central-reporting/password"


class TestYugabyteConnectionConfig:
    """yugabyte shape-mirrors postgres but uses port 5433 by default."""

    def test_default_port_is_5433(self) -> None:
        cfg = YugabyteConnectionConfig(
            datasource_type=DataSourceType.YUGABYTE,
            host="h",
            database="d",
            username="u",
            password_ref="env://PW",
        )
        assert cfg.port == 5433

    def test_shape_matches_postgres_knobs(self) -> None:
        # field-name parity: pool_min_size, pool_max_size, command_timeout_seconds
        cfg = YugabyteConnectionConfig(
            datasource_type=DataSourceType.YUGABYTE,
            host="h",
            database="d",
            username="u",
            password_ref="env://PW",
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
            password_ref="env://OTS_REDSHIFT_PASSWORD",
        )
        assert cfg.port == 5439
        # conservative pool default sized to a typical tight per-user
        # CONNECTION LIMIT; cache defaults to workers so they stay equal.
        assert cfg.executor_max_workers == 5
        assert cfg.connection_cache_size == 5
        assert cfg.query_timeout_seconds == 300

    def test_cache_defaults_to_workers_when_unset(self) -> None:
        # raising workers without setting cache keeps the pool sized as one:
        # cache follows workers so concurrency never opens past the cache.
        cfg = RedshiftConnectionConfig(
            datasource_type=DataSourceType.REDSHIFT,
            host="cluster.region.redshift.amazonaws.com",
            database="analytics",
            username="ots_user",
            password_ref="env://OTS_REDSHIFT_PASSWORD",
            executor_max_workers=15,
        )
        assert cfg.connection_cache_size == 15

    def test_cache_explicit_overrides_workers_default(self) -> None:
        # an explicit cache value is honored (diverging from workers).
        cfg = RedshiftConnectionConfig(
            datasource_type=DataSourceType.REDSHIFT,
            host="cluster.region.redshift.amazonaws.com",
            database="analytics",
            username="ots_user",
            password_ref="env://OTS_REDSHIFT_PASSWORD",
            executor_max_workers=15,
            connection_cache_size=4,
        )
        assert cfg.connection_cache_size == 4

    def test_no_pool_knobs(self) -> None:
        # negative assertion: redshift does NOT take pool_min_size /
        # pool_max_size; symmetry-by-naming would hide real differences
        cfg = RedshiftConnectionConfig(
            datasource_type=DataSourceType.REDSHIFT,
            host="h",
            database="d",
            username="u",
            password_ref="env://PW",
        )
        assert not hasattr(cfg, "pool_min_size")
        assert not hasattr(cfg, "pool_max_size")

    def test_sslmode_defaults_to_verify_ca(self) -> None:
        # default preserves the redshift_connector library default, so existing
        # callers are unaffected by the new field.
        cfg = RedshiftConnectionConfig(
            datasource_type=DataSourceType.REDSHIFT,
            host="cluster.region.redshift.amazonaws.com",
            database="analytics",
            username="ots_user",
            password_ref="env://OTS_REDSHIFT_PASSWORD",
        )
        assert cfg.sslmode == "verify-ca"

    def test_sslmode_verify_full_is_accepted(self) -> None:
        # verify-full is required to reach a proxy-fronted cluster (e.g. Satori).
        cfg = RedshiftConnectionConfig(
            datasource_type=DataSourceType.REDSHIFT,
            host="cluster.satori.example",
            database="analytics",
            username="ots_user",
            password_ref="env://OTS_REDSHIFT_PASSWORD",
            sslmode="verify-full",
        )
        assert cfg.sslmode == "verify-full"

    def test_sslmode_rejects_unsupported_mode(self) -> None:
        # redshift_connector supports only verify-ca / verify-full; an unsupported
        # libpq mode (e.g. 'require') is a config error, not a silent passthrough.
        with pytest.raises(ValidationError):
            RedshiftConnectionConfig(
                datasource_type=DataSourceType.REDSHIFT,
                host="h",
                database="d",
                username="u",
                password_ref="env://PW",
                sslmode="require",
            )


class TestSnowflakeConnectionConfig:
    """snowflake: account / warehouse / optional role; pool-sized."""

    def test_minimal(self) -> None:
        cfg = SnowflakeConnectionConfig(
            datasource_type=DataSourceType.SNOWFLAKE,
            account="abc12345",
            warehouse="WH_ANALYTICS",
            user="ots_user",
            password_ref="env://SF_PW",
        )
        assert cfg.role is None
        assert cfg.pool_size == 5
        assert cfg.query_timeout_seconds == 300

    def test_role_carries_when_set(self) -> None:
        cfg = SnowflakeConnectionConfig(
            datasource_type=DataSourceType.SNOWFLAKE,
            account="acc",
            warehouse="WH",
            user="u",
            password_ref="env://SF_PW",
            role="ANALYST_RO",
        )
        assert cfg.role == "ANALYST_RO"

    def test_password_ref_required(self) -> None:
        # unlike postgres, snowflake requires a credential reference
        with pytest.raises(ValidationError):
            SnowflakeConnectionConfig(
                datasource_type=DataSourceType.SNOWFLAKE,
                account="acc",
                warehouse="WH",
                user="u",
            )


class TestBigQueryConnectionConfig:
    """bigquery: stateless HTTPS; credentials_json_ref not password_ref."""

    def test_minimal(self) -> None:
        cfg = BigQueryConnectionConfig(
            datasource_type=DataSourceType.BIGQUERY,
            project_id="my-project",
            credentials_json_ref="env://GCP_SA_JSON",
        )
        assert cfg.executor_max_workers == 10
        assert cfg.query_timeout_seconds == 300

    def test_resolve_credentials_json_returns_secret_str(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        blob = '{"type":"service_account","private_key":"FAKE"}'
        monkeypatch.setenv("GCP_SA_JSON", blob)
        cfg = BigQueryConnectionConfig(
            datasource_type=DataSourceType.BIGQUERY,
            project_id="p",
            credentials_json_ref="env://GCP_SA_JSON",
        )
        secret = cfg.resolve_credentials_json()
        assert isinstance(secret, SecretStr)
        assert secret.get_secret_value() == blob

    def test_no_password_ref(self) -> None:
        # negative assertion: bigquery doesn't take password_ref (the
        # SA-JSON blob is the credential); resists "let me add it
        # for symmetry" drift
        cfg = BigQueryConnectionConfig(
            datasource_type=DataSourceType.BIGQUERY,
            project_id="p",
            credentials_json_ref="env://GCP_SA_JSON",
        )
        assert not hasattr(cfg, "password_ref")


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
        # password_ref on this config — the driver borrows Hub's L3
        # pool via the factory's hub_l3_pool= kwarg
        assert not hasattr(cfg, "host")
        assert not hasattr(cfg, "port")
        assert not hasattr(cfg, "database")
        assert not hasattr(cfg, "username")
        assert not hasattr(cfg, "password_ref")


# ---------------------------------------------------------------------------
# Unknown-key policy
# ---------------------------------------------------------------------------


#: one minimal, VALID body per union member, plus the unknown key that member
#: is probed with. the unknown keys are near-misses of real fields rather than
#: invented words: a transposed character is the slip that actually happens,
#: and it is the one that silently reverts a derived pool size or timeout to a
#: library default.
_UNKNOWN_KEY_PROBES: list[tuple[str, dict[str, object], str]] = [
    (
        "postgres",
        {"datasource_type": "postgres", "host": "h", "database": "d", "username": "u"},
        "pool_max_sizes",
    ),
    (
        "yugabyte",
        {"datasource_type": "yugabyte", "host": "h", "database": "d", "username": "u"},
        "command_timeout_second",
    ),
    (
        "redshift",
        {"datasource_type": "redshift", "host": "h", "database": "d", "username": "u"},
        "query_timeout_secondz",
    ),
    (
        "snowflake",
        {
            "datasource_type": "snowflake",
            "account": "a",
            "warehouse": "w",
            "user": "u",
            "password_ref": "env://PW",
        },
        "pool_sizes",
    ),
    (
        "bigquery",
        {
            "datasource_type": "bigquery",
            "project_id": "p",
            "credentials_json_ref": "env://SA",
        },
        "executor_max_worker",
    ),
    (
        "agent_internal",
        {"datasource_type": "agent_internal", "schema_name": "agent_abc"},
        "schema_names",
    ),
]


class TestConnectionConfigRefusesUnknownKeys:
    """a misspelled connection-config key is an error, never a default.

    ``DatasourceConfig`` has forbidden extras since ``datasource-task-05``
    and says why in its own comment: a typo in the reference shape must
    surface at load time instead of silently shipping the default. the
    nested per-driver members carried only ``populate_by_name=True``, so
    the rule stopped at the outer model and every pool-sizing and timeout
    value underneath it was silently droppable.

    refusal is the right policy here, unlike the relations wire that
    crosses Hub and SDK releases: tolerating an unknown key would let a
    reader run without a setting someone chose. the one version boundary
    a stored config does cross -- a consumer rolled back to an earlier
    release -- is kept open by storing only the fields that were set;
    :class:`TestStoredForm` pins that.
    """

    @pytest.mark.parametrize(("label", "body", "unknown_key"), _UNKNOWN_KEY_PROBES)
    def test_member_refuses_an_unknown_key(
        self,
        label: str,
        body: dict[str, object],
        unknown_key: str,
    ) -> None:
        """each union member rejects a near-miss key rather than defaulting it.

        :param label: union member under test, for the failure message
        :ptype label: str
        :param body: minimal valid body for that member
        :ptype body: dict[str, object]
        :param unknown_key: near-miss key the member must refuse
        :ptype unknown_key: str
        :return: none
        :rtype: None
        """
        assert _validate(dict(body)) is not None, f"{label} probe body is not itself valid"
        with pytest.raises(ValidationError, match="(?i)extra|unexpected|not permitted"):
            _validate({**body, unknown_key: 1})

    def test_a_transposed_timeout_does_not_silently_become_the_default(self) -> None:
        """the sharpest case, asserted on the VALUE rather than on the raise.

        the build datasource declares ``query_timeout_seconds: 14400``. one
        transposed character used to leave the model reading 300 -- the
        interactive default -- so a multi-hour qualification CTAS was killed
        at five minutes, hours into a build, with the file on disk still
        reading 14400. asserting the refusal here rather than the resulting
        value is what keeps the fix from being a comment.

        :return: none
        :rtype: None
        """
        good = RedshiftConnectionConfig.model_validate(
            {
                "datasource_type": "redshift",
                "host": "h",
                "database": "d",
                "username": "u",
                "query_timeout_seconds": 14400,
            }
        )
        assert good.query_timeout_seconds == 14400

        with pytest.raises(ValidationError, match="(?i)extra|unexpected|not permitted"):
            RedshiftConnectionConfig.model_validate(
                {
                    "datasource_type": "redshift",
                    "host": "h",
                    "database": "d",
                    "username": "u",
                    "query_timeout_secondz": 14400,
                }
            )

    def test_the_nested_refusal_survives_the_outer_model(self) -> None:
        """the refusal fires through ``DatasourceConfig``, not just standalone.

        the union is reached by discriminator dispatch from the parent, and a
        policy that only holds when the member is validated directly would
        never fire on the path an authored ``datasources/*.yaml`` actually
        takes.

        :return: none
        :rtype: None
        """
        with pytest.raises(ValidationError, match="(?i)extra|unexpected|not permitted"):
            DatasourceConfig.model_validate(
                {
                    "name": "seam-probe",
                    "access_mode": "build",
                    "connection_config": {
                        "datasource_type": "redshift",
                        "host": "example.invalid",
                        "database": "analytics",
                        "username": "probe",
                        "query_timeout_secondz": 14400,
                        "connection_cache_sizes": 5,
                    },
                }
            )


# ---------------------------------------------------------------------------
# Discriminator routing
# ---------------------------------------------------------------------------


class TestConnectionConfigDiscriminator:
    """pydantic routes incoming dicts to the right member on ``datasource_type``."""

    def test_routes_postgres(self) -> None:
        cfg = _validate(
            {
                "datasource_type": "postgres",
                "host": "h",
                "database": "d",
                "username": "u",
                "password_ref": "env://PW",
            }
        )
        assert isinstance(cfg, PostgresConnectionConfig)

    def test_routes_yugabyte(self) -> None:
        cfg = _validate(
            {
                "datasource_type": "yugabyte",
                "host": "h",
                "database": "d",
                "username": "u",
                "password_ref": "env://PW",
            }
        )
        assert isinstance(cfg, YugabyteConnectionConfig)

    def test_routes_redshift(self) -> None:
        cfg = _validate(
            {
                "datasource_type": "redshift",
                "host": "h",
                "database": "d",
                "username": "u",
                "password_ref": "env://PW",
            }
        )
        assert isinstance(cfg, RedshiftConnectionConfig)

    def test_routes_snowflake(self) -> None:
        cfg = _validate(
            {
                "datasource_type": "snowflake",
                "account": "a",
                "warehouse": "w",
                "user": "u",
                "password_ref": "env://PW",
            }
        )
        assert isinstance(cfg, SnowflakeConnectionConfig)

    def test_routes_bigquery(self) -> None:
        cfg = _validate(
            {
                "datasource_type": "bigquery",
                "project_id": "p",
                "credentials_json_ref": "env://GCP",
            }
        )
        assert isinstance(cfg, BigQueryConnectionConfig)

    def test_routes_agent_internal(self) -> None:
        cfg = _validate(
            {
                "datasource_type": "agent_internal",
                "schema_name": "agent_abc",
            }
        )
        assert isinstance(cfg, AgentInternalConnectionConfig)

    def test_unknown_discriminator_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _validate(
                {
                    "datasource_type": "duckdb",  # not in the union
                    "host": "h",
                    "database": "d",
                    "username": "u",
                    "password_ref": "env://PW",
                }
            )


def _validate(raw: dict[str, object]) -> ConnectionConfig:
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
        cfg = DatasourceConfig.model_validate(
            {
                "name": "central-reporting",
                "schemas": ["reporting_prod"],
                "access_mode": "read",
                "connection_config": {
                    "datasource_type": "redshift",
                    "host": "h",
                    "database": "d",
                    "username": "u",
                    "password_ref": "env://OTS_REDSHIFT_PASSWORD",
                },
            }
        )
        assert cfg.name == "central-reporting"
        assert cfg.access_mode == "read"
        assert isinstance(cfg.connection_config, RedshiftConnectionConfig)
        # convenience property exposes the nested datasource_type
        assert cfg.datasource_type == DataSourceType.REDSHIFT

    def test_no_flat_fields_accepted(self) -> None:
        # pre-shard-08 flat shape MUST fail (no aliasing)
        with pytest.raises(ValidationError):
            DatasourceConfig.model_validate(
                {
                    "name": "x",
                    "type": "redshift",
                    "host": "h",
                    "database": "d",
                    "username": "u",
                    "password_ref": "env://PW",
                }
            )

    def test_access_mode_validator(self) -> None:
        with pytest.raises(ValidationError):
            DatasourceConfig.model_validate(
                {
                    "name": "x",
                    "access_mode": "admin",  # invalid
                    "connection_config": {
                        "datasource_type": "redshift",
                        "host": "h",
                        "database": "d",
                        "username": "u",
                        "password_ref": "env://PW",
                    },
                }
            )


# ---------------------------------------------------------------------------
# Secret-redaction sanity
# ---------------------------------------------------------------------------


class TestSecretRedaction:
    """repr() / str() over a populated config don't expose secret values."""

    def test_resolved_secret_str_redacted(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MY_PW_ENV", "horse-battery-staple")
        cfg = PostgresConnectionConfig(
            datasource_type=DataSourceType.POSTGRES,
            host="h",
            database="d",
            username="u",
            password_ref="env://MY_PW_ENV",
        )
        secret = cfg.resolve_password()
        # SecretStr's repr / str redacts to '**********'
        assert "horse-battery-staple" not in repr(secret)
        assert "horse-battery-staple" not in str(secret)

    def test_password_ref_appears_but_not_resolved_value(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # the reference string (``env://MY_PW_ENV``) is safe to log; the
        # resolved VALUE is not. repr(config) shows the reference;
        # resolve_password() returns the SecretStr that redacts.
        monkeypatch.setenv("MY_PW_ENV", "horse-battery-staple")
        cfg = PostgresConnectionConfig(
            datasource_type=DataSourceType.POSTGRES,
            host="h",
            database="d",
            username="u",
            password_ref="env://MY_PW_ENV",
        )
        rendered = repr(cfg)
        # the reference string is fine to surface
        assert "env://MY_PW_ENV" in rendered
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
            {"datasource_type": "postgres", "host": "h", "database": "d", "username": "u", "password_ref": "env://PW"},
            {"datasource_type": "yugabyte", "host": "h", "database": "d", "username": "u", "password_ref": "env://PW"},
            {"datasource_type": "redshift", "host": "h", "database": "d", "username": "u", "password_ref": "env://PW"},
            {"datasource_type": "snowflake", "account": "a", "warehouse": "w", "user": "u", "password_ref": "env://PW"},
            {"datasource_type": "bigquery", "project_id": "p", "credentials_json_ref": "env://GCP"},
            {"datasource_type": "agent_internal", "schema_name": "agent_abc"},
        ],
    )
    def test_each_member_roundtrips(self, raw: dict[str, object]) -> None:
        original = _validate(raw)
        dumped = original.model_dump(mode="json")
        restored = _validate(dumped)
        assert restored == original

    def test_datasource_config_roundtrips(self) -> None:
        original = DatasourceConfig.model_validate(
            {
                "name": "central-reporting",
                "schemas": ["reporting_prod"],
                "access_mode": "read",
                "connection_config": {
                    "datasource_type": "redshift",
                    "host": "h.example.com",
                    "port": 5439,
                    "database": "analytics",
                    "username": "ots_user",
                    "password_ref": "env://OTS_REDSHIFT_PASSWORD",
                },
            }
        )
        dumped = original.model_dump(mode="json")
        restored = DatasourceConfig.model_validate(dumped)
        assert restored == original


# ---------------------------------------------------------------------------
# Stored form
# ---------------------------------------------------------------------------

_SENT_REDSHIFT: dict[str, object] = {
    "datasource_type": "redshift",
    "host": "h",
    "database": "d",
    "username": "u",
    "password_ref": "env://PW",
    "executor_max_workers": 3,
}


def _stored(config: ConnectionConfig) -> dict[str, object]:
    """the form a consumer persists: only the fields that were set.

    :param config: the validated config
    :ptype config: ConnectionConfig
    :return: the stored JSON, parsed
    :rtype: dict[str, object]
    """
    stored: dict[str, object] = json.loads(config.model_dump_json(exclude_unset=True))
    return stored


def _redshift_reader_without(field: str) -> type[BaseModel]:
    """a Redshift config as an earlier release declared it: without ``field``, unknown keys refused.

    :param field: the field the earlier release does not declare
    :ptype field: str
    :return: the earlier release's model
    :rtype: type[BaseModel]
    """
    declared: dict[str, Any] = {
        name: (info.annotation, info) for name, info in RedshiftConnectionConfig.model_fields.items() if name != field
    }
    return create_model("EarlierRedshiftConnectionConfig", __config__=ConfigDict(extra="forbid"), **declared)


class TestStoredForm:
    """a stored config reads back unchanged, and an earlier release can still read it.

    a consumer that persists a config and is rolled back reads, with the
    earlier release, what the newer one wrote. the full dump writes every
    default -- including a field the earlier release does not declare, which
    its ``extra="forbid"`` refuses -- so a rollback strands every row the
    newer release touched. the stored form writes only the fields that were
    set, so the earlier release refuses only a config that deliberately uses
    a setting it cannot honour.
    """

    @pytest.mark.parametrize(
        "raw",
        [
            {"datasource_type": "postgres", "host": "h", "database": "d", "username": "u", "password_ref": "env://PW"},
            {"datasource_type": "yugabyte", "host": "h", "database": "d", "username": "u", "password_ref": "env://PW"},
            _SENT_REDSHIFT,
            # an explicit cache equal to its default but not to the worker count:
            # a form that dropped it would re-derive 3 on read.
            pytest.param(dict(_SENT_REDSHIFT, connection_cache_size=5), id="redshift-explicit-cache-at-default"),
            {"datasource_type": "snowflake", "account": "a", "warehouse": "w", "user": "u", "password_ref": "env://PW"},
            {"datasource_type": "bigquery", "project_id": "p", "credentials_json_ref": "env://GCP"},
            {"datasource_type": "agent_internal", "schema_name": "agent_abc"},
        ],
    )
    def test_each_member_reads_back_unchanged(self, raw: dict[str, object]) -> None:
        original = _validate(raw)

        assert _validate(_stored(original)) == original

    def test_an_earlier_release_reads_a_config_that_left_a_new_field_at_its_default(self) -> None:
        config = _validate(_SENT_REDSHIFT)
        earlier = _redshift_reader_without("connect_timeout_seconds")

        earlier.model_validate(_stored(config))
        with pytest.raises(ValidationError, match="connect_timeout_seconds"):
            earlier.model_validate(json.loads(config.model_dump_json()))

    def test_an_earlier_release_refuses_a_config_that_set_a_field_it_cannot_honour(self) -> None:
        config = _validate(dict(_SENT_REDSHIFT, connect_timeout_seconds=10))

        with pytest.raises(ValidationError, match="connect_timeout_seconds"):
            _redshift_reader_without("connect_timeout_seconds").model_validate(_stored(config))

    def test_an_explicit_choice_equal_to_the_default_is_stored(self) -> None:
        default = RedshiftConnectionConfig.model_fields["query_timeout_seconds"].default
        config = _validate(dict(_SENT_REDSHIFT, query_timeout_seconds=default))

        assert _stored(config)["query_timeout_seconds"] == default

    def test_a_derived_cache_size_is_stored(self) -> None:
        stored = _stored(_validate(_SENT_REDSHIFT))

        assert stored["connection_cache_size"] == 3
