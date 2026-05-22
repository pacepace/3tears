"""tests for threetears.datasources.config.DatasourceConfig.

covers:

- YAML alias acceptance (``type:`` -> ``datasource_type``)
- access-mode + password-env validators
- round-trip via ``model_dump`` / ``model_validate``

shard-08 will introduce the discriminated-union ``ConnectionConfig``;
THESE tests pin the flat shape that shard-07 must preserve byte-for-byte.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from threetears.datasources.config import DatasourceConfig


class TestDatasourceConfigYamlShape:
    """the flat agent.yaml shape works with the canonical field names."""

    def test_minimal_valid(self) -> None:
        cfg = DatasourceConfig(
            name="central-reporting",
            type="redshift",
            host="h",
            database="d",
        )
        assert cfg.name == "central-reporting"
        assert cfg.datasource_type == "redshift"
        assert cfg.access_mode == "readwrite"

    def test_accepts_type_alias(self) -> None:
        # YAML files use ``type:`` -- pydantic exposes the field as
        # ``datasource_type`` but the alias must keep working
        cfg = DatasourceConfig.model_validate({
            "name": "x",
            "type": "postgres",
            "host": "h",
            "database": "d",
        })
        assert cfg.datasource_type == "postgres"

    def test_accepts_field_name_too(self) -> None:
        # populate_by_name=True allows both the alias and the canonical name
        cfg = DatasourceConfig.model_validate({
            "name": "x",
            "datasource_type": "postgres",
            "host": "h",
            "database": "d",
        })
        assert cfg.datasource_type == "postgres"


class TestPasswordEnvValidator:
    """``password_env`` carries an env-var name; validator rejects junk."""

    def test_accepts_valid_name(self) -> None:
        cfg = DatasourceConfig(
            name="x", type="redshift", host="h", database="d",
            password_env="OTS_REDSHIFT_PASSWORD",
        )
        assert cfg.password_env == "OTS_REDSHIFT_PASSWORD"

    def test_accepts_none(self) -> None:
        cfg = DatasourceConfig(name="x", type="redshift", host="h", database="d")
        assert cfg.password_env is None

    @pytest.mark.parametrize("bad", ["1starts-with-digit", "has space", "has-dash", "has.dot"])
    def test_rejects_invalid(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            DatasourceConfig(
                name="x", type="redshift", host="h", database="d",
                password_env=bad,
            )


class TestAccessModeValidator:
    """access_mode must be one of the three documented values."""

    @pytest.mark.parametrize("mode", ["read", "write", "readwrite"])
    def test_accepts_valid(self, mode: str) -> None:
        cfg = DatasourceConfig(
            name="x", type="redshift", host="h", database="d", access_mode=mode,
        )
        assert cfg.access_mode == mode

    def test_rejects_invalid(self) -> None:
        with pytest.raises(ValidationError):
            DatasourceConfig(
                name="x", type="redshift", host="h", database="d", access_mode="admin",
            )


class TestRoundTrip:
    """model_dump -> model_validate must be lossless."""

    def test_roundtrip_full(self) -> None:
        original = DatasourceConfig(
            name="central-reporting",
            type="redshift",
            host="h.example.com",
            port=5439,
            database="analytics",
            username="ots_user",
            password_env="OTS_REDSHIFT_PASSWORD",
            schemas=["reporting_prod"],
            access_mode="read",
        )
        dumped = original.model_dump(by_alias=True)
        restored = DatasourceConfig.model_validate(dumped)
        assert restored == original
