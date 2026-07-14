"""Unit tests for :class:`BackupConfig` — injected value object + env factory."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from threetears.backup.config import BackupConfig


def test_defaults_are_the_aibots_gfs_shape() -> None:
    config = BackupConfig(passphrase=SecretStr("pw"))
    assert (config.retention_daily, config.retention_weekly, config.retention_monthly) == (7, 4, 3)
    assert config.prefix == "backups"
    assert config.allow_delete is False


def test_is_frozen() -> None:
    config = BackupConfig(passphrase=SecretStr("pw"))
    with pytest.raises((AttributeError, TypeError)):
        config.prefix = "elsewhere"  # type: ignore[misc]


@pytest.mark.parametrize("field", ["retention_daily", "retention_weekly", "retention_monthly"])
def test_retention_must_be_at_least_one(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        BackupConfig(passphrase=SecretStr("pw"), **{field: 0})


def test_empty_passphrase_rejected() -> None:
    with pytest.raises(ValueError, match="passphrase"):
        BackupConfig(passphrase=SecretStr(""))


def test_nonpositive_timeout_rejected() -> None:
    with pytest.raises(ValueError, match="dump_timeout"):
        BackupConfig(passphrase=SecretStr("pw"), dump_timeout_seconds=0)


@pytest.mark.parametrize("bad", [0, 1, 3, 1000])
def test_work_factor_must_be_power_of_two_gt_one(bad: int) -> None:
    with pytest.raises(ValueError, match="work_factor"):
        BackupConfig(passphrase=SecretStr("pw"), encryption_work_factor=bad)


def test_work_factor_default_is_deployment_grade() -> None:
    assert BackupConfig(passphrase=SecretStr("pw")).encryption_work_factor == 2**18


def test_from_env_reads_prefixed_vars() -> None:
    env = {
        "THREETEARS_BACKUP_PASSPHRASE": "s3cret",
        "THREETEARS_BACKUP_PREFIX": "ranch-backups",
        "THREETEARS_BACKUP_RETENTION_DAILY": "14",
        "THREETEARS_BACKUP_ALLOW_DELETE": "true",
        "THREETEARS_BACKUP_DUMP_TIMEOUT_SECONDS": "900",
    }
    config = BackupConfig.from_env(env)

    assert config.passphrase.get_secret_value() == "s3cret"
    assert config.prefix == "ranch-backups"
    assert config.retention_daily == 14
    assert config.retention_weekly == 4  # untouched default
    assert config.allow_delete is True
    assert config.dump_timeout_seconds == 900


def test_from_env_requires_passphrase() -> None:
    with pytest.raises(ValueError, match="PASSPHRASE is required"):
        BackupConfig.from_env({})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1", True), ("true", True), ("YES", True), ("on", True), ("0", False), ("false", False), ("", False)],
)
def test_from_env_bool_parsing(raw: str, expected: bool) -> None:
    config = BackupConfig.from_env({"THREETEARS_BACKUP_PASSPHRASE": "x", "THREETEARS_BACKUP_ALLOW_DELETE": raw})
    assert config.allow_delete is expected
