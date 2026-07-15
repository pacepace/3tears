"""Injected configuration for the backup engine.

:class:`BackupConfig` is a frozen value object you *pass in* -- the engine never reaches for the
environment itself. Most apps build it from control-plane settings; :meth:`BackupConfig.from_env`
is a convenience that reads ``THREETEARS_BACKUP_*`` with sensible defaults for the simple case.

It is deliberately storage-agnostic: there is no bucket here. The backend is an injected
``ObjectStore`` (which already knows where it writes), so the same config drives an S3 backup or a
filesystem one. What lives here is the encryption passphrase, the key prefix, the GFS retention
counts, and the delete safety switch.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import SecretStr

__all__ = ["BackupConfig"]

_ENV_PREFIX = "THREETEARS_BACKUP_"


@dataclass(frozen=True, slots=True)
class BackupConfig:
    """Backup engine configuration (injected; never self-loaded).

    :param passphrase: AES-256-GCM encryption passphrase (per-object scrypt-derived key).
    :param prefix: object-key prefix under which backups are written/listed.
    :param retention_daily: number of daily backups to keep (>= 1).
    :param retention_weekly: number of weekly backups to keep (>= 1).
    :param retention_monthly: number of monthly backups to keep (>= 1).
    :param allow_delete: master switch for destructive operations (delete / retention prune).
    :param dump_timeout_seconds: wall-clock ceiling for a dump/restore subprocess (> 0).
    :param encryption_work_factor: scrypt cost N for the per-object key (power of two > 1); the
        default is deployment-grade, lower it only to trade brute-force resistance for speed.
    """

    passphrase: SecretStr
    prefix: str = "backups"
    retention_daily: int = 7
    retention_weekly: int = 4
    retention_monthly: int = 3
    allow_delete: bool = False
    dump_timeout_seconds: int = 3600
    encryption_work_factor: int = 2**18

    def __post_init__(self) -> None:
        if not self.prefix or self.prefix != self.prefix.strip("/"):
            raise ValueError("prefix must be non-empty with no leading/trailing '/'")
        for name in ("retention_daily", "retention_weekly", "retention_monthly"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.dump_timeout_seconds <= 0:
            raise ValueError("dump_timeout_seconds must be > 0")
        if self.encryption_work_factor <= 1 or (self.encryption_work_factor & (self.encryption_work_factor - 1)) != 0:
            raise ValueError("encryption_work_factor must be a power of two greater than 1")
        if not self.passphrase.get_secret_value():
            raise ValueError("passphrase must not be empty")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> BackupConfig:
        """Build a config from ``THREETEARS_BACKUP_*`` variables (defaults fill the rest).

        :param env: environment mapping to read (defaults to ``os.environ``).
        :raises ValueError: when ``THREETEARS_BACKUP_PASSPHRASE`` is unset, or a value is invalid.
        """
        source = os.environ if env is None else env
        passphrase = source.get(f"{_ENV_PREFIX}PASSPHRASE")
        if not passphrase:
            raise ValueError(f"{_ENV_PREFIX}PASSPHRASE is required")
        return cls(
            passphrase=SecretStr(passphrase),
            prefix=source.get(f"{_ENV_PREFIX}PREFIX", "backups"),
            retention_daily=_int(source, "RETENTION_DAILY", 7),
            retention_weekly=_int(source, "RETENTION_WEEKLY", 4),
            retention_monthly=_int(source, "RETENTION_MONTHLY", 3),
            allow_delete=_bool(source, "ALLOW_DELETE", default=False),
            dump_timeout_seconds=_int(source, "DUMP_TIMEOUT_SECONDS", 3600),
            encryption_work_factor=_int(source, "ENCRYPTION_WORK_FACTOR", 2**18),
        )


def _int(source: Mapping[str, str], suffix: str, default: int) -> int:
    raw = source.get(f"{_ENV_PREFIX}{suffix}")
    return default if raw is None else int(raw)


def _bool(source: Mapping[str, str], suffix: str, *, default: bool) -> bool:
    raw = source.get(f"{_ENV_PREFIX}{suffix}")
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}
