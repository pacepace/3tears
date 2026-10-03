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
    :param restore_copy_rows_per_transaction: how many rows one bulk COPY may commit at a time
        during a restore. Configurable because the safe value depends on how big the rows are,
        which is a property of the data rather than of this package.
    :param transient_database_prefixes: name prefixes for databases that are THROWAWAY restore
        targets rather than data. A backup that dumps one is backing up a copy of another
        database it already dumped, and paying for it twice; the defaults are the prefixes this
        package and its callers generate.
    :param dump_concurrent_ddl_retries: how many more times one database's dump is taken when the
        dump tool failed because DDL changed the catalog under it (>= 0; 0 never retries).
    :param dump_concurrent_ddl_retry_delay_seconds: pause before the first such retry, doubled
        before each later one (> 0).
    """

    passphrase: SecretStr
    prefix: str = "backups"
    retention_daily: int = 7
    retention_weekly: int = 4
    retention_monthly: int = 3
    allow_delete: bool = False
    dump_timeout_seconds: int = 3600
    encryption_work_factor: int = 2**18
    #: Yugabyte batches COPY by ROW COUNT (`yb_default_copy_from_rows_per_transaction`,
    #: default 20000) with no regard for row size, and a restore of blob-heavy rows then asks
    #: the server to commit a transaction far larger than its inbound RPC buffer, which it
    #: refuses. Measured on a live 3 GB set whose `checkpoints` rows averaged 115 KB and peaked
    #: near 196 KB: 20000 and 1000 both failed under memory pressure, 100 restored cleanly.
    #:
    #: 100 rows is roughly 20 MB even at that worst case, against a tserver read buffer of
    #: about 365 MB shared with every other caller. Raise it for narrow rows if a restore is
    #: too slow; the failure mode of raising it too far is a refused write, not corruption.
    #:
    #: Ignored by drivers whose server has no such knob -- vanilla Postgres among them.
    restore_copy_rows_per_transaction: int = 100
    #: Matched case-sensitively against the start of the database name. The
    #: defaults name the prefixes THIS package (`verify_restore_`) and its
    #: callers (`scratch_restore_`, and `scratch_` broadly) create when they need
    #: somewhere to replay a dump. Excluding them is not an optimisation: a
    #: throwaway restore target is, by construction, a partial copy of a database
    #: the set already contains, and dumping it back into the same set doubles
    #: the storage to preserve nothing.
    transient_database_prefixes: tuple[str, ...] = ("scratch_", "verify_restore_")
    #: A dump reads the catalog's list of objects, then asks the server about each one, and DDL
    #: running in the same database between the two -- an agent schema being created on a cold
    #: start, found live -- fails it with "schema with OID N does not exist". Only that database's
    #: dump is taken again, and only on that failure (``cluster.is_concurrent_ddl_failure``).
    #: Three retries over 5 + 10 + 20 seconds outlasts the DDL a cold start runs; a database still
    #: failing after them is recorded as failed, as every other dump failure is. Why the backup
    #: retries rather than taking the database's DDL lock: the ``threetears.backup.cluster``
    #: module docstring.
    dump_concurrent_ddl_retries: int = 3
    dump_concurrent_ddl_retry_delay_seconds: float = 5.0

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
        if self.dump_concurrent_ddl_retries < 0:
            raise ValueError("dump_concurrent_ddl_retries must be >= 0")
        if self.dump_concurrent_ddl_retry_delay_seconds <= 0:
            raise ValueError("dump_concurrent_ddl_retry_delay_seconds must be > 0")

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
            dump_concurrent_ddl_retries=_int(source, "DUMP_CONCURRENT_DDL_RETRIES", 3),
            dump_concurrent_ddl_retry_delay_seconds=_float(source, "DUMP_CONCURRENT_DDL_RETRY_DELAY_SECONDS", 5.0),
        )


def _int(source: Mapping[str, str], suffix: str, default: int) -> int:
    raw = source.get(f"{_ENV_PREFIX}{suffix}")
    return default if raw is None else int(raw)


def _float(source: Mapping[str, str], suffix: str, default: float) -> float:
    raw = source.get(f"{_ENV_PREFIX}{suffix}")
    return default if raw is None else float(raw)


def _bool(source: Mapping[str, str], suffix: str, *, default: bool) -> bool:
    raw = source.get(f"{_ENV_PREFIX}{suffix}")
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}
