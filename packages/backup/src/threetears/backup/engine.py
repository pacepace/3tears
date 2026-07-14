"""The backup engine — encrypted, compressed, GFS-rotated DB backups over any ObjectStore.

:class:`BackupEngine` composes the pieces: it wraps the injected backend store in
:class:`EncryptedObjectStore` (so backups are encrypted *by construction* — a caller can't
accidentally write plaintext), streams the driver's dump through gzip when the format isn't already
compressed, and writes it under a date-partitioned key. Listing, retention pruning, and restore all
funnel back through the same encrypted store. Destructive operations (delete, retention prune) are
gated on ``config.allow_delete``.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from uuid import uuid7

from threetears.media.contracts import ObjectStore
from threetears.object_store import EncryptedObjectStore
from threetears.observe import get_logger

from threetears.backup.config import BackupConfig
from threetears.backup.drivers import DbDumpDriver
from threetears.backup.gzip import gunzip_stream, gzip_stream
from threetears.backup.retention import BackupRecord, GfsRetention, RetentionDecision

__all__ = ["BackupEngine", "DeleteNotAllowedError"]

log = get_logger(__name__)

_ENCRYPTED_CONTENT_TYPE = "application/octet-stream"
_KEY_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"


def _created_at_from_key(key: str, *, fallback: datetime) -> datetime:
    """Parse the creation timestamp encoded in a backup key's filename.

    Keys are ``<prefix>/<Y>/<m>/<d>/<YYYYMMDDThhmmssZ>-<id>.<driver>.<suffix>.enc``; the stamp is
    the filename up to the first ``-``. A key we didn't write (no parseable stamp) falls back to the
    store's last-modified so a foreign object still sorts sanely.
    """
    stamp = key.rsplit("/", 1)[-1].split("-", 1)[0]
    try:
        parsed = datetime.strptime(stamp, _KEY_STAMP_FORMAT).replace(tzinfo=UTC)
    except ValueError:
        parsed = fallback
    return parsed


class DeleteNotAllowedError(RuntimeError):
    """A destructive operation was attempted while ``config.allow_delete`` is False."""


class BackupEngine:
    """Create / list / restore / prune encrypted database backups.

    :param config: the injected :class:`BackupConfig`.
    :param store: the backend :class:`ObjectStore` (S3, filesystem, …); wrapped in encryption here.
    :param driver: the :class:`DbDumpDriver` for the target database.
    :param env: environment for the dump/restore subprocess (e.g. ``PGPASSWORD``).
    """

    def __init__(
        self,
        config: BackupConfig,
        store: ObjectStore,
        driver: DbDumpDriver,
        *,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._config = config
        self._driver = driver
        self._env = env
        # encrypt by construction: everything written/read goes through the AEAD wrapper.
        self._store: ObjectStore = EncryptedObjectStore(
            store, config.passphrase, scrypt_n=config.encryption_work_factor
        )

    def _key_for(self, when: datetime) -> str:
        stamp = when.strftime("%Y%m%dT%H%M%SZ")
        suffix = "dump" if self._driver.compressed else "dump.gz"
        return f"{self._config.prefix}/{when:%Y/%m/%d}/{stamp}-{uuid7().hex[:8]}.{self._driver.name}.{suffix}.enc"

    async def create_backup(self, source_dsn: str, *, when: datetime | None = None) -> BackupRecord:
        """Dump ``source_dsn``, compress+encrypt it, and store it.

        :param source_dsn: connection string of the database to back up.
        :param when: backup timestamp (defaults to now, UTC) — also the storage partition.
        :return: the record for the written backup.
        """
        moment = when or datetime.now(UTC)
        key = self._key_for(moment)
        stream = self._driver.dump(source_dsn, env=self._env)
        if not self._driver.compressed:
            stream = gzip_stream(stream)
        await self._store.put(key, stream, content_type=_ENCRYPTED_CONTENT_TYPE)
        size = await self._size_of(key)
        log.info("backup created", extra={"extra_data": {"key": key, "size_bytes": size, "driver": self._driver.name}})
        return BackupRecord(key=key, created_at=moment, size_bytes=size)

    async def restore_into(self, target_dsn: str, key: str) -> None:
        """Restore the backup at ``key`` into ``target_dsn`` (decrypt → gunzip → driver restore)."""
        stream = self._store.open_read(key)
        if not self._driver.compressed:
            stream = gunzip_stream(stream)
        await self._driver.restore(target_dsn, stream, env=self._env)
        log.info("backup restored", extra={"extra_data": {"key": key, "driver": self._driver.name}})

    async def list_backups(self) -> list[BackupRecord]:
        """List stored backups (newest first), timed by the timestamp encoded in each key."""
        records = [
            BackupRecord(
                key=entry.key,
                created_at=_created_at_from_key(entry.key, fallback=entry.last_modified),
                size_bytes=entry.size_bytes,
            )
            async for entry in self._store.list_entries(f"{self._config.prefix}/")
        ]
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records

    async def plan_retention(self) -> RetentionDecision:
        """Compute (without deleting) which backups GFS retention would keep vs prune."""
        records = await self.list_backups()
        return GfsRetention.from_config(self._config).select(records)

    async def apply_retention(self) -> RetentionDecision:
        """Prune backups outside the GFS policy. Requires ``allow_delete``.

        :raises DeleteNotAllowedError: when ``config.allow_delete`` is False.
        """
        if not self._config.allow_delete:
            raise DeleteNotAllowedError("retention prune requires config.allow_delete=True")
        decision = await self.plan_retention()
        if decision.delete:
            await self._store.delete_many([r.key for r in decision.delete])
        log.info(
            "retention pruned", extra={"extra_data": {"deleted": len(decision.delete), "kept": len(decision.keep)}}
        )
        return decision

    async def delete_backup(self, key: str) -> None:
        """Delete one backup. Requires ``allow_delete``.

        :raises DeleteNotAllowedError: when ``config.allow_delete`` is False.
        """
        if not self._config.allow_delete:
            raise DeleteNotAllowedError("delete requires config.allow_delete=True")
        await self._store.delete(key)

    async def _size_of(self, key: str) -> int:
        size = 0
        async for entry in self._store.list_entries(key):
            if entry.key == key:
                size = entry.size_bytes
                break
        return size
