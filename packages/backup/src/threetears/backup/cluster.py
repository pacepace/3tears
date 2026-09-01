"""Cluster backup sets — every database, the globals, and a manifest that proves coverage.

``BackupEngine`` backs up ONE database per call, which leaves two holes at cluster grain: databases
nobody remembered to list (an agent or tool that created its own is invisible to a hand-maintained
list), and cluster globals (roles, grants) that live outside every database and so outside every
per-database dump. :class:`ClusterBackup` closes both: it ENUMERATES the databases from the cluster
itself at backup time — coverage by construction, never by list — dumps each one plus the globals,
and writes a :class:`~threetears.backup.manifest.BackupManifest` recording the set's stable uuid7
id, its driver, and a per-table row inventory taken at dump time.

The manifest is written last, so its existence asserts a complete set. Restores resolve the driver
FROM the manifest (:func:`~threetears.backup.drivers.driver_by_name`) — the format that wrote a
dump is the only format that can read it back, and trusting the restoring process's own driver is
how a gzipped plain-SQL dump ends up fed to ``pg_restore``.

Like the verifier, the database connection is injected (an ``asyncpg.connect``-shaped callable), so
the orchestration stays unit-testable with fakes and the package keeps asyncpg out of its hard
dependencies.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Any, Protocol, cast, runtime_checkable
from urllib.parse import urlparse, urlunparse
from uuid import UUID, uuid7

from threetears.media.contracts import ObjectStore
from threetears.object_store import EncryptedObjectStore
from threetears.observe import get_logger

from threetears.backup.config import BackupConfig
from threetears.backup.drivers import DbDumpDriver, driver_by_name, driver_for_version
from threetears.backup.gzip import gunzip_stream, gzip_stream
from threetears.backup.manifest import BackupManifest, DatabaseDump, TableCount, manifest_key
from threetears.backup.process import feed_stdin, stream_stdout

__all__ = ["ClusterBackup", "ManifestNotFoundError", "replace_database"]

log = get_logger(__name__)

_ENCRYPTED_CONTENT_TYPE = "application/octet-stream"

#: databases every cluster carries that a backup set must not: templates are scaffolding, and
#: dumping ``template0`` outright fails (it refuses connections by design).
_EXCLUDED_DATABASES = frozenset({"template0", "template1"})

_DATABASES_SQL = "SELECT datname FROM pg_database WHERE NOT datistemplate ORDER BY datname"
_TABLES_SQL = """
    SELECT table_schema, table_name
      FROM information_schema.tables
     WHERE table_type = 'BASE TABLE'
       AND table_schema NOT IN ('pg_catalog', 'information_schema')
     ORDER BY table_schema, table_name
"""


class ManifestNotFoundError(LookupError):
    """No stored manifest carries the requested backup id."""


@runtime_checkable
class _Connection(Protocol):
    async def fetch(self, query: str) -> list[Any]: ...
    async def fetchval(self, query: str) -> object: ...
    async def close(self) -> None: ...


#: an ``asyncpg.connect``-shaped callable; injected for testability, exactly as the verifier's is.
Connect = Callable[[str], Awaitable[_Connection]]


def replace_database(dsn: str, database: str) -> str:
    """Return ``dsn`` re-pointed at ``database`` on the same host/credentials."""
    parsed = urlparse(dsn)
    return urlunparse(parsed._replace(path=f"/{database}"))


class _HashingStream:
    """Wrap a byte stream, forwarding chunks while accumulating a SHA-256 of the plaintext."""

    def __init__(self, source: AsyncIterator[bytes]) -> None:
        self._source = source
        self._digest = hashlib.sha256()

    def __aiter__(self) -> _HashingStream:
        return self

    async def __anext__(self) -> bytes:
        chunk = await self._source.__anext__()
        self._digest.update(chunk)
        return chunk

    @property
    def hexdigest(self) -> str:
        return self._digest.hexdigest()


class ClusterBackup:
    """Create, list, and restore whole-cluster backup sets.

    :param config: the injected :class:`BackupConfig` (passphrase, prefix, timeouts).
    :param store: the backend :class:`ObjectStore`; wrapped in encryption here, exactly as
        :class:`~threetears.backup.engine.BackupEngine` does — nothing this class writes can be
        plaintext.
    :param connect: an ``asyncpg.connect``-shaped callable used for enumeration and inventory.
    :param env: environment for the dump/restore subprocesses (e.g. ``PGPASSWORD``).
    """

    def __init__(
        self,
        config: BackupConfig,
        store: ObjectStore,
        connect: Connect,
        *,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._config = config
        self._connect = connect
        self._env = env
        self._store: ObjectStore = EncryptedObjectStore(
            store, config.passphrase, scrypt_n=config.encryption_work_factor
        )

    # ------------------------------------------------------------------ create

    async def create_backup(self, admin_dsn: str, *, when: datetime | None = None) -> BackupManifest:
        """Dump every database in the cluster plus the globals, and write the set's manifest.

        :param admin_dsn: a dsn with rights to enumerate databases and dump each one; also the
            dsn the globals dump runs against.
        :param when: backup timestamp (defaults to now, UTC) — also the storage partition.
        :return: the written manifest, which is the set's durable identity.
        """
        moment = when or datetime.now(UTC)
        backup_id = uuid7()
        driver = await self._detect(admin_dsn)
        databases = await self._list_databases(admin_dsn)
        set_root = f"{self._config.prefix}/{moment:%Y/%m/%d}/{moment:%Y%m%dT%H%M%SZ}-{backup_id.hex[:12]}"

        globals_key = f"{set_root}/globals.sql.gz.enc"
        await self._store.put(
            globals_key,
            gzip_stream(
                stream_stdout(
                    driver.dump_globals_argv(admin_dsn),
                    env=self._env,
                    timeout=self._config.dump_timeout_seconds,
                )
            ),
            content_type=_ENCRYPTED_CONTENT_TYPE,
        )

        dumps: list[DatabaseDump] = []
        for database in databases:
            db_dsn = replace_database(admin_dsn, database)
            tables = await self._inventory(db_dsn)
            suffix = "dump" if driver.compressed else "dump.gz"
            key = f"{set_root}/{database}.{driver.name}.{suffix}.enc"
            raw = driver.dump(db_dsn, env=self._env, timeout=self._config.dump_timeout_seconds)
            hashing = _HashingStream(raw)
            stream: AsyncIterator[bytes] = hashing if driver.compressed else gzip_stream(hashing)
            await self._store.put(key, stream, content_type=_ENCRYPTED_CONTENT_TYPE)
            size = await self._size_of(key)
            dumps.append(
                DatabaseDump(
                    database=database,
                    key=key,
                    size_bytes=size,
                    sha256=hashing.hexdigest,
                    tables=tables,
                )
            )
            log.info(
                "cluster backup: database dumped",
                extra={"extra_data": {"database": database, "key": key, "tables": len(tables)}},
            )

        manifest = BackupManifest(
            backup_id=backup_id,
            created_at=moment,
            driver=driver.name,
            databases=tuple(dumps),
            globals_key=globals_key,
        )
        await self._store.put(
            manifest_key(self._config.prefix, backup_id),
            _one_chunk(manifest.to_json()),
            content_type="application/json",
        )
        log.info(
            "cluster backup complete",
            extra={
                "extra_data": {
                    "backup_id": str(backup_id),
                    "databases": len(dumps),
                    "tables": manifest.table_total,
                }
            },
        )
        return manifest

    # ------------------------------------------------------------------ list / get

    async def list_manifests(self) -> list[BackupManifest]:
        """Every stored manifest, newest first — the durable listing UIs should render."""
        prefix = f"{self._config.prefix}/manifests/"
        manifests: list[BackupManifest] = []
        async for entry in self._store.list_entries(prefix):
            manifests.append(await self._read_manifest(entry.key))
        manifests.sort(key=lambda m: m.created_at, reverse=True)
        return manifests

    async def get_manifest(self, backup_id: UUID) -> BackupManifest:
        """The manifest for one backup set.

        :raises ManifestNotFoundError: when no stored manifest carries ``backup_id``.
        """
        key = manifest_key(self._config.prefix, backup_id)
        try:
            return await self._read_manifest(key)
        except Exception as exc:
            raise ManifestNotFoundError(f"no manifest for backup {backup_id}") from exc

    # ------------------------------------------------------------------ restore

    async def restore_database(self, manifest: BackupManifest, database: str, target_dsn: str) -> None:
        """Restore ONE database's dump from a set into ``target_dsn``.

        The target must be a fresh, empty database (the verifier's temp db, a scratch db for a
        selective restore) — the drivers' restore argv assumes it. The driver comes from the
        manifest, never from this process's configuration.

        :raises LookupError: when the manifest holds no dump for ``database``.
        """
        dump = next((d for d in manifest.databases if d.database == database), None)
        if dump is None:
            raise LookupError(f"backup {manifest.backup_id} holds no database named {database!r}")
        driver = driver_by_name(manifest.driver)
        stream = self._store.open_read(dump.key)
        if not driver.compressed:
            stream = gunzip_stream(stream)
        await driver.restore(target_dsn, stream, env=self._env, timeout=self._config.dump_timeout_seconds)
        log.info(
            "cluster restore: database restored",
            extra={"extra_data": {"backup_id": str(manifest.backup_id), "database": database}},
        )

    async def restore_globals(self, manifest: BackupManifest, admin_dsn: str) -> None:
        """Replay the set's globals dump (roles, grants) against ``admin_dsn``.

        :raises LookupError: when the set carries no globals dump.
        """
        if manifest.globals_key is None:
            raise LookupError(f"backup {manifest.backup_id} carries no globals dump")
        driver = driver_by_name(manifest.driver)
        stream = gunzip_stream(self._store.open_read(manifest.globals_key))
        await feed_stdin(
            driver.restore_sql_argv(admin_dsn), stream, env=self._env, timeout=self._config.dump_timeout_seconds
        )

    # ------------------------------------------------------------------ internals

    async def _detect(self, dsn: str) -> DbDumpDriver:
        conn = await self._connect(dsn)
        try:
            version = await conn.fetchval("SELECT version()")
        finally:
            await conn.close()
        return driver_for_version(str(version))

    async def _list_databases(self, admin_dsn: str) -> list[str]:
        conn = await self._connect(admin_dsn)
        try:
            rows = await conn.fetch(_DATABASES_SQL)
        finally:
            await conn.close()
        return [row["datname"] for row in rows if row["datname"] not in _EXCLUDED_DATABASES]

    async def _inventory(self, db_dsn: str) -> tuple[TableCount, ...]:
        """Exact per-table row counts at dump time — estimates would poison later verification."""
        conn = await self._connect(db_dsn)
        try:
            tables = await conn.fetch(_TABLES_SQL)
            counts: list[TableCount] = []
            for row in tables:
                schema, table = row["table_schema"], row["table_name"]
                count = await conn.fetchval(f'SELECT count(*) FROM "{schema}"."{table}"')
                counts.append(TableCount(schema=schema, table=table, row_count=int(cast(int, count))))
        finally:
            await conn.close()
        return tuple(counts)

    async def _read_manifest(self, key: str) -> BackupManifest:
        chunks = [chunk async for chunk in self._store.open_read(key)]
        return BackupManifest.from_json(b"".join(chunks))

    async def _size_of(self, key: str) -> int:
        size = 0
        async for entry in self._store.list_entries(key):
            if entry.key == key:
                size = entry.size_bytes
                break
        return size


async def _one_chunk(data: bytes) -> AsyncIterator[bytes]:
    yield data
