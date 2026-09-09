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
from contextlib import asynccontextmanager
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
from threetears.backup.manifest import BackupManifest, DatabaseDump, DatabaseFailure, TableCount, manifest_key
from threetears.backup.process import feed_stdin, stream_stdout
from threetears.backup.retention import BackupRecord, GfsRetention, RetentionDecision

__all__ = ["ClusterBackup", "ManifestNotFoundError", "SetDeleteNotAllowedError", "replace_database"]

log = get_logger(__name__)

_ENCRYPTED_CONTENT_TYPE = "application/octet-stream"

#: databases every cluster carries that a backup set must not: templates are scaffolding, and
#: dumping ``template0`` outright fails (it refuses connections by design).
_EXCLUDED_DATABASES = frozenset({"template0", "template1"})

# Which databases are transient now lives on BackupConfig
# (`transient_database_prefixes`), so a deployment can widen it without a
# release. It was a module constant naming only the two prefixes THIS toolchain
# generates, which is why a hand-made `scratch_probe_hub2` sailed past it and
# broke every backup on a live cluster for weeks.

_DATABASES_SQL = "SELECT datname FROM pg_database WHERE NOT datistemplate ORDER BY datname"

#: The inventory and the dump must see ONE instant. This transaction holds it: repeatable read
#: fixes the session's view, `pg_export_snapshot` publishes it, and the dump tool joins it with
#: `--snapshot`. The transaction stays open for the whole dump because an exported snapshot dies
#: with the session that exported it.
_BEGIN_SNAPSHOT_SQL = "BEGIN ISOLATION LEVEL REPEATABLE READ"
_EXPORT_SNAPSHOT_SQL = "SELECT pg_export_snapshot()"
_ROLLBACK_SQL = "ROLLBACK"
_TABLES_SQL = """
    SELECT table_schema, table_name
      FROM information_schema.tables
     WHERE table_type = 'BASE TABLE'
       AND table_schema NOT IN ('pg_catalog', 'information_schema')
     ORDER BY table_schema, table_name
"""


class ManifestNotFoundError(LookupError):
    """No stored manifest carries the requested backup id."""


class ClusterBackupError(RuntimeError):
    """the set could not be taken at all.

    Distinct from an INCOMPLETE set, which is a real backup carrying a record of
    what it is missing. This is raised only when no database dumped, where a
    manifest would be an empty set wearing a backup's name -- and retention
    would go on to count it as one.
    """


class SetDeleteNotAllowedError(RuntimeError):
    """A destructive set operation was attempted while ``config.allow_delete`` is False."""


@runtime_checkable
class _Connection(Protocol):
    async def execute(self, query: str) -> object: ...
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

        log.info(
            "cluster backup: starting",
            extra={
                "extra_data": {
                    "backup_id": str(backup_id),
                    "driver": driver.name,
                    "databases": len(databases),
                }
            },
        )

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
        log.info(
            "cluster backup: globals dumped",
            extra={"extra_data": {"backup_id": str(backup_id), "key": globals_key}},
        )

        # ONE SICK DATABASE MUST NOT COST THE CLUSTER ITS BACKUP. This loop used
        # to have no handler at all, so a single database the dump tool could not
        # read aborted the whole set and left the cluster with NOTHING -- not
        # five dumps and a gap, nothing. Observed: one database whose
        # `pg_namespace` had lost its DocDB tablet took down every scheduled
        # backup, and the cluster went unbacked-up until somebody read the error.
        #
        # A failure is RECORDED, never skipped. A set that quietly omitted a
        # database would present as a complete backup of a cluster it does not
        # cover, and the first anyone would hear of it is a restore coming up
        # short.
        dumps: list[DatabaseDump] = []
        failures: list[DatabaseFailure] = []
        for position, database in enumerate(databases, start=1):
            # Logged per database, before the work rather than after it. A cluster backup is
            # minutes of silence otherwise, and an operator watching one has exactly one
            # decision to make -- keep waiting, or intervene -- which needs to know WHICH
            # database is slow and how many are left.
            log.info(
                "cluster backup: database starting",
                extra={
                    "extra_data": {
                        "backup_id": str(backup_id),
                        "database": database,
                        "position": position,
                        "of": len(databases),
                    }
                },
            )
            try:
                db_dsn = replace_database(admin_dsn, database)
                suffix = "dump" if driver.compressed else "dump.gz"
                key = f"{set_root}/{database}.{driver.name}.{suffix}.enc"
                async with self._inventory_snapshot(db_dsn) as (tables, snapshot):
                    consistent = snapshot is not None
                    log.info(
                        "cluster backup: inventory taken",
                        extra={
                            "extra_data": {
                                "backup_id": str(backup_id),
                                "database": database,
                                "tables": len(tables),
                                "snapshot_synchronized": consistent,
                            }
                        },
                    )
                    raw = driver.dump(
                        db_dsn,
                        env=self._env,
                        timeout=self._config.dump_timeout_seconds,
                        snapshot=snapshot,
                    )
                    hashing = _HashingStream(raw)
                    stream: AsyncIterator[bytes] = hashing if driver.compressed else gzip_stream(hashing)
                    await self._store.put(key, stream, content_type=_ENCRYPTED_CONTENT_TYPE)
                size = await self._size_of(key)
            except Exception as exc:  # prawduct:allow prawduct/broad-except -- see below
                # Deliberately broad, and deliberately NOT BaseException: the dump
                # tool fails in as many ways as the databases it reads (a catalog
                # inconsistency, a permission, a timeout, the store refusing a
                # put), and every one of them should cost that database and no
                # other. Cancellation is different -- a drained pod must abandon
                # the whole set, not record every remaining database as broken --
                # and CancelledError is a BaseException, so it passes through.
                log.exception(
                    "cluster backup: database FAILED to dump; the set will be incomplete",
                    extra={"extra_data": {"database": database}},
                )
                failures.append(DatabaseFailure(database=database, error=f"{type(exc).__name__}: {exc}"))
                continue
            dumps.append(
                DatabaseDump(
                    database=database,
                    key=key,
                    size_bytes=size,
                    sha256=hashing.hexdigest,
                    tables=tables,
                    inventory_snapshot_consistent=consistent,
                )
            )
            log.info(
                "cluster backup: database dumped",
                extra={
                    "extra_data": {
                        "backup_id": str(backup_id),
                        "database": database,
                        "key": key,
                        "tables": len(tables),
                        "size_bytes": size,
                    }
                },
            )

        if not dumps:
            # Nothing was backed up. A manifest here would be an empty set
            # wearing a backup's name, and retention would count it as one.
            raise ClusterBackupError(
                "cluster backup produced no database dumps; every database failed: "
                + "; ".join(f"{f.database} ({f.error})" for f in failures)
            )

        manifest = BackupManifest(
            backup_id=backup_id,
            created_at=moment,
            driver=driver.name,
            databases=tuple(dumps),
            globals_key=globals_key,
            failed_databases=tuple(failures),
        )
        await self._store.put(
            manifest_key(self._config.prefix, backup_id),
            _one_chunk(manifest.to_json()),
            content_type="application/json",
        )
        # INCOMPLETE is not "complete with a note". It is its own outcome, and it
        # is logged at a level somebody pages on, because a cluster whose backups
        # silently stopped covering a database is one restore away from finding
        # out the hard way.
        log_at = log.info if manifest.is_complete else log.error
        log_at(
            "cluster backup complete" if manifest.is_complete else "cluster backup INCOMPLETE: some databases failed",
            extra={
                "extra_data": {
                    "backup_id": str(backup_id),
                    "databases": len(dumps),
                    "failed_databases": [f.database for f in failures],
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
        # A restore is the longest silence in the system -- one measured run spent 13m34s
        # between claiming the operation and its first line of output, which is
        # indistinguishable from a wedge to whoever is watching. Say what is starting.
        log.info(
            "cluster restore: replaying dump",
            extra={
                "extra_data": {
                    "backup_id": str(manifest.backup_id),
                    "database": database,
                    "driver": driver.name,
                    "size_bytes": dump.size_bytes,
                }
            },
        )
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

    # ------------------------------------------------------------------ retention / delete

    async def plan_retention(self) -> RetentionDecision:
        """Compute (without deleting) which SETS the GFS policy would keep vs prune.

        The unit of retention is the whole set — a manifest and every dump it names live and
        die together, because a set missing one database is not a smaller backup, it is a
        broken one.
        """
        manifests = await self.list_manifests()
        records = [
            BackupRecord(key=str(m.backup_id), created_at=m.created_at, size_bytes=m.total_size_bytes)
            for m in manifests
        ]
        return GfsRetention.from_config(self._config).select(records)

    async def apply_retention(self) -> RetentionDecision:
        """Prune whole sets outside the GFS policy. Requires ``allow_delete``.

        :raises SetDeleteNotAllowedError: when ``config.allow_delete`` is False.
        """
        if not self._config.allow_delete:
            raise SetDeleteNotAllowedError("retention prune requires config.allow_delete=True")
        decision = await self.plan_retention()
        for record in decision.delete:
            await self.delete_set(UUID(record.key))
        log.info(
            "set retention pruned",
            extra={"extra_data": {"deleted": len(decision.delete), "kept": len(decision.keep)}},
        )
        return decision

    async def delete_set(self, backup_id: UUID) -> None:
        """Delete one whole set — every dump, the globals, then the manifest LAST.

        Order matters the same way it does on write: the manifest asserts a complete set, so
        it must be the last thing standing, never a survivor pointing at deleted dumps.

        :raises SetDeleteNotAllowedError: when ``config.allow_delete`` is False.
        :raises ManifestNotFoundError: when no manifest carries ``backup_id``.
        """
        if not self._config.allow_delete:
            raise SetDeleteNotAllowedError("delete requires config.allow_delete=True")
        manifest = await self.get_manifest(backup_id)
        for dump in manifest.databases:
            await self._store.delete(dump.key)
        if manifest.globals_key is not None:
            await self._store.delete(manifest.globals_key)
        await self._store.delete(manifest_key(self._config.prefix, backup_id))
        log.info("backup set deleted", extra={"extra_data": {"backup_id": str(backup_id)}})

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
        named = [row["datname"] for row in rows if row["datname"] not in _EXCLUDED_DATABASES]
        transient = [name for name in named if name.startswith(self._config.transient_database_prefixes)]
        if transient:
            # Logged rather than passed over in silence: a scratch that is still
            # here is debris from a drill or a restore that did not finish
            # tidying, and the only reason anyone ever found the one that had
            # wedged this cluster's backups was reading an error it caused.
            log.info(
                "cluster backup: skipping transient databases",
                extra={"extra_data": {"skipped": transient}},
            )
        skip = set(transient)
        return [name for name in named if name not in skip]

    async def _count_tables(self, conn: _Connection) -> tuple[TableCount, ...]:
        """Exact per-table row counts on ``conn`` — estimates would poison later verification.

        :param conn: the connection to count on; its transaction state decides which instant
            these counts describe, which is the whole point of the caller holding one.
        :ptype conn: _Connection
        :return: one count per base table, schema-qualified
        :rtype: tuple[TableCount, ...]
        """
        tables = await conn.fetch(_TABLES_SQL)
        counts: list[TableCount] = []
        for row in tables:
            schema, table = row["table_schema"], row["table_name"]
            count = await conn.fetchval(f'SELECT count(*) FROM "{schema}"."{table}"')
            counts.append(TableCount(schema=schema, table=table, row_count=int(cast(int, count))))
        return tuple(counts)

    @asynccontextmanager
    async def _inventory_snapshot(self, db_dsn: str) -> AsyncIterator[tuple[tuple[TableCount, ...], str | None]]:
        """Count every table inside one snapshot and hold that snapshot open for the dump.

        THE BUG THIS EXISTS FOR shipped and was found on a live cluster. The counts used to be
        taken on their own connection and the dump started afterwards under a snapshot of its
        own, so the manifest described the database at count time while the bytes described it
        at dump time. Every row written in between was in the dump and not in the count, and a
        dry run -- which compares a restored copy against the manifest that names it -- reported
        a mismatch for ordinary write traffic. Two audit tables, each off by one, on a cluster
        doing almost nothing. Under real load that verdict is noise, and noise is how a genuinely
        short restore gets waved through.

        The snapshot is not always available. When the export fails the database is still dumped,
        with counts taken beside it and ``None`` returned so the caller can record that the
        inventory is unsynchronized. Refusing the dump would be the wrong trade: the dump is the
        artifact and the count only describes it.

        :param db_dsn: dsn of the database to inventory
        :ptype db_dsn: str
        :yield: the counts, and the exported snapshot id or None when it could not be exported
        :rtype: AsyncIterator[tuple[tuple[TableCount, ...], str | None]]
        """
        conn = await self._connect(db_dsn)
        snapshot: str | None = None
        try:
            await conn.execute(_BEGIN_SNAPSHOT_SQL)
            try:
                snapshot = str(cast(str, await conn.fetchval(_EXPORT_SNAPSHOT_SQL)))
            except Exception as exc:  # prawduct:allow prawduct/broad-except -- see below
                # Deliberately broad because this package has no database driver to name an
                # exception type FROM: the connection is injected as a callable so asyncpg stays
                # out of the hard dependencies (see the module docstring), and the driver's own
                # error classes are therefore unreachable here. The failure is recorded in the
                # manifest, not swallowed.
                log.warning(
                    "cluster backup: snapshot export unavailable; inventory will not be synchronized with the dump",
                    extra={"extra_data": {"dsn_database": urlparse(db_dsn).path.lstrip("/"), "error": str(exc)}},
                )
                await conn.execute(_ROLLBACK_SQL)
            counts = await self._count_tables(conn)
            try:
                yield counts, snapshot
            finally:
                if snapshot is not None:
                    await conn.execute(_ROLLBACK_SQL)
        finally:
            await conn.close()

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
