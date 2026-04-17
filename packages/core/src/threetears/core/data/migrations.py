"""schema migration runner for agent data tables.

provides version-tracked migrations that execute through the L3
proxy. agent developers define migrations as async functions that
receive the DataStore and execute SQL. the runner tracks applied
versions in a _schema_migrations table and applies pending
migrations in order at startup.

example usage::

    migrations = MigrationRunner(store)

    @migrations.version(1)
    async def create_initial_tables(store):
        await store.create_table(TableDef(name="users", ...))

    @migrations.version(2)
    async def add_email_column(store):
        await store.execute("ALTER TABLE users ADD COLUMN email TEXT")

    @migrations.version(3)
    async def add_email_index(store):
        await store.execute("CREATE INDEX idx_users_email ON users (email)")

    await migrations.apply()
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from threetears.observe import get_logger, traced

if TYPE_CHECKING:
    from threetears.core.data.store import DataStore

log = get_logger(__name__)

MigrationFunc = Callable[["DataStore"], Awaitable[None]]

_CREATE_MIGRATIONS_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS _schema_migrations ("
    "version INTEGER PRIMARY KEY, "
    "description TEXT, "
    "date_applied TIMESTAMP DEFAULT now()"
    ")"
)

_SELECT_APPLIED_VERSIONS_SQL = "SELECT version FROM _schema_migrations ORDER BY version"

_SELECT_MAX_VERSION_SQL = "SELECT COALESCE(MAX(version), 0) AS max_version FROM _schema_migrations"

_INSERT_VERSION_SQL = "INSERT INTO _schema_migrations (version, description) VALUES ($1, $2)"


class MigrationRunner:
    """version-tracked schema migration runner for agent data tables.

    registers migration functions via version decorator and applies
    pending migrations in order through the DataStore L3 proxy.
    tracks applied versions in _schema_migrations table.

    :param store: DataStore instance for executing migrations
    :ptype store: DataStore
    """

    def __init__(self, store: DataStore) -> None:
        """initialize migration runner with DataStore reference.

        :param store: DataStore instance for executing migrations
        :ptype store: DataStore
        """
        self._store = store
        self._migrations: dict[int, MigrationFunc] = {}

    def version(self, n: int) -> Callable[[MigrationFunc], MigrationFunc]:
        """decorator to register migration function at version number.

        :param n: version number for this migration (must be unique)
        :ptype n: int
        :return: decorator that registers and returns migration function
        :rtype: Callable[[MigrationFunc], MigrationFunc]
        :raises ValueError: if version number already registered
        """

        def decorator(func: MigrationFunc) -> MigrationFunc:
            """register migration function at specified version.

            :param func: async migration function receiving DataStore
            :ptype func: MigrationFunc
            :return: unmodified migration function
            :rtype: MigrationFunc
            :raises ValueError: if version number already registered
            """
            if n in self._migrations:
                msg = f"migration version {n} already registered"
                raise ValueError(msg)
            self._migrations[n] = func
            return func

        return decorator

    @traced
    async def apply(self) -> int:
        """create tracking table and run all pending migrations in order.

        creates _schema_migrations table if it does not exist, queries
        for applied versions, then executes each pending migration in
        ascending version order. records each successful migration.
        stops and raises on first failure.

        :return: number of migrations applied
        :rtype: int
        :raises RuntimeError: if any migration function raises
        """
        await self._ensure_migrations_table()
        applied_versions = await self._get_applied_versions()
        pending_versions = sorted(v for v in self._migrations if v not in applied_versions)

        count = 0
        for version_num in pending_versions:
            func = self._migrations[version_num]
            description = func.__name__
            log.info(
                "applying migration version=%d description=%s",
                version_num,
                description,
            )
            await func(self._store)
            await self._record_version(version_num, description)
            count += 1
            log.info("migration version=%d applied successfully", version_num)

        return count

    async def current_version(self) -> int:
        """return highest applied migration version number.

        :return: highest version number or 0 if none applied
        :rtype: int
        """
        await self._ensure_migrations_table()
        rows = await self._store.query(_SELECT_MAX_VERSION_SQL)
        raw_version = rows[0]["max_version"] if rows else 0
        result = raw_version if raw_version is not None else 0
        return result

    async def pending(self) -> list[int]:
        """return sorted list of unapplied migration version numbers.

        :return: version numbers not yet recorded in _schema_migrations
        :rtype: list[int]
        """
        await self._ensure_migrations_table()
        applied_versions = await self._get_applied_versions()
        result = sorted(v for v in self._migrations if v not in applied_versions)
        return result

    async def _ensure_migrations_table(self) -> None:
        """create _schema_migrations table if it does not exist."""
        await self._store.execute(_CREATE_MIGRATIONS_TABLE_SQL)

    async def _get_applied_versions(self) -> set[int]:
        """query _schema_migrations for all applied version numbers.

        :return: set of applied version numbers
        :rtype: set[int]
        """
        rows = await self._store.query(_SELECT_APPLIED_VERSIONS_SQL)
        result = {row["version"] for row in rows}
        return result

    async def _record_version(self, version_num: int, description: str) -> None:
        """insert record of applied migration into _schema_migrations.

        :param version_num: migration version number
        :ptype version_num: int
        :param description: migration description (function name)
        :ptype description: str
        """
        await self._store.execute(_INSERT_VERSION_SQL, version_num, description)
