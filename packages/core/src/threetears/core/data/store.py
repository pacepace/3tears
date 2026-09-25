"""agent-scoped data layer with schema creation and collection-based access."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any
from uuid import UUID

from threetears.core.backends.protocol import L3Backend
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import CoreConfig, DefaultCoreConfig
from threetears.observe import get_logger, traced

from threetears.core.data.collection_factory import create_dynamic_collection
from threetears.core.data.migrations.ddl_lock import DdlLockPolicy, database_ddl_lock
from threetears.core.data.migrations.session import ConnectionSession, MigrationSession
from threetears.core.data.schema import TableDef
from threetears.core.data.sql_builder import build_create_index_sql, build_create_table_sql

__all__ = [
    "DataStore",
]

if TYPE_CHECKING:
    from threetears.core.data.migrations import MigrationRunner

log = get_logger(__name__)

#: the agent id recorded on a store built over a caller's session by
#: :meth:`DataStore.over_session`; such a store belongs to no agent, and
#: nothing reads the id
_SESSION_STORE_AGENT_ID = UUID(int=0)

_RAW_BACKEND_KEY = "_raw"


class DataStore:
    """agent-scoped data layer with schema creation and collection-based access.

    wraps a CollectionRegistry and creates dynamic BaseCollection subclasses
    for each agent table. tables are namespaced to the agent's private
    YugabyteDB schema. collections provide three-tier caching (L1/L2/L3),
    change tracking, and entity-style access.

    a store either sends each statement to its registry's L3 backend -- a
    pool, one borrowed connection per statement -- or is BOUND to one database
    session and sends every statement there. a bound store comes from
    :meth:`ddl_session` or :meth:`over_session`, and
    shares its registry, config and collections with the store it came from,
    so a collection a bound store creates is registered where the pool-backed
    store (and everything after the session ends) finds it.

    DDL takes the database-wide DDL lock
    (:func:`~threetears.core.data.migrations.ddl_lock.database_ddl_lock`):
    two index builds in one YugabyteDB database hang each other until the
    catalog-version wait times out. :meth:`create_table` takes it itself; a
    store from :meth:`ddl_session` already holds it and does not take it
    again.

    :param agent_id: unique agent identifier for schema namespacing
    :ptype agent_id: UUID
    :param registry: collection registry for dependency injection
    :ptype registry: CollectionRegistry
    :param config: core configuration for flush strategy and caching
    :ptype config: CoreConfig | None
    """

    def __init__(
        self,
        agent_id: UUID,
        registry: CollectionRegistry,
        config: CoreConfig | None = None,
    ) -> None:
        """initialize with agent UUID and collection registry.

        :param agent_id: unique agent identifier for schema namespacing
        :ptype agent_id: UUID
        :param registry: collection registry for dependency injection
        :ptype registry: CollectionRegistry
        :param config: core configuration for flush strategy and caching
        :ptype config: CoreConfig | None
        """
        self._agent_id = agent_id
        self._registry = registry
        self._config: CoreConfig = config or DefaultCoreConfig()
        self._schema_name = f"agent_{agent_id.hex}"
        self._collections: dict[str, BaseCollection[Any]] = {}
        self._session: MigrationSession | None = None
        self._holds_ddl_lock = False

    @classmethod
    def over_session(
        cls,
        session: MigrationSession,
        config: CoreConfig | None = None,
        *,
        holds_ddl_lock: bool = False,
    ) -> DataStore:
        """build a store that sends every statement to one existing database session.

        for a caller that already holds one connection -- wrapped in a
        :class:`~threetears.core.data.migrations.session.ConnectionSession`, or
        any other one-session store -- and needs the DataStore surface over it.
        the migration runner builds its bodies' store this way when it is given
        a plain session; given a pool-backed DataStore, it binds that store to
        one acquired connection with :meth:`ddl_session` instead.
        the store gets a registry of its own, with no L3 backend: collections
        it creates are registered there, and reach the database only through
        whoever wires that registry.

        :param session: one database session, owned by the caller
        :ptype session: MigrationSession
        :param config: core configuration for collections the store creates
        :ptype config: CoreConfig | None
        :param holds_ddl_lock: True only when ``session`` already holds the
            database-wide DDL lock for this store's whole life, or runs no
            DDL at all (a preview that captures it); the store then never
            takes the lock itself
        :ptype holds_ddl_lock: bool
        :return: a store bound to ``session``
        :rtype: DataStore
        """
        store = DataStore(_SESSION_STORE_AGENT_ID, CollectionRegistry(), config)
        store._session = session
        store._holds_ddl_lock = holds_ddl_lock
        return store

    @property
    def is_bound_to_session(self) -> bool:
        """whether every statement this store issues goes to one database session.

        :return: True for a store from :meth:`ddl_session` or
            :meth:`over_session`
        :rtype: bool
        """
        return self._session is not None

    @property
    def holds_ddl_lock(self) -> bool:
        """whether this store's session holds the database-wide DDL lock for it.

        :return: True for a store yielded by :meth:`ddl_session`, or built
            by :meth:`over_session` with ``holds_ddl_lock=True``
        :rtype: bool
        """
        return self._holds_ddl_lock

    @asynccontextmanager
    async def ddl_session(self, policy: DdlLockPolicy | None = None) -> AsyncIterator[DataStore]:
        """yield a store bound to one connection that holds the database-wide DDL lock.

        a store that already holds the lock yields itself, so DDL inside a
        migration run never takes it twice. a store bound to a session takes
        the lock on that session; any other store acquires one connection
        first. the lock is released, and an acquired connection handed back,
        when the block ends.

        :param policy: how to wait for the lock; defaults to a 1s poll with no
            deadline
        :ptype policy: DdlLockPolicy | None
        :return: async context manager yielding the locked store
        :rtype: AsyncIterator[DataStore]
        :raises RuntimeError: when no L3 backend is configured
        :raises DdlLockTimeoutError: when the lock stayed held past
            ``policy.max_wait``
        :raises DdlLockReleaseError: when the lock could not be given back
            after a block that completed
        """
        async with self._ddl_session(policy, _RAW_BACKEND_KEY) as locked:
            yield locked

    @traced
    async def create_table(self, table_def: TableDef) -> BaseCollection[Any]:
        """create table in agent schema and register dynamic collection.

        builds CREATE TABLE SQL from definition and runs it, then any index
        SQL, on one connection holding the database-wide DDL lock -- the
        store's own session when it already holds the lock (a migration body),
        otherwise one taken for this call. generates a dynamic
        BaseCollection subclass and registers it with the CollectionRegistry.

        the collection resolves its L2 (NATS) client from the registry:
        wire it via ``registry.configure(l2_client=...)`` or
        ``registry.bind_table(table_name, l2_client=...)`` -- which requires the registry
        to already carry a ``kv_key_scope``, and raises without one -- before calling
        this method.

        :param table_def: complete table definition with columns, indexes, and foreign keys
        :ptype table_def: TableDef
        :return: BaseCollection instance for created table
        :rtype: BaseCollection
        :raises RuntimeError: when no L3 backend is configured
        :raises DdlLockTimeoutError: when the DDL lock stayed held past the
            default policy's deadline (none by default)
        """
        create_sql = build_create_table_sql(table_def)
        async with self._ddl_session(None, table_def.name) as locked:
            await locked.execute(create_sql)
            for index_def in table_def.indexes:
                index_sql = build_create_index_sql(table_def.name, index_def)
                await locked.execute(index_sql)

        collection = create_dynamic_collection(
            table_def=table_def,
            registry=self._registry,
            config=self._config,
        )
        self._collections[table_def.name] = collection

        return collection

    @traced
    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """execute raw SQL query against agent schema.

        :param sql: SQL query string with $N parameter placeholders
        :ptype sql: str
        :param params: positional parameter values for query
        :ptype params: Any
        :return: list of row dictionaries from query result
        :rtype: list[dict[str, Any]]
        """
        result: list[dict[str, Any]]
        if self._session is not None:
            result = await self._session.query(sql, *params)
        else:
            rows = await self._l3_backend(_RAW_BACKEND_KEY).fetch(sql, *params)
            # convert at border: asyncpg Records iterate values, not keys --
            # honor the declared dict row shape regardless of pool driver.
            result = [dict(row) for row in rows]
        return result

    @traced
    async def execute(self, sql: str, *params: Any) -> str:
        """execute SQL statement against agent schema.

        :param sql: SQL statement string with $N parameter placeholders
        :ptype sql: str
        :param params: positional parameter values for statement
        :ptype params: Any
        :return: execution status string from database
        :rtype: str
        """
        result: str
        if self._session is not None:
            result = await self._session.execute(sql, *params)
        else:
            result = await self._l3_backend(_RAW_BACKEND_KEY).execute(sql, *params)
        return result

    @traced
    async def run_migrations(self, runner: MigrationRunner) -> int:
        """run pending agent-scope migrations against this store's schema.

        convenience method that delegates to
        :meth:`MigrationRunner.apply_for_agent_schema`, which runs the whole
        apply on one connection holding the database-wide DDL lock and hands
        each migration body this store bound to that connection.

        :param runner: migration runner with registered packages
        :ptype runner: MigrationRunner
        :return: number of migrations applied across all agent packages
        :rtype: int
        """
        result: int = await runner.apply_for_agent_schema(self)
        return result

    def __getitem__(self, table_name: str) -> BaseCollection[Any]:
        """get collection by table name for entity-style data access.

        :param table_name: name of table to get collection for
        :ptype table_name: str
        :return: BaseCollection for specified table
        :rtype: BaseCollection
        :raises KeyError: if table has not been created via create_table
        """
        result = self._collections[table_name]
        return result

    @asynccontextmanager
    async def _ddl_session(self, policy: DdlLockPolicy | None, backend_key: str) -> AsyncIterator[DataStore]:
        """take the DDL lock on one session, acquiring a connection from ``backend_key`` if needed.

        :param policy: how to wait for the lock
        :ptype policy: DdlLockPolicy | None
        :param backend_key: registry key whose L3 backend supplies the
            connection when this store is not bound to a session
        :ptype backend_key: str
        :return: async context manager yielding the locked store
        :rtype: AsyncIterator[DataStore]
        """
        if self._holds_ddl_lock:
            yield self
        elif self._session is not None:
            async with database_ddl_lock(self._session, policy):
                yield self._bound_view(self._session, holds_ddl_lock=True)
        else:
            async with self._l3_backend(backend_key).acquire() as connection:
                session = ConnectionSession(connection)
                async with database_ddl_lock(session, policy):
                    yield self._bound_view(session, holds_ddl_lock=True)

    def _bound_view(self, session: MigrationSession, *, holds_ddl_lock: bool) -> DataStore:
        """build the one kind of bound store: this store's state, over one session.

        :param session: the session every statement goes to
        :ptype session: MigrationSession
        :param holds_ddl_lock: whether the session holds the DDL lock for the view
        :ptype holds_ddl_lock: bool
        :return: a store sharing this store's registry, config and collections
        :rtype: DataStore
        """
        view = DataStore(self._agent_id, self._registry, self._config)
        view._collections = self._collections
        view._session = session
        view._holds_ddl_lock = holds_ddl_lock
        return view

    def _l3_backend(self, backend_key: str) -> L3Backend:
        """return the registry's L3 backend for ``backend_key``, or refuse.

        :param backend_key: table name, or ``"_raw"`` for raw statements
        :ptype backend_key: str
        :return: the configured backend
        :rtype: L3Backend
        :raises RuntimeError: when no L3 backend is configured
        """
        backend = self._registry.get_l3_pool(backend_key)
        if backend is None:
            raise RuntimeError("DataStore requires a configured L3 backend (CollectionRegistry.configure(l3_pool=...))")
        return backend
