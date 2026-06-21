"""agent-scoped data layer with schema creation and collection-based access."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from threetears.core.collections.base import BaseCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import CoreConfig, DefaultCoreConfig
from threetears.observe import get_logger, traced

from threetears.core.data.collection_factory import create_dynamic_collection
from threetears.core.data.schema import TableDef
from threetears.core.data.sql_builder import build_create_index_sql, build_create_table_sql

__all__ = [
    "DataStore",
]

if TYPE_CHECKING:
    from threetears.core.data.migrations import MigrationRunner

log = get_logger(__name__)


class DataStore:
    """agent-scoped data layer with schema creation and collection-based access.

    wraps a CollectionRegistry and creates dynamic BaseCollection subclasses
    for each agent table. tables are namespaced to the agent's private
    YugabyteDB schema. collections provide three-tier caching (L1/L2/L3),
    change tracking, and entity-style access.

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

    @traced
    async def create_table(self, table_def: TableDef) -> BaseCollection[Any]:
        """create table in agent schema and register dynamic collection.

        builds CREATE TABLE SQL from definition, executes via L3 pool,
        then creates and executes any index SQL. generates a dynamic
        BaseCollection subclass and registers it with the CollectionRegistry.

        the collection resolves its L2 (NATS) client from the registry:
        wire it via ``registry.configure(l2_client=...)`` or
        ``registry.bind_table(table_name, l2_client=...)`` before calling
        this method.

        :param table_def: complete table definition with columns, indexes, and foreign keys
        :ptype table_def: TableDef
        :return: BaseCollection instance for created table
        :rtype: BaseCollection
        """
        l3_pool = self._registry.get_l3_pool(table_def.name)
        if l3_pool is None:
            raise RuntimeError("DataStore requires a configured L3 backend (CollectionRegistry.configure(l3_pool=...))")

        create_sql = build_create_table_sql(table_def)
        await l3_pool.execute(create_sql)

        for index_def in table_def.indexes:
            index_sql = build_create_index_sql(table_def.name, index_def)
            await l3_pool.execute(index_sql)

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
        l3_pool = self._registry.get_l3_pool("_raw")
        if l3_pool is None:
            raise RuntimeError("DataStore requires a configured L3 backend (CollectionRegistry.configure(l3_pool=...))")
        rows = await l3_pool.fetch(sql, *params)
        # convert at border: asyncpg Records iterate values, not keys --
        # honor the declared dict row shape regardless of pool driver.
        result: list[dict[str, Any]] = [dict(row) for row in rows]
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
        l3_pool = self._registry.get_l3_pool("_raw")
        if l3_pool is None:
            raise RuntimeError("DataStore requires a configured L3 backend (CollectionRegistry.configure(l3_pool=...))")
        result: str = await l3_pool.execute(sql, *params)
        return result

    @traced
    async def run_migrations(self, runner: MigrationRunner) -> int:
        """run pending agent-scope migrations against this store's schema.

        convenience method that delegates to
        :meth:`MigrationRunner.apply_for_agent_schema`. :class:`DataStore`
        is constructed with an ``agent_id`` so its bound schema is the
        per-agent schema; the runner's agent-scope entry point is the
        only correct delegation target.

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
