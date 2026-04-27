"""Cache protocols for the three-tier checkpoint saver.

L1 and L2 are optional cache layers that sit in front of PostgreSQL.
Host applications provide implementations matching these protocols.

the module also ships :class:`AsyncpgPoolAdapter`, a thin adapter
that exposes a raw :class:`asyncpg.Pool` as an
:class:`AsyncQueryExecutor`. trusted services that hold a direct
pool (e.g. the hub) wrap once at construction; sandboxed agents
pass :class:`~threetears.core.backends.nats_proxy.NatsProxyL3Backend`
straight through because it implements the protocol natively.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import asyncpg

__all__ = [
    "AsyncQueryExecutor",
    "AsyncpgPoolAdapter",
    "CheckpointL1Cache",
    "CheckpointL2Cache",
    "FlushCallback",
]


@runtime_checkable
class CheckpointL1Cache(Protocol):
    """Protocol for L1 (local, fast) checkpoint cache.

    Implementations should be fast and synchronous-friendly (e.g. SQLite).
    All methods must be async to allow asyncio.to_thread() wrapping.
    Failures should raise exceptions — the checkpointer catches and degrades.
    """

    async def get(self, thread_id: str, checkpoint_ns: str) -> bytes | None:
        """Read a cached checkpoint blob. Returns None on miss."""
        ...

    async def put(self, thread_id: str, checkpoint_ns: str, data: bytes) -> None:
        """Write a checkpoint blob to cache."""
        ...

    async def delete(self, thread_id: str) -> None:
        """Delete all cached checkpoints for a thread."""
        ...


@runtime_checkable
class CheckpointL2Cache(Protocol):
    """Protocol for L2 (distributed, shared) checkpoint cache.

    Implementations should be network-backed (e.g. NATS KV, Redis).
    Uses simple key-value semantics with string keys and bytes values.
    Failures should raise exceptions — the checkpointer catches and degrades.
    """

    async def get(self, bucket: str, key: str) -> bytes | None:
        """Read a value from the cache. Returns None on miss."""
        ...

    async def put(self, bucket: str, key: str, value: bytes) -> None:
        """Write a value to the cache."""
        ...

    async def delete(self, bucket: str, key: str) -> None:
        """Delete a key from the cache."""
        ...


@runtime_checkable
class AsyncQueryExecutor(Protocol):
    """Protocol for async SQL query execution.

    abstracts the database access layer so checkpoint savers can work
    with any backend: direct asyncpg, NATS L3 proxy, or other transports.
    implementations return dict-like rows with string keys. row values
    are typed ``Any`` because SQL column values are dynamically typed at
    the database boundary; callers cast or validate at use sites.
    """

    async def fetch(self, query: str, *args: object) -> list[dict[str, Any]]:
        """execute query and return all matching rows as dicts."""
        ...

    async def fetchrow(self, query: str, *args: object) -> dict[str, Any] | None:
        """execute query and return first row as dict, or None if empty."""
        ...

    async def execute(self, query: str, *args: object) -> str:
        """execute statement and return status string."""
        ...


@runtime_checkable
class FlushCallback(Protocol):
    """Protocol for an optional write-buffer flush on checkpoint.

    Called after each checkpoint write to drain pending writes.
    Returns the number of items flushed.
    """

    async def __call__(self) -> int:
        """Flush pending writes. Returns count of items flushed."""
        ...


class AsyncpgPoolAdapter:
    """adapt :class:`asyncpg.Pool` to :class:`AsyncQueryExecutor`.

    :class:`asyncpg.Pool` exposes ``fetch``/``fetchrow``/``execute``
    but returns :class:`asyncpg.Record` objects rather than plain
    dicts. the :class:`AsyncQueryExecutor` protocol declares
    ``list[dict[str, Any]]`` / ``dict[str, Any] | None`` return types,
    matching what :class:`~threetears.core.backends.nats_proxy.NatsProxyL3Backend`
    produces. this adapter converts records to dicts so a direct-
    pool caller and a proxy-backed caller deliver identical row
    shapes to :class:`~threetears.langgraph.checkpoint.ThreeTierCheckpointSaver`.

    construction is cheap; callers wrap their pool once at wire-up
    and forward the adapter wherever the checkpointer wants an
    :class:`AsyncQueryExecutor`.

    :param pool: asyncpg connection pool
    :ptype pool: asyncpg.Pool
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        """capture the pool for later delegation.

        :param pool: asyncpg connection pool
        :ptype pool: asyncpg.Pool
        :return: nothing
        :rtype: None
        """
        self._pool = pool

    async def fetch(self, query: str, *args: object) -> list[dict[str, Any]]:
        """execute SELECT and return rows as dicts.

        :param query: parameterized SQL query
        :ptype query: str
        :param args: query parameter values
        :ptype args: object
        :return: list of row dictionaries
        :rtype: list[dict[str, Any]]
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
        return [dict(row) for row in rows]

    async def fetchrow(
        self,
        query: str,
        *args: object,
    ) -> dict[str, Any] | None:
        """execute SELECT and return first row as dict.

        :param query: parameterized SQL query
        :ptype query: str
        :param args: query parameter values
        :ptype args: object
        :return: first row dict or None
        :rtype: dict[str, Any] | None
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, *args)
        result: dict[str, Any] | None = dict(row) if row is not None else None
        return result

    async def execute(self, query: str, *args: object) -> str:
        """execute statement and return asyncpg status tag.

        :param query: parameterized SQL query
        :ptype query: str
        :param args: query parameter values
        :ptype args: object
        :return: asyncpg-style status tag (e.g. ``"UPDATE 3"``)
        :rtype: str
        """
        async with self._pool.acquire() as conn:
            status: str = await conn.execute(query, *args)
        return status
