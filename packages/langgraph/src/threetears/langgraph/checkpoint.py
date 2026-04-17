"""Three-tier LangGraph checkpoint saver: L1 -> L2 -> L3 PostgreSQL.

Implements BaseCheckpointSaver with a tiered caching strategy:
- L3 (PostgreSQL): Source of truth via asyncpg pool. Always available.
- L2 (distributed cache): Optional hot cache (e.g. NATS KV, Redis).
- L1 (local cache): Optional in-memory/local cache (e.g. SQLite).

All L1 and L2 operations degrade gracefully on failure -- cache misses
fall through to the next tier, and cache write failures are logged and
swallowed. The graph never crashes due to cache infrastructure issues.

Cache layers are injected via protocols (see ``protocols.py``), so any
backend that implements get/put/delete works.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, cast

import asyncpg
import uuid_utils
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from threetears.langgraph.protocols import CheckpointL1Cache, CheckpointL2Cache, FlushCallback

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_DEFAULT_L2_BUCKET = "checkpoints"


class _UUIDSafeSerializer:
    """Wraps JsonPlusSerializer to convert uuid_utils.UUID to str before packing.

    asyncpg returns uuid_utils.UUID objects (not stdlib uuid.UUID) which
    ormsgpack cannot serialize. This wrapper walks the data structure and
    converts them to plain strings so the underlying serializer succeeds.
    """

    def __init__(self) -> None:
        self._inner = JsonPlusSerializer()

    @staticmethod
    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, uuid_utils.UUID):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _UUIDSafeSerializer._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_UUIDSafeSerializer._sanitize(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(_UUIDSafeSerializer._sanitize(x) for x in obj)
        return obj

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        """Serialize, converting uuid_utils.UUID to strings first."""
        return self._inner.dumps_typed(self._sanitize(obj))

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        """Deserialize (no UUID conversion needed on this path)."""
        return self._inner.loads_typed(data)


class ThreeTierCheckpointSaver(BaseCheckpointSaver[int]):
    """LangGraph checkpoint saver using three-tier caching.

    L1 and L2 are optional cache layers that sit in front of the L3
    (PostgreSQL) source of truth. Reads check L1 -> L2 -> L3, promoting
    hits into warmer tiers. Writes always go to L3 first, then warm L2
    and L1 opportunistically.

    :param postgres_pool: asyncpg connection pool for L3 storage.
    :param l1_cache: optional L1 local cache (e.g. SQLite).
    :param l2_cache: optional L2 distributed cache (e.g. NATS KV).
    :param l2_bucket: bucket/namespace for L2 cache keys.
    :param flush_callback: optional async callback invoked after each
        checkpoint write to drain pending writes.
    """

    def __init__(
        self,
        postgres_pool: asyncpg.Pool,
        *,
        l1_cache: CheckpointL1Cache | None = None,
        l2_cache: CheckpointL2Cache | None = None,
        l2_bucket: str = _DEFAULT_L2_BUCKET,
        flush_callback: FlushCallback | None = None,
    ) -> None:
        super().__init__()
        self.serde = _UUIDSafeSerializer()
        self._pool = postgres_pool
        self._l1 = l1_cache
        self._l2 = l2_cache
        self._l2_bucket = l2_bucket
        self._flush_callback = flush_callback

    # ------------------------------------------------------------------
    # L1 helpers
    # ------------------------------------------------------------------

    async def _l1_get(self, thread_id: str, checkpoint_ns: str) -> bytes | None:
        if self._l1 is None:
            return None
        try:
            return await self._l1.get(thread_id, checkpoint_ns)
        except Exception:
            logger.warning("L1 checkpoint read failed", exc_info=True)
            return None

    async def _l1_put(self, thread_id: str, checkpoint_ns: str, data: bytes) -> None:
        if self._l1 is None:
            return
        try:
            await self._l1.put(thread_id, checkpoint_ns, data)
        except Exception:
            logger.warning("L1 checkpoint write failed", exc_info=True)

    async def _l1_delete(self, thread_id: str) -> None:
        if self._l1 is None:
            return
        try:
            await self._l1.delete(thread_id)
        except Exception:
            logger.warning("L1 checkpoint delete failed", exc_info=True)

    # ------------------------------------------------------------------
    # L2 helpers
    # ------------------------------------------------------------------

    def _l2_key(self, thread_id: str, checkpoint_ns: str) -> str:
        if checkpoint_ns == "":
            return thread_id
        return f"{thread_id}.{checkpoint_ns}"

    async def _l2_get(self, thread_id: str, checkpoint_ns: str) -> bytes | None:
        if self._l2 is None:
            return None
        try:
            return await self._l2.get(self._l2_bucket, self._l2_key(thread_id, checkpoint_ns))
        except Exception:
            logger.warning("L2 checkpoint read failed", exc_info=True)
            return None

    async def _l2_put(self, thread_id: str, checkpoint_ns: str, data: bytes) -> None:
        if self._l2 is None:
            return
        try:
            await self._l2.put(self._l2_bucket, self._l2_key(thread_id, checkpoint_ns), data)
        except Exception:
            logger.warning("L2 checkpoint write failed", exc_info=True)

    async def _l2_delete(self, thread_id: str) -> None:
        if self._l2 is None:
            return
        try:
            await self._l2.delete(self._l2_bucket, self._l2_key(thread_id, ""))
        except Exception:
            logger.warning("L2 checkpoint delete failed", exc_info=True)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def _serialize_checkpoint_tuple(
        self,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        parent_checkpoint_id: str | None,
        pending_writes: list[tuple[str, str, Any]] | None,
    ) -> bytes:
        """Serialize a full checkpoint tuple for cache storage (L1/L2)."""
        bundle = {
            "checkpoint": checkpoint,
            "metadata": metadata,
            "parent_checkpoint_id": parent_checkpoint_id,
            "pending_writes": pending_writes,
        }
        _type, blob = self.serde.dumps_typed(bundle)
        type_bytes = _type.encode("utf-8")
        return len(type_bytes).to_bytes(4, "big") + type_bytes + blob

    def _deserialize_checkpoint_tuple(self, data: bytes) -> dict[str, Any]:
        """Deserialize a checkpoint tuple from cache blob."""
        type_len = int.from_bytes(data[:4], "big")
        type_str = data[4 : 4 + type_len].decode("utf-8")
        blob = data[4 + type_len :]
        result: dict[str, Any] = self.serde.loads_typed((type_str, blob))
        return result

    # ------------------------------------------------------------------
    # Async interface -- required by LangGraph
    # ------------------------------------------------------------------

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Fetch latest or specific checkpoint: L1 -> L2 -> L3."""
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        if checkpoint_id is None:
            # --- L1 attempt ---
            cached = await self._l1_get(thread_id, checkpoint_ns)
            if cached is not None:
                try:
                    bundle = self._deserialize_checkpoint_tuple(cached)
                    return self._bundle_to_tuple(thread_id, checkpoint_ns, bundle)
                except Exception:
                    logger.warning("L1 checkpoint deserialization failed, falling through", exc_info=True)

            # --- L2 attempt ---
            cached = await self._l2_get(thread_id, checkpoint_ns)
            if cached is not None:
                try:
                    bundle = self._deserialize_checkpoint_tuple(cached)
                    tup = self._bundle_to_tuple(thread_id, checkpoint_ns, bundle)
                    await self._l1_put(thread_id, checkpoint_ns, cached)
                    return tup
                except Exception:
                    logger.warning("L2 checkpoint deserialization failed, falling through", exc_info=True)

        # --- L3 (PostgreSQL) ---
        return await self._l3_get_tuple(thread_id, checkpoint_ns, checkpoint_id)

    async def _l3_get_tuple(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str | None,
    ) -> CheckpointTuple | None:
        """Load checkpoint from PostgreSQL (L3)."""
        async with self._pool.acquire() as conn:
            if checkpoint_id:
                row = await conn.fetchrow(
                    "SELECT checkpoint_id, parent_checkpoint_id, type, "
                    "checkpoint, metadata_ "
                    "FROM checkpoints "
                    "WHERE thread_id = $1 AND checkpoint_ns = $2 "
                    "AND checkpoint_id = $3",
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                )
            else:
                row = await conn.fetchrow(
                    "SELECT checkpoint_id, parent_checkpoint_id, type, "
                    "checkpoint, metadata_ "
                    "FROM checkpoints "
                    "WHERE thread_id = $1 AND checkpoint_ns = $2 "
                    "ORDER BY checkpoint_id DESC LIMIT 1",
                    thread_id,
                    checkpoint_ns,
                )

            if row is None:
                return None

            cp_id = row["checkpoint_id"]
            parent_id = row["parent_checkpoint_id"]
            cp_type = row["type"]
            cp_blob = bytes(row["checkpoint"])
            md_blob = bytes(row["metadata_"])

            checkpoint: Checkpoint = self.serde.loads_typed((cp_type or "msgpack", cp_blob))
            metadata: CheckpointMetadata = cast(
                CheckpointMetadata,
                (self.serde.loads_typed((cp_type or "msgpack", md_blob)) if md_blob and md_blob != b"\x00" else {}),
            )

            write_rows = await conn.fetch(
                "SELECT task_id, channel, type, blob, task_path "
                "FROM checkpoint_writes "
                "WHERE thread_id = $1 AND checkpoint_ns = $2 "
                "AND checkpoint_id = $3 "
                "ORDER BY idx",
                thread_id,
                checkpoint_ns,
                cp_id,
            )
            pending_writes: list[tuple[str, str, Any]] = []
            for wr in write_rows:
                pending_writes.append(
                    (
                        wr["task_id"],
                        wr["channel"],
                        self.serde.loads_typed((wr["type"] or "msgpack", bytes(wr["blob"]))),
                    )
                )

        result_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": cp_id,
            }
        }
        parent_config: RunnableConfig | None = (
            {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns, "checkpoint_id": parent_id}}
            if parent_id
            else None
        )

        tup = CheckpointTuple(
            config=result_config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

        # Warm L1 and L2
        try:
            cache_blob = self._serialize_checkpoint_tuple(checkpoint, metadata, parent_id, pending_writes)
            await self._l2_put(thread_id, checkpoint_ns, cache_blob)
            await self._l1_put(thread_id, checkpoint_ns, cache_blob)
        except Exception:
            logger.warning("Failed to warm caches after L3 read", exc_info=True)

        return tup

    def _bundle_to_tuple(
        self,
        thread_id: str,
        checkpoint_ns: str,
        bundle: dict[str, Any],
    ) -> CheckpointTuple:
        """Convert a deserialized cache bundle back to CheckpointTuple."""
        checkpoint = bundle["checkpoint"]
        metadata = bundle["metadata"]
        parent_checkpoint_id = bundle.get("parent_checkpoint_id")
        pending_writes = bundle.get("pending_writes")

        cp_id = checkpoint.get("id", "")

        result_config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns, "checkpoint_id": cp_id}
        }
        parent_config: RunnableConfig | None = (
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_checkpoint_id,
                }
            }
            if parent_checkpoint_id
            else None
        )

        return CheckpointTuple(
            config=result_config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from L3 (PostgreSQL) directly."""
        if config is None:
            return

        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")

        query = (
            "SELECT checkpoint_id, parent_checkpoint_id, type, "
            "checkpoint, metadata_ "
            "FROM checkpoints "
            "WHERE thread_id = $1 AND checkpoint_ns = $2"
        )
        params: list[Any] = [thread_id, checkpoint_ns]

        if before and (before_id := get_checkpoint_id(before)):
            query += f" AND checkpoint_id < ${len(params) + 1}"
            params.append(before_id)

        query += " ORDER BY checkpoint_id DESC"

        if limit is not None:
            query += f" LIMIT ${len(params) + 1}"
            params.append(limit)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        for row in rows:
            cp_id = row["checkpoint_id"]
            parent_id = row["parent_checkpoint_id"]
            cp_type = row["type"]
            cp_blob = bytes(row["checkpoint"])
            md_blob = bytes(row["metadata_"])

            checkpoint: Checkpoint = self.serde.loads_typed((cp_type or "msgpack", cp_blob))
            metadata: CheckpointMetadata = cast(
                CheckpointMetadata,
                (self.serde.loads_typed((cp_type or "msgpack", md_blob)) if md_blob and md_blob != b"\x00" else {}),
            )

            if filter and not all(metadata.get(k) == v for k, v in filter.items()):
                continue

            result_config: RunnableConfig = {
                "configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns, "checkpoint_id": cp_id}
            }
            parent_config: RunnableConfig | None = (
                {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns, "checkpoint_id": parent_id}}
                if parent_id
                else None
            )

            yield CheckpointTuple(
                config=result_config,
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
                pending_writes=None,
            )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store a checkpoint: write to L3, then warm L2 and L1."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        serializable_metadata = get_checkpoint_metadata(config, metadata)

        cp_type, cp_blob = self.serde.dumps_typed(checkpoint)
        _md_type, md_blob = self.serde.dumps_typed(serializable_metadata)

        # --- L3: PostgreSQL (source of truth) ---
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO checkpoints "
                "(thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, "
                "type, checkpoint, metadata_) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7) "
                "ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id) "
                "DO UPDATE SET parent_checkpoint_id = EXCLUDED.parent_checkpoint_id, "
                "type = EXCLUDED.type, checkpoint = EXCLUDED.checkpoint, "
                "metadata_ = EXCLUDED.metadata_",
                thread_id,
                checkpoint_ns,
                checkpoint["id"],
                parent_checkpoint_id,
                cp_type,
                cp_blob,
                md_blob,
            )

        result_config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns, "checkpoint_id": checkpoint["id"]}
        }

        # --- Warm L2 and L1 caches ---
        try:
            cache_blob = self._serialize_checkpoint_tuple(checkpoint, serializable_metadata, parent_checkpoint_id, [])
            await self._l2_put(thread_id, checkpoint_ns, cache_blob)
            await self._l1_put(thread_id, checkpoint_ns, cache_blob)
        except Exception:
            logger.warning("Failed to warm caches after L3 write", exc_info=True)

        # --- Flush callback ---
        if self._flush_callback is not None:
            try:
                flushed = await self._flush_callback()
                if flushed > 0:
                    logger.debug("Flushed pending writes on checkpoint", extra={"flushed_count": flushed})
            except Exception:
                logger.warning("Failed to flush pending writes on checkpoint", exc_info=True)

        return result_config

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes to L3 only (crash recovery, not hot path)."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        async with self._pool.acquire() as conn:
            for idx, (channel, value) in enumerate(writes):
                write_idx = WRITES_IDX_MAP.get(channel, idx)
                w_type, w_blob = self.serde.dumps_typed(value)

                await conn.execute(
                    "INSERT INTO checkpoint_writes "
                    "(thread_id, checkpoint_ns, checkpoint_id, task_id, "
                    "task_path, idx, channel, type, blob) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) "
                    "ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) "
                    "DO NOTHING",
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    task_id,
                    task_path,
                    write_idx,
                    channel,
                    w_type,
                    w_blob,
                )

    async def adelete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes for a thread from all tiers."""
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = $1", thread_id)
            await conn.execute("DELETE FROM checkpoints WHERE thread_id = $1", thread_id)

        await self._l2_delete(thread_id)
        await self._l1_delete(thread_id)

    # ------------------------------------------------------------------
    # Sync methods -- not supported (async-only application)
    # ------------------------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        raise NotImplementedError("ThreeTierCheckpointSaver is async-only. Use aget_tuple().")

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        raise NotImplementedError("ThreeTierCheckpointSaver is async-only. Use alist().")

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        raise NotImplementedError("ThreeTierCheckpointSaver is async-only. Use aput().")

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        raise NotImplementedError("ThreeTierCheckpointSaver is async-only. Use aput_writes().")

    def delete_thread(self, thread_id: str) -> None:
        raise NotImplementedError("ThreeTierCheckpointSaver is async-only. Use adelete_thread().")
