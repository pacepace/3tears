"""Proxy-based LangGraph checkpoint saver for sandboxed environments.

implements BaseCheckpointSaver using an AsyncQueryExecutor protocol,
enabling checkpoint persistence through any SQL transport layer (NATS
L3 proxy, HTTP API, etc.) without requiring direct database access.

agents in sandboxed environments (no DB credentials) use this saver
to persist LangGraph state through their platform's query proxy.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, cast

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
from threetears.langgraph.protocols import (
    AsyncQueryExecutor,
    CheckpointL1Cache,
    CheckpointL2Cache,
)
from threetears.langgraph.serde import UUIDSafeSerializer

__all__ = [
    "ProxyCheckpointSaver",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ProxyCheckpointSaver(BaseCheckpointSaver[int]):
    """LangGraph checkpoint saver using an async query executor proxy.

    designed for sandboxed agents that access their database through
    a proxy layer (e.g. NATS L3 broker) rather than direct connections.
    uses the AsyncQueryExecutor protocol so any transport works.

    optional L1/L2 cache layers provide the same tiered caching as
    ThreeTierCheckpointSaver for read performance.

    :param executor: async query executor for database operations
    :ptype executor: AsyncQueryExecutor
    :param l1_cache: optional L1 local cache
    :ptype l1_cache: CheckpointL1Cache | None
    :param l2_cache: optional L2 distributed cache
    :ptype l2_cache: CheckpointL2Cache | None
    :param l2_bucket: bucket name for L2 cache keys
    :ptype l2_bucket: str
    """

    def __init__(
        self,
        executor: AsyncQueryExecutor,
        *,
        l1_cache: CheckpointL1Cache | None = None,
        l2_cache: CheckpointL2Cache | None = None,
        l2_bucket: str = "checkpoints",
    ) -> None:
        """initialize proxy checkpoint saver.

        :param executor: async query executor for database operations
        :ptype executor: AsyncQueryExecutor
        :param l1_cache: optional L1 local cache
        :ptype l1_cache: CheckpointL1Cache | None
        :param l2_cache: optional L2 distributed cache
        :ptype l2_cache: CheckpointL2Cache | None
        :param l2_bucket: bucket name for L2 cache keys
        :ptype l2_bucket: str
        """
        super().__init__()
        self.serde = UUIDSafeSerializer()
        self._exec = executor
        self._l1 = l1_cache
        self._l2 = l2_cache
        self._l2_bucket = l2_bucket

    # ------------------------------------------------------------------
    # L1 helpers
    # ------------------------------------------------------------------

    async def _l1_get(self, thread_id: str, checkpoint_ns: str) -> bytes | None:
        """read from L1 cache, returning None on miss or error."""
        if self._l1 is None:
            return None
        try:
            result = await self._l1.get(thread_id, checkpoint_ns)
        except Exception:
            logger.warning("L1 checkpoint read failed", exc_info=True)
            result = None
        return result

    async def _l1_put(self, thread_id: str, checkpoint_ns: str, data: bytes) -> None:
        """write to L1 cache, swallowing errors."""
        if self._l1 is None:
            return
        try:
            await self._l1.put(thread_id, checkpoint_ns, data)
        except Exception:
            logger.warning("L1 checkpoint write failed", exc_info=True)

    # ------------------------------------------------------------------
    # L2 helpers
    # ------------------------------------------------------------------

    def _l2_key(self, thread_id: str, checkpoint_ns: str) -> str:
        """build L2 cache key from thread and namespace."""
        if checkpoint_ns == "":
            return thread_id
        return f"{thread_id}.{checkpoint_ns}"

    async def _l2_get(self, thread_id: str, checkpoint_ns: str) -> bytes | None:
        """read from L2 cache, returning None on miss or error."""
        if self._l2 is None:
            return None
        try:
            result = await self._l2.get(self._l2_bucket, self._l2_key(thread_id, checkpoint_ns))
        except Exception:
            logger.warning("L2 checkpoint read failed", exc_info=True)
            result = None
        return result

    async def _l2_put(self, thread_id: str, checkpoint_ns: str, data: bytes) -> None:
        """write to L2 cache, swallowing errors."""
        if self._l2 is None:
            return
        try:
            await self._l2.put(self._l2_bucket, self._l2_key(thread_id, checkpoint_ns), data)
        except Exception:
            logger.warning("L2 checkpoint write failed", exc_info=True)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def _serialize_bundle(
        self,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        parent_checkpoint_id: str | None,
        pending_writes: list[tuple[str, str, Any]] | None,
    ) -> bytes:
        """serialize checkpoint tuple for cache storage."""
        bundle = {
            "checkpoint": checkpoint,
            "metadata": metadata,
            "parent_checkpoint_id": parent_checkpoint_id,
            "pending_writes": pending_writes,
        }
        _type, blob = self.serde.dumps_typed(bundle)
        type_bytes = _type.encode("utf-8")
        return len(type_bytes).to_bytes(4, "big") + type_bytes + blob

    def _deserialize_bundle(self, data: bytes) -> dict[str, Any]:
        """deserialize checkpoint tuple from cache blob."""
        type_len = int.from_bytes(data[:4], "big")
        type_str = data[4 : 4 + type_len].decode("utf-8")
        blob = data[4 + type_len :]
        result: dict[str, Any] = self.serde.loads_typed((type_str, blob))
        return result

    def _bundle_to_tuple(
        self,
        thread_id: str,
        checkpoint_ns: str,
        bundle: dict[str, Any],
    ) -> CheckpointTuple:
        """convert deserialized cache bundle to CheckpointTuple."""
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

    # ------------------------------------------------------------------
    # Async interface -- required by LangGraph
    # ------------------------------------------------------------------

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """fetch latest or specific checkpoint via proxy: L1 -> L2 -> proxy.

        :param config: runnable config with thread_id in configurable
        :ptype config: RunnableConfig
        :return: checkpoint tuple or None if not found
        :rtype: CheckpointTuple | None
        """
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        if checkpoint_id is None:
            # L1 attempt
            cached = await self._l1_get(thread_id, checkpoint_ns)
            if cached is not None:
                try:
                    bundle = self._deserialize_bundle(cached)
                    return self._bundle_to_tuple(thread_id, checkpoint_ns, bundle)
                except Exception:
                    logger.warning("L1 deserialization failed", exc_info=True)

            # L2 attempt
            cached = await self._l2_get(thread_id, checkpoint_ns)
            if cached is not None:
                try:
                    bundle = self._deserialize_bundle(cached)
                    tup = self._bundle_to_tuple(thread_id, checkpoint_ns, bundle)
                    await self._l1_put(thread_id, checkpoint_ns, cached)
                    return tup
                except Exception:
                    logger.warning("L2 deserialization failed", exc_info=True)

        # Proxy (database via executor)
        result = await self._proxy_get_tuple(thread_id, checkpoint_ns, checkpoint_id)
        return result

    async def _proxy_get_tuple(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str | None,
    ) -> CheckpointTuple | None:
        """load checkpoint from database via query executor.

        :param thread_id: conversation/thread identifier
        :ptype thread_id: str
        :param checkpoint_ns: checkpoint namespace
        :ptype checkpoint_ns: str
        :param checkpoint_id: specific checkpoint ID or None for latest
        :ptype checkpoint_id: str | None
        :return: checkpoint tuple or None
        :rtype: CheckpointTuple | None
        """
        if checkpoint_id:
            row = await self._exec.fetchrow(
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
            row = await self._exec.fetchrow(
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

        write_rows = await self._exec.fetch(
            "SELECT task_id, channel, type, blob "
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
            "configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns, "checkpoint_id": cp_id}
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

        # Warm caches
        try:
            cache_blob = self._serialize_bundle(checkpoint, metadata, parent_id, pending_writes)
            await self._l2_put(thread_id, checkpoint_ns, cache_blob)
            await self._l1_put(thread_id, checkpoint_ns, cache_blob)
        except Exception:
            logger.warning("failed to warm caches after proxy read", exc_info=True)

        return tup

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """store checkpoint via proxy executor.

        :param config: runnable config with thread_id
        :ptype config: RunnableConfig
        :param checkpoint: checkpoint state to store
        :ptype checkpoint: Checkpoint
        :param metadata: checkpoint metadata
        :ptype metadata: CheckpointMetadata
        :param new_versions: channel version updates
        :ptype new_versions: ChannelVersions
        :return: config with checkpoint_id set
        :rtype: RunnableConfig
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        serializable_metadata = get_checkpoint_metadata(config, metadata)
        cp_type, cp_blob = self.serde.dumps_typed(checkpoint)
        _md_type, md_blob = self.serde.dumps_typed(serializable_metadata)

        await self._exec.execute(
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

        # Warm caches
        try:
            cache_blob = self._serialize_bundle(checkpoint, serializable_metadata, parent_checkpoint_id, [])
            await self._l2_put(thread_id, checkpoint_ns, cache_blob)
            await self._l1_put(thread_id, checkpoint_ns, cache_blob)
        except Exception:
            logger.warning("failed to warm caches after proxy write", exc_info=True)

        return result_config

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """store intermediate writes via proxy executor.

        :param config: runnable config with thread_id and checkpoint_id
        :ptype config: RunnableConfig
        :param writes: list of (channel, value) tuples
        :ptype writes: Sequence[tuple[str, Any]]
        :param task_id: task identifier for crash recovery
        :ptype task_id: str
        :param task_path: optional task path
        :ptype task_path: str
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        for idx, (channel, value) in enumerate(writes):
            write_idx = WRITES_IDX_MAP.get(channel, idx)
            w_type, w_blob = self.serde.dumps_typed(value)

            await self._exec.execute(
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

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """list checkpoints for thread via proxy executor.

        :param config: runnable config with thread_id
        :ptype config: RunnableConfig | None
        :param filter: optional metadata filter
        :ptype filter: dict[str, Any] | None
        :param before: only return checkpoints before this config
        :ptype before: RunnableConfig | None
        :param limit: max number of checkpoints to return
        :ptype limit: int | None
        :return: async iterator of checkpoint tuples
        :rtype: AsyncIterator[CheckpointTuple]
        """
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

        rows = await self._exec.fetch(query, *params)

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

    async def adelete_thread(self, thread_id: str) -> None:
        """delete all checkpoints and writes for thread from all tiers.

        :param thread_id: thread identifier to delete
        :ptype thread_id: str
        """
        await self._exec.execute("DELETE FROM checkpoint_writes WHERE thread_id = $1", thread_id)
        await self._exec.execute("DELETE FROM checkpoints WHERE thread_id = $1", thread_id)

    # ------------------------------------------------------------------
    # Sync methods -- not supported
    # ------------------------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """not supported. use aget_tuple()."""
        raise NotImplementedError("ProxyCheckpointSaver is async-only. Use aget_tuple().")

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """not supported. use alist()."""
        raise NotImplementedError("ProxyCheckpointSaver is async-only. Use alist().")

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """not supported. use aput()."""
        raise NotImplementedError("ProxyCheckpointSaver is async-only. Use aput().")

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """not supported. use aput_writes()."""
        raise NotImplementedError("ProxyCheckpointSaver is async-only. Use aput_writes().")

    def delete_thread(self, thread_id: str) -> None:
        """not supported. use adelete_thread()."""
        raise NotImplementedError("ProxyCheckpointSaver is async-only. Use adelete_thread().")
