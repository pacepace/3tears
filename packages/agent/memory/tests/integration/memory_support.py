"""What the memory integration tests share: vectors, stub models, a pool, a Dream service, a source row.

Public names, so a test module reaches them here and never through another
test module's private ones.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import asyncpg

from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.collections import MemoriesCollection, MemoryConsolidationsCollection
from threetears.agent.memory.dream import DreamService
from threetears.agent.memory.types import MemoryConfig
from threetears.core.collections import init_connection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig


DIM = 1024


def vec(seed: float) -> list[float]:
    """a constant 1024-dim vector; equal seeds -> cosine 1.0 (cluster)."""
    return [seed] * DIM


def vec_sql(seed: float) -> str:
    return "[" + ",".join([str(seed)] * DIM) + "]"


class StubEmbeddings:
    """embedder returning a fixed gist vector (distinct from source vectors)."""

    def __init__(self, gist_vector: list[float]) -> None:
        self._gist = gist_vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._gist for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        _ = text
        return self._gist

    async def aembed_query(self, text: str) -> list[float]:
        _ = text
        return self._gist

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._gist for _ in texts]


class StubChatModel:
    def __init__(self, content: str) -> None:
        self._content = content

    async def ainvoke(self, messages: list[Any], **kwargs: Any) -> Any:
        _ = kwargs
        resp = MagicMock()
        resp.content = self._content
        return resp


class StubReflectorFactory:
    def __init__(self, content: str) -> None:
        self._content = content

    async def create_chat_model(self, purpose: str = "consolidation") -> Any:
        _ = purpose
        return StubChatModel(self._content)


async def make_pool(url: str, schema: str) -> asyncpg.Pool:
    pool: asyncpg.Pool = await asyncpg.create_pool(
        dsn=url,
        min_size=1,
        max_size=4,
        server_settings={"search_path": f"{schema}, public"},
        init=init_connection,
    )
    return pool


def make_service(
    pool: asyncpg.Pool,
    *,
    gist_vector: list[float],
    reflector_content: str = '{"gist": "merged gist", "rationale": "near-duplicates"}',
) -> DreamService:
    """wire a DreamService over real collections + stubbed embed/reflect."""
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    authorizer = MagicMock(spec=MemoryAuthorizerDependencies)
    memories = MemoriesCollection(registry=registry, config=config, authorizer=authorizer)
    edges = MemoryConsolidationsCollection(registry=registry, config=config, nats_client=None)
    return DreamService(
        config=MemoryConfig(),
        embedding_provider=StubEmbeddings(gist_vector),
        chat_model_factory=StubReflectorFactory(reflector_content),
        memories_collection=memories,
        consolidations_collection=edges,
    )


async def insert_source(
    conn: asyncpg.Connection,
    *,
    agent_id: uuid.UUID,
    customer_id: uuid.UUID | None,
    user_id: uuid.UUID | None,
    seed: float,
    content: str,
    conversation_id: uuid.UUID | None = None,
    age_days: int = 0,
    superseded_by: uuid.UUID | None = None,
    type_memory: str = "fact",
) -> tuple[uuid.UUID, uuid.UUID]:
    """insert one embedded source memory; return (memory_id, conversation_id)."""
    memory_id = uuid.uuid4()
    conv_id = conversation_id or uuid.uuid4()
    created = datetime.now(UTC) - timedelta(days=age_days)
    await conn.execute(
        "INSERT INTO memories ("
        "memory_id, agent_id, customer_id, user_id, conversation_id, "
        "type_memory, content, embedding, salience, superseded_by, "
        "date_created, date_updated"
        ") VALUES ($1,$2,$3,$4,$5,$6,$7,$8::text::public.vector,0.5,$9,$10,$10)",
        memory_id,
        agent_id,
        customer_id,
        user_id,
        conv_id,
        type_memory,
        content,
        vec_sql(seed),
        superseded_by,
        created,
    )
    return memory_id, conv_id


class PurposeChatModel:
    """chat model stub returning a preconfigured content payload."""

    def __init__(self, content: str) -> None:
        """
        :param content: text to return as ``response.content``
        :ptype content: str
        """
        self._content = content

    async def ainvoke(self, messages: list[Any], **kwargs: Any) -> Any:
        """return a MagicMock with ``content`` set to the preconfigured payload.

        Accepts and ignores the gateway identity kwargs (``user_id`` /
        ``conversation_id``) that ``_invoke_identity_kwargs`` threads onto
        the invoke call -- a real ``GatewayChatModel`` consumes them; the
        stub just tolerates them.
        """
        resp = MagicMock()
        resp.content = self._content
        return resp


class PurposeChatModelFactory:
    """factory returning per-purpose stub chat models."""

    def __init__(
        self,
        worthiness: str,
        extraction: str,
        resolution: str | None = None,
    ) -> None:
        """
        :param worthiness: JSON content for the worthiness check
        :ptype worthiness: str
        :param extraction: JSON content for the extraction list
        :ptype extraction: str
        :param resolution: JSON content for the resolution step (optional)
        :ptype resolution: str | None
        """
        self._by_purpose = {
            "worthiness": PurposeChatModel(worthiness),
            "extraction": PurposeChatModel(extraction),
            "resolution": PurposeChatModel(resolution or "[]"),
        }

    async def create_chat_model(self, purpose: str = "extraction") -> Any:
        """
        return the stub for the given purpose or a default empty-list stub.

        :param purpose: "worthiness" | "extraction" | "resolution"
        :ptype purpose: str
        :return: stub chat model
        :rtype: Any
        """
        result = self._by_purpose.get(purpose, PurposeChatModel("[]"))
        return result
