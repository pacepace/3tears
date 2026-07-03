"""memory integration for agent runtime.

wires the 3tears memory system (retrieval + extraction) into the agent
runtime. all operations soft-fail: log a warning and continue.

the memory-package Collections live on this integration object so the
framework memory-injection middleware
(:mod:`threetears.agent.memory.middleware`) + tool factories can be
wired without reaching back through the bootstrap. read OPAQUELY off
``config["configurable"]["memory_integration"]`` by the middleware --
the framework stays uncoupled from the host's memory wiring.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from langchain_core.embeddings import Embeddings
from threetears.observe import get_logger

__all__ = [
    "MemoryIntegration",
    "NatsEmbeddingAdapter",
    "extract_memories",
    "retrieve_memories",
]

log = get_logger(__name__)


class NatsEmbeddingAdapter(Embeddings):
    """thin :class:`Embeddings` wrapper around ``NatsEmbeddingModel``.

    3tears v0.6.0 retired the local ``EmbeddingProvider`` Protocol with
    ``embed_text(text) -> tuple[list[float] | None, int]`` in favour of
    LangChain's :class:`Embeddings` interface. ``NatsEmbeddingModel`` is
    already an :class:`Embeddings` subclass -- this adapter retains the
    historical injection point and exposes a stable ``dimensions``
    property the memory layer never required but downstream code reads.

    :param nats_embedding_model: NatsEmbeddingModel instance
    :ptype nats_embedding_model: Any
    :param vector_dimensions: embedding vector dimensionality
    :ptype vector_dimensions: int
    """

    def __init__(
        self,
        nats_embedding_model: Any,
        vector_dimensions: int = 1024,
    ) -> None:
        """initialize adapter wrapping NatsEmbeddingModel.

        :param nats_embedding_model: NatsEmbeddingModel instance
        :ptype nats_embedding_model: Any
        :param vector_dimensions: embedding vector dimensionality
        :ptype vector_dimensions: int
        :return: nothing
        :rtype: None
        """
        self._model = nats_embedding_model
        self._dimensions = vector_dimensions

    @property
    def dimensions(self) -> int:
        """return vector dimensions.

        :return: embedding vector dimensionality
        :rtype: int
        """
        return self._dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """delegates to ``NatsEmbeddingModel.embed_documents``.

        :param texts: list of texts to embed
        :ptype texts: list[str]
        :return: list of embedding vectors
        :rtype: list[list[float]]
        """
        result: list[list[float]] = self._model.embed_documents(texts)
        return result

    def embed_query(self, text: str) -> list[float]:
        """delegates to ``NatsEmbeddingModel.embed_query``.

        :param text: text to embed
        :ptype text: str
        :return: embedding vector
        :rtype: list[float]
        """
        result: list[float] = self._model.embed_query(text)
        return result

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """delegates to ``NatsEmbeddingModel.aembed_documents``.

        :param texts: list of texts to embed
        :ptype texts: list[str]
        :return: list of embedding vectors
        :rtype: list[list[float]]
        """
        result: list[list[float]] = await self._model.aembed_documents(texts)
        return result

    async def aembed_query(self, text: str) -> list[float]:
        """delegates to ``NatsEmbeddingModel.aembed_query``.

        :param text: text to embed
        :ptype text: str
        :return: embedding vector
        :rtype: list[float]
        """
        result: list[float] = await self._model.aembed_query(text)
        return result


class MemoryIntegration:
    """wires the 3tears memory system for agent use.

    holds references to the memory retriever, extractor, embedding
    model, and the five memory-package Collections. the framework
    memory-injection middleware + tool factories consume the
    Collections directly. the fifth collection is
    :class:`~threetears.agent.memory.collections.MemoryRefsCollection`
    for the composite-pk ``conversation_memory_refs`` surface.

    :ivar retriever: memory retriever for scoped search
    :ivar extractor: memory extractor for conversation analysis
    :ivar embedding_model: embedding model for vector operations
    :ivar memories_collection: three-tier :class:`MemoriesCollection`
    :ivar media_collection: three-tier :class:`MediaCollection`
    :ivar media_content_collection: three-tier
        :class:`MediaContentCollection`
    :ivar memory_chunk_collection: three-tier
        :class:`MemoryChunkCollection`
    :ivar memory_refs_collection: three-tier
        :class:`MemoryRefsCollection` on the composite-pk
        ``conversation_memory_refs`` table
    """

    def __init__(
        self,
        retriever: Any = None,
        extractor: Any = None,
        embedding_model: Any = None,
        memories_collection: Any = None,
        media_collection: Any = None,
        media_content_collection: Any = None,
        memory_chunk_collection: Any = None,
        memory_refs_collection: Any = None,
        access_service: Any = None,
    ) -> None:
        """initialize memory integration with optional components.

        :param retriever: memory retriever for scoped search
        :ptype retriever: Any
        :param extractor: memory extractor for conversation analysis
        :ptype extractor: Any
        :param embedding_model: embedding model for vector operations
        :ptype embedding_model: Any
        :param memories_collection: three-tier memories collection
        :ptype memories_collection: Any
        :param media_collection: three-tier media parent collection
        :ptype media_collection: Any
        :param media_content_collection: three-tier media_content
            collection
        :ptype media_content_collection: Any
        :param memory_chunk_collection: three-tier memory_chunks
            collection
        :ptype memory_chunk_collection: Any
        :param memory_refs_collection: three-tier
            ``conversation_memory_refs`` collection used by
            :class:`ToolContextManager` for surfaced-items tracking
        :ptype memory_refs_collection: Any
        :param access_service: optional
            :class:`~threetears.agent.memory.access.MemoryAccessService`
            for cross-agent retrieval. wires the cross-agent fan-out
            helper so callers can retrieve a user's memories across
            every authorized agent partition without rolling their own
            ACL composition
        :ptype access_service: Any
        :return: nothing
        :rtype: None
        """
        self.retriever = retriever
        self.extractor = extractor
        self.embedding_model = embedding_model
        self.memories_collection = memories_collection
        self.media_collection = media_collection
        self.media_content_collection = media_content_collection
        self.memory_chunk_collection = memory_chunk_collection
        self.memory_refs_collection = memory_refs_collection
        self.access_service = access_service

    async def retrieve_across_authorized_agents(
        self,
        *,
        user_id: UUID,
        caller_user_id: UUID,
        customer_id: UUID,
    ) -> list[Any]:
        """retrieve a user's memories across every agent partition the caller can read.

        thin wrapper over
        :meth:`MemoryAccessService.find_for_user_across_authorized_agents`.
        real consumer for the cross-agent ACL composition primitive;
        admin agents troubleshooting across customer agents and any
        future cross-agent memory pull runs through this helper.

        soft-fails when ``access_service`` is not wired (returns empty
        list) so a partial integration build does not crash the
        consumer.

        :param user_id: user whose memories to surface (row filter
            inside each authorized partition)
        :ptype user_id: UUID
        :param caller_user_id: invoking user UUID for the
            ``memory.read`` evaluator decision
        :ptype caller_user_id: UUID
        :param customer_id: owning customer; cross-customer namespaces
            are filtered out at the namespace enumeration step
        :ptype customer_id: UUID
        :return: list of memory entities across every authorized
            agent partition; empty when no agents resolve or when
            the access service is not wired
        :rtype: list[Any]
        """
        if self.access_service is None:
            log.debug(
                "memory cross-agent retrieval: skipping (no access service)",
            )
            return []
        try:
            result: list[Any] = await self.access_service.find_for_user_across_authorized_agents(
                user_id=user_id,
                caller_user_id=caller_user_id,
                customer_id=customer_id,
            )
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- cross-agent retrieval is best-effort; a fault yields an empty result rather than crashing the consumer
            log.warning(
                "memory cross-agent retrieval failed (soft-fail): %s",
                exc,
            )
            result = []
        return result


async def retrieve_memories(
    integration: MemoryIntegration,
    agent_id: UUID,
    customer_id: UUID,
    user_id: UUID,
    query: str,
    limit: int = 5,
) -> list[str]:
    """retrieve relevant memories for conversation context.

    delegates to :meth:`MemoryRetriever.retrieve` which returns a
    formatted context string. wraps the result in a list for uniform
    consumption. soft-fails on error -- returns an empty list, logs a
    warning.

    ``user_id`` flows through as BOTH the row filter and the
    ``caller_user_id`` for the rbac evaluator; ``agent_id`` flows
    as the owner (the conversation's own agent is the memory
    namespace owner) and as the ``caller_agent_id`` so the
    evaluator's owner short-circuit fires -- agent-internal
    retrieval never requires an explicit grant.

    :param integration: memory integration instance
    :ptype integration: MemoryIntegration
    :param agent_id: agent UUID for scoping
    :ptype agent_id: UUID
    :param customer_id: customer UUID for scoping
    :ptype customer_id: UUID
    :param user_id: user UUID for scoping
    :ptype user_id: UUID
    :param query: search query text
    :ptype query: str
    :param limit: maximum memories to return (unused, reserved for future)
    :ptype limit: int
    :return: list of memory content strings
    :rtype: list[str]
    """
    _ = limit
    if integration.retriever is None:
        return []

    try:
        context = await integration.retriever.retrieve(
            user_id,
            query,
            agent_id=agent_id,
            customer_id=customer_id,
            caller_user_id=user_id,
            caller_agent_id=agent_id,
        )
        memories: list[str] = [context] if context else []
    except Exception as exc:  # prawduct:allow prawduct/broad-except -- retrieval is best-effort context enrichment; a fault yields no memories rather than failing the turn
        log.warning("memory retrieval failed (soft-fail): %s", exc)
        memories = []

    return memories


async def extract_memories(
    integration: MemoryIntegration,
    agent_id: UUID,
    customer_id: UUID,
    user_id: UUID,
    conversation_id: UUID,
    message_id_source: UUID,
    user_message: str,
    assistant_response: str,
    turn_count: int,
) -> None:
    """extract memories from a conversation turn.

    delegates to :meth:`MemoryExtractor.extract` which is
    fire-and-forget safe. soft-fails on error, logs a warning.

    :param integration: memory integration instance
    :ptype integration: MemoryIntegration
    :param agent_id: agent UUID for scoping
    :ptype agent_id: UUID
    :param customer_id: customer UUID for scoping
    :ptype customer_id: UUID
    :param user_id: user UUID for scoping
    :ptype user_id: UUID
    :param conversation_id: conversation UUID for extraction context
    :ptype conversation_id: UUID
    :param message_id_source: source message UUID
    :ptype message_id_source: UUID
    :param user_message: raw user message text
    :ptype user_message: str
    :param assistant_response: raw assistant response text
    :ptype assistant_response: str
    :param turn_count: number of turns in conversation so far
    :ptype turn_count: int
    :return: nothing
    :rtype: None
    """
    if integration.extractor is None:
        return

    try:
        await integration.extractor.extract(
            user_id,
            conversation_id,
            message_id_source,
            user_message,
            assistant_response,
            turn_count,
            agent_id=agent_id,
            customer_id=customer_id,
        )
        log.debug("memory extraction complete for conversation %s", conversation_id)
    except Exception as exc:  # prawduct:allow prawduct/broad-except -- extraction is a fire-and-forget side-effect; a fault must not fail the turn that triggered it
        log.warning("memory extraction failed (soft-fail): %s", exc)
