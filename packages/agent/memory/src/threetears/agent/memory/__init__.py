"""Agent memory package -- memory entities, collections, and embedding protocol."""

__version__ = "0.7.0"

from threetears.agent.memory.access import MemoryAccessService
from threetears.agent.memory.events import (
    MemoryCreatedEvent,
    MemoryRetrievedEvent,
)
from threetears.agent.memory.authorize import (
    ACTION_MEMORY_EXTRACT,
    ACTION_MEMORY_READ,
    ACTION_MEMORY_WRITE,
    MEMORY_NAMESPACE_TYPE,
    MEMORY_OWNER_GROUP_PREFIX,
    MEMORY_OWNER_ROLE_NAME,
    MemoryAccessDenied,
    MemoryAuthorizerDependencies,
    authorize_memory_access,
    ensure_memory_owner_assignment,
    memory_namespace_name,
)
from threetears.agent.memory.collections import (
    MediaCollection,
    MediaContentCollection,
    MemoriesCollection,
    MemoryChunkCollection,
    MemoryRefsCollection,
    conversation_memory_refs_table,
)
from threetears.agent.memory.entities import (
    MediaContentEntity,
    MediaEntity,
    MemoryChunkEntity,
    MemoryEntity,
    MemoryRefEntity,
)
from threetears.agent.memory.extraction import ChatModelFactory, MemoryExtractor
from threetears.agent.memory.prompts import ExtractionPrompts
from threetears.agent.memory.retrieval import MemoryRetriever, RetrievalResult
from threetears.agent.memory.tools import (
    MemoryAddInput,
    MemoryRecallInput,
    MemorySearchInput,
    load_memory_add_tool,
    load_memory_recall_tool,
    load_memory_search_tool,
)
from threetears.agent.memory.types import MemoryConfig, MemoryType

__all__ = [
    "ACTION_MEMORY_EXTRACT",
    "ACTION_MEMORY_READ",
    "ACTION_MEMORY_WRITE",
    "ChatModelFactory",
    "ExtractionPrompts",
    "MEMORY_NAMESPACE_TYPE",
    "MEMORY_OWNER_GROUP_PREFIX",
    "MEMORY_OWNER_ROLE_NAME",
    "MediaCollection",
    "MediaContentCollection",
    "MediaContentEntity",
    "MediaEntity",
    "MemoriesCollection",
    "MemoryAccessDenied",
    "MemoryAccessService",
    "MemoryAddInput",
    "MemoryAuthorizerDependencies",
    "MemoryChunkCollection",
    "MemoryChunkEntity",
    "MemoryConfig",
    "MemoryCreatedEvent",
    "MemoryEntity",
    "MemoryExtractor",
    "MemoryRecallInput",
    "MemoryRefEntity",
    "MemoryRefsCollection",
    "MemoryRetrievedEvent",
    "MemoryRetriever",
    "MemorySearchInput",
    "MemoryType",
    "RetrievalResult",
    "authorize_memory_access",
    "conversation_memory_refs_table",
    "ensure_memory_owner_assignment",
    "load_memory_add_tool",
    "load_memory_recall_tool",
    "load_memory_search_tool",
    "memory_namespace_name",
]
