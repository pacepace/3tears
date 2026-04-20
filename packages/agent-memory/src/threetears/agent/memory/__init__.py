"""Agent memory package -- memory entities, collections, and embedding protocol."""

__version__ = "0.5.0"

from threetears.agent.memory.authorize import (
    ACTION_MEMORY_EXTRACT,
    ACTION_MEMORY_READ,
    ACTION_MEMORY_WRITE,
    MEMORY_NAMESPACE_TYPE,
    MEMORY_OWNER_GROUP_PREFIX,
    MEMORY_OWNER_ROLE_NAME,
    MemoryAccessDenied,
    MemoryAuthorizerDependencies,
    MemoryNamespaceResolver,
    MemoryNamespaceRow,
    MemoryOwnerAssignmentEnsurer,
    authorize_memory_access,
    memory_namespace_name,
)
from threetears.agent.memory.collections import MemoriesCollection
from threetears.agent.memory.embedding import EmbeddingProvider
from threetears.agent.memory.entities import MemoryEntity
from threetears.agent.memory.extraction import ChatModelFactory, MemoryExtractor
from threetears.agent.memory.ledger import MemoryLedger
from threetears.agent.memory.prompts import ExtractionPrompts
from threetears.agent.memory.retrieval import MemoryRetriever, RetrievalResult
from threetears.agent.memory.tools import (
    AddMemoryInput,
    MemorySearchInput,
    RecallMemoryInput,
    load_add_memory_tool,
    load_memory_search_tool,
    load_recall_memory_tool,
)
from threetears.agent.memory.types import MemoryConfig, MemoryType

__all__ = [
    "ACTION_MEMORY_EXTRACT",
    "ACTION_MEMORY_READ",
    "ACTION_MEMORY_WRITE",
    "AddMemoryInput",
    "ChatModelFactory",
    "EmbeddingProvider",
    "ExtractionPrompts",
    "MEMORY_NAMESPACE_TYPE",
    "MEMORY_OWNER_GROUP_PREFIX",
    "MEMORY_OWNER_ROLE_NAME",
    "MemoriesCollection",
    "MemoryAccessDenied",
    "MemoryAuthorizerDependencies",
    "MemoryConfig",
    "MemoryEntity",
    "MemoryExtractor",
    "MemoryLedger",
    "MemoryNamespaceResolver",
    "MemoryNamespaceRow",
    "MemoryOwnerAssignmentEnsurer",
    "MemoryRetriever",
    "MemorySearchInput",
    "MemoryType",
    "RecallMemoryInput",
    "RetrievalResult",
    "authorize_memory_access",
    "load_add_memory_tool",
    "load_memory_search_tool",
    "load_recall_memory_tool",
    "memory_namespace_name",
]
