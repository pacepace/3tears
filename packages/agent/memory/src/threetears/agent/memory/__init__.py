"""Agent memory package -- memory entities, collections, and embedding protocol."""

# Version derived from pyproject.toml so the metadata is the single
# source of truth -- a future release that bumps pyproject without
# updating ``__init__.py`` can't drift the runtime ``__version__``.
# The except guard handles the rare case where the package isn't
# installed via importlib.metadata (e.g. running directly from a
# checked-out source tree without ``uv sync``); the fallback keeps
# imports working but reports ``unknown`` rather than crashing.
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("3tears-agent-memory")
except _PackageNotFoundError:  # pragma: no cover - dev fallback
    __version__ = "unknown"

from threetears.agent.memory.access import MemoryAccessService
from threetears.agent.memory.events import (
    MemoryCreatedEvent,
    MemoryRetrievedEvent,
    default_memory_created_dispatcher,
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
    ChunkRecallInput,
    ChunkSearchInput,
    MemoryAddInput,
    MemoryRecallInput,
    MemorySearchInput,
    load_chunk_recall_tool,
    load_chunk_search_tool,
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
    "ChunkRecallInput",
    "ChunkSearchInput",
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
    "default_memory_created_dispatcher",
    "ensure_memory_owner_assignment",
    "load_chunk_recall_tool",
    "load_chunk_search_tool",
    "load_memory_add_tool",
    "load_memory_recall_tool",
    "load_memory_search_tool",
    "memory_namespace_name",
]
