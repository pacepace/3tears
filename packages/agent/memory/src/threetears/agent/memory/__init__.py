"""Agent memory package -- memory entities, collections, and embedding protocol."""

from __future__ import annotations

# Version derived from pyproject.toml so the metadata is the single
# source of truth -- a future release that bumps pyproject without
# updating ``__init__.py`` can't drift the runtime ``__version__``.
# The except guard handles the rare case where the package isn't
# installed via importlib.metadata (e.g. running directly from a
# checked-out source tree without ``uv sync``); the fallback keeps
# imports working but reports ``unknown`` rather than crashing.
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version
from typing import TYPE_CHECKING

try:
    __version__ = _version("3tears-agent-memory")
except _PackageNotFoundError:  # pragma: no cover - dev fallback
    __version__ = "unknown"

# lazy public API (PEP 562). the package namespace no longer imports its
# implementation modules eagerly: importing this package (or any of its
# submodules) costs only this file, and each public attribute resolves
# its defining module on first access. the TYPE_CHECKING block carries
# the real imports so mypy and IDEs see the full statically-typed API;
# the _LAZY map is the runtime equivalent. the three-way agreement
# between __all__, _LAZY, and the TYPE_CHECKING block is pinned by the
# package's lazy-surface consistency test.
# decision record: docs/separate-concerns-decisions.md (hand-rolled
# PEP 562 over lazy_loader -- zero added runtime deps, no stub drift).
if TYPE_CHECKING:
    from threetears.agent.memory.access import MemoryAccessService
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
        ConsolidationCycleError,
        MediaCollection,
        MediaContentCollection,
        MemoriesCollection,
        MemoryChunkCollection,
        MemoryConsolidationsCollection,
        MemoryRefsCollection,
        conversation_memory_refs_table,
        memory_consolidations_table,
    )
    from threetears.agent.memory.dream import (
        ConsolidationResult,
        DreamService,
        ReflectorChatModelFactory,
    )
    from threetears.agent.memory.embedding_utils import embedding_attribution_scope
    from threetears.agent.memory.entities import (
        MediaContentEntity,
        MediaEntity,
        MemoryChunkEntity,
        MemoryConsolidationEntity,
        MemoryEntity,
        MemoryRefEntity,
    )
    from threetears.agent.memory.events import (
        MemoryConsolidatedEvent,
        MemoryCreatedEvent,
        MemoryRetrievedEvent,
        default_memory_created_dispatcher,
    )
    from threetears.agent.memory.extraction import ChatModelFactory, MemoryExtractor
    from threetears.agent.memory.integration import (
        MemoryIntegration,
        NatsEmbeddingAdapter,
        extract_memories,
        retrieve_memories,
    )
    from threetears.agent.memory.merge import MemoryRepointResult, repoint_user
    from threetears.agent.memory.middleware import MemoryInjectionMiddleware
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

# public attribute -> (defining module, attribute name in that module)
_LAZY: dict[str, tuple[str, str]] = {
    "ACTION_MEMORY_EXTRACT": ("threetears.agent.memory.authorize", "ACTION_MEMORY_EXTRACT"),
    "ACTION_MEMORY_READ": ("threetears.agent.memory.authorize", "ACTION_MEMORY_READ"),
    "ACTION_MEMORY_WRITE": ("threetears.agent.memory.authorize", "ACTION_MEMORY_WRITE"),
    "ChatModelFactory": ("threetears.agent.memory.extraction", "ChatModelFactory"),
    "ChunkRecallInput": ("threetears.agent.memory.tools", "ChunkRecallInput"),
    "ConsolidationCycleError": ("threetears.agent.memory.collections", "ConsolidationCycleError"),
    "ChunkSearchInput": ("threetears.agent.memory.tools", "ChunkSearchInput"),
    "ConsolidationResult": ("threetears.agent.memory.dream", "ConsolidationResult"),
    "DreamService": ("threetears.agent.memory.dream", "DreamService"),
    "ExtractionPrompts": ("threetears.agent.memory.prompts", "ExtractionPrompts"),
    "MEMORY_NAMESPACE_TYPE": ("threetears.agent.memory.authorize", "MEMORY_NAMESPACE_TYPE"),
    "MEMORY_OWNER_GROUP_PREFIX": ("threetears.agent.memory.authorize", "MEMORY_OWNER_GROUP_PREFIX"),
    "MEMORY_OWNER_ROLE_NAME": ("threetears.agent.memory.authorize", "MEMORY_OWNER_ROLE_NAME"),
    "MediaCollection": ("threetears.agent.memory.collections", "MediaCollection"),
    "MediaContentCollection": ("threetears.agent.memory.collections", "MediaContentCollection"),
    "MediaContentEntity": ("threetears.agent.memory.entities", "MediaContentEntity"),
    "MediaEntity": ("threetears.agent.memory.entities", "MediaEntity"),
    "MemoriesCollection": ("threetears.agent.memory.collections", "MemoriesCollection"),
    "MemoryAccessDenied": ("threetears.agent.memory.authorize", "MemoryAccessDenied"),
    "MemoryAccessService": ("threetears.agent.memory.access", "MemoryAccessService"),
    "MemoryAddInput": ("threetears.agent.memory.tools", "MemoryAddInput"),
    "MemoryAuthorizerDependencies": ("threetears.agent.memory.authorize", "MemoryAuthorizerDependencies"),
    "MemoryChunkCollection": ("threetears.agent.memory.collections", "MemoryChunkCollection"),
    "MemoryChunkEntity": ("threetears.agent.memory.entities", "MemoryChunkEntity"),
    "MemoryConfig": ("threetears.agent.memory.types", "MemoryConfig"),
    "MemoryConsolidationEntity": ("threetears.agent.memory.entities", "MemoryConsolidationEntity"),
    "MemoryConsolidationsCollection": ("threetears.agent.memory.collections", "MemoryConsolidationsCollection"),
    "MemoryConsolidatedEvent": ("threetears.agent.memory.events", "MemoryConsolidatedEvent"),
    "MemoryCreatedEvent": ("threetears.agent.memory.events", "MemoryCreatedEvent"),
    "MemoryEntity": ("threetears.agent.memory.entities", "MemoryEntity"),
    "MemoryExtractor": ("threetears.agent.memory.extraction", "MemoryExtractor"),
    "MemoryInjectionMiddleware": ("threetears.agent.memory.middleware", "MemoryInjectionMiddleware"),
    "MemoryIntegration": ("threetears.agent.memory.integration", "MemoryIntegration"),
    "MemoryRecallInput": ("threetears.agent.memory.tools", "MemoryRecallInput"),
    "MemoryRefEntity": ("threetears.agent.memory.entities", "MemoryRefEntity"),
    "MemoryRefsCollection": ("threetears.agent.memory.collections", "MemoryRefsCollection"),
    "MemoryRetrievedEvent": ("threetears.agent.memory.events", "MemoryRetrievedEvent"),
    "MemoryRepointResult": ("threetears.agent.memory.merge", "MemoryRepointResult"),
    "MemoryRetriever": ("threetears.agent.memory.retrieval", "MemoryRetriever"),
    "MemorySearchInput": ("threetears.agent.memory.tools", "MemorySearchInput"),
    "MemoryType": ("threetears.agent.memory.types", "MemoryType"),
    "NatsEmbeddingAdapter": ("threetears.agent.memory.integration", "NatsEmbeddingAdapter"),
    "ReflectorChatModelFactory": ("threetears.agent.memory.dream", "ReflectorChatModelFactory"),
    "RetrievalResult": ("threetears.agent.memory.retrieval", "RetrievalResult"),
    "authorize_memory_access": ("threetears.agent.memory.authorize", "authorize_memory_access"),
    "conversation_memory_refs_table": ("threetears.agent.memory.collections", "conversation_memory_refs_table"),
    "default_memory_created_dispatcher": ("threetears.agent.memory.events", "default_memory_created_dispatcher"),
    "memory_consolidations_table": ("threetears.agent.memory.collections", "memory_consolidations_table"),
    "embedding_attribution_scope": ("threetears.agent.memory.embedding_utils", "embedding_attribution_scope"),
    "ensure_memory_owner_assignment": ("threetears.agent.memory.authorize", "ensure_memory_owner_assignment"),
    "extract_memories": ("threetears.agent.memory.integration", "extract_memories"),
    "load_chunk_recall_tool": ("threetears.agent.memory.tools", "load_chunk_recall_tool"),
    "load_chunk_search_tool": ("threetears.agent.memory.tools", "load_chunk_search_tool"),
    "load_memory_add_tool": ("threetears.agent.memory.tools", "load_memory_add_tool"),
    "load_memory_recall_tool": ("threetears.agent.memory.tools", "load_memory_recall_tool"),
    "load_memory_search_tool": ("threetears.agent.memory.tools", "load_memory_search_tool"),
    "memory_namespace_name": ("threetears.agent.memory.authorize", "memory_namespace_name"),
    "repoint_user": ("threetears.agent.memory.merge", "repoint_user"),
    "retrieve_memories": ("threetears.agent.memory.integration", "retrieve_memories"),
}

__all__ = [
    "ACTION_MEMORY_EXTRACT",
    "ACTION_MEMORY_READ",
    "ACTION_MEMORY_WRITE",
    "ChatModelFactory",
    "ChunkRecallInput",
    "ChunkSearchInput",
    "ConsolidationCycleError",
    "ConsolidationResult",
    "DreamService",
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
    "MemoryConsolidationEntity",
    "MemoryConsolidationsCollection",
    "MemoryConsolidatedEvent",
    "MemoryCreatedEvent",
    "MemoryEntity",
    "MemoryExtractor",
    "MemoryInjectionMiddleware",
    "MemoryIntegration",
    "MemoryRecallInput",
    "MemoryRefEntity",
    "MemoryRefsCollection",
    "MemoryRepointResult",
    "MemoryRetrievedEvent",
    "MemoryRetriever",
    "MemorySearchInput",
    "MemoryType",
    "NatsEmbeddingAdapter",
    "ReflectorChatModelFactory",
    "RetrievalResult",
    "authorize_memory_access",
    "conversation_memory_refs_table",
    "default_memory_created_dispatcher",
    "embedding_attribution_scope",
    "ensure_memory_owner_assignment",
    "extract_memories",
    "load_chunk_recall_tool",
    "load_chunk_search_tool",
    "load_memory_add_tool",
    "load_memory_recall_tool",
    "load_memory_search_tool",
    "memory_consolidations_table",
    "memory_namespace_name",
    "repoint_user",
    "retrieve_memories",
]


def __getattr__(name: str) -> object:
    """resolve a public attribute from its defining module on first access.

    :param name: attribute name being resolved
    :ptype name: str
    :return: the resolved attribute (also cached in module globals so
        ``__getattr__`` fires at most once per name)
    :rtype: object
    :raises AttributeError: when ``name`` is not part of the public API
    """
    entry = _LAZY.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module_name, attr = entry
    value: object = getattr(import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """include lazy attributes in ``dir()`` output.

    :return: sorted union of materialized globals and lazy names
    :rtype: list[str]
    """
    return sorted(set(globals()) | set(_LAZY))
