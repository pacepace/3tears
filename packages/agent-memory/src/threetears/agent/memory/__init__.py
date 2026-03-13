"""Agent memory package -- memory entities, collections, and embedding protocol."""

__version__ = "0.1.0"

from threetears.agent.memory.collections import MemoriesCollection
from threetears.agent.memory.embedding import EmbeddingProvider
from threetears.agent.memory.entities import MemoryEntity
from threetears.agent.memory.extraction import ChatModelFactory, MemoryExtractor
from threetears.agent.memory.ledger import MemoryLedger
from threetears.agent.memory.prompts import ExtractionPrompts
from threetears.agent.memory.retrieval import MemoryRetriever, RetrievalResult
from threetears.agent.memory.tools import (
    MemorySearchInput,
    RecallMemoryInput,
    load_memory_search_tool,
    load_recall_memory_tool,
)
from threetears.agent.memory.types import MemoryConfig, MemoryType

__all__ = [
    "ChatModelFactory",
    "EmbeddingProvider",
    "ExtractionPrompts",
    "MemoriesCollection",
    "MemoryConfig",
    "MemoryEntity",
    "MemoryExtractor",
    "MemoryLedger",
    "MemoryRetriever",
    "MemorySearchInput",
    "MemoryType",
    "RecallMemoryInput",
    "RetrievalResult",
    "load_memory_search_tool",
    "load_recall_memory_tool",
]
