"""3tears agent-tools: tool registry, context management, and built-in tools."""

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

try:
    __version__ = _version("3tears-agent-tools")
except _PackageNotFoundError:  # pragma: no cover - dev fallback
    __version__ = "unknown"

from threetears.agent.tools.events import TodosChangedEvent
from threetears.agent.tools.registry import ToolRegistry
from threetears.agent.tools.context import ToolContextManager
from threetears.agent.tools.entities import ContextItemEntity
from threetears.agent.tools.collections import (
    ContextItemCollection,
    context_items_table,
    migrate_context_items_schema,
)
from threetears.agent.tools.document import (
    DocumentResult,
    DocumentSection,
    OcrConfig,
    ParseDocumentInput,
    create_parse_document_tool,
    detect_mime_from_filename,
    parse_document,
)
from threetears.agent.tools.protocols import (
    GeneratedImage,
    ImageGenerationBackend,
    MediaInfo,
    MediaStorage,
    TextProvider,
    TranscriptionProvider,
    VisionProvider,
)
from threetears.agent.tools.builtin.analyze_media import (
    AnalyzerConfig,
    create_analyze_media_tool,
)
from threetears.agent.tools.builtin.image_generation import (
    ImageGenerationContext,
    ImageGenerationInput,
    create_image_generation_tool,
)
from threetears.agent.tools.chunker import (
    ChunkResult,
    ChunkStrategy,
    chunk_by_headers,
    chunk_by_lines,
    chunk_by_sections,
    chunk_content,
    register_chunk_strategy,
)
from threetears.agent.tools.todo import (
    TodoStorage,
    load_todo_tools as load_todo_tools_from_storage,
)
from threetears.agent.tools.tool_llm_dispatch import (
    InvokeToolLlmInput,
    ToolLlmInvocation,
    ToolLlmResolver,
    load_tool_llm_dispatch,
)
from threetears.agent.tools.workflow import load_workflow_tools
from threetears.agent.tools.builtin import register_builtins
from threetears.agent.tools.router import (
    DEFAULT_ROUTING_PROMPT,
    ToolRouter,
    ToolRoutingDecision,
    is_recall_intent,
)
from threetears.agent.tools.executor import ToolExecutor, ToolExecutionResult
from threetears.agent.tools.mcp import McpClient, McpTool, McpToolResult
from threetears.agent.tools.types import ChatModelFactory

__all__ = [
    "AnalyzerConfig",
    "ChunkResult",
    "ChunkStrategy",
    "ChatModelFactory",
    "ContextItemCollection",
    "ContextItemEntity",
    "context_items_table",
    "migrate_context_items_schema",
    "DEFAULT_ROUTING_PROMPT",
    "DocumentResult",
    "DocumentSection",
    "GeneratedImage",
    "ImageGenerationBackend",
    "ImageGenerationContext",
    "ImageGenerationInput",
    "InvokeToolLlmInput",
    "McpClient",
    "McpTool",
    "McpToolResult",
    "MediaInfo",
    "MediaStorage",
    "OcrConfig",
    "TextProvider",
    "ParseDocumentInput",
    "ToolContextManager",
    "TodoStorage",
    "TodosChangedEvent",
    "ToolExecutionResult",
    "ToolExecutor",
    "ToolLlmInvocation",
    "ToolLlmResolver",
    "ToolRegistry",
    "ToolRouter",
    "ToolRoutingDecision",
    "TranscriptionProvider",
    "VisionProvider",
    "chunk_by_headers",
    "chunk_by_lines",
    "chunk_by_sections",
    "chunk_content",
    "create_analyze_media_tool",
    "create_image_generation_tool",
    "create_parse_document_tool",
    "detect_mime_from_filename",
    "is_recall_intent",
    "load_tool_llm_dispatch",
    "load_todo_tools_from_storage",
    "load_workflow_tools",
    "parse_document",
    "register_builtins",
    "register_chunk_strategy",
]
