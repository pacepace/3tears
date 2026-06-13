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
from typing import TYPE_CHECKING

try:
    __version__ = _version("3tears-agent-tools")
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
    from threetears.agent.tools.builtin import register_builtins
    from threetears.agent.tools.builtin.analyze_media import AnalyzerConfig, create_analyze_media_tool
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
    from threetears.agent.tools.collections import (
        ContextItemCollection,
        context_items_table,
        migrate_context_items_schema,
    )
    from threetears.agent.tools.context import ToolContextManager
    from threetears.agent.tools.document import (
        DocumentResult,
        DocumentSection,
        OcrConfig,
        ParseDocumentInput,
        create_parse_document_tool,
        detect_mime_from_filename,
        parse_document,
    )
    from threetears.agent.tools.entities import ContextItemEntity
    from threetears.agent.tools.events import TodosChangedEvent
    from threetears.agent.tools.executor import ToolExecutionResult, ToolExecutor
    from threetears.agent.tools.mcp import McpClient, McpTool, McpToolResult
    from threetears.agent.tools.protocols import (
        GeneratedImage,
        ImageGenerationBackend,
        MediaInfo,
        MediaStorage,
        TextProvider,
        TranscriptionProvider,
        VisionProvider,
    )
    from threetears.agent.tools.registry import ToolRegistry
    from threetears.agent.tools.router import DEFAULT_ROUTING_PROMPT, ToolRouter, ToolRoutingDecision, is_recall_intent
    from threetears.agent.tools.todo import TodoStorage, load_todo_tools as load_todo_tools_from_storage
    from threetears.agent.tools.tool_llm_dispatch import (
        InvokeToolLlmInput,
        ToolLlmInvocation,
        ToolLlmResolver,
        load_tool_llm_dispatch,
    )
    from threetears.agent.tools.types import ChatModelFactory
    from threetears.agent.tools.workflow import load_workflow_tools

# public attribute -> (defining module, attribute name in that module)
_LAZY: dict[str, tuple[str, str]] = {
    "AnalyzerConfig": ("threetears.agent.tools.builtin.analyze_media", "AnalyzerConfig"),
    "ChatModelFactory": ("threetears.agent.tools.types", "ChatModelFactory"),
    "ChunkResult": ("threetears.agent.tools.chunker", "ChunkResult"),
    "ChunkStrategy": ("threetears.agent.tools.chunker", "ChunkStrategy"),
    "ContextItemCollection": ("threetears.agent.tools.collections", "ContextItemCollection"),
    "ContextItemEntity": ("threetears.agent.tools.entities", "ContextItemEntity"),
    "DEFAULT_ROUTING_PROMPT": ("threetears.agent.tools.router", "DEFAULT_ROUTING_PROMPT"),
    "DocumentResult": ("threetears.agent.tools.document", "DocumentResult"),
    "DocumentSection": ("threetears.agent.tools.document", "DocumentSection"),
    "GeneratedImage": ("threetears.agent.tools.protocols", "GeneratedImage"),
    "ImageGenerationBackend": ("threetears.agent.tools.protocols", "ImageGenerationBackend"),
    "ImageGenerationContext": ("threetears.agent.tools.builtin.image_generation", "ImageGenerationContext"),
    "ImageGenerationInput": ("threetears.agent.tools.builtin.image_generation", "ImageGenerationInput"),
    "InvokeToolLlmInput": ("threetears.agent.tools.tool_llm_dispatch", "InvokeToolLlmInput"),
    "McpClient": ("threetears.agent.tools.mcp", "McpClient"),
    "McpTool": ("threetears.agent.tools.mcp", "McpTool"),
    "McpToolResult": ("threetears.agent.tools.mcp", "McpToolResult"),
    "MediaInfo": ("threetears.agent.tools.protocols", "MediaInfo"),
    "MediaStorage": ("threetears.agent.tools.protocols", "MediaStorage"),
    "OcrConfig": ("threetears.agent.tools.document", "OcrConfig"),
    "ParseDocumentInput": ("threetears.agent.tools.document", "ParseDocumentInput"),
    "TextProvider": ("threetears.agent.tools.protocols", "TextProvider"),
    "TodoStorage": ("threetears.agent.tools.todo", "TodoStorage"),
    "TodosChangedEvent": ("threetears.agent.tools.events", "TodosChangedEvent"),
    "ToolContextManager": ("threetears.agent.tools.context", "ToolContextManager"),
    "ToolExecutionResult": ("threetears.agent.tools.executor", "ToolExecutionResult"),
    "ToolExecutor": ("threetears.agent.tools.executor", "ToolExecutor"),
    "ToolLlmInvocation": ("threetears.agent.tools.tool_llm_dispatch", "ToolLlmInvocation"),
    "ToolLlmResolver": ("threetears.agent.tools.tool_llm_dispatch", "ToolLlmResolver"),
    "ToolRegistry": ("threetears.agent.tools.registry", "ToolRegistry"),
    "ToolRouter": ("threetears.agent.tools.router", "ToolRouter"),
    "ToolRoutingDecision": ("threetears.agent.tools.router", "ToolRoutingDecision"),
    "TranscriptionProvider": ("threetears.agent.tools.protocols", "TranscriptionProvider"),
    "VisionProvider": ("threetears.agent.tools.protocols", "VisionProvider"),
    "chunk_by_headers": ("threetears.agent.tools.chunker", "chunk_by_headers"),
    "chunk_by_lines": ("threetears.agent.tools.chunker", "chunk_by_lines"),
    "chunk_by_sections": ("threetears.agent.tools.chunker", "chunk_by_sections"),
    "chunk_content": ("threetears.agent.tools.chunker", "chunk_content"),
    "context_items_table": ("threetears.agent.tools.collections", "context_items_table"),
    "create_analyze_media_tool": ("threetears.agent.tools.builtin.analyze_media", "create_analyze_media_tool"),
    "create_image_generation_tool": ("threetears.agent.tools.builtin.image_generation", "create_image_generation_tool"),
    "create_parse_document_tool": ("threetears.agent.tools.document", "create_parse_document_tool"),
    "detect_mime_from_filename": ("threetears.agent.tools.document", "detect_mime_from_filename"),
    "is_recall_intent": ("threetears.agent.tools.router", "is_recall_intent"),
    "load_todo_tools_from_storage": ("threetears.agent.tools.todo", "load_todo_tools"),
    "load_tool_llm_dispatch": ("threetears.agent.tools.tool_llm_dispatch", "load_tool_llm_dispatch"),
    "load_workflow_tools": ("threetears.agent.tools.workflow", "load_workflow_tools"),
    "migrate_context_items_schema": ("threetears.agent.tools.collections", "migrate_context_items_schema"),
    "parse_document": ("threetears.agent.tools.document", "parse_document"),
    "register_builtins": ("threetears.agent.tools.builtin", "register_builtins"),
    "register_chunk_strategy": ("threetears.agent.tools.chunker", "register_chunk_strategy"),
}

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
