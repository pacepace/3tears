"""3tears-langgraph: LangGraph integration with three-tier persistence.

provides checkpoint savers, graph builders, and context management
for building LangGraph agents backed by 3tears infrastructure.
"""

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
    __version__ = _version("3tears-langgraph")
except _PackageNotFoundError:  # pragma: no cover - dev fallback
    __version__ = "unknown"

from threetears.langgraph.caching import (
    ChatModelCapabilities,
    annotate_system_prompt,
    compute_tool_key,
    detect_capabilities,
    extract_cache_usage,
    should_bind_tools_fresh,
)
from threetears.langgraph.catalog import ObjectCataloger
from threetears.langgraph.checkpoint import ThreeTierCheckpointSaver
from threetears.langgraph.events import (
    FrameworkEvent,
    FrameworkEventRegistry,
    ImageGeneratedEvent,
    PromptBuiltEvent,
    ReasoningStreamedEvent,
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ToolCompletedEvent,
    ToolDispatchedEvent,
    ToolStartedEvent,
    WorkflowCompletedEvent,
    WorkflowStartedEvent,
    WorkflowStepCompletedEvent,
    default_registry,
    dispatch_event,
)
from threetears.langgraph.middleware import PromptCachingMiddleware
from threetears.langgraph.middleware_catalog import ObjectCatalogMiddleware
from threetears.langgraph.middleware_context import (
    ContextMergeMiddleware,
    ConversationContextProvider,
)
from threetears.langgraph.middleware_offload import ToolResultOffloadMiddleware
from threetears.langgraph.middleware_schema import SchemaPrimingMiddleware
from threetears.langgraph.middleware_summarize import SummarizationMiddleware
from threetears.langgraph.offload import (
    DEFAULT_OFFLOAD_THRESHOLD_CHARS,
    NEVER_OFFLOAD_TOOLS,
    OffloadResult,
    ToolResultOffloader,
    format_offload_handle,
    has_offload_handle,
    is_never_offload_tool,
)
from threetears.langgraph.protocols import (
    AsyncpgPoolAdapter,
    AsyncQueryExecutor,
    CheckpointL1Cache,
    CheckpointL2Cache,
    FlushCallback,
)
from threetears.langgraph.serde import UUIDSafeSerializer
from threetears.langgraph.state import merge_metadata
from threetears.langgraph.streaming import (
    NOSTREAM_TAG,
    StreamEndEvent,
    StreamErrorEvent,
    StreamEvent,
    StreamingResponse,
    StreamingResponseError,
    StreamStartEvent,
    StreamTokenEvent,
    StreamTransport,
    ToolCallEndEvent,
    ToolCallProgressEvent,
    ToolCallStartEvent,
    parse_stream_event,
)
from threetears.langgraph.summarize import (
    DEFAULT_SUMMARIZATION_PROMPT,
    summarize_older_messages,
)
from threetears.langgraph.util import summarize_args

__all__ = [
    "DEFAULT_OFFLOAD_THRESHOLD_CHARS",
    "DEFAULT_SUMMARIZATION_PROMPT",
    "NOSTREAM_TAG",
    "AsyncQueryExecutor",
    "AsyncpgPoolAdapter",
    "ChatModelCapabilities",
    "CheckpointL1Cache",
    "CheckpointL2Cache",
    "ContextMergeMiddleware",
    "ConversationContextProvider",
    "FlushCallback",
    "FrameworkEvent",
    "FrameworkEventRegistry",
    "ImageGeneratedEvent",
    "NEVER_OFFLOAD_TOOLS",
    "ObjectCataloger",
    "ObjectCatalogMiddleware",
    "OffloadResult",
    "PromptBuiltEvent",
    "PromptCachingMiddleware",
    "ReasoningStreamedEvent",
    "ResponseCompletedEvent",
    "ResponseFailedEvent",
    "SchemaPrimingMiddleware",
    "StreamEndEvent",
    "StreamErrorEvent",
    "StreamEvent",
    "StreamStartEvent",
    "StreamTokenEvent",
    "StreamTransport",
    "StreamingResponse",
    "StreamingResponseError",
    "SummarizationMiddleware",
    "ThreeTierCheckpointSaver",
    "ToolCallEndEvent",
    "ToolCallProgressEvent",
    "ToolCallStartEvent",
    "ToolCompletedEvent",
    "ToolDispatchedEvent",
    "ToolResultOffloadMiddleware",
    "ToolResultOffloader",
    "ToolStartedEvent",
    "UUIDSafeSerializer",
    "WorkflowCompletedEvent",
    "WorkflowStartedEvent",
    "WorkflowStepCompletedEvent",
    "annotate_system_prompt",
    "compute_tool_key",
    "default_registry",
    "detect_capabilities",
    "dispatch_event",
    "extract_cache_usage",
    "format_offload_handle",
    "has_offload_handle",
    "is_never_offload_tool",
    "merge_metadata",
    "parse_stream_event",
    "should_bind_tools_fresh",
    "summarize_args",
    "summarize_older_messages",
]
