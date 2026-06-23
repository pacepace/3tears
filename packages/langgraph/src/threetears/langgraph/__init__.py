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

from threetears.langgraph.builders import build_chat_agent, build_tool_agent
from threetears.langgraph.caching import (
    ChatModelCapabilities,
    annotate_system_prompt,
    compute_tool_key,
    detect_capabilities,
    extract_cache_usage,
    should_bind_tools_fresh,
)
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
from threetears.langgraph.hooks import (
    AgentNodeHook,
    PromptCachingHook,
    ToolNodeHook,
    compose_agent_node_hooks,
    compose_tool_node_hooks,
    summarize_args,
)
from threetears.langgraph.middleware import PromptCachingMiddleware
from threetears.langgraph.nodes import agent_node, has_tool_calls, tool_node
from threetears.langgraph.protocols import (
    AsyncpgPoolAdapter,
    AsyncQueryExecutor,
    CheckpointL1Cache,
    CheckpointL2Cache,
    FlushCallback,
)
from threetears.langgraph.serde import UUIDSafeSerializer
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

__all__ = [
    "DEFAULT_SUMMARIZATION_PROMPT",
    "NOSTREAM_TAG",
    "AgentNodeHook",
    "AsyncQueryExecutor",
    "AsyncpgPoolAdapter",
    "ChatModelCapabilities",
    "CheckpointL1Cache",
    "CheckpointL2Cache",
    "FlushCallback",
    "FrameworkEvent",
    "FrameworkEventRegistry",
    "ImageGeneratedEvent",
    "PromptBuiltEvent",
    "PromptCachingHook",
    "PromptCachingMiddleware",
    "ReasoningStreamedEvent",
    "ResponseCompletedEvent",
    "ResponseFailedEvent",
    "StreamEndEvent",
    "StreamErrorEvent",
    "StreamEvent",
    "StreamStartEvent",
    "StreamTokenEvent",
    "StreamTransport",
    "StreamingResponse",
    "StreamingResponseError",
    "ThreeTierCheckpointSaver",
    "ToolCallEndEvent",
    "ToolCallProgressEvent",
    "ToolCallStartEvent",
    "ToolCompletedEvent",
    "ToolDispatchedEvent",
    "ToolNodeHook",
    "ToolStartedEvent",
    "UUIDSafeSerializer",
    "WorkflowCompletedEvent",
    "WorkflowStartedEvent",
    "WorkflowStepCompletedEvent",
    "agent_node",
    "annotate_system_prompt",
    "build_chat_agent",
    "build_tool_agent",
    "compose_agent_node_hooks",
    "compose_tool_node_hooks",
    "compute_tool_key",
    "default_registry",
    "detect_capabilities",
    "dispatch_event",
    "extract_cache_usage",
    "has_tool_calls",
    "parse_stream_event",
    "should_bind_tools_fresh",
    "summarize_args",
    "summarize_older_messages",
    "tool_node",
]
