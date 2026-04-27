"""3tears-langgraph: LangGraph integration with three-tier persistence.

provides checkpoint savers, graph builders, and context management
for building LangGraph agents backed by 3tears infrastructure.
"""

__version__ = "0.5.0"

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
from threetears.langgraph.hooks import (
    AgentNodeHook,
    PromptCachingHook,
    ToolNodeHook,
    compose_agent_node_hooks,
    compose_tool_node_hooks,
    summarize_args,
)
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

__all__ = [
    "AgentNodeHook",
    "AsyncQueryExecutor",
    "AsyncpgPoolAdapter",
    "ChatModelCapabilities",
    "CheckpointL1Cache",
    "CheckpointL2Cache",
    "FlushCallback",
    "PromptCachingHook",
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
    "ToolNodeHook",
    "UUIDSafeSerializer",
    "agent_node",
    "annotate_system_prompt",
    "build_chat_agent",
    "build_tool_agent",
    "compose_agent_node_hooks",
    "compose_tool_node_hooks",
    "compute_tool_key",
    "detect_capabilities",
    "extract_cache_usage",
    "has_tool_calls",
    "parse_stream_event",
    "should_bind_tools_fresh",
    "summarize_args",
    "tool_node",
]
