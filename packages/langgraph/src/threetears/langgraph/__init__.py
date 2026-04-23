"""3tears-langgraph: LangGraph integration with three-tier persistence.

provides checkpoint savers, graph builders, and context management
for building LangGraph agents backed by 3tears infrastructure.
"""

__version__ = "0.5.0"

from threetears.langgraph.builders import build_chat_agent, build_tool_agent
from threetears.langgraph.checkpoint import ThreeTierCheckpointSaver
from threetears.langgraph.context_registry import ContextManagerRegistry, current_conversation_id
from threetears.langgraph.nodes import agent_node, has_tool_calls, tool_node
from threetears.langgraph.protocols import (
    AsyncQueryExecutor,
    CheckpointL1Cache,
    CheckpointL2Cache,
    FlushCallback,
)
from threetears.langgraph.proxy_checkpoint import ProxyCheckpointSaver
from threetears.langgraph.serde import UUIDSafeSerializer

__all__ = [
    "AsyncQueryExecutor",
    "CheckpointL1Cache",
    "CheckpointL2Cache",
    "ContextManagerRegistry",
    "FlushCallback",
    "ProxyCheckpointSaver",
    "ThreeTierCheckpointSaver",
    "UUIDSafeSerializer",
    "agent_node",
    "build_chat_agent",
    "build_tool_agent",
    "current_conversation_id",
    "has_tool_calls",
    "tool_node",
]
