"""pre-built LangGraph graph builders with 3tears integration.

provides zero-friction agent creation with checkpointing, context
memory, data store access, and tool calling pre-wired. agents compile
the returned StateGraph with their checkpointer and run via standard
LangGraph.

recognized config["configurable"] keys across all builders:
    - chat_model: BaseChatModel instance (required)
    - system_prompt: system prompt string (optional override)
    - tools: list of tool instances (optional override)
    - context_manager: ToolContextManager (optional)
    - data_store: DataStore or dict of BaseCollection instances (optional)
    - thread_id: conversation identifier for checkpoint persistence (optional)
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, MessagesState, StateGraph

from threetears.langgraph.nodes import agent_node, has_tool_calls, tool_node
from threetears.observe import get_logger

__all__ = [
    "build_chat_agent",
    "build_tool_agent",
]

log = get_logger(__name__)

_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
_DEFAULT_MAX_ITERATIONS = 10


def build_chat_agent(
    *,
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
) -> StateGraph[MessagesState, None, Any, Any]:
    """build minimal chat agent graph with no tools.

    creates graph with single flow: START -> agent -> END.

    the agent node prepends system_prompt if not already present,
    invokes the chat model, and returns the response. if a
    context_manager is in config, conversation context (variables,
    previous tool results) is injected into the system prompt.

    config["configurable"] recognized keys:
        - chat_model: BaseChatModel instance (required)
        - system_prompt: override prompt (optional)
        - context_manager: ToolContextManager (optional)
        - data_store: DataStore for three-tier entity access in custom nodes (optional)
        - thread_id: conversation ID for checkpoint persistence (optional)

    :param system_prompt: default system prompt prepended to messages
    :ptype system_prompt: str
    :return: uncompiled StateGraph ready for .compile()
    :rtype: StateGraph
    """
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)

    result = graph
    return result


def build_tool_agent(
    *,
    tools: list[Any] | None = None,
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> StateGraph[MessagesState, None, Any, Any]:
    """build tool-calling agent graph with iteration loop.

    creates graph with flow:
    START -> agent -> [has_tool_calls?] -> tools -> agent (loop)
                                        -> END

    the agent node binds tools to the model, invokes, and checks
    for tool calls. the tool node executes calls and returns results.
    loops until no more tool calls or max_iterations reached.
    if a context_manager is in config, conversation context is
    injected into the system prompt automatically.

    config["configurable"] recognized keys:
        - chat_model: BaseChatModel instance (required)
        - tools: list of tool instances (optional, overrides constructor tools)
        - system_prompt: override prompt (optional)
        - context_manager: ToolContextManager (optional)
        - data_store: DataStore for three-tier entity access in custom nodes (optional)
        - thread_id: conversation ID for checkpoint persistence (optional)

    :param tools: default tool instances for agent to use
    :ptype tools: list[Any] | None
    :param system_prompt: default system prompt prepended to messages
    :ptype system_prompt: str
    :param max_iterations: maximum number of agent-tool loop iterations
    :ptype max_iterations: int
    :return: uncompiled StateGraph ready for .compile()
    :rtype: StateGraph
    """
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        has_tool_calls,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "agent")

    result = graph
    return result
