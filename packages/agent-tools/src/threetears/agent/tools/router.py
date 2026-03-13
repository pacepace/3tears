"""Tool routing: recall-intent detection and LLM-based tool selection."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from threetears.core.logging import get_logger
from threetears.core.tracing import traced

from threetears.agent.tools.types import ChatModelFactory

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Recall-intent detection
# ---------------------------------------------------------------------------

_RECALL_VERBS = re.compile(
    r"\b(?:recall|show|display|see|view|retrieve|get|read|print|give me|look at|check)\b",
    re.IGNORECASE,
)
_RECALL_OBJECTS = re.compile(
    r"\b(?:results?|outputs?|response|responded|said|gave|returned|wrote|"
    r"produced|generated|came back|previous|earlier|last|prior|exact)\b",
    re.IGNORECASE,
)
_NEW_TASK_VERBS = re.compile(
    r"\b(?:create|write|generate|build|make|develop|implement|code|design|"
    r"analyze|search|find|look up|calculate|compute|solve|translate|convert|"
    r"summarize|explain|describe|review|refactor|fix|debug|test|benchmark)\b",
    re.IGNORECASE,
)


def is_recall_intent(message: str) -> bool:
    """Detect whether a message is asking to recall previous tool output."""
    has_recall_verb = bool(_RECALL_VERBS.search(message))
    has_recall_object = bool(_RECALL_OBJECTS.search(message))
    has_new_task = bool(_NEW_TASK_VERBS.search(message))
    return has_recall_verb and has_recall_object and not has_new_task


# ---------------------------------------------------------------------------
# Routing prompt
# ---------------------------------------------------------------------------

DEFAULT_ROUTING_PROMPT = """\
You are a tool routing assistant. Given a user message and a list of available \
tools, decide whether any tool should be invoked.

Available tools:
{tool_descriptions}

Analyze the user message and respond with ONLY valid JSON (no markdown, no \
explanation). Use one of these two formats:

If a tool should be used:
{{"tool_name": "<exact tool name>", "reasoning": "<brief reason>"}}

If no tool is needed:
{{"tool_name": null, "reasoning": "<brief reason>"}}

Rules:
- Only select a tool if the user's message clearly relates to that tool's purpose.
- If the message is general conversation, select no tool.
- Do NOT select a tool when the user is asking about, referencing, or recalling \
the output from a previous tool invocation.
- tool_name must exactly match one of the available tool names, or be null."""


# ---------------------------------------------------------------------------
# Decision dataclass
# ---------------------------------------------------------------------------


@dataclass
class ToolRoutingDecision:
    """Result of tool routing."""

    tool_name: str | None
    tool_type: str | None  # "tool_llm", "mcp", or None
    tool_id: str | None  # tool_llm_id or mcp_server_config_id
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_routing_decision(response_text: str) -> dict[str, Any]:
    """Parse a routing LLM response into a dict with tool_name and reasoning.

    Handles markdown code blocks and plain JSON.  Returns
    ``{"tool_name": None, "reasoning": "parse error"}`` on failure.
    """
    text = response_text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        # Remove opening fence (with optional language tag)
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()
    try:
        parsed = json.loads(text)
        return {
            "tool_name": parsed.get("tool_name"),
            "reasoning": parsed.get("reasoning", ""),
        }
    except (json.JSONDecodeError, AttributeError):
        log.warning(
            "Failed to parse routing response",
            extra={"extra_data": {"response": response_text[:200]}},
        )
        return {"tool_name": None, "reasoning": "parse error"}


# ---------------------------------------------------------------------------
# ToolRouter
# ---------------------------------------------------------------------------


class ToolRouter:
    """LLM-based tool router.

    Decides whether a user message should trigger a tool invocation.
    """

    def __init__(
        self,
        chat_model_factory: ChatModelFactory,
        routing_prompt: str = DEFAULT_ROUTING_PROMPT,
    ) -> None:
        self._factory = chat_model_factory
        self._routing_prompt = routing_prompt

    @traced()
    async def route(
        self,
        user_message: str,
        available_tool_llms: list[dict[str, Any]] | None = None,
        available_mcp_tools: list[dict[str, Any]] | None = None,
    ) -> ToolRoutingDecision:
        """Route to a tool based on the user message.

        Each tool_llm dict has: name, description, tool_llm_id, and optional
        model_name/provider_name.
        Each mcp_tool dict has: name, description, mcp_server_config_id.

        MCP tools are prefixed with ``mcp:`` for disambiguation in the
        routing prompt.
        """
        tool_llms = available_tool_llms or []
        mcp_tools = available_mcp_tools or []

        # No tools → nothing to route to
        if not tool_llms and not mcp_tools:
            return ToolRoutingDecision(
                tool_name=None, tool_type=None, tool_id=None, reasoning="no tools available"
            )

        # Pre-filter: recall intent
        if is_recall_intent(user_message):
            return ToolRoutingDecision(
                tool_name=None, tool_type=None, tool_id=None, reasoning="recall intent detected"
            )

        # Build description string
        descriptions: list[str] = []
        for t in tool_llms:
            descriptions.append(f"- {t['name']}: {t.get('description', '')}")
        for t in mcp_tools:
            descriptions.append(f"- mcp:{t['name']}: {t.get('description', '')}")
        tool_descriptions = "\n".join(descriptions)

        prompt_text = self._routing_prompt.format(tool_descriptions=tool_descriptions)

        # Create model and invoke
        model = await self._factory.create_chat_model(purpose="routing")
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=prompt_text),
            HumanMessage(content=user_message),
        ]
        response = await model.ainvoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        parsed = _parse_routing_decision(content)
        selected_name = parsed.get("tool_name")
        reasoning = parsed.get("reasoning", "")

        if selected_name is None:
            return ToolRoutingDecision(
                tool_name=None, tool_type=None, tool_id=None, reasoning=reasoning
            )

        # Match to tool-LLM
        for t in tool_llms:
            if t["name"] == selected_name:
                return ToolRoutingDecision(
                    tool_name=selected_name,
                    tool_type="tool_llm",
                    tool_id=t.get("tool_llm_id"),
                    reasoning=reasoning,
                )

        # Match to MCP tool (strip "mcp:" prefix if present)
        mcp_name = selected_name.removeprefix("mcp:")
        for t in mcp_tools:
            if t["name"] == mcp_name:
                return ToolRoutingDecision(
                    tool_name=mcp_name,
                    tool_type="mcp",
                    tool_id=t.get("mcp_server_config_id"),
                    reasoning=reasoning,
                )

        log.warning(
            "Routing selected unknown tool",
            extra={"extra_data": {"selected": selected_name}},
        )
        return ToolRoutingDecision(
            tool_name=None, tool_type=None, tool_id=None, reasoning=f"unknown tool: {selected_name}"
        )
