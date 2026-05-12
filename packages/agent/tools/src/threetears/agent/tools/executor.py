"""Tool execution: service tool loop for LLM-driven tool invocations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import ToolMessage

from threetears.observe import get_logger, traced

__all__ = [
    "ToolExecutionResult",
    "ToolExecutor",
]

log = get_logger(__name__)


@dataclass
class ToolExecutionResult:
    """Result of a tool execution loop."""

    output: str
    rounds_used: int
    tool_calls_made: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def _extract_text_content(response: Any) -> str:
    """Extract text content from an LLM response."""
    content = getattr(response, "content", None)
    if content is None:
        return str(response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "\n".join(parts)
    return str(content)


class ToolExecutor:
    """Reusable service tool loop for tool-LLM invocations.

    Repeatedly invokes a chat model, executing any tool calls it returns,
    until the model produces a text response or ``max_rounds`` is exhausted.
    """

    def __init__(self, max_rounds: int = 3) -> None:
        self._max_rounds = max_rounds

    @traced()
    async def invoke_with_tools(
        self,
        chat_model: Any,
        messages: list[Any],
        service_tools: list[Any],
    ) -> ToolExecutionResult:
        """Invoke a chat model with service tool support.

        Parameters
        ----------
        chat_model:
            A LangChain-compatible chat model (must support ``ainvoke``).
        messages:
            The message list to send (will be mutated in-place with tool
            messages as the loop progresses).
        service_tools:
            LangChain ``BaseTool`` instances available for the model to call.
        """
        tool_map = {t.name: t for t in service_tools}
        all_tool_calls: list[dict[str, Any]] = []

        for round_num in range(1, self._max_rounds + 1):
            response = await chat_model.ainvoke(messages)
            tool_calls = getattr(response, "tool_calls", []) or []

            if not tool_calls:
                return ToolExecutionResult(
                    output=_extract_text_content(response),
                    rounds_used=round_num,
                    tool_calls_made=all_tool_calls,
                )

            # Append the AI message (with tool_calls) to the conversation
            messages.append(response)

            for tc in tool_calls:
                tc_name = tc.get("name", "")
                tc_args = tc.get("args", {})
                tc_id = tc.get("id", "")

                all_tool_calls.append({"name": tc_name, "args": tc_args})

                tool = tool_map.get(tc_name)
                if tool is None:
                    log.warning(
                        "Tool not found during execution",
                        extra={"extra_data": {"tool_name": tc_name}},
                    )
                    messages.append(
                        ToolMessage(
                            content=f"Error: tool not found: {tc_name}",
                            tool_call_id=tc_id,
                        )
                    )
                    continue

                try:
                    result = str(await tool.ainvoke(tc_args))
                except Exception as exc:
                    log.error(
                        "Tool execution failed",
                        extra={"extra_data": {"tool_name": tc_name, "error": str(exc)}},
                    )
                    result = f"Error executing {tc_name}: {exc}"

                messages.append(ToolMessage(content=result, tool_call_id=tc_id))

        # Max rounds exhausted — do one final invocation to get text
        response = await chat_model.ainvoke(messages)
        return ToolExecutionResult(
            output=_extract_text_content(response),
            rounds_used=self._max_rounds,
            tool_calls_made=all_tool_calls,
            error="max rounds exhausted" if all_tool_calls else None,
        )
