"""``invoke_tool_llm`` service tool — dispatch to a specialist tool-LLM.

A common agent pattern is "this conversation LLM can invoke a smaller,
specialised tool-LLM as a tool" — a planner LLM dispatches to a code
LLM, a translator LLM, an extractor LLM, etc.

The dispatch lookup (which tool-LLMs are registered, how they're
instantiated, whose credentials they use) is host-specific. The
recall-intent guard, the bind_tools wiring, the structured input
schema, and the LLM-facing error / redirect formatting are not — they
are the same across every host that wants this pattern.

This module exposes the host-agnostic surface as a factory:

- :class:`ToolLlmInvocation` — return value carrying the produced output
  + observability metadata.
- :class:`ToolLlmResolver` — protocol the host implements to resolve a
  tool name and run the underlying invocation.
- :func:`load_tool_llm_dispatch` — builds the LangChain ``BaseTool`` that
  graphs bind into a model via ``bind_tools``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from threetears.agent.tools.router import is_recall_intent
from threetears.observe import get_logger

__all__ = [
    "InvokeToolLlmInput",
    "ToolLlmInvocation",
    "ToolLlmResolver",
    "load_tool_llm_dispatch",
]


_log = get_logger(__name__)


_DEFAULT_DESCRIPTION = (
    "Invoke a specialised tool-LLM by name. Tool-LLMs are domain-specific "
    "AI assistants for specialised tasks. Use the exact tool name from the "
    "Available tools list in the system prompt. Use this ONLY when the user "
    "wants NEW work done by a specialised model. NEVER use this to recall, "
    "view, or retrieve previous tool output — use recall_context instead."
)


_RECALL_REDIRECT = (
    "[REDIRECT] This input is asking to recall previous output, not "
    "requesting new work. Do NOT invoke the tool-LLM for this. Instead, "
    "use the recall_context tool with the appropriate [ctx:UUID] from "
    "the Conversation Context to retrieve the stored output, or respond "
    "directly from the conversation context summaries."
)


class InvokeToolLlmInput(BaseModel):
    """Input schema for the ``invoke_tool_llm`` tool.

    :ivar tool_name: exact name of the tool-LLM to invoke; must match one
        of the entries in the system-prompt "Available tools" list.
    :ivar input_text: self-contained prompt for the tool-LLM. Tool-LLMs
        carry no conversation history — include all necessary context,
        requirements, and specifics in this text. References like
        "the same thing" or "as above" do not work.
    """

    tool_name: str = Field(
        description=(
            "Exact name of the tool-LLM to invoke (must match one of the "
            "available tool names listed in the system prompt)"
        ),
    )
    input_text: str = Field(
        description=(
            "Self-contained prompt for the tool-LLM. Tool-LLMs have NO "
            "conversation history — include all necessary context, "
            "requirements, and specifics in this text. Never use references "
            "like 'the same thing' or 'as above'."
        ),
    )


@dataclass
class ToolLlmInvocation:
    """Return value from a successful :meth:`ToolLlmResolver.resolve_and_invoke`.

    :ivar tool_name: canonical name of the tool-LLM that ran. Hosts that
        do fuzzy-name matching set this to the resolved name (which may
        differ from the input ``tool_name``); strict-match hosts set it
        equal to the input.
    :ivar output: text response produced by the tool-LLM. Empty string
        when the model returned no content (the dispatch tool surfaces
        an empty string to the calling LLM unchanged; the calling LLM
        handles "tool returned nothing" semantics).
    :ivar duration_ms: optional wall-clock duration of the underlying
        invocation. Hosts that emit observability spans / Prometheus
        records elsewhere may leave this ``None``.
    :ivar input_tokens: optional input-token count from the underlying
        provider response (best-effort).
    :ivar output_tokens: optional output-token count from the underlying
        provider response (best-effort).
    :ivar metadata: optional host-defined extras (e.g. resolved
        ``model_id``, ``tool_llm_id``) for observability sinks the host
        controls. The dispatch tool never reads this.
    """

    tool_name: str
    output: str
    duration_ms: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    metadata: dict[str, Any] | None = None


@runtime_checkable
class ToolLlmResolver(Protocol):
    """Host-implemented tool-LLM lookup + invocation contract.

    The 3tears ``invoke_tool_llm`` service tool delegates the actual
    "given a name + input, run it, return text" work to a host-provided
    resolver. Implementations are responsible for:

    - mapping the input ``tool_name`` to a registered tool-LLM (exact,
      fuzzy, scoped-by-user, etc. — the lookup policy is host-specific);
    - constructing or pooling the underlying chat model;
    - invoking the chat model and extracting the text response;
    - any host-specific observability (token logging, audit, etc.).

    Returning ``None`` signals "no tool-LLM by that name is available";
    the dispatch tool turns that into an LLM-facing ``[TOOL ERROR] ...
    not found`` message. Raising an exception is also acceptable — the
    dispatch tool catches it and produces an LLM-facing error string.
    """

    async def resolve_and_invoke(
        self,
        tool_name: str,
        input_text: str,
    ) -> ToolLlmInvocation | None:
        """Look up ``tool_name`` and run it with ``input_text``.

        :param tool_name: tool-LLM name supplied by the calling LLM
        :ptype tool_name: str
        :param input_text: prompt body supplied by the calling LLM
        :ptype input_text: str
        :return: invocation result, or ``None`` when the tool-LLM is
            not available
        :rtype: ToolLlmInvocation | None
        """
        ...


def load_tool_llm_dispatch(
    resolver: ToolLlmResolver,
    *,
    description: str | None = None,
) -> list[BaseTool]:
    """Build the ``invoke_tool_llm`` service tool from a host resolver.

    The returned list contains exactly one ``BaseTool``. Returning a
    list keeps the call shape consistent with the other ``load_*_tools``
    factories in 3tears so callers can extend a single ``tools = []``
    accumulator without special-casing this one.

    :param resolver: host-provided tool-LLM lookup + invocation
    :ptype resolver: ToolLlmResolver
    :param description: optional override for the LLM-facing tool
        description; defaults to the platform-standard wording that
        steers the model away from using this for recall / retrieval
    :ptype description: str | None
    :return: list with the single ``invoke_tool_llm`` tool
    :rtype: list[BaseTool]
    """

    async def _invoke(tool_name: str, input_text: str) -> str:
        # Recall-intent guard — same heuristic the platform uses
        # everywhere else; prevents the calling LLM from misusing the
        # tool-LLM dispatch as a way to retrieve already-stored output.
        if is_recall_intent(input_text):
            _log.info(
                "invoke_tool_llm: recall intent detected, redirecting",
                extra={"extra_data": {
                    "tool_name": tool_name,
                    "input_preview": input_text[:120],
                }},
            )
            return _RECALL_REDIRECT

        try:
            result = await resolver.resolve_and_invoke(tool_name, input_text)
        except Exception as exc:
            _log.warning(
                "invoke_tool_llm: resolver raised",
                extra={"extra_data": {
                    "tool_name": tool_name,
                    "error": str(exc),
                }},
            )
            return f"[TOOL ERROR] invoke_tool_llm: {exc}"

        if result is None:
            return (
                f"[TOOL ERROR] invoke_tool_llm: Tool-LLM {tool_name!r} "
                "not found or not enabled."
            )

        _log.info(
            "invoke_tool_llm completed",
            extra={"extra_data": {
                "tool_name": result.tool_name,
                "duration_ms": result.duration_ms,
                "output_length": len(result.output),
            }},
        )
        return result.output

    invoke_tool_llm = StructuredTool.from_function(
        coroutine=_invoke,
        name="invoke_tool_llm",
        description=description or _DEFAULT_DESCRIPTION,
        args_schema=InvokeToolLlmInput,
    )
    return [invoke_tool_llm]
