"""3tears conversation-summarization middleware for ``langchain.agents.create_agent``.

The framework-aligned successor to the old ``SummarizationHook`` (an
``AgentNodeHook`` removed with ``hooks.py``). Re-expressed onto
:meth:`langchain.agents.middleware.AgentMiddleware.abefore_model`: before a model
call, when the active message window exceeds a message-count trigger, the older
messages are distilled via :func:`threetears.langgraph.summarize.summarize_older_messages`
and the window is replaced with ``[summary, *recent]``.

Distinct from langchain's batteries-included ``SummarizationMiddleware`` in two
ways the platform needs: (1) the summary is produced by 3tears'
``summarize_older_messages`` (one shared distillation prompt + heuristic fallback
across every 3tears product), and (2) the internal summary model call is tagged
with :data:`~threetears.langgraph.streaming.NOSTREAM_TAG` so its tokens never leak
into the user-facing token stream that :meth:`StreamingResponse.run_graph`
accumulates.

The replacement uses ``RemoveMessage(id=REMOVE_ALL_MESSAGES)`` followed by the
summary + preserved tail, the LangGraph idiom for rewriting the reduced
``messages`` channel in place. In ``create_agent`` the system prompt is carried
separately on ``ModelRequest.system_message`` (never in the reduced ``messages``
list), so replacing ``messages`` here does not touch the system prompt or the
cache prefix.

Async is the real path (the agent streams via ``astream_events`` ->
``abefore_model``). The synchronous ``before_model`` mirror cannot drive the
async ``summarize_older_messages`` and degrades to a no-op (the window rides
un-summarized), warning so the misconfiguration is visible.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid7

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import AgentState
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

from threetears.langgraph.streaming import NOSTREAM_TAG
from threetears.langgraph.summarize import summarize_older_messages
from threetears.observe import get_logger

__all__ = ["SummarizationMiddleware"]

log = get_logger(__name__)

#: Default message-count window that triggers summarization. Mirrors the retired
#: ``SummarizationHook`` default; callers override via the constructor.
_DEFAULT_TRIGGER_MESSAGES = 20

#: Default number of most-recent messages preserved verbatim after summarization.
_DEFAULT_KEEP_MESSAGES = 10


def _ensure_message_ids(messages: list[AnyMessage]) -> None:
    """Assign a stable id to any message missing one.

    The ``add_messages`` reducer keys on message id to apply
    ``RemoveMessage(id=REMOVE_ALL_MESSAGES)`` and re-add the preserved tail without
    duplication. A message minted without an id would be dropped or duplicated by
    the reducer, so every message is given one before the rewrite.

    :param messages: the active message window (mutated in place).
    :ptype messages: list[AnyMessage]
    :return: nothing.
    :rtype: None
    """
    for message in messages:
        if message.id is None:
            message.id = str(uuid7())


def _safe_cutoff(messages: list[AnyMessage], keep: int) -> int:
    """Return the index where the preserved (recent) window should start.

    Starts ``keep`` messages from the end, then advances forward past any leading
    ``ToolMessage`` so the preserved window never opens on a tool result orphaned
    from the ``AIMessage`` that requested it (which some providers reject). The
    advanced-over tool results fall into the summarized older window.

    :param messages: the full active message window.
    :ptype messages: list[AnyMessage]
    :param keep: the number of most-recent messages to try to preserve.
    :ptype keep: int
    :return: the cutoff index (start of the preserved window); ``0`` when nothing
        should be summarized.
    :rtype: int
    """
    if len(messages) <= keep:
        return 0
    cutoff = len(messages) - keep
    while cutoff < len(messages) and isinstance(messages[cutoff], ToolMessage):
        cutoff += 1
    return cutoff


class SummarizationMiddleware(AgentMiddleware):
    """Distill the older conversation window into a summary when it grows too long.

    The ``create_agent`` successor to the ``SummarizationHook``. When the message
    count exceeds ``trigger_messages``, the older messages are summarized via
    3tears' shared ``summarize_older_messages`` (NOSTREAM-tagged) and the window is
    replaced with ``[summary, *recent]``. A window at or below the trigger passes
    through unchanged.
    """

    name = "SummarizationMiddleware"

    def __init__(
        self,
        model: BaseChatModel,
        *,
        trigger_messages: int = _DEFAULT_TRIGGER_MESSAGES,
        keep_messages: int = _DEFAULT_KEEP_MESSAGES,
        custom_prompt: str | None = None,
    ) -> None:
        """Configure the summarization trigger, retention window, and model.

        :param model: the chat model used to produce the summary (typically the
            same model the agent runs on).
        :ptype model: BaseChatModel
        :param trigger_messages: summarize once the window exceeds this many
            messages.
        :ptype trigger_messages: int
        :param keep_messages: number of most-recent messages preserved verbatim.
        :ptype keep_messages: int
        :param custom_prompt: optional override for the shared 3tears summarization
            prompt.
        :ptype custom_prompt: str | None
        :return: nothing.
        :rtype: None
        """
        super().__init__()
        self.model = model
        self.trigger_messages = trigger_messages
        self.keep_messages = keep_messages
        self.custom_prompt = custom_prompt

    async def abefore_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """Summarize the older window when the message count exceeds the trigger.

        :param state: the agent state (its ``messages`` window is read).
        :ptype state: AgentState[Any]
        :param runtime: the LangGraph runtime (unused; part of the hook contract).
        :ptype runtime: Runtime[Any]
        :return: a ``messages`` rewrite when summarization ran, else ``None``.
        :rtype: dict[str, Any] | None
        """
        messages = state["messages"]
        if len(messages) <= self.trigger_messages:
            return None
        _ensure_message_ids(messages)
        cutoff = _safe_cutoff(messages, self.keep_messages)
        if cutoff <= 0:
            return None
        older = messages[:cutoff]
        preserved = messages[cutoff:]
        summary = await summarize_older_messages(
            older,
            self.model,
            custom_prompt=self.custom_prompt,
            config={"tags": [NOSTREAM_TAG]},
        )
        summary_message = HumanMessage(
            content=f"Summary of the conversation so far:\n\n{summary}",
            id=str(uuid7()),
        )
        log.info(
            "conversation summarized",
            extra={
                "extra_data": {
                    "summarized_messages": len(older),
                    "preserved_messages": len(preserved),
                }
            },
        )
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), summary_message, *preserved]}

    def before_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """Sync mirror: cannot drive the async summary call, so degrade to a no-op.

        ``summarize_older_messages`` is ``async``; a synchronous ``before_model``
        path cannot await it. Rather than fail the run, the window rides
        un-summarized (its own soft-fail fallback) and a warning is logged when the
        window was over the trigger so the misconfiguration is visible.

        :param state: the agent state.
        :ptype state: AgentState[Any]
        :param runtime: the LangGraph runtime (unused; part of the hook contract).
        :ptype runtime: Runtime[Any]
        :return: always ``None`` (no rewrite on the sync path).
        :rtype: dict[str, Any] | None
        """
        if len(state["messages"]) > self.trigger_messages:
            log.warning(
                "summarization skipped: the agent ran on the synchronous model path; "
                "3tears summarization is async-only, so the full window rides un-summarized",
            )
        return None
