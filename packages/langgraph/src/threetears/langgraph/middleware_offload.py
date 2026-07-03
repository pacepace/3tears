"""3tears large-tool-result offload middleware for ``langchain.agents.create_agent``.

The framework-aligned successor to the old ``tool_node`` offload seam (the
hand-rolled node's ``_maybe_offload_result`` branch, removed with ``nodes.py``).
Re-expressed onto :meth:`langchain.agents.middleware.AgentMiddleware.awrap_tool_call`:
after a tool runs, when its :class:`~langchain_core.messages.ToolMessage` content
exceeds the configured threshold and a
:class:`~threetears.langgraph.offload.ToolResultOffloader` is injected on
``config["configurable"]``, the full content is stored out-of-band and the model
sees ``"<summary>\\n\\n[ctx:<handle>]"`` instead of the raw dump. The structured
``artifact`` is preserved on the ToolMessage even when the content is offloaded.

Opt-in by construction: with no offloader on ``config["configurable"]`` the seam is
byte-for-byte a no-op (the ToolMessage passes through unchanged). Every other guard
of the original seam is preserved verbatim: the never-offload tool set, the size
threshold, failures ride inline (never offloaded), a missing ``conversation_id``
skips, the tool-authored summary is lifted from ``ToolMessage.artifact`` (2b) and
dropped when it is itself oversized or carries a ``[ctx:`` handle token,
``GraphBubbleUp`` propagates, and any offloader error soft-fails to the full content.

The real seam is asynchronous — the :class:`ToolResultOffloader` contract's
``offload`` is ``async def``, matching the async-only original ``tool_node``. The
synchronous :meth:`~ToolResultOffloadMiddleware.wrap_tool_call` mirror cannot drive
an async offloader, so on the sync tool-call path it degrades to the full content
(the seam's own documented soft-fail fallback) and warns when an offloader was in
fact configured.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp

from threetears.langgraph.offload import (
    DEFAULT_OFFLOAD_THRESHOLD_CHARS,
    format_offload_handle,
    is_never_offload_tool,
)
from threetears.observe import get_logger

__all__ = ["ToolResultOffloadMiddleware"]

log = get_logger(__name__)


def _tool_summary_from_artifact(artifact: Any) -> str | None:
    """lift a tool-authored summary (2b) off a ToolMessage ``artifact``.

    a content_and_artifact tool can ship a short summary under
    ``artifact["summary"]``; a content-format tool has ``artifact=None`` (no
    summary). only a non-empty ``str`` summary is honored.

    :param artifact: the ``ToolMessage.artifact`` (``None`` for content-format tools).
    :ptype artifact: Any
    :return: the tool-authored summary, or ``None``.
    :rtype: str | None
    """
    if isinstance(artifact, dict):
        raw_summary = artifact.get("summary")
        if isinstance(raw_summary, str) and raw_summary:
            return raw_summary
    return None


async def _maybe_offload_result(
    configurable: dict[str, Any],
    tool_name: str,
    content: str,
    success: bool,
    tool_summary: str | None = None,
) -> str:
    """offload an oversized tool result, returning the model-visible text.

    reads the optional :class:`~threetears.langgraph.offload.ToolResultOffloader`
    and the size threshold off ``configurable``. offload fires only when ALL hold:
    the tool succeeded, an offloader is present, a ``conversation_id`` is known, the
    tool is not in the never-offload set, and ``len(content)`` exceeds the
    threshold. on a returned :class:`~threetears.langgraph.offload.OffloadResult`
    the model sees ``"<summary>\\n\\n[ctx:<handle>]"``; otherwise (no offloader,
    under threshold, not success, a never-offload tool, declined, or a soft-failed
    offload) the original ``content`` is returned unchanged.

    SOFT-FAIL: an exception from ``offload(...)`` is logged with context and the
    full ``content`` is returned -- a big message beats a dropped tool result.
    ``GraphBubbleUp`` propagates (it is a control-flow signal, not a storage error).

    :param configurable: the ``config["configurable"]`` dict for this run.
    :ptype configurable: dict[str, Any]
    :param tool_name: name of the tool whose result may be offloaded.
    :ptype tool_name: str
    :param content: serialized tool-result content (already ``str()``-ed).
    :ptype content: str
    :param success: whether the tool invocation succeeded; failures are never
        offloaded so the error text stays inline for the model.
    :ptype success: bool
    :param tool_summary: a tool-authored summary (2b) lifted from the result's
        ``ToolMessage.artifact``, or ``None``; forwarded to the offloader (which
        prefers it) unless it meets the offload threshold or carries a ``[ctx:``
        token -- then it is dropped for the structural summary.
    :ptype tool_summary: str | None
    :return: the content to place in the ``ToolMessage`` -- either the
        ``summary + [ctx:<handle>]`` form or the original ``content``.
    :rtype: str
    """
    offloader = configurable.get("tool_result_offloader")
    threshold = int(
        configurable.get("offload_threshold_chars", DEFAULT_OFFLOAD_THRESHOLD_CHARS),
    )
    conversation_id = configurable.get("conversation_id")
    user_id = configurable.get("user_id")
    # 2b: only honor a tool-authored summary that is actually a SUMMARY -- one
    # at/above the threshold saves no context (reintroducing the very bloat
    # offload removes), and one carrying the "[ctx:" handle token could
    # surface a spurious recall handle to the model. fall back to structural.
    if tool_summary is not None and (len(tool_summary) >= threshold or "[ctx:" in tool_summary):
        tool_summary = None
    # never offload a tool whose result IS recalled content (the recall
    # tool itself): re-offloading it loops -- the model asked for the
    # bytes and would get a fresh handle instead.
    should_offload = (
        success
        and offloader is not None
        and conversation_id is not None
        and not is_never_offload_tool(tool_name)
        and len(content) > threshold
    )
    message_content = content
    if should_offload:
        offload_result = None
        try:
            offload_result = await offloader.offload(
                tool_name=tool_name,
                content=content,
                conversation_id=conversation_id,
                user_id=user_id,
                tool_summary=tool_summary,
            )
        except GraphBubbleUp:
            # control-flow signal (interrupt / subgraph bubble) -- never a
            # storage error; it MUST propagate so the broad soft-fail below
            # cannot swallow it.
            raise
        except Exception as exc:
            log.warning(
                "tool-result offload failed; falling back to full content",
                extra={
                    "extra_data": {
                        "tool_name": tool_name,
                        "content_chars": len(content),
                        "error": str(exc),
                    },
                },
                exc_info=True,
            )
        if offload_result is not None:
            message_content = f"{offload_result.summary}\n\n{format_offload_handle(offload_result.handle)}"
    return message_content


class ToolResultOffloadMiddleware(AgentMiddleware):
    """Offload an oversized tool result out-of-band, showing the model a recall handle.

    The ``create_agent`` successor to the ``tool_node`` offload seam. On every tool
    call it runs the tool, then -- when an offloader is injected and the serialized
    content exceeds the threshold -- rewrites the ToolMessage content to
    ``"<summary>\\n\\n[ctx:<handle>]"``, preserving the structured ``artifact``.
    Opt-in: no offloader on ``config["configurable"]`` -> byte-for-byte no-op.
    """

    name = "ToolResultOffloadMiddleware"

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Any,
    ) -> Any:
        """Run the tool, then offload its result when it is oversized.

        :param request: the tool-call request (``tool_call`` + ``runtime.config``).
        :ptype request: ToolCallRequest
        :param handler: async callable that executes the tool call.
        :ptype handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]
        :return: the ToolMessage (content possibly rewritten) or the raw handler
            result when it is not a ToolMessage (e.g. a ``Command``).
        :rtype: ToolMessage | Command
        """
        result = await handler(request)
        if not isinstance(result, ToolMessage):
            return result
        configurable = request.runtime.config.get("configurable", {})
        tool_name = request.tool_call["name"]
        success = result.status != "error"
        tool_summary = _tool_summary_from_artifact(result.artifact)
        message_content = await _maybe_offload_result(
            configurable,
            tool_name,
            str(result.content),
            success,
            tool_summary,
        )
        if message_content == result.content:
            return result
        # preserve artifact / status / tool_call_id / name; only the model-visible
        # content is swapped for the summary + recall handle.
        return result.model_copy(update={"content": message_content})

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Any,
    ) -> Any:
        """Sync mirror: cannot drive the async offloader, so degrade to full content.

        The :class:`~threetears.langgraph.offload.ToolResultOffloader` contract is
        ``async def``; a synchronous tool-call path cannot await it. Rather than
        break the run, the tool result rides inline unchanged (byte-for-byte the
        seam's ``no offloader`` / soft-fail fallback). A warning is logged when an
        offloader was configured so the misconfiguration is visible.

        :param request: the tool-call request.
        :ptype request: ToolCallRequest
        :param handler: sync callable that executes the tool call.
        :ptype handler: Callable[[ToolCallRequest], ToolMessage | Command]
        :return: the handler result unchanged.
        :rtype: ToolMessage | Command
        """
        result = handler(request)
        if not isinstance(result, ToolMessage):
            return result
        configurable = request.runtime.config.get("configurable", {})
        if configurable.get("tool_result_offloader") is not None:
            log.warning(
                "tool-result offload skipped: an offloader is configured but the "
                "agent ran on the synchronous tool-call path; the offloader is "
                "async-only, so the full content rides inline",
                extra={"extra_data": {"tool_name": request.tool_call["name"]}},
            )
        return result
