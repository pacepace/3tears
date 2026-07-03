"""3tears conversation-context merge middleware for ``langchain.agents.create_agent``.

The framework-aligned successor to the ``agent_node`` context-merge seam (the
hand-rolled node folded a per-conversation :class:`ToolContextManager`'s
accumulated tool-result context into the system prompt; that node was removed
with ``nodes.py``). Re-expressed onto
:meth:`langchain.agents.middleware.AgentMiddleware.awrap_model_call`: before each
model call, when a context manager is injected on
``config["configurable"]["context_manager"]``, its
:meth:`build_conversation_context` text is folded into the request's
``system_message`` so the model sees exactly ONE system message carrying the base
prompt plus the accumulated context.

Why the merge lives here and not as an extra ``SystemMessage`` in the message
list: LangChain's Anthropic binding rejects multiple non-consecutive system
messages (``ValueError("Received multiple non-consecutive system messages")``),
which the old ``context_enrichment_node`` triggered by appending a fresh
``SystemMessage`` every turn. Folding into the single ``request.system_message``
keeps one system message and sidesteps that failure.

Opt-in by construction: with no ``context_manager`` on ``config["configurable"]``
the seam is a byte-for-byte no-op (the request passes through unchanged). The
merge soft-fails: a manager that raises while building its context is logged and
the call proceeds on the un-merged system prompt rather than failing the turn.

Ordering: this middleware must sit OUTERMOST of the system-prompt-mutating
middleware (before :class:`~threetears.langgraph.middleware.PromptCachingMiddleware`)
so the cache annotation, when applied, covers the fully merged prompt.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage
from langgraph.config import get_config

from threetears.observe import get_logger

__all__ = ["ContextMergeMiddleware", "ConversationContextProvider"]

log = get_logger(__name__)


def _configurable() -> dict[str, Any]:
    """Return the active run's ``configurable`` dict, or empty when unavailable.

    Reads the run's :class:`~langchain_core.runnables.RunnableConfig` via
    :func:`langgraph.config.get_config`. The ``Runtime`` carried on a ``ModelRequest``
    does NOT expose ``config`` for the ``wrap_model_call`` seam (only the tool seam's
    ``ToolRuntime`` does), so ``get_config`` is the canonical reach to ``configurable``.
    Outside a runnable context it raises ``RuntimeError``; with no config there is no
    injected manager, so this soft-fails to an empty dict and the seam becomes a no-op.

    :return: the run's ``configurable`` mapping, or an empty dict.
    :rtype: dict[str, Any]
    """
    try:
        config = get_config()
    except RuntimeError:
        return {}
    return config.get("configurable") or {}


@runtime_checkable
class ConversationContextProvider(Protocol):
    """Structural contract for the per-conversation context manager.

    The host (e.g. the aibots SDK's ``ToolContextManager``) injects a concrete
    provider on ``config["configurable"]["context_manager"]``. Only the one
    method this middleware needs is part of the contract, so any manager that can
    render its accumulated context as text satisfies it without importing a
    3tears base class.
    """

    def build_conversation_context(self) -> str:
        """Return the accumulated conversation context as prompt text.

        :return: the context block to fold into the system prompt; an empty
            string when there is no accumulated context.
        :rtype: str
        """
        ...


def _fold_context_into_system(
    system_message: SystemMessage | None,
    context_text: str,
) -> SystemMessage:
    """Return a single ``SystemMessage`` carrying the base prompt plus context.

    Preserves the base system prompt and appends the context block after a blank
    line. When the base message has structured (list) content — e.g. a prior
    middleware already annotated it — the context is appended as an extra text
    part so existing parts (and any ``cache_control`` on them) are left intact.
    When there is no base system message, the context becomes the whole system
    message.

    :param system_message: the request's current system message, or ``None``.
    :ptype system_message: SystemMessage | None
    :param context_text: the non-empty accumulated-context block to fold in.
    :ptype context_text: str
    :return: the merged system message.
    :rtype: SystemMessage
    """
    if system_message is None:
        return SystemMessage(content=context_text)
    base_content = system_message.content
    if isinstance(base_content, list):
        merged_parts = [*base_content, {"type": "text", "text": context_text}]
        return system_message.model_copy(update={"content": merged_parts})
    base_text = base_content if isinstance(base_content, str) else str(base_content)
    return system_message.model_copy(update={"content": f"{base_text}\n\n{context_text}"})


def _merge_request_context(request: ModelRequest) -> ModelRequest:
    """Fold the injected context manager's context into ``request.system_message``.

    Reads the optional context manager off ``config["configurable"]``. Returns the
    request unchanged when no manager is injected, when it yields empty context,
    or when building the context fails (soft-fail, logged with context). Otherwise
    returns a request whose ``system_message`` carries the base prompt plus the
    accumulated context.

    :param request: the incoming model request.
    :ptype request: ModelRequest
    :return: the request, its ``system_message`` merged when context was present.
    :rtype: ModelRequest
    """
    manager = _configurable().get("context_manager")
    if manager is None:
        return request
    try:
        context_text = manager.build_conversation_context()
    except (
        Exception
    ) as exc:  # prawduct:allow prawduct/broad-except -- context render is best-effort; a fault must not fail the turn
        log.warning(
            "conversation-context merge failed; proceeding without it",
            extra={"extra_data": {"error": str(exc)}},
            exc_info=True,
        )
        return request
    if not context_text:
        return request
    merged = _fold_context_into_system(request.system_message, context_text)
    return request.override(system_message=merged)


class ContextMergeMiddleware(AgentMiddleware):
    """Fold accumulated per-conversation tool-result context into the system prompt.

    The ``create_agent`` successor to the ``agent_node`` context-merge seam. On
    every model call, when a context manager is injected on
    ``config["configurable"]["context_manager"]``, its accumulated context is
    folded into ``request.system_message`` (yielding exactly one system message).
    Opt-in: no manager on ``config["configurable"]`` -> byte-for-byte no-op.
    """

    name = "ContextMergeMiddleware"

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Any,
    ) -> ModelResponse:
        """Merge the accumulated context into the system prompt, then run the call.

        :param request: the model request (carries ``system_message`` + ``runtime``).
        :ptype request: ModelRequest
        :param handler: the next handler in the middleware chain.
        :ptype handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
        :return: the model response.
        :rtype: ModelResponse
        """
        return await handler(_merge_request_context(request))

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Any,
    ) -> ModelResponse:
        """Sync mirror of :meth:`awrap_model_call`.

        :param request: the model request.
        :ptype request: ModelRequest
        :param handler: the next handler in the middleware chain.
        :ptype handler: Callable[[ModelRequest], ModelResponse]
        :return: the model response.
        :rtype: ModelResponse
        """
        return handler(_merge_request_context(request))
