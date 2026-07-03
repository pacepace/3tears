"""3tears platform middleware for ``langchain.agents.create_agent``.

The framework-aligned successor to the hand-rolled ``AgentNodeHook`` /
``ToolNodeHook`` protocols. ``create_agent`` extends via
:class:`langchain.agents.middleware.AgentMiddleware`; this module holds the
**platform-level** middleware every 3tears consumer can reuse. Consumer-specific
policy (tool side-effects, skill-rebind, loop governance) lives in each consumer
as its own ``AgentMiddleware`` and is NOT here.

What lives here today:

- :class:`PromptCachingMiddleware` — Anthropic prompt-cache annotation
  (``cache_control``) + cache-usage normalization, re-expressed from the old
  ``PromptCachingHook`` onto ``wrap_model_call``.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, SystemMessage

from threetears.langgraph.caching import (
    annotate_system_prompt,
    detect_capabilities,
    extract_cache_usage,
)
from threetears.observe import get_logger

__all__ = ["PromptCachingMiddleware"]

log = get_logger(__name__)


def _annotate_system_message_for_cache(
    system_message: SystemMessage | None,
    chat_model: Any,
) -> SystemMessage | None:
    """Return a cache_control-annotated copy of the system message, or ``None``.

    ``create_agent`` carries the system prompt on ``ModelRequest.system_message``,
    NOT as ``messages[0]`` — the reduced ``messages`` list excludes it (it is
    prepended only at the innermost model call). When the model supports Anthropic
    ``cache_control`` and the system message is bare-string content, this returns
    its structured-content form carrying ``cache_control={"type": "ephemeral"}``.
    Returns ``None`` (meaning "leave the request unchanged") when there is no system
    message, the model does not support caching, or the content is already
    structured (idempotent).

    :param system_message: the request's system message, or ``None``
    :ptype system_message: SystemMessage | None
    :param chat_model: the chat model the request will hit (capability source)
    :ptype chat_model: Any
    :return: the annotated system message, or ``None`` when no change applies
    :rtype: SystemMessage | None
    """
    if system_message is None:
        return None
    caps = detect_capabilities(chat_model)
    if not caps.supports_anthropic_cache_control:
        return None
    if isinstance(system_message.content, list):
        return None  # already structured (a prior pass annotated it)
    prompt_text = system_message.content if isinstance(system_message.content, str) else str(system_message.content)
    return annotate_system_prompt(prompt_text, caps)


def _stamp_cache_usage(response: ModelResponse) -> None:
    """Stamp normalized ``cache_usage`` onto the response's AI message in place.

    Mirrors the old ``PromptCachingHook.after_invoke``: runs
    :func:`extract_cache_usage` and writes it under
    ``message.usage_metadata["cache_usage"]`` so downstream readers (gateway
    telemetry, tests) get a uniform shape. Best-effort — never raises.

    :param response: the ModelResponse returned by the wrapped handler
    :ptype response: ModelResponse
    :return: nothing
    :rtype: None
    """
    messages = getattr(response, "result", None) or []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        usage = extract_cache_usage(message)
        existing = getattr(message, "usage_metadata", None)
        if isinstance(existing, dict):
            existing["cache_usage"] = usage
        else:
            try:
                message.usage_metadata = {"cache_usage": usage}  # type: ignore[assignment]
            except Exception:  # noqa: BLE001 - exotic response shapes may refuse the assignment
                log.debug(
                    "response message %s does not accept usage_metadata assignment",
                    type(message).__name__,
                )


class PromptCachingMiddleware(AgentMiddleware):
    """Annotate the system prompt for Anthropic prompt caching + normalize usage.

    The ``create_agent`` successor to ``PromptCachingHook`` (an ``AgentNodeHook``).
    On every model call it rewrites the request's bare-string ``system_message`` to
    the ``cache_control`` structured form when the model supports it, then normalizes
    the response's cache-hit/creation counters onto ``usage_metadata["cache_usage"]``.
    The system prompt is annotated on ``request.system_message`` (where
    ``create_agent`` carries it) — NOT ``messages[0]``, which excludes the system
    message.

    Deliberately NOT ported: the old hook's per-round ``bind_tools`` memoization
    (the ``_BOUND_MODEL_CACHE`` WeakKeyDictionary). That optimization existed because
    the hand-rolled ``agent_node`` re-bound tools on every round; ``create_agent``
    binds once at construction, so the memoization is moot. If a future need for
    cross-call bind caching appears, it belongs in ``wrap_model_call`` via
    ``request.override(model=...)`` — added then, not speculatively now.
    """

    name = "PromptCachingMiddleware"

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Any,
    ) -> ModelResponse:
        """Annotate the request's system prompt, run the call, stamp cache usage.

        :param request: the model request (carries ``model`` + ``messages``)
        :ptype request: ModelRequest
        :param handler: the next handler in the middleware chain
        :ptype handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
        :return: the model response with ``cache_usage`` normalized
        :rtype: ModelResponse
        """
        annotated = _annotate_system_message_for_cache(request.system_message, request.model)
        if annotated is not None:
            request = request.override(system_message=annotated)
        response = await handler(request)
        _stamp_cache_usage(response)
        return response

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Any,
    ) -> ModelResponse:
        """Sync mirror of :meth:`awrap_model_call`.

        :param request: the model request
        :ptype request: ModelRequest
        :param handler: the next handler in the middleware chain
        :ptype handler: Callable[[ModelRequest], ModelResponse]
        :return: the model response with ``cache_usage`` normalized
        :rtype: ModelResponse
        """
        annotated = _annotate_system_message_for_cache(request.system_message, request.model)
        if annotated is not None:
            request = request.override(system_message=annotated)
        response = handler(request)
        _stamp_cache_usage(response)
        return response
