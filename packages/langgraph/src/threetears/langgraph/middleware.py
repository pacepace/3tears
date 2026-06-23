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


def _annotate_messages_for_cache(messages: list[Any], chat_model: Any) -> list[Any]:
    """Return ``messages`` with a leading SystemMessage cache_control-annotated.

    Mirrors the old ``PromptCachingHook._rewrite_system_prompt_for_cache``: when the
    model supports Anthropic ``cache_control`` and ``messages[0]`` is a bare-string
    :class:`SystemMessage`, replace it with the structured-content form carrying
    ``cache_control={"type": "ephemeral"}``. Idempotent (already-structured content is
    left alone); a no-op for non-caching models.

    :param messages: the request message list (system prefix at index 0 when present)
    :ptype messages: list[Any]
    :param chat_model: the chat model the request will hit (capability source)
    :ptype chat_model: Any
    :return: the message list, index 0 optionally rewritten
    :rtype: list[Any]
    """
    if not messages or not isinstance(messages[0], SystemMessage):
        return messages
    caps = detect_capabilities(chat_model)
    if not caps.supports_anthropic_cache_control:
        return messages
    head = messages[0]
    if isinstance(head.content, list):
        return messages  # already structured (caller or a prior pass annotated it)
    prompt_text = head.content if isinstance(head.content, str) else str(head.content)
    result = list(messages)
    result[0] = annotate_system_prompt(prompt_text, caps)
    return result


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
            except (AttributeError, ValueError):
                log.debug(
                    "response message %s does not accept usage_metadata assignment",
                    type(message).__name__,
                )


class PromptCachingMiddleware(AgentMiddleware):
    """Annotate the system prompt for Anthropic prompt caching + normalize usage.

    The ``create_agent`` successor to ``PromptCachingHook`` (an ``AgentNodeHook``).
    On every model call it rewrites a leading bare-string ``SystemMessage`` to the
    ``cache_control`` structured form when the model supports it, then normalizes the
    response's cache-hit/creation counters onto ``usage_metadata["cache_usage"]``.

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
        annotated = _annotate_messages_for_cache(request.messages, request.model)
        if annotated is not request.messages:
            request = request.override(messages=annotated)
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
        annotated = _annotate_messages_for_cache(request.messages, request.model)
        if annotated is not request.messages:
            request = request.override(messages=annotated)
        response = handler(request)
        _stamp_cache_usage(response)
        return response
