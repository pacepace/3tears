"""openrouter chat provider adapter wrapping langchain-openrouter."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from threetears.models.messages import ChatMessage, ToolDefinition
from threetears.models.providers._conversions import (
    ai_chunk_to_chat_chunk,
    ai_message_to_result,
    messages_to_lc,
    tool_def_to_lc,
)
from threetears.models.results import ChatChunk, ChatResult

__all__ = [
    "OpenRouterChatProvider",
]


class OpenRouterChatProvider:
    """chat provider adapter for OpenRouter models via langchain-openrouter.

    wraps ChatOpenRouter with lazy instantiation, converting between
    threetears message types and LangChain message types at boundaries.
    accepts timeout in seconds and converts to milliseconds for
    ChatOpenRouter which expects millisecond values.

    :param model_name: OpenRouter model identifier (e.g. deepseek/deepseek-chat-v3-0324)
    :ptype model_name: str
    :param api_key: OpenRouter API key for authentication
    :ptype api_key: str
    :param timeout: request timeout in seconds (converted to milliseconds internally)
    :ptype timeout: int
    :param max_retries: maximum retry attempts for failed requests
    :ptype max_retries: int
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        *,
        timeout: int = 120,
        max_retries: int = 2,
    ) -> None:
        self._model_name = model_name
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._model: Any = None
        self._tools: list[ToolDefinition] | None = None

    def _get_model(self) -> Any:
        """lazily creates and caches ChatOpenRouter instance.

        imports langchain_openrouter on first call to avoid module-level
        dependency on optional package. converts stored timeout from
        seconds to milliseconds as required by ChatOpenRouter.

        :return: configured ChatOpenRouter instance, optionally with tools bound
        :rtype: Any
        """
        if self._model is not None:
            return self._model

        from langchain_openrouter import ChatOpenRouter

        kwargs: dict[str, Any] = {
            "model": self._model_name,
            "api_key": self._api_key,
            "timeout": self._timeout * 1000,
            "max_retries": self._max_retries,
        }

        base_model: Any = ChatOpenRouter(**kwargs)

        if self._tools:
            lc_tools = [tool_def_to_lc(t) for t in self._tools]
            base_model = base_model.bind_tools(lc_tools)

        self._model = base_model
        return self._model

    async def complete(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResult:
        """generates chat completion from message history.

        converts threetears messages to LangChain format, invokes model,
        and converts response back to ChatResult.

        :param messages: ordered list of conversation messages
        :ptype messages: list[ChatMessage]
        :param kwargs: additional parameters passed to LangChain ainvoke
        :ptype kwargs: Any
        :return: chat completion result with content, tool calls, and usage
        :rtype: ChatResult
        """
        lc_messages = messages_to_lc(messages)
        response = await self._get_model().ainvoke(lc_messages, **kwargs)
        result = ai_message_to_result(response)
        return result

    async def stream(self, messages: list[ChatMessage], **kwargs: Any) -> AsyncIterator[ChatChunk]:
        """streams chat completion chunks from message history.

        converts threetears messages to LangChain format and yields
        converted chunks from async stream.

        :param messages: ordered list of conversation messages
        :ptype messages: list[ChatMessage]
        :param kwargs: additional parameters passed to LangChain astream
        :ptype kwargs: Any
        :return: async iterator of chat completion chunks
        :rtype: AsyncIterator[ChatChunk]
        """
        lc_messages = messages_to_lc(messages)
        async for chunk in self._get_model().astream(lc_messages, **kwargs):
            yield ai_chunk_to_chat_chunk(chunk)

    def bind_tools(self, tools: list[ToolDefinition]) -> None:
        """binds tool definitions for subsequent completions.

        stores tools and clears cached model instance so next call
        recreates model with tools bound.

        :param tools: tool definitions available to model
        :ptype tools: list[ToolDefinition]
        """
        self._tools = list(tools)
        self._model = None

    def preprocess(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """preprocesses messages before sending to OpenRouter model.

        applies capability-based transforms via preprocessing pipeline.
        OpenRouter models do not require alternating roles by default,
        so this is effectively passthrough for standard configurations.

        :param messages: raw conversation messages
        :ptype messages: list[ChatMessage]
        :return: preprocessed messages ready for model
        :rtype: list[ChatMessage]
        """
        from threetears.models.capabilities import ModelCapabilities
        from threetears.models.enums import ModelStatus, ModelTier, ModelType
        from threetears.models.preprocessing import preprocess_messages

        capabilities = ModelCapabilities(
            model_name=self._model_name,
            model_type=ModelType.CHAT,
            model_tier=ModelTier.LARGE,
            model_status=ModelStatus.ACTIVE,
            requires_alternating_roles=False,
        )
        result = preprocess_messages(messages, capabilities)
        return result
