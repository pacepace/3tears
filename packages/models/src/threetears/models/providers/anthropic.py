"""anthropic chat provider adapter wrapping langchain-anthropic."""

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

# backward-compatible aliases for existing imports of private names
_messages_to_lc = messages_to_lc
_ai_message_to_result = ai_message_to_result
_ai_chunk_to_chat_chunk = ai_chunk_to_chat_chunk
_tool_def_to_lc = tool_def_to_lc


class AnthropicChatProvider:
    """chat provider adapter for Anthropic models via langchain-anthropic.

    wraps ChatAnthropic with lazy instantiation, converting between
    threetears message types and LangChain message types at boundaries.

    :param model_name: Anthropic model identifier (e.g. claude-sonnet-4-20250514)
    :ptype model_name: str
    :param api_key: Anthropic API key for authentication
    :ptype api_key: str
    :param base_url: optional custom API base URL
    :ptype base_url: str | None
    :param timeout: request timeout in seconds
    :ptype timeout: int
    :param max_retries: maximum retry attempts for failed requests
    :ptype max_retries: int
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        *,
        base_url: str | None = None,
        timeout: int = 120,
        max_retries: int = 2,
    ) -> None:
        self._model_name = model_name
        self._api_key = api_key
        self._base_url = _strip_v1_suffix(base_url) if base_url else None
        self._timeout = timeout
        self._max_retries = max_retries
        self._model: Any = None
        self._tools: list[ToolDefinition] | None = None

    def _get_model(self) -> Any:
        """lazily creates and caches ChatAnthropic instance.

        imports langchain_anthropic on first call to avoid module-level
        dependency on optional package.

        :return: configured ChatAnthropic instance, optionally with tools bound
        :rtype: Any
        """
        if self._model is not None:
            return self._model

        from langchain_anthropic import ChatAnthropic

        kwargs: dict[str, Any] = {
            "model_name": self._model_name,
            "api_key": self._api_key,
            "timeout": self._timeout,
            "max_retries": self._max_retries,
        }
        if self._base_url is not None:
            kwargs["base_url"] = self._base_url

        base_model: Any = ChatAnthropic(**kwargs)

        if self._tools:
            lc_tools = [_tool_def_to_lc(t) for t in self._tools]
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
        lc_messages = _messages_to_lc(messages)
        response = await self._get_model().ainvoke(lc_messages, **kwargs)
        result = _ai_message_to_result(response)
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
        lc_messages = _messages_to_lc(messages)
        async for chunk in self._get_model().astream(lc_messages, **kwargs):
            yield _ai_chunk_to_chat_chunk(chunk)

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
        """preprocesses messages before sending to Anthropic model.

        applies capability-based transforms via preprocessing pipeline.
        Anthropic models do not require alternating roles, so this is
        effectively passthrough for standard configurations.

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


def _strip_v1_suffix(url: str) -> str:
    """strips trailing /v1 or /v1/ from URL.

    Anthropic SDK auto-appends /v1, so passing a URL ending with /v1
    would cause doubled path segments.

    :param url: base URL to clean
    :ptype url: str
    :return: URL with /v1 suffix removed if present
    :rtype: str
    """
    if url.endswith("/v1"):
        return url[:-3]
    if url.endswith("/v1/"):
        return url[:-4]
    return url
