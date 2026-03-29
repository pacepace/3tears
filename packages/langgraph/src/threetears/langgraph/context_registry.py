"""per-conversation context manager registry.

routes context operations to the correct ToolContextManager based on
the active conversation_id. creates managers lazily on first access.
uses contextvars for async-safe conversation routing.

usage:
    from threetears.langgraph import ContextManagerRegistry, current_conversation_id

    registry = ContextManagerRegistry(context_collection=collection)

    # set active conversation (typically in message handler)
    current_conversation_id.set(str(conversation_id))

    # all context ops route to the active conversation's manager
    await registry.save_tool_result("my_tool", result, "tool did X")
    prompt = registry.build_context_prompt()
"""

from __future__ import annotations

import contextvars
import logging
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

current_conversation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_conversation_id", default="default",
)


class ContextManagerRegistry:
    """per-conversation context manager multiplexer.

    routes context operations to the correct ToolContextManager
    based on the current_conversation_id contextvar. creates managers
    lazily on first access for each conversation.

    :param context_collection: three-tier collection for context storage
    :ptype context_collection: Any
    :param l3_pool: optional L3 pool for direct persistence
    :ptype l3_pool: Any
    :param var_limit: max variables per conversation
    :ptype var_limit: int
    :param result_limit: max tool results before LRU eviction
    :ptype result_limit: int | None
    :param max_managers: max concurrent conversation managers before LRU eviction
    :ptype max_managers: int
    """

    def __init__(
        self,
        context_collection: Any,
        l3_pool: Any = None,
        var_limit: int = 50,
        result_limit: int | None = None,
        max_managers: int = 500,
    ) -> None:
        """initialize registry with shared collection and pool.

        :param context_collection: three-tier collection for context storage
        :ptype context_collection: Any
        :param l3_pool: optional L3 pool for direct persistence
        :ptype l3_pool: Any
        :param var_limit: max variables per conversation
        :ptype var_limit: int
        :param result_limit: max tool results before LRU eviction
        :ptype result_limit: int | None
        :param max_managers: max concurrent conversation managers before LRU eviction
        :ptype max_managers: int
        """
        self._collection = context_collection
        self._l3_pool = l3_pool
        self._var_limit = var_limit
        self._result_limit = result_limit
        self._max_managers = max_managers
        self._managers: dict[str, Any] = {}
        self._access_order: list[str] = []

    def _get_current(self) -> Any:
        """resolve context manager for active conversation.

        creates a new ToolContextManager on first access for each
        conversation_id. evicts least-recently-used managers when
        max_managers is exceeded. uses current_conversation_id
        contextvar to determine which conversation is active.

        :return: ToolContextManager for the active conversation
        :rtype: Any
        """
        conv_id = current_conversation_id.get()
        if conv_id not in self._managers:
            self._evict_if_full()
            from threetears.agent.tools.context import ToolContextManager

            conv_uuid = UUID(conv_id) if conv_id != "default" else UUID(int=0)
            self._managers[conv_id] = ToolContextManager(
                collection=self._collection,
                conversation_id=conv_uuid,
                user_id=UUID(int=0),
                var_limit=self._var_limit,
                result_limit=self._result_limit,
                l3_pool=self._l3_pool,
            )
        # update access order for LRU
        if conv_id in self._access_order:
            self._access_order.remove(conv_id)
        self._access_order.append(conv_id)
        return self._managers[conv_id]

    def _evict_if_full(self) -> None:
        """evict least-recently-used manager if at capacity.

        removes the oldest accessed manager to stay within
        max_managers limit. the underlying context data is
        persisted in three-tier storage and will be reloaded
        if the conversation resumes.
        """
        while len(self._managers) >= self._max_managers and self._access_order:
            oldest = self._access_order.pop(0)
            self._managers.pop(oldest, None)

    async def save_context_item(self, **kwargs: Any) -> Any:
        """save context item to active conversation.

        :param kwargs: arguments forwarded to ToolContextManager.save_context_item
        :ptype kwargs: Any
        :return: context item identifier
        :rtype: Any
        """
        return await self._get_current().save_context_item(**kwargs)

    async def save_tool_result(
        self,
        tool_name: str,
        result: str,
        short_desc: str,
        **kwargs: Any,
    ) -> Any:
        """save tool execution result to active conversation.

        :param tool_name: name of tool that produced the result
        :ptype tool_name: str
        :param result: full result content
        :ptype result: str
        :param short_desc: token-efficient summary (max 200 chars)
        :ptype short_desc: str
        :param kwargs: additional arguments forwarded to ToolContextManager
        :ptype kwargs: Any
        :return: context item identifier
        :rtype: Any
        """
        return await self._get_current().save_tool_result(
            tool_name, result, short_desc, **kwargs,
        )

    async def set_variable(self, key: str, value: str, **kwargs: Any) -> Any:
        """set named variable in active conversation.

        :param key: variable name
        :ptype key: str
        :param value: variable value
        :ptype value: str
        :param kwargs: additional arguments
        :ptype kwargs: Any
        :return: context item identifier
        :rtype: Any
        """
        return await self._get_current().set_variable(key, value, **kwargs)

    async def get_variable(self, key: str) -> Any:
        """get named variable from active conversation.

        :param key: variable name
        :ptype key: str
        :return: variable data dict or None
        :rtype: Any
        """
        return await self._get_current().get_variable(key)

    async def delete_variable(self, key: str) -> bool:
        """delete named variable from active conversation.

        :param key: variable name
        :ptype key: str
        :return: True if deleted, False if not found
        :rtype: bool
        """
        return await self._get_current().delete_variable(key)

    def build_conversation_context(self) -> str | None:
        """build formatted context for system prompt injection.

        :return: formatted context string or None if empty
        :rtype: str | None
        """
        return self._get_current().build_conversation_context()

    def build_context_prompt(self) -> str:
        """build context prompt section for system message.

        :return: formatted context including variables and tool results
        :rtype: str
        """
        return self._get_current().build_context_prompt()

    def build_ledger_prompt(self) -> str:
        """build ledger prompt listing previously surfaced items.

        :return: formatted ledger section
        :rtype: str
        """
        return self._get_current().build_ledger_prompt()

    def build_workflow_prompt(self) -> str:
        """build workflow prompt with active checklist.

        :return: formatted workflow section
        :rtype: str
        """
        return self._get_current().build_workflow_prompt()
