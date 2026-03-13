"""Protocol types for agent-tools."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ChatModelFactory(Protocol):
    """Protocol for creating chat models for routing/execution."""

    async def create_chat_model(self, purpose: str = "routing") -> Any:
        """Create a chat model. Returns a LangChain BaseChatModel or compatible."""
        ...
