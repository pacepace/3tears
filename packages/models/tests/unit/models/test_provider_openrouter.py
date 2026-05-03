"""tests for ``create_openrouter_chat`` factory and capability registration."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from threetears.models.capabilities import get_capabilities
from threetears.models.enums import ModelTier, ModelType
from threetears.models.providers.openrouter import (
    OPENROUTER_PROVIDER_NAME,
    create_openrouter_chat,
)


class TestCreateOpenRouterChat:
    """tests for ``create_openrouter_chat`` factory."""

    def test_returns_base_chat_model(self) -> None:
        """factory returns a ``BaseChatModel`` subclass instance."""
        model = create_openrouter_chat("deepseek/deepseek-chat-v3-0324", "sk-test")
        assert isinstance(model, BaseChatModel)


class TestOpenRouterCapabilityRegistration:
    """tests that openrouter canonical models register at import time."""

    def test_deepseek_chat_registered(self) -> None:
        """``deepseek/deepseek-chat-v3-0324`` resolves to openrouter chat capabilities."""
        caps = get_capabilities("deepseek/deepseek-chat-v3-0324")
        assert caps is not None
        assert caps.provider_name == OPENROUTER_PROVIDER_NAME
        assert caps.model_type == ModelType.CHAT
        assert caps.model_tier == ModelTier.LARGE
        assert caps.requires_alternating_roles is True
