"""tests for :func:`create_anthropic_chat` factory and capability registration."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

from threetears.models.capabilities import get_capabilities
from threetears.models.enums import ModelTier, ModelType
from threetears.models.providers.anthropic import (
    ANTHROPIC_PROVIDER_NAME,
    create_anthropic_chat,
    strip_v1_suffix,
)


class TestCreateAnthropicChat:
    """tests for the ``create_anthropic_chat`` factory function."""

    def test_returns_base_chat_model(self) -> None:
        """factory returns a ``BaseChatModel`` subclass instance."""
        model = create_anthropic_chat("claude-sonnet-4-20250514", "sk-test")
        assert isinstance(model, BaseChatModel)
        assert isinstance(model, ChatAnthropic)

    def test_model_name_propagated(self) -> None:
        """factory forwards model_name to ``ChatAnthropic``."""
        model = create_anthropic_chat("claude-3-5-haiku-20241022", "sk-test")
        assert model.model == "claude-3-5-haiku-20241022"

    def test_strips_v1_suffix_from_base_url(self) -> None:
        """trailing ``/v1`` is stripped from base_url before instantiation."""
        model = create_anthropic_chat(
            "claude-sonnet-4-20250514",
            "sk-test",
            base_url="https://api.anthropic.com/v1",
        )
        assert model.anthropic_api_url == "https://api.anthropic.com"

    def test_strips_v1_slash_suffix(self) -> None:
        """trailing ``/v1/`` is also stripped."""
        model = create_anthropic_chat(
            "claude-sonnet-4-20250514",
            "sk-test",
            base_url="https://api.anthropic.com/v1/",
        )
        assert model.anthropic_api_url == "https://api.anthropic.com"

    def test_no_base_url_means_default(self) -> None:
        """omitting base_url leaves ``ChatAnthropic`` to apply its default."""
        model = create_anthropic_chat("claude-sonnet-4-20250514", "sk-test")
        # ChatAnthropic defaults the URL itself; just verify it is not None.
        assert model.anthropic_api_url is not None


class TestStripV1Suffix:
    """tests for the ``strip_v1_suffix`` helper."""

    def test_strips_v1(self) -> None:
        """``/v1`` suffix is removed."""
        assert strip_v1_suffix("https://api.anthropic.com/v1") == "https://api.anthropic.com"

    def test_strips_v1_slash(self) -> None:
        """``/v1/`` suffix is removed."""
        assert strip_v1_suffix("https://api.anthropic.com/v1/") == "https://api.anthropic.com"

    def test_no_v1_unchanged(self) -> None:
        """URL without ``/v1`` suffix is returned unchanged."""
        assert strip_v1_suffix("https://custom.api.com") == "https://custom.api.com"

    def test_v1_in_middle_unchanged(self) -> None:
        """URL with ``/v1`` in middle path is returned unchanged."""
        assert strip_v1_suffix("https://api.com/v1/extra") == "https://api.com/v1/extra"


class TestAnthropicCapabilityRegistration:
    """tests that anthropic-canonical models register at import time."""

    def test_sonnet_registered(self) -> None:
        """``claude-sonnet-4-20250514`` resolves to anthropic chat capabilities."""
        caps = get_capabilities("claude-sonnet-4-20250514")
        assert caps is not None
        assert caps.provider_name == ANTHROPIC_PROVIDER_NAME
        assert caps.model_type == ModelType.CHAT
        assert caps.model_tier == ModelTier.LARGE
        assert caps.supports_tools is True

    def test_haiku_registered(self) -> None:
        """``claude-3-5-haiku-20241022`` resolves to anthropic small-tier chat."""
        caps = get_capabilities("claude-3-5-haiku-20241022")
        assert caps is not None
        assert caps.provider_name == ANTHROPIC_PROVIDER_NAME
        assert caps.model_tier == ModelTier.SMALL
