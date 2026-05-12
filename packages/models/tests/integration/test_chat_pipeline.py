"""integration tests for the LangChain-native chat factory pipeline."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from threetears.models.capabilities import get_capabilities
from threetears.models.factory import create_chat_model


class TestChatFactoryPipeline:
    """integration tests covering capability lookup, factory dispatch, and callbacks."""

    def test_anthropic_chat_model_construction(self) -> None:
        """factory builds a callback-wrapped anthropic chat model."""
        model = create_chat_model("claude-sonnet-4-20250514", api_key="sk-test")
        assert isinstance(model, Runnable)
        bound = getattr(model, "bound", model)
        assert isinstance(bound, BaseChatModel)

    def test_openai_chat_model_construction(self) -> None:
        """factory builds a callback-wrapped openai chat model."""
        model = create_chat_model("gpt-4o-mini", api_key="sk-test")
        assert isinstance(model, Runnable)

    def test_openrouter_chat_model_construction(self) -> None:
        """factory builds a callback-wrapped openrouter chat model."""
        model = create_chat_model("deepseek/deepseek-chat-v3-0324", api_key="sk-test")
        assert isinstance(model, Runnable)

    def test_factory_attaches_callbacks(self) -> None:
        """``with_config(callbacks=[...])`` wires both tracker and breaker callbacks."""
        model = create_chat_model("claude-sonnet-4-20250514", api_key="sk-test")
        # RunnableBinding stores config in `config`. retrieve and assert callbacks are present.
        config = getattr(model, "config", {})
        callbacks = config.get("callbacks", [])
        assert len(callbacks) >= 2

    def test_capabilities_resolve_for_factory_targets(self) -> None:
        """all factory-target ids are in the capability registry."""
        for model_id in (
            "claude-sonnet-4-20250514",
            "gpt-4o",
            "deepseek/deepseek-chat-v3-0324",
        ):
            caps = get_capabilities(model_id)
            assert caps is not None, f"missing capabilities for {model_id}"
