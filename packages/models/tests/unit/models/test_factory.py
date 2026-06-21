"""tests for the LangChain-native factory entry points."""

from __future__ import annotations

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from threetears.models import DEFAULT_CHAT_MODEL, DEFAULT_LARGE_MODEL
from threetears.models.factory import create_chat_model, create_embedding_model


class TestCreateChatModel:
    """tests for :func:`create_chat_model`."""

    def test_returns_runnable_chat_model_for_anthropic(self) -> None:
        """``DEFAULT_CHAT_MODEL`` resolves through the registry."""
        model = create_chat_model(
            DEFAULT_CHAT_MODEL,
            api_key="sk-test",
        )
        # `with_config(callbacks=...)` returns a RunnableBinding wrapping the chat model.
        assert isinstance(model, Runnable)
        # the bound model is still a BaseChatModel under the hood.
        bound = getattr(model, "bound", model)
        assert isinstance(bound, BaseChatModel)

    def test_returns_runnable_chat_model_for_openai(self) -> None:
        """``gpt-4o`` resolves through the registry."""
        model = create_chat_model(
            "gpt-4o",
            api_key="sk-test",
        )
        assert isinstance(model, Runnable)

    def test_unregistered_model_requires_explicit_provider(self) -> None:
        """unknown ``model_id`` without ``provider`` kwarg raises ``ValueError``."""
        with pytest.raises(ValueError, match="Cannot resolve provider"):
            create_chat_model("unknown-model-id", api_key="sk-test")

    def test_unregistered_model_with_explicit_provider_works(self) -> None:
        """explicit ``provider=`` lets the factory dispatch even without registration."""
        model = create_chat_model(
            DEFAULT_LARGE_MODEL,
            api_key="sk-test",
            provider="anthropic",
        )
        assert isinstance(model, Runnable)

    def test_embedding_model_id_rejected(self) -> None:
        """passing an embedding model id to chat factory raises ``ValueError``."""
        with pytest.raises(ValueError, match="registered as embedding"):
            create_chat_model(
                "text-embedding-3-small",
                api_key="sk-test",
            )


class TestCreateEmbeddingModel:
    """tests for :func:`create_embedding_model`."""

    def test_returns_embeddings_for_openai(self) -> None:
        """``text-embedding-3-small`` resolves through the registry."""
        model = create_embedding_model(
            "text-embedding-3-small",
            api_key="sk-test",
        )
        assert isinstance(model, Embeddings)

    def test_unregistered_model_requires_explicit_provider(self) -> None:
        """unknown embedding model_id without provider raises ``ValueError``."""
        with pytest.raises(ValueError, match="Cannot resolve provider"):
            create_embedding_model("unknown-embedding", api_key="sk-test")

    def test_chat_model_id_rejected(self) -> None:
        """passing a chat model id to embedding factory raises ``ValueError``."""
        with pytest.raises(ValueError, match="registered as chat"):
            create_embedding_model(
                DEFAULT_CHAT_MODEL,
                api_key="sk-test",
            )
