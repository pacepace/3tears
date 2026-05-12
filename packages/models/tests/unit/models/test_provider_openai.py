"""tests for OpenAI chat and embedding factories and capability registration."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from threetears.models.capabilities import get_capabilities
from threetears.models.enums import ModelTier, ModelType
from threetears.models.providers.openai import (
    OPENAI_PROVIDER_NAME,
    create_openai_chat,
    create_openai_embedding,
)


class TestCreateOpenAIChat:
    """tests for ``create_openai_chat`` factory."""

    def test_returns_base_chat_model(self) -> None:
        """factory returns a ``BaseChatModel`` subclass instance."""
        model = create_openai_chat("gpt-4o", "sk-test")
        assert isinstance(model, BaseChatModel)
        assert isinstance(model, ChatOpenAI)

    def test_model_name_propagated(self) -> None:
        """factory forwards model name to ``ChatOpenAI``."""
        model = create_openai_chat("gpt-4o-mini", "sk-test")
        assert model.model_name == "gpt-4o-mini"

    def test_base_url_passed_through(self) -> None:
        """custom base_url is forwarded to ``ChatOpenAI``."""
        model = create_openai_chat("gpt-4o", "sk-test", base_url="https://api.example.com")
        assert str(model.openai_api_base) == "https://api.example.com"


class TestCreateOpenAIEmbedding:
    """tests for ``create_openai_embedding`` factory."""

    def test_returns_embeddings(self) -> None:
        """factory returns an ``Embeddings`` subclass instance."""
        model = create_openai_embedding("text-embedding-3-small", "sk-test")
        assert isinstance(model, Embeddings)
        assert isinstance(model, OpenAIEmbeddings)

    def test_dimensions_forwarded(self) -> None:
        """``embedding_dimensions`` is mapped to ``OpenAIEmbeddings.dimensions``."""
        model = create_openai_embedding("text-embedding-3-small", "sk-test", embedding_dimensions=512)
        assert model.dimensions == 512


class TestOpenAICapabilityRegistration:
    """tests that openai canonical models register at import time."""

    def test_gpt4o_registered(self) -> None:
        """``gpt-4o`` resolves to openai LARGE chat capabilities."""
        caps = get_capabilities("gpt-4o")
        assert caps is not None
        assert caps.provider_name == OPENAI_PROVIDER_NAME
        assert caps.model_type == ModelType.CHAT
        assert caps.model_tier == ModelTier.LARGE

    def test_text_embedding_3_small_registered(self) -> None:
        """``text-embedding-3-small`` resolves to openai embedding capabilities."""
        caps = get_capabilities("text-embedding-3-small")
        assert caps is not None
        assert caps.provider_name == OPENAI_PROVIDER_NAME
        assert caps.model_type == ModelType.EMBEDDING
        assert caps.embedding_dimensions == 1536
