"""tests for ``create_voyageai_embedding`` factory and capability registration."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from threetears.models.capabilities import get_capabilities
from threetears.models.enums import ModelTier, ModelType
from threetears.models.providers.voyageai import (
    VOYAGEAI_PROVIDER_NAME,
    create_voyageai_embedding,
)


class TestCreateVoyageAIEmbedding:
    """tests for ``create_voyageai_embedding`` factory."""

    def test_returns_embeddings(self) -> None:
        """factory returns an ``Embeddings`` subclass instance."""
        model = create_voyageai_embedding("pa-test-key", model_name="voyage-3-lite")
        assert isinstance(model, Embeddings)


class TestVoyageAICapabilityRegistration:
    """tests that voyageai canonical models register at import time."""

    def test_voyage_3_registered(self) -> None:
        """``voyage-3`` resolves to voyageai LARGE embedding capabilities."""
        caps = get_capabilities("voyage-3")
        assert caps is not None
        assert caps.provider_name == VOYAGEAI_PROVIDER_NAME
        assert caps.model_type == ModelType.EMBEDDING
        assert caps.model_tier == ModelTier.LARGE
        assert caps.embedding_dimensions == 1024

    def test_voyage_3_lite_registered(self) -> None:
        """``voyage-3-lite`` resolves to voyageai SMALL embedding capabilities."""
        caps = get_capabilities("voyage-3-lite")
        assert caps is not None
        assert caps.provider_name == VOYAGEAI_PROVIDER_NAME
        assert caps.model_type == ModelType.EMBEDDING
        assert caps.model_tier == ModelTier.SMALL
