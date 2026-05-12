"""integration tests for the LangChain-native embedding factory pipeline."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from threetears.models.factory import create_embedding_model


class TestEmbeddingFactoryPipeline:
    """integration tests covering capability lookup and factory dispatch."""

    def test_openai_embedding_model_construction(self) -> None:
        """factory builds an ``Embeddings`` for ``text-embedding-3-small``."""
        model = create_embedding_model("text-embedding-3-small", api_key="sk-test")
        assert isinstance(model, Embeddings)

    def test_voyageai_embedding_model_construction(self) -> None:
        """factory builds an ``Embeddings`` for ``voyage-3-lite``."""
        model = create_embedding_model("voyage-3-lite", api_key="pa-test")
        assert isinstance(model, Embeddings)
