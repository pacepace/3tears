"""Tests for the ``EmbeddingProvider`` alias for ``langchain_core.embeddings.Embeddings``."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from threetears.agent.memory.embedding import EmbeddingProvider


class StubEmbedder(Embeddings):
    """Concrete ``Embeddings`` implementation for testing."""

    def __init__(self, dims: int = 1024) -> None:
        self._dims = dims

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """returns a fixed-length zero vector for each input text."""
        return [[0.1] * self._dims for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """returns a fixed-length vector or empty list for empty text."""
        if not text:
            return []
        return [0.1] * self._dims


class TestEmbeddingProvider:
    """``EmbeddingProvider`` is now an alias for ``Embeddings``."""

    def test_alias_resolves_to_embeddings(self) -> None:
        """``EmbeddingProvider`` is the same object as ``Embeddings``."""
        assert EmbeddingProvider is Embeddings

    def test_stub_subclasses_embeddings(self) -> None:
        """concrete subclasses of ``Embeddings`` satisfy ``isinstance`` checks."""
        embedder = StubEmbedder(dims=1024)
        assert isinstance(embedder, EmbeddingProvider)
        assert isinstance(embedder, Embeddings)

    def test_embed_query_returns_vector(self) -> None:
        """``embed_query`` returns a vector of the expected dimensionality."""
        embedder = StubEmbedder(dims=4)
        vector = embedder.embed_query("hello world")
        assert len(vector) == 4

    def test_embed_query_returns_empty_on_empty(self) -> None:
        """``embed_query`` returns an empty list for empty input."""
        embedder = StubEmbedder()
        vector = embedder.embed_query("")
        assert vector == []
