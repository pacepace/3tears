"""Tests for EmbeddingProvider protocol."""

from __future__ import annotations

from threetears.agent.memory.embedding import EmbeddingProvider


class StubEmbedder:
    """Concrete implementation of EmbeddingProvider for testing."""

    def __init__(self, dims: int = 1024) -> None:
        self._dims = dims

    async def embed_text(self, text: str) -> tuple[list[float] | None, int]:
        if not text:
            return None, 0
        vector = [0.1] * self._dims
        return vector, len(text.split())

    @property
    def dimensions(self) -> int:
        return self._dims


class TestEmbeddingProtocol:
    def test_stub_satisfies_protocol(self) -> None:
        embedder = StubEmbedder(dims=1024)
        assert isinstance(embedder, EmbeddingProvider)

    def test_dimensions_property(self) -> None:
        embedder = StubEmbedder(dims=768)
        assert embedder.dimensions == 768

    async def test_embed_text_returns_vector(self) -> None:
        embedder = StubEmbedder(dims=4)
        vector, tokens = await embedder.embed_text("hello world")

        assert vector is not None
        assert len(vector) == 4
        assert tokens == 2

    async def test_embed_text_returns_none_on_empty(self) -> None:
        embedder = StubEmbedder()
        vector, tokens = await embedder.embed_text("")

        assert vector is None
        assert tokens == 0

    def test_class_without_dimensions_not_provider(self) -> None:
        class NotAnEmbedder:
            async def embed_text(self, text: str) -> tuple[list[float] | None, int]:
                return None, 0

        assert not isinstance(NotAnEmbedder(), EmbeddingProvider)

    def test_class_without_embed_text_not_provider(self) -> None:
        class AlsoNotAnEmbedder:
            @property
            def dimensions(self) -> int:
                return 1024

        assert not isinstance(AlsoNotAnEmbedder(), EmbeddingProvider)
