"""integration tests for full embedding pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.models.cache import ModelCache
from threetears.models.providers.openai import OpenAIEmbeddingProvider
from threetears.models.results import EmbeddingResult


class TestEmbeddingPipeline:
    """integration tests for full embedding pipeline."""

    @pytest.mark.asyncio
    async def test_embedding_pipeline(self) -> None:
        """full embedding pipeline: create, cache, embed, verify."""
        provider = OpenAIEmbeddingProvider("text-embedding-3-small", "sk-test")
        mock_model = MagicMock()
        mock_model.aembed_documents = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        provider.model = mock_model

        cache = ModelCache()
        cache.put("openai", "text-embedding-3-small", provider)

        cached_provider = cache.get("openai", "text-embedding-3-small")
        assert cached_provider is provider

        result = await provider.embed("test text")
        assert isinstance(result, EmbeddingResult)
        assert len(result.vector) == 3
        assert result.vector == [0.1, 0.2, 0.3]
        assert result.dimensions == 3
        assert result.model == "text-embedding-3-small"

    @pytest.mark.asyncio
    async def test_batch_embedding_pipeline(self) -> None:
        """batch embedding with multiple texts."""
        provider = OpenAIEmbeddingProvider("text-embedding-3-small", "sk-test")
        mock_model = MagicMock()
        mock_model.aembed_documents = AsyncMock(
            return_value=[
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ]
        )
        provider.model = mock_model

        cache = ModelCache()
        cache.put("openai", "text-embedding-3-small", provider)

        texts = ["first text", "second text", "third text"]
        results = await provider.embed_batch(texts)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, EmbeddingResult)
            assert result.dimensions == 3
            assert result.model == "text-embedding-3-small"

        assert results[0].vector == [0.1, 0.2, 0.3]
        assert results[1].vector == [0.4, 0.5, 0.6]
        assert results[2].vector == [0.7, 0.8, 0.9]
