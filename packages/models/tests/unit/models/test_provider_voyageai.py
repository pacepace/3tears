"""tests for VoyageAIEmbeddingProvider adapter."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

from threetears.models.protocol import EmbeddingProvider
from threetears.models.providers.voyageai import (
    VoyageAIEmbeddingProvider,
    _DEFAULT_EMBEDDING_DIMENSIONS,
)
from threetears.models.results import EmbeddingResult


class TestVoyageAICompat:
    """tests for voyageai Python 3.14 compatibility shim."""

    def test_compat_stubs_multimodal_module(self) -> None:
        """apply_voyageai_compat injects stub module into sys.modules on 3.14+."""
        from threetears.models.providers._voyageai_compat import apply_voyageai_compat

        apply_voyageai_compat()
        if sys.version_info >= (3, 14):
            mod = sys.modules.get("voyageai.object.multimodal_embeddings")
            assert mod is not None
            assert hasattr(mod, "MultimodalEmbeddingsObject")
            assert hasattr(mod, "MultimodalInputRequest")

    def test_compat_safe_to_call_twice(self) -> None:
        """apply_voyageai_compat is safe to call multiple times."""
        from threetears.models.providers._voyageai_compat import apply_voyageai_compat

        apply_voyageai_compat()
        apply_voyageai_compat()

    def test_compat_allows_voyageai_import(self) -> None:
        """after compat patch, langchain_voyageai can be imported on 3.14+."""
        from threetears.models.providers._voyageai_compat import apply_voyageai_compat

        apply_voyageai_compat()
        if sys.version_info >= (3, 14):
            from langchain_voyageai import VoyageAIEmbeddings

            assert hasattr(VoyageAIEmbeddings, "aembed_documents")


class TestVoyageAIEmbeddingProvider:
    """tests for VoyageAIEmbeddingProvider class."""

    def test_satisfies_embedding_provider_protocol(self) -> None:
        """VoyageAIEmbeddingProvider instance satisfies EmbeddingProvider protocol check."""
        provider = VoyageAIEmbeddingProvider("pa-test-key")
        assert isinstance(provider, EmbeddingProvider)

    def test_dimensions_property_returns_configured(self) -> None:
        """dimensions property returns configured embedding_dimensions."""
        provider = VoyageAIEmbeddingProvider(
            "pa-test-key",
            embedding_dimensions=512,
        )
        assert provider.dimensions == 512

    def test_dimensions_property_default(self) -> None:
        """dimensions property returns default when not configured."""
        provider = VoyageAIEmbeddingProvider("pa-test-key")
        assert provider.dimensions == _DEFAULT_EMBEDDING_DIMENSIONS
        assert provider.dimensions == 1024

    def test_default_model_name(self) -> None:
        """default model_name is voyage-3-lite."""
        provider = VoyageAIEmbeddingProvider("pa-test-key")
        assert provider._model_name == "voyage-3-lite"

    def test_custom_model_name(self) -> None:
        """custom model_name is stored correctly."""
        provider = VoyageAIEmbeddingProvider(
            "pa-test-key",
            model_name="voyage-3",
        )
        assert provider._model_name == "voyage-3"

    async def test_embed_returns_single_result(self) -> None:
        """embed returns single EmbeddingResult from aembed_documents."""
        provider = VoyageAIEmbeddingProvider("pa-test-key")
        mock_model = MagicMock()
        mock_model.aembed_documents = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        provider.model = mock_model

        result = await provider.embed("test text")

        assert isinstance(result, EmbeddingResult)
        assert result.vector == [0.1, 0.2, 0.3]
        assert result.dimensions == 3
        assert result.model == "voyage-3-lite"

    async def test_embed_batch_returns_multiple_results(self) -> None:
        """embed_batch returns list of EmbeddingResult for each input text."""
        provider = VoyageAIEmbeddingProvider("pa-test-key")
        mock_model = MagicMock()
        mock_model.aembed_documents = AsyncMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        provider.model = mock_model

        results = await provider.embed_batch(["text one", "text two"])

        assert len(results) == 2
        assert results[0].vector == [0.1, 0.2, 0.3]
        assert results[0].model == "voyage-3-lite"
        assert results[1].vector == [0.4, 0.5, 0.6]
        assert results[1].dimensions == 3

    async def test_embed_batch_token_count_estimation(self) -> None:
        """embed_batch estimates token_count as len(text) // 4."""
        provider = VoyageAIEmbeddingProvider("pa-test-key")
        mock_model = MagicMock()
        mock_model.aembed_documents = AsyncMock(return_value=[[0.1, 0.2]])
        provider.model = mock_model

        # 20 characters -> 20 // 4 = 5 estimated tokens
        results = await provider.embed_batch(["twelve chars plus 8x"])

        assert results[0].token_count == 5

    async def test_embed_batch_dimensions_from_vector(self) -> None:
        """embed_batch dimensions reflect actual vector length, not configured dimensions."""
        provider = VoyageAIEmbeddingProvider(
            "pa-test-key",
            embedding_dimensions=1024,
        )
        mock_model = MagicMock()
        mock_model.aembed_documents = AsyncMock(return_value=[[0.1, 0.2, 0.3, 0.4, 0.5]])
        provider.model = mock_model

        results = await provider.embed_batch(["some text"])

        assert results[0].dimensions == 5
        assert provider.dimensions == 1024

    def test_base_url_stored(self) -> None:
        """base_url is stored on provider."""
        provider = VoyageAIEmbeddingProvider(
            "pa-test-key",
            base_url="https://custom.voyageai.com/v1",
        )
        assert provider._base_url == "https://custom.voyageai.com/v1"
