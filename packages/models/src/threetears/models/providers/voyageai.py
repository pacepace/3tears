"""voyageai embedding provider adapter wrapping langchain-voyageai."""

from __future__ import annotations

from typing import Any

from threetears.models.results import EmbeddingResult

# default embedding dimensions for voyage-3 / voyage-3-lite
_DEFAULT_EMBEDDING_DIMENSIONS = 1024


class VoyageAIEmbeddingProvider:
    """embedding provider adapter for VoyageAI models via langchain-voyageai.

    wraps VoyageAIEmbeddings with lazy instantiation for single and batch
    embedding operations. maps standard api_key parameter to voyage_api_key
    and dimensions to output_dimension as required by VoyageAI SDK.

    :param api_key: API key for VoyageAI authentication
    :ptype api_key: str
    :param model_name: VoyageAI embedding model identifier
    :ptype model_name: str
    :param base_url: optional custom API base URL
    :ptype base_url: str | None
    :param embedding_dimensions: optional output vector dimensionality
    :ptype embedding_dimensions: int | None
    """

    def __init__(
        self,
        api_key: str,
        *,
        model_name: str = "voyage-3-lite",
        base_url: str | None = None,
        embedding_dimensions: int | None = None,
    ) -> None:
        self._model_name = model_name
        self._api_key = api_key
        self._base_url = base_url
        self._embedding_dimensions = embedding_dimensions
        self._model: Any = None

    def _get_model(self) -> Any:
        """lazily creates and caches VoyageAIEmbeddings instance.

        imports langchain_voyageai on first call to avoid module-level
        dependency on optional package. maps api_key to voyage_api_key
        and embedding_dimensions to output_dimension for VoyageAI SDK.

        :return: configured VoyageAIEmbeddings instance
        :rtype: Any
        """
        if self._model is not None:
            return self._model

        from threetears.models.providers._voyageai_compat import apply_voyageai_compat

        apply_voyageai_compat()

        from langchain_voyageai import VoyageAIEmbeddings

        kwargs: dict[str, Any] = {
            "model": self._model_name,
            "api_key": self._api_key,
        }
        if self._base_url is not None:
            kwargs["base_url"] = self._base_url
        if self._embedding_dimensions is not None:
            kwargs["output_dimension"] = self._embedding_dimensions

        self._model = VoyageAIEmbeddings(**kwargs)
        return self._model

    @property
    def dimensions(self) -> int:
        """number of dimensions in embedding vectors.

        returns configured dimensions if set, otherwise defaults to
        1024 for voyage-3 / voyage-3-lite compatibility.

        :return: embedding vector dimensionality
        :rtype: int
        """
        if self._embedding_dimensions is not None:
            return self._embedding_dimensions
        return _DEFAULT_EMBEDDING_DIMENSIONS

    async def embed(self, text: str) -> EmbeddingResult:
        """generates embedding vector for single text input.

        delegates to embed_batch and returns first result.

        :param text: text to embed
        :ptype text: str
        :return: embedding result with vector and metadata
        :rtype: EmbeddingResult
        """
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """generates embedding vectors for batch of text inputs.

        calls VoyageAIEmbeddings.aembed_documents and converts raw float
        vectors into EmbeddingResult objects with metadata.

        :param texts: list of texts to embed
        :ptype texts: list[str]
        :return: list of embedding results in same order as inputs
        :rtype: list[EmbeddingResult]
        """
        raw_embeddings: list[list[float]] = await self._get_model().aembed_documents(texts)

        results: list[EmbeddingResult] = []
        for text, embedding in zip(texts, raw_embeddings):
            results.append(
                EmbeddingResult(
                    vector=embedding,
                    dimensions=len(embedding),
                    model=self._model_name,
                    token_count=len(text) // 4,
                )
            )
        return results
