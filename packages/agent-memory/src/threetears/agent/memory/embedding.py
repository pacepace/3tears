"""Embedding provider protocol for vector generation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding generation backends."""

    async def embed_text(self, text: str) -> tuple[list[float] | None, int]:
        """Generate an embedding vector for the given text.

        Returns (embedding_vector, token_count). Vector is None on failure.
        """
        ...

    @property
    def dimensions(self) -> int:
        """Vector dimensions (e.g., 1024)."""
        ...
