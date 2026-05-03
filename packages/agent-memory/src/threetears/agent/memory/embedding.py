"""Embedding interface alias for vector generation.

3tears v0.6.0+: ``EmbeddingProvider`` is now an alias for
:class:`langchain_core.embeddings.Embeddings`. Concrete implementations
must override :meth:`Embeddings.embed_documents` and
:meth:`Embeddings.embed_query` (and their ``aembed_*`` async variants).

The retired :func:`embed_text(text) -> tuple[list[float] | None, int]`
shape has been replaced by direct calls to
:meth:`Embeddings.aembed_query`. Token counts (previously the second
element of that tuple) are now estimated locally where they're needed
since the LangChain ``Embeddings`` interface does not surface them.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

__all__ = [
    "EmbeddingProvider",
]


# alias kept for ergonomic imports; new code should prefer Embeddings
# directly so the LangChain provenance is explicit at the call site.
EmbeddingProvider = Embeddings
