"""The shared embed helpers answer a usable vector or raise, never ``None``.

Three packages each carried a soft-fail shim that answered ``None`` for an embedder that raised,
answered an empty vector, or answered a short batch; every caller then read the ``None`` as its
own quiet answer. These pin the one refusal that replaced them.
"""

from __future__ import annotations

import pytest
from langchain_core.embeddings import Embeddings

from threetears.langgraph.embeddings import EmbeddingUnavailableError, embed_documents, embed_query


class _Scripted(Embeddings):
    """answers what it is given, or raises it."""

    def __init__(self, query: list[float] | Exception, documents: list[list[float]] | Exception) -> None:
        self._query = query
        self._documents = documents

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    async def aembed_query(self, text: str) -> list[float]:
        if isinstance(self._query, Exception):
            raise self._query
        return self._query

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if isinstance(self._documents, Exception):
            raise self._documents
        return self._documents


async def test_a_query_vector_is_answered() -> None:
    assert await embed_query(_Scripted([0.1, 0.2], []), "hello") == [0.1, 0.2]


@pytest.mark.parametrize(
    ("embedder", "text", "why"),
    [
        (_Scripted(ConnectionError("gateway down"), []), "hello", "the embedder failed: ConnectionError: gateway down"),
        (_Scripted([], []), "hello", "answered an empty vector"),
        (_Scripted([0.1], []), "", "there is no text to embed"),
    ],
)
async def test_a_query_that_gives_no_usable_vector_is_refused(embedder: Embeddings, text: str, why: str) -> None:
    with pytest.raises(EmbeddingUnavailableError, match=why):
        await embed_query(embedder, text)


async def test_the_provider_fault_is_chained() -> None:
    with pytest.raises(EmbeddingUnavailableError) as caught:
        await embed_query(_Scripted(ConnectionError("gateway down"), []), "hello")
    assert isinstance(caught.value.__cause__, ConnectionError)


async def test_a_batch_is_answered_one_vector_per_text() -> None:
    assert await embed_documents(_Scripted([], [[1.0], [2.0]]), ["a", "b"]) == [[1.0], [2.0]]


async def test_an_empty_batch_makes_no_call() -> None:
    assert await embed_documents(_Scripted([], ConnectionError("never called")), []) == []


@pytest.mark.parametrize(
    ("documents", "why"),
    [
        (ConnectionError("gateway down"), "the embedder failed: ConnectionError: gateway down"),
        ([[1.0]], "answered 1 vector"),
        ([[1.0], []], "empty vector for at least one text"),
    ],
)
async def test_a_batch_that_gives_no_usable_vectors_is_refused(
    documents: list[list[float]] | Exception, why: str
) -> None:
    with pytest.raises(EmbeddingUnavailableError, match=why):
        await embed_documents(_Scripted([], documents), ["a", "b"])
