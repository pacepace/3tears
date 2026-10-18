"""The shared embed helpers answer a usable vector or raise, never ``None``.

Three packages each carried a soft-fail shim that answered ``None`` for an embedder that raised,
answered an empty vector, or answered a short batch; every caller then read the ``None`` as its
own quiet answer. These pin the one refusal that replaced them.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.embeddings import Embeddings

from threetears.langgraph.embeddings import EmbeddingUnavailableError, embed_documents, embed_query


class _Array(list[float]):
    """a vector whose truth value is ambiguous, as a numpy array's is."""

    def __bool__(self) -> bool:
        raise ValueError("The truth value of an array with more than one element is ambiguous.")


class _Scripted(Embeddings):
    """answers what it is given, or raises it."""

    def __init__(self, query: Any, documents: Any) -> None:
        self._query = query
        self._documents = documents
        self.sent: list[list[str]] = []

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    async def aembed_query(self, text: str) -> list[float]:
        if isinstance(self._query, Exception):
            raise self._query
        return self._query

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.sent.append(list(texts))
        if isinstance(self._documents, Exception):
            raise self._documents
        return self._documents


async def test_a_query_vector_is_answered() -> None:
    assert await embed_query(_Scripted([0.1, 0.2], []), "hello") == [0.1, 0.2]


@pytest.mark.parametrize(
    ("embedder", "text", "why"),
    [
        (_Scripted(ConnectionError("gateway down"), []), "hello", r"the embedder failed \(ConnectionError\)"),
        (_Scripted([], []), "hello", "answered an empty vector"),
        (_Scripted(_Array(), []), "hello", "answered an empty vector"),
        (_Scripted([0.1], []), "", "there is no text to embed"),
        (_Scripted([0.1], []), "  \n ", "there is no text to embed"),
    ],
)
async def test_a_query_that_gives_no_usable_vector_is_refused(embedder: Embeddings, text: str, why: str) -> None:
    with pytest.raises(EmbeddingUnavailableError, match=why):
        await embed_query(embedder, text)


async def test_a_numpy_vector_is_answered_as_floats() -> None:
    assert await embed_query(_Scripted(_Array([0.5, 0.25]), []), "hello") == [0.5, 0.25]


async def test_the_providers_own_message_never_reaches_the_refusal() -> None:
    """a caller hands this message to a model; a provider's text can carry a key."""
    leak = ValueError("Incorrect API key provided: sk-proj-FAKE1234567890 at https://api.example.test/v1")
    for call in (embed_query(_Scripted(leak, []), "hello"), embed_documents(_Scripted([], leak), ["a"])):
        with pytest.raises(EmbeddingUnavailableError) as caught:
            await call
        assert "sk-proj" not in str(caught.value) and "https://" not in str(caught.value)
        assert caught.value.__cause__ is leak, "the full text stays on the cause, for contain() to log"


async def test_the_provider_fault_is_chained() -> None:
    with pytest.raises(EmbeddingUnavailableError) as caught:
        await embed_query(_Scripted(ConnectionError("gateway down"), []), "hello")
    assert isinstance(caught.value.__cause__, ConnectionError)


async def test_a_batch_is_answered_one_vector_per_text() -> None:
    assert await embed_documents(_Scripted([], [[1.0], [2.0]]), ["a", "b"]) == [[1.0], [2.0]]


async def test_an_empty_batch_makes_no_call() -> None:
    assert await embed_documents(_Scripted([], ConnectionError("never called")), []) == []


async def test_a_whitespace_text_is_not_sent_and_comes_back_without_a_vector() -> None:
    embedder = _Scripted([], [[1.0]])
    assert await embed_documents(embedder, ["  ", "b"]) == [None, [1.0]]
    assert embedder.sent == [["b"]]


async def test_one_empty_vector_leaves_that_text_without_one_and_keeps_the_rest() -> None:
    assert await embed_documents(_Scripted([], [[1.0], _Array()]), ["a", "b"]) == [[1.0], None]


@pytest.mark.parametrize(
    ("documents", "why"),
    [
        (ConnectionError("gateway down"), r"the embedder failed \(ConnectionError\)"),
        ([[1.0]], "answered 1 vector"),
        ([[], []], "no usable vector for any text"),
    ],
)
async def test_a_batch_that_gives_no_usable_vectors_is_refused(
    documents: list[list[float]] | Exception, why: str
) -> None:
    with pytest.raises(EmbeddingUnavailableError, match=why):
        await embed_documents(_Scripted([], documents), ["a", "b"])
