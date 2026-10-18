"""embedding calls that either answer a usable vector or raise.

a LangChain :class:`~langchain_core.embeddings.Embeddings` call can fail three ways that matter to
a caller: it raises (network error, auth failure, upstream outage), it answers an empty vector, or
it answers a batch of the wrong length. each package used to wrap the call in its own soft-fail
shim that answered ``None`` for all three, and every caller read that ``None`` as its own quiet
answer -- "no memories", "no similar wants", "nothing relevant" -- with the cause left in a log
line. these helpers turn every one of them into one :class:`EmbeddingUnavailableError` naming why,
so the caller decides what an outage costs and the decision is visible where it is made.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.embeddings import Embeddings

__all__ = ["EmbeddingUnavailableError", "embed_documents", "embed_query"]


class EmbeddingUnavailableError(RuntimeError):
    """the embedder gave no usable vector; the message says why.

    raised in place of the ``None`` the per-package embed shims used to answer, so a caller
    that cannot embed says so -- a tool answers an error naming the cause, retrieval refuses
    rather than reporting "nothing relevant" -- instead of reading the outage as an answer.
    """


async def embed_query(embedder: Embeddings, text: str) -> list[float]:
    """embed one text, refusing anything that is not a usable vector.

    :param embedder: LangChain embedding model
    :ptype embedder: Embeddings
    :param text: text to embed
    :ptype text: str
    :return: the embedding vector
    :rtype: list[float]
    :raises EmbeddingUnavailableError: there is no text, the embedder raised, or it answered an
        empty vector
    """
    if not text:
        raise EmbeddingUnavailableError("there is no text to embed")
    try:
        result = await embedder.aembed_query(text)
    except Exception as exc:  # every provider fault becomes the one named refusal callers handle
        raise EmbeddingUnavailableError(f"the embedder failed: {type(exc).__name__}: {exc}") from exc
    if not result:
        raise EmbeddingUnavailableError("the embedder answered an empty vector")
    return list(result)


async def embed_documents(embedder: Embeddings, texts: Sequence[str]) -> list[list[float]]:
    """embed a batch of texts, one usable vector per text or a refusal.

    :param embedder: LangChain embedding model
    :ptype embedder: Embeddings
    :param texts: texts to embed; an empty batch answers ``[]`` without a call
    :ptype texts: Sequence[str]
    :return: one vector per text, in order
    :rtype: list[list[float]]
    :raises EmbeddingUnavailableError: the embedder raised, answered a different number of
        vectors than texts, or answered an empty vector for any of them
    """
    if not texts:
        return []
    try:
        result = await embedder.aembed_documents(list(texts))
    except Exception as exc:  # every provider fault becomes the one named refusal callers handle
        raise EmbeddingUnavailableError(f"the embedder failed: {type(exc).__name__}: {exc}") from exc
    if len(result) != len(texts):
        raise EmbeddingUnavailableError(f"the embedder answered {len(result)} vector(s) for {len(texts)} text(s)")
    if not all(result):
        raise EmbeddingUnavailableError("the embedder answered an empty vector for at least one text")
    return [list(vector) for vector in result]
