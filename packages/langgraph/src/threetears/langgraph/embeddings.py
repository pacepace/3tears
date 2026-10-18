"""embedding calls that either answer a usable vector or raise.

a LangChain :class:`~langchain_core.embeddings.Embeddings` call can fail three ways that matter to
a caller: it raises (network error, auth failure, upstream outage), it answers an empty vector, or
it answers a batch of the wrong length. each package used to wrap the call in its own soft-fail
shim that answered ``None`` for all three, and every caller read that ``None`` as its own quiet
answer -- "no memories", "no similar wants", "nothing relevant" -- with the cause left in a log
line. these helpers turn every one of them into one :class:`EmbeddingUnavailableError` naming why,
so the caller decides what an outage costs and the decision is visible where it is made.

the refusal's message names the provider's exception CLASS only. a provider's own message can
carry a URL, a request body or a key ("Incorrect API key provided: sk-..."), and callers put this
message in front of a model; the full provider exception rides on ``__cause__``, where
:func:`threetears.observe.contain` logs it with the traceback.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.embeddings import Embeddings

__all__ = ["EmbeddingUnavailableError", "embed_documents", "embed_query"]


def _is_empty(vector: Sequence[float]) -> bool:
    """whether ``vector`` holds no values -- by length, so a numpy array (whose truth value is
    ambiguous) is read the same as a list."""
    return len(vector) == 0


class EmbeddingUnavailableError(RuntimeError):
    """the embedder gave no usable vector; the message says why, without the provider's own text.

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
    :raises EmbeddingUnavailableError: there is no text (or only whitespace), the embedder
        raised, or it answered an empty vector
    """
    if not text.strip():
        raise EmbeddingUnavailableError("there is no text to embed")
    try:
        result = await embedder.aembed_query(text)
    except Exception as exc:  # every provider fault becomes the one named refusal callers handle
        raise EmbeddingUnavailableError(f"the embedder failed ({type(exc).__name__})") from exc
    if result is None or _is_empty(result):
        raise EmbeddingUnavailableError("the embedder answered an empty vector")
    return [float(value) for value in result]


async def embed_documents(embedder: Embeddings, texts: Sequence[str]) -> list[list[float] | None]:
    """embed a batch of texts, one usable vector per text or a refusal.

    :param embedder: LangChain embedding model
    :ptype embedder: Embeddings
    :param texts: texts to embed; an empty batch answers ``[]`` without a call
    :ptype texts: Sequence[str]
    :return: one vector per text, in order; ``None`` in the place of a text that is empty or
        whitespace-only (never sent to the provider) or that the provider answered an empty
        vector for, so a caller can leave that one text unranked without losing the batch
    :rtype: list[list[float] | None]
    :raises EmbeddingUnavailableError: the embedder raised, answered a different number of
        vectors than texts it was sent, or answered no usable vector for any text at all
    """
    if not texts:
        return []
    sent = [i for i, text in enumerate(texts) if text.strip()]
    vectors: list[list[float] | None] = [None] * len(texts)
    if not sent:
        raise EmbeddingUnavailableError("there is no text to embed")
    try:
        result = await embedder.aembed_documents([texts[i] for i in sent])
    except Exception as exc:  # every provider fault becomes the one named refusal callers handle
        raise EmbeddingUnavailableError(f"the embedder failed ({type(exc).__name__})") from exc
    if len(result) != len(sent):
        raise EmbeddingUnavailableError(f"the embedder answered {len(result)} vector(s) for {len(sent)} text(s)")
    for i, vector in zip(sent, result, strict=True):
        if vector is not None and not _is_empty(vector):
            vectors[i] = [float(value) for value in vector]
    if all(vector is None for vector in vectors):
        raise EmbeddingUnavailableError("the embedder answered no usable vector for any text")
    return vectors
