"""Internal helpers for working with LangChain ``Embeddings`` instances.

Adds two thin shims that the agent-memory call sites need now that the
local ``EmbeddingProvider`` Protocol has been replaced by
:class:`langchain_core.embeddings.Embeddings`:

- :func:`_safe_aembed_query` -- wraps :meth:`Embeddings.aembed_query` in
  try/except, returning ``None`` on failure (callers used to rely on the
  ``embed_text`` tuple's ``None`` sentinel for soft-fail).
- :func:`_estimate_tokens` -- returns a coarse token count estimate. The
  retired ``embed_text`` returned a real token count from the embedding
  provider; LangChain's ``Embeddings`` interface does not surface one, so
  call sites that need it estimate locally (used only for observability,
  not billing).
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from threetears.observe import get_logger

# no ``__all__`` — both helpers are intentionally package-private
# (``_``-prefixed) and only consumed by sibling ``threetears.agent.memory``
# modules. listing private names in ``__all__`` is contradictory and the
# underscore-access enforcement (shape E) flags it.

_log = get_logger(__name__)

# rough words-to-tokens ratio for English. used only for observability
# (RetrievalResult.embed_tokens), not for billing or rate-limiting.
_TOKENS_PER_WORD = 1.3


async def _safe_aembed_query(
    embedder: Embeddings,
    text: str,
) -> list[float] | None:
    """invokes :meth:`Embeddings.aembed_query` with soft-fail semantics.

    returns ``None`` when the embedder raises (network error, auth failure,
    upstream outage). matches the ``embed_text(text) -> (None, 0)``
    behaviour of the retired protocol so soft-fail call sites continue to
    work without changes.

    :param embedder: LangChain embedding model
    :ptype embedder: Embeddings
    :param text: text to embed
    :ptype text: str
    :return: embedding vector or ``None`` on failure
    :rtype: list[float] | None
    """
    if not text:
        return None
    try:
        result = await embedder.aembed_query(text)
    except Exception as exc:
        _log.warning(
            "embedding aembed_query failed (soft-fail): %s",
            exc,
        )
        return None
    if not result:
        return None
    return result


def _estimate_tokens(text: str) -> int:
    """returns a coarse token-count estimate for ``text``.

    used by :class:`~threetears.agent.memory.retrieval.RetrievalResult` for
    observability only. retains parity with the previous embed_text
    contract (which returned ``len(text.split())`` from the stub
    implementations).

    :param text: text to estimate
    :ptype text: str
    :return: estimated token count
    :rtype: int
    """
    if not text:
        return 0
    word_count = len(text.split())
    return max(int(word_count * _TOKENS_PER_WORD), 1)
