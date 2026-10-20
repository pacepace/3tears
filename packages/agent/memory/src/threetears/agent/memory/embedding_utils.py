"""Internal helpers for working with LangChain ``Embeddings`` instances.

- :func:`_estimate_tokens` -- returns a coarse token count estimate. The
  retired ``embed_text`` returned a real token count from the embedding
  provider; LangChain's ``Embeddings`` interface does not surface one, so
  call sites that need it estimate locally (used only for observability,
  not billing).
- :func:`embedding_attribution_scope` -- binds the usage-attribution user for
  the embeddings sent inside it.

The embed call itself is :func:`threetears.langgraph.embeddings.embed_query`, shared with
the other agent packages: it raises
:class:`~threetears.langgraph.embeddings.EmbeddingUnavailableError` where this module's
soft-fail shim used to answer ``None`` -- the same ``None`` every caller read as "nothing
relevant".
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import UUID

# ``__all__`` lists only the public embedding-attribution scope helper. The
# ``_``-prefixed :func:`_estimate_tokens` stays package-private and is intentionally
# excluded -- listing a private name in ``__all__`` is contradictory and the
# underscore-access enforcement (shape E) flags it.
__all__ = ["embedding_attribution_scope"]

# rough words-to-tokens ratio for English. used only for observability
# (RetrievalResult.embed_tokens), not for billing or rate-limiting.
_TOKENS_PER_WORD = 1.3


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


#: usage-attribution principal for the embedding(s) about to be sent. read by the
#: gateway embedding model when it builds the request and stamped onto the request's
#: ``attribution_user_id`` -- the gateway records it on the ``gateway_usage`` row.
#: this is the USAGE principal only; the AUTHORIZATION principal stays ``None`` (the
#: agent), so a chat user who lacks the model grant never denies the agent-internal
#: embedding.
#:
#: a ContextVar (not a constructor arg) because the langchain
#: ``Embeddings.aembed_query(text)`` surface has no per-call slot, and the single
#: embedder instance is shared across concurrently-multiplexed conversations. set +
#: read happen inside ONE coroutine (the embed call site sets it immediately before
#: awaiting the embed), so this never crosses a LangGraph node boundary -- where
#: contextvar propagation is NOT guaranteed (the runtime threads identity through
#: ``config`` for exactly that reason). default ``None`` means "no originating user"
#: and records a NULL attribution.
_attribution_user: ContextVar[UUID | None] = ContextVar(
    "threetears_embedding_attribution_user",
    default=None,
)


@contextmanager
def embedding_attribution_scope(user_id: UUID | None) -> Iterator[None]:
    """bind the usage-attribution user for embeddings sent inside this scope.

    set the originating user immediately before awaiting an embed call so the gateway
    embedding model stamps it onto the request's ``attribution_user_id``. the scope
    MUST wrap the embed call directly (no intervening LangGraph node boundary) --
    contextvars do not reliably propagate across nodes; within a single coroutine
    they do.

    a ``None`` user_id is a valid scope (background / wake-initiated embedding with no
    originating user) and records a NULL attribution.

    :param user_id: originating user UUID to attribute usage to, or ``None`` for an
        embedding with no originating user
    :ptype user_id: UUID | None
    :return: nothing yielded; the scope is a side-effect on the contextvar
    :rtype: Iterator[None]
    """
    token = _attribution_user.set(user_id)
    try:
        yield
    finally:
        _attribution_user.reset(token)
