"""SearchRequest -- what the caller asks for, and nothing it must not carry.

Fields exist only where a requirement forces them: query text, the open
criteria vocabulary, requested fidelity (SR-B6), the opt-in record flag
(SR-F6), and budget scope tags (SR-D2's plural, non-interchangeable
scopes). Query text is user content (D11): the contract makes it available
for redaction; redaction policy stays with the consumer.

Canonical serialization is exposed here as a public contract feature
(D26, SR-F1): :meth:`SearchRequest.canonical_form` and
:meth:`SearchRequest.canonical_digest` render the explicitly-set
*semantic* parameters only -- query, criteria, fidelity -- with absent and
defaulted canonically identical and criteria order canonically
irrelevant. The operational fields (``record``, ``budget_scope_tags``)
never participate: a recording is made with ``record=True`` by definition
(SR-F6), so keying the flag would make every recorded search's digest one
that no later replay of the same search derives (SR-F7), and scope tags
carry per-run identity (SR-D2), which would give every eval run a unique
digest and defeat SR-F1's attributability (Gate A, 2026-08-10).
"""

from __future__ import annotations

from typing import ClassVar

from threetears.search.contracts._base import ContractModel
from threetears.search.contracts._canonical import canonical_digest, canonicalize
from threetears.search.contracts.criteria import Criterion

__all__ = ["SearchRequest"]


class SearchRequest(ContractModel):
    """One search, as the caller states it."""

    CANONICAL_ORDER_INSENSITIVE: ClassVar[frozenset[str]] = frozenset({"criteria"})
    CANONICAL_EXCLUDED: ClassVar[frozenset[str]] = frozenset({"record", "budget_scope_tags"})

    #: the query text. User content (D11) -- never logged or persisted by
    #: this package's layers except where the consumer's policy says so.
    query: str
    #: constraints on what comes back, in the one open vocabulary
    #: (SR-B1). The response answers per criterion (SR-B2).
    criteria: tuple[Criterion, ...] = ()
    #: the fidelity the consumer needs
    #: (:mod:`threetears.search.contracts.fidelity` vocabulary); None asks
    #: for the provider's listing-grade default.
    fidelity: str | None = None
    #: opt-in replay recording for this call (SR-F6). Recording is never
    #: ambient.
    record: bool = False
    #: which budget scopes this call debits (SR-D2: per-persona-per-day,
    #: per-invocation, per-run, ... -- plural and not interchangeable).
    #: Tags name scopes; the port that enforces them is a parameter, not
    #: payload.
    budget_scope_tags: tuple[str, ...] = ()

    def canonical_form(self) -> str:
        """Render this request in canonical form (D26, SR-F1).

        Explicitly-set semantic parameters only (``record`` and
        ``budget_scope_tags`` are operational and never appear); absent
        and defaulted identical; criteria order irrelevant; stable output.

        :return: canonical JSON text
        :rtype: str
        """
        return canonicalize(self)

    def canonical_digest(self) -> str:
        """Digest this request's canonical form for equality keying.

        The replay key derivation (SR-F8) combines this with provider
        instance identity, profile digest, and a key-derivation version;
        eval identity consumes the same canonical form (SR-F1).

        :return: hex SHA-256 of :meth:`canonical_form`
        :rtype: str
        """
        return canonical_digest(self)
