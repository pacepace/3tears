"""The ranker slot -- a shape Select consumes and this package never fills.

`family-convergence.md` §4.14 rules reranking **out of search**: MMR lives in
``agent-memory``, rerank capability metadata and pricing in ``3tears-models``,
and a cross-encoder arrives as a models provider when a consumer pulls for
one. Select owns the criteria negotiation and the cull, and composes a ranker
in. This module is the seam and deliberately ships no implementation.

**Not even a pass-through default**, and that is the part worth stating.
SR-L2/P8 require unranked output be *marked* unranked. Something occupying
this slot while returning input order would be a ranking implementation that
lies about being one, and the mark would either disappear or become false.
Absent ranker means absent ranker, and
:attr:`~threetears.search.contracts.shortlist.Shortlist.ranked` says so.

There is a second reason the slot stays empty, and it is a success check
rather than a matter of taste: *"a Pi deployment installs it without torch"*
(check 5). A bundled ranker is how torch arrives.

A ranker **orders and never culls** -- see :meth:`Ranker.rank`. Filtering is
Select's, and a ranker that also filtered would apply criteria nothing
recorded a disposition for (SR-B2).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from threetears.search.contracts.corpus import CorpusEntry

__all__ = ["Ranker"]


@runtime_checkable
class Ranker(Protocol):
    """Orders corpus entries. Supplied by a consumer; never by this package."""

    @property
    def name(self) -> str:
        """Who did the ranking, recorded on the shortlist for provenance.

        An instance name where that distinction matters (two cross-encoders
        are two rankers), following the ``provider_instance`` convention.

        :return: the ranker's name
        :rtype: str
        """
        ...

    def rank(self, entries: Sequence[CorpusEntry], /) -> Sequence[CorpusEntry]:
        """Return the same entries in a new order.

        MUST return exactly the entries it was given -- same identities, same
        count. Dropping one applies a constraint no disposition answers for
        (SR-B2), and adding one invents a result. Select checks this rather
        than trusting it, because a ranker is consumer code.

        :param entries: the entries to order
        :ptype entries: Sequence[CorpusEntry]
        :return: the same entries, reordered
        :rtype: Sequence[CorpusEntry]
        """
        ...
