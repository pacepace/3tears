"""Shortlist -- what the cull returns: an ordered, filtered subset that says so.

Carries the corpus's spend and notices forward so a consumer holds one object
rather than reconciling two, and adds the two facts only Select knows: whether
anything ranked this, and how each criterion was finally answered.

**Unranked is a stated fact, not an absence** (SR-L2, P8). :attr:`ranked`
defaults to False, which is the honest value for a shortlist nothing ordered:
entries are in corpus accumulation order, which is provider order, which is
not a ranking anyone produced.
"""

from __future__ import annotations

from pydantic import Field

from threetears.search.contracts._base import ContractModel
from threetears.search.contracts.corpus import CorpusEntry
from threetears.search.contracts.criteria import CriterionDisposition
from threetears.search.contracts.spend import Spend

__all__ = ["Shortlist"]


class Shortlist(ContractModel):
    """Candidates plus criteria, culled and ordered (search-spec.md §3.6)."""

    #: the surviving entries, ordered by the ranker when one ran and in
    #: corpus order when none did.
    entries: tuple[CorpusEntry, ...] = ()
    #: whether a ranker ordered these. False means the order carries no
    #: ranking judgment and must not be read as one (SR-L2).
    ranked: bool = False
    #: which ranker ordered them, when one did. None whenever
    #: :attr:`ranked` is False.
    ranker: str | None = None
    #: how every criterion was finally answered, after local application
    #: (SR-B2, P8).
    dispositions: tuple[CriterionDisposition, ...] = ()
    #: what producing the corpus cost, carried forward unchanged -- Select
    #: filters candidates and never spend.
    spend: Spend = Field(default_factory=Spend)
    #: degradations inherited from the corpus, plus any the cull recorded
    #: (candidates dropped for data a criterion needed and they lacked).
    notices: tuple[str, ...] = ()
