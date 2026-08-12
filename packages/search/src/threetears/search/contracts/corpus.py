"""Corpus -- Aggregate's accumulation type, and the reason it is not a bag of candidates (D2, SR-A5).

Call returns a :class:`~threetears.search.contracts.candidate.CandidateSet`:
one query, one adapter. Accumulation across calls, dedup and merge belong
here instead, because D2 rules two types with two dedup/merge stories
rather than one type that grows a second meaning.

**Why an entry holds its contributions whole rather than merging them.**
:attr:`~threetears.search.contracts.candidate.Candidate.provenance` is
*singular*. So the obvious dedup -- collapse two providers' copies of one
URL into one candidate -- has exactly one slot for an origin and must
discard the other. That destroys the per-result grounding SR-A3 exists to
keep, and :mod:`threetears.search.contracts.provenance` already states why
nothing can reconstruct it afterwards: *"the grounding question a consumer
eventually asks -- does this claim appear on the page it was cited from --
is per-result, and no aggregate can answer it after the fact."*

So a :class:`CorpusEntry` keeps every contributing candidate intact and
exposes merged *views* over them. Every provenance survives and every score
entry keeps its own ``source``.

**A stage's own judgment is entry-level, not provider-level.** Rank fusion
produces a score about the *entry*, and writing it onto one contribution
would claim a provider reported something it never did. Those live on
:attr:`CorpusEntry.derived_scores`, which is also the only place a
``comparable=True`` score may appear (SR-A4: only a stage that normalised
across providers may declare one).

**Scores are never combined into a value** (D1). SearXNG's engine-fusion
weight is unbounded above -- two agreeing engines score 4.0, three score
9.0 -- while Tavily's relevance lives in [0, 1]. Averaging those produces a
meaningful-*looking* number that means nothing, so :attr:`CorpusEntry.scores`
is a union of distinct entries, each still naming its source and each still
non-comparable.

Entry order is accumulation order and is **not** a ranking (SR-L2).
Ordering is Select's business.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from threetears.search.contracts._base import ContractModel
from threetears.search.contracts.candidate import Candidate, ContentSlot
from threetears.search.contracts.criteria import CriterionDisposition
from threetears.search.contracts.provenance import Provenance
from threetears.search.contracts.scores import ScoreEntry
from threetears.search.contracts.spend import Spend

__all__ = ["Corpus", "CorpusEntry"]


class CorpusEntry(ContractModel):
    """Every candidate that shares one identity, kept whole.

    A merge is a view over :attr:`contributions`, never a mutation of them.
    Two providers returning the same URL produce one entry with two
    contributions -- not one candidate with one provenance and a discarded
    second.
    """

    #: the dedup key these contributions grouped under. That is the shared
    #: ``Candidate.identity`` in the default case, and whatever a
    #: caller-supplied normalising key derived otherwise -- so it is not
    #: safe to assume it equals any contribution's own identity.
    identity: str
    #: the contributing candidates, in accumulation order, each intact.
    #: Never empty: an entry with nothing in it has no identity to be.
    contributions: tuple[Candidate, ...]
    #: judgments a pipeline stage made about this entry, as distinct from
    #: the ones providers reported about their own results. Rank fusion
    #: writes here; nothing else does yet.
    derived_scores: tuple[ScoreEntry, ...] = ()

    @model_validator(mode="after")
    def _check_contributions(self) -> CorpusEntry:
        """Refuse an entry with nothing in it.

        Deliberately **not** checked: that every contribution's own
        ``identity`` equals :attr:`identity`. It does under the default key
        and does not under a normalising one, and a rule that holds only
        for the default would forbid the very lever the default exists to
        offer.

        :return: the validated entry
        :rtype: CorpusEntry
        :raises ValueError: when there are no contributions
        """
        if not self.contributions:
            raise ValueError(f"corpus entry {self.identity!r} has no contributions")
        return self

    @property
    def scores(self) -> tuple[ScoreEntry, ...]:
        """Every score entry about this entry, unioned and never combined (D1).

        Provider-reported entries stay distinct and stay non-comparable; a
        consumer selecting by ``name`` may get several answers, which is the
        honest shape because two providers scoring "relevance" did not score
        the same thing. Stage-derived entries follow them.

        :return: contributed scores in contribution order, then derived ones
        :rtype: tuple[ScoreEntry, ...]
        """
        contributed = tuple(score for contribution in self.contributions for score in contribution.scores)
        return (*contributed, *self.derived_scores)

    @property
    def provenances(self) -> tuple[Provenance, ...]:
        """Where every contribution came from (SR-A3).

        :return: one provenance per contribution, in contribution order
        :rtype: tuple[Provenance, ...]
        """
        return tuple(contribution.provenance for contribution in self.contributions)

    @property
    def title(self) -> str | None:
        """The first title any contributor supplied.

        "First" is accumulation order, not a quality judgment -- providers
        disagree about titles and nothing here is entitled to arbitrate.

        :return: the title, or None when no contributor had one
        :rtype: str | None
        """
        return next((c.title for c in self.contributions if c.title is not None), None)

    @property
    def snippet(self) -> str | None:
        """The first snippet any contributor supplied.

        :return: the snippet, or None when no contributor had one
        :rtype: str | None
        """
        return next((c.snippet for c in self.contributions if c.snippet is not None), None)

    @property
    def content(self) -> ContentSlot | None:
        """The first content slot any contributor supplied (SR-A2).

        This is what stops a mixed corpus re-fetching what one provider
        already returned: Tavily ships page text with the search response,
        and an entry that also has a SearXNG contribution still carries it.

        :return: the content slot, or None when no contributor had one
        :rtype: ContentSlot | None
        """
        return next((c.content for c in self.contributions if c.content is not None), None)


class Corpus(ContractModel):
    """Many calls, one accumulated set (D2).

    An empty corpus is a success value, for the reason an empty
    ``CandidateSet`` is (SR-J2): a search that found nothing answered the
    question.
    """

    #: the deduplicated entries, in accumulation order. **Not a ranking**
    #: (SR-L2) -- Select orders, this accumulates.
    entries: tuple[CorpusEntry, ...] = ()
    #: per-criterion answers across every contributing call, reporting the
    #: weakest honest answer where contributors disagreed (SR-B2, P8).
    dispositions: tuple[CriterionDisposition, ...] = ()
    #: what the whole fan-out consumed, summed across contributing calls --
    #: including calls that failed and bought nothing (D4).
    spend: Spend = Field(default_factory=Spend)
    #: degradations gathered from contributing calls, plus any the
    #: aggregation itself recorded (a failed sibling, a dropped provider).
    #: Empty means nothing went wrong anywhere (P8).
    notices: tuple[str, ...] = ()
