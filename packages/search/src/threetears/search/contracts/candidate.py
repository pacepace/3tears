"""Candidate -- the carrier-neutral result core (SR-C1), and its set (D2).

A candidate is identity + locators + provenance + scores + fidelity + an
optional content slot + open facets. There is deliberately NO closed
carrier union ("web | image | pdf" is prohibited): carrier-specific data
rides ``facets``, keyed by the ``media-contracts`` vocabulary, and a
consumer that does not recognise a facet ignores it rather than failing
(SR-C2, SR-C3). Adding a carrier must require no change here.

:class:`CandidateSet` is what Call returns -- one query through one
adapter. It is NOT the corpus: accumulation across calls, dedup, and merge
belong to Aggregate's own named type (``Corpus``, D2/SR-A5, Phase 3).
An empty ``candidates`` tuple is a *success* (SR-J2).
"""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, Field, JsonValue

from threetears.search.contracts._base import ContractModel
from threetears.search.contracts.criteria import CriterionDisposition
from threetears.search.contracts.provenance import Provenance
from threetears.search.contracts.scores import ScoreEntry
from threetears.search.contracts.spend import Spend

__all__ = [
    "Candidate",
    "CandidateSet",
    "ContentSlot",
    "Locator",
]


class Locator(ContractModel):
    """One address where the candidate (or a rendition of it) can be reached.

    ``rel`` is an open vocabulary with conventional values -- ``canonical``,
    ``direct-file``, ``containing-page``, ``thumbnail`` -- because the
    direct-file-versus-containing-page distinction is load-bearing for
    image work (SR-C3) and more relations will follow.
    """

    #: the URL itself.
    url: str
    #: what this URL is to the candidate; open vocabulary, ``canonical``
    #: by convention for the primary locator.
    rel: str = "canonical"


class ContentSlot(ContractModel):
    """The information itself, when a candidate carries it (SR-A2).

    Records whether the content arrived with the search response (the
    Tavily case -- already bought, never re-fetch it) or from a later
    fetch. Extract is a no-op when ``origin`` says the provider already
    supplied it.
    """

    #: extracted text content.
    text: str
    #: where the content came from: with the search response, from a
    #: subsequent fetch, or -- for content the caller already held and
    #: upstream confirmed still current -- from a revalidation (D30).
    #: ``revalidated`` keeps SR-A2's "where did this content come from"
    #: question answerable rather than letting a confirmed copy pass as a
    #: fresh fetch.
    origin: Literal["provider-response", "later-fetch", "revalidated"]
    #: MIME type of the source the text was extracted from, when known.
    mime_type: str | None = None
    #: size of the fetched source in bytes, when a fetch happened.
    size_bytes: int | None = None
    #: the ``ETag`` the content arrived with, when upstream gave one, to be
    #: echoed back in ``If-None-Match`` on a later conditional read (D30 /
    #: SR-M4). A first-class field rather than a facet, by the argument
    #: :attr:`Candidate.published_at` already makes: facets are the
    #: carrier-specific escape hatch (SR-C2/C3), and an HTTP validator is
    #: not carrier-specific.
    etag: str | None = None
    #: the ``Last-Modified`` header **as its own string**, deliberately not
    #: a datetime. It is an opaque token to be echoed back verbatim in
    #: ``If-Modified-Since``; parsing and re-rendering it risks changing the
    #: bytes and failing the match, and buys nothing -- nobody compares it,
    #: they only return it. This is the one place a date stays a string, and
    #: the reason is that it is not being read as a date.
    last_modified: str | None = None


class Candidate(ContractModel):
    """One retrieved result, whatever carries it."""

    #: stable identity for dedup and citation -- by convention the
    #: canonical URL; providers without URLs use their native id.
    identity: str
    #: where the candidate can be reached; at least the canonical locator.
    locators: tuple[Locator, ...]
    #: recorded origin (SR-A3): query, provider instance, native ids,
    #: retrieval time, egress, producer class.
    provenance: Provenance
    #: result title, when the provider gave one.
    title: str | None = None
    #: provider snippet / summary text, when given.
    snippet: str | None = None
    #: when the candidate was published, where the provider reported it and
    #: the report was parseable. A first-class field rather than a facet: a
    #: publication date is not carrier-specific, and adapters MUST keep what
    #: the provider returned in typed form rather than a disclaimed blob
    #: (search-spec.md §3.2). Timezone-aware by construction; an adapter
    #: reading a naive provider date states the zone it assumed, and keeps
    #: the raw string on :attr:`Provenance.provider_ids` so the assumption
    #: stays inspectable.
    published_at: AwareDatetime | None = None
    #: named, provenanced judgments (D1). Never a single ``score`` field.
    scores: tuple[ScoreEntry, ...] = ()
    #: the best fidelity the provider can supply for this candidate
    #: (:mod:`threetears.search.contracts.fidelity` vocabulary).
    fidelity_available: str | None = None
    #: the fidelity actually achieved for this candidate so far.
    fidelity_achieved: str | None = None
    #: the information itself, when present (SR-A2).
    content: ContentSlot | None = None
    #: additive carrier facets, keyed by the ``media-contracts`` vocabulary
    #: (``media_category``, ``extraction_status``, ``has_downloadable_data``,
    #: pixel dimensions, rights status, ...). Unrecognised facets are
    #: ignorable by contract (SR-C2); never a closed carrier union.
    facets: dict[str, JsonValue] = Field(default_factory=dict)


class CandidateSet(ContractModel):
    """What one Call returns: one query, one adapter, one candidate set (D2).

    Zero candidates is a success value, not an error (SR-J2). The
    dispositions answer for every criterion the request carried (SR-B2);
    spend is attached whether the call was priced or free (SR-E1).
    """

    #: the candidates, in provider order (ranking is Select's business;
    #: unranked output is known-unranked -- SR-L2).
    candidates: tuple[Candidate, ...] = ()
    #: per-criterion answers: pushdown, local, unsatisfied, or
    #: ignored-unknown (SR-B2, SR-B3).
    dispositions: tuple[CriterionDisposition, ...] = ()
    #: what the call consumed (SR-E1); zero-valued for a free provider.
    spend: Spend = Field(default_factory=Spend)
    #: degradations the provider reported that are neither per-candidate nor
    #: per-criterion -- engines that did not answer, a partial fan-in, output
    #: known to be unranked (SR-L2). Named in the typed response because a
    #: partial answer that reads as complete is exactly the defect P8 exists
    #: to prevent; empty means the provider reported nothing wrong.
    notices: tuple[str, ...] = ()
